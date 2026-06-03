use std::fmt;

use super::super::{
    ControlTarget, MemoryAddress, MemoryAddressBase, MemoryAddressImmediate, MemorySpace,
    PredicateCondition, RegisterRef,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassAnalysisModule {
    pub target: Option<String>,
    pub functions: Vec<SassAnalysisFunction>,
}

impl SassAnalysisModule {
    pub fn block_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.blocks.len())
            .sum()
    }

    pub fn edge_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.edges.len())
            .sum()
    }

    pub fn dataflow_op_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.dataflow.len())
            .sum()
    }

    pub fn reaching_use_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.reaching_uses.len())
            .sum()
    }

    pub fn live_range_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.live_ranges.len())
            .sum()
    }

    pub fn memory_access_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.memory_accesses.len())
            .sum()
    }

    pub fn dominator_block_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.dominators.len())
            .sum()
    }

    pub fn natural_loop_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.natural_loops.len())
            .sum()
    }

    pub fn region_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.regions.len())
            .sum()
    }

    pub fn ssa_value_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.ssa_values.len())
            .sum()
    }

    pub fn def_use_edge_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.def_use_edges.len())
            .sum()
    }

    pub fn value_op_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.value_ops.len())
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassAnalysisFunction {
    pub name: String,
    pub blocks: Vec<SassBasicBlock>,
    pub edges: Vec<SassCfgEdge>,
    pub dominators: Vec<SassDominatorBlock>,
    pub natural_loops: Vec<SassNaturalLoop>,
    pub regions: Vec<SassRegion>,
    pub dataflow: Vec<SassDataflowOp>,
    pub reaching_uses: Vec<SassReachingUse>,
    pub ssa_values: Vec<SassSsaValue>,
    pub def_use_edges: Vec<SassDefUseEdge>,
    pub value_ops: Vec<SassValueOp>,
    pub live_ranges: Vec<SassLiveRange>,
    pub memory_accesses: Vec<SassMemoryAccess>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassBasicBlock {
    pub id: usize,
    pub label: Option<String>,
    pub start_address: u64,
    pub end_address: u64,
    pub start_op_index: usize,
    pub end_op_index: usize,
    pub instruction_count: usize,
    pub terminator: SassBlockTerminator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassBlockTerminator {
    Fallthrough,
    Branch,
    Call,
    Return,
    Exit,
}

impl fmt::Display for SassBlockTerminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fallthrough => f.write_str("fallthrough"),
            Self::Branch => f.write_str("branch"),
            Self::Call => f.write_str("call"),
            Self::Return => f.write_str("return"),
            Self::Exit => f.write_str("exit"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCfgEdge {
    pub from_block: usize,
    pub to_block: Option<usize>,
    pub kind: SassCfgEdgeKind,
    pub condition: Option<PredicateCondition>,
    pub target: Option<ControlTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassCfgEdgeKind {
    Fallthrough,
    Branch,
    Call,
    Return,
    Exit,
}

impl fmt::Display for SassCfgEdgeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fallthrough => f.write_str("fallthrough"),
            Self::Branch => f.write_str("branch"),
            Self::Call => f.write_str("call"),
            Self::Return => f.write_str("return"),
            Self::Exit => f.write_str("exit"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassDominatorBlock {
    pub block_id: usize,
    pub reachable: bool,
    pub immediate_dominator: Option<usize>,
    pub dominators: Vec<usize>,
    pub dominated_blocks: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassNaturalLoop {
    pub header_block: usize,
    pub latch_block: usize,
    pub reachable: bool,
    pub blocks: Vec<usize>,
    pub edge_condition: Option<PredicateCondition>,
    pub edge_target: Option<ControlTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassRegion {
    pub id: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub depth: usize,
    pub path: SassRegionPath,
    pub local_rank: usize,
    pub kind: SassRegionKind,
    pub header_block: Option<usize>,
    pub latch_block: Option<usize>,
    pub branch_block: Option<usize>,
    pub entry_blocks: Vec<usize>,
    pub blocks: Vec<usize>,
    pub op_addresses: Vec<u64>,
    pub opcode_closure: Vec<String>,
    pub condition: Option<PredicateCondition>,
    pub target: Option<ControlTarget>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassRegionPath {
    ranks: Vec<usize>,
}

impl SassRegionPath {
    pub fn new(ranks: Vec<usize>) -> Self {
        Self { ranks }
    }

    pub fn len(&self) -> usize {
        self.ranks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    pub fn push(&mut self, rank: usize) {
        self.ranks.push(rank);
    }

    pub fn starts_with(&self, parent: &Self) -> bool {
        self.ranks.starts_with(&parent.ranks)
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.ranks
    }

    pub fn to_vec(&self) -> Vec<usize> {
        self.ranks.clone()
    }
}

impl From<Vec<usize>> for SassRegionPath {
    fn from(ranks: Vec<usize>) -> Self {
        Self::new(ranks)
    }
}

impl AsRef<[usize]> for SassRegionPath {
    fn as_ref(&self) -> &[usize] {
        self.as_slice()
    }
}

impl fmt::Display for SassRegionPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, rank) in self.ranks.iter().enumerate() {
            if index > 0 {
                f.write_str(".")?;
            }
            write!(f, "{rank}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassRegionKind {
    Function,
    NaturalLoop,
    Branch,
    BranchArm,
}

impl fmt::Display for SassRegionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function => f.write_str("function"),
            Self::NaturalLoop => f.write_str("natural-loop"),
            Self::Branch => f.write_str("branch"),
            Self::BranchArm => f.write_str("branch-arm"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassDataflowOp {
    pub address: u64,
    pub predicate: Option<String>,
    pub defines: Vec<RegisterRef>,
    pub uses: Vec<RegisterRef>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassReachingUse {
    pub address: u64,
    pub register: RegisterRef,
    pub reaching_def_addresses: Vec<u64>,
    pub reaches_entry: bool,
}

impl SassReachingUse {
    pub fn sources_text(&self) -> String {
        let mut sources = self
            .reaching_def_addresses
            .iter()
            .map(|address| format!("{address:#06x}"))
            .collect::<Vec<_>>();
        if self.reaches_entry {
            sources.insert(0, "entry".to_string());
        }
        sources.join(",")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassSsaValue {
    pub value_id: usize,
    pub register: RegisterRef,
    pub def_address: Option<u64>,
    pub source: Option<String>,
    pub use_addresses: Vec<u64>,
}

impl SassSsaValue {
    pub fn name(&self) -> String {
        format_register_definition(&self.register, self.def_address)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassDefUseEdge {
    pub value_id: usize,
    pub register: RegisterRef,
    pub def_address: Option<u64>,
    pub use_address: u64,
    pub use_source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassValueOp {
    pub address: u64,
    pub block_id: Option<usize>,
    pub predicate: Option<String>,
    pub opcode: String,
    pub kind: String,
    pub input_registers: Vec<RegisterRef>,
    pub output_registers: Vec<RegisterRef>,
    pub input_value_ids: Vec<usize>,
    pub output_value_ids: Vec<usize>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassLiveRange {
    pub register: RegisterRef,
    pub def_address: Option<u64>,
    pub start_address: u64,
    pub end_address: u64,
    pub use_addresses: Vec<u64>,
}

impl SassLiveRange {
    pub fn def_text(&self) -> String {
        self.def_address
            .map(|address| format!("{address:#06x}"))
            .unwrap_or_else(|| "entry".to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassMemoryAccess {
    pub address: u64,
    pub predicate: Option<String>,
    pub kind: SassMemoryAccessKind,
    pub space: MemorySpace,
    pub width_bits: Option<u32>,
    pub value_register: RegisterRef,
    pub memory_address: MemoryAddress,
    pub address_registers: Vec<RegisterRef>,
    pub address_base: Option<MemoryAddressBase>,
    pub offset: Option<MemoryAddressImmediate>,
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassMemoryAccessKind {
    Load,
    Store,
    LoadConst,
}

impl fmt::Display for SassMemoryAccessKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load => f.write_str("load"),
            Self::Store => f.write_str("store"),
            Self::LoadConst => f.write_str("load-const"),
        }
    }
}

pub(super) fn format_register_definition(
    register: &RegisterRef,
    def_address: Option<u64>,
) -> String {
    def_address
        .map(|address| format!("{register}@{address:#06x}"))
        .unwrap_or_else(|| format!("{register}@entry"))
}
