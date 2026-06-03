use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
};

use super::{KernelIrFunction, KernelIrModule, KernelIrOp, KernelIrOpKind, MemorySpace};

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

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        if let Some(target) = &self.target {
            writeln!(out, "target {target}").expect("write to string");
        }
        for function in &self.functions {
            writeln!(out, "fn {} {{", function.name).expect("write to string");
            writeln!(out, "  blocks").expect("write to string");
            for block in &function.blocks {
                writeln!(
                    out,
                    "    b{} {:#06x}-{:#06x} label={} instructions={} terminator={}",
                    block.id,
                    block.start_address,
                    block.end_address,
                    block.label.as_deref().unwrap_or("-"),
                    block.instruction_count,
                    block.terminator
                )
                .expect("write to string");
            }
            writeln!(out, "  edges").expect("write to string");
            for edge in &function.edges {
                writeln!(
                    out,
                    "    b{} -> {} [{} condition={} target={}]",
                    edge.from_block,
                    edge.to_block
                        .map(|block| format!("b{block}"))
                        .unwrap_or_else(|| "external".to_string()),
                    edge.kind,
                    edge.condition.as_deref().unwrap_or("-"),
                    edge.target.as_deref().unwrap_or("-")
                )
                .expect("write to string");
            }
            writeln!(out, "  dataflow").expect("write to string");
            for dataflow in &function.dataflow {
                writeln!(
                    out,
                    "    {:#06x}: def=[{}] use=[{}] <- {}",
                    dataflow.address,
                    dataflow.defines.join(","),
                    dataflow.uses.join(","),
                    dataflow.source
                )
                .expect("write to string");
            }
            writeln!(out, "  reaching_uses").expect("write to string");
            for reaching in &function.reaching_uses {
                writeln!(
                    out,
                    "    {:#06x}: {} <- [{}]",
                    reaching.address,
                    reaching.register,
                    reaching.sources_text()
                )
                .expect("write to string");
            }
            writeln!(out, "  live_ranges").expect("write to string");
            for range in &function.live_ranges {
                writeln!(
                    out,
                    "    {}@{} {:#06x}-{:#06x} uses=[{}]",
                    range.register,
                    range.def_text(),
                    range.start_address,
                    range.end_address,
                    format_addresses(&range.use_addresses)
                )
                .expect("write to string");
            }
            writeln!(out, "  memory_accesses").expect("write to string");
            for access in &function.memory_accesses {
                writeln!(
                    out,
                    "    {:#06x}: {} {} value={} addr={} regs=[{}] base={} offset={} width={} predicate={} <- {}",
                    access.address,
                    access.kind,
                    access.space,
                    access.value_register,
                    access.address_expr,
                    access.address_registers.join(","),
                    access.address_base.as_deref().unwrap_or("-"),
                    access.offset.as_deref().unwrap_or("-"),
                    access
                        .width_bits
                        .map(|bits| bits.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    access.predicate.as_deref().unwrap_or("-"),
                    access.source
                )
                .expect("write to string");
            }
            writeln!(out, "}}").expect("write to string");
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassAnalysisFunction {
    pub name: String,
    pub blocks: Vec<SassBasicBlock>,
    pub edges: Vec<SassCfgEdge>,
    pub dataflow: Vec<SassDataflowOp>,
    pub reaching_uses: Vec<SassReachingUse>,
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

impl std::fmt::Display for SassBlockTerminator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    pub condition: Option<String>,
    pub target: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassCfgEdgeKind {
    Fallthrough,
    Branch,
    Call,
    Return,
    Exit,
}

impl std::fmt::Display for SassCfgEdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
pub struct SassDataflowOp {
    pub address: u64,
    pub predicate: Option<String>,
    pub defines: Vec<String>,
    pub uses: Vec<String>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassReachingUse {
    pub address: u64,
    pub register: String,
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
pub struct SassLiveRange {
    pub register: String,
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
    pub value_register: String,
    pub address_expr: String,
    pub address_registers: Vec<String>,
    pub address_base: Option<String>,
    pub offset: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassMemoryAccessKind {
    Load,
    Store,
    LoadConst,
}

impl std::fmt::Display for SassMemoryAccessKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load => f.write_str("load"),
            Self::Store => f.write_str("store"),
            Self::LoadConst => f.write_str("load-const"),
        }
    }
}

pub fn analyze_sass_ir(module: &KernelIrModule) -> SassAnalysisModule {
    SassAnalysisModule {
        target: module.target.clone(),
        functions: module.functions.iter().map(analyze_function).collect(),
    }
}

fn analyze_function(function: &KernelIrFunction) -> SassAnalysisFunction {
    let blocks = build_blocks(function);
    let edges = build_edges(function, &blocks);
    let dataflow = function
        .ops
        .iter()
        .map(analyze_dataflow)
        .collect::<Vec<_>>();
    let (reaching_uses, live_ranges) = analyze_reaching_defs(&blocks, &edges, &dataflow);
    let memory_accesses = analyze_memory_accesses(function);
    SassAnalysisFunction {
        name: function.name.clone(),
        blocks,
        edges,
        dataflow,
        reaching_uses,
        live_ranges,
        memory_accesses,
    }
}

fn analyze_memory_accesses(function: &KernelIrFunction) -> Vec<SassMemoryAccess> {
    let mut accesses = function
        .ops
        .iter()
        .filter_map(|op| match &op.kind {
            KernelIrOpKind::Load {
                dst,
                address,
                space,
            } => Some(memory_access(
                op,
                SassMemoryAccessKind::Load,
                *space,
                dst,
                address,
            )),
            KernelIrOpKind::LoadConst { dst, source } => Some(memory_access(
                op,
                SassMemoryAccessKind::LoadConst,
                MemorySpace::Constant,
                dst,
                source,
            )),
            KernelIrOpKind::Store {
                address,
                value,
                space,
            } => Some(memory_access(
                op,
                SassMemoryAccessKind::Store,
                *space,
                value,
                address,
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    accesses.sort_by_key(|access| access.address);
    accesses
}

fn memory_access(
    op: &KernelIrOp,
    kind: SassMemoryAccessKind,
    space: MemorySpace,
    value_register: &str,
    address_expr: &str,
) -> SassMemoryAccess {
    let (address_base, offset) = memory_base_and_offset(space, address_expr);
    SassMemoryAccess {
        address: op.address,
        predicate: op.predicate.clone(),
        kind,
        space,
        width_bits: memory_width_bits(&op.source_modifiers),
        value_register: value_register.to_string(),
        address_expr: address_expr.to_string(),
        address_registers: extract_registers(address_expr),
        address_base,
        offset,
        source: op.source.clone(),
    }
}

fn memory_width_bits(modifiers: &[String]) -> Option<u32> {
    modifiers.iter().find_map(|modifier| {
        modifier
            .strip_prefix('U')
            .or_else(|| modifier.strip_prefix('S'))
            .and_then(|bits| bits.parse::<u32>().ok())
            .or_else(|| modifier.parse::<u32>().ok())
    })
}

fn memory_base_and_offset(
    space: MemorySpace,
    address_expr: &str,
) -> (Option<String>, Option<String>) {
    match space {
        MemorySpace::Descriptor => descriptor_base_and_offset(address_expr),
        MemorySpace::Constant => constant_base_and_offset(address_expr),
        _ => (None, offset_after_plus(address_expr)),
    }
}

fn descriptor_base_and_offset(address_expr: &str) -> (Option<String>, Option<String>) {
    let Some(rest) = address_expr.strip_prefix("desc") else {
        return (None, offset_after_plus(address_expr));
    };
    let Some((descriptor, rest)) = bracketed(rest) else {
        return (None, offset_after_plus(address_expr));
    };
    let offset = bracketed(rest)
        .and_then(|(address, tail)| tail.trim().is_empty().then_some(address))
        .and_then(offset_after_plus);
    (Some(descriptor.to_string()), offset)
}

fn constant_base_and_offset(address_expr: &str) -> (Option<String>, Option<String>) {
    let Some(rest) = address_expr.strip_prefix('c') else {
        return (None, offset_after_plus(address_expr));
    };
    let Some((bank, rest)) = bracketed(rest) else {
        return (None, offset_after_plus(address_expr));
    };
    let offset = bracketed(rest)
        .and_then(|(offset, tail)| tail.trim().is_empty().then_some(offset.to_string()));
    (Some(bank.to_string()), offset)
}

fn bracketed(text: &str) -> Option<(&str, &str)> {
    let text = text.strip_prefix('[')?;
    let end = text.find(']')?;
    Some((&text[..end], &text[end + 1..]))
}

fn offset_after_plus(text: &str) -> Option<String> {
    text.split_once('+')
        .map(|(_, offset)| offset.trim().to_string())
        .filter(|offset| !offset.is_empty())
}

fn build_blocks(function: &KernelIrFunction) -> Vec<SassBasicBlock> {
    if function.ops.is_empty() {
        return Vec::new();
    }

    let mut starts = BTreeSet::new();
    starts.insert(0usize);
    for (index, op) in function.ops.iter().enumerate() {
        if op.label.is_some() {
            starts.insert(index);
        }
        if is_block_boundary_terminator(op) && index + 1 < function.ops.len() {
            starts.insert(index + 1);
        }
        if let Some(target) = branch_target(op) {
            if let Some(target_index) = label_index(function, target) {
                starts.insert(target_index);
            }
        }
    }

    let starts = starts.into_iter().collect::<Vec<_>>();
    let mut blocks = Vec::new();
    for (id, start) in starts.iter().copied().enumerate() {
        let end = starts
            .get(id + 1)
            .copied()
            .unwrap_or(function.ops.len())
            .saturating_sub(1);
        let start_op = &function.ops[start];
        let end_op = &function.ops[end];
        blocks.push(SassBasicBlock {
            id,
            label: start_op.label.clone(),
            start_address: start_op.address,
            end_address: end_op.address,
            start_op_index: start,
            end_op_index: end,
            instruction_count: end.saturating_sub(start) + 1,
            terminator: terminator_for(end_op),
        });
    }
    blocks
}

fn build_edges(function: &KernelIrFunction, blocks: &[SassBasicBlock]) -> Vec<SassCfgEdge> {
    let mut edges = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        let Some(last_op) = function.ops.get(block.end_op_index) else {
            continue;
        };
        let next_block = blocks.get(index + 1).map(|block| block.id);
        match &last_op.kind {
            KernelIrOpKind::Branch { target, condition } => {
                let target_block = target
                    .as_deref()
                    .and_then(|target| label_index(function, target))
                    .and_then(|target_index| block_id_for_op_index(blocks, target_index));
                edges.push(SassCfgEdge {
                    from_block: block.id,
                    to_block: target_block,
                    kind: SassCfgEdgeKind::Branch,
                    condition: condition.clone(),
                    target: target.clone(),
                });
                if condition.is_some() {
                    if let Some(next_block) = next_block {
                        edges.push(SassCfgEdge {
                            from_block: block.id,
                            to_block: Some(next_block),
                            kind: SassCfgEdgeKind::Fallthrough,
                            condition: None,
                            target: None,
                        });
                    }
                }
            }
            KernelIrOpKind::Call { target, .. } => {
                edges.push(SassCfgEdge {
                    from_block: block.id,
                    to_block: None,
                    kind: SassCfgEdgeKind::Call,
                    condition: None,
                    target: target.clone(),
                });
                if let Some(next_block) = next_block {
                    edges.push(SassCfgEdge {
                        from_block: block.id,
                        to_block: Some(next_block),
                        kind: SassCfgEdgeKind::Fallthrough,
                        condition: None,
                        target: None,
                    });
                }
            }
            KernelIrOpKind::Return { target, .. } => {
                edges.push(SassCfgEdge {
                    from_block: block.id,
                    to_block: None,
                    kind: SassCfgEdgeKind::Return,
                    condition: last_op.predicate.clone(),
                    target: target.clone(),
                });
                if last_op.predicate.is_some() {
                    if let Some(next_block) = next_block {
                        edges.push(SassCfgEdge {
                            from_block: block.id,
                            to_block: Some(next_block),
                            kind: SassCfgEdgeKind::Fallthrough,
                            condition: None,
                            target: None,
                        });
                    }
                }
            }
            KernelIrOpKind::Exit { condition } => {
                edges.push(SassCfgEdge {
                    from_block: block.id,
                    to_block: None,
                    kind: SassCfgEdgeKind::Exit,
                    condition: condition.clone(),
                    target: None,
                });
                if condition.is_some() {
                    if let Some(next_block) = next_block {
                        edges.push(SassCfgEdge {
                            from_block: block.id,
                            to_block: Some(next_block),
                            kind: SassCfgEdgeKind::Fallthrough,
                            condition: None,
                            target: None,
                        });
                    }
                }
            }
            _ => {
                if let Some(next_block) = next_block {
                    edges.push(SassCfgEdge {
                        from_block: block.id,
                        to_block: Some(next_block),
                        kind: SassCfgEdgeKind::Fallthrough,
                        condition: None,
                        target: None,
                    });
                }
            }
        }
    }
    edges
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReachingDef {
    register: String,
    address: Option<u64>,
}

fn analyze_reaching_defs(
    blocks: &[SassBasicBlock],
    edges: &[SassCfgEdge],
    dataflow: &[SassDataflowOp],
) -> (Vec<SassReachingUse>, Vec<SassLiveRange>) {
    if blocks.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut definitions = Vec::new();
    let mut def_by_op_register = BTreeMap::<(usize, String), usize>::new();
    let mut defs_by_register = BTreeMap::<String, BTreeSet<usize>>::new();
    let mut registers = BTreeSet::<String>::new();
    for op in dataflow {
        registers.extend(op.defines.iter().cloned());
        registers.extend(op.uses.iter().cloned());
    }
    for register in registers {
        let def_id = definitions.len();
        definitions.push(ReachingDef {
            register: register.clone(),
            address: None,
        });
        defs_by_register.entry(register).or_default().insert(def_id);
    }
    for (op_index, op) in dataflow.iter().enumerate() {
        for register in &op.defines {
            let def_id = definitions.len();
            definitions.push(ReachingDef {
                register: register.clone(),
                address: Some(op.address),
            });
            def_by_op_register.insert((op_index, register.clone()), def_id);
            defs_by_register
                .entry(register.clone())
                .or_default()
                .insert(def_id);
        }
    }

    let predecessors = predecessors_by_block(blocks.len(), edges);
    let entry_defs = definitions
        .iter()
        .enumerate()
        .filter_map(|(def_id, definition)| definition.address.is_none().then_some(def_id))
        .collect::<BTreeSet<_>>();
    let mut in_sets = vec![BTreeSet::<usize>::new(); blocks.len()];
    let mut out_sets = vec![BTreeSet::<usize>::new(); blocks.len()];
    loop {
        let mut changed = false;
        for block in blocks {
            let mut next_in = BTreeSet::new();
            if block.id == 0 {
                next_in.extend(entry_defs.iter().copied());
            }
            for pred in &predecessors[block.id] {
                next_in.extend(out_sets[*pred].iter().copied());
            }
            let next_out = transfer_block(
                block,
                dataflow,
                &def_by_op_register,
                &defs_by_register,
                next_in.clone(),
            );
            if next_in != in_sets[block.id] || next_out != out_sets[block.id] {
                in_sets[block.id] = next_in;
                out_sets[block.id] = next_out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut reaching_uses = Vec::new();
    let mut live_range_uses = BTreeMap::<(String, Option<u64>), BTreeSet<u64>>::new();
    for definition in &definitions {
        if let Some(address) = definition.address {
            live_range_uses
                .entry((definition.register.clone(), Some(address)))
                .or_default();
        }
    }

    for block in blocks {
        let mut state = in_sets[block.id].clone();
        for op_index in block.start_op_index..=block.end_op_index {
            let op = &dataflow[op_index];
            for register in &op.uses {
                let reaching = reaching_defs_for_register(&state, &definitions, register);
                let reaches_entry = reaching.is_empty();
                let reaches_entry = reaches_entry
                    || reaching
                        .iter()
                        .any(|def_id| definitions[*def_id].address.is_none());
                let reaching_def_addresses = reaching
                    .iter()
                    .filter_map(|def_id| definitions[*def_id].address)
                    .collect::<Vec<_>>();
                if reaches_entry {
                    live_range_uses
                        .entry((register.clone(), None))
                        .or_default()
                        .insert(op.address);
                }
                for address in &reaching_def_addresses {
                    live_range_uses
                        .entry((register.clone(), Some(*address)))
                        .or_default()
                        .insert(op.address);
                }
                reaching_uses.push(SassReachingUse {
                    address: op.address,
                    register: register.clone(),
                    reaching_def_addresses,
                    reaches_entry,
                });
            }
            apply_defs(
                op_index,
                op,
                &mut state,
                &def_by_op_register,
                &defs_by_register,
            );
        }
    }

    let mut live_ranges = live_range_uses
        .into_iter()
        .map(|((register, def_address), uses)| {
            let start_address = def_address
                .or_else(|| uses.iter().next().copied())
                .unwrap_or(0);
            let end_address = uses.iter().next_back().copied().unwrap_or(start_address);
            SassLiveRange {
                register,
                def_address,
                start_address,
                end_address,
                use_addresses: uses.into_iter().collect(),
            }
        })
        .collect::<Vec<_>>();
    live_ranges.sort_by(|lhs, rhs| {
        lhs.register
            .cmp(&rhs.register)
            .then_with(|| lhs.start_address.cmp(&rhs.start_address))
            .then_with(|| lhs.def_address.cmp(&rhs.def_address))
    });
    reaching_uses.sort_by(|lhs, rhs| {
        lhs.address
            .cmp(&rhs.address)
            .then_with(|| lhs.register.cmp(&rhs.register))
    });
    (reaching_uses, live_ranges)
}

fn predecessors_by_block(block_count: usize, edges: &[SassCfgEdge]) -> Vec<Vec<usize>> {
    let mut predecessors = vec![Vec::new(); block_count];
    for edge in edges {
        if let Some(to_block) = edge.to_block {
            if to_block < block_count && !predecessors[to_block].contains(&edge.from_block) {
                predecessors[to_block].push(edge.from_block);
            }
        }
    }
    predecessors
}

fn transfer_block(
    block: &SassBasicBlock,
    dataflow: &[SassDataflowOp],
    def_by_op_register: &BTreeMap<(usize, String), usize>,
    defs_by_register: &BTreeMap<String, BTreeSet<usize>>,
    mut state: BTreeSet<usize>,
) -> BTreeSet<usize> {
    for op_index in block.start_op_index..=block.end_op_index {
        apply_defs(
            op_index,
            &dataflow[op_index],
            &mut state,
            def_by_op_register,
            defs_by_register,
        );
    }
    state
}

fn apply_defs(
    op_index: usize,
    op: &SassDataflowOp,
    state: &mut BTreeSet<usize>,
    def_by_op_register: &BTreeMap<(usize, String), usize>,
    defs_by_register: &BTreeMap<String, BTreeSet<usize>>,
) {
    let conditional_write = op.predicate.is_some();
    for register in &op.defines {
        if !conditional_write {
            if let Some(kill_set) = defs_by_register.get(register) {
                for def_id in kill_set {
                    state.remove(def_id);
                }
            }
        }
        if let Some(def_id) = def_by_op_register.get(&(op_index, register.clone())) {
            state.insert(*def_id);
        }
    }
}

fn reaching_defs_for_register(
    state: &BTreeSet<usize>,
    definitions: &[ReachingDef],
    register: &str,
) -> Vec<usize> {
    state
        .iter()
        .copied()
        .filter(|def_id| definitions[*def_id].register == register)
        .collect()
}

fn analyze_dataflow(op: &KernelIrOp) -> SassDataflowOp {
    let mut defines = Vec::new();
    let mut uses = Vec::new();
    if let Some(predicate) = &op.predicate {
        push_registers(predicate, &mut uses);
    }
    match &op.kind {
        KernelIrOpKind::ReadSpecialRegister { dst, special } => {
            push_registers(dst, &mut defines);
            push_registers(special, &mut uses);
        }
        KernelIrOpKind::Move { dst, src } => {
            push_registers(dst, &mut defines);
            push_registers(src, &mut uses);
        }
        KernelIrOpKind::LoadConst { dst, source } => {
            push_registers(dst, &mut defines);
            push_registers(source, &mut uses);
        }
        KernelIrOpKind::Load { dst, address, .. } => {
            push_registers(dst, &mut defines);
            push_registers(address, &mut uses);
        }
        KernelIrOpKind::Store { address, value, .. } => {
            push_registers(address, &mut uses);
            push_registers(value, &mut uses);
        }
        KernelIrOpKind::IntegerAdd { dst, inputs, .. }
        | KernelIrOpKind::PackedHalfAdd { dst, inputs, .. }
        | KernelIrOpKind::PackedHalfMul { dst, inputs, .. }
        | KernelIrOpKind::Shift { dst, inputs }
        | KernelIrOpKind::LogicLut { dst, inputs }
        | KernelIrOpKind::Permute { dst, inputs }
        | KernelIrOpKind::AddressCalc { dst, inputs } => {
            push_registers(dst, &mut defines);
            for input in inputs {
                push_registers(input, &mut uses);
            }
        }
        KernelIrOpKind::FloatAdd { dst, lhs, rhs } | KernelIrOpKind::FloatMul { dst, lhs, rhs } => {
            push_registers(dst, &mut defines);
            push_registers(lhs, &mut uses);
            push_registers(rhs, &mut uses);
        }
        KernelIrOpKind::FusedMultiplyAdd { dst, a, b, c, .. }
        | KernelIrOpKind::IntegerMad { dst, a, b, c, .. } => {
            push_registers(dst, &mut defines);
            push_registers(a, &mut uses);
            push_registers(b, &mut uses);
            push_registers(c, &mut uses);
        }
        KernelIrOpKind::CompareSet { dst, lhs, rhs, .. } => {
            push_registers(dst, &mut defines);
            push_registers(lhs, &mut uses);
            push_registers(rhs, &mut uses);
        }
        KernelIrOpKind::Branch { condition, .. } | KernelIrOpKind::Exit { condition } => {
            if let Some(condition) = condition {
                push_registers(condition, &mut uses);
            }
        }
        KernelIrOpKind::Call { operands, .. }
        | KernelIrOpKind::Return { operands, .. }
        | KernelIrOpKind::Sync { operands, .. } => {
            for operand in operands {
                push_registers(operand, &mut uses);
            }
        }
        KernelIrOpKind::WarpShuffle {
            predicate,
            dst,
            src,
            offset,
            mask,
            ..
        } => {
            push_registers(dst, &mut defines);
            push_registers(predicate, &mut uses);
            push_registers(src, &mut uses);
            push_registers(offset, &mut uses);
            push_registers(mask, &mut uses);
        }
        KernelIrOpKind::NoOp | KernelIrOpKind::Unsupported { .. } => {}
    }
    SassDataflowOp {
        address: op.address,
        predicate: op.predicate.clone(),
        defines,
        uses,
        source: op.source.clone(),
    }
}

fn is_block_boundary_terminator(op: &KernelIrOp) -> bool {
    matches!(
        op.kind,
        KernelIrOpKind::Branch { .. }
            | KernelIrOpKind::Call { .. }
            | KernelIrOpKind::Return { .. }
            | KernelIrOpKind::Exit { .. }
    )
}

fn terminator_for(op: &KernelIrOp) -> SassBlockTerminator {
    match op.kind {
        KernelIrOpKind::Branch { .. } => SassBlockTerminator::Branch,
        KernelIrOpKind::Call { .. } => SassBlockTerminator::Call,
        KernelIrOpKind::Return { .. } => SassBlockTerminator::Return,
        KernelIrOpKind::Exit { .. } => SassBlockTerminator::Exit,
        _ => SassBlockTerminator::Fallthrough,
    }
}

fn branch_target(op: &KernelIrOp) -> Option<&str> {
    match &op.kind {
        KernelIrOpKind::Branch {
            target: Some(target),
            ..
        } => Some(target),
        _ => None,
    }
}

fn label_index(function: &KernelIrFunction, label: &str) -> Option<usize> {
    function
        .ops
        .iter()
        .position(|op| op.label.as_deref() == Some(label))
}

fn block_id_for_op_index(blocks: &[SassBasicBlock], op_index: usize) -> Option<usize> {
    blocks
        .iter()
        .find(|block| block.start_op_index <= op_index && op_index <= block.end_op_index)
        .map(|block| block.id)
}

fn push_registers(text: &str, out: &mut Vec<String>) {
    for register in extract_registers(text) {
        if !out.contains(&register) {
            out.push(register);
        }
    }
}

fn extract_registers(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut index = 0usize;
    let mut registers = Vec::new();
    while index < bytes.len() {
        let Some((register, consumed)) = parse_register_at(text, index) else {
            index += 1;
            continue;
        };
        if !is_pseudo_register(&register) && !registers.contains(&register) {
            registers.push(register);
        }
        index += consumed;
    }
    registers
}

fn parse_register_at(text: &str, index: usize) -> Option<(String, usize)> {
    if !is_token_boundary(text, index) {
        return None;
    }
    let tail = &text[index..];
    for literal in ["SR_", "URZ", "UPT", "UR", "UP", "RZ", "PT", "R", "P", "B"] {
        if let Some(register) = parse_register_prefix(tail, literal) {
            return Some(register);
        }
    }
    None
}

fn parse_register_prefix(tail: &str, prefix: &str) -> Option<(String, usize)> {
    let rest = tail.strip_prefix(prefix)?;
    match prefix {
        "SR_" => {
            let len = rest
                .char_indices()
                .take_while(|(_, ch)| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '.')
                .map(|(index, ch)| index + ch.len_utf8())
                .last()
                .unwrap_or(0);
            (len > 0).then(|| (tail[..prefix.len() + len].to_string(), prefix.len() + len))
        }
        "URZ" | "UPT" | "RZ" | "PT" => Some((prefix.to_string(), prefix.len())),
        "UR" | "UP" | "R" | "P" | "B" => {
            let len = rest
                .char_indices()
                .take_while(|(_, ch)| ch.is_ascii_digit())
                .map(|(index, ch)| index + ch.len_utf8())
                .last()
                .unwrap_or(0);
            (len > 0).then(|| (tail[..prefix.len() + len].to_string(), prefix.len() + len))
        }
        _ => None,
    }
}

fn is_token_boundary(text: &str, index: usize) -> bool {
    if index == 0 {
        return true;
    }
    let before = text[..index]
        .chars()
        .next_back()
        .expect("index > 0 should have previous char");
    !before.is_ascii_alphanumeric() && before != '_'
}

fn is_pseudo_register(register: &str) -> bool {
    matches!(register, "RZ" | "URZ" | "PT" | "UPT")
}

fn format_addresses(addresses: &[u64]) -> String {
    addresses
        .iter()
        .map(|address| format!("{address:#06x}"))
        .collect::<Vec<_>>()
        .join(",")
}
