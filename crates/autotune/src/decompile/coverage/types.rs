use std::{collections::BTreeSet, fmt, path::PathBuf};

use nn_rust_inference::runtime;

use super::super::{
    AggregateOperand, ControlTarget, MemoryAddress, MemoryAddressBase, MemoryAddressImmediate,
    MemorySpace, PredicateCondition, RegisterRef, SassArchitecture, SassBlockTerminator,
    SassCfgEdgeKind, SassDataflowSite, SassInstruction, SassLiftedOpClass, SassLiftedOpDetail,
    SassLiftedOpKind, SassLiftedSemantics, SassLiftedValueRef, SassMemoryAccessKind, SassModifier,
    SassOpcode, SassOpcodeCatalogClass, SassOpcodeCatalogKind, SassOpcodeCatalogSource,
    SassParseError, SassPatternConfidence, SassRegionKind, SassRegionPath,
    SassSemanticPatternCategory, SassSemanticPatternKind, SassSymbol, SassTarget,
    SassUnsupportedReason, SassValueOpKind,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageOptions {
    pub root: PathBuf,
    pub output_dir: PathBuf,
}

impl SassCoverageOptions {
    pub fn default_artifact_scan() -> Self {
        let artifact_root = runtime::default_artifact_dir();
        Self {
            root: artifact_root.clone(),
            output_dir: artifact_root.join("decompile-coverage"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageReport {
    pub root: PathBuf,
    pub output_dir: PathBuf,
    pub summary_path: PathBuf,
    pub files_path: PathBuf,
    pub opcode_catalog_path: PathBuf,
    pub opcode_probe_targets_path: PathBuf,
    pub opcode_frequency_path: PathBuf,
    pub opcode_signature_frequency_path: PathBuf,
    pub semantic_patterns_path: PathBuf,
    pub semantic_pattern_frequency_path: PathBuf,
    pub cfg_blocks_path: PathBuf,
    pub cfg_edges_path: PathBuf,
    pub dominators_path: PathBuf,
    pub natural_loops_path: PathBuf,
    pub regions_path: PathBuf,
    pub dataflow_path: PathBuf,
    pub reaching_uses_path: PathBuf,
    pub ssa_values_path: PathBuf,
    pub def_use_edges_path: PathBuf,
    pub value_ops_path: PathBuf,
    pub lifted_ops_path: PathBuf,
    pub live_ranges_path: PathBuf,
    pub memory_accesses_path: PathBuf,
    pub unsupported_instructions_path: PathBuf,
    pub files: Vec<SassCoverageFileReport>,
    pub scanned_architectures: Vec<SassArchitecture>,
    pub opcode_catalog: Vec<SassOpcodeCatalogEntry>,
    pub opcode_probe_targets: Vec<SassOpcodeProbeTarget>,
    pub opcode_counts: Vec<SassOpcodeCount>,
    pub opcode_signature_counts: Vec<SassOpcodeSignatureCount>,
    pub semantic_pattern_counts: Vec<SassSemanticPatternCount>,
    pub semantic_patterns: Vec<SassCoverageSemanticPattern>,
    pub cfg_blocks: Vec<SassCoverageBasicBlock>,
    pub cfg_edges: Vec<SassCoverageCfgEdge>,
    pub dominators: Vec<SassCoverageDominatorBlock>,
    pub natural_loops: Vec<SassCoverageNaturalLoop>,
    pub regions: Vec<SassCoverageRegion>,
    pub dataflow: Vec<SassCoverageDataflowOp>,
    pub reaching_uses: Vec<SassCoverageReachingUse>,
    pub ssa_values: Vec<SassCoverageSsaValue>,
    pub def_use_edges: Vec<SassCoverageDefUseEdge>,
    pub value_ops: Vec<SassCoverageValueOp>,
    pub lifted_ops: Vec<SassCoverageLiftedOp>,
    pub live_ranges: Vec<SassCoverageLiveRange>,
    pub memory_accesses: Vec<SassCoverageMemoryAccess>,
    pub unsupported_instructions: Vec<SassUnsupportedInstruction>,
    pub parsed_file_count: usize,
    pub parse_error_count: usize,
    pub parsed_instruction_count: usize,
    pub cfg_block_count: usize,
    pub cfg_edge_count: usize,
    pub dominator_block_count: usize,
    pub natural_loop_count: usize,
    pub region_count: usize,
    pub dataflow_op_count: usize,
    pub reaching_use_count: usize,
    pub ssa_value_count: usize,
    pub def_use_edge_count: usize,
    pub value_op_count: usize,
    pub lifted_op_count: usize,
    pub live_range_count: usize,
    pub memory_access_count: usize,
    pub semantic_pattern_count: usize,
    pub known_opcode_count: usize,
    pub locally_mapped_opcode_count: usize,
    pub known_unobserved_opcode_count: usize,
    pub opcode_probe_target_count: usize,
    pub known_unmapped_opcode_count: usize,
    pub observed_unregistered_opcode_count: usize,
    pub observed_unmapped_opcode_count: usize,
    pub unsupported_instruction_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageFileReport {
    pub sass_path: PathBuf,
    pub target: Option<SassTarget>,
    pub ir_path: Option<PathBuf>,
    pub lifted_ir_path: Option<PathBuf>,
    pub analysis_path: Option<PathBuf>,
    pub pattern_path: Option<PathBuf>,
    pub side_by_side_path: Option<PathBuf>,
    pub parsed_instruction_count: usize,
    pub cfg_block_count: usize,
    pub cfg_edge_count: usize,
    pub dominator_block_count: usize,
    pub natural_loop_count: usize,
    pub region_count: usize,
    pub reaching_use_count: usize,
    pub ssa_value_count: usize,
    pub def_use_edge_count: usize,
    pub value_op_count: usize,
    pub lifted_op_count: usize,
    pub live_range_count: usize,
    pub memory_access_count: usize,
    pub semantic_pattern_count: usize,
    pub unsupported_instruction_count: usize,
    pub parse_error: Option<SassParseError>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassOpcodeCount {
    pub opcode: SassOpcode,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassOpcodeSignatureCount {
    pub signature: SassOpcodeSignature,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassSemanticPatternCount {
    pub category: SassSemanticPatternCategory,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassOpcodeProbeTarget {
    pub opcode: SassOpcode,
    pub priority: u8,
    pub architectures: Vec<SassArchitecture>,
    pub matching_scanned_architectures: Vec<SassArchitecture>,
    pub classes: Vec<SassOpcodeCatalogClass>,
    pub kinds: Vec<SassOpcodeCatalogKind>,
    pub known_sources: Vec<SassOpcodeCatalogSource>,
    pub locally_mapped: bool,
    pub recommended_action: SassOpcodeProbeAction,
    pub reason: SassOpcodeProbeReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassOpcodeProbeAction {
    GenerateSassArtifact,
    ScanArchitectureArtifact,
    AddLifterMapping,
}

impl fmt::Display for SassOpcodeProbeAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GenerateSassArtifact => f.write_str("generate-sass-artifact"),
            Self::ScanArchitectureArtifact => f.write_str("scan-architecture-artifact"),
            Self::AddLifterMapping => f.write_str("add-lifter-mapping"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassOpcodeProbeReason {
    MissingLocalLifterMapping,
    ArchitectureArtifactNotScanned,
    TensorCoreMappedUnobserved,
    TensorMemoryMappedUnobserved,
    WarpGroupMappedUnobserved,
    ArchitectureSpecificMappedUnobserved,
    ScalarMappedUnobserved,
}

impl fmt::Display for SassOpcodeProbeReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingLocalLifterMapping => {
                f.write_str("known opcode has no local lifter mapping")
            }
            Self::ArchitectureArtifactNotScanned => f.write_str(
                "mapped opcode requires a SASS artifact for an architecture not present in this scan",
            ),
            Self::TensorCoreMappedUnobserved => f.write_str(
                "tensor-core opcode is mapped but unobserved in generated SASS artifacts",
            ),
            Self::TensorMemoryMappedUnobserved => f.write_str(
                "tensor-memory opcode is mapped but unobserved in generated SASS artifacts",
            ),
            Self::WarpGroupMappedUnobserved => {
                f.write_str("warpgroup opcode is mapped but unobserved in generated SASS artifacts")
            }
            Self::ArchitectureSpecificMappedUnobserved => f.write_str(
                "architecture-specific opcode is mapped but unobserved in generated SASS artifacts",
            ),
            Self::ScalarMappedUnobserved => {
                f.write_str("mapped scalar opcode is unobserved in generated SASS artifacts")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassOpcodeCatalogEntry {
    pub opcode: SassOpcode,
    pub known: bool,
    pub observed: bool,
    pub locally_mapped: bool,
    pub instruction_count: usize,
    pub signature_count: usize,
    pub signatures: Vec<SassOpcodeSignature>,
    pub source_formats: Vec<SassCoverageSourceFormat>,
    pub architectures: Vec<SassArchitecture>,
    pub observed_architectures: Vec<SassArchitecture>,
    pub known_sources: Vec<SassOpcodeCatalogSource>,
    pub classes: Vec<SassOpcodeCatalogClass>,
    pub kinds: Vec<SassOpcodeCatalogKind>,
    pub support: SassOpcodeSupport,
    pub coverage: SassOpcodeCoverageState,
    pub unsupported_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassOpcodeSupport {
    Mapped,
    Unsupported,
    Mixed,
    Unmapped,
}

impl SassOpcodeSupport {
    fn from_counts(
        locally_mapped: bool,
        observed: bool,
        instruction_count: usize,
        unsupported_count: usize,
    ) -> Self {
        if locally_mapped && unsupported_count == 0 {
            Self::Mapped
        } else if observed && unsupported_count == instruction_count {
            Self::Unsupported
        } else if observed && unsupported_count > 0 {
            Self::Mixed
        } else {
            Self::Unmapped
        }
    }
}

impl fmt::Display for SassOpcodeSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mapped => f.write_str("mapped"),
            Self::Unsupported => f.write_str("unsupported"),
            Self::Mixed => f.write_str("mixed"),
            Self::Unmapped => f.write_str("unmapped"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassOpcodeCoverageState {
    KnownObservedMapped,
    KnownObservedPartial,
    KnownObservedUnmapped,
    KnownUnobservedMapped,
    KnownUnobservedUnmapped,
    ObservedMapped,
    ObservedPartial,
    ObservedUnregisteredUnmapped,
    Empty,
    Absent,
}

impl SassOpcodeCoverageState {
    fn from_catalog(known: bool, observed: bool, support: SassOpcodeSupport) -> Self {
        match (known, observed, support) {
            (true, true, SassOpcodeSupport::Mapped) => Self::KnownObservedMapped,
            (true, true, SassOpcodeSupport::Mixed) => Self::KnownObservedPartial,
            (true, true, _) => Self::KnownObservedUnmapped,
            (true, false, SassOpcodeSupport::Mapped) => Self::KnownUnobservedMapped,
            (true, false, _) => Self::KnownUnobservedUnmapped,
            (false, true, SassOpcodeSupport::Mapped) => Self::ObservedMapped,
            (false, true, SassOpcodeSupport::Mixed) => Self::ObservedPartial,
            (false, true, _) => Self::ObservedUnregisteredUnmapped,
            (false, false, _) => Self::Empty,
        }
    }
}

impl fmt::Display for SassOpcodeCoverageState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KnownObservedMapped => f.write_str("known-observed-mapped"),
            Self::KnownObservedPartial => f.write_str("known-observed-partial"),
            Self::KnownObservedUnmapped => f.write_str("known-observed-unmapped"),
            Self::KnownUnobservedMapped => f.write_str("known-unobserved-mapped"),
            Self::KnownUnobservedUnmapped => f.write_str("known-unobserved-unmapped"),
            Self::ObservedMapped => f.write_str("observed-mapped"),
            Self::ObservedPartial => f.write_str("observed-partial"),
            Self::ObservedUnregisteredUnmapped => f.write_str("observed-unregistered-unmapped"),
            Self::Empty => f.write_str("empty"),
            Self::Absent => f.write_str("absent"),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct OpcodeCatalogBuilder {
    pub(super) known: bool,
    pub(super) locally_mapped: bool,
    pub(super) instruction_count: usize,
    pub(super) signatures: BTreeSet<SassOpcodeSignature>,
    pub(super) source_formats: BTreeSet<SassCoverageSourceFormat>,
    pub(super) architectures: BTreeSet<SassArchitecture>,
    pub(super) observed_architectures: BTreeSet<SassArchitecture>,
    pub(super) known_sources: BTreeSet<SassOpcodeCatalogSource>,
    pub(super) classes: BTreeSet<SassOpcodeCatalogClass>,
    pub(super) kinds: BTreeSet<SassOpcodeCatalogKind>,
    pub(super) unsupported_count: usize,
}

impl OpcodeCatalogBuilder {
    pub(super) fn into_entry(self, opcode: SassOpcode) -> SassOpcodeCatalogEntry {
        let observed = self.instruction_count > 0;
        let support = SassOpcodeSupport::from_counts(
            self.locally_mapped,
            observed,
            self.instruction_count,
            self.unsupported_count,
        );
        let coverage = SassOpcodeCoverageState::from_catalog(self.known, observed, support);
        let signatures = self.signatures.into_iter().collect::<Vec<_>>();
        SassOpcodeCatalogEntry {
            opcode,
            known: self.known,
            observed,
            locally_mapped: self.locally_mapped,
            instruction_count: self.instruction_count,
            signature_count: signatures.len(),
            signatures,
            source_formats: self.source_formats.into_iter().collect(),
            architectures: self.architectures.into_iter().collect(),
            observed_architectures: self.observed_architectures.into_iter().collect(),
            known_sources: self.known_sources.into_iter().collect(),
            classes: self.classes.into_iter().collect(),
            kinds: self.kinds.into_iter().collect(),
            support,
            coverage,
            unsupported_count: self.unsupported_count,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SassOpcodeSignature {
    pub opcode: SassOpcode,
    pub modifiers: Vec<SassModifier>,
}

impl SassOpcodeSignature {
    pub(super) fn from_instruction(instruction: &SassInstruction) -> Self {
        Self {
            opcode: SassOpcode::new(instruction.opcode.clone()),
            modifiers: instruction
                .modifiers
                .iter()
                .map(|modifier| SassModifier::parse(modifier.as_str()))
                .collect(),
        }
    }
}

impl fmt::Display for SassOpcodeSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.opcode)?;
        for modifier in &self.modifiers {
            write!(f, ".{modifier}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassCoverageSourceFormat {
    Cuobjdump,
    Nvdisasm,
    Sass,
}

impl fmt::Display for SassCoverageSourceFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuobjdump => f.write_str("cuobjdump"),
            Self::Nvdisasm => f.write_str("nvdisasm"),
            Self::Sass => f.write_str("sass"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassUnsupportedInstruction {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub address: u64,
    pub opcode: SassOpcode,
    pub reason: SassUnsupportedReason,
    pub source_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageSemanticPattern {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub start_address: u64,
    pub end_address: u64,
    pub kind: SassSemanticPatternKind,
    pub confidence: SassPatternConfidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageBasicBlock {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub id: usize,
    pub label: Option<SassSymbol>,
    pub start_address: u64,
    pub end_address: u64,
    pub instruction_count: usize,
    pub terminator: SassBlockTerminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageCfgEdge {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub from_block: usize,
    pub to_block: Option<usize>,
    pub kind: SassCfgEdgeKind,
    pub condition: Option<PredicateCondition>,
    pub target: Option<ControlTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageDominatorBlock {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub block_id: usize,
    pub reachable: bool,
    pub immediate_dominator: Option<usize>,
    pub dominators: Vec<usize>,
    pub dominated_blocks: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageNaturalLoop {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub header_block: usize,
    pub latch_block: usize,
    pub reachable: bool,
    pub blocks: Vec<usize>,
    pub edge_condition: Option<PredicateCondition>,
    pub edge_target: Option<ControlTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageRegion {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
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
    pub opcode_closure: Vec<SassOpcode>,
    pub condition: Option<PredicateCondition>,
    pub target: Option<ControlTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageDataflowOp {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub address: u64,
    pub defines: Vec<RegisterRef>,
    pub uses: Vec<RegisterRef>,
    pub source_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageReachingUse {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub address: u64,
    pub register: RegisterRef,
    pub reaching_def_addresses: Vec<u64>,
    pub reaches_entry: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageSsaValue {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub value_id: usize,
    pub register: RegisterRef,
    pub def_address: Option<u64>,
    pub origin: SassDataflowSite,
    pub use_addresses: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageDefUseEdge {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub value_id: usize,
    pub register: RegisterRef,
    pub def_address: Option<u64>,
    pub use_address: u64,
    pub use_site: SassDataflowSite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageValueOp {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub address: u64,
    pub block_id: Option<usize>,
    pub predicate: Option<PredicateCondition>,
    pub opcode: SassOpcode,
    pub kind: SassValueOpKind,
    pub input_registers: Vec<RegisterRef>,
    pub output_registers: Vec<RegisterRef>,
    pub input_value_ids: Vec<usize>,
    pub output_value_ids: Vec<usize>,
    pub source_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageLiftedOp {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub address: u64,
    pub block_id: Option<usize>,
    pub predicate: Option<PredicateCondition>,
    pub opcode: SassOpcode,
    pub class: SassLiftedOpClass,
    pub kind: SassLiftedOpKind,
    pub semantics: SassLiftedSemantics,
    pub inputs: Vec<SassLiftedValueRef>,
    pub outputs: Vec<SassLiftedValueRef>,
    pub source_operands: Vec<AggregateOperand>,
    pub detail: SassLiftedOpDetail,
    pub source_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageLiveRange {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub register: RegisterRef,
    pub def_address: Option<u64>,
    pub start_address: u64,
    pub end_address: u64,
    pub use_addresses: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageMemoryAccess {
    pub sass_path: PathBuf,
    pub function: SassSymbol,
    pub address: u64,
    pub predicate: Option<PredicateCondition>,
    pub kind: SassMemoryAccessKind,
    pub space: MemorySpace,
    pub width_bits: Option<u32>,
    pub value_register: RegisterRef,
    pub memory_address: MemoryAddress,
    pub address_registers: Vec<RegisterRef>,
    pub address_base: Option<MemoryAddressBase>,
    pub offset: Option<MemoryAddressImmediate>,
    pub source_text: String,
}
