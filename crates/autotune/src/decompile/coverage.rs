use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt::{self, Write as _},
    fs,
    path::{Path, PathBuf},
};

use nn_rust_inference::runtime;

use super::{
    ControlTarget, KernelIrModule, KernelIrOpKind, KnownSassOpcode, MemoryAddressBase,
    MemoryAddressImmediate, MemorySpace, PredicateCondition, RegisterRef, SassAnalysisModule,
    SassBlockTerminator, SassCfgEdgeKind, SassLiftedModule, SassLiftedOpClass, SassLiftedOpDetail,
    SassLiftedOpKind, SassLiftedSemantics, SassLiftedValueRef, SassMemoryAccessKind, SassModifier,
    SassOpcode, SassOpcodeCatalogClass, SassOpcodeCatalogKind, SassOpcodeCatalogSource,
    SassPatternConfidence, SassPatternModule, SassRegionKind, SassRegionPath,
    SassSemanticPatternCategory, SassSemanticPatternKind, SassValueOpKind, analyze_sass_ir,
    known_sass_opcodes, lift_sass_value_ir, parse_nvidia_sass, recover_sass_patterns,
    render_sass_file_side_by_side,
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
    pub parse_error: Option<String>,
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
    pub architectures: Vec<String>,
    pub classes: Vec<SassOpcodeCatalogClass>,
    pub kinds: Vec<SassOpcodeCatalogKind>,
    pub known_sources: Vec<SassOpcodeCatalogSource>,
    pub locally_mapped: bool,
    pub recommended_action: String,
    pub reason: String,
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
    pub architectures: Vec<String>,
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
struct OpcodeCatalogBuilder {
    known: bool,
    locally_mapped: bool,
    instruction_count: usize,
    signatures: BTreeSet<SassOpcodeSignature>,
    source_formats: BTreeSet<SassCoverageSourceFormat>,
    architectures: BTreeSet<String>,
    known_sources: BTreeSet<SassOpcodeCatalogSource>,
    classes: BTreeSet<SassOpcodeCatalogClass>,
    kinds: BTreeSet<SassOpcodeCatalogKind>,
    unsupported_count: usize,
}

impl OpcodeCatalogBuilder {
    fn into_entry(self, opcode: SassOpcode) -> SassOpcodeCatalogEntry {
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
    fn from_instruction(instruction: &super::SassInstruction) -> Self {
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
    pub function: String,
    pub address: u64,
    pub opcode: SassOpcode,
    pub reason: String,
    pub raw: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageSemanticPattern {
    pub sass_path: PathBuf,
    pub function: String,
    pub start_address: u64,
    pub end_address: u64,
    pub kind: SassSemanticPatternKind,
    pub confidence: SassPatternConfidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageBasicBlock {
    pub sass_path: PathBuf,
    pub function: String,
    pub id: usize,
    pub label: Option<String>,
    pub start_address: u64,
    pub end_address: u64,
    pub instruction_count: usize,
    pub terminator: SassBlockTerminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageCfgEdge {
    pub sass_path: PathBuf,
    pub function: String,
    pub from_block: usize,
    pub to_block: Option<usize>,
    pub kind: SassCfgEdgeKind,
    pub condition: Option<PredicateCondition>,
    pub target: Option<ControlTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageDominatorBlock {
    pub sass_path: PathBuf,
    pub function: String,
    pub block_id: usize,
    pub reachable: bool,
    pub immediate_dominator: Option<usize>,
    pub dominators: Vec<usize>,
    pub dominated_blocks: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageNaturalLoop {
    pub sass_path: PathBuf,
    pub function: String,
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
    pub function: String,
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
    pub function: String,
    pub address: u64,
    pub defines: Vec<RegisterRef>,
    pub uses: Vec<RegisterRef>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageReachingUse {
    pub sass_path: PathBuf,
    pub function: String,
    pub address: u64,
    pub register: RegisterRef,
    pub reaching_def_addresses: Vec<u64>,
    pub reaches_entry: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageSsaValue {
    pub sass_path: PathBuf,
    pub function: String,
    pub value_id: usize,
    pub register: RegisterRef,
    pub def_address: Option<u64>,
    pub source: Option<String>,
    pub use_addresses: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageDefUseEdge {
    pub sass_path: PathBuf,
    pub function: String,
    pub value_id: usize,
    pub register: RegisterRef,
    pub def_address: Option<u64>,
    pub use_address: u64,
    pub use_source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageValueOp {
    pub sass_path: PathBuf,
    pub function: String,
    pub address: u64,
    pub block_id: Option<usize>,
    pub predicate: Option<String>,
    pub opcode: SassOpcode,
    pub kind: SassValueOpKind,
    pub input_registers: Vec<RegisterRef>,
    pub output_registers: Vec<RegisterRef>,
    pub input_value_ids: Vec<usize>,
    pub output_value_ids: Vec<usize>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageLiftedOp {
    pub sass_path: PathBuf,
    pub function: String,
    pub address: u64,
    pub block_id: Option<usize>,
    pub predicate: Option<String>,
    pub opcode: SassOpcode,
    pub class: SassLiftedOpClass,
    pub kind: SassLiftedOpKind,
    pub semantics: SassLiftedSemantics,
    pub inputs: Vec<SassLiftedValueRef>,
    pub outputs: Vec<SassLiftedValueRef>,
    pub source_operands: Vec<String>,
    pub detail: SassLiftedOpDetail,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageLiveRange {
    pub sass_path: PathBuf,
    pub function: String,
    pub register: RegisterRef,
    pub def_address: Option<u64>,
    pub start_address: u64,
    pub end_address: u64,
    pub use_addresses: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageMemoryAccess {
    pub sass_path: PathBuf,
    pub function: String,
    pub address: u64,
    pub predicate: Option<String>,
    pub kind: SassMemoryAccessKind,
    pub space: MemorySpace,
    pub width_bits: Option<u32>,
    pub value_register: RegisterRef,
    pub address_expr: String,
    pub address_registers: Vec<RegisterRef>,
    pub address_base: Option<MemoryAddressBase>,
    pub offset: Option<MemoryAddressImmediate>,
    pub source: String,
}

pub fn run_sass_coverage_scan(
    options: &SassCoverageOptions,
) -> Result<SassCoverageReport, Box<dyn Error>> {
    let mut sass_paths = Vec::new();
    collect_sass_paths(&options.root, &mut sass_paths)?;
    sass_paths.sort();

    fs::create_dir_all(&options.output_dir)?;
    let mut files = Vec::new();
    let mut opcode_catalog = BTreeMap::<SassOpcode, OpcodeCatalogBuilder>::new();
    seed_known_opcode_catalog(&mut opcode_catalog);
    let mut opcode_counts = BTreeMap::<SassOpcode, usize>::new();
    let mut opcode_signature_counts = BTreeMap::<SassOpcodeSignature, usize>::new();
    let mut semantic_pattern_counts = BTreeMap::<SassSemanticPatternCategory, usize>::new();
    let mut semantic_patterns = Vec::new();
    let mut cfg_blocks = Vec::new();
    let mut cfg_edges = Vec::new();
    let mut dominators = Vec::new();
    let mut natural_loops = Vec::new();
    let mut regions = Vec::new();
    let mut dataflow = Vec::new();
    let mut reaching_uses = Vec::new();
    let mut ssa_values = Vec::new();
    let mut def_use_edges = Vec::new();
    let mut value_ops = Vec::new();
    let mut lifted_ops = Vec::new();
    let mut live_ranges = Vec::new();
    let mut memory_accesses = Vec::new();
    let mut unsupported_instructions = Vec::new();
    let files_output_root = options.output_dir.join("files");

    for sass_path in sass_paths {
        let sass = fs::read_to_string(&sass_path)?;
        let relative = relative_sass_path(&options.root, &sass_path);
        let source_format = sass_source_format(&sass_path);
        match parse_nvidia_sass(&sass) {
            Ok(parsed) => {
                for function in &parsed.functions {
                    for instruction in &function.instructions {
                        let opcode = SassOpcode::new(instruction.opcode.clone());
                        let signature = SassOpcodeSignature::from_instruction(instruction);
                        *opcode_counts.entry(opcode.clone()).or_default() += 1;
                        *opcode_signature_counts
                            .entry(signature.clone())
                            .or_default() += 1;
                        let catalog_entry = opcode_catalog.entry(opcode).or_default();
                        catalog_entry.instruction_count += 1;
                        catalog_entry.signatures.insert(signature);
                        catalog_entry.source_formats.insert(source_format);
                    }
                }

                let project_ir = super::lift_sass_module(&parsed);
                let analysis = analyze_sass_ir(&project_ir);
                let lifted = lift_sass_value_ir(&project_ir, &analysis);
                let patterns = recover_sass_patterns(&project_ir);
                for function in &patterns.functions {
                    for pattern in &function.patterns {
                        *semantic_pattern_counts
                            .entry(pattern.kind.category())
                            .or_default() += 1;
                    }
                }
                append_analysis(
                    &sass_path,
                    &analysis,
                    &lifted,
                    &mut cfg_blocks,
                    &mut cfg_edges,
                    &mut dominators,
                    &mut natural_loops,
                    &mut regions,
                    &mut dataflow,
                    &mut reaching_uses,
                    &mut ssa_values,
                    &mut def_use_edges,
                    &mut value_ops,
                    &mut lifted_ops,
                    &mut live_ranges,
                    &mut memory_accesses,
                );
                append_semantic_patterns(&sass_path, &patterns, &mut semantic_patterns);
                append_unsupported(&sass_path, &project_ir, &mut unsupported_instructions);
                append_opcode_catalog_lifted_ops(&lifted, &mut opcode_catalog);
                append_opcode_catalog_unsupported(&project_ir, &mut opcode_catalog);

                let ir_path = files_output_root
                    .join(&relative)
                    .with_extension("lifted.ir.txt");
                if let Some(parent) = ir_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&ir_path, project_ir.to_text().as_bytes())?;

                let lifted_ir_path = files_output_root
                    .join(&relative)
                    .with_extension("lifted-value-ir.txt");
                fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;

                let analysis_path = files_output_root
                    .join(&relative)
                    .with_extension("analysis.txt");
                fs::write(&analysis_path, analysis.to_text().as_bytes())?;

                let pattern_path = files_output_root
                    .join(&relative)
                    .with_extension("patterns.txt");
                fs::write(&pattern_path, patterns.to_text().as_bytes())?;

                let side_by_side_path = files_output_root
                    .join(&relative)
                    .with_extension("source-sass-ir.txt");
                let side_by_side =
                    render_sass_file_side_by_side(&sass_path, None, &sass, &project_ir);
                fs::write(&side_by_side_path, side_by_side.as_bytes())?;

                files.push(SassCoverageFileReport {
                    sass_path,
                    ir_path: Some(ir_path),
                    lifted_ir_path: Some(lifted_ir_path),
                    analysis_path: Some(analysis_path),
                    pattern_path: Some(pattern_path),
                    side_by_side_path: Some(side_by_side_path),
                    parsed_instruction_count: parsed.instruction_count(),
                    cfg_block_count: analysis.block_count(),
                    cfg_edge_count: analysis.edge_count(),
                    dominator_block_count: analysis.dominator_block_count(),
                    natural_loop_count: analysis.natural_loop_count(),
                    region_count: analysis.region_count(),
                    reaching_use_count: analysis.reaching_use_count(),
                    ssa_value_count: analysis.ssa_value_count(),
                    def_use_edge_count: analysis.def_use_edge_count(),
                    value_op_count: analysis.value_op_count(),
                    lifted_op_count: lifted.op_count(),
                    live_range_count: analysis.live_range_count(),
                    memory_access_count: analysis.memory_access_count(),
                    semantic_pattern_count: patterns.pattern_count(),
                    unsupported_instruction_count: project_ir.unsupported_instruction_count(),
                    parse_error: None,
                });
            }
            Err(error) => {
                files.push(SassCoverageFileReport {
                    sass_path,
                    ir_path: None,
                    lifted_ir_path: None,
                    analysis_path: None,
                    pattern_path: None,
                    side_by_side_path: None,
                    parsed_instruction_count: 0,
                    cfg_block_count: 0,
                    cfg_edge_count: 0,
                    dominator_block_count: 0,
                    natural_loop_count: 0,
                    region_count: 0,
                    reaching_use_count: 0,
                    ssa_value_count: 0,
                    def_use_edge_count: 0,
                    value_op_count: 0,
                    lifted_op_count: 0,
                    live_range_count: 0,
                    memory_access_count: 0,
                    semantic_pattern_count: 0,
                    unsupported_instruction_count: 0,
                    parse_error: Some(error.to_string()),
                });
            }
        }
    }

    let opcode_probe_targets = opcode_probe_targets(&opcode_catalog);
    let opcode_catalog = opcode_catalog_entries(opcode_catalog);
    let opcode_counts = sorted_opcode_counts(opcode_counts);
    let opcode_signature_counts = sorted_opcode_signature_counts(opcode_signature_counts);
    let semantic_pattern_counts = sorted_semantic_pattern_counts(semantic_pattern_counts);
    let parsed_file_count = files
        .iter()
        .filter(|file| file.parse_error.is_none())
        .count();
    let parse_error_count = files.len().saturating_sub(parsed_file_count);
    let parsed_instruction_count = files.iter().map(|file| file.parsed_instruction_count).sum();
    let cfg_block_count = cfg_blocks.len();
    let cfg_edge_count = cfg_edges.len();
    let dominator_block_count = dominators.len();
    let natural_loop_count = natural_loops.len();
    let region_count = regions.len();
    let dataflow_op_count = dataflow.len();
    let reaching_use_count = reaching_uses.len();
    let ssa_value_count = ssa_values.len();
    let def_use_edge_count = def_use_edges.len();
    let value_op_count = value_ops.len();
    let lifted_op_count = lifted_ops.len();
    let live_range_count = live_ranges.len();
    let memory_access_count = memory_accesses.len();
    let semantic_pattern_count = semantic_patterns.len();
    let known_opcode_count = opcode_catalog.iter().filter(|entry| entry.known).count();
    let locally_mapped_opcode_count = opcode_catalog
        .iter()
        .filter(|entry| entry.locally_mapped)
        .count();
    let known_unobserved_opcode_count = opcode_catalog
        .iter()
        .filter(|entry| entry.known && entry.instruction_count == 0)
        .count();
    let opcode_probe_target_count = opcode_probe_targets.len();
    let known_unmapped_opcode_count = opcode_catalog
        .iter()
        .filter(|entry| entry.known && !entry.locally_mapped)
        .count();
    let observed_unregistered_opcode_count = opcode_catalog
        .iter()
        .filter(|entry| !entry.known && entry.instruction_count > 0)
        .count();
    let observed_unmapped_opcode_count = opcode_catalog
        .iter()
        .filter(|entry| entry.instruction_count > 0 && !entry.locally_mapped)
        .count();
    let unsupported_instruction_count = unsupported_instructions.len();

    let summary_path = options.output_dir.join("summary.txt");
    let files_path = options.output_dir.join("files.tsv");
    let opcode_catalog_path = options.output_dir.join("opcode-catalog.tsv");
    let opcode_probe_targets_path = options.output_dir.join("opcode-probe-targets.tsv");
    let opcode_frequency_path = options.output_dir.join("opcode-frequency.tsv");
    let opcode_signature_frequency_path = options.output_dir.join("opcode-signature-frequency.tsv");
    let semantic_patterns_path = options.output_dir.join("semantic-patterns.tsv");
    let semantic_pattern_frequency_path = options.output_dir.join("semantic-pattern-frequency.tsv");
    let cfg_blocks_path = options.output_dir.join("cfg-blocks.tsv");
    let cfg_edges_path = options.output_dir.join("cfg-edges.tsv");
    let dominators_path = options.output_dir.join("dominators.tsv");
    let natural_loops_path = options.output_dir.join("natural-loops.tsv");
    let regions_path = options.output_dir.join("regions.tsv");
    let dataflow_path = options.output_dir.join("dataflow.tsv");
    let reaching_uses_path = options.output_dir.join("reaching-uses.tsv");
    let ssa_values_path = options.output_dir.join("ssa-values.tsv");
    let def_use_edges_path = options.output_dir.join("def-use-edges.tsv");
    let value_ops_path = options.output_dir.join("value-ops.tsv");
    let lifted_ops_path = options.output_dir.join("lifted-ops.tsv");
    let live_ranges_path = options.output_dir.join("live-ranges.tsv");
    let memory_accesses_path = options.output_dir.join("memory-accesses.tsv");
    let unsupported_instructions_path = options.output_dir.join("unsupported-instructions.tsv");

    let report = SassCoverageReport {
        root: options.root.clone(),
        output_dir: options.output_dir.clone(),
        summary_path,
        files_path,
        opcode_catalog_path,
        opcode_probe_targets_path,
        opcode_frequency_path,
        opcode_signature_frequency_path,
        semantic_patterns_path,
        semantic_pattern_frequency_path,
        cfg_blocks_path,
        cfg_edges_path,
        dominators_path,
        natural_loops_path,
        regions_path,
        dataflow_path,
        reaching_uses_path,
        ssa_values_path,
        def_use_edges_path,
        value_ops_path,
        lifted_ops_path,
        live_ranges_path,
        memory_accesses_path,
        unsupported_instructions_path,
        files,
        opcode_catalog,
        opcode_probe_targets,
        opcode_counts,
        opcode_signature_counts,
        semantic_pattern_counts,
        semantic_patterns,
        cfg_blocks,
        cfg_edges,
        dominators,
        natural_loops,
        regions,
        dataflow,
        reaching_uses,
        ssa_values,
        def_use_edges,
        value_ops,
        lifted_ops,
        live_ranges,
        memory_accesses,
        unsupported_instructions,
        parsed_file_count,
        parse_error_count,
        parsed_instruction_count,
        cfg_block_count,
        cfg_edge_count,
        dominator_block_count,
        natural_loop_count,
        region_count,
        dataflow_op_count,
        reaching_use_count,
        ssa_value_count,
        def_use_edge_count,
        value_op_count,
        lifted_op_count,
        live_range_count,
        memory_access_count,
        semantic_pattern_count,
        known_opcode_count,
        locally_mapped_opcode_count,
        known_unobserved_opcode_count,
        opcode_probe_target_count,
        known_unmapped_opcode_count,
        observed_unregistered_opcode_count,
        observed_unmapped_opcode_count,
        unsupported_instruction_count,
    };

    write_coverage_reports(&report)?;
    Ok(report)
}

fn collect_sass_paths(root: &Path, out: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    if root.is_file() {
        if is_sass_file(root) {
            out.push(root.to_path_buf());
        }
        return Ok(());
    }

    let mut entries = fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_sass_paths(&path, out)?;
        } else if is_sass_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_sass_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".sass"))
}

fn relative_sass_path(root: &Path, sass_path: &Path) -> PathBuf {
    if root.is_file() {
        return sass_path
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("sass.sass"));
    }
    sass_path
        .strip_prefix(root)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| {
            sass_path
                .file_name()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("sass.sass"))
        })
}

fn sass_source_format(path: &Path) -> SassCoverageSourceFormat {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if name.ends_with(".cuobjdump.sass") {
        SassCoverageSourceFormat::Cuobjdump
    } else if name.ends_with(".nvdisasm.sass") {
        SassCoverageSourceFormat::Nvdisasm
    } else {
        SassCoverageSourceFormat::Sass
    }
}

fn seed_known_opcode_catalog(opcode_catalog: &mut BTreeMap<SassOpcode, OpcodeCatalogBuilder>) {
    for known in known_sass_opcodes() {
        append_known_opcode(opcode_catalog, known);
    }
}

fn append_known_opcode(
    opcode_catalog: &mut BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
    known: &KnownSassOpcode,
) {
    let entry = opcode_catalog
        .entry(SassOpcode::from_kind(known.opcode.clone()))
        .or_default();
    entry.known = true;
    entry.locally_mapped |= known.locally_mapped;
    entry.classes.insert(known.class);
    entry.kinds.insert(known.kind);
    entry.known_sources.insert(known.source);
    for architecture in known.architectures {
        entry.architectures.insert((*architecture).to_string());
    }
}

fn append_opcode_catalog_lifted_ops(
    lifted: &SassLiftedModule,
    opcode_catalog: &mut BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
) {
    for function in &lifted.functions {
        for op in &function.ops {
            let entry = opcode_catalog.entry(op.opcode.clone()).or_default();
            if op.class != SassLiftedOpClass::Unsupported {
                entry.locally_mapped = true;
            }
            entry.classes.insert(op.class.into());
            entry.kinds.insert(op.kind.into());
        }
    }
}

fn append_opcode_catalog_unsupported(
    project_ir: &KernelIrModule,
    opcode_catalog: &mut BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
) {
    for function in &project_ir.functions {
        for op in &function.ops {
            let KernelIrOpKind::Unsupported { opcode, .. } = &op.kind else {
                continue;
            };
            opcode_catalog
                .entry(opcode.clone())
                .or_default()
                .unsupported_count += 1;
        }
    }
}

fn opcode_catalog_entries(
    opcode_catalog: BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
) -> Vec<SassOpcodeCatalogEntry> {
    opcode_catalog
        .into_iter()
        .map(|(opcode, entry)| entry.into_entry(opcode))
        .collect()
}

fn opcode_probe_targets(
    opcode_catalog: &BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
) -> Vec<SassOpcodeProbeTarget> {
    let mut targets = opcode_catalog
        .iter()
        .filter(|(_, entry)| entry.known && entry.instruction_count == 0)
        .map(|(opcode, entry)| SassOpcodeProbeTarget {
            opcode: opcode.clone(),
            priority: opcode_probe_priority(entry),
            architectures: entry.architectures.iter().cloned().collect(),
            classes: entry.classes.iter().copied().collect(),
            kinds: entry.kinds.iter().copied().collect(),
            known_sources: entry.known_sources.iter().copied().collect(),
            locally_mapped: entry.locally_mapped,
            recommended_action: if entry.locally_mapped {
                "generate-sass-artifact"
            } else {
                "add-lifter-mapping"
            }
            .to_string(),
            reason: opcode_probe_reason(entry),
        })
        .collect::<Vec<_>>();
    targets.sort_by(|left, right| {
        right
            .priority
            .cmp(&left.priority)
            .then_with(|| left.opcode.cmp(&right.opcode))
    });
    targets
}

fn opcode_probe_priority(entry: &OpcodeCatalogBuilder) -> u8 {
    if !entry.locally_mapped {
        100
    } else if opcode_has_class(entry, SassOpcodeCatalogClass::TensorCore)
        || opcode_has_class(entry, SassOpcodeCatalogClass::TensorMemory)
        || opcode_has_class(entry, SassOpcodeCatalogClass::WarpGroup)
    {
        90
    } else if !entry.architectures.is_empty() {
        70
    } else {
        50
    }
}

fn opcode_probe_reason(entry: &OpcodeCatalogBuilder) -> String {
    if !entry.locally_mapped {
        return "known opcode has no local lifter mapping".to_string();
    }
    if opcode_has_class(entry, SassOpcodeCatalogClass::TensorCore) {
        return "tensor-core opcode is mapped but unobserved in generated SASS artifacts"
            .to_string();
    }
    if opcode_has_class(entry, SassOpcodeCatalogClass::TensorMemory) {
        return "tensor-memory opcode is mapped but unobserved in generated SASS artifacts"
            .to_string();
    }
    if opcode_has_class(entry, SassOpcodeCatalogClass::WarpGroup) {
        return "warpgroup opcode is mapped but unobserved in generated SASS artifacts".to_string();
    }
    if !entry.architectures.is_empty() {
        return "architecture-specific opcode is mapped but unobserved in generated SASS artifacts"
            .to_string();
    }
    "mapped scalar opcode is unobserved in generated SASS artifacts".to_string()
}

fn opcode_has_class(entry: &OpcodeCatalogBuilder, class: SassOpcodeCatalogClass) -> bool {
    entry.classes.contains(&class)
}

fn append_unsupported(
    sass_path: &Path,
    project_ir: &KernelIrModule,
    unsupported_instructions: &mut Vec<SassUnsupportedInstruction>,
) {
    for function in &project_ir.functions {
        for op in &function.ops {
            let KernelIrOpKind::Unsupported { opcode, reason } = &op.kind else {
                continue;
            };
            unsupported_instructions.push(SassUnsupportedInstruction {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                address: op.address,
                opcode: opcode.clone(),
                reason: reason.clone(),
                raw: op.source.clone(),
            });
        }
    }
}

fn append_semantic_patterns(
    sass_path: &Path,
    patterns: &SassPatternModule,
    semantic_patterns: &mut Vec<SassCoverageSemanticPattern>,
) {
    for function in &patterns.functions {
        for pattern in &function.patterns {
            semantic_patterns.push(SassCoverageSemanticPattern {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                start_address: pattern.start_address,
                end_address: pattern.end_address,
                kind: pattern.kind.clone(),
                confidence: pattern.confidence,
            });
        }
    }
}

fn append_analysis(
    sass_path: &Path,
    analysis: &SassAnalysisModule,
    lifted: &SassLiftedModule,
    cfg_blocks: &mut Vec<SassCoverageBasicBlock>,
    cfg_edges: &mut Vec<SassCoverageCfgEdge>,
    dominators: &mut Vec<SassCoverageDominatorBlock>,
    natural_loops: &mut Vec<SassCoverageNaturalLoop>,
    regions: &mut Vec<SassCoverageRegion>,
    dataflow: &mut Vec<SassCoverageDataflowOp>,
    reaching_uses: &mut Vec<SassCoverageReachingUse>,
    ssa_values: &mut Vec<SassCoverageSsaValue>,
    def_use_edges: &mut Vec<SassCoverageDefUseEdge>,
    value_ops: &mut Vec<SassCoverageValueOp>,
    lifted_ops: &mut Vec<SassCoverageLiftedOp>,
    live_ranges: &mut Vec<SassCoverageLiveRange>,
    memory_accesses: &mut Vec<SassCoverageMemoryAccess>,
) {
    for function in &analysis.functions {
        for block in &function.blocks {
            cfg_blocks.push(SassCoverageBasicBlock {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                id: block.id,
                label: block.label.clone(),
                start_address: block.start_address,
                end_address: block.end_address,
                instruction_count: block.instruction_count,
                terminator: block.terminator,
            });
        }
        for edge in &function.edges {
            cfg_edges.push(SassCoverageCfgEdge {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                from_block: edge.from_block,
                to_block: edge.to_block,
                kind: edge.kind,
                condition: edge.condition.clone(),
                target: edge.target.clone(),
            });
        }
        for dominator in &function.dominators {
            dominators.push(SassCoverageDominatorBlock {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                block_id: dominator.block_id,
                reachable: dominator.reachable,
                immediate_dominator: dominator.immediate_dominator,
                dominators: dominator.dominators.clone(),
                dominated_blocks: dominator.dominated_blocks.clone(),
            });
        }
        for natural_loop in &function.natural_loops {
            natural_loops.push(SassCoverageNaturalLoop {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                header_block: natural_loop.header_block,
                latch_block: natural_loop.latch_block,
                reachable: natural_loop.reachable,
                blocks: natural_loop.blocks.clone(),
                edge_condition: natural_loop.edge_condition.clone(),
                edge_target: natural_loop.edge_target.clone(),
            });
        }
        for region in &function.regions {
            regions.push(SassCoverageRegion {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                id: region.id,
                parent: region.parent,
                children: region.children.clone(),
                depth: region.depth,
                path: region.path.clone(),
                local_rank: region.local_rank,
                kind: region.kind,
                header_block: region.header_block,
                latch_block: region.latch_block,
                branch_block: region.branch_block,
                entry_blocks: region.entry_blocks.clone(),
                blocks: region.blocks.clone(),
                op_addresses: region.op_addresses.clone(),
                opcode_closure: region.opcode_closure.clone(),
                condition: region.condition.clone(),
                target: region.target.clone(),
            });
        }
        for op in &function.dataflow {
            dataflow.push(SassCoverageDataflowOp {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                address: op.address,
                defines: op.defines.clone(),
                uses: op.uses.clone(),
                source: op.source.clone(),
            });
        }
        for use_site in &function.reaching_uses {
            reaching_uses.push(SassCoverageReachingUse {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                address: use_site.address,
                register: use_site.register.clone(),
                reaching_def_addresses: use_site.reaching_def_addresses.clone(),
                reaches_entry: use_site.reaches_entry,
            });
        }
        for value in &function.ssa_values {
            ssa_values.push(SassCoverageSsaValue {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                value_id: value.value_id,
                register: value.register.clone(),
                def_address: value.def_address,
                source: value.source.clone(),
                use_addresses: value.use_addresses.clone(),
            });
        }
        for edge in &function.def_use_edges {
            def_use_edges.push(SassCoverageDefUseEdge {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                value_id: edge.value_id,
                register: edge.register.clone(),
                def_address: edge.def_address,
                use_address: edge.use_address,
                use_source: edge.use_source.clone(),
            });
        }
        for op in &function.value_ops {
            value_ops.push(SassCoverageValueOp {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                address: op.address,
                block_id: op.block_id,
                predicate: op.predicate.clone(),
                opcode: op.opcode.clone(),
                kind: op.kind,
                input_registers: op.input_registers.clone(),
                output_registers: op.output_registers.clone(),
                input_value_ids: op.input_value_ids.clone(),
                output_value_ids: op.output_value_ids.clone(),
                source: op.source.clone(),
            });
        }
        if let Some(lifted_function) = lifted
            .functions
            .iter()
            .find(|lifted| lifted.name == function.name)
        {
            for op in &lifted_function.ops {
                lifted_ops.push(SassCoverageLiftedOp {
                    sass_path: sass_path.to_path_buf(),
                    function: function.name.clone(),
                    address: op.address,
                    block_id: op.block_id,
                    predicate: op.predicate.clone(),
                    opcode: op.opcode.clone(),
                    class: op.class,
                    kind: op.kind,
                    semantics: op.semantics.clone(),
                    inputs: op.inputs.clone(),
                    outputs: op.outputs.clone(),
                    source_operands: op.source_operands.clone(),
                    detail: op.detail,
                    source: op.source.clone(),
                });
            }
        }
        for range in &function.live_ranges {
            live_ranges.push(SassCoverageLiveRange {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                register: range.register.clone(),
                def_address: range.def_address,
                start_address: range.start_address,
                end_address: range.end_address,
                use_addresses: range.use_addresses.clone(),
            });
        }
        for access in &function.memory_accesses {
            memory_accesses.push(SassCoverageMemoryAccess {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                address: access.address,
                predicate: access.predicate.clone(),
                kind: access.kind,
                space: access.space,
                width_bits: access.width_bits,
                value_register: access.value_register.clone(),
                address_expr: access.memory_address.to_string(),
                address_registers: access.address_registers.clone(),
                address_base: access.address_base.clone(),
                offset: access.offset.clone(),
                source: access.source.clone(),
            });
        }
    }
}

fn sorted_opcode_counts(counts: BTreeMap<SassOpcode, usize>) -> Vec<SassOpcodeCount> {
    let mut counts = counts
        .into_iter()
        .map(|(opcode, count)| SassOpcodeCount { opcode, count })
        .collect::<Vec<_>>();
    counts.sort_by(|lhs, rhs| {
        rhs.count
            .cmp(&lhs.count)
            .then_with(|| lhs.opcode.cmp(&rhs.opcode))
    });
    counts
}

fn sorted_opcode_signature_counts(
    counts: BTreeMap<SassOpcodeSignature, usize>,
) -> Vec<SassOpcodeSignatureCount> {
    let mut counts = counts
        .into_iter()
        .map(|(signature, count)| SassOpcodeSignatureCount { signature, count })
        .collect::<Vec<_>>();
    counts.sort_by(|lhs, rhs| {
        rhs.count
            .cmp(&lhs.count)
            .then_with(|| lhs.signature.cmp(&rhs.signature))
    });
    counts
}

fn sorted_semantic_pattern_counts(
    counts: BTreeMap<SassSemanticPatternCategory, usize>,
) -> Vec<SassSemanticPatternCount> {
    let mut counts = counts
        .into_iter()
        .map(|(category, count)| SassSemanticPatternCount { category, count })
        .collect::<Vec<_>>();
    counts.sort_by(|lhs, rhs| {
        rhs.count
            .cmp(&lhs.count)
            .then_with(|| lhs.category.cmp(&rhs.category))
    });
    counts
}

fn write_coverage_reports(report: &SassCoverageReport) -> Result<(), Box<dyn Error>> {
    fs::write(
        &report.summary_path,
        render_coverage_summary(report).as_bytes(),
    )?;
    fs::write(&report.files_path, render_files_tsv(report).as_bytes())?;
    fs::write(
        &report.opcode_catalog_path,
        render_opcode_catalog_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.opcode_probe_targets_path,
        render_opcode_probe_targets_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.opcode_frequency_path,
        render_opcode_counts_tsv(&report.opcode_counts).as_bytes(),
    )?;
    fs::write(
        &report.opcode_signature_frequency_path,
        render_opcode_signature_counts_tsv(&report.opcode_signature_counts).as_bytes(),
    )?;
    fs::write(
        &report.semantic_patterns_path,
        render_semantic_patterns_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.semantic_pattern_frequency_path,
        render_semantic_pattern_counts_tsv(&report.semantic_pattern_counts).as_bytes(),
    )?;
    fs::write(
        &report.cfg_blocks_path,
        render_cfg_blocks_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.cfg_edges_path,
        render_cfg_edges_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.dominators_path,
        render_dominators_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.natural_loops_path,
        render_natural_loops_tsv(report).as_bytes(),
    )?;
    fs::write(&report.regions_path, render_regions_tsv(report).as_bytes())?;
    fs::write(
        &report.dataflow_path,
        render_dataflow_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.reaching_uses_path,
        render_reaching_uses_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.ssa_values_path,
        render_ssa_values_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.def_use_edges_path,
        render_def_use_edges_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.value_ops_path,
        render_value_ops_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.lifted_ops_path,
        render_lifted_ops_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.live_ranges_path,
        render_live_ranges_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.memory_accesses_path,
        render_memory_accesses_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.unsupported_instructions_path,
        render_unsupported_tsv(report).as_bytes(),
    )?;
    Ok(())
}

fn render_coverage_summary(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(out, "root={}", report.root.display()).expect("write to string");
    writeln!(out, "output_dir={}", report.output_dir.display()).expect("write to string");
    writeln!(out, "files_seen={}", report.files.len()).expect("write to string");
    writeln!(out, "files_parsed={}", report.parsed_file_count).expect("write to string");
    writeln!(out, "parse_errors={}", report.parse_error_count).expect("write to string");
    writeln!(
        out,
        "parsed_instructions={}",
        report.parsed_instruction_count
    )
    .expect("write to string");
    writeln!(out, "cfg_blocks={}", report.cfg_block_count).expect("write to string");
    writeln!(out, "cfg_edges={}", report.cfg_edge_count).expect("write to string");
    writeln!(out, "dominator_blocks={}", report.dominator_block_count).expect("write to string");
    writeln!(out, "natural_loops={}", report.natural_loop_count).expect("write to string");
    writeln!(out, "regions={}", report.region_count).expect("write to string");
    writeln!(out, "dataflow_ops={}", report.dataflow_op_count).expect("write to string");
    writeln!(out, "reaching_uses={}", report.reaching_use_count).expect("write to string");
    writeln!(out, "ssa_values={}", report.ssa_value_count).expect("write to string");
    writeln!(out, "def_use_edges={}", report.def_use_edge_count).expect("write to string");
    writeln!(out, "value_ops={}", report.value_op_count).expect("write to string");
    writeln!(out, "lifted_ops={}", report.lifted_op_count).expect("write to string");
    writeln!(out, "live_ranges={}", report.live_range_count).expect("write to string");
    writeln!(out, "memory_accesses={}", report.memory_access_count).expect("write to string");
    writeln!(out, "semantic_patterns={}", report.semantic_pattern_count).expect("write to string");
    writeln!(out, "known_opcodes={}", report.known_opcode_count).expect("write to string");
    writeln!(
        out,
        "locally_mapped_opcodes={}",
        report.locally_mapped_opcode_count
    )
    .expect("write to string");
    writeln!(
        out,
        "known_unobserved_opcodes={}",
        report.known_unobserved_opcode_count
    )
    .expect("write to string");
    writeln!(
        out,
        "opcode_probe_targets={}",
        report.opcode_probe_target_count
    )
    .expect("write to string");
    writeln!(
        out,
        "known_unmapped_opcodes={}",
        report.known_unmapped_opcode_count
    )
    .expect("write to string");
    writeln!(
        out,
        "observed_unregistered_opcodes={}",
        report.observed_unregistered_opcode_count
    )
    .expect("write to string");
    writeln!(
        out,
        "observed_unmapped_opcodes={}",
        report.observed_unmapped_opcode_count
    )
    .expect("write to string");
    writeln!(
        out,
        "unsupported_instructions={}",
        report.unsupported_instruction_count
    )
    .expect("write to string");
    writeln!(out, "unique_opcodes={}", report.opcode_counts.len()).expect("write to string");
    writeln!(
        out,
        "opcode_catalog={}",
        report.opcode_catalog_path.display()
    )
    .expect("write to string");
    writeln!(
        out,
        "unique_opcode_signatures={}",
        report.opcode_signature_counts.len()
    )
    .expect("write to string");
    writeln!(out, "regions_path={}", report.regions_path.display()).expect("write to string");
    writeln!(out).expect("write to string");
    writeln!(out, "top_opcodes").expect("write to string");
    for count in report.opcode_counts.iter().take(32) {
        writeln!(out, "{}\t{}", count.opcode, count.count).expect("write to string");
    }
    if !report.semantic_pattern_counts.is_empty() {
        writeln!(out).expect("write to string");
        writeln!(out, "top_semantic_patterns").expect("write to string");
        for count in report.semantic_pattern_counts.iter().take(32) {
            writeln!(out, "{}\t{}", count.category, count.count).expect("write to string");
        }
    }
    if !report.opcode_probe_targets.is_empty() {
        writeln!(out).expect("write to string");
        writeln!(out, "top_opcode_probe_targets").expect("write to string");
        writeln!(out, "opcode\tpriority\trecommended_action\treason").expect("write to string");
        for target in report.opcode_probe_targets.iter().take(16) {
            writeln!(
                out,
                "{}\t{}\t{}\t{}",
                target.opcode, target.priority, target.recommended_action, target.reason
            )
            .expect("write to string");
        }
    }
    if !report.unsupported_instructions.is_empty() {
        writeln!(out).expect("write to string");
        writeln!(out, "unsupported_by_opcode").expect("write to string");
        let mut by_opcode = BTreeMap::<SassOpcode, usize>::new();
        for instruction in &report.unsupported_instructions {
            *by_opcode.entry(instruction.opcode.clone()).or_default() += 1;
        }
        for count in sorted_opcode_counts(by_opcode) {
            writeln!(out, "{}\t{}", count.opcode, count.count).expect("write to string");
        }
    }
    out
}

fn render_opcode_catalog_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "opcode\tknown\tobserved\tlocally_mapped\tinstruction_count\tsignature_count\tsignatures\tsource_formats\tarchitectures\tknown_sources\tclasses\tkinds\tsupport\tcoverage\tunsupported_count"
    )
    .expect("write to string");
    for entry in &report.opcode_catalog {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&entry.opcode.to_string()),
            entry.known,
            entry.observed,
            entry.locally_mapped,
            entry.instruction_count,
            entry.signature_count,
            tsv(&display_list(&entry.signatures)),
            tsv(&display_list(&entry.source_formats)),
            tsv(&entry.architectures.join(",")),
            tsv(&display_list(&entry.known_sources)),
            tsv(&display_list(&entry.classes)),
            tsv(&display_list(&entry.kinds)),
            tsv(&entry.support.to_string()),
            tsv(&entry.coverage.to_string()),
            entry.unsupported_count,
        )
        .expect("write to string");
    }
    out
}

fn render_opcode_probe_targets_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "opcode\tpriority\tlocally_mapped\tarchitectures\tclasses\tkinds\tknown_sources\trecommended_action\treason"
    )
    .expect("write to string");
    for target in &report.opcode_probe_targets {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&target.opcode.to_string()),
            target.priority,
            target.locally_mapped,
            tsv(&target.architectures.join(",")),
            tsv(&display_list(&target.classes)),
            tsv(&display_list(&target.kinds)),
            tsv(&display_list(&target.known_sources)),
            tsv(&target.recommended_action),
            tsv(&target.reason),
        )
        .expect("write to string");
    }
    out
}

fn render_files_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "status\tsass_path\tparsed_instructions\tcfg_blocks\tcfg_edges\tdominator_blocks\tnatural_loops\tregions\treaching_uses\tssa_values\tdef_use_edges\tvalue_ops\tlifted_ops\tlive_ranges\tmemory_accesses\tsemantic_patterns\tunsupported_instructions\tir_path\tlifted_ir_path\tanalysis_path\tpatterns_path\tside_by_side_path\terror"
    )
    .expect("write to string");
    for file in &report.files {
        let status = if file.parse_error.is_some() {
            "parse-error"
        } else {
            "parsed"
        };
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            status,
            tsv(&file.sass_path.display().to_string()),
            file.parsed_instruction_count,
            file.cfg_block_count,
            file.cfg_edge_count,
            file.dominator_block_count,
            file.natural_loop_count,
            file.region_count,
            file.reaching_use_count,
            file.ssa_value_count,
            file.def_use_edge_count,
            file.value_op_count,
            file.lifted_op_count,
            file.live_range_count,
            file.memory_access_count,
            file.semantic_pattern_count,
            file.unsupported_instruction_count,
            tsv(&optional_path(&file.ir_path)),
            tsv(&optional_path(&file.lifted_ir_path)),
            tsv(&optional_path(&file.analysis_path)),
            tsv(&optional_path(&file.pattern_path)),
            tsv(&optional_path(&file.side_by_side_path)),
            tsv(file.parse_error.as_deref().unwrap_or(""))
        )
        .expect("write to string");
    }
    out
}

fn render_semantic_patterns_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tstart_address\tend_address\tkind\tconfidence\tdetail"
    )
    .expect("write to string");
    for pattern in &report.semantic_patterns {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{:#06x}\t{}\t{}\t{}",
            tsv(&pattern.sass_path.display().to_string()),
            tsv(&pattern.function),
            pattern.start_address,
            pattern.end_address,
            tsv(pattern.kind.name()),
            tsv(&pattern.confidence.to_string()),
            tsv(&format!("{:?}", pattern.kind)),
        )
        .expect("write to string");
    }
    out
}

fn render_cfg_blocks_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tblock_id\tlabel\tstart_address\tend_address\tinstruction_count\tterminator"
    )
    .expect("write to string");
    for block in &report.cfg_blocks {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{:#06x}\t{:#06x}\t{}\t{}",
            tsv(&block.sass_path.display().to_string()),
            tsv(&block.function),
            block.id,
            tsv(block.label.as_deref().unwrap_or("")),
            block.start_address,
            block.end_address,
            block.instruction_count,
            tsv(&block.terminator.to_string()),
        )
        .expect("write to string");
    }
    out
}

fn render_cfg_edges_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tfrom_block\tto_block\tkind\tcondition\ttarget"
    )
    .expect("write to string");
    for edge in &report.cfg_edges {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&edge.sass_path.display().to_string()),
            tsv(&edge.function),
            edge.from_block,
            edge.to_block
                .map(|block| block.to_string())
                .unwrap_or_default(),
            tsv(&edge.kind.to_string()),
            tsv(&display_optional(edge.condition.as_ref())),
            tsv(&display_optional(edge.target.as_ref())),
        )
        .expect("write to string");
    }
    out
}

fn render_dominators_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tblock_id\treachable\timmediate_dominator\tdominators\tdominated_blocks"
    )
    .expect("write to string");
    for dominator in &report.dominators {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&dominator.sass_path.display().to_string()),
            tsv(&dominator.function),
            dominator.block_id,
            dominator.reachable,
            dominator
                .immediate_dominator
                .map(|block| block.to_string())
                .unwrap_or_default(),
            tsv(&format_blocks(&dominator.dominators)),
            tsv(&format_blocks(&dominator.dominated_blocks)),
        )
        .expect("write to string");
    }
    out
}

fn render_natural_loops_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\theader_block\tlatch_block\treachable\tblocks\tedge_condition\tedge_target"
    )
    .expect("write to string");
    for natural_loop in &report.natural_loops {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&natural_loop.sass_path.display().to_string()),
            tsv(&natural_loop.function),
            natural_loop.header_block,
            natural_loop.latch_block,
            natural_loop.reachable,
            tsv(&format_blocks(&natural_loop.blocks)),
            tsv(&display_optional(natural_loop.edge_condition.as_ref())),
            tsv(&display_optional(natural_loop.edge_target.as_ref())),
        )
        .expect("write to string");
    }
    out
}

fn render_regions_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tregion_id\tparent_region\tchildren\tdepth\tpath\tlocal_rank\tkind\theader_block\tlatch_block\tbranch_block\tentry_blocks\tblocks\top_addresses\topcode_closure\tcondition\ttarget"
    )
    .expect("write to string");
    for region in &report.regions {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&region.sass_path.display().to_string()),
            tsv(&region.function),
            region.id,
            region.parent.map(|id| id.to_string()).unwrap_or_default(),
            tsv(&format_blocks(&region.children)),
            region.depth,
            tsv(&region.path.to_string()),
            region.local_rank,
            tsv(&region.kind.to_string()),
            region
                .header_block
                .map(|block| block.to_string())
                .unwrap_or_default(),
            region
                .latch_block
                .map(|block| block.to_string())
                .unwrap_or_default(),
            region
                .branch_block
                .map(|block| block.to_string())
                .unwrap_or_default(),
            tsv(&format_blocks(&region.entry_blocks)),
            tsv(&format_blocks(&region.blocks)),
            tsv(&format_addresses(&region.op_addresses)),
            tsv(&display_list(&region.opcode_closure)),
            tsv(&display_optional(region.condition.as_ref())),
            tsv(&display_optional(region.target.as_ref())),
        )
        .expect("write to string");
    }
    out
}

fn render_dataflow_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(out, "sass_path\tfunction\taddress\tdefines\tuses\traw").expect("write to string");
    for op in &report.dataflow {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}",
            tsv(&op.sass_path.display().to_string()),
            tsv(&op.function),
            op.address,
            tsv(&display_list(&op.defines)),
            tsv(&display_list(&op.uses)),
            tsv(&op.source),
        )
        .expect("write to string");
    }
    out
}

fn render_reaching_uses_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\taddress\tregister\treaching_defs\treaches_entry"
    )
    .expect("write to string");
    for use_site in &report.reaching_uses {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}",
            tsv(&use_site.sass_path.display().to_string()),
            tsv(&use_site.function),
            use_site.address,
            tsv(&use_site.register.to_string()),
            tsv(&format_reaching_defs(
                &use_site.reaching_def_addresses,
                use_site.reaches_entry,
            )),
            use_site.reaches_entry,
        )
        .expect("write to string");
    }
    out
}

fn render_ssa_values_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tvalue_id\tregister\tdef\tuses\tsource"
    )
    .expect("write to string");
    for value in &report.ssa_values {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&value.sass_path.display().to_string()),
            tsv(&value.function),
            value.value_id,
            tsv(&value.register.to_string()),
            tsv(&format_optional_address(value.def_address)),
            tsv(&format_addresses(&value.use_addresses)),
            tsv(value.source.as_deref().unwrap_or("entry")),
        )
        .expect("write to string");
    }
    out
}

fn render_def_use_edges_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tuse_address\tregister\tvalue_id\tdef\tuse_source"
    )
    .expect("write to string");
    for edge in &report.def_use_edges {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}\t{}",
            tsv(&edge.sass_path.display().to_string()),
            tsv(&edge.function),
            edge.use_address,
            tsv(&edge.register.to_string()),
            edge.value_id,
            tsv(&format_optional_address(edge.def_address)),
            tsv(&edge.use_source),
        )
        .expect("write to string");
    }
    out
}

fn render_value_ops_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\taddress\tblock_id\tpredicate\topcode\tinput_registers\toutput_registers\tinput_values\toutput_values\tkind\traw"
    )
    .expect("write to string");
    for op in &report.value_ops {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&op.sass_path.display().to_string()),
            tsv(&op.function),
            op.address,
            op.block_id
                .map(|block| block.to_string())
                .unwrap_or_default(),
            tsv(op.predicate.as_deref().unwrap_or("")),
            tsv(&op.opcode.to_string()),
            tsv(&display_list(&op.input_registers)),
            tsv(&display_list(&op.output_registers)),
            tsv(&format_values(&op.input_value_ids)),
            tsv(&format_values(&op.output_value_ids)),
            tsv(&op.kind.to_string()),
            tsv(&op.source),
        )
        .expect("write to string");
    }
    out
}

fn render_lifted_ops_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\taddress\tblock_id\tpredicate\topcode\tclass\tkind\tsemantics\tinputs\toutputs\tsource_operands\tdetail\traw"
    )
    .expect("write to string");
    for op in &report.lifted_ops {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&op.sass_path.display().to_string()),
            tsv(&op.function),
            op.address,
            op.block_id
                .map(|block| block.to_string())
                .unwrap_or_default(),
            tsv(op.predicate.as_deref().unwrap_or("")),
            tsv(&op.opcode.to_string()),
            tsv(&op.class.to_string()),
            tsv(&op.kind.to_string()),
            tsv(&op.semantics.to_string()),
            tsv(&display_lifted_value_refs(&op.inputs)),
            tsv(&display_lifted_value_refs(&op.outputs)),
            tsv(&op.source_operands.join(",")),
            tsv(&op.detail.to_string()),
            tsv(&op.source),
        )
        .expect("write to string");
    }
    out
}

fn render_live_ranges_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tregister\tdef\tstart_address\tend_address\tuses"
    )
    .expect("write to string");
    for range in &report.live_ranges {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{:#06x}\t{:#06x}\t{}",
            tsv(&range.sass_path.display().to_string()),
            tsv(&range.function),
            tsv(&range.register.to_string()),
            tsv(&format_optional_address(range.def_address)),
            range.start_address,
            range.end_address,
            tsv(&format_addresses(&range.use_addresses)),
        )
        .expect("write to string");
    }
    out
}

fn render_memory_accesses_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\taddress\tpredicate\tkind\tspace\twidth_bits\tvalue_register\taddress_expr\taddress_registers\taddress_base\toffset\traw"
    )
    .expect("write to string");
    for access in &report.memory_accesses {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&access.sass_path.display().to_string()),
            tsv(&access.function),
            access.address,
            tsv(access.predicate.as_deref().unwrap_or("")),
            tsv(&access.kind.to_string()),
            tsv(&access.space.to_string()),
            access
                .width_bits
                .map(|bits| bits.to_string())
                .unwrap_or_default(),
            tsv(&access.value_register.to_string()),
            tsv(&access.address_expr),
            tsv(&display_list(&access.address_registers)),
            tsv(&display_optional(access.address_base.as_ref())),
            tsv(&display_optional(access.offset.as_ref())),
            tsv(&access.source),
        )
        .expect("write to string");
    }
    out
}

fn render_opcode_counts_tsv(counts: &[SassOpcodeCount]) -> String {
    let mut out = String::new();
    writeln!(out, "opcode\tcount").expect("write to string");
    for count in counts {
        writeln!(out, "{}\t{}", tsv(&count.opcode.to_string()), count.count)
            .expect("write to string");
    }
    out
}

fn render_opcode_signature_counts_tsv(counts: &[SassOpcodeSignatureCount]) -> String {
    let mut out = String::new();
    writeln!(out, "opcode_signature\tcount").expect("write to string");
    for count in counts {
        writeln!(
            out,
            "{}\t{}",
            tsv(&count.signature.to_string()),
            count.count
        )
        .expect("write to string");
    }
    out
}

fn render_semantic_pattern_counts_tsv(counts: &[SassSemanticPatternCount]) -> String {
    let mut out = String::new();
    writeln!(out, "semantic_pattern\tcount").expect("write to string");
    for count in counts {
        writeln!(out, "{}\t{}", tsv(&count.category.to_string()), count.count)
            .expect("write to string");
    }
    out
}

fn display_list<T: fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn display_optional<T: fmt::Display>(value: Option<&T>) -> String {
    value.map(ToString::to_string).unwrap_or_default()
}

fn display_lifted_value_refs(values: &[SassLiftedValueRef]) -> String {
    values
        .iter()
        .map(SassLiftedValueRef::name)
        .collect::<Vec<_>>()
        .join(",")
}

fn render_unsupported_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(out, "sass_path\tfunction\taddress\topcode\treason\traw").expect("write to string");
    for instruction in &report.unsupported_instructions {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}",
            tsv(&instruction.sass_path.display().to_string()),
            tsv(&instruction.function),
            instruction.address,
            tsv(&instruction.opcode.to_string()),
            tsv(&instruction.reason),
            tsv(&instruction.raw)
        )
        .expect("write to string");
    }
    out
}

fn optional_path(path: &Option<PathBuf>) -> String {
    path.as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_default()
}

fn format_reaching_defs(addresses: &[u64], reaches_entry: bool) -> String {
    let mut parts = addresses
        .iter()
        .map(|address| format!("{address:#06x}"))
        .collect::<Vec<_>>();
    if reaches_entry {
        parts.insert(0, "entry".to_string());
    }
    parts.join(",")
}

fn format_optional_address(address: Option<u64>) -> String {
    address
        .map(|address| format!("{address:#06x}"))
        .unwrap_or_else(|| "entry".to_string())
}

fn format_addresses(addresses: &[u64]) -> String {
    addresses
        .iter()
        .map(|address| format!("{address:#06x}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_blocks(blocks: &[usize]) -> String {
    blocks
        .iter()
        .map(|block| block.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn format_values(values: &[usize]) -> String {
    values
        .iter()
        .map(|value| format!("v{value}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn tsv(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
}
