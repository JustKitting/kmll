use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use super::super::{
    KernelIrModule, KernelIrOpKind, SassAnalysisModule, SassArchitecture, SassLiftedModule,
    SassOpcode, SassPatternModule, SassSemanticPatternCategory, SassTarget, analyze_sass_ir,
    lift_sass_module, lift_sass_value_ir, parse_nvidia_sass, recover_sass_patterns,
    render_sass_file_side_by_side,
};

use super::catalog::{
    append_opcode_catalog_lifted_ops, append_opcode_catalog_unsupported, opcode_catalog_entries,
    opcode_probe_targets, seed_known_opcode_catalog, sm120_tensor_core_support,
    sorted_opcode_counts, sorted_opcode_signature_counts, sorted_semantic_pattern_counts,
};
use super::render::write_coverage_reports;
use super::types::*;

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
    let mut scanned_architectures = BTreeSet::<SassArchitecture>::new();
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
                let target = parsed.target.clone().map(SassTarget::parse);
                let target_architecture = target.as_ref().and_then(SassTarget::architecture);
                if let Some(architecture) = target_architecture {
                    scanned_architectures.insert(architecture);
                }
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
                        if let Some(architecture) = target_architecture {
                            catalog_entry.observed_architectures.insert(architecture);
                        }
                    }
                }

                let project_ir = lift_sass_module(&parsed);
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
                    target,
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
                    target: None,
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
                    parse_error: Some(error),
                });
            }
        }
    }

    let scanned_architectures = scanned_architectures.into_iter().collect::<Vec<_>>();
    let opcode_probe_targets = opcode_probe_targets(&opcode_catalog, &scanned_architectures);
    let opcode_catalog = opcode_catalog_entries(opcode_catalog);
    let sm120_tensor_core_support = sm120_tensor_core_support(&opcode_catalog);
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
    let sm120_tensor_core_required_count = sm120_tensor_core_support.len();
    let sm120_tensor_core_supported_count = sm120_tensor_core_support
        .iter()
        .filter(|entry| entry.status == Sm120TensorCoreSupportStatus::Supported)
        .count();
    let sm120_tensor_core_missing_count = sm120_tensor_core_support
        .len()
        .saturating_sub(sm120_tensor_core_supported_count);
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
    let sm120_tensor_core_support_path = options.output_dir.join("sm120-tensor-core-support.tsv");
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
        sm120_tensor_core_support_path,
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
        scanned_architectures,
        opcode_catalog,
        opcode_probe_targets,
        sm120_tensor_core_support,
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
        sm120_tensor_core_required_count,
        sm120_tensor_core_supported_count,
        sm120_tensor_core_missing_count,
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
                source_text: op.source_text.clone(),
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
                source_text: op.source_text.clone(),
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
                origin: value.origin,
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
                use_site: edge.use_site,
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
                source_text: op.source_text.clone(),
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
                    source_text: op.source_text.clone(),
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
                memory_address: access.memory_address.clone(),
                address_registers: access.address_registers.clone(),
                address_base: access.address_base.clone(),
                offset: access.offset.clone(),
                source_text: access.source_text.clone(),
            });
        }
    }
}
