use std::{
    collections::BTreeMap,
    error::Error,
    fmt::{self, Write as _},
    fs,
    path::PathBuf,
};

use super::super::{SassLiftedValueRef, SassOpcode};
use super::catalog::sorted_opcode_counts;
use super::types::{
    SassCoverageReport, SassOpcodeCount, SassOpcodeSignatureCount, SassSemanticPatternCount,
};

pub(super) fn write_coverage_reports(report: &SassCoverageReport) -> Result<(), Box<dyn Error>> {
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
        &report.sm120_tensor_core_support_path,
        render_sm120_tensor_core_support_tsv(report).as_bytes(),
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
    writeln!(
        out,
        "scanned_architectures={}",
        display_list(&report.scanned_architectures)
    )
    .expect("write to string");
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
        "sm120_tensor_core_required={}",
        report.sm120_tensor_core_required_count
    )
    .expect("write to string");
    writeln!(
        out,
        "sm120_tensor_core_supported={}",
        report.sm120_tensor_core_supported_count
    )
    .expect("write to string");
    writeln!(
        out,
        "sm120_tensor_core_missing={}",
        report.sm120_tensor_core_missing_count
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
        "sm120_tensor_core_support={}",
        report.sm120_tensor_core_support_path.display()
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
    if !report.sm120_tensor_core_support.is_empty() {
        writeln!(out).expect("write to string");
        writeln!(out, "sm120_tensor_core_support").expect("write to string");
        writeln!(out, "opcode\trequired_architecture\tstatus\trequirement")
            .expect("write to string");
        for entry in &report.sm120_tensor_core_support {
            writeln!(
                out,
                "{}\t{}\t{}\t{}",
                entry.opcode, entry.required_architecture, entry.status, entry.requirement
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

fn render_sm120_tensor_core_support_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "opcode\trequired_architecture\tfamily\trequirement\tobserved\tlocally_mapped\tinstruction_count\tobserved_architectures\tstatus"
    )
    .expect("write to string");
    for entry in &report.sm120_tensor_core_support {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&entry.opcode.to_string()),
            entry.required_architecture,
            entry.family,
            tsv(entry.requirement),
            entry.observed,
            entry.locally_mapped,
            entry.instruction_count,
            tsv(&display_list(&entry.observed_architectures)),
            entry.status,
        )
        .expect("write to string");
    }
    out
}

fn render_opcode_catalog_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "opcode\tknown\tobserved\tlocally_mapped\tinstruction_count\tsignature_count\tsignatures\tsource_formats\tarchitectures\tobserved_architectures\tknown_sources\tclasses\tkinds\tsupport\tcoverage\tunsupported_count"
    )
    .expect("write to string");
    for entry in &report.opcode_catalog {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&entry.opcode.to_string()),
            entry.known,
            entry.observed,
            entry.locally_mapped,
            entry.instruction_count,
            entry.signature_count,
            tsv(&display_list(&entry.signatures)),
            tsv(&display_list(&entry.source_formats)),
            tsv(&display_list(&entry.architectures)),
            tsv(&display_list(&entry.observed_architectures)),
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
        "opcode\tpriority\tlocally_mapped\tarchitectures\tmatching_scanned_architectures\tclasses\tkinds\tknown_sources\trecommended_action\treason"
    )
    .expect("write to string");
    for target in &report.opcode_probe_targets {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&target.opcode.to_string()),
            target.priority,
            target.locally_mapped,
            tsv(&display_list(&target.architectures)),
            tsv(&display_list(&target.matching_scanned_architectures)),
            tsv(&display_list(&target.classes)),
            tsv(&display_list(&target.kinds)),
            tsv(&display_list(&target.known_sources)),
            tsv(&target.recommended_action.to_string()),
            tsv(&target.reason.to_string()),
        )
        .expect("write to string");
    }
    out
}

fn render_files_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "status\tsass_path\ttarget\tparsed_instructions\tcfg_blocks\tcfg_edges\tdominator_blocks\tnatural_loops\tregions\treaching_uses\tssa_values\tdef_use_edges\tvalue_ops\tlifted_ops\tlive_ranges\tmemory_accesses\tsemantic_patterns\tunsupported_instructions\tir_path\tlifted_ir_path\tanalysis_path\tpatterns_path\tside_by_side_path\terror"
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
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            status,
            tsv(&file.sass_path.display().to_string()),
            tsv(&file.target.as_ref().map(ToString::to_string).unwrap_or_default()),
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
            tsv(&file.parse_error.as_ref().map(ToString::to_string).unwrap_or_default())
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
            tsv_display(&pattern.function),
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
            tsv_display(&block.function),
            block.id,
            tsv(&display_optional(block.label.as_ref())),
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
            tsv_display(&edge.function),
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
            tsv_display(&dominator.function),
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
            tsv_display(&natural_loop.function),
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
            tsv_display(&region.function),
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
    writeln!(
        out,
        "sass_path\tfunction\taddress\tdefines\tuses\tsource_text"
    )
    .expect("write to string");
    for op in &report.dataflow {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}",
            tsv(&op.sass_path.display().to_string()),
            tsv_display(&op.function),
            op.address,
            tsv(&display_list(&op.defines)),
            tsv(&display_list(&op.uses)),
            tsv(&op.source_text),
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
            tsv_display(&use_site.function),
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
        "sass_path\tfunction\tvalue_id\tregister\tdef\tuses\torigin"
    )
    .expect("write to string");
    for value in &report.ssa_values {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&value.sass_path.display().to_string()),
            tsv_display(&value.function),
            value.value_id,
            tsv(&value.register.to_string()),
            tsv(&format_optional_address(value.def_address)),
            tsv(&format_addresses(&value.use_addresses)),
            tsv(&value.origin.to_string()),
        )
        .expect("write to string");
    }
    out
}

fn render_def_use_edges_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tuse_address\tregister\tvalue_id\tdef\tuse_site"
    )
    .expect("write to string");
    for edge in &report.def_use_edges {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}\t{}",
            tsv(&edge.sass_path.display().to_string()),
            tsv_display(&edge.function),
            edge.use_address,
            tsv(&edge.register.to_string()),
            edge.value_id,
            tsv(&format_optional_address(edge.def_address)),
            tsv(&edge.use_site.to_string()),
        )
        .expect("write to string");
    }
    out
}

fn render_value_ops_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\taddress\tblock_id\tpredicate\topcode\tinput_registers\toutput_registers\tinput_values\toutput_values\tkind\tsource_text"
    )
    .expect("write to string");
    for op in &report.value_ops {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&op.sass_path.display().to_string()),
            tsv_display(&op.function),
            op.address,
            op.block_id
                .map(|block| block.to_string())
                .unwrap_or_default(),
            tsv(&display_optional(op.predicate.as_ref())),
            tsv(&op.opcode.to_string()),
            tsv(&display_list(&op.input_registers)),
            tsv(&display_list(&op.output_registers)),
            tsv(&format_values(&op.input_value_ids)),
            tsv(&format_values(&op.output_value_ids)),
            tsv(&op.kind.to_string()),
            tsv(&op.source_text),
        )
        .expect("write to string");
    }
    out
}

fn render_lifted_ops_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\taddress\tblock_id\tpredicate\topcode\tclass\tkind\tsemantics\tinputs\toutputs\tsource_operands\tdetail\tsource_text"
    )
    .expect("write to string");
    for op in &report.lifted_ops {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&op.sass_path.display().to_string()),
            tsv_display(&op.function),
            op.address,
            op.block_id
                .map(|block| block.to_string())
                .unwrap_or_default(),
            tsv(&display_optional(op.predicate.as_ref())),
            tsv(&op.opcode.to_string()),
            tsv(&op.class.to_string()),
            tsv(&op.kind.to_string()),
            tsv(&op.semantics.to_string()),
            tsv(&display_lifted_value_refs(&op.inputs)),
            tsv(&display_lifted_value_refs(&op.outputs)),
            tsv(&display_list(&op.source_operands)),
            tsv(&op.detail.to_string()),
            tsv(&op.source_text),
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
            tsv_display(&range.function),
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
        "sass_path\tfunction\taddress\tpredicate\tkind\tspace\twidth_bits\tvalue_register\taddress_expr\taddress_registers\taddress_base\toffset\tsource_text"
    )
    .expect("write to string");
    for access in &report.memory_accesses {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&access.sass_path.display().to_string()),
            tsv_display(&access.function),
            access.address,
            tsv(&display_optional(access.predicate.as_ref())),
            tsv(&access.kind.to_string()),
            tsv(&access.space.to_string()),
            access
                .width_bits
                .map(|bits| bits.to_string())
                .unwrap_or_default(),
            tsv(&access.value_register.to_string()),
            tsv(&access.memory_address.to_string()),
            tsv(&display_list(&access.address_registers)),
            tsv(&display_optional(access.address_base.as_ref())),
            tsv(&display_optional(access.offset.as_ref())),
            tsv(&access.source_text),
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
    writeln!(
        out,
        "sass_path\tfunction\taddress\topcode\treason\tsource_text"
    )
    .expect("write to string");
    for instruction in &report.unsupported_instructions {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}",
            tsv(&instruction.sass_path.display().to_string()),
            tsv_display(&instruction.function),
            instruction.address,
            tsv(&instruction.opcode.to_string()),
            tsv(&instruction.reason.to_string()),
            tsv(&instruction.source_text)
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

fn tsv_display(value: &impl fmt::Display) -> String {
    tsv(&value.to_string())
}

fn tsv(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
}
