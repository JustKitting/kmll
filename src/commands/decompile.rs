use std::{fmt, path::PathBuf};

use nn_rust_autotune::AutoOptimizeConfig;
use nn_rust_autotune::decompile::{
    DecompileAutotuneGemmOptions, DecompileAutotuneMatvecOptions, DecompileAutotuneMeasureOptions,
    DecompileAutotuneSassOptions, DecompileFixtureCoverageOptions, DecompileFixtureOptions,
    DecompilePtxProbeOptions, DecompiledAutotuneShape, PtxDecompileProbeKind,
    SassCoverageComparisonOptions, SassCoverageOptions, SassFileDecompileOptions,
    SimpleKernelFixtureKind, all_ptx_decompile_probe_kinds, all_simple_kernel_fixture_kinds,
    run_decompile_autotune_gemm, run_decompile_autotune_matvec, run_decompile_autotune_sass,
    run_decompile_fixture_coverage, run_decompile_fixtures, run_decompile_ptx_probes,
    run_sass_coverage_comparison, run_sass_coverage_scan, run_sass_file_decompile,
};
use nn_rust_inference::runtime;

use crate::{AppResult, invalid_input, parse_required_flag_value, parse_required_usize};

const DECOMPILE_FIXTURES_USAGE: &str =
    "kernel-decompile-fixtures [--fixture NAME|all] [--artifact-root PATH] [--compile-arch sm_120]";
const DECOMPILE_PTX_PROBES_USAGE: &str = "kernel-decompile-ptx-probes [--probe NAME|all] [--artifact-root PATH] [--compile-arch auto|sm_120]";
const DECOMPILE_FIXTURE_COVERAGE_USAGE: &str = "kernel-decompile-fixture-coverage [--fixture NAME|all] [--artifact-root PATH] [--compile-arch sm_120] [--out-dir PATH]";
const DECOMPILE_SASS_USAGE: &str =
    "kernel-decompile-sass SASS_PATH [--source PATH] [--out-dir PATH]";
const DECOMPILE_AUTOTUNE_MATVEC_USAGE: &str = "kernel-decompile-autotune-matvec ROWS COLS [--artifact-root PATH] [--compile-arch sm_120] [--beam-width N] [--max-steps N] [--min-score-improvement VALUE] [--require-launchable] [--measure] [--measure-repeat N] [--measure-warmup N] [--measure-device N]";
const DECOMPILE_AUTOTUNE_GEMM_USAGE: &str = "kernel-decompile-autotune-gemm M N K [--artifact-root PATH] [--compile-arch sm_120] [--beam-width N] [--max-steps N] [--min-score-improvement VALUE] [--require-launchable]";
const DECOMPILE_AUTOTUNE_SASS_USAGE: &str = "kernel-decompile-autotune-sass SASS_PATH (--matvec-bf16-row-major ROWS COLS | --gemm-f32-bf16-row-col-row M N K) [--function SYMBOL] [--source PATH] [--out-dir PATH] [--artifact-root PATH] [--compile-arch sm_120] [--beam-width N] [--max-steps N] [--min-score-improvement VALUE] [--require-launchable]";
const DECOMPILE_COVERAGE_USAGE: &str =
    "kernel-decompile-coverage [ROOT] [--root PATH] [--out-dir PATH]";
const DECOMPILE_COVERAGE_COMPARE_USAGE: &str =
    "kernel-decompile-coverage-compare BASELINE_ROOT CANDIDATE_ROOT [--out-dir PATH]";

pub(crate) fn run_kernel_decompile_coverage(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let mut options = SassCoverageOptions::default_artifact_scan();
    let mut saw_positional_root = false;

    while index < args.len() {
        match args[index].as_str() {
            "--root" => {
                options.root =
                    PathBuf::from(parse_required_flag_value(args, &mut index, "--root")?);
            }
            "--out-dir" => {
                options.output_dir =
                    PathBuf::from(parse_required_flag_value(args, &mut index, "--out-dir")?);
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-coverage unknown argument {flag:?}; usage: {DECOMPILE_COVERAGE_USAGE}"
                )));
            }
            root if !saw_positional_root => {
                options.root = PathBuf::from(root);
                saw_positional_root = true;
                index += 1;
            }
            extra => {
                return Err(invalid_input(format!(
                    "kernel-decompile-coverage unexpected argument {extra:?}; usage: {DECOMPILE_COVERAGE_USAGE}"
                )));
            }
        }
    }

    let report = run_sass_coverage_scan(&options)?;
    println!(
        "kernel_decompile_coverage root={} files_seen={} files_parsed={} scanned_architectures={} parse_errors={} parsed_instructions={} cfg_blocks={} cfg_edges={} dominator_blocks={} natural_loops={} regions={} dataflow_ops={} reaching_uses={} ssa_values={} def_use_edges={} value_ops={} lifted_ops={} live_ranges={} memory_accesses={} semantic_patterns={} known_opcodes={} locally_mapped_opcodes={} known_unobserved_opcodes={} opcode_probe_targets={} sm120_tensor_core_required={} sm120_tensor_core_supported={} sm120_tensor_core_missing={} known_unmapped_opcodes={} observed_unregistered_opcodes={} observed_unmapped_opcodes={} unsupported_instructions={} summary_path={} files_path={} opcode_catalog_path={} opcode_probe_targets_path={} sm120_tensor_core_support_path={} opcode_frequency_path={} opcode_signature_frequency_path={} cfg_blocks_path={} cfg_edges_path={} dominators_path={} natural_loops_path={} regions_path={} dataflow_path={} reaching_uses_path={} ssa_values_path={} def_use_edges_path={} value_ops_path={} lifted_ops_path={} live_ranges_path={} memory_accesses_path={} semantic_patterns_path={} semantic_pattern_frequency_path={} unsupported_instructions_path={}",
        report.root.display(),
        report.files.len(),
        report.parsed_file_count,
        display_list(&report.scanned_architectures),
        report.parse_error_count,
        report.parsed_instruction_count,
        report.cfg_block_count,
        report.cfg_edge_count,
        report.dominator_block_count,
        report.natural_loop_count,
        report.region_count,
        report.dataflow_op_count,
        report.reaching_use_count,
        report.ssa_value_count,
        report.def_use_edge_count,
        report.value_op_count,
        report.lifted_op_count,
        report.live_range_count,
        report.memory_access_count,
        report.semantic_pattern_count,
        report.known_opcode_count,
        report.locally_mapped_opcode_count,
        report.known_unobserved_opcode_count,
        report.opcode_probe_target_count,
        report.sm120_tensor_core_required_count,
        report.sm120_tensor_core_supported_count,
        report.sm120_tensor_core_missing_count,
        report.known_unmapped_opcode_count,
        report.observed_unregistered_opcode_count,
        report.observed_unmapped_opcode_count,
        report.unsupported_instruction_count,
        report.summary_path.display(),
        report.files_path.display(),
        report.opcode_catalog_path.display(),
        report.opcode_probe_targets_path.display(),
        report.sm120_tensor_core_support_path.display(),
        report.opcode_frequency_path.display(),
        report.opcode_signature_frequency_path.display(),
        report.cfg_blocks_path.display(),
        report.cfg_edges_path.display(),
        report.dominators_path.display(),
        report.natural_loops_path.display(),
        report.regions_path.display(),
        report.dataflow_path.display(),
        report.reaching_uses_path.display(),
        report.ssa_values_path.display(),
        report.def_use_edges_path.display(),
        report.value_ops_path.display(),
        report.lifted_ops_path.display(),
        report.live_ranges_path.display(),
        report.memory_accesses_path.display(),
        report.semantic_patterns_path.display(),
        report.semantic_pattern_frequency_path.display(),
        report.unsupported_instructions_path.display(),
    );
    Ok(())
}

pub(crate) fn run_kernel_decompile_coverage_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let mut baseline_root = None::<PathBuf>;
    let mut candidate_root = None::<PathBuf>;
    let mut output_dir = runtime::default_artifact_dir().join("decompile-coverage-compare");

    while index < args.len() {
        match args[index].as_str() {
            "--out-dir" => {
                output_dir =
                    PathBuf::from(parse_required_flag_value(args, &mut index, "--out-dir")?);
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-coverage-compare unknown argument {flag:?}; usage: {DECOMPILE_COVERAGE_COMPARE_USAGE}"
                )));
            }
            value if baseline_root.is_none() => {
                baseline_root = Some(PathBuf::from(value));
                index += 1;
            }
            value if candidate_root.is_none() => {
                candidate_root = Some(PathBuf::from(value));
                index += 1;
            }
            extra => {
                return Err(invalid_input(format!(
                    "kernel-decompile-coverage-compare unexpected argument {extra:?}; usage: {DECOMPILE_COVERAGE_COMPARE_USAGE}"
                )));
            }
        }
    }

    let baseline_root = baseline_root.ok_or_else(|| {
        invalid_input(format!(
            "kernel-decompile-coverage-compare requires BASELINE_ROOT; usage: {DECOMPILE_COVERAGE_COMPARE_USAGE}"
        ))
    })?;
    let candidate_root = candidate_root.ok_or_else(|| {
        invalid_input(format!(
            "kernel-decompile-coverage-compare requires CANDIDATE_ROOT; usage: {DECOMPILE_COVERAGE_COMPARE_USAGE}"
        ))
    })?;

    let report = run_sass_coverage_comparison(&SassCoverageComparisonOptions {
        baseline_root,
        candidate_root,
        output_dir,
    })?;
    println!(
        "kernel_decompile_coverage_compare baseline_root={} candidate_root={} baseline_files_seen={} candidate_files_seen={} baseline_probe_targets={} candidate_probe_targets={} newly_observed_opcodes={} no_longer_observed_opcodes={} coverage_changed_opcodes={} count_changed_opcodes={} resolved_probe_targets={} new_probe_targets={} summary_path={} opcode_delta_path={} resolved_probe_targets_path={} new_probe_targets_path={}",
        report.baseline_report.root.display(),
        report.candidate_report.root.display(),
        report.baseline_report.files.len(),
        report.candidate_report.files.len(),
        report.baseline_report.opcode_probe_target_count,
        report.candidate_report.opcode_probe_target_count,
        report.newly_observed_opcode_count,
        report.no_longer_observed_opcode_count,
        report.coverage_changed_opcode_count,
        report.count_changed_opcode_count,
        report.resolved_probe_targets.len(),
        report.new_probe_targets.len(),
        report.summary_path.display(),
        report.opcode_delta_path.display(),
        report.resolved_probe_targets_path.display(),
        report.new_probe_targets_path.display(),
    );
    Ok(())
}

pub(crate) fn run_kernel_decompile_sass(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    if index >= args.len() {
        return Err(invalid_input(format!(
            "kernel-decompile-sass requires SASS_PATH; usage: {DECOMPILE_SASS_USAGE}"
        )));
    }
    let sass_path = PathBuf::from(&args[index]);
    index += 1;
    let mut source_path = None;
    let mut output_dir = None;

    while index < args.len() {
        match args[index].as_str() {
            "--source" => {
                source_path = Some(PathBuf::from(parse_required_flag_value(
                    args, &mut index, "--source",
                )?));
            }
            "--out-dir" => {
                output_dir = Some(PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--out-dir",
                )?));
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-sass unknown argument {flag:?}; usage: {DECOMPILE_SASS_USAGE}"
                )));
            }
            extra => {
                return Err(invalid_input(format!(
                    "kernel-decompile-sass unexpected argument {extra:?}; usage: {DECOMPILE_SASS_USAGE}"
                )));
            }
        }
    }

    let report = run_sass_file_decompile(&SassFileDecompileOptions {
        sass_path,
        source_path,
        output_dir,
    })?;
    println!(
        "kernel_decompile_sass parsed_instructions={} cfg_blocks={} cfg_edges={} dominator_blocks={} natural_loops={} regions={} reaching_uses={} ssa_values={} def_use_edges={} value_ops={} lifted_ops={} live_ranges={} memory_accesses={} semantic_patterns={} unsupported_instructions={} sass_path={} ir_path={} lifted_ir_path={} analysis_path={} pattern_path={} side_by_side_path={}",
        report.parsed_instruction_count,
        report.cfg_block_count,
        report.cfg_edge_count,
        report.dominator_block_count,
        report.natural_loop_count,
        report.region_count,
        report.reaching_use_count,
        report.ssa_value_count,
        report.def_use_edge_count,
        report.value_op_count,
        report.lifted_op_count,
        report.live_range_count,
        report.memory_access_count,
        report.semantic_pattern_count,
        report.unsupported_instruction_count,
        report.sass_path.display(),
        report.ir_path.display(),
        report.lifted_ir_path.display(),
        report.analysis_path.display(),
        report.pattern_path.display(),
        report.side_by_side_path.display(),
    );
    Ok(())
}

pub(crate) fn run_kernel_decompile_autotune_matvec(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let rows = parse_required_usize(args, &mut index, "rows", "kernel-decompile-autotune-matvec")?;
    let cols = parse_required_usize(args, &mut index, "cols", "kernel-decompile-autotune-matvec")?;
    let mut options = DecompileAutotuneMatvecOptions::sm120_default(rows, cols);
    let mut config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };

    while index < args.len() {
        match args[index].as_str() {
            "--artifact-root" => {
                options.artifact_root = PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--artifact-root",
                )?);
            }
            "--compile-arch" => {
                options.compile_arch =
                    parse_required_flag_value(args, &mut index, "--compile-arch")?.to_string();
            }
            "--beam-width" => {
                let value = parse_required_flag_value(args, &mut index, "--beam-width")?;
                config.beam_width = value.parse().map_err(|error| {
                    invalid_input(format!("invalid --beam-width {value:?}: {error}"))
                })?;
            }
            "--max-steps" => {
                let value = parse_required_flag_value(args, &mut index, "--max-steps")?;
                config.max_steps = value.parse().map_err(|error| {
                    invalid_input(format!("invalid --max-steps {value:?}: {error}"))
                })?;
            }
            "--min-score-improvement" => {
                let value = parse_required_flag_value(args, &mut index, "--min-score-improvement")?;
                config.min_score_improvement = value.parse().map_err(|error| {
                    invalid_input(format!(
                        "invalid --min-score-improvement {value:?}: {error}"
                    ))
                })?;
            }
            "--require-launchable" => {
                config.require_launchable = true;
                index += 1;
            }
            "--measure" => {
                options.measure = Some(DecompileAutotuneMeasureOptions::default_sm120());
                index += 1;
            }
            "--measure-repeat" => {
                let value = parse_required_flag_value(args, &mut index, "--measure-repeat")?;
                ensure_matvec_measure_options(&mut options).repeat_count =
                    parse_positive_usize_flag(
                        value,
                        "kernel-decompile-autotune-matvec",
                        "--measure-repeat",
                    )?;
            }
            "--measure-warmup" => {
                let value = parse_required_flag_value(args, &mut index, "--measure-warmup")?;
                ensure_matvec_measure_options(&mut options).warmup_count = parse_usize_flag(
                    value,
                    "kernel-decompile-autotune-matvec",
                    "--measure-warmup",
                )?;
            }
            "--measure-device" => {
                let value = parse_required_flag_value(args, &mut index, "--measure-device")?;
                ensure_matvec_measure_options(&mut options).device_index = parse_usize_flag(
                    value,
                    "kernel-decompile-autotune-matvec",
                    "--measure-device",
                )?;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-autotune-matvec unknown argument {flag:?}; usage: {DECOMPILE_AUTOTUNE_MATVEC_USAGE}"
                )));
            }
            extra => {
                return Err(invalid_input(format!(
                    "kernel-decompile-autotune-matvec unexpected argument {extra:?}; usage: {DECOMPILE_AUTOTUNE_MATVEC_USAGE}"
                )));
            }
        }
    }
    options.config = config;

    let report = run_decompile_autotune_matvec(&options)?;
    println!(
        "kernel_decompile_autotune_matvec rows={} cols={} naive_symbol={} parsed_instructions={} semantic_patterns={} unsupported_instructions={} has_bf16_descriptor_load={} has_bf16_widen={} has_f32_mul_add={} has_f32_fused_multiply_add={} has_warp_reduce_sum={} best_symbol={} best_action_count={} best_action_ops={} source_score={} source_score_source={} best_score={} best_score_source={} explored={} rejected={} improving_steps={} optimized_parsed_instructions={} optimized_semantic_patterns={} optimized_unsupported_instructions={} optimized_has_bf16_descriptor_load={} optimized_has_bf16_widen={} optimized_has_f32_mul_add={} optimized_has_f32_fused_multiply_add={} optimized_has_warp_reduce_sum={} source_path={} ptx_path={} cubin_path={} sass_path={} ir_path={} pattern_path={} side_by_side_path={} auto_report_path={} auto_report_visual_svg_path={} auto_report_visual_html_path={} overview_path={} overview_graph_path={} optimized_source_path={} optimized_ptx_path={} optimized_cubin_path={} optimized_sass_path={} optimized_ir_path={} optimized_pattern_path={} optimized_side_by_side_path={}",
        report.rows,
        report.cols,
        report.naive_symbol,
        report.parsed_instruction_count,
        report.semantic_pattern_count,
        report.unsupported_instruction_count,
        report.evidence.has_bf16_descriptor_load,
        report.evidence.has_bf16_widen,
        report.evidence.has_f32_mul_add,
        report.evidence.has_f32_fused_multiply_add,
        report.evidence.has_warp_reduce_sum,
        report.best_symbol,
        report.best_action_count,
        report.best_action_ops.join(","),
        optional_f64_string(report.source_score),
        optional_string(report.source_score_source.as_deref()),
        report
            .best_score
            .map(|score| score.to_string())
            .unwrap_or_else(|| "none".to_string()),
        optional_string(report.best_score_source.as_deref()),
        report.explored,
        report.rejected,
        report.improving_step_count,
        report.optimized_parsed_instruction_count,
        report.optimized_semantic_pattern_count,
        report.optimized_unsupported_instruction_count,
        report.optimized_evidence.has_bf16_descriptor_load,
        report.optimized_evidence.has_bf16_widen,
        report.optimized_evidence.has_f32_mul_add,
        report.optimized_evidence.has_f32_fused_multiply_add,
        report.optimized_evidence.has_warp_reduce_sum,
        report.source_path.display(),
        report.ptx_path.display(),
        report.cubin_path.display(),
        report.sass_path.display(),
        report.ir_path.display(),
        report.pattern_path.display(),
        report.side_by_side_path.display(),
        report.auto_report_path.display(),
        report.auto_report_visual_svg_path.display(),
        report.auto_report_visual_html_path.display(),
        report.overview_path.display(),
        report.overview_graph_path.display(),
        report.optimized_source_path.display(),
        report.optimized_ptx_path.display(),
        report.optimized_cubin_path.display(),
        report.optimized_sass_path.display(),
        report.optimized_ir_path.display(),
        report.optimized_pattern_path.display(),
        report.optimized_side_by_side_path.display(),
    );
    Ok(())
}

fn ensure_matvec_measure_options(
    options: &mut DecompileAutotuneMatvecOptions,
) -> &mut DecompileAutotuneMeasureOptions {
    options
        .measure
        .get_or_insert_with(DecompileAutotuneMeasureOptions::default_sm120)
}

fn parse_positive_usize_flag(value: &str, command: &str, flag: &str) -> AppResult<usize> {
    let parsed = parse_usize_flag(value, command, flag)?;
    if parsed == 0 {
        return Err(invalid_input(format!("{command} {flag} must be nonzero")));
    }
    Ok(parsed)
}

fn parse_usize_flag(value: &str, command: &str, flag: &str) -> AppResult<usize> {
    value
        .parse()
        .map_err(|error| invalid_input(format!("invalid {flag} {value:?} for {command}: {error}")))
}

fn optional_f64_string(value: Option<f64>) -> String {
    value
        .map(|score| score.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn optional_string(value: Option<&str>) -> String {
    value.unwrap_or("none").to_string()
}

pub(crate) fn run_kernel_decompile_autotune_gemm(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let m = parse_required_usize(args, &mut index, "m", "kernel-decompile-autotune-gemm")?;
    let n = parse_required_usize(args, &mut index, "n", "kernel-decompile-autotune-gemm")?;
    let k = parse_required_usize(args, &mut index, "k", "kernel-decompile-autotune-gemm")?;
    let mut options = DecompileAutotuneGemmOptions::sm120_default(m, n, k);
    let mut config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };

    while index < args.len() {
        match args[index].as_str() {
            "--artifact-root" => {
                options.artifact_root = PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--artifact-root",
                )?);
            }
            "--compile-arch" => {
                options.compile_arch =
                    parse_required_flag_value(args, &mut index, "--compile-arch")?.to_string();
            }
            "--beam-width" => {
                let value = parse_required_flag_value(args, &mut index, "--beam-width")?;
                config.beam_width = value.parse().map_err(|error| {
                    invalid_input(format!("invalid --beam-width {value:?}: {error}"))
                })?;
            }
            "--max-steps" => {
                let value = parse_required_flag_value(args, &mut index, "--max-steps")?;
                config.max_steps = value.parse().map_err(|error| {
                    invalid_input(format!("invalid --max-steps {value:?}: {error}"))
                })?;
            }
            "--min-score-improvement" => {
                let value = parse_required_flag_value(args, &mut index, "--min-score-improvement")?;
                config.min_score_improvement = value.parse().map_err(|error| {
                    invalid_input(format!(
                        "invalid --min-score-improvement {value:?}: {error}"
                    ))
                })?;
            }
            "--require-launchable" => {
                config.require_launchable = true;
                index += 1;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-autotune-gemm unknown argument {flag:?}; usage: {DECOMPILE_AUTOTUNE_GEMM_USAGE}"
                )));
            }
            extra => {
                return Err(invalid_input(format!(
                    "kernel-decompile-autotune-gemm unexpected argument {extra:?}; usage: {DECOMPILE_AUTOTUNE_GEMM_USAGE}"
                )));
            }
        }
    }
    options.config = config;

    let report = run_decompile_autotune_gemm(&options)?;
    println!(
        "kernel_decompile_autotune_gemm m={} n={} k={} source_symbol={} parsed_instructions={} semantic_patterns={} unsupported_instructions={} has_f32_descriptor_load={} has_bf16_descriptor_load={} has_bf16_widen={} has_shared_store={} has_shared_load={} has_barrier={} has_f32_mul_add={} has_f32_fused_multiply_add={} has_descriptor_store={} best_symbol={} best_action_count={} best_action_ops={} best_score={} explored={} rejected={} improving_steps={} optimized_parsed_instructions={} optimized_semantic_patterns={} optimized_unsupported_instructions={} optimized_has_f32_descriptor_load={} optimized_has_bf16_descriptor_load={} optimized_has_bf16_widen={} optimized_has_shared_store={} optimized_has_shared_load={} optimized_has_barrier={} optimized_has_f32_mul_add={} optimized_has_f32_fused_multiply_add={} optimized_has_descriptor_store={} source_path={} ptx_path={} cubin_path={} sass_path={} ir_path={} pattern_path={} side_by_side_path={} auto_report_path={} auto_report_visual_svg_path={} auto_report_visual_html_path={} overview_path={} overview_graph_path={} optimized_source_path={} optimized_ptx_path={} optimized_cubin_path={} optimized_sass_path={} optimized_ir_path={} optimized_pattern_path={} optimized_side_by_side_path={}",
        report.m,
        report.n,
        report.k,
        report.source_symbol,
        report.parsed_instruction_count,
        report.semantic_pattern_count,
        report.unsupported_instruction_count,
        report.evidence.has_f32_descriptor_load,
        report.evidence.has_bf16_descriptor_load,
        report.evidence.has_bf16_widen,
        report.evidence.has_shared_store,
        report.evidence.has_shared_load,
        report.evidence.has_barrier,
        report.evidence.has_f32_mul_add,
        report.evidence.has_f32_fused_multiply_add,
        report.evidence.has_descriptor_store,
        report.best_symbol,
        report.best_action_count,
        report.best_action_ops.join(","),
        report
            .best_score
            .map(|score| score.to_string())
            .unwrap_or_else(|| "none".to_string()),
        report.explored,
        report.rejected,
        report.improving_step_count,
        report.optimized_parsed_instruction_count,
        report.optimized_semantic_pattern_count,
        report.optimized_unsupported_instruction_count,
        report.optimized_evidence.has_f32_descriptor_load,
        report.optimized_evidence.has_bf16_descriptor_load,
        report.optimized_evidence.has_bf16_widen,
        report.optimized_evidence.has_shared_store,
        report.optimized_evidence.has_shared_load,
        report.optimized_evidence.has_barrier,
        report.optimized_evidence.has_f32_mul_add,
        report.optimized_evidence.has_f32_fused_multiply_add,
        report.optimized_evidence.has_descriptor_store,
        report.source_path.display(),
        report.ptx_path.display(),
        report.cubin_path.display(),
        report.sass_path.display(),
        report.ir_path.display(),
        report.pattern_path.display(),
        report.side_by_side_path.display(),
        report.auto_report_path.display(),
        report.auto_report_visual_svg_path.display(),
        report.auto_report_visual_html_path.display(),
        report.overview_path.display(),
        report.overview_graph_path.display(),
        report.optimized_source_path.display(),
        report.optimized_ptx_path.display(),
        report.optimized_cubin_path.display(),
        report.optimized_sass_path.display(),
        report.optimized_ir_path.display(),
        report.optimized_pattern_path.display(),
        report.optimized_side_by_side_path.display(),
    );
    Ok(())
}

pub(crate) fn run_kernel_decompile_autotune_sass(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    if index >= args.len() {
        return Err(invalid_input(format!(
            "kernel-decompile-autotune-sass requires SASS_PATH; usage: {DECOMPILE_AUTOTUNE_SASS_USAGE}"
        )));
    }
    let sass_path = PathBuf::from(&args[index]);
    index += 1;
    let mut shape = None::<DecompiledAutotuneShape>;
    let mut source_path = None::<PathBuf>;
    let mut output_dir = None::<PathBuf>;
    let mut artifact_root = runtime::default_artifact_dir()
        .join("decompile-autotune")
        .join("input-sass");
    let mut compile_arch = "sm_120".to_string();
    let mut function_symbol = None::<String>;
    let mut config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };

    while index < args.len() {
        match args[index].as_str() {
            "--matvec-bf16-row-major" => {
                if shape.is_some() {
                    return Err(invalid_input(
                        "kernel-decompile-autotune-sass accepts exactly one shape flag",
                    ));
                }
                index += 1;
                let rows = parse_required_usize(
                    args,
                    &mut index,
                    "rows",
                    "kernel-decompile-autotune-sass",
                )?;
                let cols = parse_required_usize(
                    args,
                    &mut index,
                    "cols",
                    "kernel-decompile-autotune-sass",
                )?;
                shape = Some(DecompiledAutotuneShape::MatvecBf16RowMajor { rows, cols });
            }
            "--gemm-f32-bf16-row-col-row" => {
                if shape.is_some() {
                    return Err(invalid_input(
                        "kernel-decompile-autotune-sass accepts exactly one shape flag",
                    ));
                }
                index += 1;
                let m =
                    parse_required_usize(args, &mut index, "m", "kernel-decompile-autotune-sass")?;
                let n =
                    parse_required_usize(args, &mut index, "n", "kernel-decompile-autotune-sass")?;
                let k =
                    parse_required_usize(args, &mut index, "k", "kernel-decompile-autotune-sass")?;
                shape = Some(DecompiledAutotuneShape::GemmF32Bf16RowColRow { m, n, k });
            }
            "--function" => {
                function_symbol =
                    Some(parse_required_flag_value(args, &mut index, "--function")?.to_string());
            }
            "--source" => {
                source_path = Some(PathBuf::from(parse_required_flag_value(
                    args, &mut index, "--source",
                )?));
            }
            "--out-dir" => {
                output_dir = Some(PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--out-dir",
                )?));
            }
            "--artifact-root" => {
                artifact_root = PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--artifact-root",
                )?);
            }
            "--compile-arch" => {
                compile_arch =
                    parse_required_flag_value(args, &mut index, "--compile-arch")?.to_string();
            }
            "--beam-width" => {
                let value = parse_required_flag_value(args, &mut index, "--beam-width")?;
                config.beam_width = value.parse().map_err(|error| {
                    invalid_input(format!("invalid --beam-width {value:?}: {error}"))
                })?;
            }
            "--max-steps" => {
                let value = parse_required_flag_value(args, &mut index, "--max-steps")?;
                config.max_steps = value.parse().map_err(|error| {
                    invalid_input(format!("invalid --max-steps {value:?}: {error}"))
                })?;
            }
            "--min-score-improvement" => {
                let value = parse_required_flag_value(args, &mut index, "--min-score-improvement")?;
                config.min_score_improvement = value.parse().map_err(|error| {
                    invalid_input(format!(
                        "invalid --min-score-improvement {value:?}: {error}"
                    ))
                })?;
            }
            "--require-launchable" => {
                config.require_launchable = true;
                index += 1;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-autotune-sass unknown argument {flag:?}; usage: {DECOMPILE_AUTOTUNE_SASS_USAGE}"
                )));
            }
            extra => {
                return Err(invalid_input(format!(
                    "kernel-decompile-autotune-sass unexpected argument {extra:?}; usage: {DECOMPILE_AUTOTUNE_SASS_USAGE}"
                )));
            }
        }
    }

    let shape = shape.ok_or_else(|| {
        invalid_input(format!(
            "kernel-decompile-autotune-sass requires a shape flag; usage: {DECOMPILE_AUTOTUNE_SASS_USAGE}"
        ))
    })?;
    let report = run_decompile_autotune_sass(&DecompileAutotuneSassOptions {
        sass_path,
        source_path,
        output_dir,
        artifact_root,
        compile_arch,
        function_symbol,
        shape,
        config,
    })?;
    println!(
        "kernel_decompile_autotune_sass shape={} function_symbol={} parsed_instructions={} semantic_patterns={} unsupported_instructions={} has_f32_descriptor_load={} has_bf16_descriptor_load={} has_bf16_widen={} has_shared_store={} has_shared_load={} has_barrier={} has_f32_mul_add={} has_f32_fused_multiply_add={} has_descriptor_store={} has_warp_reduce_sum={} best_symbol={} best_action_count={} best_action_ops={} best_score={} explored={} rejected={} improving_steps={} optimized_parsed_instructions={} optimized_semantic_patterns={} optimized_unsupported_instructions={} optimized_has_f32_descriptor_load={} optimized_has_bf16_descriptor_load={} optimized_has_bf16_widen={} optimized_has_shared_store={} optimized_has_shared_load={} optimized_has_barrier={} optimized_has_f32_mul_add={} optimized_has_f32_fused_multiply_add={} optimized_has_descriptor_store={} optimized_has_warp_reduce_sum={} sass_path={} source_path={} output_dir={} ir_path={} pattern_path={} side_by_side_path={} auto_report_path={} auto_report_visual_svg_path={} auto_report_visual_html_path={} overview_path={} overview_graph_path={} optimized_source_path={} optimized_ptx_path={} optimized_cubin_path={} optimized_sass_path={} optimized_ir_path={} optimized_pattern_path={} optimized_side_by_side_path={}",
        decompiled_shape_label(shape),
        report.function_symbol,
        report.parsed_instruction_count,
        report.semantic_pattern_count,
        report.unsupported_instruction_count,
        report.evidence.has_f32_descriptor_load,
        report.evidence.has_bf16_descriptor_load,
        report.evidence.has_bf16_widen,
        report.evidence.has_shared_store,
        report.evidence.has_shared_load,
        report.evidence.has_barrier,
        report.evidence.has_f32_mul_add,
        report.evidence.has_f32_fused_multiply_add,
        report.evidence.has_descriptor_store,
        report.evidence.has_warp_reduce_sum,
        report.best_symbol,
        report.best_action_count,
        report.best_action_ops.join(","),
        report
            .best_score
            .map(|score| score.to_string())
            .unwrap_or_else(|| "none".to_string()),
        report.explored,
        report.rejected,
        report.improving_step_count,
        report.optimized_parsed_instruction_count,
        report.optimized_semantic_pattern_count,
        report.optimized_unsupported_instruction_count,
        report.optimized_evidence.has_f32_descriptor_load,
        report.optimized_evidence.has_bf16_descriptor_load,
        report.optimized_evidence.has_bf16_widen,
        report.optimized_evidence.has_shared_store,
        report.optimized_evidence.has_shared_load,
        report.optimized_evidence.has_barrier,
        report.optimized_evidence.has_f32_mul_add,
        report.optimized_evidence.has_f32_fused_multiply_add,
        report.optimized_evidence.has_descriptor_store,
        report.optimized_evidence.has_warp_reduce_sum,
        report.sass_path.display(),
        report
            .source_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
        report.output_dir.display(),
        report.ir_path.display(),
        report.pattern_path.display(),
        report.side_by_side_path.display(),
        report.auto_report_path.display(),
        report.auto_report_visual_svg_path.display(),
        report.auto_report_visual_html_path.display(),
        report.overview_path.display(),
        report.overview_graph_path.display(),
        report.optimized_source_path.display(),
        report.optimized_ptx_path.display(),
        report.optimized_cubin_path.display(),
        report.optimized_sass_path.display(),
        report.optimized_ir_path.display(),
        report.optimized_pattern_path.display(),
        report.optimized_side_by_side_path.display(),
    );
    Ok(())
}

fn decompiled_shape_label(shape: DecompiledAutotuneShape) -> &'static str {
    match shape {
        DecompiledAutotuneShape::MatvecBf16RowMajor { .. } => "matvec-bf16-row-major",
        DecompiledAutotuneShape::GemmF32Bf16RowColRow { .. } => "gemm-f32-bf16-row-col-row",
    }
}

pub(crate) fn run_kernel_decompile_fixtures(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let mut artifact_root = runtime::default_artifact_dir().join("decompile-fixtures");
    let mut compile_arch = "sm_120".to_string();
    let mut fixtures = Vec::new();

    while index < args.len() {
        match args[index].as_str() {
            "--artifact-root" => {
                artifact_root = PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--artifact-root",
                )?);
            }
            "--compile-arch" => {
                compile_arch =
                    parse_required_flag_value(args, &mut index, "--compile-arch")?.to_string();
            }
            "--fixture" => {
                let value = parse_required_flag_value(args, &mut index, "--fixture")?;
                push_fixture_arg(value, &mut fixtures)?;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-fixtures unknown argument {flag:?}; usage: {DECOMPILE_FIXTURES_USAGE}"
                )));
            }
            value => {
                push_fixture_arg(value, &mut fixtures)?;
                index += 1;
            }
        }
    }

    if fixtures.is_empty() {
        fixtures.push(SimpleKernelFixtureKind::I32Add);
    }

    let options = DecompileFixtureOptions {
        artifact_root,
        compile_arch,
        fixtures,
    };
    let reports = run_decompile_fixtures(&options)?;
    for report in reports {
        println!(
            "kernel_decompile_fixture fixture={} symbol={} parsed_instructions={} cfg_blocks={} cfg_edges={} dominator_blocks={} natural_loops={} regions={} reaching_uses={} ssa_values={} def_use_edges={} value_ops={} lifted_ops={} live_ranges={} memory_accesses={} semantic_patterns={} unsupported_instructions={} source_path={} ptx_path={} cubin_path={} sass_path={} ir_path={} lifted_ir_path={} analysis_path={} pattern_path={} side_by_side_path={}",
            report.fixture.name(),
            report.symbol,
            report.parsed_instruction_count,
            report.cfg_block_count,
            report.cfg_edge_count,
            report.dominator_block_count,
            report.natural_loop_count,
            report.region_count,
            report.reaching_use_count,
            report.ssa_value_count,
            report.def_use_edge_count,
            report.value_op_count,
            report.lifted_op_count,
            report.live_range_count,
            report.memory_access_count,
            report.semantic_pattern_count,
            report.unsupported_instruction_count,
            report.source_path.display(),
            report.ptx_path.display(),
            report.cubin_path.display(),
            report.sass_path.display(),
            report.ir_path.display(),
            report.lifted_ir_path.display(),
            report.analysis_path.display(),
            report.pattern_path.display(),
            report.side_by_side_path.display(),
        );
    }
    Ok(())
}

pub(crate) fn run_kernel_decompile_ptx_probes(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let mut options = DecompilePtxProbeOptions::sm120_default();
    let mut saw_probe_arg = false;

    while index < args.len() {
        match args[index].as_str() {
            "--artifact-root" => {
                options.artifact_root = PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--artifact-root",
                )?);
            }
            "--compile-arch" => {
                options.compile_arch =
                    parse_required_flag_value(args, &mut index, "--compile-arch")?.to_string();
            }
            "--probe" => {
                let value = parse_required_flag_value(args, &mut index, "--probe")?;
                if !saw_probe_arg {
                    options.probes.clear();
                    saw_probe_arg = true;
                }
                push_ptx_probe_arg(value, &mut options.probes)?;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-ptx-probes unknown argument {flag:?}; usage: {DECOMPILE_PTX_PROBES_USAGE}"
                )));
            }
            value => {
                if !saw_probe_arg {
                    options.probes.clear();
                    saw_probe_arg = true;
                }
                push_ptx_probe_arg(value, &mut options.probes)?;
                index += 1;
            }
        }
    }

    if options.probes.is_empty() {
        options.probes = all_ptx_decompile_probe_kinds();
    }

    let reports = run_decompile_ptx_probes(&options)?;
    for report in reports {
        println!(
            "kernel_decompile_ptx_probe probe={} symbol={} compile_arch={} parsed_instructions={} unsupported_instructions={} ptx_path={} cubin_path={} nvdisasm_sass_path={} cuobjdump_sass_path={}",
            report.probe.name(),
            report.symbol,
            report.compile_arch,
            report.parsed_instruction_count,
            report.unsupported_instruction_count,
            report.ptx_path.display(),
            report.cubin_path.display(),
            report.nvdisasm_sass_path.display(),
            report.cuobjdump_sass_path.display(),
        );
    }
    Ok(())
}

pub(crate) fn run_kernel_decompile_fixture_coverage(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let mut options = DecompileFixtureCoverageOptions::sm120_all_default();
    let mut saw_coverage_out_dir = false;
    let mut saw_fixture_arg = false;

    while index < args.len() {
        match args[index].as_str() {
            "--artifact-root" => {
                let artifact_root = PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--artifact-root",
                )?);
                if !saw_coverage_out_dir {
                    options.coverage_output_dir = artifact_root.join("coverage");
                }
                options.fixture_options.artifact_root = artifact_root;
            }
            "--compile-arch" => {
                options.fixture_options.compile_arch =
                    parse_required_flag_value(args, &mut index, "--compile-arch")?.to_string();
            }
            "--fixture" => {
                let value = parse_required_flag_value(args, &mut index, "--fixture")?;
                if !saw_fixture_arg {
                    options.fixture_options.fixtures.clear();
                    saw_fixture_arg = true;
                }
                push_fixture_arg(value, &mut options.fixture_options.fixtures)?;
            }
            "--out-dir" => {
                options.coverage_output_dir =
                    PathBuf::from(parse_required_flag_value(args, &mut index, "--out-dir")?);
                saw_coverage_out_dir = true;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-fixture-coverage unknown argument {flag:?}; usage: {DECOMPILE_FIXTURE_COVERAGE_USAGE}"
                )));
            }
            value => {
                if !saw_fixture_arg {
                    options.fixture_options.fixtures.clear();
                    saw_fixture_arg = true;
                }
                push_fixture_arg(value, &mut options.fixture_options.fixtures)?;
                index += 1;
            }
        }
    }

    if options.fixture_options.fixtures.is_empty() {
        options.fixture_options.fixtures = all_simple_kernel_fixture_kinds();
    }

    let report = run_decompile_fixture_coverage(&options)?;
    println!(
        "kernel_decompile_fixture_coverage fixtures={} root={} files_seen={} files_parsed={} scanned_architectures={} parse_errors={} parsed_instructions={} known_opcodes={} locally_mapped_opcodes={} known_unobserved_opcodes={} opcode_probe_targets={} sm120_tensor_core_required={} sm120_tensor_core_supported={} sm120_tensor_core_missing={} observed_unregistered_opcodes={} observed_unmapped_opcodes={} unsupported_instructions={} summary_path={} files_path={} opcode_catalog_path={} opcode_probe_targets_path={} sm120_tensor_core_support_path={}",
        report.fixture_reports.len(),
        report.coverage_report.root.display(),
        report.coverage_report.files.len(),
        report.coverage_report.parsed_file_count,
        display_list(&report.coverage_report.scanned_architectures),
        report.coverage_report.parse_error_count,
        report.coverage_report.parsed_instruction_count,
        report.coverage_report.known_opcode_count,
        report.coverage_report.locally_mapped_opcode_count,
        report.coverage_report.known_unobserved_opcode_count,
        report.coverage_report.opcode_probe_target_count,
        report.coverage_report.sm120_tensor_core_required_count,
        report.coverage_report.sm120_tensor_core_supported_count,
        report.coverage_report.sm120_tensor_core_missing_count,
        report.coverage_report.observed_unregistered_opcode_count,
        report.coverage_report.observed_unmapped_opcode_count,
        report.coverage_report.unsupported_instruction_count,
        report.coverage_report.summary_path.display(),
        report.coverage_report.files_path.display(),
        report.coverage_report.opcode_catalog_path.display(),
        report.coverage_report.opcode_probe_targets_path.display(),
        report
            .coverage_report
            .sm120_tensor_core_support_path
            .display(),
    );
    for fixture in report.fixture_reports {
        println!(
            "kernel_decompile_fixture fixture={} symbol={} parsed_instructions={} unsupported_instructions={} sass_path={}",
            fixture.fixture.name(),
            fixture.symbol,
            fixture.parsed_instruction_count,
            fixture.unsupported_instruction_count,
            fixture.sass_path.display(),
        );
    }
    Ok(())
}

fn push_fixture_arg(value: &str, fixtures: &mut Vec<SimpleKernelFixtureKind>) -> AppResult<()> {
    if value == "all" {
        for fixture in all_simple_kernel_fixture_kinds() {
            push_unique_fixture(fixtures, fixture);
        }
        return Ok(());
    }
    let fixture = SimpleKernelFixtureKind::parse(value).ok_or_else(|| {
        invalid_input(format!(
            "unknown decompile fixture {value:?}; usage: {DECOMPILE_FIXTURES_USAGE}"
        ))
    })?;
    push_unique_fixture(fixtures, fixture);
    Ok(())
}

fn push_ptx_probe_arg(value: &str, probes: &mut Vec<PtxDecompileProbeKind>) -> AppResult<()> {
    if value == "all" {
        for probe in all_ptx_decompile_probe_kinds() {
            push_unique_ptx_probe(probes, probe);
        }
        return Ok(());
    }
    let probe = PtxDecompileProbeKind::parse(value).ok_or_else(|| {
        invalid_input(format!(
            "unknown PTX decompile probe {value:?}; usage: {DECOMPILE_PTX_PROBES_USAGE}"
        ))
    })?;
    push_unique_ptx_probe(probes, probe);
    Ok(())
}

fn push_unique_fixture(
    fixtures: &mut Vec<SimpleKernelFixtureKind>,
    fixture: SimpleKernelFixtureKind,
) {
    if !fixtures.contains(&fixture) {
        fixtures.push(fixture);
    }
}

fn push_unique_ptx_probe(probes: &mut Vec<PtxDecompileProbeKind>, probe: PtxDecompileProbeKind) {
    if !probes.contains(&probe) {
        probes.push(probe);
    }
}

fn display_list<T: fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}
