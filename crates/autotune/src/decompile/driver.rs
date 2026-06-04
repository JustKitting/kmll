use std::{
    error::Error,
    fmt, fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
};

use nn_rust_inference::runtime;

use crate::autotune::{
    AutoOptimizeConfig, GemmRustCudaGenerator, GemmSearchProblem, GemmTileShape,
    InferenceKernelRustCudaGenerator, KernelArtifactStore, KernelSourceGenerator,
    MatvecRustCudaGenerator, MatvecSearchProblem, auto_optimize_inference_kernel,
    compile_standalone_kernel_crate, standalone_cargo_toml, standalone_main_source,
};

use super::{
    driver_support::{
        absolute_path, render_sass_file_side_by_side, render_side_by_side, run_capture, run_checked,
    },
    *,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureOptions {
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub fixtures: Vec<SimpleKernelFixtureKind>,
}

impl DecompileFixtureOptions {
    pub fn sm120_default() -> Self {
        Self {
            artifact_root: runtime::default_artifact_dir().join("decompile-fixtures"),
            compile_arch: "sm_120".to_string(),
            fixtures: vec![SimpleKernelFixtureKind::I32Add],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureCoverageOptions {
    pub fixture_options: DecompileFixtureOptions,
    pub coverage_output_dir: PathBuf,
}

impl DecompileFixtureCoverageOptions {
    pub fn sm120_all_default() -> Self {
        let artifact_root = runtime::default_artifact_dir().join("decompile-fixtures");
        Self {
            fixture_options: DecompileFixtureOptions {
                artifact_root: artifact_root.clone(),
                compile_arch: "sm_120".to_string(),
                fixtures: all_simple_kernel_fixture_kinds(),
            },
            coverage_output_dir: artifact_root.join("coverage"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureReport {
    pub fixture: SimpleKernelFixtureKind,
    pub symbol: String,
    pub crate_dir: PathBuf,
    pub source_path: PathBuf,
    pub ptx_path: PathBuf,
    pub cubin_path: PathBuf,
    pub sass_path: PathBuf,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureCoverageReport {
    pub fixture_reports: Vec<DecompileFixtureReport>,
    pub coverage_report: SassCoverageReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompilePtxProbeOptions {
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub probes: Vec<PtxDecompileProbeKind>,
}

impl DecompilePtxProbeOptions {
    pub fn sm120_default() -> Self {
        Self {
            artifact_root: runtime::default_artifact_dir().join("decompile-probes"),
            compile_arch: "sm_120".to_string(),
            probes: vec![
                PtxDecompileProbeKind::TensorCoreHmma,
                PtxDecompileProbeKind::TensorCoreImma,
                PtxDecompileProbeKind::TensorCoreDmma,
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompilePtxProbeReport {
    pub probe: PtxDecompileProbeKind,
    pub symbol: String,
    pub probe_dir: PathBuf,
    pub ptx_path: PathBuf,
    pub cubin_path: PathBuf,
    pub nvdisasm_sass_path: PathBuf,
    pub cuobjdump_sass_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneMatvecOptions {
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub rows: usize,
    pub cols: usize,
    pub config: AutoOptimizeConfig,
}

impl DecompileAutotuneMatvecOptions {
    pub fn sm120_default(rows: usize, cols: usize) -> Self {
        Self {
            artifact_root: runtime::default_artifact_dir()
                .join("decompile-autotune")
                .join("matvec-bf16-row-major"),
            compile_arch: "sm_120".to_string(),
            rows,
            cols,
            config: AutoOptimizeConfig {
                beam_width: 4,
                max_steps: 2,
                require_launchable: false,
                min_score_improvement: 0.0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneMatvecReport {
    pub rows: usize,
    pub cols: usize,
    pub naive_symbol: String,
    pub source_path: PathBuf,
    pub ptx_path: PathBuf,
    pub cubin_path: PathBuf,
    pub sass_path: PathBuf,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
    pub semantic_pattern_count: usize,
    pub evidence: DecompiledAutotuneEvidence,
    pub operation_name: String,
    pub auto_report_path: PathBuf,
    pub optimized_source_path: PathBuf,
    pub optimized_ptx_path: PathBuf,
    pub optimized_cubin_path: PathBuf,
    pub optimized_sass_path: PathBuf,
    pub optimized_ir_path: PathBuf,
    pub optimized_lifted_ir_path: PathBuf,
    pub optimized_analysis_path: PathBuf,
    pub optimized_pattern_path: PathBuf,
    pub optimized_side_by_side_path: PathBuf,
    pub optimized_parsed_instruction_count: usize,
    pub optimized_unsupported_instruction_count: usize,
    pub optimized_semantic_pattern_count: usize,
    pub optimized_evidence: DecompiledAutotuneEvidence,
    pub best_symbol: String,
    pub best_action_ops: Vec<String>,
    pub best_action_count: usize,
    pub best_score: Option<f64>,
    pub explored: usize,
    pub rejected: usize,
    pub improving_step_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneGemmOptions {
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub config: AutoOptimizeConfig,
}

impl DecompileAutotuneGemmOptions {
    pub fn sm120_default(m: usize, n: usize, k: usize) -> Self {
        Self {
            artifact_root: runtime::default_artifact_dir()
                .join("decompile-autotune")
                .join("gemm-f32-bf16-row-col-row"),
            compile_arch: "sm_120".to_string(),
            m,
            n,
            k,
            config: AutoOptimizeConfig {
                beam_width: 4,
                max_steps: 2,
                require_launchable: false,
                min_score_improvement: 0.0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneGemmReport {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub source_symbol: String,
    pub source_path: PathBuf,
    pub ptx_path: PathBuf,
    pub cubin_path: PathBuf,
    pub sass_path: PathBuf,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
    pub semantic_pattern_count: usize,
    pub evidence: DecompiledAutotuneEvidence,
    pub operation_name: String,
    pub auto_report_path: PathBuf,
    pub optimized_source_path: PathBuf,
    pub optimized_ptx_path: PathBuf,
    pub optimized_cubin_path: PathBuf,
    pub optimized_sass_path: PathBuf,
    pub optimized_ir_path: PathBuf,
    pub optimized_lifted_ir_path: PathBuf,
    pub optimized_analysis_path: PathBuf,
    pub optimized_pattern_path: PathBuf,
    pub optimized_side_by_side_path: PathBuf,
    pub optimized_parsed_instruction_count: usize,
    pub optimized_unsupported_instruction_count: usize,
    pub optimized_semantic_pattern_count: usize,
    pub optimized_evidence: DecompiledAutotuneEvidence,
    pub best_symbol: String,
    pub best_action_ops: Vec<String>,
    pub best_action_count: usize,
    pub best_score: Option<f64>,
    pub explored: usize,
    pub rejected: usize,
    pub improving_step_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneSassOptions {
    pub sass_path: PathBuf,
    pub source_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub function_symbol: Option<String>,
    pub shape: DecompiledAutotuneShape,
    pub config: AutoOptimizeConfig,
}

impl DecompileAutotuneSassOptions {
    pub fn sm120_default(sass_path: PathBuf, shape: DecompiledAutotuneShape) -> Self {
        Self {
            sass_path,
            source_path: None,
            output_dir: None,
            artifact_root: runtime::default_artifact_dir()
                .join("decompile-autotune")
                .join("input-sass"),
            compile_arch: "sm_120".to_string(),
            function_symbol: None,
            shape,
            config: AutoOptimizeConfig {
                beam_width: 4,
                max_steps: 2,
                require_launchable: false,
                min_score_improvement: 0.0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneSassReport {
    pub sass_path: PathBuf,
    pub source_path: Option<PathBuf>,
    pub function_symbol: String,
    pub output_dir: PathBuf,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
    pub semantic_pattern_count: usize,
    pub evidence: DecompiledAutotuneEvidence,
    pub operation_name: String,
    pub auto_report_path: PathBuf,
    pub optimized_source_path: PathBuf,
    pub optimized_ptx_path: PathBuf,
    pub optimized_cubin_path: PathBuf,
    pub optimized_sass_path: PathBuf,
    pub optimized_ir_path: PathBuf,
    pub optimized_lifted_ir_path: PathBuf,
    pub optimized_analysis_path: PathBuf,
    pub optimized_pattern_path: PathBuf,
    pub optimized_side_by_side_path: PathBuf,
    pub optimized_parsed_instruction_count: usize,
    pub optimized_unsupported_instruction_count: usize,
    pub optimized_semantic_pattern_count: usize,
    pub optimized_evidence: DecompiledAutotuneEvidence,
    pub best_symbol: String,
    pub best_action_ops: Vec<String>,
    pub best_action_count: usize,
    pub best_score: Option<f64>,
    pub explored: usize,
    pub rejected: usize,
    pub improving_step_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassFileDecompileOptions {
    pub sass_path: PathBuf,
    pub source_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassFileDecompileReport {
    pub sass_path: PathBuf,
    pub source_path: Option<PathBuf>,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
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
}

pub fn run_decompile_fixtures(
    options: &DecompileFixtureOptions,
) -> Result<Vec<DecompileFixtureReport>, Box<dyn Error>> {
    let mut reports = Vec::new();
    let fixtures = simple_kernel_fixtures();
    for requested in &options.fixtures {
        let fixture = fixtures
            .iter()
            .find(|fixture| fixture.kind == *requested)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("unknown decompile fixture {}", requested.name()),
                )
            })?;
        reports.push(run_decompile_fixture(options, fixture)?);
    }
    Ok(reports)
}

pub fn run_decompile_fixture_coverage(
    options: &DecompileFixtureCoverageOptions,
) -> Result<DecompileFixtureCoverageReport, Box<dyn Error>> {
    let fixture_reports = run_decompile_fixtures(&options.fixture_options)?;
    let coverage_report = run_sass_coverage_scan(&SassCoverageOptions {
        root: options.fixture_options.artifact_root.clone(),
        output_dir: options.coverage_output_dir.clone(),
    })?;
    Ok(DecompileFixtureCoverageReport {
        fixture_reports,
        coverage_report,
    })
}

pub fn run_decompile_ptx_probes(
    options: &DecompilePtxProbeOptions,
) -> Result<Vec<DecompilePtxProbeReport>, Box<dyn Error>> {
    let mut reports = Vec::new();
    let probes = ptx_decompile_probes();
    for requested in &options.probes {
        let probe = probes
            .iter()
            .find(|probe| probe.kind == *requested)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("unknown PTX decompile probe {}", requested.name()),
                )
            })?;
        reports.push(run_decompile_ptx_probe(options, probe)?);
    }
    Ok(reports)
}

pub fn run_decompile_autotune_matvec(
    options: &DecompileAutotuneMatvecOptions,
) -> Result<DecompileAutotuneMatvecReport, Box<dyn Error>> {
    if options.rows == 0 || options.cols == 0 {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "decompile autotune matvec dimensions must be nonzero",
        )));
    }

    let problem = MatvecSearchProblem::bf16_row_major(options.rows, options.cols);
    let naive_candidate = problem.generated_naive_candidate();
    let naive_source = MatvecRustCudaGenerator.source_for(&naive_candidate)?;
    let artifact_root = absolute_path(&options.artifact_root)?;
    let run_dir = artifact_root
        .join(format!("{}x{}", options.rows, options.cols))
        .join("naive");
    let crate_dir = run_dir.join("standalone-crate");
    let source_path = crate_dir.join("src").join("main.rs");
    let cargo_toml_path = crate_dir.join("Cargo.toml");
    let package_stem = format!(
        "nn_rust_decompile_autotune_matvec_{}x{}_naive",
        options.rows, options.cols
    );
    fs::create_dir_all(
        source_path
            .parent()
            .expect("source path should have a parent"),
    )?;
    fs::write(&cargo_toml_path, standalone_cargo_toml(&package_stem))?;
    fs::write(&source_path, standalone_main_source(&naive_source.source))?;

    let ptx_output_dir = run_dir.join("ptx");
    let target_dir = artifact_root.join("standalone-target");
    let compiled = compile_standalone_kernel_crate(
        &crate_dir,
        &ptx_output_dir,
        &package_stem,
        Some(&options.compile_arch),
        Some(&target_dir),
    )?;

    let cubin_path = run_dir.join(format!(
        "{}.{}.cubin",
        naive_source.symbol, options.compile_arch
    ));
    run_checked(
        "ptxas",
        &[
            format!("-arch={}", options.compile_arch),
            "-o".to_string(),
            cubin_path.display().to_string(),
            compiled.ptx_path.display().to_string(),
        ],
        &run_dir,
    )?;

    let sass_path = run_dir.join(format!(
        "{}.{}.nvdisasm.sass",
        naive_source.symbol, options.compile_arch
    ));
    let sass = run_capture("nvdisasm", &[cubin_path.display().to_string()], &run_dir)?;
    fs::write(&sass_path, sass.as_bytes())?;

    let parsed = parse_nvidia_sass(&sass)?;
    let project_ir = lift_sass_module(&parsed);
    let analysis = analyze_sass_ir(&project_ir);
    let lifted = lift_sass_value_ir(&project_ir, &analysis);
    let patterns = recover_sass_patterns(&project_ir);
    let ir_path = run_dir.join("lifted.ir.txt");
    fs::write(&ir_path, project_ir.to_text().as_bytes())?;
    let lifted_ir_path = run_dir.join("lifted-value-ir.txt");
    fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;
    let analysis_path = run_dir.join("analysis.txt");
    fs::write(&analysis_path, analysis.to_text().as_bytes())?;
    let pattern_path = run_dir.join("patterns.txt");
    fs::write(&pattern_path, patterns.to_text().as_bytes())?;
    let side_by_side = render_sass_file_side_by_side(
        &sass_path,
        Some((&source_path, &naive_source.source)),
        &sass,
        &project_ir,
    );
    let side_by_side_path = run_dir.join("source-sass-ir.txt");
    fs::write(&side_by_side_path, side_by_side.as_bytes())?;

    let function = project_ir
        .functions
        .iter()
        .find(|function| function.name.as_str() == naive_source.symbol)
        .or_else(|| project_ir.functions.first())
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "compiled naive matvec SASS did not contain any functions",
            )
        })?;
    let routed = decompiled_autotune_operation_with_module(
        &project_ir,
        function,
        DecompiledAutotuneShape::MatvecBf16RowMajor {
            rows: options.rows,
            cols: options.cols,
        },
    )
    .map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "compiled naive matvec SASS did not provide supported autotune evidence: {error:?}"
            ),
        )
    })?;

    let optimization = auto_optimize_inference_kernel(&routed.operation, options.config)?;
    let best = optimization.best_candidate().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            "decompiled matvec autotune did not produce any candidates",
        )
    })?;
    let generated_store = KernelArtifactStore::new(artifact_root.join("generated"));
    let auto_report = optimization.auto_optimization_report(options.config);
    let emitted_report = generated_store.emit_auto_search_report(&auto_report)?;
    let emitted_optimized =
        generated_store.emit_standalone_crate(best, &InferenceKernelRustCudaGenerator)?;
    let optimized_output_dir = generated_store.paths_for(best).directory.join("ptx");
    let compiled_optimized = compile_standalone_kernel_crate(
        &emitted_optimized.paths.crate_dir,
        &optimized_output_dir,
        &emitted_optimized.package_name,
        Some(&options.compile_arch),
        Some(&generated_store.standalone_target_root()),
    )?;
    let optimized_sass_dir = generated_store.paths_for(best).directory.join("sass");
    fs::create_dir_all(&optimized_sass_dir)?;
    let optimized_cubin_path = optimized_sass_dir.join(format!(
        "{}.{}.cubin",
        emitted_optimized.symbol, options.compile_arch
    ));
    run_checked(
        "ptxas",
        &[
            format!("-arch={}", options.compile_arch),
            "-o".to_string(),
            optimized_cubin_path.display().to_string(),
            compiled_optimized.ptx_path.display().to_string(),
        ],
        &optimized_sass_dir,
    )?;
    let optimized_sass_path = optimized_sass_dir.join(format!(
        "{}.{}.nvdisasm.sass",
        emitted_optimized.symbol, options.compile_arch
    ));
    let optimized_sass = run_capture(
        "nvdisasm",
        &[optimized_cubin_path.display().to_string()],
        &optimized_sass_dir,
    )?;
    fs::write(&optimized_sass_path, optimized_sass.as_bytes())?;
    let optimized_parsed = parse_nvidia_sass(&optimized_sass)?;
    let optimized_ir = lift_sass_module(&optimized_parsed);
    let optimized_analysis = analyze_sass_ir(&optimized_ir);
    let optimized_lifted = lift_sass_value_ir(&optimized_ir, &optimized_analysis);
    let optimized_patterns = recover_sass_patterns(&optimized_ir);
    let optimized_ir_path = optimized_sass_dir.join("lifted.ir.txt");
    fs::write(&optimized_ir_path, optimized_ir.to_text().as_bytes())?;
    let optimized_lifted_ir_path = optimized_sass_dir.join("lifted-value-ir.txt");
    fs::write(
        &optimized_lifted_ir_path,
        optimized_lifted.to_text().as_bytes(),
    )?;
    let optimized_analysis_path = optimized_sass_dir.join("analysis.txt");
    fs::write(
        &optimized_analysis_path,
        optimized_analysis.to_text().as_bytes(),
    )?;
    let optimized_pattern_path = optimized_sass_dir.join("patterns.txt");
    fs::write(
        &optimized_pattern_path,
        optimized_patterns.to_text().as_bytes(),
    )?;
    let optimized_source_text = fs::read_to_string(&emitted_optimized.paths.source_path)?;
    let optimized_side_by_side = render_sass_file_side_by_side(
        &optimized_sass_path,
        Some((
            emitted_optimized.paths.source_path.as_path(),
            optimized_source_text.as_str(),
        )),
        &optimized_sass,
        &optimized_ir,
    );
    let optimized_side_by_side_path = optimized_sass_dir.join("source-sass-ir.txt");
    fs::write(
        &optimized_side_by_side_path,
        optimized_side_by_side.as_bytes(),
    )?;
    let optimized_function = optimized_ir
        .functions
        .iter()
        .find(|function| function.name.as_str() == emitted_optimized.symbol)
        .or_else(|| optimized_ir.functions.first())
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "compiled optimized matvec SASS did not contain any functions",
            )
        })?;
    let optimized_routed = decompiled_autotune_operation_with_module(
        &optimized_ir,
        optimized_function,
        DecompiledAutotuneShape::MatvecBf16RowMajor {
            rows: options.rows,
            cols: options.cols,
        },
    )
    .map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "compiled optimized matvec SASS did not provide supported autotune evidence: {error:?}"
            ),
        )
    })?;
    let best_action_ops = best
        .action_trace
        .iter()
        .map(|action| action.op.label().to_string())
        .collect::<Vec<_>>();
    let improving_step_count = optimization
        .result
        .steps
        .iter()
        .filter(|step| {
            step.improvement
                .is_some_and(|improvement| improvement > 0.0)
        })
        .count();

    Ok(DecompileAutotuneMatvecReport {
        rows: options.rows,
        cols: options.cols,
        naive_symbol: naive_source.symbol,
        source_path,
        ptx_path: compiled.ptx_path,
        cubin_path,
        sass_path,
        ir_path,
        lifted_ir_path,
        analysis_path,
        pattern_path,
        side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        unsupported_instruction_count: project_ir.unsupported_instruction_count(),
        semantic_pattern_count: patterns.pattern_count(),
        evidence: routed.evidence,
        operation_name: routed.operation.name,
        auto_report_path: emitted_report.report_path,
        optimized_source_path: emitted_optimized.paths.source_path,
        optimized_ptx_path: compiled_optimized.ptx_path,
        optimized_cubin_path,
        optimized_sass_path,
        optimized_ir_path,
        optimized_lifted_ir_path,
        optimized_analysis_path,
        optimized_pattern_path,
        optimized_side_by_side_path,
        optimized_parsed_instruction_count: optimized_parsed.instruction_count(),
        optimized_unsupported_instruction_count: optimized_ir.unsupported_instruction_count(),
        optimized_semantic_pattern_count: optimized_patterns.pattern_count(),
        optimized_evidence: optimized_routed.evidence,
        best_symbol: emitted_optimized.symbol,
        best_action_count: best_action_ops.len(),
        best_action_ops,
        best_score: best.score.map(|score| score.value),
        explored: optimization.result.explored,
        rejected: optimization.result.rejected,
        improving_step_count,
    })
}

pub fn run_decompile_autotune_gemm(
    options: &DecompileAutotuneGemmOptions,
) -> Result<DecompileAutotuneGemmReport, Box<dyn Error>> {
    if options.m == 0 || options.n == 0 || options.k == 0 {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "decompile autotune GEMM dimensions must be nonzero",
        )));
    }

    let problem = GemmSearchProblem::f32_bf16_row_col_row(options.m, options.n, options.k);
    let source_tile = GemmTileShape::new(16, 16, 8);
    let source_candidate = problem.candidate_for_tile(source_tile);
    let source_kernel = GemmRustCudaGenerator.source_for(&source_candidate)?;
    let artifact_root = absolute_path(&options.artifact_root)?;
    let run_dir = artifact_root
        .join(format!("{}x{}x{}", options.m, options.n, options.k))
        .join(format!(
            "source-tile-{}x{}x{}",
            source_tile.m, source_tile.n, source_tile.k
        ));
    let crate_dir = run_dir.join("standalone-crate");
    let source_path = crate_dir.join("src").join("main.rs");
    let cargo_toml_path = crate_dir.join("Cargo.toml");
    let package_stem = format!(
        "nn_rust_decompile_autotune_gemm_{}x{}x{}_source",
        options.m, options.n, options.k
    );
    fs::create_dir_all(
        source_path
            .parent()
            .expect("source path should have a parent"),
    )?;
    fs::write(&cargo_toml_path, standalone_cargo_toml(&package_stem))?;
    fs::write(&source_path, standalone_main_source(&source_kernel.source))?;

    let ptx_output_dir = run_dir.join("ptx");
    let target_dir = artifact_root.join("standalone-target");
    let compiled = compile_standalone_kernel_crate(
        &crate_dir,
        &ptx_output_dir,
        &package_stem,
        Some(&options.compile_arch),
        Some(&target_dir),
    )?;

    let cubin_path = run_dir.join(format!(
        "{}.{}.cubin",
        source_kernel.symbol, options.compile_arch
    ));
    run_checked(
        "ptxas",
        &[
            format!("-arch={}", options.compile_arch),
            "-o".to_string(),
            cubin_path.display().to_string(),
            compiled.ptx_path.display().to_string(),
        ],
        &run_dir,
    )?;

    let sass_path = run_dir.join(format!(
        "{}.{}.nvdisasm.sass",
        source_kernel.symbol, options.compile_arch
    ));
    let sass = run_capture("nvdisasm", &[cubin_path.display().to_string()], &run_dir)?;
    fs::write(&sass_path, sass.as_bytes())?;

    let parsed = parse_nvidia_sass(&sass)?;
    let project_ir = lift_sass_module(&parsed);
    let analysis = analyze_sass_ir(&project_ir);
    let lifted = lift_sass_value_ir(&project_ir, &analysis);
    let patterns = recover_sass_patterns(&project_ir);
    let ir_path = run_dir.join("lifted.ir.txt");
    fs::write(&ir_path, project_ir.to_text().as_bytes())?;
    let lifted_ir_path = run_dir.join("lifted-value-ir.txt");
    fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;
    let analysis_path = run_dir.join("analysis.txt");
    fs::write(&analysis_path, analysis.to_text().as_bytes())?;
    let pattern_path = run_dir.join("patterns.txt");
    fs::write(&pattern_path, patterns.to_text().as_bytes())?;
    let side_by_side = render_sass_file_side_by_side(
        &sass_path,
        Some((&source_path, &source_kernel.source)),
        &sass,
        &project_ir,
    );
    let side_by_side_path = run_dir.join("source-sass-ir.txt");
    fs::write(&side_by_side_path, side_by_side.as_bytes())?;

    let function = project_ir
        .functions
        .iter()
        .find(|function| function.name.as_str() == source_kernel.symbol)
        .or_else(|| project_ir.functions.first())
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "compiled source GEMM SASS did not contain any functions",
            )
        })?;
    let routed = decompiled_autotune_operation_with_module(
        &project_ir,
        function,
        DecompiledAutotuneShape::GemmF32Bf16RowColRow {
            m: options.m,
            n: options.n,
            k: options.k,
        },
    )
    .map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "compiled source GEMM SASS did not provide supported autotune evidence: {error:?}"
            ),
        )
    })?;

    let optimization = auto_optimize_inference_kernel(&routed.operation, options.config)?;
    let best = optimization.best_candidate().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            "decompiled GEMM autotune did not produce any candidates",
        )
    })?;
    let generated_store = KernelArtifactStore::new(artifact_root.join("generated"));
    let auto_report = optimization.auto_optimization_report(options.config);
    let emitted_report = generated_store.emit_auto_search_report(&auto_report)?;
    let emitted_optimized =
        generated_store.emit_standalone_crate(best, &InferenceKernelRustCudaGenerator)?;
    let optimized_output_dir = generated_store.paths_for(best).directory.join("ptx");
    let compiled_optimized = compile_standalone_kernel_crate(
        &emitted_optimized.paths.crate_dir,
        &optimized_output_dir,
        &emitted_optimized.package_name,
        Some(&options.compile_arch),
        Some(&generated_store.standalone_target_root()),
    )?;
    let optimized_sass_dir = generated_store.paths_for(best).directory.join("sass");
    fs::create_dir_all(&optimized_sass_dir)?;
    let optimized_cubin_path = optimized_sass_dir.join(format!(
        "{}.{}.cubin",
        emitted_optimized.symbol, options.compile_arch
    ));
    run_checked(
        "ptxas",
        &[
            format!("-arch={}", options.compile_arch),
            "-o".to_string(),
            optimized_cubin_path.display().to_string(),
            compiled_optimized.ptx_path.display().to_string(),
        ],
        &optimized_sass_dir,
    )?;
    let optimized_sass_path = optimized_sass_dir.join(format!(
        "{}.{}.nvdisasm.sass",
        emitted_optimized.symbol, options.compile_arch
    ));
    let optimized_sass = run_capture(
        "nvdisasm",
        &[optimized_cubin_path.display().to_string()],
        &optimized_sass_dir,
    )?;
    fs::write(&optimized_sass_path, optimized_sass.as_bytes())?;
    let optimized_parsed = parse_nvidia_sass(&optimized_sass)?;
    let optimized_ir = lift_sass_module(&optimized_parsed);
    let optimized_analysis = analyze_sass_ir(&optimized_ir);
    let optimized_lifted = lift_sass_value_ir(&optimized_ir, &optimized_analysis);
    let optimized_patterns = recover_sass_patterns(&optimized_ir);
    let optimized_ir_path = optimized_sass_dir.join("lifted.ir.txt");
    fs::write(&optimized_ir_path, optimized_ir.to_text().as_bytes())?;
    let optimized_lifted_ir_path = optimized_sass_dir.join("lifted-value-ir.txt");
    fs::write(
        &optimized_lifted_ir_path,
        optimized_lifted.to_text().as_bytes(),
    )?;
    let optimized_analysis_path = optimized_sass_dir.join("analysis.txt");
    fs::write(
        &optimized_analysis_path,
        optimized_analysis.to_text().as_bytes(),
    )?;
    let optimized_pattern_path = optimized_sass_dir.join("patterns.txt");
    fs::write(
        &optimized_pattern_path,
        optimized_patterns.to_text().as_bytes(),
    )?;
    let optimized_source_text = fs::read_to_string(&emitted_optimized.paths.source_path)?;
    let optimized_side_by_side = render_sass_file_side_by_side(
        &optimized_sass_path,
        Some((
            emitted_optimized.paths.source_path.as_path(),
            optimized_source_text.as_str(),
        )),
        &optimized_sass,
        &optimized_ir,
    );
    let optimized_side_by_side_path = optimized_sass_dir.join("source-sass-ir.txt");
    fs::write(
        &optimized_side_by_side_path,
        optimized_side_by_side.as_bytes(),
    )?;
    let optimized_function = optimized_ir
        .functions
        .iter()
        .find(|function| function.name.as_str() == emitted_optimized.symbol)
        .or_else(|| optimized_ir.functions.first())
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "compiled optimized GEMM SASS did not contain any functions",
            )
        })?;
    let optimized_routed = decompiled_autotune_operation_with_module(
        &optimized_ir,
        optimized_function,
        DecompiledAutotuneShape::GemmF32Bf16RowColRow {
            m: options.m,
            n: options.n,
            k: options.k,
        },
    )
    .map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "compiled optimized GEMM SASS did not provide supported autotune evidence: {error:?}"
            ),
        )
    })?;
    let best_action_ops = best
        .action_trace
        .iter()
        .map(|action| action.op.label().to_string())
        .collect::<Vec<_>>();
    let improving_step_count = optimization
        .result
        .steps
        .iter()
        .filter(|step| {
            step.improvement
                .is_some_and(|improvement| improvement > 0.0)
        })
        .count();

    Ok(DecompileAutotuneGemmReport {
        m: options.m,
        n: options.n,
        k: options.k,
        source_symbol: source_kernel.symbol,
        source_path,
        ptx_path: compiled.ptx_path,
        cubin_path,
        sass_path,
        ir_path,
        lifted_ir_path,
        analysis_path,
        pattern_path,
        side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        unsupported_instruction_count: project_ir.unsupported_instruction_count(),
        semantic_pattern_count: patterns.pattern_count(),
        evidence: routed.evidence,
        operation_name: routed.operation.name,
        auto_report_path: emitted_report.report_path,
        optimized_source_path: emitted_optimized.paths.source_path,
        optimized_ptx_path: compiled_optimized.ptx_path,
        optimized_cubin_path,
        optimized_sass_path,
        optimized_ir_path,
        optimized_lifted_ir_path,
        optimized_analysis_path,
        optimized_pattern_path,
        optimized_side_by_side_path,
        optimized_parsed_instruction_count: optimized_parsed.instruction_count(),
        optimized_unsupported_instruction_count: optimized_ir.unsupported_instruction_count(),
        optimized_semantic_pattern_count: optimized_patterns.pattern_count(),
        optimized_evidence: optimized_routed.evidence,
        best_symbol: emitted_optimized.symbol,
        best_action_count: best_action_ops.len(),
        best_action_ops,
        best_score: best.score.map(|score| score.value),
        explored: optimization.result.explored,
        rejected: optimization.result.rejected,
        improving_step_count,
    })
}

pub fn run_decompile_autotune_sass(
    options: &DecompileAutotuneSassOptions,
) -> Result<DecompileAutotuneSassReport, Box<dyn Error>> {
    validate_decompiled_shape(options.shape)?;

    let sass_path = absolute_path(&options.sass_path)?;
    let source_path = options
        .source_path
        .as_ref()
        .map(|path| absolute_path(path))
        .transpose()?;
    let artifact_root = absolute_path(&options.artifact_root)?;
    let sass = fs::read_to_string(&sass_path)?;
    let source = match &source_path {
        Some(path) => Some((path.clone(), fs::read_to_string(path)?)),
        None => None,
    };
    let parsed = parse_nvidia_sass(&sass)?;
    let project_ir = lift_sass_module(&parsed);
    let analysis = analyze_sass_ir(&project_ir);
    let lifted = lift_sass_value_ir(&project_ir, &analysis);
    let patterns = recover_sass_patterns(&project_ir);
    let output_dir = match &options.output_dir {
        Some(path) => absolute_path(path)?,
        None => artifact_root.join("input").join(
            sass_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("sass"),
        ),
    };
    fs::create_dir_all(&output_dir)?;
    let ir_path = output_dir.join("lifted.ir.txt");
    fs::write(&ir_path, project_ir.to_text().as_bytes())?;
    let lifted_ir_path = output_dir.join("lifted-value-ir.txt");
    fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;
    let analysis_path = output_dir.join("analysis.txt");
    fs::write(&analysis_path, analysis.to_text().as_bytes())?;
    let pattern_path = output_dir.join("patterns.txt");
    fs::write(&pattern_path, patterns.to_text().as_bytes())?;
    let side_by_side = render_sass_file_side_by_side(
        &sass_path,
        source
            .as_ref()
            .map(|(path, text)| (path.as_path(), text.as_str())),
        &sass,
        &project_ir,
    );
    let side_by_side_path = output_dir.join("source-sass-ir.txt");
    fs::write(&side_by_side_path, side_by_side.as_bytes())?;

    let function = match &options.function_symbol {
        Some(symbol) => project_ir
            .functions
            .iter()
            .find(|function| function.name.as_str() == symbol)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "SASS function {symbol:?} was not found in {}",
                        sass_path.display()
                    ),
                )
            })?,
        None => project_ir.functions.first().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "input SASS did not contain any functions",
            )
        })?,
    };
    let routed = decompiled_autotune_operation_with_module(&project_ir, function, options.shape)
        .map_err(|error| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!("input SASS did not provide supported autotune evidence: {error:?}"),
            )
        })?;

    let optimization = auto_optimize_inference_kernel(&routed.operation, options.config)?;
    let best = optimization.best_candidate().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            "decompiled SASS autotune did not produce any candidates",
        )
    })?;
    let generated_store = KernelArtifactStore::new(artifact_root.join("generated"));
    let auto_report = optimization.auto_optimization_report(options.config);
    let emitted_report = generated_store.emit_auto_search_report(&auto_report)?;
    let emitted_optimized =
        generated_store.emit_standalone_crate(best, &InferenceKernelRustCudaGenerator)?;
    let optimized_output_dir = generated_store.paths_for(best).directory.join("ptx");
    let compiled_optimized = compile_standalone_kernel_crate(
        &emitted_optimized.paths.crate_dir,
        &optimized_output_dir,
        &emitted_optimized.package_name,
        Some(&options.compile_arch),
        Some(&generated_store.standalone_target_root()),
    )?;
    let optimized_sass_dir = generated_store.paths_for(best).directory.join("sass");
    fs::create_dir_all(&optimized_sass_dir)?;
    let optimized_cubin_path = optimized_sass_dir.join(format!(
        "{}.{}.cubin",
        emitted_optimized.symbol, options.compile_arch
    ));
    run_checked(
        "ptxas",
        &[
            format!("-arch={}", options.compile_arch),
            "-o".to_string(),
            optimized_cubin_path.display().to_string(),
            compiled_optimized.ptx_path.display().to_string(),
        ],
        &optimized_sass_dir,
    )?;
    let optimized_sass_path = optimized_sass_dir.join(format!(
        "{}.{}.nvdisasm.sass",
        emitted_optimized.symbol, options.compile_arch
    ));
    let optimized_sass = run_capture(
        "nvdisasm",
        &[optimized_cubin_path.display().to_string()],
        &optimized_sass_dir,
    )?;
    fs::write(&optimized_sass_path, optimized_sass.as_bytes())?;
    let optimized_parsed = parse_nvidia_sass(&optimized_sass)?;
    let optimized_ir = lift_sass_module(&optimized_parsed);
    let optimized_analysis = analyze_sass_ir(&optimized_ir);
    let optimized_lifted = lift_sass_value_ir(&optimized_ir, &optimized_analysis);
    let optimized_patterns = recover_sass_patterns(&optimized_ir);
    let optimized_ir_path = optimized_sass_dir.join("lifted.ir.txt");
    fs::write(&optimized_ir_path, optimized_ir.to_text().as_bytes())?;
    let optimized_lifted_ir_path = optimized_sass_dir.join("lifted-value-ir.txt");
    fs::write(
        &optimized_lifted_ir_path,
        optimized_lifted.to_text().as_bytes(),
    )?;
    let optimized_analysis_path = optimized_sass_dir.join("analysis.txt");
    fs::write(
        &optimized_analysis_path,
        optimized_analysis.to_text().as_bytes(),
    )?;
    let optimized_pattern_path = optimized_sass_dir.join("patterns.txt");
    fs::write(
        &optimized_pattern_path,
        optimized_patterns.to_text().as_bytes(),
    )?;
    let optimized_source_text = fs::read_to_string(&emitted_optimized.paths.source_path)?;
    let optimized_side_by_side = render_sass_file_side_by_side(
        &optimized_sass_path,
        Some((
            emitted_optimized.paths.source_path.as_path(),
            optimized_source_text.as_str(),
        )),
        &optimized_sass,
        &optimized_ir,
    );
    let optimized_side_by_side_path = optimized_sass_dir.join("source-sass-ir.txt");
    fs::write(
        &optimized_side_by_side_path,
        optimized_side_by_side.as_bytes(),
    )?;
    let optimized_function = optimized_ir
        .functions
        .iter()
        .find(|function| function.name.as_str() == emitted_optimized.symbol)
        .or_else(|| optimized_ir.functions.first())
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "compiled optimized SASS did not contain any functions",
            )
        })?;
    let optimized_routed = decompiled_autotune_operation_with_module(
        &optimized_ir,
        optimized_function,
        options.shape,
    )
    .map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "compiled optimized SASS did not provide supported autotune evidence: {error:?}"
            ),
        )
    })?;
    let best_action_ops = best
        .action_trace
        .iter()
        .map(|action| action.op.label().to_string())
        .collect::<Vec<_>>();
    let improving_step_count = optimization
        .result
        .steps
        .iter()
        .filter(|step| {
            step.improvement
                .is_some_and(|improvement| improvement > 0.0)
        })
        .count();

    Ok(DecompileAutotuneSassReport {
        sass_path,
        source_path,
        function_symbol: function.name.as_str().to_string(),
        output_dir,
        ir_path,
        lifted_ir_path,
        analysis_path,
        pattern_path,
        side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        unsupported_instruction_count: project_ir.unsupported_instruction_count(),
        semantic_pattern_count: patterns.pattern_count(),
        evidence: routed.evidence,
        operation_name: routed.operation.name,
        auto_report_path: emitted_report.report_path,
        optimized_source_path: emitted_optimized.paths.source_path,
        optimized_ptx_path: compiled_optimized.ptx_path,
        optimized_cubin_path,
        optimized_sass_path,
        optimized_ir_path,
        optimized_lifted_ir_path,
        optimized_analysis_path,
        optimized_pattern_path,
        optimized_side_by_side_path,
        optimized_parsed_instruction_count: optimized_parsed.instruction_count(),
        optimized_unsupported_instruction_count: optimized_ir.unsupported_instruction_count(),
        optimized_semantic_pattern_count: optimized_patterns.pattern_count(),
        optimized_evidence: optimized_routed.evidence,
        best_symbol: emitted_optimized.symbol,
        best_action_count: best_action_ops.len(),
        best_action_ops,
        best_score: best.score.map(|score| score.value),
        explored: optimization.result.explored,
        rejected: optimization.result.rejected,
        improving_step_count,
    })
}

fn validate_decompiled_shape(shape: DecompiledAutotuneShape) -> Result<(), Box<dyn Error>> {
    let invalid = match shape {
        DecompiledAutotuneShape::MatvecBf16RowMajor { rows, cols } => rows == 0 || cols == 0,
        DecompiledAutotuneShape::GemmF32Bf16RowColRow { m, n, k } => m == 0 || n == 0 || k == 0,
    };
    if invalid {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "decompile autotune SASS dimensions must be nonzero",
        )));
    }
    Ok(())
}

pub fn run_sass_file_decompile(
    options: &SassFileDecompileOptions,
) -> Result<SassFileDecompileReport, Box<dyn Error>> {
    let sass = fs::read_to_string(&options.sass_path)?;
    let source = match &options.source_path {
        Some(path) => Some((path.clone(), fs::read_to_string(path)?)),
        None => None,
    };
    let parsed = parse_nvidia_sass(&sass)?;
    let project_ir = lift_sass_module(&parsed);
    let analysis = analyze_sass_ir(&project_ir);
    let lifted = lift_sass_value_ir(&project_ir, &analysis);
    let patterns = recover_sass_patterns(&project_ir);
    let output_dir = options.output_dir.clone().unwrap_or_else(|| {
        options
            .sass_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    });
    fs::create_dir_all(&output_dir)?;
    let stem = options
        .sass_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("sass");
    let ir_path = output_dir.join(format!("{stem}.lifted.ir.txt"));
    fs::write(&ir_path, project_ir.to_text().as_bytes())?;
    let lifted_ir_path = output_dir.join(format!("{stem}.lifted-value-ir.txt"));
    fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;
    let analysis_path = output_dir.join(format!("{stem}.analysis.txt"));
    fs::write(&analysis_path, analysis.to_text().as_bytes())?;
    let pattern_path = output_dir.join(format!("{stem}.patterns.txt"));
    fs::write(&pattern_path, patterns.to_text().as_bytes())?;
    let side_by_side_path = output_dir.join(format!("{stem}.source-sass-ir.txt"));
    let side_by_side = render_sass_file_side_by_side(
        options.sass_path.as_path(),
        source
            .as_ref()
            .map(|(path, text)| (path.as_path(), text.as_str())),
        &sass,
        &project_ir,
    );
    fs::write(&side_by_side_path, side_by_side.as_bytes())?;

    Ok(SassFileDecompileReport {
        sass_path: options.sass_path.clone(),
        source_path: options.source_path.clone(),
        ir_path,
        lifted_ir_path,
        analysis_path,
        pattern_path,
        side_by_side_path,
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
    })
}

fn run_decompile_ptx_probe(
    options: &DecompilePtxProbeOptions,
    probe: &PtxDecompileProbe,
) -> Result<DecompilePtxProbeReport, Box<dyn Error>> {
    let probe_dir = options.artifact_root.join(probe.kind.name());
    fs::create_dir_all(&probe_dir)?;
    let ptx_path = probe_dir.join(format!("{}.ptx", probe.symbol));
    fs::write(&ptx_path, probe.source.as_bytes())?;

    let cubin_path = probe_dir.join(format!("{}.{}.cubin", probe.symbol, options.compile_arch));
    run_checked(
        "ptxas",
        &[
            format!("-arch={}", options.compile_arch),
            "-o".to_string(),
            cubin_path.display().to_string(),
            ptx_path.display().to_string(),
        ],
        &probe_dir,
    )?;

    let nvdisasm_sass_path = probe_dir.join(format!(
        "{}.{}.nvdisasm.sass",
        probe.symbol, options.compile_arch
    ));
    let nvdisasm_sass = run_capture("nvdisasm", &[cubin_path.display().to_string()], &probe_dir)?;
    fs::write(&nvdisasm_sass_path, nvdisasm_sass.as_bytes())?;

    let cuobjdump_sass_path = probe_dir.join(format!(
        "{}.{}.cuobjdump.sass",
        probe.symbol, options.compile_arch
    ));
    let cuobjdump_sass = run_capture(
        "cuobjdump",
        &["--dump-sass".to_string(), cubin_path.display().to_string()],
        &probe_dir,
    )?;
    fs::write(&cuobjdump_sass_path, cuobjdump_sass.as_bytes())?;

    let nvdisasm_module = parse_nvidia_sass(&nvdisasm_sass)?;
    let cuobjdump_module = parse_nvidia_sass(&cuobjdump_sass)?;
    let nvdisasm_ir = lift_sass_module(&nvdisasm_module);
    let cuobjdump_ir = lift_sass_module(&cuobjdump_module);

    Ok(DecompilePtxProbeReport {
        probe: probe.kind,
        symbol: probe.symbol.to_string(),
        probe_dir,
        ptx_path,
        cubin_path,
        nvdisasm_sass_path,
        cuobjdump_sass_path,
        parsed_instruction_count: nvdisasm_module.instruction_count()
            + cuobjdump_module.instruction_count(),
        unsupported_instruction_count: nvdisasm_ir.unsupported_instruction_count()
            + cuobjdump_ir.unsupported_instruction_count(),
    })
}

fn run_decompile_fixture(
    options: &DecompileFixtureOptions,
    fixture: &SimpleKernelFixture,
) -> Result<DecompileFixtureReport, Box<dyn Error>> {
    let fixture_dir = options.artifact_root.join(fixture.kind.name());
    let crate_dir = fixture_dir.join("standalone-crate");
    let source_path = crate_dir.join("src").join("main.rs");
    let cargo_toml_path = crate_dir.join("Cargo.toml");
    let package_stem = format!(
        "nn_rust_sass_fixture_{}",
        fixture.kind.name().replace('-', "_")
    );
    fs::create_dir_all(
        source_path
            .parent()
            .expect("source path should have a parent"),
    )?;
    fs::write(&cargo_toml_path, standalone_cargo_toml(&package_stem))?;
    fs::write(&source_path, standalone_main_source(fixture.source))?;

    let ptx_output_dir = fixture_dir.join("ptx");
    let target_dir = options.artifact_root.join("standalone-target");
    let compiled = compile_standalone_kernel_crate(
        &crate_dir,
        &ptx_output_dir,
        &package_stem,
        Some(&options.compile_arch),
        Some(&target_dir),
    )?;

    let cubin_path = fixture_dir.join(format!("{}.{}.cubin", fixture.symbol, options.compile_arch));
    run_checked(
        "ptxas",
        &[
            format!("-arch={}", options.compile_arch),
            "-o".to_string(),
            cubin_path.display().to_string(),
            compiled.ptx_path.display().to_string(),
        ],
        &fixture_dir,
    )?;

    let sass_path = fixture_dir.join(format!(
        "{}.{}.nvdisasm.sass",
        fixture.symbol, options.compile_arch
    ));
    let sass = run_capture(
        "nvdisasm",
        &[cubin_path.display().to_string()],
        &fixture_dir,
    )?;
    fs::write(&sass_path, sass.as_bytes())?;

    let parsed = parse_nvidia_sass(&sass)?;
    let project_ir = lift_sass_module(&parsed);
    let analysis = analyze_sass_ir(&project_ir);
    let lifted = lift_sass_value_ir(&project_ir, &analysis);
    let patterns = recover_sass_patterns(&project_ir);
    let ir_text = project_ir.to_text();
    let ir_path = fixture_dir.join("lifted.ir.txt");
    fs::write(&ir_path, ir_text.as_bytes())?;
    let lifted_ir_path = fixture_dir.join("lifted-value-ir.txt");
    fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;
    let analysis_path = fixture_dir.join("analysis.txt");
    fs::write(&analysis_path, analysis.to_text().as_bytes())?;
    let pattern_path = fixture_dir.join("patterns.txt");
    fs::write(&pattern_path, patterns.to_text().as_bytes())?;

    let side_by_side = render_side_by_side(fixture, &sass, &project_ir);
    let side_by_side_path = fixture_dir.join("source-sass-ir.txt");
    fs::write(&side_by_side_path, side_by_side.as_bytes())?;

    Ok(DecompileFixtureReport {
        fixture: fixture.kind,
        symbol: fixture.symbol.to_string(),
        crate_dir,
        source_path,
        ptx_path: compiled.ptx_path,
        cubin_path,
        sass_path,
        ir_path,
        lifted_ir_path,
        analysis_path,
        pattern_path,
        side_by_side_path,
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
    })
}

impl fmt::Display for SimpleKernelFixtureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
