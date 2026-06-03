use nn_rust_inference::autotune::{
    AutoOptimizeConfig, CachedInferenceKernelAutoOptimize, GemmF32Bf16MeasuredAutotuneScorer,
    InferenceKernelAutoOptimize, InferenceKernelRustCudaGenerator, KernelArtifactStore,
    KernelAutotuneMeasureOptions, KernelCandidateMetadata, KernelExpansionPolicy,
    KernelOptimizationCacheKey, MatvecBf16MeasuredAutotuneScorer,
    auto_optimize_inference_kernel_with_selection_cache_and_policy,
    auto_optimize_inference_kernel_with_selection_cache_and_policy_scorer, cached_measured_score,
    compile_standalone_kernel_crate,
};
use nn_rust_profiling::TypedOperationSpec;

use super::{
    operation::{gemm_autotune_operation, matvec_autotune_operation},
    options::{AutotuneCliOptions, GEMM_USAGE, MATVEC_USAGE},
    output::{
        print_kernel_candidate, print_kernel_expansion_policy, print_selection_cache_status,
        print_selection_cache_write,
    },
};
use crate::{AppResult, cuda_handles, invalid_input, parse_required_usize};

pub(crate) fn run_kernel_autotune_gemm(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let m = parse_required_usize(args, &mut index, "m", "kernel-autotune-gemm")?;
    let n = parse_required_usize(args, &mut index, "n", "kernel-autotune-gemm")?;
    let k = parse_required_usize(args, &mut index, "k", "kernel-autotune-gemm")?;
    if m == 0 || n == 0 || k == 0 {
        return Err(invalid_input(
            "kernel-autotune-gemm dimensions must be nonzero",
        ));
    }

    let options = AutotuneCliOptions::parse(args, &mut index, "kernel-autotune-gemm", GEMM_USAGE)?;
    let operation = gemm_autotune_operation(m, n, k);
    let config = options.config();
    let policy = options.policy();
    let store = options.store();
    let cached = run_gemm_search(&operation, m, n, k, &options, config, policy, &store)?;
    let result = &cached.optimization.result;
    let best = result
        .best
        .as_ref()
        .ok_or_else(|| invalid_input("kernel-autotune-gemm did not produce any candidates"))?;

    println!(
        "kernel_autotune_gemm m={m} n={n} k={k} beam_width={} max_depth={} min_score_improvement={} allow_generated={} measure={}",
        options.beam_width,
        options.max_depth,
        options.min_score_improvement,
        options.allow_generated,
        options.measure
    );
    print_search_result(&cached, policy);
    emit_requested_artifacts(
        &store,
        best,
        &cached.cache_key,
        &cached.optimization,
        config,
        &options,
    )
}

pub(crate) fn run_kernel_autotune_matvec(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let rows = parse_required_usize(args, &mut index, "rows", "kernel-autotune-matvec")?;
    let cols = parse_required_usize(args, &mut index, "cols", "kernel-autotune-matvec")?;
    if rows == 0 || cols == 0 {
        return Err(invalid_input(
            "kernel-autotune-matvec dimensions must be nonzero",
        ));
    }

    let options =
        AutotuneCliOptions::parse(args, &mut index, "kernel-autotune-matvec", MATVEC_USAGE)?;
    let operation = matvec_autotune_operation(rows, cols);
    let config = options.config();
    let policy = options.policy();
    let store = options.store();
    let cached = run_matvec_search(&operation, rows, cols, &options, config, policy, &store)?;
    let result = &cached.optimization.result;
    let best = result
        .best
        .as_ref()
        .ok_or_else(|| invalid_input("kernel-autotune-matvec did not produce any candidates"))?;

    println!(
        "kernel_autotune_matvec rows={rows} cols={cols} beam_width={} max_depth={} min_score_improvement={} allow_generated={} measure={}",
        options.beam_width,
        options.max_depth,
        options.min_score_improvement,
        options.allow_generated,
        options.measure
    );
    print_search_result(&cached, policy);
    emit_requested_artifacts(
        &store,
        best,
        &cached.cache_key,
        &cached.optimization,
        config,
        &options,
    )
}

fn run_gemm_search(
    operation: &TypedOperationSpec,
    m: usize,
    n: usize,
    k: usize,
    options: &AutotuneCliOptions,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
    store: &KernelArtifactStore,
) -> AppResult<CachedInferenceKernelAutoOptimize> {
    let score_namespace = options.score_namespace();
    if !options.measure {
        return Ok(
            auto_optimize_inference_kernel_with_selection_cache_and_policy(
                store,
                operation,
                config,
                policy,
                &score_namespace,
            )?,
        );
    }

    store.remove_compile_scratch()?;
    let (stream, module) = cuda_handles()?;
    let measure_options = KernelAutotuneMeasureOptions {
        repeat_count: options.measure_repeat_count,
        warmup_count: options.measure_warmup_count,
    };
    let mut bench = GemmF32Bf16MeasuredAutotuneScorer::new(
        &stream,
        &module,
        m,
        n,
        k,
        measure_options,
        store.clone(),
    )?;
    let mut first_measure_error = None;
    let mut score_cache_error = None;
    let cached = auto_optimize_inference_kernel_with_selection_cache_and_policy_scorer(
        store,
        operation,
        config,
        policy,
        &score_namespace,
        |candidate, _problem| {
            cached_measured_score(
                store,
                &score_namespace,
                candidate,
                &mut first_measure_error,
                &mut score_cache_error,
                |candidate| bench.score_candidate(candidate),
            )
        },
    )?;
    finish_measured_search(
        cached,
        score_cache_error,
        first_measure_error,
        "kernel-autotune-gemm",
    )
}

fn run_matvec_search(
    operation: &TypedOperationSpec,
    rows: usize,
    cols: usize,
    options: &AutotuneCliOptions,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
    store: &KernelArtifactStore,
) -> AppResult<CachedInferenceKernelAutoOptimize> {
    let score_namespace = options.score_namespace();
    if !options.measure {
        return Ok(
            auto_optimize_inference_kernel_with_selection_cache_and_policy(
                store,
                operation,
                config,
                policy,
                &score_namespace,
            )?,
        );
    }

    store.remove_compile_scratch()?;
    let (stream, module) = cuda_handles()?;
    let measure_options = KernelAutotuneMeasureOptions {
        repeat_count: options.measure_repeat_count,
        warmup_count: options.measure_warmup_count,
    };
    let mut bench = MatvecBf16MeasuredAutotuneScorer::new(
        &stream,
        &module,
        rows,
        cols,
        measure_options,
        store.clone(),
    )?;
    let mut first_measure_error = None;
    let mut score_cache_error = None;
    let cached = auto_optimize_inference_kernel_with_selection_cache_and_policy_scorer(
        store,
        operation,
        config,
        policy,
        &score_namespace,
        |candidate, _problem| {
            cached_measured_score(
                store,
                &score_namespace,
                candidate,
                &mut first_measure_error,
                &mut score_cache_error,
                |candidate| bench.score_candidate(candidate),
            )
        },
    )?;
    finish_measured_search(
        cached,
        score_cache_error,
        first_measure_error,
        "kernel-autotune-matvec",
    )
}

fn finish_measured_search(
    cached: CachedInferenceKernelAutoOptimize,
    score_cache_error: Option<String>,
    first_measure_error: Option<String>,
    command: &str,
) -> AppResult<CachedInferenceKernelAutoOptimize> {
    if let Some(error) = score_cache_error {
        return Err(invalid_input(format!(
            "{command} score cache failed: {error}"
        )));
    }
    if cached.result().best.is_none()
        && let Some(error) = first_measure_error
    {
        return Err(invalid_input(format!(
            "{command} measured search did not produce a candidate; first measurement error: {error}"
        )));
    }
    Ok(cached)
}

fn print_search_result(cached: &CachedInferenceKernelAutoOptimize, policy: KernelExpansionPolicy) {
    let result = &cached.optimization.result;
    print_kernel_expansion_policy(policy);
    println!(
        "search explored={} rejected={} duplicates={} beam_len={} steps={} exit={}",
        result.explored,
        result.rejected,
        result.duplicates,
        result.beam.len(),
        result.steps.len(),
        result.exit_reason.label()
    );
    print_selection_cache_status(&cached.cache_key, &cached.cache_status);
    if let Some(emitted) = &cached.cache_write {
        print_selection_cache_write(&cached.cache_key, emitted);
    }
    for (rank, candidate) in result.beam.iter().enumerate() {
        print_kernel_candidate(rank, candidate);
    }
}

fn emit_requested_artifacts(
    store: &KernelArtifactStore,
    best: &KernelCandidateMetadata,
    selection_cache_key: &KernelOptimizationCacheKey,
    optimization: &InferenceKernelAutoOptimize,
    config: AutoOptimizeConfig,
    options: &AutotuneCliOptions,
) -> AppResult<()> {
    if !options.emit && !options.emit_crate {
        return Ok(());
    }
    if options.emit {
        let emitted = store.emit_metadata(best)?;
        println!(
            "emitted_metadata rank=0 artifact_key={} manifest_path={} manifest_bytes={}",
            emitted.artifact_key.hex(),
            emitted.paths.manifest_path.display(),
            emitted.manifest_bytes
        );
        let emitted_selection = store.emit_selection_for_candidate(best)?;
        println!(
            "emitted_selection rank=0 artifact_key={} selection_path={} selection_bytes={}",
            emitted_selection.artifact_key,
            emitted_selection.selection_path.display(),
            emitted_selection.selection_bytes
        );
        let emitted_cached_selection =
            store.emit_selection_cache_for_candidate(selection_cache_key, best)?;
        println!(
            "emitted_selection_cache cache_key={} artifact_key={} selection_path={} selection_bytes={}",
            selection_cache_key.hex(),
            emitted_cached_selection.artifact_key,
            emitted_cached_selection.selection_path.display(),
            emitted_cached_selection.selection_bytes
        );
        let report = optimization.auto_optimization_report(config);
        let emitted_report = store.emit_auto_search_report(&report)?;
        println!(
            "emitted_auto_search_report report_key={} report_path={} report_bytes={}",
            emitted_report.report_key.hex(),
            emitted_report.report_path.display(),
            emitted_report.report_bytes
        );
    }
    if options.emit_crate {
        let emitted_crate = store.emit_standalone_crate(best, &InferenceKernelRustCudaGenerator)?;
        println!(
            "emitted_crate rank=0 artifact_key={} package={} symbol={} crate_dir={} cargo_toml={} source_path={} cargo_toml_bytes={} source_bytes={}",
            emitted_crate.artifact_key.hex(),
            emitted_crate.package_name,
            emitted_crate.symbol,
            emitted_crate.paths.crate_dir.display(),
            emitted_crate.paths.cargo_toml_path.display(),
            emitted_crate.paths.source_path.display(),
            emitted_crate.cargo_toml_bytes,
            emitted_crate.source_bytes
        );
        if options.compile {
            let output_dir = store.paths_for(best).directory;
            let compiled = compile_standalone_kernel_crate(
                &emitted_crate.paths.crate_dir,
                &output_dir,
                &emitted_crate.package_name,
                options.compile_arch.as_deref(),
                None,
            )?;
            println!(
                "compiled_crate crate_dir={} output_dir={} ptx_path={} stdout_bytes={} stderr_bytes={}",
                compiled.crate_dir.display(),
                compiled.output_dir.display(),
                compiled.ptx_path.display(),
                compiled.stdout_bytes,
                compiled.stderr_bytes
            );
        }
    }
    Ok(())
}
