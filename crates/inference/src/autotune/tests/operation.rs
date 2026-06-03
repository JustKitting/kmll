use super::*;

fn matvec_operation(rows: usize, cols: usize) -> TypedOperationSpec {
    TypedOperationSpec::new(
        "test-matvec",
        OperationKind::Matvec,
        OperationRoute::CudaKernel,
    )
    .with_input(
        TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [cols]).with_layout("contiguous"),
    )
    .with_input(
        TensorTypeSpec::new(NumericKind::Bf16, NumericKind::F32, [rows, cols])
            .with_layout("row-major"),
    )
    .with_output(
        TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [rows]).with_layout("contiguous"),
    )
}

fn gemm_operation(m: usize, n: usize, k: usize) -> TypedOperationSpec {
    TypedOperationSpec::new("test-gemm", OperationKind::Gemm, OperationRoute::CudaKernel)
        .with_input(
            TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [m, k])
                .with_layout("row-major"),
        )
        .with_input(
            TensorTypeSpec::new(NumericKind::Bf16, NumericKind::F32, [k, n])
                .with_layout("column-major"),
        )
        .with_output(
            TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [m, n])
                .with_layout("row-major"),
        )
}

#[test]
fn operation_problem_routes_matvec_search_and_source_generation() {
    let problem = InferenceKernelOptimizationProblem::from_operation(&matvec_operation(128, 256))
        .expect("supported matvec operation should route to an autotune problem");
    let InferenceKernelOptimizationProblem::MatvecBf16RowMajor(matvec) = problem else {
        panic!("matvec operation should route to matvec search");
    };
    assert_eq!(matvec.rows, 128);
    assert_eq!(matvec.cols, 256);

    let result = beam_search_metadata(
        &problem,
        BeamSearchConfig {
            beam_width: 4,
            max_depth: 1,
            require_launchable: false,
        },
    );
    assert_eq!(result.explored, 36);
    let seed = problem.seed();
    let generated = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("generic matvec problem should replay split action");
    let source = InferenceKernelRustCudaGenerator
        .source_for(&generated)
        .expect("generic generator should render matvec source");

    assert_eq!(problem.family(), "matvec-bf16-row-major");
    assert_eq!(generated.family, "matvec-bf16-row-major");
    assert_eq!(source.symbol, "matvec_bf16_rows8");
    assert!(source.source.contains("pub fn matvec_bf16_rows8("));
}

#[test]
fn operation_auto_optimize_renders_best_matvec_source() {
    let operation = matvec_operation(128, 256);
    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };

    let generated = generate_inference_kernel_source(&operation, config)
        .expect("supported matvec operation should optimize and render source");
    let report = generated.optimization.auto_optimization_report(config);

    assert_eq!(
        generated.optimization.problem.family(),
        "matvec-bf16-row-major"
    );
    assert_eq!(generated.optimization.operation.name, "test-matvec");
    assert_eq!(
        generated
            .optimization
            .best_candidate()
            .map(|candidate| candidate.artifact_key()),
        Some(generated.candidate.artifact_key())
    );
    assert_eq!(generated.source.symbol, generated.candidate.launch.kernel);
    assert!(generated.source.source.contains("#[kernel]"));
    assert!(generated.source.source.contains("pub fn matvec_bf16_"));
    assert_eq!(report.family, "matvec-bf16-row-major");
    assert_eq!(report.config.beam_width, 4);
    assert!(report.best.is_some());
    assert!(report.action_space.is_some());
}

#[test]
fn operation_problem_routes_gemm_search_and_source_generation() {
    let problem =
        InferenceKernelOptimizationProblem::from_operation(&gemm_operation(128, 128, 256))
            .expect("supported GEMM operation should route to an autotune problem");
    let InferenceKernelOptimizationProblem::GemmF32Bf16RowColRow(gemm) = problem else {
        panic!("GEMM operation should route to GEMM search");
    };
    assert_eq!(gemm.m, 128);
    assert_eq!(gemm.n, 128);
    assert_eq!(gemm.k, 256);

    let result = beam_search_metadata(
        &problem,
        BeamSearchConfig {
            beam_width: 4,
            max_depth: 1,
            require_launchable: false,
        },
    );
    assert_eq!(result.explored, 26);
    let seed = problem.seed();
    let generated = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::tile_gemm(
                16,
                32,
                16,
                KernelActionMaterialization::DeferredGenerated,
            ),
        )
        .expect("generic GEMM problem should replay tile action");
    let source = InferenceKernelRustCudaGenerator
        .source_for(&generated)
        .expect("generic generator should render GEMM source");

    assert_eq!(problem.family(), "gemm-f32-bf16-row-col-row");
    assert_eq!(generated.family, "gemm-f32-bf16-row-col-row");
    assert_eq!(source.symbol, "gemm_f32_bf16_tile_16x32x16");
    assert!(
        source
            .source
            .contains("pub fn gemm_f32_bf16_tile_16x32x16(")
    );
}

#[test]
fn operation_auto_optimize_with_scorer_can_drive_gemm_generation() {
    let operation = gemm_operation(128, 128, 256);
    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };
    let optimization =
        auto_optimize_inference_kernel_with_scorer(&operation, config, |candidate| {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if plan.reduce_unroll == 4 {
                SearchScore::measured(f64::from(plan.tile.m + plan.tile.n + plan.tile.k))
            } else {
                SearchScore::measured(1000.0 + f64::from(plan.tile.m + plan.tile.n + plan.tile.k))
            }
        })
        .expect("supported GEMM operation should optimize through generic scorer");
    let source = optimization
        .render_best_source()
        .expect("optimized GEMM candidate should render source");
    let best = optimization
        .best_candidate()
        .expect("generic GEMM optimizer should select a best candidate");
    let best_plan = schedule_gemm_plan(&best.schedule).expect("best GEMM should have plan");

    assert_eq!(optimization.problem.family(), "gemm-f32-bf16-row-col-row");
    assert_eq!(best_plan.reduce_unroll, 4);
    assert!(source.symbol.contains("_u4"));
    assert!(
        source
            .source
            .contains(&format!("pub fn {}(", source.symbol))
    );
}

fn score_gemm_threads(candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
    let threads = candidate.resources?.threads_per_block;
    if threads > 128 {
        SearchScore::measured(1.0)
    } else {
        SearchScore::measured(10.0 + f64::from(threads))
    }
}

#[test]
fn operation_auto_optimize_with_policy_filters_overbudget_gemm_candidates() {
    let operation = gemm_operation(128, 128, 256);
    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 1,
        require_launchable: false,
        min_score_improvement: 0.0,
    };
    let default_policy = KernelExpansionPolicy::for_search_config(false);
    let capped_policy = default_policy.with_max_threads_per_block(Some(128));

    let uncapped = auto_optimize_inference_kernel_with_policy_scorer(
        &operation,
        config,
        default_policy,
        |candidate, _problem| score_gemm_threads(candidate),
    )
    .expect("uncapped operation-level GEMM optimization should run");
    let uncapped_threads = uncapped
        .best_candidate()
        .and_then(|candidate| candidate.resources)
        .expect("uncapped best should carry resource metadata")
        .threads_per_block;

    let capped = auto_optimize_inference_kernel_with_policy_scorer(
        &operation,
        config,
        capped_policy,
        |candidate, _problem| score_gemm_threads(candidate),
    )
    .expect("policy-capped operation-level GEMM optimization should run");
    let capped_threads = capped
        .best_candidate()
        .and_then(|candidate| candidate.resources)
        .expect("capped best should carry resource metadata")
        .threads_per_block;

    assert!(uncapped_threads > 128);
    assert!(capped_threads <= 128);
    assert!(capped.result.rejected > 0);
}

#[test]
fn operation_generation_reuses_selection_cache() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let operation = matvec_operation(128, 256);
    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };

    let first = generate_inference_kernel_source_with_selection_cache(
        &store,
        &operation,
        config,
        "heuristic",
    )
    .expect("first operation-level generation should optimize and write selection cache");
    let second = generate_inference_kernel_source_with_selection_cache(
        &store,
        &operation,
        config,
        "heuristic",
    )
    .expect("second operation-level generation should replay selection cache");

    assert_eq!(first.optimization.cache_status, SelectionCacheStatus::Miss);
    assert!(first.optimization.cache_write.is_some());
    assert_eq!(second.optimization.cache_status, SelectionCacheStatus::Hit);
    assert!(second.optimization.cache_write.is_none());
    assert_eq!(second.optimization.result().explored, 0);
    assert_eq!(second.optimization.result().rejected, 0);
    assert_eq!(
        first.candidate.artifact_key(),
        second.candidate.artifact_key()
    );
    assert_eq!(first.source.symbol, second.source.symbol);
    assert!(
        second
            .source
            .source
            .contains(&format!("pub fn {}(", second.source.symbol))
    );

    remove_test_generated_root(&root);
}

#[test]
fn operation_generation_selection_cache_is_policy_specific() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let operation = matvec_operation(128, 256);
    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 1,
        require_launchable: false,
        min_score_improvement: 0.0,
    };
    let capped_policy =
        KernelExpansionPolicy::for_search_config(false).with_max_threads_per_block(Some(64));

    let default = generate_inference_kernel_source_with_selection_cache(
        &store,
        &operation,
        config,
        "heuristic",
    )
    .expect("default operation-level generation should write selection cache");
    let capped = generate_inference_kernel_source_with_selection_cache_and_policy(
        &store,
        &operation,
        config,
        capped_policy,
        "heuristic",
    )
    .expect("policy-capped operation-level generation should write selection cache");

    assert_eq!(
        default.optimization.cache_status,
        SelectionCacheStatus::Miss
    );
    assert_eq!(capped.optimization.cache_status, SelectionCacheStatus::Miss);
    assert_ne!(
        default.optimization.cache_key,
        capped.optimization.cache_key
    );
    assert!(
        capped.candidate.launch.block_dim.x
            <= capped_policy
                .max_threads_per_block
                .expect("policy should cap threads")
    );

    remove_test_generated_root(&root);
}

#[test]
fn operation_problem_rejects_unsupported_layouts_explicitly() {
    let operation = TypedOperationSpec::new(
        "bad-matvec-layout",
        OperationKind::Matvec,
        OperationRoute::CudaKernel,
    )
    .with_input(
        TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [256]).with_layout("contiguous"),
    )
    .with_input(
        TensorTypeSpec::new(NumericKind::Bf16, NumericKind::F32, [128, 256])
            .with_layout("column-major"),
    )
    .with_output(
        TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [128]).with_layout("contiguous"),
    );

    let error = InferenceKernelOptimizationProblem::from_operation(&operation)
        .expect_err("unsupported matvec layout should be rejected");

    let KernelGenerationError::UnsupportedOperation { name, kind, reason } = error else {
        panic!("unsupported operation should report the operation-level reason");
    };
    assert_eq!(name, "bad-matvec-layout");
    assert_eq!(kind, OperationKind::Matvec);
    assert!(reason.contains("row-major weights"));
}
