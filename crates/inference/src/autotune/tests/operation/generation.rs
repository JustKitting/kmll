use super::*;

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
