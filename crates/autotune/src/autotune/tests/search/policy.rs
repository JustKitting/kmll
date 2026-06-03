use super::*;

#[test]
fn policy_aware_search_rejects_overbudget_candidates_during_beam_expansion() {
    let problem = BudgetedSearchProblem;
    let beam_config = BeamSearchConfig {
        beam_width: 2,
        max_depth: 1,
        require_launchable: false,
    };
    let default_policy = KernelExpansionPolicy::for_search_config(false);

    let unconstrained = beam_search_metadata_with_policy(&problem, beam_config, default_policy);
    let unconstrained_best = unconstrained
        .best
        .as_ref()
        .expect("unconstrained search should find a best candidate");

    assert_eq!(unconstrained.rejected, 0);
    assert_eq!(
        unconstrained_best
            .resources
            .expect("candidate should carry resource metadata")
            .accumulator_elements_per_thread,
        16
    );

    let capped_policy = default_policy.with_max_accumulator_elements_per_thread(Some(8));
    let capped = beam_search_metadata_with_policy(&problem, beam_config, capped_policy);
    let capped_best = capped
        .best
        .as_ref()
        .expect("capped search should still find an admissible candidate");

    assert_eq!(capped.rejected, 1);
    assert_eq!(
        capped_best
            .resources
            .expect("candidate should carry resource metadata")
            .accumulator_elements_per_thread,
        4
    );

    let auto_config = AutoOptimizeConfig::from_beam_search_config(beam_config);
    let capped_auto = auto_optimize_metadata_with_policy(&problem, auto_config, capped_policy);
    let capped_auto_best = capped_auto
        .best
        .as_ref()
        .expect("policy-aware auto optimize should find an admissible best candidate");

    assert_eq!(capped_auto.rejected, 1);
    assert_eq!(
        capped_auto_best
            .resources
            .expect("candidate should carry resource metadata")
            .accumulator_elements_per_thread,
        4
    );
}

#[test]
fn beam_and_auto_search_results_track_duplicate_expansions() {
    let problem = BudgetedSearchProblem;
    let beam_config = BeamSearchConfig {
        beam_width: 2,
        max_depth: 2,
        require_launchable: false,
    };

    let result = beam_search_metadata(&problem, beam_config);
    assert_eq!(result.duplicates, 4);
    let report = result.optimization_report("budget-test", beam_config);
    assert_eq!(report.duplicates, 4);

    let auto_config = AutoOptimizeConfig::from_beam_search_config(beam_config);
    let auto = auto_optimize_metadata(&problem, auto_config);
    assert_eq!(auto.duplicates, 4);
    assert_eq!(auto.steps.len(), 2);
    assert_eq!(auto.steps[1].duplicates, 4);
    let auto_report = auto.auto_optimization_report("budget-test", auto_config);
    assert_eq!(auto_report.duplicates, 4);
    assert_eq!(auto_report.steps[1].duplicates, 4);
}

#[test]
fn expansion_policy_rejects_overbudget_kernel_resources() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let overbudget =
        problem.candidate_for_plan(GemmSchedulePlan::new(GemmTileShape::new(128, 128, 64)));

    assert_eq!(
        KernelExpansionPolicy::default().allows(&overbudget),
        Err(KernelCandidateRejectReason::ThreadsPerBlock {
            actual: 16_384,
            max: 1024
        })
    );
    assert_eq!(
        KernelExpansionPolicy::default()
            .with_max_threads_per_block(None)
            .allows(&overbudget),
        Err(KernelCandidateRejectReason::SharedMemoryBytes {
            actual: 65_536,
            max: 48 * 1024
        })
    );

    let register_heavy = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 16, 16))
            .with_m_per_thread(4)
            .with_n_per_thread(4),
    );
    assert_eq!(
        KernelExpansionPolicy::default()
            .with_max_accumulator_elements_per_thread(Some(8))
            .allows(&register_heavy),
        Err(KernelCandidateRejectReason::AccumulatorElementsPerThread { actual: 16, max: 8 })
    );
}
