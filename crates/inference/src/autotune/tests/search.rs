use super::*;

#[test]
fn schedule_action_template_captures_tinygrad_like_beam_factors() {
    let template = KernelScheduleActionTemplate::TINYGRAD_LIKE;

    assert_eq!(template.upcast_factors, &[2, 3, 4, 5, 7]);
    assert_eq!(template.unroll_factors, &[4, 7]);
    assert_eq!(template.local_tile_factors, &[2, 3, 4, 8, 13, 16, 29, 32]);
    assert_eq!(
        template.group_top_factors,
        &[13, 16, 28, 29, 32, 49, 64, 256]
    );
    assert_eq!(
        template.thread_group_factors,
        &[2, 3, 4, 5, 8, 12, 16, 24, 32, 64]
    );
}

#[test]
fn inference_template_keeps_arbitrary_legal_split_and_unroll_factors() {
    let template = KernelScheduleActionTemplate::INFERENCE_DEFAULT;

    assert_eq!(
        template.bounded_split_factors(4096, 32, None),
        vec![8, 13, 16, 24, 32]
    );
    assert_eq!(template.bounded_split_factors(10, 32, None), vec![8, 10]);
    assert_eq!(
        template.exhaustive_unroll_factors(4096, 32, Some(4)),
        (1..=32).filter(|factor| *factor != 4).collect::<Vec<_>>()
    );
    assert_eq!(
        template.legal_upcast_factors(|factor| factor <= 4 && 16 % factor == 0),
        vec![2, 4]
    );
    assert_eq!(
        template.legal_thread_group_factors(|factor| factor >= 32 && factor < 128),
        vec![32, 64]
    );
}

#[test]
fn auto_optimize_preserves_parent_when_children_do_not_improve() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let result = auto_optimize_metadata_with_scorer(
        &problem,
        AutoOptimizeConfig {
            beam_width: 8,
            max_steps: 2,
            require_launchable: false,
            min_score_improvement: 0.0,
        },
        |candidate| {
            let plan = schedule_matvec_plan(&candidate.schedule)?;
            if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL
                && plan.row_upcast.is_default()
                && plan.thread_group.is_default()
            {
                SearchScore::measured(1.0)
            } else {
                SearchScore::measured(10.0)
            }
        },
    );
    let best = result
        .best
        .expect("auto optimize should keep the best parent candidate");
    let best_plan = schedule_matvec_plan(&best.schedule).expect("best candidate should plan");

    assert_eq!(
        result.exit_reason,
        AutoOptimizeExitReason::NoImprovement { best_delta: -9.0 }
    );
    assert_eq!(result.steps.len(), 2);
    assert_eq!(
        best_plan.reduce_unroll,
        MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL
    );
    assert_eq!(best.score.and_then(|score| Some(score.value)), Some(1.0));
}

#[test]
fn auto_optimize_accepts_improving_generated_action() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let result = auto_optimize_metadata_with_scorer(
        &problem,
        AutoOptimizeConfig {
            beam_width: 8,
            max_steps: 2,
            require_launchable: false,
            min_score_improvement: 0.0,
        },
        |candidate| {
            let plan = schedule_matvec_plan(&candidate.schedule)?;
            match plan.reduce_unroll {
                7 => SearchScore::measured(1.0),
                factor if factor == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL => {
                    SearchScore::measured(10.0)
                }
                _ => SearchScore::measured(5.0),
            }
        },
    );
    let best = result
        .best
        .expect("auto optimize should accept the improving generated candidate");
    let best_plan = schedule_matvec_plan(&best.schedule).expect("best candidate should plan");

    assert_eq!(result.exit_reason, AutoOptimizeExitReason::MaxSteps);
    assert_eq!(result.steps.len(), 2);
    assert_eq!(result.steps[1].improvement, Some(9.0));
    assert_eq!(best_plan.reduce_unroll, 7);
    assert_eq!(best.score.and_then(|score| Some(score.value)), Some(1.0));
}

#[test]
fn candidate_projects_to_profiling_optimization_spec() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let mut candidate = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("row split action should produce candidate metadata");
    candidate.score = SearchScore::heuristic(1.0);

    let spec = candidate.optimization_spec();

    assert_eq!(spec.family, "matvec-bf16-row-major");
    assert_eq!(spec.artifact_key, candidate.artifact_key().hex());
    assert_eq!(spec.generator, "row-major-matvec-generator");
    assert!(!spec.launchable);
    assert_eq!(spec.launch.kernel, "matvec_bf16_rows8");
    assert_eq!(spec.operation.kind, OperationKind::Matvec);
    assert_eq!(spec.action_trace, candidate.action_trace);
    assert_eq!(spec.score, candidate.score);
}

#[test]
fn optimization_candidate_spec_replay_checks_artifact_key() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::unroll(1, 8),
        ],
    )
    .expect("valid matvec action trace should replay into candidate metadata");
    let spec = candidate.optimization_spec();

    let replayed = replay_optimization_candidate_spec(&problem, &spec)
        .expect("matching optimization spec should replay");
    assert_eq!(replayed.artifact_key(), candidate.artifact_key());

    let mut stale_spec = spec;
    stale_spec.artifact_key = "0000000000000000".to_string();
    assert_eq!(
        replay_optimization_candidate_spec(&problem, &stale_spec),
        Err(KernelActionReplayError::ArtifactKeyMismatch {
            expected: "0000000000000000".to_string(),
            actual: candidate.artifact_key().hex(),
        })
    );
}

#[test]
fn action_trace_replay_rejects_invalid_action_sequence() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let action = KernelScheduleAction::unroll(1, 8);

    assert_eq!(
        replay_schedule_actions(&problem, std::slice::from_ref(&action)),
        Err(KernelActionReplayError::InvalidAction { index: 0, action })
    );
}

#[test]
fn metadata_expansion_filters_launchability_and_tracks_duplicates() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let mut seen = HashSet::new();
    seen.insert(seed.artifact_key());

    let expansion = expand_metadata_candidates(
        &problem,
        &seed,
        KernelExpansionPolicy::for_search_config(true),
        &mut seen,
    );

    assert_eq!(expansion.accepted, 4);
    assert_eq!(expansion.rejected, 32);
    assert_eq!(expansion.explored(), 36);
    assert_eq!(expansion.duplicates, 0);
    assert_eq!(
        expansion.last_reject_reason,
        Some(KernelCandidateRejectReason::DeferredGenerated)
    );
    assert!(
        expansion
            .candidates
            .iter()
            .all(|candidate| candidate.is_launchable())
    );

    let duplicate_expansion = expand_metadata_candidates(
        &problem,
        &seed,
        KernelExpansionPolicy::for_search_config(true),
        &mut seen,
    );
    assert_eq!(duplicate_expansion.accepted, 0);
    assert_eq!(duplicate_expansion.rejected, 0);
    assert_eq!(duplicate_expansion.explored(), 0);
    assert_eq!(duplicate_expansion.duplicates, 36);
}

#[derive(Debug, Clone, Copy)]
struct BudgetedSearchProblem;

impl BudgetedSearchProblem {
    fn candidate(factor: u32, accumulators: u32) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new(
            format!("budget_candidate_{factor}"),
            (1, 1, 1),
            (1, 1, 1),
            0,
        );
        let operation = TypedOperationSpec::new(
            format!("budget-candidate-{factor}"),
            OperationKind::Gemm,
            OperationRoute::CudaKernel,
        )
        .with_launch(launch.clone());
        let mut candidate = KernelCandidateMetadata::new(
            "budget-test",
            vec![KernelAxis::spatial(0, "x", 1, Some(1))],
            KernelSchedule::new().with_transform(ScheduleTransform::Split { axis: 0, factor }),
            "budget-generator",
            KernelMaterialization::DeferredGenerated {
                symbol_hint: format!("budget_candidate_{factor}"),
                reason: "test candidate only carries metadata".to_string(),
            },
            launch,
            operation,
        );
        candidate.resources = Some(KernelResourceUsage::new(
            1,
            0,
            accumulators,
            accumulators,
            1,
        ));
        candidate
    }
}

impl KernelMetadataSearchProblem for BudgetedSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        Self::candidate(0, 1)
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        if candidate.schedule.depth() == 1 {
            vec![Self::candidate(1, 4), Self::candidate(2, 16)]
        } else {
            Vec::new()
        }
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let accumulators = candidate.resources?.accumulator_elements_per_thread;
        if candidate
            .schedule
            .transforms
            .iter()
            .any(|transform| matches!(transform, ScheduleTransform::Split { axis: 0, factor: 0 }))
        {
            SearchScore::measured(20.0)
        } else if accumulators > 8 {
            SearchScore::measured(1.0)
        } else {
            SearchScore::measured(10.0)
        }
    }
}

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
