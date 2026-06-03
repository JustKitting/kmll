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
        assert!(expansion.candidates.iter().all(|candidate| candidate.is_launchable()));

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
            Err(KernelCandidateRejectReason::AccumulatorElementsPerThread {
                actual: 16,
                max: 8
            })
        );
    }
