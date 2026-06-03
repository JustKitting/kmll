use super::super::*;

pub fn auto_optimize_metadata<P>(problem: &P, config: AutoOptimizeConfig) -> AutoOptimizeResult
where
    P: KernelMetadataSearchProblem,
{
    auto_optimize_metadata_with_scorer(problem, config, |candidate| problem.score(candidate))
}

pub fn auto_optimize_metadata_with_scorer<P, F>(
    problem: &P,
    config: AutoOptimizeConfig,
    score_candidate: F,
) -> AutoOptimizeResult
where
    P: KernelMetadataSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    auto_optimize_metadata_with_policy_scorer(
        problem,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
        score_candidate,
    )
}

pub fn auto_optimize_metadata_with_policy<P>(
    problem: &P,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
) -> AutoOptimizeResult
where
    P: KernelMetadataSearchProblem,
{
    auto_optimize_metadata_with_policy_scorer(problem, config, policy, |candidate| {
        problem.score(candidate)
    })
}

pub fn auto_optimize_metadata_with_policy_scorer<P, F>(
    problem: &P,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
    mut score_candidate: F,
) -> AutoOptimizeResult
where
    P: KernelMetadataSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    assert!(config.beam_width > 0, "beam width must be nonzero");
    assert!(
        config.min_score_improvement.is_finite() && config.min_score_improvement >= 0.0,
        "minimum score improvement must be finite and nonnegative"
    );

    let mut seed = problem.seed();
    seed.score = score_candidate(&seed);
    let mut seen = HashSet::new();
    seen.insert(seed.implementation_key());
    let mut beam = vec![seed];
    let mut explored = 0;
    let mut rejected = 0;
    let mut duplicates = 0;
    let mut steps = Vec::new();
    let mut exit_reason = AutoOptimizeExitReason::MaxSteps;

    for depth in 0..config.max_steps {
        let input_beam_len = beam.len();
        let best_before = beam.first().and_then(|candidate| candidate.score);
        let mut candidates = Vec::new();
        let mut generated = 0;
        let mut step_rejected = 0;
        let mut step_duplicates = 0;

        for candidate in &beam {
            let expansion = expand_metadata_candidates(problem, candidate, policy, &mut seen);
            explored += expansion.explored();
            generated += expansion.explored();
            rejected += expansion.rejected;
            step_rejected += expansion.rejected;
            duplicates += expansion.duplicates;
            step_duplicates += expansion.duplicates;
            for mut next in expansion.candidates {
                match score_candidate(&next) {
                    Some(score) => {
                        next.score = Some(score);
                        candidates.push(next);
                    }
                    None => {
                        rejected += 1;
                        step_rejected += 1;
                    }
                }
            }
        }

        if candidates.is_empty() {
            steps.push(AutoOptimizeStep {
                depth,
                input_beam_len,
                generated,
                accepted: 0,
                rejected: step_rejected,
                duplicates: step_duplicates,
                best_before,
                best_after: None,
                best_candidate: None,
                improvement: None,
            });
            exit_reason = AutoOptimizeExitReason::NoCandidates;
            break;
        }

        candidates.sort_by(compare_candidates);
        let accepted = candidates.len().min(config.beam_width);
        let next_beam = candidates
            .into_iter()
            .take(config.beam_width)
            .collect::<Vec<_>>();
        let best_after = next_beam.first().and_then(|candidate| candidate.score);
        let best_candidate = next_beam.first().cloned();
        let improvement =
            best_before.and_then(|before| best_after.map(|after| before.value - after.value));
        let stop_for_no_improvement = improvement
            .map(|delta| delta <= config.min_score_improvement)
            .unwrap_or(false);

        steps.push(AutoOptimizeStep {
            depth,
            input_beam_len,
            generated,
            accepted,
            rejected: step_rejected,
            duplicates: step_duplicates,
            best_before,
            best_after,
            best_candidate: best_candidate.clone(),
            improvement,
        });

        if stop_for_no_improvement {
            if improvement.is_some_and(|delta| delta > 0.0)
                && let Some(best_next) = best_candidate
            {
                beam = vec![best_next];
            }
            exit_reason = AutoOptimizeExitReason::NoImprovement {
                best_delta: improvement.unwrap_or(0.0),
            };
            break;
        }

        beam = next_beam;
    }

    let best = beam.first().cloned();
    AutoOptimizeResult {
        best,
        beam,
        explored,
        rejected,
        duplicates,
        steps,
        exit_reason,
    }
}
