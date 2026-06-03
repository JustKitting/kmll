use super::*;

pub fn beam_search_metadata<P>(problem: &P, config: BeamSearchConfig) -> BeamSearchResult
where
    P: KernelMetadataSearchProblem,
{
    beam_search_metadata_with_scorer(problem, config, |candidate| problem.score(candidate))
}

pub fn beam_search_metadata_with_scorer<P, F>(
    problem: &P,
    config: BeamSearchConfig,
    mut score_candidate: F,
) -> BeamSearchResult
where
    P: KernelMetadataSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    assert!(config.beam_width > 0, "beam width must be nonzero");

    let mut seed = problem.seed();
    seed.score = score_candidate(&seed);
    let mut seen = HashSet::new();
    seen.insert(seed.artifact_key());
    let mut beam = vec![seed];
    let mut explored = 0;
    let mut rejected = 0;
    let expansion_policy = KernelExpansionPolicy::for_search_config(config.require_launchable);

    for _ in 0..config.max_depth {
        let mut candidates = Vec::new();
        for candidate in &beam {
            let expansion =
                expand_metadata_candidates(problem, candidate, expansion_policy, &mut seen);
            explored += expansion.explored();
            rejected += expansion.rejected;
            for mut next in expansion.candidates {
                match score_candidate(&next) {
                    Some(score) => {
                        next.score = Some(score);
                        candidates.push(next);
                    }
                    None => rejected += 1,
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        candidates.sort_by(compare_candidates);
        beam = candidates.into_iter().take(config.beam_width).collect();
    }

    let best = beam.first().cloned();
    BeamSearchResult {
        best,
        beam,
        explored,
        rejected,
    }
}

pub fn auto_optimize_metadata<P>(problem: &P, config: AutoOptimizeConfig) -> AutoOptimizeResult
where
    P: KernelMetadataSearchProblem,
{
    auto_optimize_metadata_with_scorer(problem, config, |candidate| problem.score(candidate))
}

pub fn auto_optimize_metadata_with_scorer<P, F>(
    problem: &P,
    config: AutoOptimizeConfig,
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
    seen.insert(seed.artifact_key());
    let mut beam = vec![seed];
    let mut explored = 0;
    let mut rejected = 0;
    let mut steps = Vec::new();
    let mut exit_reason = AutoOptimizeExitReason::MaxSteps;
    let expansion_policy = KernelExpansionPolicy::for_search_config(config.require_launchable);

    for depth in 0..config.max_steps {
        let input_beam_len = beam.len();
        let best_before = beam.first().and_then(|candidate| candidate.score);
        let mut candidates = Vec::new();
        let mut generated = 0;
        let mut step_rejected = 0;

        for candidate in &beam {
            let expansion =
                expand_metadata_candidates(problem, candidate, expansion_policy, &mut seen);
            explored += expansion.explored();
            generated += expansion.explored();
            rejected += expansion.rejected;
            step_rejected += expansion.rejected;
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
        steps,
        exit_reason,
    }
}

pub fn beam_search_metadata_with_selection_cache<P, F>(
    store: &KernelArtifactStore,
    problem: &P,
    config: BeamSearchConfig,
    score_namespace: &str,
    mut score_candidate: F,
) -> Result<CachedBeamSearchResult, KernelGenerationError>
where
    P: KernelActionSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    let cache_key = optimization_selection_cache_key(problem, config, score_namespace);
    let cache_status = match store.read_selection_cache(&cache_key)? {
        Some(selection) => match selection.replay(problem) {
            Ok(mut candidate) => {
                candidate.score = selection.score;
                return Ok(CachedBeamSearchResult {
                    result: BeamSearchResult {
                        best: Some(candidate.clone()),
                        beam: vec![candidate],
                        explored: 0,
                        rejected: 0,
                    },
                    cache_key,
                    cache_status: SelectionCacheStatus::Hit,
                    cache_write: None,
                });
            }
            Err(error) => SelectionCacheStatus::Stale {
                reason: error.to_string(),
            },
        },
        None => SelectionCacheStatus::Miss,
    };
    let result =
        beam_search_metadata_with_scorer(problem, config, |candidate| score_candidate(candidate));
    let cache_write = result
        .best
        .as_ref()
        .map(|candidate| store.emit_selection_cache_for_candidate(&cache_key, candidate))
        .transpose()?;
    Ok(CachedBeamSearchResult {
        result,
        cache_key,
        cache_status,
        cache_write,
    })
}

pub fn auto_optimize_metadata_with_selection_cache<P, F>(
    store: &KernelArtifactStore,
    problem: &P,
    config: AutoOptimizeConfig,
    score_namespace: &str,
    mut score_candidate: F,
) -> Result<CachedAutoOptimizeResult, KernelGenerationError>
where
    P: KernelActionSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    let cache_key = auto_optimization_selection_cache_key(problem, config, score_namespace);
    let cache_status = match store.read_selection_cache(&cache_key)? {
        Some(selection) => match selection.replay(problem) {
            Ok(mut candidate) => {
                candidate.score = selection.score;
                return Ok(CachedAutoOptimizeResult {
                    result: AutoOptimizeResult {
                        best: Some(candidate.clone()),
                        beam: vec![candidate],
                        explored: 0,
                        rejected: 0,
                        steps: Vec::new(),
                        exit_reason: AutoOptimizeExitReason::CacheHit,
                    },
                    cache_key,
                    cache_status: SelectionCacheStatus::Hit,
                    cache_write: None,
                });
            }
            Err(error) => SelectionCacheStatus::Stale {
                reason: error.to_string(),
            },
        },
        None => SelectionCacheStatus::Miss,
    };
    let result =
        auto_optimize_metadata_with_scorer(problem, config, |candidate| score_candidate(candidate));
    let cache_write = result
        .best
        .as_ref()
        .map(|candidate| store.emit_selection_cache_for_candidate(&cache_key, candidate))
        .transpose()?;
    Ok(CachedAutoOptimizeResult {
        result,
        cache_key,
        cache_status,
        cache_write,
    })
}

pub fn optimization_selection_cache_key<P>(
    problem: &P,
    config: BeamSearchConfig,
    score_namespace: &str,
) -> KernelOptimizationCacheKey
where
    P: KernelActionSearchProblem,
{
    let seed = problem.seed();
    let mut state = FNV_OFFSET;
    state = hash_str(state, "optimization-selection-cache");
    state = hash_str(state, score_namespace);
    state = hash_u64(state, config.beam_width as u64);
    state = hash_u64(state, config.max_depth as u64);
    state = hash_u64(state, u64::from(config.require_launchable));
    state = hash_optimization_candidate(state, &seed.optimization_spec());
    state = hash_action_space_set(state, &problem.search_space());
    KernelOptimizationCacheKey {
        family: seed.family,
        key: KernelMetadataKey(state),
    }
}

pub fn auto_optimization_selection_cache_key<P>(
    problem: &P,
    config: AutoOptimizeConfig,
    score_namespace: &str,
) -> KernelOptimizationCacheKey
where
    P: KernelActionSearchProblem,
{
    let seed = problem.seed();
    let mut state = FNV_OFFSET;
    state = hash_str(state, "auto-optimization-selection-cache");
    state = hash_str(state, score_namespace);
    state = hash_u64(state, config.beam_width as u64);
    state = hash_u64(state, config.max_steps as u64);
    state = hash_u64(state, u64::from(config.require_launchable));
    state = hash_u64(state, config.min_score_improvement.to_bits());
    state = hash_optimization_candidate(state, &seed.optimization_spec());
    state = hash_action_space_set(state, &problem.search_space());
    KernelOptimizationCacheKey {
        family: seed.family,
        key: KernelMetadataKey(state),
    }
}
