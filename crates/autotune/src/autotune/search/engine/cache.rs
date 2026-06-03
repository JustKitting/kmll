use super::super::*;

pub fn beam_search_metadata_with_selection_cache<P, F>(
    store: &KernelArtifactStore,
    problem: &P,
    config: BeamSearchConfig,
    score_namespace: &str,
    score_candidate: F,
) -> Result<CachedBeamSearchResult, KernelGenerationError>
where
    P: KernelActionSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    beam_search_metadata_with_selection_cache_and_policy(
        store,
        problem,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
        score_namespace,
        score_candidate,
    )
}

pub fn beam_search_metadata_with_selection_cache_and_policy<P, F>(
    store: &KernelArtifactStore,
    problem: &P,
    config: BeamSearchConfig,
    policy: KernelExpansionPolicy,
    score_namespace: &str,
    mut score_candidate: F,
) -> Result<CachedBeamSearchResult, KernelGenerationError>
where
    P: KernelActionSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    let cache_key =
        optimization_selection_cache_key_with_policy(problem, config, policy, score_namespace);
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
                        duplicates: 0,
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
    let result = beam_search_metadata_with_policy_scorer(problem, config, policy, |candidate| {
        score_candidate(candidate)
    });
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
    score_candidate: F,
) -> Result<CachedAutoOptimizeResult, KernelGenerationError>
where
    P: KernelActionSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    auto_optimize_metadata_with_selection_cache_and_policy(
        store,
        problem,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
        score_namespace,
        score_candidate,
    )
}

pub fn auto_optimize_metadata_with_selection_cache_and_policy<P, F>(
    store: &KernelArtifactStore,
    problem: &P,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
    score_namespace: &str,
    mut score_candidate: F,
) -> Result<CachedAutoOptimizeResult, KernelGenerationError>
where
    P: KernelActionSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    let cache_key =
        auto_optimization_selection_cache_key_with_policy(problem, config, policy, score_namespace);
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
                        duplicates: 0,
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
    let result = auto_optimize_metadata_with_policy_scorer(problem, config, policy, |candidate| {
        score_candidate(candidate)
    });
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
    optimization_selection_cache_key_with_policy(
        problem,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
        score_namespace,
    )
}

pub fn optimization_selection_cache_key_with_policy<P>(
    problem: &P,
    config: BeamSearchConfig,
    policy: KernelExpansionPolicy,
    score_namespace: &str,
) -> KernelOptimizationCacheKey
where
    P: KernelActionSearchProblem,
{
    let seed = problem.seed();
    let mut state = FNV_OFFSET;
    state = hash_str(state, "optimization-selection-cache");
    state = hash_str(state, "implementation-dedupe-v1");
    state = hash_str(state, score_namespace);
    state = hash_u64(state, config.beam_width as u64);
    state = hash_u64(state, config.max_depth as u64);
    state = hash_u64(state, u64::from(config.require_launchable));
    state = hash_expansion_policy(state, policy);
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
    auto_optimization_selection_cache_key_with_policy(
        problem,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
        score_namespace,
    )
}

pub fn auto_optimization_selection_cache_key_with_policy<P>(
    problem: &P,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
    score_namespace: &str,
) -> KernelOptimizationCacheKey
where
    P: KernelActionSearchProblem,
{
    let seed = problem.seed();
    let mut state = FNV_OFFSET;
    state = hash_str(state, "auto-optimization-selection-cache");
    state = hash_str(state, "implementation-dedupe-v1");
    state = hash_str(state, score_namespace);
    state = hash_u64(state, config.beam_width as u64);
    state = hash_u64(state, config.max_steps as u64);
    state = hash_u64(state, u64::from(config.require_launchable));
    state = hash_u64(state, config.min_score_improvement.to_bits());
    state = hash_expansion_policy(state, policy);
    state = hash_optimization_candidate(state, &seed.optimization_spec());
    state = hash_action_space_set(state, &problem.search_space());
    KernelOptimizationCacheKey {
        family: seed.family,
        key: KernelMetadataKey(state),
    }
}

fn hash_expansion_policy(mut state: u64, policy: KernelExpansionPolicy) -> u64 {
    state = hash_str(state, "expansion-policy");
    state = hash_u64(state, u64::from(policy.require_launchable));
    state = hash_optional_u32(state, policy.max_threads_per_block);
    state = hash_optional_u32(state, policy.max_shared_memory_bytes);
    state = hash_optional_u32(state, policy.max_accumulator_elements_per_thread);
    state = hash_optional_u32(state, policy.max_output_elements_per_thread);
    state = hash_optional_u32(state, policy.max_load_elements_per_block);
    state
}

fn hash_optional_u32(mut state: u64, value: Option<u32>) -> u64 {
    match value {
        Some(value) => {
            state = hash_u64(state, 1);
            hash_u64(state, u64::from(value))
        }
        None => hash_u64(state, 0),
    }
}
