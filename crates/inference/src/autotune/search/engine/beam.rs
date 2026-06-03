use super::super::*;

pub fn beam_search_metadata<P>(problem: &P, config: BeamSearchConfig) -> BeamSearchResult
where
    P: KernelMetadataSearchProblem,
{
    beam_search_metadata_with_scorer(problem, config, |candidate| problem.score(candidate))
}

pub fn beam_search_metadata_with_scorer<P, F>(
    problem: &P,
    config: BeamSearchConfig,
    score_candidate: F,
) -> BeamSearchResult
where
    P: KernelMetadataSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    beam_search_metadata_with_policy_scorer(
        problem,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
        score_candidate,
    )
}

pub fn beam_search_metadata_with_policy<P>(
    problem: &P,
    config: BeamSearchConfig,
    policy: KernelExpansionPolicy,
) -> BeamSearchResult
where
    P: KernelMetadataSearchProblem,
{
    beam_search_metadata_with_policy_scorer(problem, config, policy, |candidate| {
        problem.score(candidate)
    })
}

pub fn beam_search_metadata_with_policy_scorer<P, F>(
    problem: &P,
    config: BeamSearchConfig,
    policy: KernelExpansionPolicy,
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
    seen.insert(seed.implementation_key());
    let mut beam = vec![seed];
    let mut explored = 0;
    let mut rejected = 0;
    let mut duplicates = 0;

    for _ in 0..config.max_depth {
        let mut candidates = Vec::new();
        for candidate in &beam {
            let expansion = expand_metadata_candidates(problem, candidate, policy, &mut seen);
            explored += expansion.explored();
            rejected += expansion.rejected;
            duplicates += expansion.duplicates;
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
        duplicates,
    }
}
