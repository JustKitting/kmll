use super::*;

#[test]
fn selection_cache_key_tracks_problem_config_and_score_namespace() {
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let same_problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let different_problem = MatvecSearchProblem::bf16_row_major(128, 512);
    let config = BeamSearchConfig {
        beam_width: 4,
        max_depth: 2,
        require_launchable: false,
    };

    let key = optimization_selection_cache_key(&problem, config, "heuristic");

    assert_eq!(
        key,
        optimization_selection_cache_key(&same_problem, config, "heuristic")
    );
    assert_ne!(
        key,
        optimization_selection_cache_key(&different_problem, config, "heuristic")
    );
    assert_ne!(
        key,
        optimization_selection_cache_key(
            &problem,
            BeamSearchConfig {
                beam_width: 8,
                ..config
            },
            "heuristic"
        )
    );
    assert_ne!(
        key,
        optimization_selection_cache_key(&problem, config, "measured-cuda-event-r5-w2")
    );
    assert_ne!(
        key,
        optimization_selection_cache_key_with_policy(
            &problem,
            config,
            KernelExpansionPolicy::for_search_config(false).with_max_threads_per_block(Some(64)),
            "heuristic"
        )
    );
    assert_ne!(
        key,
        optimization_selection_cache_key(&MatvecWithoutUnrollSpace(problem), config, "heuristic")
    );
    assert_eq!(key.family, "matvec-bf16-row-major");
    assert_eq!(key.hex().len(), 16);
}

#[test]
fn auto_selection_cache_key_tracks_auto_config_and_action_space() {
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };

    let key = auto_optimization_selection_cache_key(&problem, config, "heuristic");

    assert_eq!(
        key,
        auto_optimization_selection_cache_key(&problem, config, "heuristic")
    );
    assert_ne!(
        key,
        auto_optimization_selection_cache_key(
            &problem,
            AutoOptimizeConfig {
                min_score_improvement: 0.5,
                ..config
            },
            "heuristic"
        )
    );
    assert_ne!(
        key,
        auto_optimization_selection_cache_key(
            &problem,
            AutoOptimizeConfig {
                max_steps: 3,
                ..config
            },
            "heuristic"
        )
    );
    assert_ne!(
        key,
        auto_optimization_selection_cache_key(
            &MatvecWithoutUnrollSpace(problem),
            config,
            "heuristic"
        )
    );
    assert_ne!(
        key,
        auto_optimization_selection_cache_key_with_policy(
            &problem,
            config,
            KernelExpansionPolicy::for_search_config(false).with_max_threads_per_block(Some(64)),
            "heuristic"
        )
    );
    assert_eq!(key.family, "matvec-bf16-row-major");
    assert_eq!(key.hex().len(), 16);
}
