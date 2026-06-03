use super::*;

#[test]
fn cached_auto_optimize_replays_hit_without_expanding_beam() {
    let root = test_generated_root();
    fs::create_dir_all(&root).expect("test-generated root should be creatable");
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };

    let first = auto_optimize_metadata_with_selection_cache(
        &store,
        &problem,
        config,
        "heuristic",
        |candidate| problem.score(candidate),
    )
    .expect("cache miss should run auto optimize");
    assert_eq!(first.cache_status, SelectionCacheStatus::Miss);
    assert!(first.cache_write.is_some());
    assert!(first.result.explored > 0);
    let first_best = first
        .result
        .best
        .as_ref()
        .expect("auto optimize should find a best candidate")
        .artifact_key();

    let second = auto_optimize_metadata_with_selection_cache(
        &store,
        &problem,
        config,
        "heuristic",
        |candidate| problem.score(candidate),
    )
    .expect("cache hit should replay selected auto-optimized candidate");
    let second_best = second
        .result
        .best
        .as_ref()
        .expect("cache hit should have selected candidate");

    assert_eq!(second.cache_status, SelectionCacheStatus::Hit);
    assert!(second.cache_write.is_none());
    assert_eq!(second.result.exit_reason, AutoOptimizeExitReason::CacheHit);
    assert_eq!(second.result.explored, 0);
    assert_eq!(second.result.rejected, 0);
    assert!(second.result.steps.is_empty());
    assert_eq!(second.result.beam.len(), 1);
    assert_eq!(second_best.artifact_key(), first_best);

    remove_test_generated_root(&root);
}
