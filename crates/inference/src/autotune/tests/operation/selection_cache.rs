use super::*;

#[test]
fn operation_generation_reuses_selection_cache() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let operation = matvec_operation(128, 256);
    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };

    let first = generate_inference_kernel_source_with_selection_cache(
        &store,
        &operation,
        config,
        "heuristic",
    )
    .expect("first operation-level generation should optimize and write selection cache");
    let second = generate_inference_kernel_source_with_selection_cache(
        &store,
        &operation,
        config,
        "heuristic",
    )
    .expect("second operation-level generation should replay selection cache");

    assert_eq!(first.optimization.cache_status, SelectionCacheStatus::Miss);
    assert!(first.optimization.cache_write.is_some());
    assert_eq!(second.optimization.cache_status, SelectionCacheStatus::Hit);
    assert!(second.optimization.cache_write.is_none());
    assert_eq!(second.optimization.result().explored, 0);
    assert_eq!(second.optimization.result().rejected, 0);
    assert_eq!(
        first.candidate.artifact_key(),
        second.candidate.artifact_key()
    );
    assert_eq!(first.source.symbol, second.source.symbol);
    assert!(
        second
            .source
            .source
            .contains(&format!("pub fn {}(", second.source.symbol))
    );

    remove_test_generated_root(&root);
}

#[test]
fn operation_generation_selection_cache_is_policy_specific() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let operation = matvec_operation(128, 256);
    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 1,
        require_launchable: false,
        min_score_improvement: 0.0,
    };
    let capped_policy =
        KernelExpansionPolicy::for_search_config(false).with_max_threads_per_block(Some(64));

    let default = generate_inference_kernel_source_with_selection_cache(
        &store,
        &operation,
        config,
        "heuristic",
    )
    .expect("default operation-level generation should write selection cache");
    let capped = generate_inference_kernel_source_with_selection_cache_and_policy(
        &store,
        &operation,
        config,
        capped_policy,
        "heuristic",
    )
    .expect("policy-capped operation-level generation should write selection cache");

    assert_eq!(
        default.optimization.cache_status,
        SelectionCacheStatus::Miss
    );
    assert_eq!(capped.optimization.cache_status, SelectionCacheStatus::Miss);
    assert_ne!(
        default.optimization.cache_key,
        capped.optimization.cache_key
    );
    assert!(
        capped.candidate.launch.block_dim.x
            <= capped_policy
                .max_threads_per_block
                .expect("policy should cap threads")
    );

    remove_test_generated_root(&root);
}
