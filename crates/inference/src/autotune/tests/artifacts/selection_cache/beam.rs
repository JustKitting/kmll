use super::*;

#[test]
fn artifact_store_writes_and_reads_selection_cache() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let config = BeamSearchConfig {
        beam_width: 4,
        max_depth: 3,
        require_launchable: false,
    };
    let cache_key = optimization_selection_cache_key(&problem, config, "heuristic");
    let candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::tile_gemm(
                16,
                32,
                16,
                KernelActionMaterialization::DeferredGenerated,
            ),
            KernelScheduleAction::unroll(2, 4),
            KernelScheduleAction::stride_order(vec![2, 1]),
        ],
    )
    .expect("valid GEMM action trace should replay into candidate metadata");

    assert!(
        store
            .read_selection_cache(&cache_key)
            .expect("missing selection cache should not be an error")
            .is_none()
    );

    let emitted = store
        .emit_selection_cache_for_candidate(&cache_key, &candidate)
        .expect("artifact store should write selection cache metadata");
    assert!(emitted.selection_path.starts_with(store.root()));
    assert!(
        emitted
            .selection_path
            .components()
            .any(|component| component.as_os_str() == "selection-cache")
    );
    assert_eq!(
        emitted
            .selection_path
            .file_stem()
            .and_then(|stem| stem.to_str()),
        Some(cache_key.hex().as_str())
    );

    let selection = store
        .read_selection_cache(&cache_key)
        .expect("selection cache should parse")
        .expect("selection cache should exist after write");
    let replayed = selection
        .replay(&problem)
        .expect("cached selection should replay into selected candidate");
    assert_eq!(selection.artifact_key, candidate.artifact_key().hex());
    assert_eq!(replayed.artifact_key(), candidate.artifact_key());
    assert_eq!(replayed.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u4_bk");

    let cache_text =
        fs::read_to_string(&emitted.selection_path).expect("selection cache should be readable");
    assert!(cache_text.contains("\"action_trace\""));
    assert!(!cache_text.contains("#[kernel]"));
    assert!(!cache_text.contains("pub fn gemm_f32_bf16"));

    remove_test_generated_root(&root);
}

#[test]
fn cached_beam_search_reports_miss_and_runs_search() {
    let root = test_generated_root();
    fs::create_dir_all(&root).expect("test-generated root should be creatable");
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let config = BeamSearchConfig {
        beam_width: 4,
        max_depth: 2,
        require_launchable: false,
    };

    let cached = beam_search_metadata_with_selection_cache(
        &store,
        &problem,
        config,
        "heuristic",
        |candidate| problem.score(candidate),
    )
    .expect("cache miss should still run beam search");

    assert_eq!(cached.cache_status, SelectionCacheStatus::Miss);
    assert!(cached.result.explored > 0);
    assert!(cached.result.best.is_some());
    assert!(cached.cache_write.is_some());
    assert_eq!(
        cached.cache_key,
        optimization_selection_cache_key(&problem, config, "heuristic")
    );
    let cached_selection = store
        .read_selection_cache(&cached.cache_key)
        .expect("selection cache read should succeed")
        .expect("miss fallback should write selection cache metadata");
    let replayed = cached_selection
        .replay(&problem)
        .expect("written cache selection should replay");
    assert_eq!(
        replayed.artifact_key(),
        cached
            .result
            .best
            .as_ref()
            .expect("miss fallback should have a best candidate")
            .artifact_key()
    );

    remove_test_generated_root(&root);
}

#[test]
fn cached_beam_search_replays_hit_without_expanding_beam() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let config = BeamSearchConfig {
        beam_width: 4,
        max_depth: 2,
        require_launchable: false,
    };
    let cache_key = optimization_selection_cache_key(&problem, config, "heuristic");
    let mut candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::unroll(1, 8),
        ],
    )
    .expect("valid matvec action trace should replay into candidate metadata");
    candidate.score = SearchScore::heuristic(12.0);
    store
        .emit_selection_cache_for_candidate(&cache_key, &candidate)
        .expect("artifact store should write selection cache metadata");

    let cached = beam_search_metadata_with_selection_cache(
        &store,
        &problem,
        config,
        "heuristic",
        |candidate| problem.score(candidate),
    )
    .expect("cache hit should replay selected candidate");
    let best = cached
        .result
        .best
        .expect("cache hit should produce selected candidate");

    assert_eq!(cached.cache_status, SelectionCacheStatus::Hit);
    assert!(cached.cache_write.is_none());
    assert_eq!(cached.result.explored, 0);
    assert_eq!(cached.result.rejected, 0);
    assert_eq!(cached.result.beam.len(), 1);
    assert_eq!(best.artifact_key(), candidate.artifact_key());
    assert_eq!(best.score, SearchScore::heuristic(12.0));

    remove_test_generated_root(&root);
}

#[test]
fn cached_beam_search_falls_back_on_stale_selection() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let config = BeamSearchConfig {
        beam_width: 4,
        max_depth: 2,
        require_launchable: false,
    };
    let cache_key = optimization_selection_cache_key(&problem, config, "heuristic");
    let candidate = replay_schedule_actions(
        &problem,
        &[KernelScheduleAction::split(
            0,
            8,
            KernelActionMaterialization::DeferredGenerated,
        )],
    )
    .expect("valid matvec action trace should replay into candidate metadata");
    let mut stale_selection = KernelOptimizationSelection::from_candidate(&candidate);
    stale_selection.artifact_key = "0000000000000000".to_string();
    store
        .emit_selection_cache(&cache_key, &stale_selection)
        .expect("artifact store should write stale selection cache metadata");

    let cached = beam_search_metadata_with_selection_cache(
        &store,
        &problem,
        config,
        "heuristic",
        |candidate| problem.score(candidate),
    )
    .expect("stale cache should fall back to beam search");

    assert!(matches!(
        cached.cache_status,
        SelectionCacheStatus::Stale { .. }
    ));
    assert!(cached.result.explored > 0);
    assert!(cached.result.best.is_some());
    assert!(cached.cache_write.is_some());
    let refreshed = store
        .read_selection_cache(&cached.cache_key)
        .expect("selection cache read should succeed after stale fallback")
        .expect("stale fallback should refresh selection cache metadata");
    let refreshed_candidate = refreshed
        .replay(&problem)
        .expect("refreshed cache selection should replay");
    assert_eq!(
        refreshed_candidate.artifact_key(),
        cached
            .result
            .best
            .as_ref()
            .expect("stale fallback should have a best candidate")
            .artifact_key()
    );

    remove_test_generated_root(&root);
}
