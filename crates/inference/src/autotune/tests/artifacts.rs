use super::*;

#[test]
fn artifact_store_writes_metadata_manifest_without_kernel_source() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
    let optimization_spec = candidate.optimization_spec();

    assert_eq!(optimization_spec.resources, candidate.resources);

    let emitted = store
        .emit_metadata(&candidate)
        .expect("artifact store should write generated metadata manifest");

    assert_eq!(emitted.artifact_key, candidate.artifact_key());
    assert!(emitted.paths.directory.starts_with(store.root()));
    assert!(emitted.paths.manifest_path.starts_with(store.root()));
    assert!(emitted.manifest_bytes > 0);
    assert!(
        !emitted.paths.directory.join("kernel.rs").exists(),
        "metadata emission should not persist generated kernel source"
    );

    let manifest_text = fs::read_to_string(&emitted.paths.manifest_path)
        .expect("generated manifest should be readable");
    let manifest: Value =
        serde_json::from_str(&manifest_text).expect("manifest should be valid JSON");
    let artifact_key = candidate.artifact_key().hex();
    assert_eq!(
        manifest["artifact_key"].as_str(),
        Some(artifact_key.as_str())
    );
    assert_eq!(
        manifest["family"].as_str(),
        Some("gemm-f32-bf16-row-col-row")
    );
    assert_eq!(manifest["generator"].as_str(), Some("tiled-gemm-generator"));
    assert_eq!(
        manifest["materialization"]["kind"].as_str(),
        Some("generated")
    );
    assert_eq!(
        manifest["materialization"]["symbol"].as_str(),
        Some("gemm_f32_bf16_tile_16x32x16")
    );
    assert_eq!(manifest["schedule"][0]["op"].as_str(), Some("tile-gemm"));
    assert_eq!(manifest["schedule"][0]["n"].as_u64(), Some(32));
    assert_eq!(manifest["action_trace"].as_array().map(Vec::len), Some(0));
    assert_eq!(
        manifest["resources"]["threads_per_block"].as_u64(),
        Some(512)
    );
    assert_eq!(
        manifest["resources"]["shared_memory_bytes"].as_u64(),
        Some(3072)
    );
    assert_eq!(
        manifest["resources"]["accumulator_elements_per_thread"].as_u64(),
        Some(1)
    );
    assert_eq!(
        manifest["resources"]["load_elements_per_block"].as_u64(),
        Some(768)
    );

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_records_action_trace_metadata_without_changing_kernel_key() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let seed = problem.seed();
    let tile_action =
        KernelScheduleAction::tile_gemm(16, 32, 16, KernelActionMaterialization::DeferredGenerated);
    let tile_candidate = problem
        .apply_schedule_action(&seed, &tile_action)
        .expect("tile action should produce candidate metadata");
    let candidate = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::unroll(2, 4))
        .expect("unroll action should produce candidate metadata");
    let direct_candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16)).with_reduce_unroll(4),
    );

    assert_eq!(candidate.artifact_key(), direct_candidate.artifact_key());
    assert_eq!(
        candidate.action_trace,
        vec![tile_action, KernelScheduleAction::unroll(2, 4)]
    );

    let emitted = store
        .emit_metadata(&candidate)
        .expect("artifact store should write action trace metadata manifest");
    let manifest_text = fs::read_to_string(&emitted.paths.manifest_path)
        .expect("generated manifest should be readable");
    let manifest: Value =
        serde_json::from_str(&manifest_text).expect("manifest should be valid JSON");

    assert_eq!(manifest["action_trace"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        manifest["action_trace"][0]["op"].as_str(),
        Some("tile-gemm")
    );
    assert_eq!(
        manifest["action_trace"][0]["materialization"].as_str(),
        Some("deferred-generated")
    );
    assert_eq!(
        manifest["action_trace"][0]["arg"]["kind"].as_str(),
        Some("tile-3d")
    );
    assert_eq!(manifest["action_trace"][0]["arg"]["n"].as_u64(), Some(32));
    assert_eq!(manifest["action_trace"][1]["op"].as_str(), Some("unroll"));
    assert_eq!(manifest["action_trace"][1]["axis"].as_u64(), Some(2));
    assert_eq!(
        manifest["action_trace"][1]["arg"]["value"].as_u64(),
        Some(4)
    );

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_records_measured_timing_metadata() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let seed = problem.seed();
    let mut candidate = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::Existing),
        )
        .expect("row split action should produce candidate metadata");
    let samples =
        nn_rust_profiling::SampleStats::from_finite_samples(&[0.000002, 0.000003, 0.000004])
            .expect("sample stats should accept finite samples");
    let selected = nn_rust_profiling::ProfileDuration::from_seconds_f64(samples.median)
        .expect("median should be a valid duration");
    let timing = OptimizationTiming::new(
        nn_rust_profiling::ProfileTimeSource::CudaEvent,
        1,
        samples,
        selected,
    )
    .with_setup_segment(OptimizationTimingSegment::new(
        "compile-standalone-crate",
        ProfileTimeSource::WallClock,
        ProfileDuration::from_seconds_f64(0.125).expect("setup segment duration should be valid"),
    ))
    .with_setup_segment(OptimizationTimingSegment::new(
        "cleanup-compile-scratch",
        ProfileTimeSource::WallClock,
        ProfileDuration::from_seconds_f64(0.015625)
            .expect("cleanup segment duration should be valid"),
    ));
    candidate.score = SearchScore::measured_with_timing(samples.median, timing);

    let emitted = store
        .emit_metadata(&candidate)
        .expect("artifact store should write measured timing metadata manifest");
    let manifest_text = fs::read_to_string(&emitted.paths.manifest_path)
        .expect("generated manifest should be readable");
    let manifest: Value =
        serde_json::from_str(&manifest_text).expect("manifest should be valid JSON");

    assert_eq!(manifest["score"]["source"].as_str(), Some("measured"));
    assert_eq!(
        manifest["score"]["timing"]["source"].as_str(),
        Some("cuda-event")
    );
    assert_eq!(
        manifest["score"]["timing"]["warmup_count"].as_u64(),
        Some(1)
    );
    assert_eq!(
        manifest["score"]["timing"]["samples"]["count"].as_u64(),
        Some(3)
    );
    assert_eq!(
        manifest["score"]["timing"]["setup_segments"][0]["name"].as_str(),
        Some("compile-standalone-crate")
    );
    assert_eq!(
        manifest["score"]["timing"]["setup_segments"][0]["source"].as_str(),
        Some("wall-clock")
    );
    assert_eq!(
        manifest["score"]["timing"]["setup_segments"][1]["name"].as_str(),
        Some("cleanup-compile-scratch")
    );

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_writes_and_reads_candidate_score_cache() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let seed = problem.seed();
    let mut candidate = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::Existing),
        )
        .expect("row split action should produce candidate metadata");
    let samples = SampleStats::from_finite_samples(&[0.000002, 0.000003, 0.000004])
        .expect("sample stats should accept finite samples");
    let timing = OptimizationTiming::new(
        ProfileTimeSource::CudaEvent,
        1,
        samples,
        ProfileDuration::from_seconds_f64(samples.median)
            .expect("median should be a valid duration"),
    );
    let score = SearchScore::measured_with_timing(samples.median, timing)
        .expect("measured score should be finite");
    candidate.score = Some(score);

    let emitted = store
        .emit_score_cache_for_candidate("measured-cuda-event-r3-w1", &candidate)
        .expect("score cache write should succeed")
        .expect("scored candidate should produce a cache record");

    assert!(emitted.score_path.starts_with(store.root()));
    assert!(
        emitted
            .score_path
            .components()
            .any(|component| component.as_os_str() == "score-cache")
    );
    assert!(emitted.score_bytes > 0);

    let cached = store
        .read_score_cache_for_candidate("measured-cuda-event-r3-w1", &candidate)
        .expect("score cache read should succeed")
        .expect("matching score cache should return a score");

    assert_eq!(cached, score);

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_ignores_stale_candidate_score_cache() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let seed = problem.seed();
    let mut rows8 = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::Existing),
        )
        .expect("rows8 split action should produce candidate metadata");
    rows8.score = SearchScore::heuristic(8.0);
    let rows4 = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 4, KernelActionMaterialization::Existing),
        )
        .expect("rows4 split action should produce candidate metadata");

    let emitted = store
        .emit_score_cache_for_candidate("heuristic", &rows8)
        .expect("score cache write should succeed")
        .expect("scored candidate should produce a cache record");
    let stale_path = store.score_cache_path_for("heuristic", &rows4);
    fs::create_dir_all(
        stale_path
            .parent()
            .expect("score cache path should have a parent"),
    )
    .expect("stale score cache parent should be creatable");
    fs::copy(&emitted.score_path, &stale_path)
        .expect("stale score cache record should be copied into requested path");

    assert_eq!(
        store
            .read_score_cache_for_candidate("heuristic", &rows4)
            .expect("stale score cache read should not fail"),
        None
    );

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_writes_selection_metadata_and_replays_action_trace() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let mut candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::unroll(1, 8),
        ],
    )
    .expect("valid matvec action trace should replay into candidate metadata");
    let samples = SampleStats::from_finite_samples(&[0.000002, 0.000003, 0.000004])
        .expect("sample stats should accept finite samples");
    let timing = OptimizationTiming::new(
        ProfileTimeSource::CudaEvent,
        1,
        samples,
        ProfileDuration::from_seconds_f64(samples.median)
            .expect("median should be a valid duration"),
    )
    .with_setup_segment(OptimizationTimingSegment::new(
        "load-generated-module",
        ProfileTimeSource::WallClock,
        ProfileDuration::from_seconds_f64(0.03125).expect("setup segment duration should be valid"),
    ));
    candidate.score = SearchScore::measured_with_timing(samples.median, timing);

    let emitted = store
        .emit_selection_for_candidate(&candidate)
        .expect("artifact store should write selected action metadata");

    assert!(emitted.selection_path.starts_with(store.root()));
    assert!(
        emitted
            .selection_path
            .components()
            .any(|component| component.as_os_str() == "selections")
    );
    assert!(emitted.selection_bytes > 0);

    let selection_text =
        fs::read_to_string(&emitted.selection_path).expect("selection should be readable");
    assert!(selection_text.contains("\"action_trace\""));
    assert!(selection_text.contains("\"op\": \"split\""));
    assert!(selection_text.contains("\"op\": \"unroll\""));
    assert!(selection_text.contains("\"source\": \"measured\""));
    assert!(selection_text.contains("\"setup_segments\""));
    assert!(selection_text.contains("\"load-generated-module\""));
    assert!(!selection_text.contains("#[kernel]"));
    assert!(!selection_text.contains("pub fn matvec_bf16"));

    let selection = store
        .read_selection(&emitted.selection_path)
        .expect("selection should parse back from JSON");
    assert_eq!(selection.family, "matvec-bf16-row-major");
    assert_eq!(selection.artifact_key, candidate.artifact_key().hex());
    assert_eq!(selection.generator, "row-major-matvec-generator");
    assert!(!selection.launchable);
    assert_eq!(selection.action_trace, candidate.action_trace);
    assert_eq!(
        selection
            .score
            .and_then(|score| score.timing)
            .map(|timing| timing.samples.count),
        Some(3)
    );
    assert_eq!(
        selection
            .score
            .and_then(|score| score.timing)
            .and_then(|timing| timing.setup_segments[0])
            .map(|segment| segment.name),
        Some("load-generated-module")
    );

    let replayed = selection
        .replay(&problem)
        .expect("selection should replay into selected candidate");
    assert_eq!(replayed.artifact_key(), candidate.artifact_key());
    assert_eq!(replayed.launch.kernel, "matvec_bf16_rows8_u8");

    remove_test_generated_root(&root);
}

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

#[test]
fn artifact_store_writes_search_report_metadata_without_kernel_source() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let config = BeamSearchConfig {
        beam_width: 4,
        max_depth: 2,
        require_launchable: false,
    };
    let result = beam_search_metadata_with_scorer(&problem, config, |candidate| {
        let plan = schedule_matvec_plan(&candidate.schedule)?;
        let score = (8.0 - f64::from(plan.rows.rows_per_block())) * 10.0
            + (64.0 - f64::from(plan.reduce_unroll));
        SearchScore::measured(score)
    });
    let report = result.optimization_report_with_action_space(
        "matvec-bf16-row-major",
        config,
        &problem.search_space(),
    );

    let emitted = store
        .emit_search_report(&report)
        .expect("artifact store should write compact search report");

    assert!(emitted.report_path.starts_with(store.root()));
    assert!(
        emitted
            .report_path
            .components()
            .any(|component| component.as_os_str() == "search-reports")
    );
    assert!(emitted.report_bytes > 0);
    assert!(
        !emitted
            .report_path
            .with_file_name("standalone-crate")
            .exists()
    );

    let report_text =
        fs::read_to_string(&emitted.report_path).expect("search report should be readable");
    let report_json: Value =
        serde_json::from_str(&report_text).expect("search report should be valid JSON");

    assert_eq!(
        report_json["family"].as_str(),
        Some("matvec-bf16-row-major")
    );
    assert_eq!(report_json["config"]["beam_width"].as_u64(), Some(4));
    assert_eq!(
        report_json["config"]["require_launchable"].as_bool(),
        Some(false)
    );
    assert_eq!(
        report_json["duplicates"].as_u64(),
        Some(result.duplicates as u64)
    );
    assert_eq!(
        report_json["action_space"]["total_actions"].as_u64(),
        Some(problem.search_space().actions().len() as u64)
    );
    assert_eq!(
        report_json["action_space"]["spaces"][0]["op"].as_str(),
        Some("split")
    );
    assert_eq!(
        report_json["action_space"]["spaces"][1]["op"].as_str(),
        Some("upcast")
    );
    assert_eq!(
        report_json["action_space"]["spaces"][2]["op"].as_str(),
        Some("unroll")
    );
    assert_eq!(report_json["best"]["launchable"].as_bool(), Some(false));
    assert_eq!(
        report_json["best"]["materialization"].as_str(),
        Some("generated")
    );
    assert_eq!(
        report_json["best"]["launch"]["kernel"].as_str(),
        Some("matvec_bf16_rows32_u32")
    );
    assert_eq!(
        report_json["best"]["action_trace"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(
        report_json["best"]["action_trace"][1]["op"].as_str(),
        Some("unroll")
    );
    assert_eq!(
        report_json["best"]["score"]["source"].as_str(),
        Some("measured")
    );
    assert!(
        report_json["beam"]
            .as_array()
            .is_some_and(|beam| !beam.is_empty())
    );
    assert!(!report_text.contains("#[kernel]"));
    assert!(!report_text.contains("pub fn matvec_bf16"));
    assert!(!report_text.contains("pub struct Bf16"));

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_writes_auto_search_report_with_step_metadata() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };
    let result = auto_optimize_metadata_with_scorer(&problem, config, |candidate| {
        let plan = schedule_matvec_plan(&candidate.schedule)?;
        if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
            SearchScore::heuristic(1.0)
        } else {
            SearchScore::heuristic(10.0)
        }
    });
    let report = result.auto_optimization_report_with_action_space(
        "matvec-bf16-row-major",
        config,
        &problem.search_space(),
    );

    let emitted = store
        .emit_auto_search_report(&report)
        .expect("artifact store should write compact auto-search report");

    assert!(emitted.report_path.starts_with(store.root()));
    assert!(
        emitted
            .report_path
            .components()
            .any(|component| component.as_os_str() == "auto-search-reports")
    );
    assert!(emitted.report_bytes > 0);

    let report_text =
        fs::read_to_string(&emitted.report_path).expect("auto-search report should be readable");
    let report_json: Value =
        serde_json::from_str(&report_text).expect("auto-search report should be valid JSON");

    assert_eq!(
        report_json["family"].as_str(),
        Some("matvec-bf16-row-major")
    );
    assert_eq!(report_json["config"]["beam_width"].as_u64(), Some(4));
    assert_eq!(report_json["config"]["max_steps"].as_u64(), Some(2));
    assert_eq!(
        report_json["duplicates"].as_u64(),
        Some(result.duplicates as u64)
    );
    assert_eq!(
        report_json["action_space"]["total_actions"].as_u64(),
        Some(problem.search_space().actions().len() as u64)
    );
    assert_eq!(
        report_json["action_space"]["spaces"][0]["variants"][0]["materialization"].as_str(),
        Some("existing")
    );
    assert_eq!(
        report_json["exit_reason"]["label"].as_str(),
        Some("no-improvement")
    );
    assert!(
        report_json["steps"]
            .as_array()
            .is_some_and(|steps| !steps.is_empty())
    );
    let steps = report_json["steps"]
        .as_array()
        .expect("steps should be serialized as an array");
    assert!(
        steps
            .iter()
            .any(|step| step["best_before"].is_object() && step["best_after"].is_object())
    );
    assert!(
        steps
            .iter()
            .all(|step| step["duplicates"].as_u64().is_some())
    );
    assert!(steps.iter().any(|step| {
        step["best_candidate"]["action_trace"]
            .as_array()
            .is_some_and(|actions| !actions.is_empty())
    }));
    assert_eq!(
        report_json["best"]["action_trace"][0]["op"].as_str(),
        Some("split")
    );
    assert!(!report_text.contains("#[kernel]"));
    assert!(!report_text.contains("pub fn matvec_bf16"));
    assert!(!report_text.contains("pub struct Bf16"));

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_records_b_load_stride_order_metadata() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
            .with_b_load_order(GemmBTileLoadOrder::KContiguous),
    );

    let emitted = store
        .emit_metadata(&candidate)
        .expect("artifact store should write stride-order metadata manifest");
    let manifest_text = fs::read_to_string(&emitted.paths.manifest_path)
        .expect("generated manifest should be readable");
    let manifest: Value =
        serde_json::from_str(&manifest_text).expect("manifest should be valid JSON");

    assert_eq!(
        manifest["materialization"]["symbol"].as_str(),
        Some("gemm_f32_bf16_tile_16x32x16_bk")
    );
    assert_eq!(manifest["schedule"][1]["op"].as_str(), Some("stride-order"));
    assert_eq!(manifest["schedule"][1]["axes"][0].as_u64(), Some(2));
    assert_eq!(manifest["schedule"][1]["axes"][1].as_u64(), Some(1));

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_writes_standalone_crate_for_generated_matvec_kernel() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_rows(RowMajorWarpRows::Rows8);

    let emitted = store
        .emit_standalone_crate(&candidate, &MatvecRustCudaGenerator)
        .expect("artifact store should write standalone generated matvec kernel crate");

    assert_eq!(emitted.artifact_key, candidate.artifact_key());
    assert_eq!(emitted.symbol, "matvec_bf16_rows8");
    assert!(emitted.paths.crate_dir.starts_with(store.root()));
    assert!(emitted.paths.cargo_toml_path.starts_with(store.root()));
    assert!(emitted.paths.source_path.starts_with(store.root()));

    let source = fs::read_to_string(&emitted.paths.source_path)
        .expect("standalone matvec main.rs should be readable");
    assert!(source.contains("pub fn matvec_bf16_rows8("));
    assert!(source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
    assert!(source.contains("fn main() {}"));

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_writes_standalone_crate_to_scratch_dir_without_persistent_source() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_rows(RowMajorWarpRows::Rows8);
    let persistent_paths = store.standalone_crate_paths_for(&candidate);
    let scratch_root = root
        .join("compile-scratch")
        .join(candidate.artifact_key().hex());
    let scratch_crate_dir = scratch_root.join("standalone-crate");

    let emitted = store
        .emit_standalone_crate_to_dir(&candidate, &MatvecRustCudaGenerator, &scratch_crate_dir)
        .expect("artifact store should write scratch standalone generated matvec crate");

    assert_eq!(emitted.artifact_key, candidate.artifact_key());
    assert_eq!(emitted.symbol, "matvec_bf16_rows8");
    assert_eq!(emitted.paths.crate_dir, scratch_crate_dir);
    assert!(emitted.paths.cargo_toml_path.starts_with(&scratch_root));
    assert!(emitted.paths.source_path.starts_with(&scratch_root));
    assert!(!persistent_paths.crate_dir.exists());
    assert!(!persistent_paths.cargo_toml_path.exists());
    assert!(!persistent_paths.source_path.exists());

    let source = fs::read_to_string(&emitted.paths.source_path)
        .expect("scratch standalone matvec main.rs should be readable");
    assert!(source.contains("pub fn matvec_bf16_rows8("));

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_removes_stale_compile_scratch_root() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let stale_source_path = store
        .compile_scratch_root()
        .join("stale-candidate")
        .join("standalone-crate")
        .join("src")
        .join("main.rs");
    fs::create_dir_all(
        stale_source_path
            .parent()
            .expect("stale scratch path should have a parent"),
    )
    .expect("stale scratch directory should be creatable");
    fs::write(&stale_source_path, "fn main() {}\n")
        .expect("stale scratch source should be writable");

    store
        .remove_compile_scratch()
        .expect("stale compile scratch root should be removable");

    assert!(!store.compile_scratch_root().exists());
    assert_eq!(
        store.standalone_target_root(),
        root.join("standalone-target")
    );
    assert!(root.exists());

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_compile_scratch_cleanup_is_idempotent() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);

    store
        .remove_compile_scratch()
        .expect("missing compile scratch root should be accepted");

    fs::create_dir_all(&root).expect("test root should be creatable");
    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_writes_standalone_crate_for_generated_kernel() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));

    let emitted = store
        .emit_standalone_crate(&candidate, &GemmRustCudaGenerator)
        .expect("artifact store should write standalone generated kernel crate");

    assert_eq!(emitted.artifact_key, candidate.artifact_key());
    assert_eq!(emitted.package_name, "nn_rust_kernel_172b6682003af067");
    assert_eq!(emitted.symbol, "gemm_f32_bf16_tile_16x32x16");
    assert!(emitted.paths.crate_dir.starts_with(store.root()));
    assert!(emitted.paths.cargo_toml_path.starts_with(store.root()));
    assert!(emitted.paths.source_path.starts_with(store.root()));

    let cargo_toml = fs::read_to_string(&emitted.paths.cargo_toml_path)
        .expect("standalone Cargo.toml should be readable");
    assert!(cargo_toml.contains("name = \"nn_rust_kernel_172b6682003af067\""));
    assert!(cargo_toml.contains("cuda-device"));
    assert!(cargo_toml.contains("cuda-host"));

    let source = fs::read_to_string(&emitted.paths.source_path)
        .expect("standalone main.rs should be readable");
    assert!(source.contains("pub fn gemm_f32_bf16_tile_16x32x16("));
    assert!(source.contains("pub struct Bf16(u16);"));
    assert!(source.contains("fn main() {}"));

    remove_test_generated_root(&root);
}

#[test]
fn standalone_manifest_can_render_cuda_oxide_path_dependencies() {
    let mut manifest = String::new();

    write_cuda_oxide_dependency(
        &mut manifest,
        "cuda-device",
        Some(Path::new("/tmp/cuda oxide/root")),
    );
    write_cuda_oxide_dependency(&mut manifest, "cuda-host", None);

    assert!(
        manifest.contains("cuda-device = { path = \"/tmp/cuda oxide/root/crates/cuda-device\" }")
    );
    assert!(manifest.contains(
        "cuda-host = { git = \"https://github.com/NVlabs/cuda-oxide.git\", tag = \"v0.1.0\" }"
    ));
}

#[test]
fn cuda_oxide_checkout_detection_requires_kernel_crates() {
    let root = test_generated_root();
    let checkout = root.join("cuda-oxide");
    fs::create_dir_all(checkout.join("crates").join("cuda-device"))
        .expect("cuda-device directory should be creatable");
    fs::write(
        checkout
            .join("crates")
            .join("cuda-device")
            .join("Cargo.toml"),
        "[package]\nname = \"cuda-device\"\n",
    )
    .expect("cuda-device manifest should be writable");

    assert!(!cuda_oxide_checkout_has_kernel_crates(&checkout));

    fs::create_dir_all(checkout.join("crates").join("cuda-host"))
        .expect("cuda-host directory should be creatable");
    fs::write(
        checkout.join("crates").join("cuda-host").join("Cargo.toml"),
        "[package]\nname = \"cuda-host\"\n",
    )
    .expect("cuda-host manifest should be writable");

    assert!(cuda_oxide_checkout_has_kernel_crates(&checkout));

    remove_test_generated_root(&root);
}
