use super::*;

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
    let score_text =
        fs::read_to_string(&emitted.score_path).expect("score cache should be readable");
    assert!(score_text.contains("\"materialization\""));
    assert!(score_text.contains("\"kind\": \"existing\""));
    assert!(score_text.contains("\"symbol\": \"matvec_bf16_kernel\""));

    let cached = store
        .read_score_cache_for_candidate("measured-cuda-event-r3-w1", &candidate)
        .expect("score cache read should succeed")
        .expect("matching score cache should return a score");

    assert_eq!(cached, score);

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_reads_legacy_candidate_score_cache_without_materialization() {
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
    candidate.score = SearchScore::heuristic(8.0);
    let emitted = store
        .emit_score_cache_for_candidate("heuristic", &candidate)
        .expect("score cache write should succeed")
        .expect("scored candidate should produce a cache record");
    let score_text =
        fs::read_to_string(&emitted.score_path).expect("score cache should be readable");
    let mut score_json: Value =
        serde_json::from_str(&score_text).expect("score cache should be valid JSON");
    score_json
        .as_object_mut()
        .expect("score cache should be a JSON object")
        .remove("materialization");
    fs::write(
        &emitted.score_path,
        serde_json::to_vec_pretty(&score_json).expect("legacy score cache should serialize"),
    )
    .expect("legacy score cache should be writable");

    let cached = store
        .read_score_cache_for_candidate("heuristic", &candidate)
        .expect("legacy score cache read should not fail")
        .expect("legacy score cache should still match candidate");
    assert_eq!(
        cached,
        SearchScore::heuristic(8.0).expect("score should be finite")
    );

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
fn artifact_store_ignores_stale_candidate_score_cache_materialization() {
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
    candidate.score = SearchScore::heuristic(8.0);
    let emitted = store
        .emit_score_cache_for_candidate("heuristic", &candidate)
        .expect("score cache write should succeed")
        .expect("scored candidate should produce a cache record");
    let score_text =
        fs::read_to_string(&emitted.score_path).expect("score cache should be readable");
    let mut score_json: Value =
        serde_json::from_str(&score_text).expect("score cache should be valid JSON");
    score_json["materialization"] = json!({
        "kind": "generated",
        "symbol": "matvec_bf16_rows8",
    });
    fs::write(
        &emitted.score_path,
        serde_json::to_vec_pretty(&score_json)
            .expect("stale materialization score cache should serialize"),
    )
    .expect("stale materialization score cache should be writable");

    assert_eq!(
        store
            .read_score_cache_for_candidate("heuristic", &candidate)
            .expect("stale materialization score cache read should not fail"),
        None
    );

    remove_test_generated_root(&root);
}
