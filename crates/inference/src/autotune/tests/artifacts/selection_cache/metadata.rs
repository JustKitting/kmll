use super::*;

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
    assert!(selection_text.contains("\"materialization\""));
    assert!(selection_text.contains("\"kind\": \"generated\""));
    assert!(selection_text.contains("\"symbol\": \"matvec_bf16_rows8_u8\""));
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
    assert_eq!(
        selection.materialization,
        Some(KernelMaterializationDescriptor::Generated {
            symbol: "matvec_bf16_rows8_u8".to_string()
        })
    );
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
fn artifact_store_reads_legacy_selection_without_materialization() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(128, 256);
    let candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::unroll(1, 8),
        ],
    )
    .expect("valid matvec action trace should replay into candidate metadata");
    let emitted = store
        .emit_selection_for_candidate(&candidate)
        .expect("artifact store should write selected action metadata");
    let selection_text =
        fs::read_to_string(&emitted.selection_path).expect("selection should be readable");
    let mut selection_json: Value =
        serde_json::from_str(&selection_text).expect("selection should be valid JSON");
    selection_json
        .as_object_mut()
        .expect("selection should be a JSON object")
        .remove("materialization");
    fs::write(
        &emitted.selection_path,
        serde_json::to_vec_pretty(&selection_json).expect("legacy selection should serialize"),
    )
    .expect("legacy selection should be writable");

    let selection = store
        .read_selection(&emitted.selection_path)
        .expect("legacy selection without materialization should parse");
    assert_eq!(selection.materialization, None);
    let replayed = selection
        .replay(&problem)
        .expect("legacy selection should replay using action trace and artifact key");
    assert_eq!(replayed.artifact_key(), candidate.artifact_key());

    remove_test_generated_root(&root);
}
