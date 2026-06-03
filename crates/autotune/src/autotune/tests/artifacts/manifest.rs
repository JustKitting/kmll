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
