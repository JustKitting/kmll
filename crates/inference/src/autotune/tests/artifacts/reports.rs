use super::*;

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
