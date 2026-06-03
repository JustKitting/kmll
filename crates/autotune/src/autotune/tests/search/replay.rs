use super::*;

#[test]
fn candidate_projects_to_profiling_optimization_spec() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let mut candidate = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("row split action should produce candidate metadata");
    candidate.score = SearchScore::heuristic(1.0);

    let spec = candidate.optimization_spec();

    assert_eq!(spec.family, "matvec-bf16-row-major");
    assert_eq!(spec.artifact_key, candidate.artifact_key().hex());
    assert_eq!(spec.generator, "row-major-matvec-generator");
    assert!(!spec.launchable);
    assert_eq!(
        spec.materialization,
        ProfilingCandidateMaterialization::Generated
    );
    assert_eq!(spec.launch.kernel, "matvec_bf16_rows8");
    assert_eq!(spec.operation.kind, OperationKind::Matvec);
    assert_eq!(spec.action_trace, candidate.action_trace);
    assert_eq!(spec.score, candidate.score);
}

#[test]
fn optimization_candidate_spec_replay_checks_artifact_key() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::unroll(1, 8),
        ],
    )
    .expect("valid matvec action trace should replay into candidate metadata");
    let spec = candidate.optimization_spec();

    let replayed = replay_optimization_candidate_spec(&problem, &spec)
        .expect("matching optimization spec should replay");
    assert_eq!(replayed.artifact_key(), candidate.artifact_key());

    let mut stale_spec = spec;
    stale_spec.artifact_key = "0000000000000000".to_string();
    assert_eq!(
        replay_optimization_candidate_spec(&problem, &stale_spec),
        Err(KernelActionReplayError::ArtifactKeyMismatch {
            expected: "0000000000000000".to_string(),
            actual: candidate.artifact_key().hex(),
        })
    );
}

#[test]
fn action_trace_replay_rejects_invalid_action_sequence() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let action = KernelScheduleAction::unroll(1, 8);

    assert_eq!(
        replay_schedule_actions(&problem, std::slice::from_ref(&action)),
        Err(KernelActionReplayError::InvalidAction { index: 0, action })
    );
}
