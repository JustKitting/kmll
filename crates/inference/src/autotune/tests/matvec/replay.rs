use super::*;

#[test]
fn action_trace_replay_reconstructs_generated_matvec_candidate() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let actions = vec![
        KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
        KernelScheduleAction::unroll(1, 8),
    ];

    let candidate = replay_schedule_actions(&problem, &actions)
        .expect("valid matvec action trace should replay into candidate metadata");

    assert_eq!(candidate.family, "matvec-bf16-row-major");
    assert_eq!(candidate.action_trace, actions);
    assert_eq!(candidate.launch.kernel, "matvec_bf16_rows8_u8");
    assert_eq!(schedule_rows_per_block(&candidate.schedule), Some(8));
    assert_eq!(schedule_matvec_reduce_unroll(&candidate.schedule), Some(8));
    assert!(!candidate.is_launchable());

    let generated = MatvecRustCudaGenerator
        .source_for(&candidate)
        .expect("replayed generated candidate should render source on demand");
    assert_eq!(generated.symbol, "matvec_bf16_rows8_u8");
    assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
    assert!(generated.source.contains("const REDUCE_UNROLL: u32 = 8;"));
}
