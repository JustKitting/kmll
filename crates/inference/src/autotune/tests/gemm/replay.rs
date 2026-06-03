use super::*;

#[test]
fn action_trace_replay_reconstructs_generated_gemm_candidate() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let actions = vec![
        KernelScheduleAction::tile_gemm(16, 32, 16, KernelActionMaterialization::DeferredGenerated),
        KernelScheduleAction::unroll(2, 4),
        KernelScheduleAction::stride_order(vec![2, 1]),
    ];

    let candidate = replay_schedule_actions(&problem, &actions)
        .expect("valid GEMM action trace should replay into candidate metadata");
    let plan = schedule_gemm_plan(&candidate.schedule).expect("replayed GEMM should have plan");

    assert_eq!(candidate.family, "gemm-f32-bf16-row-col-row");
    assert_eq!(candidate.action_trace, actions);
    assert_eq!(candidate.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u4_bk");
    assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
    assert_eq!(plan.reduce_unroll, 4);
    assert_eq!(plan.b_load_order, GemmBTileLoadOrder::KContiguous);
    assert!(!candidate.is_launchable());

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("replayed generated GEMM candidate should render source on demand");
    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_u4_bk");
    assert!(generated.source.contains("const TILE_M: usize = 16;"));
    assert!(generated.source.contains("const TILE_N: usize = 32;"));
    assert!(generated.source.contains("const REDUCE_UNROLL: usize = 4;"));
    assert!(generated.source.contains("let tile_row = load % TILE_K;"));
}

#[test]
fn action_trace_replay_reconstructs_2d_upcast_gemm_candidate() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let actions = vec![
        KernelScheduleAction::tile_gemm(16, 32, 16, KernelActionMaterialization::DeferredGenerated),
        KernelScheduleAction::upcast(0, 2),
        KernelScheduleAction::upcast(1, 2),
    ];

    let candidate = replay_schedule_actions(&problem, &actions)
        .expect("valid GEMM 2D upcast trace should replay into candidate metadata");
    let plan = schedule_gemm_plan(&candidate.schedule).expect("upcast GEMM should have plan");

    assert_eq!(candidate.family, "gemm-f32-bf16-row-col-row");
    assert_eq!(candidate.action_trace, actions);
    assert_eq!(
        candidate.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2"
    );
    assert_eq!(candidate.launch.block_dim.x, 16);
    assert_eq!(candidate.launch.block_dim.y, 8);
    assert_eq!(candidate.launch.block_dim.z, 1);
    assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
    assert_eq!(plan.m_per_thread, 2);
    assert_eq!(plan.n_per_thread, 2);
    assert!(candidate.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::Upcast { axis: 0, factor: 2 })
    }));
    assert!(candidate.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::Upcast { axis: 1, factor: 2 })
    }));

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("2D upcast generated GEMM candidate should render source");
    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_mt2_nt2");
    assert!(generated.source.contains("const THREAD_TILE_M: usize = 2;"));
    assert!(generated.source.contains("const THREAD_TILE_N: usize = 2;"));
    assert!(
        generated
            .source
            .contains("let tile_row1 = thread_m * THREAD_TILE_M + 1;")
    );
    assert!(
        generated
            .source
            .contains("let tile_col1 = thread_n * THREAD_TILE_N + 1;")
    );
    assert!(generated.source.contains("let mut acc3 = 0.0_f32;"));
    assert!(generated.source.contains("TILE_A[tile_row1 * TILE_K + kk]"));
    assert!(generated.source.contains("TILE_B[kk * TILE_N + tile_col1]"));
    assert!(
        generated
            .source
            .contains("*c_elem = alpha * acc3 + beta * current;")
    );
}

#[test]
fn gemm_describes_generated_tiles_without_storing_kernel_payloads() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let actions = vec![
        KernelScheduleAction::split(0, 13, KernelActionMaterialization::DeferredGenerated),
        KernelScheduleAction::split(1, 24, KernelActionMaterialization::DeferredGenerated),
        KernelScheduleAction::split(2, 13, KernelActionMaterialization::DeferredGenerated),
    ];
    let generated = replay_schedule_actions(&problem, &actions)
        .expect("GEMM split trace should expose arbitrary generated tile metadata");
    assert_eq!(
        schedule_gemm_tile(&generated.schedule),
        Some(GemmTileShape::new(13, 24, 13))
    );
    assert!(!generated.is_launchable());
    assert_eq!(generated.launch.kernel, "gemm_f32_bf16_tile_13x24x13");
    assert_eq!(
        generated.generated.materialization,
        KernelMaterialization::Generated {
            symbol: "gemm_f32_bf16_tile_13x24x13".to_string()
        }
    );
    assert_eq!(generated.action_trace, actions);
    assert_eq!(generated.generated.generator, "tiled-gemm-generator");
}
