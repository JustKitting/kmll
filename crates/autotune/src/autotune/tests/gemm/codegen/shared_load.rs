use super::super::*;

#[test]
fn gemm_generator_renders_shared_load_unroll_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::tile_gemm(
                16,
                32,
                16,
                KernelActionMaterialization::DeferredGenerated,
            ),
            KernelScheduleAction::upcast(0, 2),
            KernelScheduleAction::upcast(1, 2),
            KernelScheduleAction::unroll(3, 2),
            KernelScheduleAction::unroll(4, 2),
        ],
    )
    .expect("valid shared-load-unroll GEMM action trace should replay");
    let plan =
        schedule_gemm_plan(&candidate.schedule).expect("load-unrolled candidate should plan");
    assert_eq!(plan.m_per_thread, 2);
    assert_eq!(plan.n_per_thread, 2);
    assert_eq!(plan.a_load_unroll, 2);
    assert_eq!(plan.b_load_unroll, 2);

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render shared-load-unrolled source");

    assert_eq!(
        generated.symbol,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_au2_bu2"
    );
    assert!(generated.source.contains("const A_LOAD_UNROLL: usize = 2;"));
    assert!(generated.source.contains("const B_LOAD_UNROLL: usize = 2;"));
    assert!(
        generated
            .source
            .contains("let a_load1 = load + A_LOAD_THREADS;")
    );
    assert!(generated.source.contains("if a_load1 < TILE_A_ELEMS"));
    assert!(
        generated
            .source
            .contains("load += A_LOAD_THREADS * A_LOAD_UNROLL;")
    );
    assert!(
        generated
            .source
            .contains("let b_load1 = load + B_LOAD_THREADS;")
    );
    assert!(generated.source.contains("if b_load1 < TILE_B_ELEMS"));
    assert!(
        generated
            .source
            .contains("load += B_LOAD_THREADS * B_LOAD_UNROLL;")
    );
}

#[test]
fn gemm_generator_renders_shared_load_group_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::tile_gemm(
                16,
                32,
                16,
                KernelActionMaterialization::DeferredGenerated,
            ),
            KernelScheduleAction::upcast(0, 2),
            KernelScheduleAction::upcast(1, 2),
            KernelScheduleAction::group(3, 32),
            KernelScheduleAction::group(4, 64),
        ],
    )
    .expect("valid shared-load-group GEMM action trace should replay");
    let plan = schedule_gemm_plan(&candidate.schedule).expect("load-grouped candidate should plan");
    assert_eq!(plan.a_load_thread_count(), 32);
    assert_eq!(plan.b_load_thread_count(), 64);

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render shared-load-grouped source");

    assert_eq!(
        generated.symbol,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_atg32_btg64"
    );
    assert!(
        generated
            .source
            .contains("const A_LOAD_THREADS: usize = 32;")
    );
    assert!(
        generated
            .source
            .contains("const B_LOAD_THREADS: usize = 64;")
    );
    assert!(
        generated
            .source
            .contains("let mut load = if tid < A_LOAD_THREADS { tid } else { TILE_A_ELEMS };")
    );
    assert!(
        generated
            .source
            .contains("load = if tid < B_LOAD_THREADS { tid } else { TILE_B_ELEMS };")
    );
    assert!(generated.source.contains("load += A_LOAD_THREADS;"));
    assert!(generated.source.contains("load += B_LOAD_THREADS;"));
}
