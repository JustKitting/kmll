use super::super::*;

#[test]
fn gemm_generator_renders_thread_axis_swap_source_on_demand() {
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
            KernelScheduleAction::swap(0, 1),
        ],
    )
    .expect("valid swapped GEMM action trace should replay");
    let plan = schedule_gemm_plan(&candidate.schedule).expect("swapped candidate should plan");
    assert_eq!(plan.thread_order, GemmThreadOrder::MThenN);
    assert_eq!(candidate.launch.block_dim.x, 16);
    assert_eq!(candidate.launch.block_dim.y, 32);

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render thread-axis-swapped source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_sw01");
    assert!(
        generated
            .source
            .contains("let thread_m = thread::threadIdx_x() as usize;")
    );
    assert!(
        generated
            .source
            .contains("let thread_n = thread::threadIdx_y() as usize;")
    );
    assert!(
        generated
            .source
            .contains("if thread_m >= THREADS_M || thread_n >= THREADS_N")
    );
    assert!(
        generated
            .source
            .contains("let tid = thread_n * THREADS_M + thread_m;")
    );
    assert!(
        generated
            .source
            .contains("let tile_row0 = thread_m * THREAD_TILE_M;")
    );
    assert!(
        generated
            .source
            .contains("let tile_col0 = thread_n * THREAD_TILE_N;")
    );
}
