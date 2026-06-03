use super::super::*;

#[test]
fn gemm_generator_renders_a_load_stride_order_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
            .with_a_load_order(GemmATileLoadOrder::MContiguous),
    );
    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render A stride-order source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_am");
    assert!(generated.source.contains("let tile_row = load % TILE_M;"));
    assert!(generated.source.contains("let tile_col = load / TILE_M;"));
    assert!(
        generated
            .source
            .contains("let a_smem_index = tile_row * TILE_K + tile_col;")
    );
    assert!(generated.source.contains("TILE_A[a_smem_index] = if"));
}

#[test]
fn gemm_generator_renders_b_load_stride_order_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
            .with_b_load_order(GemmBTileLoadOrder::KContiguous),
    );
    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render stride-order source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_bk");
    assert!(generated.source.contains("let tile_row = load % TILE_K;"));
    assert!(generated.source.contains("let tile_col = load / TILE_K;"));
    assert!(
        generated
            .source
            .contains("let b_smem_index = tile_row * TILE_N + tile_col;")
    );
    assert!(generated.source.contains("TILE_B[b_smem_index] = if"));
}
