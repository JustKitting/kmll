use super::super::*;

#[test]
fn gemm_generator_renders_reduce_unroll_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16)).with_reduce_unroll(7),
    );
    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render reduce-unrolled source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_u7");
    assert!(generated.source.contains("const REDUCE_UNROLL: usize = 7;"));
    assert!(
        generated
            .source
            .contains("while kk + REDUCE_UNROLL <= TILE_K")
    );
    assert!(
        generated
            .source
            .contains("TILE_A[tile_row0 * TILE_K + kk + 6]")
    );
    assert!(
        generated
            .source
            .contains("TILE_B[(kk + 6) * TILE_N + tile_col0]")
    );
    assert!(generated.source.contains("while kk < TILE_K"));
}

#[test]
fn gemm_generator_renders_reduce_group_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16)).with_reduce_group(13),
    );
    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render reduce-grouped source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_kg13");
    assert!(generated.source.contains("const REDUCE_GROUP: usize = 13;"));
    assert!(generated.source.contains("let mut kk_group = 0;"));
    assert!(generated.source.contains("while kk_group < TILE_K"));
    assert!(
        generated
            .source
            .contains("while kk + REDUCE_UNROLL <= kk_group_end")
    );
    assert!(generated.source.contains("while kk < kk_group_end"));
    assert!(generated.source.contains("kk_group += REDUCE_GROUP;"));
}
