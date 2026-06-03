use super::super::*;

#[test]
fn gemm_generator_renders_tile_specific_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_tile(GemmTileShape::new(13, 24, 13));
    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render deferred tile source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_13x24x13");
    assert!(generated.source.contains("#[kernel]"));
    assert!(
        generated
            .source
            .contains("pub fn gemm_f32_bf16_tile_13x24x13(")
    );
    assert!(generated.source.contains("pub struct Bf16(u16);"));
    assert!(generated.source.contains(".to_f32()"));
    assert!(generated.source.contains("const TILE_M: usize = 13;"));
    assert!(generated.source.contains("const TILE_N: usize = 24;"));
    assert!(generated.source.contains("const TILE_K: usize = 13;"));
    assert!(generated.source.contains("while load < TILE_A_ELEMS"));
    assert!(generated.source.contains("while load < TILE_B_ELEMS"));
}
