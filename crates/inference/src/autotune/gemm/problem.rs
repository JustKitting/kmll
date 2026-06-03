use super::*;

mod actions;
mod candidate;
mod factors;
mod metadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmSearchProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_dtype: NumericKind,
    pub b_dtype: NumericKind,
    pub c_dtype: NumericKind,
    pub accumulator: NumericKind,
}

impl GemmSearchProblem {
    pub(in crate::autotune) const EXISTING_TILE: GemmTileShape = GemmTileShape::new(16, 16, 16);
    const MAX_TILE_DIM: u32 = 32;
    const MAX_THREADS_PER_BLOCK: u32 = 1024;
    const MAX_SHARED_MEMORY_BYTES: u32 = 48 * 1024;
    const MAX_ACCUMULATOR_ELEMENTS_PER_THREAD: u32 = 16;
    const MAX_REDUCE_UNROLL_FACTOR: u32 = 32;
    const MAX_LOAD_UNROLL_FACTOR: u32 = 4;

    pub const fn f32_bf16_row_col_row(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            a_dtype: NumericKind::F32,
            b_dtype: NumericKind::Bf16,
            c_dtype: NumericKind::F32,
            accumulator: NumericKind::F32,
        }
    }
}
