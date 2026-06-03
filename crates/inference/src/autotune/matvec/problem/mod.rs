use super::*;

mod actions;
mod candidate;
mod factors;
mod metadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatvecSearchProblem {
    pub rows: usize,
    pub cols: usize,
    pub input_dtype: NumericKind,
    pub weight_dtype: NumericKind,
    pub accumulator: NumericKind,
}

impl MatvecSearchProblem {
    const MAX_ROWS_PER_BLOCK: u32 = MatvecRowSplit::MAX_ROWS_PER_BLOCK;
    const MAX_REDUCE_UNROLL_FACTOR: u32 = 32;

    pub const fn bf16_row_major(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            input_dtype: NumericKind::F32,
            weight_dtype: NumericKind::Bf16,
            accumulator: NumericKind::F32,
        }
    }
}
