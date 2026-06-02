use std::{fmt, marker::PhantomData};

use cuda_core::DeviceBuffer;

use crate::{
    backends::Cuda,
    dtypes::{AccumulatorWith, DeviceFloat, DeviceStorageElement, TensorElement},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape2D {
    pub rows: usize,
    pub cols: usize,
}

impl Shape2D {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub fn element_count(self) -> usize {
        self.rows
            .checked_mul(self.cols)
            .expect("matrix shape element count overflow")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stride2D {
    pub row: usize,
    pub col: usize,
}

impl Stride2D {
    pub const fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape3D {
    pub outer: usize,
    pub middle: usize,
    pub inner: usize,
}

impl Shape3D {
    pub const fn new(outer: usize, middle: usize, inner: usize) -> Self {
        Self {
            outer,
            middle,
            inner,
        }
    }

    pub fn element_count(self) -> usize {
        self.outer
            .checked_mul(self.middle)
            .and_then(|n| n.checked_mul(self.inner))
            .expect("tensor shape element count overflow")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stride3D {
    pub outer: usize,
    pub middle: usize,
    pub inner: usize,
}

impl Stride3D {
    pub const fn new(outer: usize, middle: usize, inner: usize) -> Self {
        Self {
            outer,
            middle,
            inner,
        }
    }
}

pub trait Layout2D: Copy + fmt::Debug + 'static {
    const NAME: &'static str;

    fn packed_stride(shape: Shape2D) -> Stride2D;
}

pub trait Layout3D: Copy + fmt::Debug + 'static {
    const NAME: &'static str;

    fn packed_stride(shape: Shape3D) -> Stride3D;
}

pub trait MatvecTile: Copy + fmt::Debug + 'static {
    const LANES_PER_ROW: u32;
    const ROWS_PER_BLOCK: u32;

    fn block_threads() -> u32 {
        Self::LANES_PER_ROW
            .checked_mul(Self::ROWS_PER_BLOCK)
            .expect("matvec tile thread count overflow")
    }

    fn grid_rows(rows: usize) -> u32 {
        let rows = rows as u32;
        rows.div_ceil(Self::ROWS_PER_BLOCK)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WarpRowTile;

impl MatvecTile for WarpRowTile {
    const LANES_PER_ROW: u32 = 32;
    const ROWS_PER_BLOCK: u32 = 1;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WarpRows4Tile;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WarpRows2Tile;

impl MatvecTile for WarpRows2Tile {
    const LANES_PER_ROW: u32 = 32;
    const ROWS_PER_BLOCK: u32 = 2;
}

impl MatvecTile for WarpRows4Tile {
    const LANES_PER_ROW: u32 = 32;
    const ROWS_PER_BLOCK: u32 = 4;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WarpRows8Tile;

impl MatvecTile for WarpRows8Tile {
    const LANES_PER_ROW: u32 = 32;
    const ROWS_PER_BLOCK: u32 = 8;
}

pub trait MatvecKernelPlan: Copy + fmt::Debug + 'static {
    type Layout: Layout2D;
    type Tile: MatvecTile;

    const NAME: &'static str;

    fn grid_rows(rows: usize) -> u32 {
        Self::Tile::grid_rows(rows)
    }

    fn block_threads() -> u32 {
        Self::Tile::block_threads()
    }

    fn rows_per_block() -> u32 {
        Self::Tile::ROWS_PER_BLOCK
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowMajorWarpRowMatvecPlan;

impl MatvecKernelPlan for RowMajorWarpRowMatvecPlan {
    type Layout = RowMajor;
    type Tile = WarpRowTile;

    const NAME: &'static str = "row-major-warp-row-matvec";
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowMajorWarpRows2MatvecPlan;

impl MatvecKernelPlan for RowMajorWarpRows2MatvecPlan {
    type Layout = RowMajor;
    type Tile = WarpRows2Tile;

    const NAME: &'static str = "row-major-warp-rows2-matvec";
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowMajorWarpRows4MatvecPlan;

impl MatvecKernelPlan for RowMajorWarpRows4MatvecPlan {
    type Layout = RowMajor;
    type Tile = WarpRows4Tile;

    const NAME: &'static str = "row-major-warp-rows4-matvec";
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowMajorWarpRows8MatvecPlan;

impl MatvecKernelPlan for RowMajorWarpRows8MatvecPlan {
    type Layout = RowMajor;
    type Tile = WarpRows8Tile;

    const NAME: &'static str = "row-major-warp-rows8-matvec";
}

pub trait GemmTile: Copy + fmt::Debug + 'static {
    const M: u32;
    const N: u32;
    const K: u32;

    fn block_dim() -> (u32, u32, u32) {
        (Self::N, Self::M, 1)
    }

    fn grid_dim(m: usize, n: usize) -> (u32, u32, u32) {
        let m = m as u32;
        let n = n as u32;
        (n.div_ceil(Self::N), m.div_ceil(Self::M), 1)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tile16x16x16;

impl GemmTile for Tile16x16x16 {
    const M: u32 = 16;
    const N: u32 = 16;
    const K: u32 = 16;
}

pub trait GemmKernelPlan: Copy + fmt::Debug + 'static {
    type ALayout: Layout2D;
    type BLayout: Layout2D;
    type CLayout: Layout2D;
    type Tile: GemmTile;

    const NAME: &'static str;

    fn block_dim() -> (u32, u32, u32) {
        Self::Tile::block_dim()
    }

    fn grid_dim(m: usize, n: usize) -> (u32, u32, u32) {
        Self::Tile::grid_dim(m, n)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TiledGemm16Plan<ALayout = RowMajor, BLayout = RowMajor, CLayout = RowMajor> {
    _layouts: PhantomData<(ALayout, BLayout, CLayout)>,
}

impl<ALayout, BLayout, CLayout> GemmKernelPlan for TiledGemm16Plan<ALayout, BLayout, CLayout>
where
    ALayout: Layout2D,
    BLayout: Layout2D,
    CLayout: Layout2D,
{
    type ALayout = ALayout;
    type BLayout = BLayout;
    type CLayout = CLayout;
    type Tile = Tile16x16x16;

    const NAME: &'static str = "tiled-gemm-16x16x16";
}

pub trait RmsNormKernelPlan: Copy + fmt::Debug + 'static {
    const NAME: &'static str;
    const THREADS: u32;

    fn block_threads() -> u32 {
        Self::THREADS
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockRmsNorm256Plan;

impl RmsNormKernelPlan for BlockRmsNorm256Plan {
    const NAME: &'static str = "block-rmsnorm-256";
    const THREADS: u32 = 256;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormEpsilon<Acc> {
    value: f32,
    _accumulator: PhantomData<Acc>,
}

impl<Acc: DeviceFloat> RmsNormEpsilon<Acc> {
    pub const fn from_f32(value: f32) -> Self {
        Self {
            value,
            _accumulator: PhantomData,
        }
    }

    pub const fn as_f32(self) -> f32 {
        self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormOperation<Plan, Acc> {
    epsilon: RmsNormEpsilon<Acc>,
    _plan: PhantomData<Plan>,
}

impl<Plan, Acc> RmsNormOperation<Plan, Acc>
where
    Plan: RmsNormKernelPlan,
    Acc: DeviceFloat,
{
    pub const fn new(epsilon: RmsNormEpsilon<Acc>) -> Self {
        Self {
            epsilon,
            _plan: PhantomData,
        }
    }

    pub const fn from_f32_epsilon(epsilon: f32) -> Self {
        Self::new(RmsNormEpsilon::<Acc>::from_f32(epsilon))
    }

    pub const fn epsilon(&self) -> RmsNormEpsilon<Acc> {
        self.epsilon
    }

    pub const fn epsilon_f32(&self) -> f32 {
        self.epsilon.as_f32()
    }

    pub fn block_threads(&self) -> u32 {
        Plan::block_threads()
    }
}

pub trait LogitSelectionTile: Copy + fmt::Debug + 'static {
    const THREADS: u32;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockLogitSelectionTile;

impl LogitSelectionTile for BlockLogitSelectionTile {
    const THREADS: u32 = 256;
}

pub trait LogitSelectionKernelPlan: Copy + fmt::Debug + 'static {
    type Tile: LogitSelectionTile;

    const NAME: &'static str;

    fn block_threads() -> u32 {
        Self::Tile::THREADS
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockLogitSelection256Plan;

impl LogitSelectionKernelPlan for BlockLogitSelection256Plan {
    type Tile = BlockLogitSelectionTile;

    const NAME: &'static str = "block-logit-selection-256";
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowMajor;

impl Layout2D for RowMajor {
    const NAME: &'static str = "row-major";

    fn packed_stride(shape: Shape2D) -> Stride2D {
        Stride2D::new(shape.cols, 1)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColumnMajor;

impl Layout2D for ColumnMajor {
    const NAME: &'static str = "column-major";

    fn packed_stride(shape: Shape2D) -> Stride2D {
        Stride2D::new(1, shape.rows)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SeqHeadDimMajor;

impl Layout3D for SeqHeadDimMajor {
    const NAME: &'static str = "seq-head-dim-major";

    fn packed_stride(shape: Shape3D) -> Stride3D {
        Stride3D::new(shape.middle * shape.inner, shape.inner, 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout<L> {
    shape: Shape2D,
    stride: Stride2D,
    _layout: PhantomData<L>,
}

impl<L: Layout2D> MatrixLayout<L> {
    pub fn packed(rows: usize, cols: usize) -> Self {
        let shape = Shape2D::new(rows, cols);
        Self {
            shape,
            stride: L::packed_stride(shape),
            _layout: PhantomData,
        }
    }
}

impl<L> MatrixLayout<L> {
    pub const fn new(shape: Shape2D, stride: Stride2D) -> Self {
        Self {
            shape,
            stride,
            _layout: PhantomData,
        }
    }

    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    pub const fn stride(&self) -> Stride2D {
        self.stride
    }

    pub fn offset(&self, row: usize, col: usize) -> usize {
        assert!(row < self.shape.rows, "matrix row index out of bounds");
        assert!(col < self.shape.cols, "matrix col index out of bounds");
        row.checked_mul(self.stride.row)
            .and_then(|row_offset| {
                col.checked_mul(self.stride.col)
                    .and_then(|c| row_offset.checked_add(c))
            })
            .expect("matrix layout offset overflow")
    }

    pub fn capacity(&self) -> usize {
        if self.shape.rows == 0 || self.shape.cols == 0 {
            return 0;
        }

        self.offset(self.shape.rows - 1, self.shape.cols - 1)
            .checked_add(1)
            .expect("matrix layout capacity overflow")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowwiseScaledMatrixLayout<L> {
    values: MatrixLayout<L>,
    scales_len: usize,
}

impl<L: Layout2D> RowwiseScaledMatrixLayout<L> {
    pub fn packed(rows: usize, cols: usize) -> Self {
        Self {
            values: MatrixLayout::<L>::packed(rows, cols),
            scales_len: rows,
        }
    }
}

impl<L> RowwiseScaledMatrixLayout<L> {
    pub fn new(values: MatrixLayout<L>, scales_len: usize) -> Result<Self, LayoutError> {
        let rows = values.shape().rows;
        if scales_len != rows {
            return Err(LayoutError::RowwiseScaleLengthMismatch {
                scales_len,
                weight_rows: rows,
            });
        }

        Ok(Self { values, scales_len })
    }

    pub const fn values(&self) -> &MatrixLayout<L> {
        &self.values
    }

    pub const fn scales_len(&self) -> usize {
        self.scales_len
    }

    pub fn rows(&self) -> usize {
        self.values.shape().rows
    }

    pub fn cols(&self) -> usize {
        self.values.shape().cols
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorLayout3D<L> {
    shape: Shape3D,
    stride: Stride3D,
    _layout: PhantomData<L>,
}

impl<L: Layout3D> TensorLayout3D<L> {
    pub fn packed(outer: usize, middle: usize, inner: usize) -> Self {
        let shape = Shape3D::new(outer, middle, inner);
        Self {
            shape,
            stride: L::packed_stride(shape),
            _layout: PhantomData,
        }
    }
}

impl<L> TensorLayout3D<L> {
    pub const fn new(shape: Shape3D, stride: Stride3D) -> Self {
        Self {
            shape,
            stride,
            _layout: PhantomData,
        }
    }

    pub const fn shape(&self) -> Shape3D {
        self.shape
    }

    pub const fn stride(&self) -> Stride3D {
        self.stride
    }

    pub fn offset(&self, outer: usize, middle: usize, inner: usize) -> usize {
        assert!(outer < self.shape.outer, "tensor outer index out of bounds");
        assert!(
            middle < self.shape.middle,
            "tensor middle index out of bounds"
        );
        assert!(inner < self.shape.inner, "tensor inner index out of bounds");
        outer
            .checked_mul(self.stride.outer)
            .and_then(|outer_offset| {
                middle
                    .checked_mul(self.stride.middle)
                    .and_then(|middle_offset| outer_offset.checked_add(middle_offset))
            })
            .and_then(|base| {
                inner
                    .checked_mul(self.stride.inner)
                    .and_then(|inner_offset| base.checked_add(inner_offset))
            })
            .expect("tensor layout offset overflow")
    }

    pub fn capacity(&self) -> usize {
        if self.shape.outer == 0 || self.shape.middle == 0 || self.shape.inner == 0 {
            return 0;
        }

        self.offset(
            self.shape.outer - 1,
            self.shape.middle - 1,
            self.shape.inner - 1,
        )
        .checked_add(1)
        .expect("tensor layout capacity overflow")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutError {
    BufferLengthMismatch {
        len: usize,
        required: usize,
        rows: usize,
        cols: usize,
        layout: &'static str,
    },
    VectorLengthMismatch {
        len: usize,
        expected: usize,
    },
    LinearShapeMismatch {
        input_len: usize,
        weight_rows: usize,
        weight_cols: usize,
        output_len: usize,
    },
    GemmShapeMismatch {
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
        c_rows: usize,
        c_cols: usize,
    },
    RowwiseScaleLengthMismatch {
        scales_len: usize,
        weight_rows: usize,
    },
    TensorLengthMismatch {
        len: usize,
        required: usize,
        outer: usize,
        middle: usize,
        inner: usize,
        layout: &'static str,
    },
    AttentionSequenceLengthMismatch {
        seq_len: usize,
        max_seq_len: usize,
    },
    AttentionHeadMismatch {
        n_heads: usize,
        n_kv_heads: usize,
    },
    AttentionShapeMismatch {
        query_len: usize,
        key_cache_len: usize,
        scores_len: usize,
        output_len: usize,
        expected_query_len: usize,
        expected_cache_len: usize,
        expected_scores_len: usize,
        expected_output_len: usize,
    },
    KvCacheWriteShapeMismatch {
        input_len: usize,
        cache_len: usize,
        expected_input_len: usize,
        expected_cache_len: usize,
        position: usize,
        max_seq_len: usize,
    },
    RmsNormShapeMismatch {
        input_len: usize,
        weight_len: usize,
        output_len: usize,
    },
    BinaryElementwiseShapeMismatch {
        op: &'static str,
        lhs_len: usize,
        rhs_len: usize,
        output_len: usize,
    },
    EmbeddingTableShapeMismatch {
        len: usize,
        dim: usize,
    },
    EmbeddingLookupShapeMismatch {
        token_id: u32,
        token_count: usize,
        dim: usize,
        output_len: usize,
    },
    ArgmaxShapeMismatch {
        logits_len: usize,
        token_out_len: usize,
        logit_out_len: usize,
    },
    TopKShapeMismatch {
        logits_len: usize,
        k: usize,
        token_out_len: usize,
        logit_out_len: usize,
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::BufferLengthMismatch {
                len,
                required,
                rows,
                cols,
                layout,
            } => write!(
                f,
                "{layout} matrix buffer has {len} elements, expected {required} for packed shape {rows}x{cols}"
            ),
            Self::VectorLengthMismatch { len, expected } => {
                write!(f, "vector has {len} elements, expected {expected}")
            }
            Self::LinearShapeMismatch {
                input_len,
                weight_rows,
                weight_cols,
                output_len,
            } => write!(
                f,
                "linear shape mismatch: input={input_len}, weight={weight_rows}x{weight_cols}, output={output_len}"
            ),
            Self::GemmShapeMismatch {
                a_rows,
                a_cols,
                b_rows,
                b_cols,
                c_rows,
                c_cols,
            } => write!(
                f,
                "GEMM shape mismatch: A={a_rows}x{a_cols}, B={b_rows}x{b_cols}, C={c_rows}x{c_cols}"
            ),
            Self::RowwiseScaleLengthMismatch {
                scales_len,
                weight_rows,
            } => write!(
                f,
                "rowwise scale length mismatch: scales={scales_len}, weight_rows={weight_rows}"
            ),
            Self::TensorLengthMismatch {
                len,
                required,
                outer,
                middle,
                inner,
                layout,
            } => write!(
                f,
                "{layout} tensor buffer has {len} elements, expected {required} for packed shape {outer}x{middle}x{inner}"
            ),
            Self::AttentionSequenceLengthMismatch {
                seq_len,
                max_seq_len,
            } => write!(
                f,
                "attention sequence length mismatch: seq_len={seq_len}, max_seq_len={max_seq_len}"
            ),
            Self::AttentionHeadMismatch {
                n_heads,
                n_kv_heads,
            } => write!(
                f,
                "attention head mismatch: n_heads={n_heads}, n_kv_heads={n_kv_heads}"
            ),
            Self::AttentionShapeMismatch {
                query_len,
                key_cache_len,
                scores_len,
                output_len,
                expected_query_len,
                expected_cache_len,
                expected_scores_len,
                expected_output_len,
            } => write!(
                f,
                "attention shape mismatch: query={query_len}/{expected_query_len}, key_cache={key_cache_len}/{expected_cache_len}, scores={scores_len}/{expected_scores_len}, output={output_len}/{expected_output_len}"
            ),
            Self::KvCacheWriteShapeMismatch {
                input_len,
                cache_len,
                expected_input_len,
                expected_cache_len,
                position,
                max_seq_len,
            } => write!(
                f,
                "KV cache write shape mismatch: input={input_len}/{expected_input_len}, cache={cache_len}/{expected_cache_len}, position={position}, max_seq_len={max_seq_len}"
            ),
            Self::RmsNormShapeMismatch {
                input_len,
                weight_len,
                output_len,
            } => write!(
                f,
                "RMSNorm shape mismatch: input={input_len}, weight={weight_len}, output={output_len}"
            ),
            Self::BinaryElementwiseShapeMismatch {
                op,
                lhs_len,
                rhs_len,
                output_len,
            } => write!(
                f,
                "{op} shape mismatch: lhs={lhs_len}, rhs={rhs_len}, output={output_len}"
            ),
            Self::EmbeddingTableShapeMismatch { len, dim } => write!(
                f,
                "embedding table shape mismatch: weights_len={len}, dim={dim}"
            ),
            Self::EmbeddingLookupShapeMismatch {
                token_id,
                token_count,
                dim,
                output_len,
            } => write!(
                f,
                "embedding lookup shape mismatch: token_id={token_id}, token_count={token_count}, dim={dim}, output={output_len}"
            ),
            Self::ArgmaxShapeMismatch {
                logits_len,
                token_out_len,
                logit_out_len,
            } => write!(
                f,
                "argmax shape mismatch: logits={logits_len}, token_out={token_out_len}, logit_out={logit_out_len}"
            ),
            Self::TopKShapeMismatch {
                logits_len,
                k,
                token_out_len,
                logit_out_len,
            } => write!(
                f,
                "top-k shape mismatch: logits={logits_len}, k={k}, token_out={token_out_len}, logit_out={logit_out_len}"
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

pub struct DeviceVector<'a, T> {
    buffer: &'a DeviceBuffer<T>,
    len: usize,
}

impl<'a, T> DeviceVector<'a, T> {
    pub fn new(buffer: &'a DeviceBuffer<T>) -> Self {
        Self {
            buffer,
            len: buffer.len(),
        }
    }

    pub fn with_len(buffer: &'a DeviceBuffer<T>, len: usize) -> Result<Self, LayoutError> {
        if buffer.len() != len {
            return Err(LayoutError::VectorLengthMismatch {
                len: buffer.len(),
                expected: len,
            });
        }
        Ok(Self { buffer, len })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn buffer(&self) -> &'a DeviceBuffer<T> {
        self.buffer
    }
}

pub struct DeviceVectorMut<'a, T> {
    buffer: &'a mut DeviceBuffer<T>,
    len: usize,
}

impl<'a, T> DeviceVectorMut<'a, T> {
    pub fn new(buffer: &'a mut DeviceBuffer<T>) -> Self {
        let len = buffer.len();
        Self { buffer, len }
    }

    pub fn with_len(buffer: &'a mut DeviceBuffer<T>, len: usize) -> Result<Self, LayoutError> {
        if buffer.len() != len {
            return Err(LayoutError::VectorLengthMismatch {
                len: buffer.len(),
                expected: len,
            });
        }
        Ok(Self { buffer, len })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn buffer_mut(&mut self) -> &mut DeviceBuffer<T> {
        self.buffer
    }
}

pub struct DeviceMatrix<'a, T, L> {
    buffer: &'a DeviceBuffer<T>,
    layout: MatrixLayout<L>,
}

impl<'a, T, L: Layout2D> DeviceMatrix<'a, T, L> {
    pub fn packed(
        buffer: &'a DeviceBuffer<T>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, LayoutError> {
        let layout = MatrixLayout::<L>::packed(rows, cols);
        let required = layout.capacity();
        if buffer.len() != required {
            return Err(LayoutError::BufferLengthMismatch {
                len: buffer.len(),
                required,
                rows,
                cols,
                layout: L::NAME,
            });
        }

        Ok(Self { buffer, layout })
    }
}

impl<'a, T, L> DeviceMatrix<'a, T, L> {
    pub const fn layout(&self) -> &MatrixLayout<L> {
        &self.layout
    }

    pub const fn buffer(&self) -> &'a DeviceBuffer<T> {
        self.buffer
    }

    pub fn rows(&self) -> usize {
        self.layout.shape().rows
    }

    pub fn cols(&self) -> usize {
        self.layout.shape().cols
    }
}

pub struct DeviceRowwiseScaledMatrix<'a, Weight, Scale, L> {
    values: DeviceMatrix<'a, Weight, L>,
    scales: DeviceVector<'a, Scale>,
    layout: RowwiseScaledMatrixLayout<L>,
}

impl<'a, Weight, Scale, L: Layout2D> DeviceRowwiseScaledMatrix<'a, Weight, Scale, L> {
    pub fn packed(
        values: &'a DeviceBuffer<Weight>,
        scales: &'a DeviceBuffer<Scale>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, LayoutError> {
        let values = DeviceMatrix::<Weight, L>::packed(values, rows, cols)?;
        let scales = DeviceVector::with_len(scales, rows)?;
        Self::new(values, scales)
    }
}

impl<'a, Weight, Scale, L> DeviceRowwiseScaledMatrix<'a, Weight, Scale, L> {
    pub fn new(
        values: DeviceMatrix<'a, Weight, L>,
        scales: DeviceVector<'a, Scale>,
    ) -> Result<Self, LayoutError> {
        let value_layout =
            MatrixLayout::<L>::new(values.layout().shape(), values.layout().stride());
        let layout = RowwiseScaledMatrixLayout::new(value_layout, scales.len())?;
        Ok(Self {
            values,
            scales,
            layout,
        })
    }

    pub const fn layout(&self) -> &RowwiseScaledMatrixLayout<L> {
        &self.layout
    }

    pub const fn values(&self) -> &DeviceMatrix<'a, Weight, L> {
        &self.values
    }

    pub const fn scales(&self) -> &DeviceVector<'a, Scale> {
        &self.scales
    }

    pub fn rows(&self) -> usize {
        self.layout.rows()
    }

    pub fn cols(&self) -> usize {
        self.layout.cols()
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceMatrix<'a, Weight, L>,
        DeviceVector<'a, Scale>,
        RowwiseScaledMatrixLayout<L>,
    ) {
        (self.values, self.scales, self.layout)
    }
}

pub struct DeviceMatrixMut<'a, T, L> {
    buffer: &'a mut DeviceBuffer<T>,
    layout: MatrixLayout<L>,
}

impl<'a, T, L: Layout2D> DeviceMatrixMut<'a, T, L> {
    pub fn packed(
        buffer: &'a mut DeviceBuffer<T>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, LayoutError> {
        let layout = MatrixLayout::<L>::packed(rows, cols);
        let required = layout.capacity();
        if buffer.len() != required {
            return Err(LayoutError::BufferLengthMismatch {
                len: buffer.len(),
                required,
                rows,
                cols,
                layout: L::NAME,
            });
        }

        Ok(Self { buffer, layout })
    }
}

impl<'a, T, L> DeviceMatrixMut<'a, T, L> {
    pub const fn layout(&self) -> &MatrixLayout<L> {
        &self.layout
    }

    pub fn buffer_mut(&mut self) -> &mut DeviceBuffer<T> {
        self.buffer
    }

    pub fn rows(&self) -> usize {
        self.layout.shape().rows
    }

    pub fn cols(&self) -> usize {
        self.layout.shape().cols
    }
}

pub struct DeviceEmbeddingTable<'a, T> {
    weight: DeviceMatrix<'a, T, RowMajor>,
}

impl<'a, T: DeviceFloat> DeviceEmbeddingTable<'a, T> {
    pub fn packed(buffer: &'a DeviceBuffer<T>, dim: usize) -> Result<Self, LayoutError> {
        if dim == 0 || buffer.len() % dim != 0 {
            return Err(LayoutError::EmbeddingTableShapeMismatch {
                len: buffer.len(),
                dim,
            });
        }

        let token_count = buffer.len() / dim;
        Ok(Self {
            weight: DeviceMatrix::<T, RowMajor>::packed(buffer, token_count, dim)?,
        })
    }
}

impl<'a, T> DeviceEmbeddingTable<'a, T> {
    pub const fn layout(&self) -> &MatrixLayout<RowMajor> {
        self.weight.layout()
    }

    pub const fn buffer(&self) -> &'a DeviceBuffer<T> {
        self.weight.buffer()
    }

    pub fn token_count(&self) -> usize {
        self.weight.rows()
    }

    pub fn dim(&self) -> usize {
        self.weight.cols()
    }
}

pub struct DeviceTensor3<'a, T, L> {
    buffer: &'a DeviceBuffer<T>,
    layout: TensorLayout3D<L>,
}

impl<'a, T, L: Layout3D> DeviceTensor3<'a, T, L> {
    pub fn packed(
        buffer: &'a DeviceBuffer<T>,
        outer: usize,
        middle: usize,
        inner: usize,
    ) -> Result<Self, LayoutError> {
        let layout = TensorLayout3D::<L>::packed(outer, middle, inner);
        let required = layout.capacity();
        if buffer.len() != required {
            return Err(LayoutError::TensorLengthMismatch {
                len: buffer.len(),
                required,
                outer,
                middle,
                inner,
                layout: L::NAME,
            });
        }

        Ok(Self { buffer, layout })
    }
}

impl<'a, T, L> DeviceTensor3<'a, T, L> {
    pub const fn layout(&self) -> &TensorLayout3D<L> {
        &self.layout
    }

    pub const fn buffer(&self) -> &'a DeviceBuffer<T> {
        self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

pub struct DeviceTensor3Mut<'a, T, L> {
    buffer: &'a mut DeviceBuffer<T>,
    layout: TensorLayout3D<L>,
}

impl<'a, T, L: Layout3D> DeviceTensor3Mut<'a, T, L> {
    pub fn packed(
        buffer: &'a mut DeviceBuffer<T>,
        outer: usize,
        middle: usize,
        inner: usize,
    ) -> Result<Self, LayoutError> {
        let layout = TensorLayout3D::<L>::packed(outer, middle, inner);
        let required = layout.capacity();
        if buffer.len() != required {
            return Err(LayoutError::TensorLengthMismatch {
                len: buffer.len(),
                required,
                outer,
                middle,
                inner,
                layout: L::NAME,
            });
        }

        Ok(Self { buffer, layout })
    }
}

impl<'a, T, L> DeviceTensor3Mut<'a, T, L> {
    pub const fn layout(&self) -> &TensorLayout3D<L> {
        &self.layout
    }

    pub fn buffer_mut(&mut self) -> &mut DeviceBuffer<T> {
        self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheGeometry {
    pub max_seq_len: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCacheGeometry {
    pub const fn new(max_seq_len: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            max_seq_len,
            n_kv_heads,
            head_dim,
        }
    }

    pub fn token_len(self) -> usize {
        self.n_kv_heads
            .checked_mul(self.head_dim)
            .expect("KV token length overflow")
    }

    pub fn cache_len(self) -> usize {
        self.max_seq_len
            .checked_mul(self.token_len())
            .expect("KV cache length overflow")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionGeometry {
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionGeometry {
    pub fn new(
        seq_len: usize,
        max_seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, LayoutError> {
        if seq_len == 0 || seq_len > max_seq_len {
            return Err(LayoutError::AttentionSequenceLengthMismatch {
                seq_len,
                max_seq_len,
            });
        }
        if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(LayoutError::AttentionHeadMismatch {
                n_heads,
                n_kv_heads,
            });
        }

        Ok(Self {
            seq_len,
            max_seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    pub fn kv_cache_geometry(self) -> KvCacheGeometry {
        KvCacheGeometry::new(self.max_seq_len, self.n_kv_heads, self.head_dim)
    }

    pub fn query_len(self) -> usize {
        self.n_heads
            .checked_mul(self.head_dim)
            .expect("attention query length overflow")
    }

    pub fn output_len(self) -> usize {
        self.query_len()
    }

    pub fn scores_len(self) -> usize {
        self.n_heads
            .checked_mul(self.max_seq_len)
            .expect("attention scores length overflow")
    }
}

pub trait CudaLinearRoute<Weight>: DeviceFloat
where
    Weight: DeviceFloat,
    Self: AccumulatorWith<Weight, Cuda>,
{
    type Output: TensorElement;
}

impl CudaLinearRoute<crate::dtypes::Bf16> for f32 {
    type Output = f32;
}

impl CudaLinearRoute<f32> for f32 {
    type Output = f32;
}

pub trait CudaRowwiseScaledLinearRoute<Weight, Scale>: DeviceFloat
where
    Weight: DeviceStorageElement,
    Scale: DeviceFloat,
{
    type Accumulator: DeviceFloat;
    type Output: TensorElement;
}

impl CudaRowwiseScaledLinearRoute<i8, f32> for f32 {
    type Accumulator = f32;
    type Output = f32;
}

pub trait CudaRmsNormRoute<Weight>: DeviceFloat
where
    Weight: DeviceFloat,
{
    type Accumulator: DeviceFloat;
    type Output: TensorElement;
}

impl CudaRmsNormRoute<crate::dtypes::Bf16> for f32 {
    type Accumulator = f32;
    type Output = f32;
}

impl CudaRmsNormRoute<f32> for f32 {
    type Accumulator = f32;
    type Output = f32;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SiluMulOp;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AddOp;

pub trait BinaryElementwiseOp: Copy + fmt::Debug + 'static {
    const NAME: &'static str;
}

impl BinaryElementwiseOp for SiluMulOp {
    const NAME: &'static str = "silu_mul";
}

impl BinaryElementwiseOp for AddOp {
    const NAME: &'static str = "add";
}

pub trait CudaBinaryElementwiseRoute<Rhs, Op>: DeviceFloat
where
    Rhs: DeviceFloat,
    Op: BinaryElementwiseOp,
{
    type Output: TensorElement;
}

impl CudaBinaryElementwiseRoute<f32, SiluMulOp> for f32 {
    type Output = f32;
}

impl CudaBinaryElementwiseRoute<f32, AddOp> for f32 {
    type Output = f32;
}

pub trait CudaEmbeddingRoute: DeviceFloat {
    type Output: TensorElement;
}

impl CudaEmbeddingRoute for crate::dtypes::Bf16 {
    type Output = f32;
}

impl CudaEmbeddingRoute for f32 {
    type Output = f32;
}

pub struct LinearProblem<'a, Input, Weight, Output, L>
where
    Input: CudaLinearRoute<Weight, Output = Output>,
    Weight: DeviceFloat,
    Output: TensorElement,
{
    input: DeviceVector<'a, Input>,
    weight: DeviceMatrix<'a, Weight, L>,
    output: DeviceVectorMut<'a, Output>,
    _accumulator: PhantomData<<Input as AccumulatorWith<Weight, Cuda>>::Accumulator>,
}

impl<'a, Input, Weight, Output, L> LinearProblem<'a, Input, Weight, Output, L>
where
    Input: CudaLinearRoute<Weight, Output = Output>,
    Weight: DeviceFloat,
    Output: TensorElement,
{
    pub fn new(
        input: DeviceVector<'a, Input>,
        weight: DeviceMatrix<'a, Weight, L>,
        output: DeviceVectorMut<'a, Output>,
    ) -> Result<Self, LayoutError> {
        if input.len() != weight.cols() || output.len() != weight.rows() {
            return Err(LayoutError::LinearShapeMismatch {
                input_len: input.len(),
                weight_rows: weight.rows(),
                weight_cols: weight.cols(),
                output_len: output.len(),
            });
        }

        Ok(Self {
            input,
            weight,
            output,
            _accumulator: PhantomData,
        })
    }

    pub const fn input(&self) -> &DeviceVector<'a, Input> {
        &self.input
    }

    pub const fn weight(&self) -> &DeviceMatrix<'a, Weight, L> {
        &self.weight
    }

    pub fn output_mut(&mut self) -> &mut DeviceVectorMut<'a, Output> {
        &mut self.output
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceVector<'a, Input>,
        DeviceMatrix<'a, Weight, L>,
        DeviceVectorMut<'a, Output>,
    ) {
        (self.input, self.weight, self.output)
    }
}

pub struct GemmProblem<'a, A, B, C, LA, LB, LC>
where
    A: DeviceFloat,
    B: DeviceFloat,
    C: DeviceFloat,
{
    a: DeviceMatrix<'a, A, LA>,
    b: DeviceMatrix<'a, B, LB>,
    c: DeviceMatrixMut<'a, C, LC>,
    alpha: f32,
    beta: f32,
}

impl<'a, A, B, C, LA, LB, LC> GemmProblem<'a, A, B, C, LA, LB, LC>
where
    A: DeviceFloat,
    B: DeviceFloat,
    C: DeviceFloat,
{
    pub fn new(
        a: DeviceMatrix<'a, A, LA>,
        b: DeviceMatrix<'a, B, LB>,
        c: DeviceMatrixMut<'a, C, LC>,
        alpha: f32,
        beta: f32,
    ) -> Result<Self, LayoutError> {
        if a.cols() != b.rows() || c.rows() != a.rows() || c.cols() != b.cols() {
            return Err(LayoutError::GemmShapeMismatch {
                a_rows: a.rows(),
                a_cols: a.cols(),
                b_rows: b.rows(),
                b_cols: b.cols(),
                c_rows: c.rows(),
                c_cols: c.cols(),
            });
        }

        Ok(Self {
            a,
            b,
            c,
            alpha,
            beta,
        })
    }

    pub fn m(&self) -> usize {
        self.a.rows()
    }

    pub fn n(&self) -> usize {
        self.b.cols()
    }

    pub fn k(&self) -> usize {
        self.a.cols()
    }

    pub const fn alpha(&self) -> f32 {
        self.alpha
    }

    pub const fn beta(&self) -> f32 {
        self.beta
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceMatrix<'a, A, LA>,
        DeviceMatrix<'a, B, LB>,
        DeviceMatrixMut<'a, C, LC>,
        f32,
        f32,
    ) {
        (self.a, self.b, self.c, self.alpha, self.beta)
    }
}

pub struct RowwiseScaledLinearProblem<'a, Input, Weight, Scale, Output, L>
where
    Input: CudaRowwiseScaledLinearRoute<Weight, Scale, Output = Output>,
    Weight: DeviceStorageElement,
    Scale: DeviceFloat,
    Output: TensorElement,
{
    input: DeviceVector<'a, Input>,
    weight: DeviceRowwiseScaledMatrix<'a, Weight, Scale, L>,
    output: DeviceVectorMut<'a, Output>,
    _accumulator: PhantomData<<Input as CudaRowwiseScaledLinearRoute<Weight, Scale>>::Accumulator>,
}

impl<'a, Input, Weight, Scale, Output, L>
    RowwiseScaledLinearProblem<'a, Input, Weight, Scale, Output, L>
where
    Input: CudaRowwiseScaledLinearRoute<Weight, Scale, Output = Output>,
    Weight: DeviceStorageElement,
    Scale: DeviceFloat,
    Output: TensorElement,
{
    pub fn new(
        input: DeviceVector<'a, Input>,
        weight: DeviceRowwiseScaledMatrix<'a, Weight, Scale, L>,
        output: DeviceVectorMut<'a, Output>,
    ) -> Result<Self, LayoutError> {
        if input.len() != weight.cols() || output.len() != weight.rows() {
            return Err(LayoutError::LinearShapeMismatch {
                input_len: input.len(),
                weight_rows: weight.rows(),
                weight_cols: weight.cols(),
                output_len: output.len(),
            });
        }

        Ok(Self {
            input,
            weight,
            output,
            _accumulator: PhantomData,
        })
    }

    pub const fn input(&self) -> &DeviceVector<'a, Input> {
        &self.input
    }

    pub const fn weight(&self) -> &DeviceRowwiseScaledMatrix<'a, Weight, Scale, L> {
        &self.weight
    }

    pub fn output_mut(&mut self) -> &mut DeviceVectorMut<'a, Output> {
        &mut self.output
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceVector<'a, Input>,
        DeviceRowwiseScaledMatrix<'a, Weight, Scale, L>,
        DeviceVectorMut<'a, Output>,
    ) {
        (self.input, self.weight, self.output)
    }
}

pub struct EmbeddingLookupProblem<'a, Weight, Output>
where
    Weight: CudaEmbeddingRoute<Output = Output>,
    Output: TensorElement,
{
    weight: DeviceEmbeddingTable<'a, Weight>,
    token_id: u32,
    output: DeviceVectorMut<'a, Output>,
}

impl<'a, Weight, Output> EmbeddingLookupProblem<'a, Weight, Output>
where
    Weight: CudaEmbeddingRoute<Output = Output>,
    Output: TensorElement,
{
    pub fn new(
        weight: DeviceEmbeddingTable<'a, Weight>,
        token_id: u32,
        output: DeviceVectorMut<'a, Output>,
    ) -> Result<Self, LayoutError> {
        if token_id as usize >= weight.token_count() || output.len() != weight.dim() {
            return Err(LayoutError::EmbeddingLookupShapeMismatch {
                token_id,
                token_count: weight.token_count(),
                dim: weight.dim(),
                output_len: output.len(),
            });
        }

        Ok(Self {
            weight,
            token_id,
            output,
        })
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceEmbeddingTable<'a, Weight>,
        u32,
        DeviceVectorMut<'a, Output>,
    ) {
        (self.weight, self.token_id, self.output)
    }
}

pub struct RmsNormProblem<'a, Input, Weight, Output, Plan>
where
    Input: CudaRmsNormRoute<Weight, Output = Output>,
    Weight: DeviceFloat,
    Output: TensorElement,
    Plan: RmsNormKernelPlan,
{
    operation: RmsNormOperation<Plan, <Input as CudaRmsNormRoute<Weight>>::Accumulator>,
    input: DeviceVector<'a, Input>,
    weight: DeviceVector<'a, Weight>,
    output: DeviceVectorMut<'a, Output>,
    _accumulator: PhantomData<<Input as CudaRmsNormRoute<Weight>>::Accumulator>,
}

impl<'a, Input, Weight, Output, Plan> RmsNormProblem<'a, Input, Weight, Output, Plan>
where
    Input: CudaRmsNormRoute<Weight, Output = Output>,
    Weight: DeviceFloat,
    Output: TensorElement,
    Plan: RmsNormKernelPlan,
{
    pub fn new(
        operation: RmsNormOperation<Plan, <Input as CudaRmsNormRoute<Weight>>::Accumulator>,
        input: DeviceVector<'a, Input>,
        weight: DeviceVector<'a, Weight>,
        output: DeviceVectorMut<'a, Output>,
    ) -> Result<Self, LayoutError> {
        if input.len() != weight.len() || input.len() != output.len() {
            return Err(LayoutError::RmsNormShapeMismatch {
                input_len: input.len(),
                weight_len: weight.len(),
                output_len: output.len(),
            });
        }

        Ok(Self {
            operation,
            input,
            weight,
            output,
            _accumulator: PhantomData,
        })
    }

    pub fn into_parts(
        self,
    ) -> (
        RmsNormOperation<Plan, <Input as CudaRmsNormRoute<Weight>>::Accumulator>,
        DeviceVector<'a, Input>,
        DeviceVector<'a, Weight>,
        DeviceVectorMut<'a, Output>,
    ) {
        (self.operation, self.input, self.weight, self.output)
    }
}

pub struct BinaryElementwiseProblem<'a, Lhs, Rhs, Output, Op>
where
    Lhs: CudaBinaryElementwiseRoute<Rhs, Op, Output = Output>,
    Rhs: DeviceFloat,
    Output: TensorElement,
    Op: BinaryElementwiseOp,
{
    lhs: DeviceVector<'a, Lhs>,
    rhs: DeviceVector<'a, Rhs>,
    output: DeviceVectorMut<'a, Output>,
    _op: PhantomData<Op>,
}

impl<'a, Lhs, Rhs, Output, Op> BinaryElementwiseProblem<'a, Lhs, Rhs, Output, Op>
where
    Lhs: CudaBinaryElementwiseRoute<Rhs, Op, Output = Output>,
    Rhs: DeviceFloat,
    Output: TensorElement,
    Op: BinaryElementwiseOp,
{
    pub fn new(
        lhs: DeviceVector<'a, Lhs>,
        rhs: DeviceVector<'a, Rhs>,
        output: DeviceVectorMut<'a, Output>,
    ) -> Result<Self, LayoutError> {
        if lhs.len() != rhs.len() || lhs.len() != output.len() {
            return Err(LayoutError::BinaryElementwiseShapeMismatch {
                op: Op::NAME,
                lhs_len: lhs.len(),
                rhs_len: rhs.len(),
                output_len: output.len(),
            });
        }

        Ok(Self {
            lhs,
            rhs,
            output,
            _op: PhantomData,
        })
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceVector<'a, Lhs>,
        DeviceVector<'a, Rhs>,
        DeviceVectorMut<'a, Output>,
    ) {
        (self.lhs, self.rhs, self.output)
    }
}

pub struct ArgmaxProblem<'a> {
    logits: DeviceVector<'a, f32>,
    token_out: DeviceVectorMut<'a, u32>,
    logit_out: DeviceVectorMut<'a, f32>,
}

impl<'a> ArgmaxProblem<'a> {
    pub fn new(
        logits: DeviceVector<'a, f32>,
        token_out: DeviceVectorMut<'a, u32>,
        logit_out: DeviceVectorMut<'a, f32>,
    ) -> Result<Self, LayoutError> {
        if logits.len() == 0 || token_out.len() != 1 || logit_out.len() != 1 {
            return Err(LayoutError::ArgmaxShapeMismatch {
                logits_len: logits.len(),
                token_out_len: token_out.len(),
                logit_out_len: logit_out.len(),
            });
        }

        Ok(Self {
            logits,
            token_out,
            logit_out,
        })
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceVector<'a, f32>,
        DeviceVectorMut<'a, u32>,
        DeviceVectorMut<'a, f32>,
    ) {
        (self.logits, self.token_out, self.logit_out)
    }
}

pub struct TopKProblem<'a> {
    logits: DeviceVector<'a, f32>,
    k: usize,
    token_out: DeviceVectorMut<'a, u32>,
    logit_out: DeviceVectorMut<'a, f32>,
}

impl<'a> TopKProblem<'a> {
    pub fn new(
        logits: DeviceVector<'a, f32>,
        k: usize,
        token_out: DeviceVectorMut<'a, u32>,
        logit_out: DeviceVectorMut<'a, f32>,
    ) -> Result<Self, LayoutError> {
        if logits.len() == 0 || k == 0 || token_out.len() < k || logit_out.len() < k {
            return Err(LayoutError::TopKShapeMismatch {
                logits_len: logits.len(),
                k,
                token_out_len: token_out.len(),
                logit_out_len: logit_out.len(),
            });
        }

        Ok(Self {
            logits,
            k,
            token_out,
            logit_out,
        })
    }

    pub const fn k(&self) -> usize {
        self.k
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceVector<'a, f32>,
        DeviceVectorMut<'a, u32>,
        DeviceVectorMut<'a, f32>,
    ) {
        (self.logits, self.token_out, self.logit_out)
    }
}

pub struct KvCacheWriteProblem<'a> {
    input: DeviceVector<'a, f32>,
    cache: DeviceTensor3Mut<'a, f32, SeqHeadDimMajor>,
    position: usize,
    geometry: KvCacheGeometry,
}

impl<'a> KvCacheWriteProblem<'a> {
    pub fn new(
        input: DeviceVector<'a, f32>,
        cache: DeviceTensor3Mut<'a, f32, SeqHeadDimMajor>,
        position: usize,
        geometry: KvCacheGeometry,
    ) -> Result<Self, LayoutError> {
        let expected_input_len = geometry.token_len();
        let expected_cache_len = geometry.cache_len();
        if position >= geometry.max_seq_len
            || input.len() != expected_input_len
            || cache.len() != expected_cache_len
        {
            return Err(LayoutError::KvCacheWriteShapeMismatch {
                input_len: input.len(),
                cache_len: cache.len(),
                expected_input_len,
                expected_cache_len,
                position,
                max_seq_len: geometry.max_seq_len,
            });
        }

        Ok(Self {
            input,
            cache,
            position,
            geometry,
        })
    }

    pub const fn position(&self) -> usize {
        self.position
    }

    pub const fn geometry(&self) -> KvCacheGeometry {
        self.geometry
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceVector<'a, f32>,
        DeviceTensor3Mut<'a, f32, SeqHeadDimMajor>,
    ) {
        (self.input, self.cache)
    }
}

pub struct AttentionScoresProblem<'a> {
    geometry: AttentionGeometry,
    query: DeviceVector<'a, f32>,
    key_cache: DeviceTensor3<'a, f32, SeqHeadDimMajor>,
    scores: DeviceMatrixMut<'a, f32, RowMajor>,
}

impl<'a> AttentionScoresProblem<'a> {
    pub fn new(
        geometry: AttentionGeometry,
        query: DeviceVector<'a, f32>,
        key_cache: DeviceTensor3<'a, f32, SeqHeadDimMajor>,
        scores: DeviceMatrixMut<'a, f32, RowMajor>,
    ) -> Result<Self, LayoutError> {
        let expected_query_len = geometry.query_len();
        let expected_cache_len = geometry.kv_cache_geometry().cache_len();
        let expected_scores_len = geometry.scores_len();
        if query.len() != expected_query_len
            || key_cache.len() != expected_cache_len
            || scores.rows() != geometry.n_heads
            || scores.cols() != geometry.max_seq_len
        {
            return Err(LayoutError::AttentionShapeMismatch {
                query_len: query.len(),
                key_cache_len: key_cache.len(),
                scores_len: scores.rows() * scores.cols(),
                output_len: 0,
                expected_query_len,
                expected_cache_len,
                expected_scores_len,
                expected_output_len: 0,
            });
        }

        Ok(Self {
            geometry,
            query,
            key_cache,
            scores,
        })
    }

    pub const fn geometry(&self) -> AttentionGeometry {
        self.geometry
    }

    pub fn into_parts(
        self,
    ) -> (
        AttentionGeometry,
        DeviceVector<'a, f32>,
        DeviceTensor3<'a, f32, SeqHeadDimMajor>,
        DeviceMatrixMut<'a, f32, RowMajor>,
    ) {
        (self.geometry, self.query, self.key_cache, self.scores)
    }
}

pub struct SoftmaxValueProblem<'a> {
    geometry: AttentionGeometry,
    scores: DeviceMatrix<'a, f32, RowMajor>,
    value_cache: DeviceTensor3<'a, f32, SeqHeadDimMajor>,
    output: DeviceVectorMut<'a, f32>,
}

impl<'a> SoftmaxValueProblem<'a> {
    pub fn new(
        geometry: AttentionGeometry,
        scores: DeviceMatrix<'a, f32, RowMajor>,
        value_cache: DeviceTensor3<'a, f32, SeqHeadDimMajor>,
        output: DeviceVectorMut<'a, f32>,
    ) -> Result<Self, LayoutError> {
        let expected_query_len = geometry.query_len();
        let expected_cache_len = geometry.kv_cache_geometry().cache_len();
        let expected_scores_len = geometry.scores_len();
        let expected_output_len = geometry.output_len();
        if scores.rows() != geometry.n_heads
            || scores.cols() != geometry.max_seq_len
            || value_cache.len() != expected_cache_len
            || output.len() != expected_output_len
        {
            return Err(LayoutError::AttentionShapeMismatch {
                query_len: 0,
                key_cache_len: value_cache.len(),
                scores_len: scores.rows() * scores.cols(),
                output_len: output.len(),
                expected_query_len,
                expected_cache_len,
                expected_scores_len,
                expected_output_len,
            });
        }

        Ok(Self {
            geometry,
            scores,
            value_cache,
            output,
        })
    }

    pub const fn geometry(&self) -> AttentionGeometry {
        self.geometry
    }

    pub fn into_parts(
        self,
    ) -> (
        AttentionGeometry,
        DeviceMatrix<'a, f32, RowMajor>,
        DeviceTensor3<'a, f32, SeqHeadDimMajor>,
        DeviceVectorMut<'a, f32>,
    ) {
        (self.geometry, self.scores, self.value_cache, self.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_offsets_match_cute_layout_demo() {
        let layout = MatrixLayout::<RowMajor>::packed(4, 8);
        assert_eq!(layout.offset(2, 3), 19);
        assert_eq!(layout.capacity(), 32);
    }

    #[test]
    fn column_major_offsets_match_cute_layout_demo() {
        let layout = MatrixLayout::<ColumnMajor>::packed(4, 8);
        assert_eq!(layout.offset(2, 3), 14);
        assert_eq!(layout.capacity(), 32);
    }

    #[test]
    fn rowwise_scaled_layout_pairs_matrix_rows_with_scales() {
        let layout = RowwiseScaledMatrixLayout::<RowMajor>::packed(4, 8);
        assert_eq!(layout.rows(), 4);
        assert_eq!(layout.cols(), 8);
        assert_eq!(layout.scales_len(), 4);
        assert_eq!(layout.values().offset(2, 3), 19);
    }

    #[test]
    fn rowwise_scaled_layout_rejects_scale_row_mismatch() {
        let matrix = MatrixLayout::<RowMajor>::packed(4, 8);
        let error = RowwiseScaledMatrixLayout::new(matrix, 3).unwrap_err();
        assert_eq!(
            error,
            LayoutError::RowwiseScaleLengthMismatch {
                scales_len: 3,
                weight_rows: 4
            }
        );
    }

    #[test]
    fn seq_head_dim_offsets_match_packed_cache_layout() {
        let layout = TensorLayout3D::<SeqHeadDimMajor>::packed(5, 6, 7);
        assert_eq!(layout.offset(2, 3, 4), 109);
        assert_eq!(layout.capacity(), 210);
    }

    #[test]
    fn warp_row_tile_groups_one_output_row_per_block() {
        assert_eq!(WarpRowTile::LANES_PER_ROW, 32);
        assert_eq!(WarpRowTile::ROWS_PER_BLOCK, 1);
        assert_eq!(WarpRowTile::block_threads(), 32);
        assert_eq!(WarpRowTile::grid_rows(4096), 4096);
        assert_eq!(WarpRowTile::grid_rows(4099), 4099);
    }

    #[test]
    fn warp_rows2_tile_groups_two_output_rows_per_block() {
        assert_eq!(WarpRows2Tile::LANES_PER_ROW, 32);
        assert_eq!(WarpRows2Tile::ROWS_PER_BLOCK, 2);
        assert_eq!(WarpRows2Tile::block_threads(), 64);
        assert_eq!(WarpRows2Tile::grid_rows(4096), 2048);
        assert_eq!(WarpRows2Tile::grid_rows(4099), 2050);
    }

    #[test]
    fn warp_rows4_tile_groups_four_output_rows_per_block() {
        assert_eq!(WarpRows4Tile::LANES_PER_ROW, 32);
        assert_eq!(WarpRows4Tile::ROWS_PER_BLOCK, 4);
        assert_eq!(WarpRows4Tile::block_threads(), 128);
        assert_eq!(WarpRows4Tile::grid_rows(4096), 1024);
        assert_eq!(WarpRows4Tile::grid_rows(4099), 1025);
    }

    #[test]
    fn warp_rows8_tile_groups_eight_output_rows_per_block() {
        assert_eq!(WarpRows8Tile::LANES_PER_ROW, 32);
        assert_eq!(WarpRows8Tile::ROWS_PER_BLOCK, 8);
        assert_eq!(WarpRows8Tile::block_threads(), 256);
        assert_eq!(WarpRows8Tile::grid_rows(4096), 512);
        assert_eq!(WarpRows8Tile::grid_rows(4099), 513);
    }

    #[test]
    fn rmsnorm_operation_carries_plan_and_accumulator_epsilon() {
        let operation = RmsNormOperation::<BlockRmsNorm256Plan, f32>::from_f32_epsilon(1.0e-5);
        assert_eq!(operation.epsilon_f32(), 1.0e-5);
        assert_eq!(operation.block_threads(), 256);
    }

    #[test]
    fn block_logit_selection_tile_uses_one_reduction_block() {
        assert_eq!(BlockLogitSelectionTile::THREADS, 256);
    }

    #[test]
    fn matvec_plan_routes_to_row_major_warp_row() {
        assert_eq!(RowMajorWarpRowMatvecPlan::NAME, "row-major-warp-row-matvec");
        assert_eq!(
            <RowMajorWarpRowMatvecPlan as MatvecKernelPlan>::Layout::NAME,
            "row-major"
        );
        assert_eq!(RowMajorWarpRowMatvecPlan::rows_per_block(), 1);
        assert_eq!(RowMajorWarpRowMatvecPlan::block_threads(), 32);
        assert_eq!(RowMajorWarpRowMatvecPlan::grid_rows(4099), 4099);
    }

    #[test]
    fn matvec_plan_routes_to_row_major_warp_rows2() {
        assert_eq!(
            RowMajorWarpRows2MatvecPlan::NAME,
            "row-major-warp-rows2-matvec"
        );
        assert_eq!(
            <RowMajorWarpRows2MatvecPlan as MatvecKernelPlan>::Layout::NAME,
            "row-major"
        );
        assert_eq!(RowMajorWarpRows2MatvecPlan::rows_per_block(), 2);
        assert_eq!(RowMajorWarpRows2MatvecPlan::block_threads(), 64);
        assert_eq!(RowMajorWarpRows2MatvecPlan::grid_rows(4099), 2050);
    }

    #[test]
    fn matvec_plan_routes_to_row_major_warp_rows4() {
        assert_eq!(
            RowMajorWarpRows4MatvecPlan::NAME,
            "row-major-warp-rows4-matvec"
        );
        assert_eq!(
            <RowMajorWarpRows4MatvecPlan as MatvecKernelPlan>::Layout::NAME,
            "row-major"
        );
        assert_eq!(RowMajorWarpRows4MatvecPlan::rows_per_block(), 4);
        assert_eq!(RowMajorWarpRows4MatvecPlan::block_threads(), 128);
        assert_eq!(RowMajorWarpRows4MatvecPlan::grid_rows(4099), 1025);
    }

    #[test]
    fn matvec_plan_routes_to_row_major_warp_rows8() {
        assert_eq!(
            RowMajorWarpRows8MatvecPlan::NAME,
            "row-major-warp-rows8-matvec"
        );
        assert_eq!(
            <RowMajorWarpRows8MatvecPlan as MatvecKernelPlan>::Layout::NAME,
            "row-major"
        );
        assert_eq!(RowMajorWarpRows8MatvecPlan::rows_per_block(), 8);
        assert_eq!(RowMajorWarpRows8MatvecPlan::block_threads(), 256);
        assert_eq!(RowMajorWarpRows8MatvecPlan::grid_rows(4099), 513);
    }

    #[test]
    fn tiled_gemm_plan_carries_layouts_and_tile_shape() {
        type Plan = TiledGemm16Plan<RowMajor, ColumnMajor, RowMajor>;
        assert_eq!(Plan::NAME, "tiled-gemm-16x16x16");
        assert_eq!(<Plan as GemmKernelPlan>::ALayout::NAME, "row-major");
        assert_eq!(<Plan as GemmKernelPlan>::BLayout::NAME, "column-major");
        assert_eq!(<Plan as GemmKernelPlan>::CLayout::NAME, "row-major");
        assert_eq!(Plan::block_dim(), (16, 16, 1));
        assert_eq!(Plan::grid_dim(17, 33), (3, 2, 1));
    }

    #[test]
    fn reduction_plans_encode_thread_counts() {
        assert_eq!(BlockRmsNorm256Plan::NAME, "block-rmsnorm-256");
        assert_eq!(BlockRmsNorm256Plan::block_threads(), 256);
        assert_eq!(
            BlockLogitSelection256Plan::NAME,
            "block-logit-selection-256"
        );
        assert_eq!(BlockLogitSelection256Plan::block_threads(), 256);
    }
}
