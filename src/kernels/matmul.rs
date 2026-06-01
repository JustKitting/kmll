use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

use crate::{
    backends::Cuda,
    dtypes::{AccumulateToF32, Bf16},
};

const GEMM_TILE: usize = 16;
const GEMM_TILE_ELEMS: usize = GEMM_TILE * GEMM_TILE;

#[inline(always)]
fn gemm_f32_accum_tiled_impl<B>(
    a: &[f32],
    b: &[B],
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    c_row_stride: u32,
    c_col_stride: u32,
    alpha: f32,
    beta: f32,
    mut c: DisjointSlice<f32>,
) where
    B: AccumulateToF32<Cuda>,
{
    static mut TILE_A: SharedArray<f32, GEMM_TILE_ELEMS> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, GEMM_TILE_ELEMS> = SharedArray::UNINIT;

    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    if tx >= GEMM_TILE || ty >= GEMM_TILE {
        return;
    }

    let row = thread::blockIdx_y() as usize * GEMM_TILE + ty;
    let col = thread::blockIdx_x() as usize * GEMM_TILE + tx;
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let a_row_stride = a_row_stride as usize;
    let a_col_stride = a_col_stride as usize;
    let b_row_stride = b_row_stride as usize;
    let b_col_stride = b_col_stride as usize;
    let c_row_stride = c_row_stride as usize;
    let c_col_stride = c_col_stride as usize;

    let mut acc = 0.0_f32;
    let tile_count = k.div_ceil(GEMM_TILE);
    let smem_index = ty * GEMM_TILE + tx;

    let mut tile = 0;
    while tile < tile_count {
        let k_base = tile * GEMM_TILE;
        let a_col = k_base + tx;
        let b_row = k_base + ty;

        unsafe {
            if row < m && a_col < k {
                TILE_A[smem_index] = a[row * a_row_stride + a_col * a_col_stride];
            } else {
                TILE_A[smem_index] = 0.0;
            }

            // Packed column-major RHS stores K contiguously for each output
            // column, so load that tile with K on threadIdx.x for coalescing.
            if b_row_stride == 1 {
                let b_tile_row = tx;
                let b_tile_col = ty;
                let b_global_row = k_base + b_tile_row;
                let b_global_col = thread::blockIdx_x() as usize * GEMM_TILE + b_tile_col;
                let b_smem_index = b_tile_row * GEMM_TILE + b_tile_col;
                if b_global_row < k && b_global_col < n {
                    TILE_B[b_smem_index] = b
                        [b_global_row * b_row_stride + b_global_col * b_col_stride]
                        .to_f32_accumulator();
                } else {
                    TILE_B[b_smem_index] = 0.0;
                }
            } else if b_row < k && col < n {
                TILE_B[smem_index] =
                    b[b_row * b_row_stride + col * b_col_stride].to_f32_accumulator();
            } else {
                TILE_B[smem_index] = 0.0;
            }
        }

        thread::sync_threads();

        unsafe {
            let mut kk = 0;
            while kk < GEMM_TILE {
                acc += TILE_A[ty * GEMM_TILE + kk] * TILE_B[kk * GEMM_TILE + tx];
                kk += 1;
            }
        }

        thread::sync_threads();
        tile += 1;
    }

    if row < m && col < n {
        let c_offset = row * c_row_stride + col * c_col_stride;
        unsafe {
            let c_elem = c.get_unchecked_mut(c_offset);
            let current = *c_elem;
            *c_elem = alpha * acc + beta * current;
        }
    }
}

#[inline(always)]
fn gemm_f32_i8_scaled_tiled_impl(
    a: &[f32],
    b: &[i8],
    scales: &[f32],
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    c_row_stride: u32,
    c_col_stride: u32,
    alpha: f32,
    beta: f32,
    mut c: DisjointSlice<f32>,
) {
    static mut TILE_A: SharedArray<f32, GEMM_TILE_ELEMS> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, GEMM_TILE_ELEMS> = SharedArray::UNINIT;

    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    if tx >= GEMM_TILE || ty >= GEMM_TILE {
        return;
    }

    let row = thread::blockIdx_y() as usize * GEMM_TILE + ty;
    let col = thread::blockIdx_x() as usize * GEMM_TILE + tx;
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let a_row_stride = a_row_stride as usize;
    let a_col_stride = a_col_stride as usize;
    let b_row_stride = b_row_stride as usize;
    let b_col_stride = b_col_stride as usize;
    let c_row_stride = c_row_stride as usize;
    let c_col_stride = c_col_stride as usize;

    let mut acc = 0.0_f32;
    let tile_count = k.div_ceil(GEMM_TILE);
    let smem_index = ty * GEMM_TILE + tx;

    let mut tile = 0;
    while tile < tile_count {
        let k_base = tile * GEMM_TILE;
        let a_col = k_base + tx;
        let b_row = k_base + ty;

        unsafe {
            if row < m && a_col < k {
                TILE_A[smem_index] = a[row * a_row_stride + a_col * a_col_stride];
            } else {
                TILE_A[smem_index] = 0.0;
            }

            if b_row_stride == 1 {
                let b_tile_row = tx;
                let b_tile_col = ty;
                let b_global_row = k_base + b_tile_row;
                let b_global_col = thread::blockIdx_x() as usize * GEMM_TILE + b_tile_col;
                let b_smem_index = b_tile_row * GEMM_TILE + b_tile_col;
                if b_global_row < k && b_global_col < n {
                    TILE_B[b_smem_index] =
                        b[b_global_row * b_row_stride + b_global_col * b_col_stride] as f32
                            * scales[b_global_col];
                } else {
                    TILE_B[b_smem_index] = 0.0;
                }
            } else if b_row < k && col < n {
                TILE_B[smem_index] =
                    b[b_row * b_row_stride + col * b_col_stride] as f32 * scales[col];
            } else {
                TILE_B[smem_index] = 0.0;
            }
        }

        thread::sync_threads();

        unsafe {
            let mut kk = 0;
            while kk < GEMM_TILE {
                acc += TILE_A[ty * GEMM_TILE + kk] * TILE_B[kk * GEMM_TILE + tx];
                kk += 1;
            }
        }

        thread::sync_threads();
        tile += 1;
    }

    if row < m && col < n {
        let c_offset = row * c_row_stride + col * c_col_stride;
        unsafe {
            let c_elem = c.get_unchecked_mut(c_offset);
            let current = *c_elem;
            *c_elem = alpha * acc + beta * current;
        }
    }
}

#[kernel]
pub fn gemm_f32_tiled_kernel(
    a: &[f32],
    b: &[f32],
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    c_row_stride: u32,
    c_col_stride: u32,
    alpha: f32,
    beta: f32,
    c: DisjointSlice<f32>,
) {
    gemm_f32_accum_tiled_impl(
        a,
        b,
        m,
        n,
        k,
        a_row_stride,
        a_col_stride,
        b_row_stride,
        b_col_stride,
        c_row_stride,
        c_col_stride,
        alpha,
        beta,
        c,
    );
}

#[kernel]
pub fn gemm_f32_bf16_tiled_kernel(
    a: &[f32],
    b: &[Bf16],
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    c_row_stride: u32,
    c_col_stride: u32,
    alpha: f32,
    beta: f32,
    c: DisjointSlice<f32>,
) {
    gemm_f32_accum_tiled_impl(
        a,
        b,
        m,
        n,
        k,
        a_row_stride,
        a_col_stride,
        b_row_stride,
        b_col_stride,
        c_row_stride,
        c_col_stride,
        alpha,
        beta,
        c,
    );
}

#[kernel]
pub fn gemm_f32_i8_scaled_tiled_kernel(
    a: &[f32],
    b: &[i8],
    scales: &[f32],
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    c_row_stride: u32,
    c_col_stride: u32,
    alpha: f32,
    beta: f32,
    c: DisjointSlice<f32>,
) {
    gemm_f32_i8_scaled_tiled_impl(
        a,
        b,
        scales,
        m,
        n,
        k,
        a_row_stride,
        a_col_stride,
        b_row_stride,
        b_col_stride,
        c_row_stride,
        c_col_stride,
        alpha,
        beta,
        c,
    );
}
