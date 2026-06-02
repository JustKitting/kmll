use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};

use crate::{
    backends::Cuda,
    dtypes::{AccumulateToF32, Bf16},
    math,
};

const RMSNORM_REDUCE_THREADS: usize = 256;
const SOFTMAX_VALUE_THREADS: usize = 256;
const SINGLE_QUERY_ATTENTION_MAX_SEQ: usize = 4096;
const SINGLE_QUERY_ATTENTION_THREADS: usize = 256;
const SINGLE_QUERY_ATTENTION_WARPS: usize = SINGLE_QUERY_ATTENTION_THREADS / 32;
const LOGIT_SELECTION_THREADS: usize = 256;
const LOGIT_SELECTION_TOP_K_CAPACITY: usize = 16;
const LOGIT_SELECTION_SHARED_SLOTS: usize =
    LOGIT_SELECTION_THREADS * LOGIT_SELECTION_TOP_K_CAPACITY;
const MATVEC_TOP1_ROWS_PER_BLOCK_MAX: usize = 8;

#[inline(always)]
fn embedding_impl<W: AccumulateToF32<Cuda>>(
    weight: &[W],
    token_id: u32,
    dim: u32,
    mut out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out_elem) = out.get_mut(idx) {
        let offset = token_id as usize * dim as usize + i;
        *out_elem = weight[offset].to_f32_accumulator();
    }
}

#[inline(always)]
fn embedding_tokens_impl<W: AccumulateToF32<Cuda>>(
    weight: &[W],
    tokens: &[u32],
    token_count: u32,
    dim: u32,
    mut out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    let total = token_count as usize * dim as usize;

    if i < total {
        let dim = dim as usize;
        let token_index = i / dim;
        let col = i - token_index * dim;
        let token_id = tokens[token_index] as usize;
        unsafe {
            *out.get_unchecked_mut(i) = weight[token_id * dim + col].to_f32_accumulator();
        }
    }
}

#[inline(always)]
fn rmsnorm_impl<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    weight: &[W],
    eps: f32,
    mut out: DisjointSlice<f32>,
) {
    static mut PARTIAL_SUMS: SharedArray<f32, RMSNORM_REDUCE_THREADS> = SharedArray::UNINIT;

    let n = out.len();
    if n == 0 {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > RMSNORM_REDUCE_THREADS {
        return;
    }

    let mut sum_sq = 0.0;
    let mut i = tid;
    while i < n {
        let x = input[i];
        sum_sq += x * x;
        i += block_threads;
    }

    unsafe {
        PARTIAL_SUMS[tid] = sum_sq;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                PARTIAL_SUMS[tid] += PARTIAL_SUMS[tid + stride];
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    let inv_scale = unsafe { math::rsqrt::<Cuda, f32>(PARTIAL_SUMS[0] / n as f32 + eps) };
    let mut i = tid;
    while i < n {
        let value = input[i] * inv_scale * weight[i].to_f32_accumulator();
        // SAFETY: each thread writes indices separated by block_threads, so no
        // two threads in the block write the same output element.
        unsafe {
            *out.get_unchecked_mut(i) = value;
        }
        i += block_threads;
    }
}

#[inline(always)]
fn qwen_rmsnorm_impl<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    weight: &[W],
    eps: f32,
    mut out: DisjointSlice<f32>,
) {
    static mut PARTIAL_SUMS: SharedArray<f32, RMSNORM_REDUCE_THREADS> = SharedArray::UNINIT;

    let n = out.len();
    if n == 0 {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > RMSNORM_REDUCE_THREADS {
        return;
    }

    let mut sum_sq = 0.0;
    let mut i = tid;
    while i < n {
        let x = input[i];
        sum_sq += x * x;
        i += block_threads;
    }

    unsafe {
        PARTIAL_SUMS[tid] = sum_sq;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                PARTIAL_SUMS[tid] += PARTIAL_SUMS[tid + stride];
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    let inv_scale = unsafe { math::rsqrt::<Cuda, f32>(PARTIAL_SUMS[0] / n as f32 + eps) };
    let mut i = tid;
    while i < n {
        let value = input[i] * inv_scale * (1.0 + weight[i].to_f32_accumulator());
        unsafe {
            *out.get_unchecked_mut(i) = value;
        }
        i += block_threads;
    }
}

#[inline(always)]
fn rmsnorm_batched_impl<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    weight: &[W],
    batch: u32,
    dim: u32,
    eps: f32,
    mut out: DisjointSlice<f32>,
) {
    static mut PARTIAL_SUMS: SharedArray<f32, RMSNORM_REDUCE_THREADS> = SharedArray::UNINIT;

    let row = thread::blockIdx_x() as usize;
    if row >= batch as usize {
        return;
    }

    let dim = dim as usize;
    if dim == 0 {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > RMSNORM_REDUCE_THREADS {
        return;
    }

    let row_offset = row * dim;
    let mut sum_sq = 0.0;
    let mut i = tid;
    while i < dim {
        let x = input[row_offset + i];
        sum_sq += x * x;
        i += block_threads;
    }

    unsafe {
        PARTIAL_SUMS[tid] = sum_sq;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                PARTIAL_SUMS[tid] += PARTIAL_SUMS[tid + stride];
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    let inv_scale = unsafe { math::rsqrt::<Cuda, f32>(PARTIAL_SUMS[0] / dim as f32 + eps) };
    let mut i = tid;
    while i < dim {
        let value = input[row_offset + i] * inv_scale * weight[i].to_f32_accumulator();
        unsafe {
            *out.get_unchecked_mut(row_offset + i) = value;
        }
        i += block_threads;
    }
}

#[inline(always)]
fn qwen_rmsnorm_batched_impl<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    weight: &[W],
    batch: u32,
    dim: u32,
    eps: f32,
    mut out: DisjointSlice<f32>,
) {
    static mut PARTIAL_SUMS: SharedArray<f32, RMSNORM_REDUCE_THREADS> = SharedArray::UNINIT;

    let row = thread::blockIdx_x() as usize;
    if row >= batch as usize {
        return;
    }

    let dim = dim as usize;
    if dim == 0 {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > RMSNORM_REDUCE_THREADS {
        return;
    }

    let row_offset = row * dim;
    let mut sum_sq = 0.0;
    let mut i = tid;
    while i < dim {
        let x = input[row_offset + i];
        sum_sq += x * x;
        i += block_threads;
    }

    unsafe {
        PARTIAL_SUMS[tid] = sum_sq;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                PARTIAL_SUMS[tid] += PARTIAL_SUMS[tid + stride];
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    let inv_scale = unsafe { math::rsqrt::<Cuda, f32>(PARTIAL_SUMS[0] / dim as f32 + eps) };
    let mut i = tid;
    while i < dim {
        let value = input[row_offset + i] * inv_scale * (1.0 + weight[i].to_f32_accumulator());
        unsafe {
            *out.get_unchecked_mut(row_offset + i) = value;
        }
        i += block_threads;
    }
}

#[inline(always)]
fn warp_reduce_sum(mut acc: f32) -> f32 {
    acc += warp::shuffle_down_f32(acc, 16);
    acc += warp::shuffle_down_f32(acc, 8);
    acc += warp::shuffle_down_f32(acc, 4);
    acc += warp::shuffle_down_f32(acc, 2);
    acc += warp::shuffle_down_f32(acc, 1);
    acc
}

#[inline(always)]
fn matvec_row_accum<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    weight: &[W],
    row: usize,
    lane: u32,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
) -> f32 {
    let mut acc = 0.0;
    let mut col = lane as usize;
    let row_base = row * row_stride;

    while col + 96 < cols {
        let col0 = col;
        let col1 = col + 32;
        let col2 = col + 64;
        let col3 = col + 96;
        acc += weight[row_base + col0 * col_stride].to_f32_accumulator() * input[col0];
        acc += weight[row_base + col1 * col_stride].to_f32_accumulator() * input[col1];
        acc += weight[row_base + col2 * col_stride].to_f32_accumulator() * input[col2];
        acc += weight[row_base + col3 * col_stride].to_f32_accumulator() * input[col3];
        col += 128;
    }

    while col < cols {
        acc += weight[row_base + col * col_stride].to_f32_accumulator() * input[col];
        col += 32;
    }

    acc
}

#[inline(always)]
fn matvec_impl<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    weight: &[W],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    mut out: DisjointSlice<f32>,
) {
    let thread_x = thread::threadIdx_x();
    let row_in_block = thread_x / 32;
    if row_in_block >= rows_per_block {
        return;
    }

    let row = (thread::blockIdx_x() * rows_per_block + row_in_block) as usize;
    if row >= rows as usize {
        return;
    }

    let lane = warp::lane_id();
    let cols = cols as usize;
    let row_stride = row_stride as usize;
    let col_stride = col_stride as usize;
    let acc = warp_reduce_sum(matvec_row_accum(
        input, weight, row, lane, cols, row_stride, col_stride,
    ));

    if lane == 0 {
        // SAFETY: the launch maps one warp lane-0 writer to each output row.
        unsafe {
            *out.get_unchecked_mut(row) = acc;
        }
    }
}

#[inline(always)]
fn matvec_top1_stage_impl<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    weight: &[W],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    mut partial_tokens: DisjointSlice<u32>,
    mut partial_logits: DisjointSlice<f32>,
) {
    static mut BLOCK_TOKENS: SharedArray<u32, MATVEC_TOP1_ROWS_PER_BLOCK_MAX> = SharedArray::UNINIT;
    static mut BLOCK_LOGITS: SharedArray<f32, MATVEC_TOP1_ROWS_PER_BLOCK_MAX> = SharedArray::UNINIT;

    let thread_x = thread::threadIdx_x();
    let row_in_block = thread_x / 32;
    if row_in_block >= rows_per_block || rows_per_block as usize > MATVEC_TOP1_ROWS_PER_BLOCK_MAX {
        return;
    }

    let row_u32 = thread::blockIdx_x() * rows_per_block + row_in_block;
    let row = row_u32 as usize;
    let valid_row = row < rows as usize;
    let lane = warp::lane_id();
    let cols = cols as usize;
    let row_stride = row_stride as usize;
    let col_stride = col_stride as usize;
    let mut acc = if valid_row {
        matvec_row_accum(input, weight, row, lane, cols, row_stride, col_stride)
    } else {
        f32::NEG_INFINITY
    };

    acc = warp_reduce_sum(acc);

    if lane == 0 {
        let slot = row_in_block as usize;
        unsafe {
            BLOCK_TOKENS[slot] = if valid_row { row_u32 } else { 0xffff_ffff };
            BLOCK_LOGITS[slot] = acc;
        }
    }
    thread::sync_threads();

    if thread_x == 0 {
        let mut best_token = unsafe { BLOCK_TOKENS[0] };
        let mut best_logit = unsafe { BLOCK_LOGITS[0] };
        let mut i = 1;
        while i < rows_per_block as usize && i < MATVEC_TOP1_ROWS_PER_BLOCK_MAX {
            let candidate_token = unsafe { BLOCK_TOKENS[i] };
            let candidate_logit = unsafe { BLOCK_LOGITS[i] };
            if candidate_logit > best_logit
                || (candidate_logit == best_logit && candidate_token < best_token)
            {
                best_token = candidate_token;
                best_logit = candidate_logit;
            }
            i += 1;
        }

        let block = thread::blockIdx_x() as usize;
        unsafe {
            *partial_tokens.get_unchecked_mut(block) = best_token;
            *partial_logits.get_unchecked_mut(block) = best_logit;
        }
    }
}

#[inline(always)]
fn matvec_top1_i8_scaled_stage_impl(
    input: &[f32],
    weight: &[i8],
    scales: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    mut partial_tokens: DisjointSlice<u32>,
    mut partial_logits: DisjointSlice<f32>,
) {
    static mut BLOCK_TOKENS: SharedArray<u32, MATVEC_TOP1_ROWS_PER_BLOCK_MAX> = SharedArray::UNINIT;
    static mut BLOCK_LOGITS: SharedArray<f32, MATVEC_TOP1_ROWS_PER_BLOCK_MAX> = SharedArray::UNINIT;

    let thread_x = thread::threadIdx_x();
    let row_in_block = thread_x / 32;
    if row_in_block >= rows_per_block || rows_per_block as usize > MATVEC_TOP1_ROWS_PER_BLOCK_MAX {
        return;
    }

    let row_u32 = thread::blockIdx_x() * rows_per_block + row_in_block;
    let row = row_u32 as usize;
    let valid_row = row < rows as usize;
    let lane = warp::lane_id();
    let cols = cols as usize;
    let row_stride = row_stride as usize;
    let col_stride = col_stride as usize;
    let mut acc = f32::NEG_INFINITY;

    if valid_row {
        let scale = scales[row];
        let row_base = row * row_stride;
        let mut partial = 0.0_f32;
        let mut col = lane as usize;
        while col + 96 < cols {
            let col0 = col;
            let col1 = col + 32;
            let col2 = col + 64;
            let col3 = col + 96;
            partial += weight[row_base + col0 * col_stride] as f32 * scale * input[col0];
            partial += weight[row_base + col1 * col_stride] as f32 * scale * input[col1];
            partial += weight[row_base + col2 * col_stride] as f32 * scale * input[col2];
            partial += weight[row_base + col3 * col_stride] as f32 * scale * input[col3];
            col += 128;
        }
        while col < cols {
            partial += weight[row_base + col * col_stride] as f32 * scale * input[col];
            col += 32;
        }
        acc = partial;
    }

    acc = warp_reduce_sum(acc);

    if lane == 0 {
        let slot = row_in_block as usize;
        unsafe {
            BLOCK_TOKENS[slot] = if valid_row { row_u32 } else { 0xffff_ffff };
            BLOCK_LOGITS[slot] = acc;
        }
    }
    thread::sync_threads();

    if thread_x == 0 {
        let mut best_token = unsafe { BLOCK_TOKENS[0] };
        let mut best_logit = unsafe { BLOCK_LOGITS[0] };
        let mut i = 1;
        while i < rows_per_block as usize && i < MATVEC_TOP1_ROWS_PER_BLOCK_MAX {
            let candidate_token = unsafe { BLOCK_TOKENS[i] };
            let candidate_logit = unsafe { BLOCK_LOGITS[i] };
            if candidate_logit > best_logit
                || (candidate_logit == best_logit && candidate_token < best_token)
            {
                best_token = candidate_token;
                best_logit = candidate_logit;
            }
            i += 1;
        }

        let block = thread::blockIdx_x() as usize;
        unsafe {
            *partial_tokens.get_unchecked_mut(block) = best_token;
            *partial_logits.get_unchecked_mut(block) = best_logit;
        }
    }
}

#[inline(always)]
fn matvec_residual_impl<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    weight: &[W],
    residual: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    mut out: DisjointSlice<f32>,
) {
    let thread_x = thread::threadIdx_x();
    let row_in_block = thread_x / 32;
    if row_in_block >= rows_per_block {
        return;
    }

    let row = (thread::blockIdx_x() * rows_per_block + row_in_block) as usize;
    if row >= rows as usize {
        return;
    }

    let lane = warp::lane_id();
    let cols = cols as usize;
    let row_stride = row_stride as usize;
    let col_stride = col_stride as usize;
    let acc = warp_reduce_sum(matvec_row_accum(
        input, weight, row, lane, cols, row_stride, col_stride,
    ));

    if lane == 0 {
        unsafe {
            *out.get_unchecked_mut(row) = residual[row] + acc;
        }
    }
}

#[inline(always)]
fn matvec_pair_impl<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    weight_a: &[W],
    weight_b: &[W],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    mut out_a: DisjointSlice<f32>,
    mut out_b: DisjointSlice<f32>,
) {
    let thread_x = thread::threadIdx_x();
    let row_in_block = thread_x / 32;
    if row_in_block >= rows_per_block {
        return;
    }

    let row = (thread::blockIdx_x() * rows_per_block + row_in_block) as usize;
    if row >= rows as usize {
        return;
    }

    let lane = warp::lane_id();
    let cols = cols as usize;
    let row_stride = row_stride as usize;
    let col_stride = col_stride as usize;
    let mut acc_a = 0.0;
    let mut acc_b = 0.0;
    let mut col = lane as usize;
    let row_base = row * row_stride;

    while col + 96 < cols {
        let col0 = col;
        let col1 = col + 32;
        let col2 = col + 64;
        let col3 = col + 96;
        let x0 = input[col0];
        let x1 = input[col1];
        let x2 = input[col2];
        let x3 = input[col3];
        let offset0 = row_base + col0 * col_stride;
        let offset1 = row_base + col1 * col_stride;
        let offset2 = row_base + col2 * col_stride;
        let offset3 = row_base + col3 * col_stride;
        acc_a += weight_a[offset0].to_f32_accumulator() * x0;
        acc_b += weight_b[offset0].to_f32_accumulator() * x0;
        acc_a += weight_a[offset1].to_f32_accumulator() * x1;
        acc_b += weight_b[offset1].to_f32_accumulator() * x1;
        acc_a += weight_a[offset2].to_f32_accumulator() * x2;
        acc_b += weight_b[offset2].to_f32_accumulator() * x2;
        acc_a += weight_a[offset3].to_f32_accumulator() * x3;
        acc_b += weight_b[offset3].to_f32_accumulator() * x3;
        col += 128;
    }

    while col < cols {
        let weight_offset = row_base + col * col_stride;
        let x = input[col];
        acc_a += weight_a[weight_offset].to_f32_accumulator() * x;
        acc_b += weight_b[weight_offset].to_f32_accumulator() * x;
        col += 32;
    }

    acc_a = warp_reduce_sum(acc_a);
    acc_b = warp_reduce_sum(acc_b);

    if lane == 0 {
        // SAFETY: the launch maps one warp lane-0 writer to each output row.
        unsafe {
            *out_a.get_unchecked_mut(row) = acc_a;
            *out_b.get_unchecked_mut(row) = acc_b;
        }
    }
}

#[inline(always)]
fn matvec_silu_gate_up_impl<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    gate_weight: &[W],
    up_weight: &[W],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    mut out: DisjointSlice<f32>,
) {
    let thread_x = thread::threadIdx_x();
    let row_in_block = thread_x / 32;
    if row_in_block >= rows_per_block {
        return;
    }

    let row = (thread::blockIdx_x() * rows_per_block + row_in_block) as usize;
    if row >= rows as usize {
        return;
    }

    let lane = warp::lane_id();
    let cols = cols as usize;
    let row_stride = row_stride as usize;
    let col_stride = col_stride as usize;
    let mut gate_acc = 0.0;
    let mut up_acc = 0.0;
    let mut col = lane as usize;
    let row_base = row * row_stride;

    while col + 224 < cols {
        let col0 = col;
        let col1 = col + 32;
        let col2 = col + 64;
        let col3 = col + 96;
        let col4 = col + 128;
        let col5 = col + 160;
        let col6 = col + 192;
        let col7 = col + 224;
        let x0 = input[col0];
        let x1 = input[col1];
        let x2 = input[col2];
        let x3 = input[col3];
        let x4 = input[col4];
        let x5 = input[col5];
        let x6 = input[col6];
        let x7 = input[col7];
        let offset0 = row_base + col0 * col_stride;
        let offset1 = row_base + col1 * col_stride;
        let offset2 = row_base + col2 * col_stride;
        let offset3 = row_base + col3 * col_stride;
        let offset4 = row_base + col4 * col_stride;
        let offset5 = row_base + col5 * col_stride;
        let offset6 = row_base + col6 * col_stride;
        let offset7 = row_base + col7 * col_stride;
        gate_acc += gate_weight[offset0].to_f32_accumulator() * x0;
        up_acc += up_weight[offset0].to_f32_accumulator() * x0;
        gate_acc += gate_weight[offset1].to_f32_accumulator() * x1;
        up_acc += up_weight[offset1].to_f32_accumulator() * x1;
        gate_acc += gate_weight[offset2].to_f32_accumulator() * x2;
        up_acc += up_weight[offset2].to_f32_accumulator() * x2;
        gate_acc += gate_weight[offset3].to_f32_accumulator() * x3;
        up_acc += up_weight[offset3].to_f32_accumulator() * x3;
        gate_acc += gate_weight[offset4].to_f32_accumulator() * x4;
        up_acc += up_weight[offset4].to_f32_accumulator() * x4;
        gate_acc += gate_weight[offset5].to_f32_accumulator() * x5;
        up_acc += up_weight[offset5].to_f32_accumulator() * x5;
        gate_acc += gate_weight[offset6].to_f32_accumulator() * x6;
        up_acc += up_weight[offset6].to_f32_accumulator() * x6;
        gate_acc += gate_weight[offset7].to_f32_accumulator() * x7;
        up_acc += up_weight[offset7].to_f32_accumulator() * x7;
        col += 256;
    }

    while col < cols {
        let weight_offset = row_base + col * col_stride;
        let x = input[col];
        gate_acc += gate_weight[weight_offset].to_f32_accumulator() * x;
        up_acc += up_weight[weight_offset].to_f32_accumulator() * x;
        col += 32;
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc = warp_reduce_sum(up_acc);

    if lane == 0 {
        unsafe {
            *out.get_unchecked_mut(row) = math::silu::<Cuda, f32>(gate_acc) * up_acc;
        }
    }
}

#[inline(always)]
fn matvec_triple_impl<W: AccumulateToF32<Cuda>>(
    input: &[f32],
    weight_a: &[W],
    weight_b: &[W],
    weight_c: &[W],
    rows_a: u32,
    rows_b: u32,
    rows_c: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    mut out_a: DisjointSlice<f32>,
    mut out_b: DisjointSlice<f32>,
    mut out_c: DisjointSlice<f32>,
) {
    let thread_x = thread::threadIdx_x();
    let row_in_block = thread_x / 32;
    if row_in_block >= rows_per_block {
        return;
    }

    let row = (thread::blockIdx_x() * rows_per_block + row_in_block) as usize;
    let rows_a = rows_a as usize;
    let rows_b = rows_b as usize;
    let rows_c = rows_c as usize;
    if row >= rows_a && row >= rows_b && row >= rows_c {
        return;
    }

    let lane = warp::lane_id();
    let cols = cols as usize;
    let row_stride = row_stride as usize;
    let col_stride = col_stride as usize;
    let has_a = row < rows_a;
    let has_b = row < rows_b;
    let has_c = row < rows_c;
    let mut acc_a = 0.0;
    let mut acc_b = 0.0;
    let mut acc_c = 0.0;
    let mut col = lane as usize;
    let row_base = row * row_stride;

    while col + 96 < cols {
        let col0 = col;
        let col1 = col + 32;
        let col2 = col + 64;
        let col3 = col + 96;
        let x0 = input[col0];
        let x1 = input[col1];
        let x2 = input[col2];
        let x3 = input[col3];
        let offset0 = row_base + col0 * col_stride;
        let offset1 = row_base + col1 * col_stride;
        let offset2 = row_base + col2 * col_stride;
        let offset3 = row_base + col3 * col_stride;
        if has_a {
            acc_a += weight_a[offset0].to_f32_accumulator() * x0;
            acc_a += weight_a[offset1].to_f32_accumulator() * x1;
            acc_a += weight_a[offset2].to_f32_accumulator() * x2;
            acc_a += weight_a[offset3].to_f32_accumulator() * x3;
        }
        if has_b {
            acc_b += weight_b[offset0].to_f32_accumulator() * x0;
            acc_b += weight_b[offset1].to_f32_accumulator() * x1;
            acc_b += weight_b[offset2].to_f32_accumulator() * x2;
            acc_b += weight_b[offset3].to_f32_accumulator() * x3;
        }
        if has_c {
            acc_c += weight_c[offset0].to_f32_accumulator() * x0;
            acc_c += weight_c[offset1].to_f32_accumulator() * x1;
            acc_c += weight_c[offset2].to_f32_accumulator() * x2;
            acc_c += weight_c[offset3].to_f32_accumulator() * x3;
        }
        col += 128;
    }

    while col < cols {
        let weight_offset = row_base + col * col_stride;
        let x = input[col];
        if has_a {
            acc_a += weight_a[weight_offset].to_f32_accumulator() * x;
        }
        if has_b {
            acc_b += weight_b[weight_offset].to_f32_accumulator() * x;
        }
        if has_c {
            acc_c += weight_c[weight_offset].to_f32_accumulator() * x;
        }
        col += 32;
    }

    acc_a = warp_reduce_sum(acc_a);
    acc_b = warp_reduce_sum(acc_b);
    acc_c = warp_reduce_sum(acc_c);

    if lane == 0 {
        // SAFETY: the launch maps one warp lane-0 writer to each output row.
        unsafe {
            if has_a {
                *out_a.get_unchecked_mut(row) = acc_a;
            }
            if has_b {
                *out_b.get_unchecked_mut(row) = acc_b;
            }
            if has_c {
                *out_c.get_unchecked_mut(row) = acc_c;
            }
        }
    }
}

#[inline(always)]
fn matvec_i8_scaled_impl(
    input: &[f32],
    weight: &[i8],
    scales: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    mut out: DisjointSlice<f32>,
) {
    let thread_x = thread::threadIdx_x();
    let row_in_block = thread_x / 32;
    if row_in_block >= rows_per_block {
        return;
    }

    let row = (thread::blockIdx_x() * rows_per_block + row_in_block) as usize;
    if row >= rows as usize {
        return;
    }

    let lane = warp::lane_id();
    let cols = cols as usize;
    let row_stride = row_stride as usize;
    let col_stride = col_stride as usize;
    let scale = scales[row];
    let mut acc = 0.0;
    let mut col = lane as usize;
    let row_base = row * row_stride;

    while col + 96 < cols {
        let col0 = col;
        let col1 = col + 32;
        let col2 = col + 64;
        let col3 = col + 96;
        acc += weight[row_base + col0 * col_stride] as f32 * scale * input[col0];
        acc += weight[row_base + col1 * col_stride] as f32 * scale * input[col1];
        acc += weight[row_base + col2 * col_stride] as f32 * scale * input[col2];
        acc += weight[row_base + col3 * col_stride] as f32 * scale * input[col3];
        col += 128;
    }

    while col < cols {
        let weight_offset = row_base + col * col_stride;
        acc += weight[weight_offset] as f32 * scale * input[col];
        col += 32;
    }

    acc = warp_reduce_sum(acc);

    if lane == 0 {
        // SAFETY: the launch maps exactly one warp-sized block to each output
        // row and only lane 0 writes, so each row is written by one thread.
        unsafe {
            *out.get_unchecked_mut(row) = acc;
        }
    }
}

#[inline(always)]
fn copy_matrix_row_to_vector_impl(
    matrix: &[f32],
    row: u32,
    cols: u32,
    mut out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let col = idx.get();
    if col < cols as usize {
        unsafe {
            *out.get_unchecked_mut(col) = matrix[row as usize * cols as usize + col];
        }
    }
}

#[inline(always)]
fn copy_vector_to_matrix_row_impl(
    input: &[f32],
    row: u32,
    cols: u32,
    mut matrix: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let col = idx.get();
    if col < cols as usize {
        unsafe {
            *matrix.get_unchecked_mut(row as usize * cols as usize + col) = input[col];
        }
    }
}

#[inline(always)]
fn rope_matrix_row_value(
    matrix: &[f32],
    row_base: usize,
    i: usize,
    freqs: &[f32],
    position: u32,
    head_dim: usize,
) -> f32 {
    let half = freqs.len();
    let rotary_dim = half * 2;
    let head = i / head_dim;
    let dim = i - head * head_dim;
    if dim >= rotary_dim {
        return matrix[row_base + i];
    }
    let pair_dim = if dim < half { dim } else { dim - half };
    let base = row_base + head * head_dim;
    let first = matrix[base + pair_dim];
    let second = matrix[base + pair_dim + half];
    let angle = position as f32 * freqs[pair_dim];
    let sin = math::sin::<Cuda, f32>(angle);
    let cos = math::cos::<Cuda, f32>(angle);

    if dim < half {
        first * cos - second * sin
    } else {
        second * cos + first * sin
    }
}

#[inline(always)]
fn rope_vector_value(
    input: &[f32],
    i: usize,
    freqs: &[f32],
    position: u32,
    head_dim: usize,
) -> f32 {
    let half = freqs.len();
    let rotary_dim = half * 2;
    let head = i / head_dim;
    let dim = i - head * head_dim;
    if dim >= rotary_dim {
        return input[i];
    }
    let pair_dim = if dim < half { dim } else { dim - half };
    let base = head * head_dim;
    let first = input[base + pair_dim];
    let second = input[base + pair_dim + half];
    let angle = position as f32 * freqs[pair_dim];
    let sin = math::sin::<Cuda, f32>(angle);
    let cos = math::cos::<Cuda, f32>(angle);

    if dim < half {
        first * cos - second * sin
    } else {
        second * cos + first * sin
    }
}

#[inline(always)]
fn prepare_prefill_attention_batch_impl(
    query_batch: &[f32],
    key_batch: &[f32],
    value_batch: &[f32],
    freqs: &[f32],
    prompt_len: u32,
    q_len: u32,
    kv_len: u32,
    head_dim: u32,
    mut query_rot_batch: DisjointSlice<f32>,
    mut key_cache: DisjointSlice<f32>,
    mut value_cache: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    let prompt_len = prompt_len as usize;
    let q_len = q_len as usize;
    let kv_len = kv_len as usize;
    let head_dim = head_dim as usize;
    let width = if q_len > kv_len { q_len } else { kv_len };
    let total = prompt_len * width;

    if i >= total {
        return;
    }

    let position = i / width;
    let col = i - position * width;
    if col < q_len {
        let query_base = position * q_len;
        let value = rope_matrix_row_value(
            query_batch,
            query_base,
            col,
            freqs,
            position as u32,
            head_dim,
        );
        unsafe {
            *query_rot_batch.get_unchecked_mut(query_base + col) = value;
        }
    }

    if col < kv_len {
        let row_base = position * kv_len;
        let key_value =
            rope_matrix_row_value(key_batch, row_base, col, freqs, position as u32, head_dim);
        let cache_offset = row_base + col;
        unsafe {
            *key_cache.get_unchecked_mut(cache_offset) = key_value;
            *value_cache.get_unchecked_mut(cache_offset) = value_batch[row_base + col];
        }
    }
}

#[inline(always)]
fn prepare_incremental_attention_impl(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    freqs: &[f32],
    position: u32,
    q_len: u32,
    kv_len: u32,
    head_dim: u32,
    mut query_rot: DisjointSlice<f32>,
    mut key_cache: DisjointSlice<f32>,
    mut value_cache: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    let q_len = q_len as usize;
    let kv_len = kv_len as usize;
    let head_dim = head_dim as usize;
    let position = position as usize;

    if i < q_len {
        let query_value = rope_vector_value(query, i, freqs, position as u32, head_dim);
        unsafe {
            *query_rot.get_unchecked_mut(i) = query_value;
        }
    }

    if i < kv_len {
        let key_value = rope_vector_value(key, i, freqs, position as u32, head_dim);
        let cache_offset = position * kv_len + i;
        unsafe {
            *key_cache.get_unchecked_mut(cache_offset) = key_value;
            *value_cache.get_unchecked_mut(cache_offset) = value[i];
        }
    }
}

#[inline(always)]
fn silu_mul_impl(gate: &[f32], up: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = math::silu::<Cuda, f32>(gate[i]) * up[i];
    }
}

#[inline(always)]
fn sigmoid_mul_impl(gate: &[f32], up: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out_elem) = out.get_mut(idx) {
        let sigmoid = 1.0 / (1.0 + math::exp::<Cuda, f32>(-gate[i]));
        *out_elem = sigmoid * up[i];
    }
}

#[inline(always)]
fn qwen_split_query_gate_impl(
    q_gate: &[f32],
    n_heads: u32,
    head_dim: u32,
    mut query: DisjointSlice<f32>,
    mut gate: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    let q_len = n_heads as usize * head_dim as usize;

    if i < q_len {
        let head_dim = head_dim as usize;
        let head = i / head_dim;
        let dim = i - head * head_dim;
        let src = head * head_dim * 2 + dim;
        unsafe {
            *query.get_unchecked_mut(i) = q_gate[src];
            *gate.get_unchecked_mut(i) = q_gate[src + head_dim];
        }
    }
}

#[inline(always)]
fn add_impl(lhs: &[f32], rhs: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = lhs[i] + rhs[i];
    }
}

#[inline(always)]
fn silu_mul_n_impl(gate: &[f32], up: &[f32], count: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();

    if i < count as usize {
        unsafe {
            *out.get_unchecked_mut(i) = math::silu::<Cuda, f32>(gate[i]) * up[i];
        }
    }
}

#[inline(always)]
fn add_n_impl(lhs: &[f32], rhs: &[f32], count: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();

    if i < count as usize {
        unsafe {
            *out.get_unchecked_mut(i) = lhs[i] + rhs[i];
        }
    }
}

#[inline(always)]
fn single_token_gqa_impl(
    value: &[f32],
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    mut out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out_elem) = out.get_mut(idx) {
        let head_dim = head_dim as usize;
        let q_head = i / head_dim;
        let dim = i - q_head * head_dim;
        let heads_per_kv = n_heads as usize / n_kv_heads as usize;
        let kv_head = q_head / heads_per_kv;
        *out_elem = value[kv_head * head_dim + dim];
    }
}

#[inline(always)]
fn apply_rope_impl(
    input: &[f32],
    freqs: &[f32],
    position: u32,
    head_dim: u32,
    mut out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out_elem) = out.get_mut(idx) {
        let head_dim = head_dim as usize;
        let half = freqs.len();
        let rotary_dim = half * 2;
        let head = i / head_dim;
        let dim = i - head * head_dim;
        if dim >= rotary_dim {
            *out_elem = input[i];
            return;
        }
        let pair_dim = if dim < half { dim } else { dim - half };
        let base = head * head_dim;
        let first = input[base + pair_dim];
        let second = input[base + pair_dim + half];
        let angle = position as f32 * freqs[pair_dim];
        let sin = math::sin::<Cuda, f32>(angle);
        let cos = math::cos::<Cuda, f32>(angle);

        *out_elem = if dim < half {
            first * cos - second * sin
        } else {
            second * cos + first * sin
        };
    }
}

#[inline(always)]
fn apply_rope_write_kv_cache_impl(
    input: &[f32],
    freqs: &[f32],
    position: u32,
    head_dim: u32,
    mut cache: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();

    if i < input.len() {
        let head_dim = head_dim as usize;
        let half = freqs.len();
        let rotary_dim = half * 2;
        let head = i / head_dim;
        let dim = i - head * head_dim;
        if dim >= rotary_dim {
            let offset = position as usize * input.len() + i;
            unsafe {
                *cache.get_unchecked_mut(offset) = input[i];
            }
            return;
        }
        let pair_dim = if dim < half { dim } else { dim - half };
        let base = head * head_dim;
        let first = input[base + pair_dim];
        let second = input[base + pair_dim + half];
        let angle = position as f32 * freqs[pair_dim];
        let sin = math::sin::<Cuda, f32>(angle);
        let cos = math::cos::<Cuda, f32>(angle);
        let value = if dim < half {
            first * cos - second * sin
        } else {
            second * cos + first * sin
        };
        let offset = position as usize * input.len() + i;

        // SAFETY: `i` is unique per thread, so `position * input.len() + i`
        // is unique for this launch. The host wrapper checks cache capacity.
        unsafe {
            *cache.get_unchecked_mut(offset) = value;
        }
    }
}

#[inline(always)]
fn write_kv_cache_impl(input: &[f32], position: u32, mut cache: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();

    if i < input.len() {
        let offset = position as usize * input.len() + i;
        // SAFETY: `i` is unique per thread, so `position * input.len() + i`
        // is unique for this kernel launch. The host wrapper bounds-checks
        // the cache capacity for the requested position.
        unsafe {
            *cache.get_unchecked_mut(offset) = input[i];
        }
    }
}

#[inline(always)]
fn attention_scores_impl(
    query: &[f32],
    key_cache: &[f32],
    seq_len: u32,
    max_seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scores_per_block: u32,
    mut scores: DisjointSlice<f32>,
) {
    let seq_len = seq_len as usize;
    let max_seq_len = max_seq_len as usize;
    let n_heads = n_heads as usize;
    let n_kv_heads = n_kv_heads as usize;
    let head_dim = head_dim as usize;

    let thread_x = thread::threadIdx_x();
    let score_in_block = thread_x / 32;
    if score_in_block >= scores_per_block {
        return;
    }

    let score_index = (thread::blockIdx_x() * scores_per_block + score_in_block) as usize;
    let active_scores = n_heads * seq_len;
    if score_index >= active_scores {
        return;
    }

    let head = score_index / seq_len;
    let pos = score_index - head * seq_len;
    let lane = warp::lane_id();
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_head = head / heads_per_kv;
    let q_offset = head * head_dim;
    let k_offset = pos * n_kv_heads * head_dim + kv_head * head_dim;
    let mut dot = 0.0;
    let mut d = lane as usize;
    while d < head_dim {
        dot += query[q_offset + d] * key_cache[k_offset + d];
        d += 32;
    }

    dot += warp::shuffle_down_f32(dot, 16);
    dot += warp::shuffle_down_f32(dot, 8);
    dot += warp::shuffle_down_f32(dot, 4);
    dot += warp::shuffle_down_f32(dot, 2);
    dot += warp::shuffle_down_f32(dot, 1);

    if lane == 0 {
        let score_offset = head * max_seq_len + pos;
        // SAFETY: launch indices map one-to-one to `(head, pos)` score slots.
        unsafe {
            *scores.get_unchecked_mut(score_offset) =
                dot * math::rsqrt::<Cuda, f32>(head_dim as f32);
        }
    }
}

#[inline(always)]
fn attention_scores_from_matrix_row_impl(
    query_batch: &[f32],
    key_cache: &[f32],
    query_row: u32,
    seq_len: u32,
    max_seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scores_per_block: u32,
    mut scores: DisjointSlice<f32>,
) {
    let seq_len = seq_len as usize;
    let max_seq_len = max_seq_len as usize;
    let n_heads = n_heads as usize;
    let n_kv_heads = n_kv_heads as usize;
    let head_dim = head_dim as usize;
    let query_row = query_row as usize;
    let q_len = n_heads * head_dim;

    let thread_x = thread::threadIdx_x();
    let score_in_block = thread_x / 32;
    if score_in_block >= scores_per_block {
        return;
    }

    let score_index = (thread::blockIdx_x() * scores_per_block + score_in_block) as usize;
    let active_scores = n_heads * seq_len;
    if score_index >= active_scores {
        return;
    }

    let head = score_index / seq_len;
    let pos = score_index - head * seq_len;
    let lane = warp::lane_id();
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_head = head / heads_per_kv;
    let q_offset = query_row * q_len + head * head_dim;
    let k_offset = pos * n_kv_heads * head_dim + kv_head * head_dim;
    let mut dot = 0.0;
    let mut d = lane as usize;
    while d < head_dim {
        dot += query_batch[q_offset + d] * key_cache[k_offset + d];
        d += 32;
    }

    dot += warp::shuffle_down_f32(dot, 16);
    dot += warp::shuffle_down_f32(dot, 8);
    dot += warp::shuffle_down_f32(dot, 4);
    dot += warp::shuffle_down_f32(dot, 2);
    dot += warp::shuffle_down_f32(dot, 1);

    if lane == 0 {
        let score_offset = head * max_seq_len + pos;
        unsafe {
            *scores.get_unchecked_mut(score_offset) =
                dot * math::rsqrt::<Cuda, f32>(head_dim as f32);
        }
    }
}

#[inline(always)]
fn softmax_value_impl(
    scores: &[f32],
    value_cache: &[f32],
    seq_len: u32,
    max_seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    out: DisjointSlice<f32>,
) {
    softmax_value_write_impl(
        scores,
        value_cache,
        seq_len,
        max_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        0,
        out,
    );
}

#[inline(always)]
fn softmax_value_write_impl(
    scores: &[f32],
    value_cache: &[f32],
    seq_len: u32,
    max_seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    output_base: u32,
    mut out: DisjointSlice<f32>,
) {
    static mut SOFTMAX_SCRATCH: SharedArray<f32, SOFTMAX_VALUE_THREADS> = SharedArray::UNINIT;

    let seq_len = seq_len as usize;
    let max_seq_len = max_seq_len as usize;
    let n_heads = n_heads as usize;
    let n_kv_heads = n_kv_heads as usize;
    let head_dim = head_dim as usize;
    let output_base = output_base as usize;
    let head = thread::blockIdx_x() as usize;

    if head >= n_heads {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > SOFTMAX_VALUE_THREADS {
        return;
    }

    let score_base = head * max_seq_len;
    let mut max_score = f32::NEG_INFINITY;
    let mut pos = tid;
    while pos < seq_len {
        let score = scores[score_base + pos];
        if score > max_score {
            max_score = score;
        }
        pos += block_threads;
    }

    unsafe {
        SOFTMAX_SCRATCH[tid] = max_score;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                if SOFTMAX_SCRATCH[tid + stride] > SOFTMAX_SCRATCH[tid] {
                    SOFTMAX_SCRATCH[tid] = SOFTMAX_SCRATCH[tid + stride];
                }
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    let max_score = unsafe { SOFTMAX_SCRATCH[0] };
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_head = head / heads_per_kv;
    let mut denom = 0.0;
    pos = tid;
    while pos < seq_len {
        let weight = math::exp::<Cuda, f32>(scores[score_base + pos] - max_score);
        denom += weight;
        pos += block_threads;
    }

    unsafe {
        SOFTMAX_SCRATCH[tid] = denom;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                SOFTMAX_SCRATCH[tid] += SOFTMAX_SCRATCH[tid + stride];
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    let denom = unsafe { SOFTMAX_SCRATCH[0] };
    let mut dim = tid;
    while dim < head_dim {
        let mut acc = 0.0;
        pos = 0;
        while pos < seq_len {
            let weight = math::exp::<Cuda, f32>(scores[score_base + pos] - max_score);
            let value = value_cache[pos * n_kv_heads * head_dim + kv_head * head_dim + dim];
            acc += weight * value;
            pos += 1;
        }

        // SAFETY: this kernel maps one CTA to each head and strides dimensions
        // by thread id, so each `(head, dim)` output is written once.
        unsafe {
            *out.get_unchecked_mut(output_base + head * head_dim + dim) = acc / denom;
        }
        dim += block_threads;
    }
}

#[inline(always)]
fn single_query_attention_impl(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    mut out: DisjointSlice<f32>,
) {
    static mut SCORES: SharedArray<f32, SINGLE_QUERY_ATTENTION_MAX_SEQ> = SharedArray::UNINIT;
    static mut PARTIALS: SharedArray<f32, SINGLE_QUERY_ATTENTION_THREADS> = SharedArray::UNINIT;

    let seq_len = seq_len as usize;
    let n_heads = n_heads as usize;
    let n_kv_heads = n_kv_heads as usize;
    let head_dim = head_dim as usize;
    let head = thread::blockIdx_x() as usize;
    if head >= n_heads || seq_len > SINGLE_QUERY_ATTENTION_MAX_SEQ {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > SINGLE_QUERY_ATTENTION_THREADS {
        return;
    }

    let lane = warp::lane_id();
    let warp_index = tid / 32;
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_head = head / heads_per_kv;
    let q_offset = head * head_dim;
    let score_scale = math::rsqrt::<Cuda, f32>(head_dim as f32);

    let mut pos = warp_index;
    while pos < seq_len {
        let k_offset = pos * n_kv_heads * head_dim + kv_head * head_dim;
        let mut dot = 0.0;
        let mut dim = lane as usize;
        while dim < head_dim {
            dot += query[q_offset + dim] * key_cache[k_offset + dim];
            dim += 32;
        }

        dot += warp::shuffle_down_f32(dot, 16);
        dot += warp::shuffle_down_f32(dot, 8);
        dot += warp::shuffle_down_f32(dot, 4);
        dot += warp::shuffle_down_f32(dot, 2);
        dot += warp::shuffle_down_f32(dot, 1);

        if lane == 0 {
            unsafe {
                SCORES[pos] = dot * score_scale;
            }
        }
        pos += SINGLE_QUERY_ATTENTION_WARPS;
    }
    thread::sync_threads();

    let mut max_score = f32::NEG_INFINITY;
    pos = tid;
    while pos < seq_len {
        let score = unsafe { SCORES[pos] };
        if score > max_score {
            max_score = score;
        }
        pos += block_threads;
    }

    unsafe {
        PARTIALS[tid] = max_score;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                if PARTIALS[tid + stride] > PARTIALS[tid] {
                    PARTIALS[tid] = PARTIALS[tid + stride];
                }
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    let max_score = unsafe { PARTIALS[0] };
    let mut denom = 0.0;
    pos = tid;
    while pos < seq_len {
        let weight = math::exp::<Cuda, f32>(unsafe { SCORES[pos] } - max_score);
        denom += weight;
        pos += block_threads;
    }

    unsafe {
        PARTIALS[tid] = denom;
    }
    thread::sync_threads();

    stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                PARTIALS[tid] += PARTIALS[tid + stride];
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    let denom = unsafe { PARTIALS[0] };
    let mut dim = tid;
    while dim < head_dim {
        let mut acc = 0.0;
        pos = 0;
        while pos < seq_len {
            let weight = math::exp::<Cuda, f32>(unsafe { SCORES[pos] } - max_score);
            let value = value_cache[pos * n_kv_heads * head_dim + kv_head * head_dim + dim];
            acc += weight * value;
            pos += 1;
        }

        unsafe {
            *out.get_unchecked_mut(head * head_dim + dim) = acc / denom;
        }
        dim += block_threads;
    }
}

#[inline(always)]
fn prefill_causal_attention_impl(
    query_batch: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    prompt_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    mut out: DisjointSlice<f32>,
) {
    static mut SCORES: SharedArray<f32, SINGLE_QUERY_ATTENTION_MAX_SEQ> = SharedArray::UNINIT;
    static mut PARTIALS: SharedArray<f32, SINGLE_QUERY_ATTENTION_THREADS> = SharedArray::UNINIT;

    let prompt_len = prompt_len as usize;
    let n_heads = n_heads as usize;
    let n_kv_heads = n_kv_heads as usize;
    let head_dim = head_dim as usize;
    let head = thread::blockIdx_x() as usize;
    let row = thread::blockIdx_y() as usize;
    if head >= n_heads || row >= prompt_len || prompt_len > SINGLE_QUERY_ATTENTION_MAX_SEQ {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > SINGLE_QUERY_ATTENTION_THREADS {
        return;
    }

    let seq_len = row + 1;
    let lane = warp::lane_id();
    let warp_index = tid / 32;
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_head = head / heads_per_kv;
    let q_len = n_heads * head_dim;
    let q_offset = row * q_len + head * head_dim;
    let score_scale = math::rsqrt::<Cuda, f32>(head_dim as f32);

    let mut pos = warp_index;
    while pos < seq_len {
        let k_offset = pos * n_kv_heads * head_dim + kv_head * head_dim;
        let mut dot = 0.0;
        let mut dim = lane as usize;
        while dim < head_dim {
            dot += query_batch[q_offset + dim] * key_cache[k_offset + dim];
            dim += 32;
        }

        dot += warp::shuffle_down_f32(dot, 16);
        dot += warp::shuffle_down_f32(dot, 8);
        dot += warp::shuffle_down_f32(dot, 4);
        dot += warp::shuffle_down_f32(dot, 2);
        dot += warp::shuffle_down_f32(dot, 1);

        if lane == 0 {
            unsafe {
                SCORES[pos] = dot * score_scale;
            }
        }
        pos += SINGLE_QUERY_ATTENTION_WARPS;
    }
    thread::sync_threads();

    let mut max_score = f32::NEG_INFINITY;
    pos = tid;
    while pos < seq_len {
        let score = unsafe { SCORES[pos] };
        if score > max_score {
            max_score = score;
        }
        pos += block_threads;
    }

    unsafe {
        PARTIALS[tid] = max_score;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                if PARTIALS[tid + stride] > PARTIALS[tid] {
                    PARTIALS[tid] = PARTIALS[tid + stride];
                }
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    let max_score = unsafe { PARTIALS[0] };
    let mut denom = 0.0;
    pos = tid;
    while pos < seq_len {
        let weight = math::exp::<Cuda, f32>(unsafe { SCORES[pos] } - max_score);
        denom += weight;
        pos += block_threads;
    }

    unsafe {
        PARTIALS[tid] = denom;
    }
    thread::sync_threads();

    stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                PARTIALS[tid] += PARTIALS[tid + stride];
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    let denom = unsafe { PARTIALS[0] };
    let mut dim = tid;
    while dim < head_dim {
        let mut acc = 0.0;
        pos = 0;
        while pos < seq_len {
            let weight = math::exp::<Cuda, f32>(unsafe { SCORES[pos] } - max_score);
            let value = value_cache[pos * n_kv_heads * head_dim + kv_head * head_dim + dim];
            acc += weight * value;
            pos += 1;
        }

        unsafe {
            *out.get_unchecked_mut(row * q_len + head * head_dim + dim) = acc / denom;
        }
        dim += block_threads;
    }
}

#[inline(always)]
fn argmax_f32_impl(
    logits: &[f32],
    mut token_out: DisjointSlice<u32>,
    mut logit_out: DisjointSlice<f32>,
) {
    static mut BEST_TOKENS: SharedArray<u32, LOGIT_SELECTION_THREADS> = SharedArray::UNINIT;
    static mut BEST_LOGITS: SharedArray<f32, LOGIT_SELECTION_THREADS> = SharedArray::UNINIT;

    if logits.is_empty() {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > LOGIT_SELECTION_THREADS {
        return;
    }

    let mut best_token = tid;
    let mut best_logit = f32::NEG_INFINITY;
    let mut i = tid;
    while i < logits.len() {
        let candidate = logits[i];
        if candidate > best_logit || (candidate == best_logit && i < best_token) {
            best_token = i;
            best_logit = candidate;
        }
        i += block_threads;
    }

    unsafe {
        BEST_TOKENS[tid] = best_token as u32;
        BEST_LOGITS[tid] = best_logit;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                let other_logit = BEST_LOGITS[tid + stride];
                let other_token = BEST_TOKENS[tid + stride];
                if other_logit > BEST_LOGITS[tid]
                    || (other_logit == BEST_LOGITS[tid] && other_token < BEST_TOKENS[tid])
                {
                    BEST_LOGITS[tid] = other_logit;
                    BEST_TOKENS[tid] = other_token;
                }
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        unsafe {
            *token_out.get_unchecked_mut(0) = BEST_TOKENS[0];
            *logit_out.get_unchecked_mut(0) = BEST_LOGITS[0];
        }
    }
}

#[inline(always)]
fn argmax_f32_packed_impl(logits: &[f32], mut packed_out: DisjointSlice<u64>) {
    static mut BEST_TOKENS: SharedArray<u32, LOGIT_SELECTION_THREADS> = SharedArray::UNINIT;
    static mut BEST_LOGITS: SharedArray<f32, LOGIT_SELECTION_THREADS> = SharedArray::UNINIT;

    if logits.is_empty() {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > LOGIT_SELECTION_THREADS {
        return;
    }

    let mut best_token = tid;
    let mut best_logit = f32::NEG_INFINITY;
    let mut i = tid;
    while i < logits.len() {
        let candidate = logits[i];
        if candidate > best_logit || (candidate == best_logit && i < best_token) {
            best_token = i;
            best_logit = candidate;
        }
        i += block_threads;
    }

    unsafe {
        BEST_TOKENS[tid] = best_token as u32;
        BEST_LOGITS[tid] = best_logit;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                let other_logit = BEST_LOGITS[tid + stride];
                let other_token = BEST_TOKENS[tid + stride];
                if other_logit > BEST_LOGITS[tid]
                    || (other_logit == BEST_LOGITS[tid] && other_token < BEST_TOKENS[tid])
                {
                    BEST_LOGITS[tid] = other_logit;
                    BEST_TOKENS[tid] = other_token;
                }
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let token = unsafe { BEST_TOKENS[0] } as u64;
        let logit_bits = unsafe { BEST_LOGITS[0] }.to_bits() as u64;
        unsafe {
            *packed_out.get_unchecked_mut(0) = (logit_bits << 32) | token;
        }
    }
}

#[inline(always)]
fn argmax_pairs_f32_packed_impl(
    tokens: &[u32],
    logits: &[f32],
    count: u32,
    mut packed_out: DisjointSlice<u64>,
) {
    static mut BEST_TOKENS: SharedArray<u32, LOGIT_SELECTION_THREADS> = SharedArray::UNINIT;
    static mut BEST_LOGITS: SharedArray<f32, LOGIT_SELECTION_THREADS> = SharedArray::UNINIT;

    let count = count as usize;
    if count == 0 {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > LOGIT_SELECTION_THREADS {
        return;
    }

    let mut best_token = 0xffff_ffff_u32;
    let mut best_logit = f32::NEG_INFINITY;
    let mut i = tid;
    while i < count {
        let candidate_token = tokens[i];
        let candidate_logit = logits[i];
        if candidate_logit > best_logit
            || (candidate_logit == best_logit && candidate_token < best_token)
        {
            best_token = candidate_token;
            best_logit = candidate_logit;
        }
        i += block_threads;
    }

    unsafe {
        BEST_TOKENS[tid] = best_token;
        BEST_LOGITS[tid] = best_logit;
    }
    thread::sync_threads();

    let mut stride = block_threads / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                let other_logit = BEST_LOGITS[tid + stride];
                let other_token = BEST_TOKENS[tid + stride];
                if other_logit > BEST_LOGITS[tid]
                    || (other_logit == BEST_LOGITS[tid] && other_token < BEST_TOKENS[tid])
                {
                    BEST_LOGITS[tid] = other_logit;
                    BEST_TOKENS[tid] = other_token;
                }
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let token = unsafe { BEST_TOKENS[0] } as u64;
        let logit_bits = unsafe { BEST_LOGITS[0] }.to_bits() as u64;
        unsafe {
            *packed_out.get_unchecked_mut(0) = (logit_bits << 32) | token;
        }
    }
}

#[inline(always)]
fn top_k_f32_impl(
    logits: &[f32],
    k: u32,
    mut token_out: DisjointSlice<u32>,
    mut logit_out: DisjointSlice<f32>,
) {
    static mut CANDIDATE_TOKENS: SharedArray<u32, LOGIT_SELECTION_SHARED_SLOTS> =
        SharedArray::UNINIT;
    static mut CANDIDATE_LOGITS: SharedArray<f32, LOGIT_SELECTION_SHARED_SLOTS> =
        SharedArray::UNINIT;

    if logits.is_empty() {
        return;
    }

    let requested_k = k as usize;
    let k = if requested_k > LOGIT_SELECTION_TOP_K_CAPACITY {
        LOGIT_SELECTION_TOP_K_CAPACITY
    } else {
        requested_k
    };
    if k == 0 {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    if block_threads > LOGIT_SELECTION_THREADS {
        return;
    }

    let mut top_tokens = [0u32; LOGIT_SELECTION_TOP_K_CAPACITY];
    let mut top_logits = [f32::NEG_INFINITY; LOGIT_SELECTION_TOP_K_CAPACITY];

    let mut token = tid;
    while token < logits.len() {
        let logit = logits[token];
        let candidate_token = token as u32;
        let mut rank = 0;
        while rank < k {
            let current_logit = top_logits[rank];
            let current_token = top_tokens[rank];
            if logit > current_logit || (logit == current_logit && candidate_token < current_token)
            {
                let mut shift = k - 1;
                while shift > rank {
                    top_tokens[shift] = top_tokens[shift - 1];
                    top_logits[shift] = top_logits[shift - 1];
                    shift -= 1;
                }
                top_tokens[rank] = candidate_token;
                top_logits[rank] = logit;
                break;
            }
            rank += 1;
        }
        token += block_threads;
    }

    let base = tid * LOGIT_SELECTION_TOP_K_CAPACITY;
    let mut rank = 0;
    while rank < k {
        unsafe {
            CANDIDATE_TOKENS[base + rank] = top_tokens[rank];
            CANDIDATE_LOGITS[base + rank] = top_logits[rank];
        }
        rank += 1;
    }
    thread::sync_threads();

    if tid == 0 {
        let mut final_tokens = [0u32; LOGIT_SELECTION_TOP_K_CAPACITY];
        let mut final_logits = [f32::NEG_INFINITY; LOGIT_SELECTION_TOP_K_CAPACITY];
        let mut candidate_thread = 0;
        while candidate_thread < block_threads {
            let candidate_base = candidate_thread * LOGIT_SELECTION_TOP_K_CAPACITY;
            rank = 0;
            while rank < k {
                unsafe {
                    let candidate_token = CANDIDATE_TOKENS[candidate_base + rank];
                    let candidate_logit = CANDIDATE_LOGITS[candidate_base + rank];
                    let mut final_rank = 0;
                    while final_rank < k {
                        let current_logit = final_logits[final_rank];
                        let current_token = final_tokens[final_rank];
                        if candidate_logit > current_logit
                            || (candidate_logit == current_logit && candidate_token < current_token)
                        {
                            let mut shift = k - 1;
                            while shift > final_rank {
                                final_tokens[shift] = final_tokens[shift - 1];
                                final_logits[shift] = final_logits[shift - 1];
                                shift -= 1;
                            }
                            final_tokens[final_rank] = candidate_token;
                            final_logits[final_rank] = candidate_logit;
                            break;
                        }
                        final_rank += 1;
                    }
                }
                rank += 1;
            }
            candidate_thread += 1;
        }

        rank = 0;
        while rank < k {
            unsafe {
                *token_out.get_unchecked_mut(rank) = final_tokens[rank];
                *logit_out.get_unchecked_mut(rank) = final_logits[rank];
            }
            rank += 1;
        }
    }
}

#[kernel]
pub fn embedding_bf16_kernel(weight: &[Bf16], token_id: u32, dim: u32, out: DisjointSlice<f32>) {
    embedding_impl(weight, token_id, dim, out);
}

#[kernel]
pub fn embedding_tokens_bf16_kernel(
    weight: &[Bf16],
    tokens: &[u32],
    token_count: u32,
    dim: u32,
    out: DisjointSlice<f32>,
) {
    embedding_tokens_impl(weight, tokens, token_count, dim, out);
}

#[kernel]
pub fn rmsnorm_bf16_kernel(input: &[f32], weight: &[Bf16], eps: f32, out: DisjointSlice<f32>) {
    rmsnorm_impl(input, weight, eps, out);
}

#[kernel]
pub fn qwen_rmsnorm_bf16_kernel(input: &[f32], weight: &[Bf16], eps: f32, out: DisjointSlice<f32>) {
    qwen_rmsnorm_impl(input, weight, eps, out);
}

#[kernel]
pub fn rmsnorm_batched_bf16_kernel(
    input: &[f32],
    weight: &[Bf16],
    batch: u32,
    dim: u32,
    eps: f32,
    out: DisjointSlice<f32>,
) {
    rmsnorm_batched_impl(input, weight, batch, dim, eps, out);
}

#[kernel]
pub fn qwen_rmsnorm_batched_bf16_kernel(
    input: &[f32],
    weight: &[Bf16],
    batch: u32,
    dim: u32,
    eps: f32,
    out: DisjointSlice<f32>,
) {
    qwen_rmsnorm_batched_impl(input, weight, batch, dim, eps, out);
}

#[kernel]
pub fn matvec_bf16_kernel(
    input: &[f32],
    weight: &[Bf16],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    out: DisjointSlice<f32>,
) {
    matvec_impl(
        input,
        weight,
        rows,
        cols,
        row_stride,
        col_stride,
        rows_per_block,
        out,
    );
}

#[kernel]
pub fn matvec_residual_bf16_kernel(
    input: &[f32],
    weight: &[Bf16],
    residual: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    out: DisjointSlice<f32>,
) {
    matvec_residual_impl(
        input,
        weight,
        residual,
        rows,
        cols,
        row_stride,
        col_stride,
        rows_per_block,
        out,
    );
}

#[kernel]
pub fn matvec_pair_bf16_kernel(
    input: &[f32],
    weight_a: &[Bf16],
    weight_b: &[Bf16],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    out_a: DisjointSlice<f32>,
    out_b: DisjointSlice<f32>,
) {
    matvec_pair_impl(
        input,
        weight_a,
        weight_b,
        rows,
        cols,
        row_stride,
        col_stride,
        rows_per_block,
        out_a,
        out_b,
    );
}

#[kernel]
pub fn matvec_silu_gate_up_bf16_kernel(
    input: &[f32],
    gate_weight: &[Bf16],
    up_weight: &[Bf16],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    out: DisjointSlice<f32>,
) {
    matvec_silu_gate_up_impl(
        input,
        gate_weight,
        up_weight,
        rows,
        cols,
        row_stride,
        col_stride,
        rows_per_block,
        out,
    );
}

#[kernel]
pub fn matvec_triple_bf16_kernel(
    input: &[f32],
    weight_a: &[Bf16],
    weight_b: &[Bf16],
    weight_c: &[Bf16],
    rows_a: u32,
    rows_b: u32,
    rows_c: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    out_a: DisjointSlice<f32>,
    out_b: DisjointSlice<f32>,
    out_c: DisjointSlice<f32>,
) {
    matvec_triple_impl(
        input,
        weight_a,
        weight_b,
        weight_c,
        rows_a,
        rows_b,
        rows_c,
        cols,
        row_stride,
        col_stride,
        rows_per_block,
        out_a,
        out_b,
        out_c,
    );
}

#[kernel]
pub fn matvec_i8_scaled_kernel(
    input: &[f32],
    weight: &[i8],
    scales: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    out: DisjointSlice<f32>,
) {
    matvec_i8_scaled_impl(
        input,
        weight,
        scales,
        rows,
        cols,
        row_stride,
        col_stride,
        rows_per_block,
        out,
    );
}

#[kernel]
pub fn silu_mul_kernel(gate: &[f32], up: &[f32], out: DisjointSlice<f32>) {
    silu_mul_impl(gate, up, out);
}

#[kernel]
pub fn sigmoid_mul_kernel(gate: &[f32], up: &[f32], out: DisjointSlice<f32>) {
    sigmoid_mul_impl(gate, up, out);
}

#[kernel]
pub fn qwen_split_query_gate_kernel(
    q_gate: &[f32],
    n_heads: u32,
    head_dim: u32,
    query: DisjointSlice<f32>,
    gate: DisjointSlice<f32>,
) {
    qwen_split_query_gate_impl(q_gate, n_heads, head_dim, query, gate);
}

#[kernel]
pub fn silu_mul_n_kernel(gate: &[f32], up: &[f32], count: u32, out: DisjointSlice<f32>) {
    silu_mul_n_impl(gate, up, count, out);
}

#[kernel]
pub fn add_kernel(lhs: &[f32], rhs: &[f32], out: DisjointSlice<f32>) {
    add_impl(lhs, rhs, out);
}

#[kernel]
pub fn add_n_kernel(lhs: &[f32], rhs: &[f32], count: u32, out: DisjointSlice<f32>) {
    add_n_impl(lhs, rhs, count, out);
}

#[kernel]
pub fn copy_matrix_row_to_vector_kernel(
    matrix: &[f32],
    row: u32,
    cols: u32,
    out: DisjointSlice<f32>,
) {
    copy_matrix_row_to_vector_impl(matrix, row, cols, out);
}

#[kernel]
pub fn copy_vector_to_matrix_row_kernel(
    input: &[f32],
    row: u32,
    cols: u32,
    matrix: DisjointSlice<f32>,
) {
    copy_vector_to_matrix_row_impl(input, row, cols, matrix);
}

#[kernel]
pub fn prepare_prefill_attention_batch_kernel(
    query_batch: &[f32],
    key_batch: &[f32],
    value_batch: &[f32],
    freqs: &[f32],
    prompt_len: u32,
    q_len: u32,
    kv_len: u32,
    head_dim: u32,
    query_rot_batch: DisjointSlice<f32>,
    key_cache: DisjointSlice<f32>,
    value_cache: DisjointSlice<f32>,
) {
    prepare_prefill_attention_batch_impl(
        query_batch,
        key_batch,
        value_batch,
        freqs,
        prompt_len,
        q_len,
        kv_len,
        head_dim,
        query_rot_batch,
        key_cache,
        value_cache,
    );
}

#[kernel]
pub fn prepare_incremental_attention_kernel(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    freqs: &[f32],
    position: u32,
    q_len: u32,
    kv_len: u32,
    head_dim: u32,
    query_rot: DisjointSlice<f32>,
    key_cache: DisjointSlice<f32>,
    value_cache: DisjointSlice<f32>,
) {
    prepare_incremental_attention_impl(
        query,
        key,
        value,
        freqs,
        position,
        q_len,
        kv_len,
        head_dim,
        query_rot,
        key_cache,
        value_cache,
    );
}

#[kernel]
pub fn single_token_gqa_kernel(
    value: &[f32],
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    out: DisjointSlice<f32>,
) {
    single_token_gqa_impl(value, n_heads, n_kv_heads, head_dim, out);
}

#[kernel]
pub fn apply_rope_kernel(
    input: &[f32],
    freqs: &[f32],
    position: u32,
    head_dim: u32,
    out: DisjointSlice<f32>,
) {
    apply_rope_impl(input, freqs, position, head_dim, out);
}

#[kernel]
pub fn apply_rope_write_kv_cache_kernel(
    input: &[f32],
    freqs: &[f32],
    position: u32,
    head_dim: u32,
    cache: DisjointSlice<f32>,
) {
    apply_rope_write_kv_cache_impl(input, freqs, position, head_dim, cache);
}

#[kernel]
pub fn write_kv_cache_kernel(input: &[f32], position: u32, cache: DisjointSlice<f32>) {
    write_kv_cache_impl(input, position, cache);
}

#[kernel]
pub fn attention_scores_kernel(
    query: &[f32],
    key_cache: &[f32],
    seq_len: u32,
    max_seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scores_per_block: u32,
    scores: DisjointSlice<f32>,
) {
    attention_scores_impl(
        query,
        key_cache,
        seq_len,
        max_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        scores_per_block,
        scores,
    );
}

#[kernel]
pub fn attention_scores_from_matrix_row_kernel(
    query_batch: &[f32],
    key_cache: &[f32],
    query_row: u32,
    seq_len: u32,
    max_seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scores_per_block: u32,
    scores: DisjointSlice<f32>,
) {
    attention_scores_from_matrix_row_impl(
        query_batch,
        key_cache,
        query_row,
        seq_len,
        max_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        scores_per_block,
        scores,
    );
}

#[kernel]
pub fn softmax_value_kernel(
    scores: &[f32],
    value_cache: &[f32],
    seq_len: u32,
    max_seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    out: DisjointSlice<f32>,
) {
    softmax_value_impl(
        scores,
        value_cache,
        seq_len,
        max_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        out,
    );
}

#[kernel]
pub fn softmax_value_to_matrix_row_kernel(
    scores: &[f32],
    value_cache: &[f32],
    seq_len: u32,
    max_seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    output_row: u32,
    out: DisjointSlice<f32>,
) {
    softmax_value_write_impl(
        scores,
        value_cache,
        seq_len,
        max_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        output_row * n_heads * head_dim,
        out,
    );
}

#[kernel]
pub fn single_query_attention_kernel(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    out: DisjointSlice<f32>,
) {
    single_query_attention_impl(
        query,
        key_cache,
        value_cache,
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        out,
    );
}

#[kernel]
pub fn prefill_causal_attention_kernel(
    query_batch: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    prompt_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    out: DisjointSlice<f32>,
) {
    prefill_causal_attention_impl(
        query_batch,
        key_cache,
        value_cache,
        prompt_len,
        n_heads,
        n_kv_heads,
        head_dim,
        out,
    );
}

#[kernel]
pub fn argmax_f32_kernel(
    logits: &[f32],
    token_out: DisjointSlice<u32>,
    logit_out: DisjointSlice<f32>,
) {
    argmax_f32_impl(logits, token_out, logit_out);
}

#[kernel]
pub fn argmax_f32_packed_kernel(logits: &[f32], packed_out: DisjointSlice<u64>) {
    argmax_f32_packed_impl(logits, packed_out);
}

#[kernel]
pub fn matvec_top1_bf16_stage_kernel(
    input: &[f32],
    weight: &[Bf16],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    partial_tokens: DisjointSlice<u32>,
    partial_logits: DisjointSlice<f32>,
) {
    matvec_top1_stage_impl(
        input,
        weight,
        rows,
        cols,
        row_stride,
        col_stride,
        rows_per_block,
        partial_tokens,
        partial_logits,
    );
}

#[kernel]
pub fn matvec_top1_i8_scaled_stage_kernel(
    input: &[f32],
    weight: &[i8],
    scales: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    rows_per_block: u32,
    partial_tokens: DisjointSlice<u32>,
    partial_logits: DisjointSlice<f32>,
) {
    matvec_top1_i8_scaled_stage_impl(
        input,
        weight,
        scales,
        rows,
        cols,
        row_stride,
        col_stride,
        rows_per_block,
        partial_tokens,
        partial_logits,
    );
}

#[kernel]
pub fn argmax_pairs_f32_packed_kernel(
    tokens: &[u32],
    logits: &[f32],
    count: u32,
    packed_out: DisjointSlice<u64>,
) {
    argmax_pairs_f32_packed_impl(tokens, logits, count, packed_out);
}

#[kernel]
pub fn top_k_f32_kernel(
    logits: &[f32],
    k: u32,
    token_out: DisjointSlice<u32>,
    logit_out: DisjointSlice<f32>,
) {
    top_k_f32_impl(logits, k, token_out, logit_out);
}
