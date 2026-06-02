use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

use crate::{
    backends::Cuda,
    dtypes::{AccumulateToF32, Bf16},
    math,
};

const RMSNORM_REDUCE_THREADS: usize = 256;

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
fn qwen_linear_conv_silu_step_impl(
    input: &[f32],
    weight: &[Bf16],
    conv_state_in: &[f32],
    channels: u32,
    kernel_size: u32,
    mut conv_state_out: DisjointSlice<f32>,
    mut out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let channel = idx.get();
    let channels = channels as usize;
    let kernel_size = kernel_size as usize;

    if channel < channels {
        let base = channel * kernel_size;
        let mut k = 0;
        while k + 1 < kernel_size {
            unsafe {
                *conv_state_out.get_unchecked_mut(base + k) = conv_state_in[base + k + 1];
            }
            k += 1;
        }
        unsafe {
            *conv_state_out.get_unchecked_mut(base + kernel_size - 1) = input[channel];
        }

        let mut acc = 0.0;
        let mut k = 0;
        while k < kernel_size {
            let state = if k + 1 < kernel_size {
                conv_state_in[base + k + 1]
            } else {
                input[channel]
            };
            acc += state * weight[base + k].to_f32_accumulator();
            k += 1;
        }
        let silu = acc / (1.0 + math::exp::<Cuda, f32>(-acc));
        unsafe {
            *out.get_unchecked_mut(channel) = silu;
        }
    }
}

#[inline(always)]
fn qwen_split_linear_qkv_impl(
    qkv: &[f32],
    value_heads: u32,
    key_heads: u32,
    key_dim: u32,
    value_dim: u32,
    mut query: DisjointSlice<f32>,
    mut key: DisjointSlice<f32>,
    mut value: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    let query_len = value_heads as usize * key_dim as usize;
    let value_len = value_heads as usize * value_dim as usize;
    let total = if query_len > value_len {
        query_len
    } else {
        value_len
    };

    if i < total {
        let value_heads = value_heads as usize;
        let key_heads = key_heads as usize;
        let key_dim = key_dim as usize;
        let value_dim = value_dim as usize;
        let repeat = value_heads / key_heads;

        if i < value_heads * key_dim {
            let head = i / key_dim;
            let dim = i - head * key_dim;
            let key_head = head / repeat;
            let key_offset = key_head * key_dim + dim;
            unsafe {
                *query.get_unchecked_mut(i) = qkv[key_offset];
                *key.get_unchecked_mut(i) = qkv[key_heads * key_dim + key_offset];
            }
        }
        if i < value_heads * value_dim {
            unsafe {
                *value.get_unchecked_mut(i) = qkv[2 * key_heads * key_dim + i];
            }
        }
    }
}

#[inline(always)]
fn qwen_gated_delta_decay_impl(
    a: &[f32],
    a_log: &[Bf16],
    dt_bias: &[Bf16],
    mut out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out_elem) = out.get_mut(idx) {
        let dt = a[i] + dt_bias[i].to_f32_accumulator();
        let softplus = if dt > 20.0 {
            dt
        } else {
            math::log::<Cuda, f32>(1.0 + math::exp::<Cuda, f32>(dt))
        };
        *out_elem = -math::exp::<Cuda, f32>(a_log[i].to_f32_accumulator()) * softplus;
    }
}

#[inline(always)]
fn qwen_gated_delta_step_impl(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    g: &[f32],
    beta_logits: &[f32],
    state_in: &[f32],
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
    mut out: DisjointSlice<f32>,
    mut state_out: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    let value_heads = value_heads as usize;
    let key_dim = key_dim as usize;
    let value_dim = value_dim as usize;
    let value_len = value_heads * value_dim;

    if i < value_len {
        let head = i / value_dim;
        let value_col = i - head * value_dim;
        let head_offset = head * key_dim;
        let state_head_offset = head * key_dim * value_dim;

        let mut q_sum_sq = 0.0;
        let mut k_sum_sq = 0.0;
        let mut d = 0;
        while d < key_dim {
            let q = query[head_offset + d];
            let k = key[head_offset + d];
            q_sum_sq += q * q;
            k_sum_sq += k * k;
            d += 1;
        }

        let q_inv = math::rsqrt::<Cuda, f32>(q_sum_sq + 1.0e-6);
        let k_inv = math::rsqrt::<Cuda, f32>(k_sum_sq + 1.0e-6);
        let scale = math::rsqrt::<Cuda, f32>(key_dim as f32);
        let decay = math::exp::<Cuda, f32>(g[head]);
        let beta = 1.0 / (1.0 + math::exp::<Cuda, f32>(-beta_logits[head]));

        let mut kv_mem = 0.0;
        d = 0;
        while d < key_dim {
            let k_norm = key[head_offset + d] * k_inv;
            let state_index = state_head_offset + d * value_dim + value_col;
            kv_mem += state_in[state_index] * decay * k_norm;
            d += 1;
        }

        let delta = (value[i] - kv_mem) * beta;
        let mut acc = 0.0;
        d = 0;
        while d < key_dim {
            let q_norm = query[head_offset + d] * q_inv * scale;
            let k_norm = key[head_offset + d] * k_inv;
            let state_index = state_head_offset + d * value_dim + value_col;
            let new_state = state_in[state_index] * decay + k_norm * delta;
            unsafe {
                *state_out.get_unchecked_mut(state_index) = new_state;
            }
            acc += new_state * q_norm;
            d += 1;
        }

        unsafe {
            *out.get_unchecked_mut(i) = acc;
        }
    }
}

#[inline(always)]
fn qwen_gated_rmsnorm_bf16_impl(
    input: &[f32],
    gate: &[f32],
    weight: &[Bf16],
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
        let gate_value = gate[row_offset + i];
        let silu_gate = gate_value / (1.0 + math::exp::<Cuda, f32>(-gate_value));
        let value = input[row_offset + i] * inv_scale * weight[i].to_f32_accumulator() * silu_gate;
        unsafe {
            *out.get_unchecked_mut(row_offset + i) = value;
        }
        i += block_threads;
    }
}

#[kernel]
pub fn qwen_rmsnorm_bf16_kernel(input: &[f32], weight: &[Bf16], eps: f32, out: DisjointSlice<f32>) {
    qwen_rmsnorm_impl(input, weight, eps, out);
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
pub fn qwen_linear_conv_silu_step_kernel(
    input: &[f32],
    weight: &[Bf16],
    conv_state_in: &[f32],
    channels: u32,
    kernel_size: u32,
    conv_state_out: DisjointSlice<f32>,
    out: DisjointSlice<f32>,
) {
    qwen_linear_conv_silu_step_impl(
        input,
        weight,
        conv_state_in,
        channels,
        kernel_size,
        conv_state_out,
        out,
    );
}

#[kernel]
pub fn qwen_split_linear_qkv_kernel(
    qkv: &[f32],
    value_heads: u32,
    key_heads: u32,
    key_dim: u32,
    value_dim: u32,
    query: DisjointSlice<f32>,
    key: DisjointSlice<f32>,
    value: DisjointSlice<f32>,
) {
    qwen_split_linear_qkv_impl(
        qkv,
        value_heads,
        key_heads,
        key_dim,
        value_dim,
        query,
        key,
        value,
    );
}

#[kernel]
pub fn qwen_gated_delta_decay_kernel(
    a: &[f32],
    a_log: &[Bf16],
    dt_bias: &[Bf16],
    out: DisjointSlice<f32>,
) {
    qwen_gated_delta_decay_impl(a, a_log, dt_bias, out);
}

#[kernel]
pub fn qwen_gated_delta_step_kernel(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    g: &[f32],
    beta_logits: &[f32],
    state_in: &[f32],
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
    out: DisjointSlice<f32>,
    state_out: DisjointSlice<f32>,
) {
    qwen_gated_delta_step_impl(
        query,
        key,
        value,
        g,
        beta_logits,
        state_in,
        value_heads,
        key_dim,
        value_dim,
        out,
        state_out,
    );
}

#[kernel]
pub fn qwen_gated_rmsnorm_bf16_kernel(
    input: &[f32],
    gate: &[f32],
    weight: &[Bf16],
    batch: u32,
    dim: u32,
    eps: f32,
    out: DisjointSlice<f32>,
) {
    qwen_gated_rmsnorm_bf16_impl(input, gate, weight, batch, dim, eps, out);
}
