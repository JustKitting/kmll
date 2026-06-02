use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_host::cuda_launch;

use crate::{
    dtypes::Bf16,
    layout::{
        ColumnMajor, DeviceMatrix, DeviceMatrixMut, GemmKernelPlan, GemmProblem, Layout2D,
        MatrixLayout, RowMajor, TiledGemm16Plan,
    },
    rowwise_scaled::DeviceRowwiseScaledI8Matrix,
};

// cuda_launch! resolves cuda-oxide's generated __*_CudaKernel marker types by
// bare name, so keep the wildcard import for kernels launched from this module.
use crate::kernels::matmul::*;

pub fn gemm_f32<ALayout, BLayout, CLayout>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &mut DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), DriverError>
where
    ALayout: Layout2D,
    BLayout: Layout2D,
    CLayout: Layout2D,
{
    type Plan<ALayout, BLayout, CLayout> = TiledGemm16Plan<ALayout, BLayout, CLayout>;

    let a = DeviceMatrix::<f32, ALayout>::packed(a, m, k).expect("GEMM A shape mismatch");
    let b = DeviceMatrix::<f32, BLayout>::packed(b, k, n).expect("GEMM B shape mismatch");
    let c = DeviceMatrixMut::<f32, CLayout>::packed(c, m, n).expect("GEMM C shape mismatch");
    let problem = GemmProblem::new(a, b, c, alpha, beta).expect("GEMM operand shape mismatch");
    let m = problem.m();
    let n = problem.n();
    let k = problem.k();
    let alpha = problem.alpha();
    let beta = problem.beta();
    let (a, b, mut c, _, _) = problem.into_parts();
    let a_stride = a.layout().stride();
    let b_stride = b.layout().stride();
    let c_stride = c.layout().stride();

    cuda_launch! {
        kernel: gemm_f32_tiled_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (Plan::<ALayout, BLayout, CLayout>::grid_dim(m, n)),
            block_dim: (Plan::<ALayout, BLayout, CLayout>::block_dim()),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*a.buffer()),
            slice(*b.buffer()),
            m as u32,
            n as u32,
            k as u32,
            a_stride.row as u32,
            a_stride.col as u32,
            b_stride.row as u32,
            b_stride.col as u32,
            c_stride.row as u32,
            c_stride.col as u32,
            alpha,
            beta,
            slice_mut(*c.buffer_mut())
        ]
    }
}

pub fn gemm_f32_bf16<ALayout, BLayout, CLayout>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<Bf16>,
    c: &mut DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), DriverError>
where
    ALayout: Layout2D,
    BLayout: Layout2D,
    CLayout: Layout2D,
{
    type Plan<ALayout, BLayout, CLayout> = TiledGemm16Plan<ALayout, BLayout, CLayout>;

    let a = DeviceMatrix::<f32, ALayout>::packed(a, m, k).expect("GEMM A shape mismatch");
    let b = DeviceMatrix::<Bf16, BLayout>::packed(b, k, n).expect("GEMM B shape mismatch");
    let c = DeviceMatrixMut::<f32, CLayout>::packed(c, m, n).expect("GEMM C shape mismatch");
    let problem = GemmProblem::new(a, b, c, alpha, beta).expect("GEMM operand shape mismatch");
    let m = problem.m();
    let n = problem.n();
    let k = problem.k();
    let alpha = problem.alpha();
    let beta = problem.beta();
    let (a, b, mut c, _, _) = problem.into_parts();
    let a_stride = a.layout().stride();
    let b_stride = b.layout().stride();
    let c_stride = c.layout().stride();

    cuda_launch! {
        kernel: gemm_f32_bf16_tiled_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (Plan::<ALayout, BLayout, CLayout>::grid_dim(m, n)),
            block_dim: (Plan::<ALayout, BLayout, CLayout>::block_dim()),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*a.buffer()),
            slice(*b.buffer()),
            m as u32,
            n as u32,
            k as u32,
            a_stride.row as u32,
            a_stride.col as u32,
            b_stride.row as u32,
            b_stride.col as u32,
            c_stride.row as u32,
            c_stride.col as u32,
            alpha,
            beta,
            slice_mut(*c.buffer_mut())
        ]
    }
}

pub fn gemm_f32_bf16_prefix<ALayout, BLayout, CLayout>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<Bf16>,
    c: &mut DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), DriverError>
where
    ALayout: Layout2D,
    BLayout: Layout2D,
    CLayout: Layout2D,
{
    type Plan<ALayout, BLayout, CLayout> = TiledGemm16Plan<ALayout, BLayout, CLayout>;

    let a_layout = MatrixLayout::<ALayout>::packed(m, k);
    let b_layout = MatrixLayout::<BLayout>::packed(k, n);
    let c_layout = MatrixLayout::<CLayout>::packed(m, n);
    assert!(
        a.len() >= a_layout.capacity(),
        "GEMM A buffer too short: {} < {}",
        a.len(),
        a_layout.capacity()
    );
    assert!(
        b.len() >= b_layout.capacity(),
        "GEMM B buffer too short: {} < {}",
        b.len(),
        b_layout.capacity()
    );
    assert!(
        c.len() >= c_layout.capacity(),
        "GEMM C buffer too short: {} < {}",
        c.len(),
        c_layout.capacity()
    );
    let a_stride = a_layout.stride();
    let b_stride = b_layout.stride();
    let c_stride = c_layout.stride();

    cuda_launch! {
        kernel: gemm_f32_bf16_tiled_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (Plan::<ALayout, BLayout, CLayout>::grid_dim(m, n)),
            block_dim: (Plan::<ALayout, BLayout, CLayout>::block_dim()),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*a),
            slice(*b),
            m as u32,
            n as u32,
            k as u32,
            a_stride.row as u32,
            a_stride.col as u32,
            b_stride.row as u32,
            b_stride.col as u32,
            c_stride.row as u32,
            c_stride.col as u32,
            alpha,
            beta,
            slice_mut(*c)
        ]
    }
}

pub fn gemm_f32_i8_scaled_prefix<ALayout, BLayout, CLayout>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    a: &DeviceBuffer<f32>,
    b: &DeviceRowwiseScaledI8Matrix,
    c: &mut DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), DriverError>
where
    ALayout: Layout2D,
    BLayout: Layout2D,
    CLayout: Layout2D,
{
    type Plan<ALayout, BLayout, CLayout> = TiledGemm16Plan<ALayout, BLayout, CLayout>;

    let a_layout = MatrixLayout::<ALayout>::packed(m, k);
    let b_layout = MatrixLayout::<BLayout>::packed(k, n);
    let c_layout = MatrixLayout::<CLayout>::packed(m, n);
    assert!(
        a.len() >= a_layout.capacity(),
        "scaled-i8 GEMM A buffer too short: {} < {}",
        a.len(),
        a_layout.capacity()
    );
    assert!(
        b.values.len() >= b_layout.capacity(),
        "scaled-i8 GEMM B buffer too short: {} < {}",
        b.values.len(),
        b_layout.capacity()
    );
    assert_eq!(b.rows, n, "scaled-i8 GEMM scale row count mismatch");
    assert_eq!(b.cols, k, "scaled-i8 GEMM weight column count mismatch");
    assert!(
        b.scales.len() >= n,
        "scaled-i8 GEMM scales buffer too short: {} < {}",
        b.scales.len(),
        n
    );
    assert!(
        c.len() >= c_layout.capacity(),
        "scaled-i8 GEMM C buffer too short: {} < {}",
        c.len(),
        c_layout.capacity()
    );
    let a_stride = a_layout.stride();
    let b_stride = b_layout.stride();
    let c_stride = c_layout.stride();
    let b_values = &b.values;
    let b_scales = &b.scales;

    cuda_launch! {
        kernel: gemm_f32_i8_scaled_tiled_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (Plan::<ALayout, BLayout, CLayout>::grid_dim(m, n)),
            block_dim: (Plan::<ALayout, BLayout, CLayout>::block_dim()),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*a),
            slice(*b_values),
            slice(*b_scales),
            m as u32,
            n as u32,
            k as u32,
            a_stride.row as u32,
            a_stride.col as u32,
            b_stride.row as u32,
            b_stride.col as u32,
            c_stride.row as u32,
            c_stride.col as u32,
            alpha,
            beta,
            slice_mut(*c)
        ]
    }
}

pub fn linear_qkv_batched_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    wq: &DeviceBuffer<Bf16>,
    wk: &DeviceBuffer<Bf16>,
    wv: &DeviceBuffer<Bf16>,
    batch: usize,
    input_dim: usize,
    q_output_dim: usize,
    kv_output_dim: usize,
    query_out: &mut DeviceBuffer<f32>,
    key_out: &mut DeviceBuffer<f32>,
    value_out: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let input_len = batch
        .checked_mul(input_dim)
        .expect("batched QKV input shape overflow");
    let q_weight_len = q_output_dim
        .checked_mul(input_dim)
        .expect("batched Q weight shape overflow");
    let kv_weight_len = kv_output_dim
        .checked_mul(input_dim)
        .expect("batched KV weight shape overflow");
    let q_output_len = batch
        .checked_mul(q_output_dim)
        .expect("batched Q output shape overflow");
    let kv_output_len = batch
        .checked_mul(kv_output_dim)
        .expect("batched KV output shape overflow");
    assert!(
        input.len() >= input_len,
        "batched QKV input too short: {} < {}",
        input.len(),
        input_len
    );
    assert_eq!(wq.len(), q_weight_len, "batched Q weight length mismatch");
    assert_eq!(wk.len(), kv_weight_len, "batched K weight length mismatch");
    assert_eq!(wv.len(), kv_weight_len, "batched V weight length mismatch");
    assert!(
        query_out.len() >= q_output_len,
        "batched Q output too short: {} < {}",
        query_out.len(),
        q_output_len
    );
    assert!(
        key_out.len() >= kv_output_len,
        "batched K output too short: {} < {}",
        key_out.len(),
        kv_output_len
    );
    assert!(
        value_out.len() >= kv_output_len,
        "batched V output too short: {} < {}",
        value_out.len(),
        kv_output_len
    );

    gemm_f32_bf16_prefix::<RowMajor, ColumnMajor, RowMajor>(
        stream,
        module,
        input,
        wq,
        query_out,
        batch,
        q_output_dim,
        input_dim,
        1.0,
        0.0,
    )?;
    gemm_f32_bf16_prefix::<RowMajor, ColumnMajor, RowMajor>(
        stream,
        module,
        input,
        wk,
        key_out,
        batch,
        kv_output_dim,
        input_dim,
        1.0,
        0.0,
    )?;
    gemm_f32_bf16_prefix::<RowMajor, ColumnMajor, RowMajor>(
        stream,
        module,
        input,
        wv,
        value_out,
        batch,
        kv_output_dim,
        input_dim,
        1.0,
        0.0,
    )
}

pub fn linear_batched_i8_scaled(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceRowwiseScaledI8Matrix,
    batch: usize,
    input_dim: usize,
    output_dim: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let input_len = batch
        .checked_mul(input_dim)
        .expect("batched scaled-i8 linear input shape overflow");
    let output_len = batch
        .checked_mul(output_dim)
        .expect("batched scaled-i8 linear output shape overflow");
    assert!(
        input.len() >= input_len,
        "batched scaled-i8 linear input too short: {} < {}",
        input.len(),
        input_len
    );
    assert_eq!(
        weight.rows, output_dim,
        "batched scaled-i8 linear output dim mismatch"
    );
    assert_eq!(
        weight.cols, input_dim,
        "batched scaled-i8 linear input dim mismatch"
    );
    assert!(
        output.len() >= output_len,
        "batched scaled-i8 linear output too short: {} < {}",
        output.len(),
        output_len
    );

    gemm_f32_i8_scaled_prefix::<RowMajor, ColumnMajor, RowMajor>(
        stream, module, input, weight, output, batch, output_dim, input_dim, 1.0, 0.0,
    )
}

pub fn linear_qkv_batched_i8_scaled(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    wq: &DeviceRowwiseScaledI8Matrix,
    wk: &DeviceRowwiseScaledI8Matrix,
    wv: &DeviceRowwiseScaledI8Matrix,
    batch: usize,
    input_dim: usize,
    q_output_dim: usize,
    kv_output_dim: usize,
    query_out: &mut DeviceBuffer<f32>,
    key_out: &mut DeviceBuffer<f32>,
    value_out: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    linear_batched_i8_scaled(
        stream,
        module,
        input,
        wq,
        batch,
        input_dim,
        q_output_dim,
        query_out,
    )?;
    linear_batched_i8_scaled(
        stream,
        module,
        input,
        wk,
        batch,
        input_dim,
        kv_output_dim,
        key_out,
    )?;
    linear_batched_i8_scaled(
        stream,
        module,
        input,
        wv,
        batch,
        input_dim,
        kv_output_dim,
        value_out,
    )
}
