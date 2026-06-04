use std::{
    ffi::c_void,
    sync::{Arc, Mutex, OnceLock},
};

use cuda_core::{CudaContext, CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use crate::dtypes::Bf16;

const GEMM_F32_BF16_TF32_LINEAR_PTX: &str = include_str!("ptx/gemm_f32_bf16_tf32_linear.ptx");
const GEMM_F32_BF16_TF32_LINEAR_SYMBOL: &str = "gemm_f32_bf16_tf32_linear_kernel";
const TILE_M: usize = 16;
const TILE_N: usize = 8;

static TENSOR_CORE_MODULES: OnceLock<Mutex<Vec<(Arc<CudaContext>, Arc<CudaModule>)>>> =
    OnceLock::new();

pub fn try_linear_batched_bf16_tf32(
    stream: &Arc<CudaStream>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    batch: usize,
    input_dim: usize,
    output_dim: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<bool, DriverError> {
    if batch == 0 || input_dim == 0 || output_dim == 0 {
        return Ok(true);
    }
    if !supports_tf32_tensor_cores(stream)? {
        return Ok(false);
    }

    let input_len = batch
        .checked_mul(input_dim)
        .expect("tensor-core batched linear input shape overflow");
    let weight_len = output_dim
        .checked_mul(input_dim)
        .expect("tensor-core batched linear weight shape overflow");
    let output_len = batch
        .checked_mul(output_dim)
        .expect("tensor-core batched linear output shape overflow");
    assert!(
        input.len() >= input_len,
        "tensor-core batched linear input too short: {} < {}",
        input.len(),
        input_len
    );
    assert_eq!(
        weight.len(),
        weight_len,
        "tensor-core batched linear weight length mismatch"
    );
    assert!(
        output.len() >= output_len,
        "tensor-core batched linear output too short: {} < {}",
        output.len(),
        output_len
    );

    let module = tensor_core_module(stream)?;
    let function = module.load_function(GEMM_F32_BF16_TF32_LINEAR_SYMBOL)?;
    let config = LaunchConfig {
        grid_dim: (
            output_dim.div_ceil(TILE_N) as u32,
            batch.div_ceil(TILE_M) as u32,
            1,
        ),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut a_ptr = input.cu_deviceptr();
    let mut b_ptr = weight.cu_deviceptr();
    let mut c_ptr = output.cu_deviceptr();
    let mut m_arg = batch as u32;
    let mut n_arg = output_dim as u32;
    let mut k_arg = input_dim as u32;
    let mut args = vec![
        &mut a_ptr as *mut _ as *mut c_void,
        &mut b_ptr as *mut _ as *mut c_void,
        &mut c_ptr as *mut _ as *mut c_void,
        &mut m_arg as *mut _ as *mut c_void,
        &mut n_arg as *mut _ as *mut c_void,
        &mut k_arg as *mut _ as *mut c_void,
    ];

    unsafe {
        cuda_core::launch_kernel_on_stream(
            &function,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            stream.as_ref(),
            &mut args,
        )?;
    }
    Ok(true)
}

fn supports_tf32_tensor_cores(stream: &Arc<CudaStream>) -> Result<bool, DriverError> {
    let (major, _) = stream.context().compute_capability()?;
    Ok(major >= 8)
}

fn tensor_core_module(stream: &Arc<CudaStream>) -> Result<Arc<CudaModule>, DriverError> {
    let ctx = stream.context().clone();
    let modules = TENSOR_CORE_MODULES.get_or_init(|| Mutex::new(Vec::new()));
    let mut modules = modules.lock().expect("tensor-core module cache poisoned");
    if let Some((_, module)) = modules
        .iter()
        .find(|(module_ctx, _)| module_ctx.as_ref() == ctx.as_ref())
    {
        return Ok(module.clone());
    }

    let module = ctx.load_module_from_ptx_src(GEMM_F32_BF16_TF32_LINEAR_PTX)?;
    modules.push((ctx, module.clone()));
    Ok(module)
}
