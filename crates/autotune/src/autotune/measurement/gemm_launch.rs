use std::sync::Arc;

use cuda_core::{CudaFunction, CudaModule, CudaStream, DeviceBuffer};

use super::super::KernelCandidateMetadata;
use super::KernelAutotuneMeasureResult;
use nn_rust_inference::{
    dtypes::Bf16,
    layout::{ColumnMajor, MatrixLayout, RowMajor},
};

pub(super) struct GeneratedGemmModule {
    pub(super) _module: Arc<CudaModule>,
    pub(super) function: CudaFunction,
}

pub(super) fn launch_generated_gemm_symbol(
    stream: &Arc<CudaStream>,
    generated: &GeneratedGemmModule,
    candidate: &KernelCandidateMetadata,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<Bf16>,
    c: &mut DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> KernelAutotuneMeasureResult<()> {
    let a_layout = MatrixLayout::<RowMajor>::packed(m, k);
    let b_layout = MatrixLayout::<ColumnMajor>::packed(k, n);
    let c_layout = MatrixLayout::<RowMajor>::packed(m, n);
    let a_stride = a_layout.stride();
    let b_stride = b_layout.stride();
    let c_stride = c_layout.stride();

    let mut a_ptr = a.cu_deviceptr();
    let mut a_len = a.len() as u64;
    let mut b_ptr = b.cu_deviceptr();
    let mut b_len = b.len() as u64;
    let mut m_arg = m as u32;
    let mut n_arg = n as u32;
    let mut k_arg = k as u32;
    let mut a_row_stride = a_stride.row as u32;
    let mut a_col_stride = a_stride.col as u32;
    let mut b_row_stride = b_stride.row as u32;
    let mut b_col_stride = b_stride.col as u32;
    let mut c_row_stride = c_stride.row as u32;
    let mut c_col_stride = c_stride.col as u32;
    let mut alpha = 1.0_f32;
    let mut beta = 0.0_f32;
    let mut c_ptr = c.cu_deviceptr();
    let mut c_len = c.len() as u64;
    let mut args = vec![
        &mut a_ptr as *mut _ as *mut std::ffi::c_void,
        &mut a_len as *mut _ as *mut std::ffi::c_void,
        &mut b_ptr as *mut _ as *mut std::ffi::c_void,
        &mut b_len as *mut _ as *mut std::ffi::c_void,
        &mut m_arg as *mut _ as *mut std::ffi::c_void,
        &mut n_arg as *mut _ as *mut std::ffi::c_void,
        &mut k_arg as *mut _ as *mut std::ffi::c_void,
        &mut a_row_stride as *mut _ as *mut std::ffi::c_void,
        &mut a_col_stride as *mut _ as *mut std::ffi::c_void,
        &mut b_row_stride as *mut _ as *mut std::ffi::c_void,
        &mut b_col_stride as *mut _ as *mut std::ffi::c_void,
        &mut c_row_stride as *mut _ as *mut std::ffi::c_void,
        &mut c_col_stride as *mut _ as *mut std::ffi::c_void,
        &mut alpha as *mut _ as *mut std::ffi::c_void,
        &mut beta as *mut _ as *mut std::ffi::c_void,
        &mut c_ptr as *mut _ as *mut std::ffi::c_void,
        &mut c_len as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe {
        cuda_core::launch_kernel_on_stream(
            &generated.function,
            (
                candidate.launch.grid_dim.x,
                candidate.launch.grid_dim.y,
                candidate.launch.grid_dim.z,
            ),
            (
                candidate.launch.block_dim.x,
                candidate.launch.block_dim.y,
                candidate.launch.block_dim.z,
            ),
            candidate.launch.shared_mem_bytes,
            stream.as_ref(),
            &mut args,
        )?;
    }
    Ok(())
}
