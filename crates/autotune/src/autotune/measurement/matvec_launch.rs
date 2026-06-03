use std::sync::Arc;

use cuda_core::{CudaFunction, CudaModule, CudaStream, DeviceBuffer};

use super::super::{KernelCandidateMetadata, ScheduleTransform};
use super::{KernelAutotuneMeasureResult, invalid_input};
use nn_rust_inference::{
    dtypes::Bf16,
    layout::{MatrixLayout, RowMajor},
};

pub(super) struct GeneratedMatvecModule {
    pub(super) _module: Arc<CudaModule>,
    pub(super) function: CudaFunction,
}

pub(super) fn launch_matvec_symbol(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    candidate: &KernelCandidateMetadata,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
) -> KernelAutotuneMeasureResult<()> {
    let weight_layout = MatrixLayout::<RowMajor>::packed(rows, cols);
    let stride = weight_layout.stride();
    let rows_per_block = matvec_rows_per_block(candidate)?;

    let mut input_ptr = input.cu_deviceptr();
    let mut input_len = input.len() as u64;
    let mut weight_ptr = weight.cu_deviceptr();
    let mut weight_len = weight.len() as u64;
    let mut rows_arg = rows as u32;
    let mut cols_arg = cols as u32;
    let mut row_stride = stride.row as u32;
    let mut col_stride = stride.col as u32;
    let mut rows_per_block_arg = rows_per_block;
    let mut output_ptr = output.cu_deviceptr();
    let mut output_len = output.len() as u64;
    let mut args = vec![
        &mut input_ptr as *mut _ as *mut std::ffi::c_void,
        &mut input_len as *mut _ as *mut std::ffi::c_void,
        &mut weight_ptr as *mut _ as *mut std::ffi::c_void,
        &mut weight_len as *mut _ as *mut std::ffi::c_void,
        &mut rows_arg as *mut _ as *mut std::ffi::c_void,
        &mut cols_arg as *mut _ as *mut std::ffi::c_void,
        &mut row_stride as *mut _ as *mut std::ffi::c_void,
        &mut col_stride as *mut _ as *mut std::ffi::c_void,
        &mut rows_per_block_arg as *mut _ as *mut std::ffi::c_void,
        &mut output_ptr as *mut _ as *mut std::ffi::c_void,
        &mut output_len as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe {
        cuda_core::launch_kernel_on_stream(
            function,
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

fn matvec_rows_per_block(candidate: &KernelCandidateMetadata) -> KernelAutotuneMeasureResult<u32> {
    candidate
        .schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Split { axis: 0, factor }
            | ScheduleTransform::GroupTop { axis: 0, factor } => Some(*factor),
            _ => None,
        })
        .ok_or_else(|| {
            invalid_input(format!(
                "matvec candidate {} is missing row split schedule",
                candidate.artifact_key().hex()
            ))
        })
}
