use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_host::cuda_launch;

// cuda_launch! resolves cuda-oxide's generated __*_CudaKernel marker types by
// bare name, so keep the wildcard import for kernels launched from this module.
use crate::kernels::activation::*;

pub fn relu(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(
        input.len(),
        output.len(),
        "relu input/output length mismatch"
    );

    cuda_launch! {
        kernel: relu_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(input.len() as u32),
        args: [slice(*input), slice_mut(*output)]
    }
}

pub fn swiglu(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(
        input.len(),
        output.len(),
        "swiglu input/output length mismatch"
    );

    cuda_launch! {
        kernel: swiglu_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(input.len() as u32),
        args: [slice(*input), slice_mut(*output)]
    }
}
