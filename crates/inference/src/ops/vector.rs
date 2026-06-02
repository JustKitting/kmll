use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_host::cuda_launch;

// cuda_launch! resolves cuda-oxide's generated __*_CudaKernel marker types by
// bare name, so keep the wildcard import for kernels launched from this module.
use crate::kernels::vector::*;

pub fn vecadd(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(a.len(), b.len(), "vecadd input length mismatch");
    assert_eq!(a.len(), output.len(), "vecadd output length mismatch");

    cuda_launch! {
        kernel: vecadd_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(a.len() as u32),
        args: [slice(*a), slice(*b), slice_mut(*output)]
    }
}
