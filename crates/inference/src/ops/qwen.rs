use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_host::cuda_launch;
use nn_rust_profiling::{
    Bf16ProfileType, CudaLaunchSpec, F32ProfileType, OperationKind, OperationRoute, TensorType,
    TypedOperationSpec,
};

use crate::{
    dtypes::Bf16,
    layout::{BlockRmsNorm256Plan, RmsNormKernelPlan},
};
// cuda_launch! resolves cuda-oxide's generated __*_CudaKernel marker types by
// bare name, so keep the wildcard import for kernels launched from this module.
use crate::kernels::qwen::*;

type DefaultRmsNormPlan = BlockRmsNorm256Plan;

#[derive(Debug, Clone)]
pub struct QwenLaunchPlan {
    pub config: LaunchConfig,
    pub operation: TypedOperationSpec,
}

fn contiguous_f32(len: usize) -> nn_rust_profiling::TensorTypeSpec {
    TensorType::<F32ProfileType, 1>::new([len])
        .with_static_layout("contiguous")
        .erase()
}

fn contiguous_bf16(len: usize) -> nn_rust_profiling::TensorTypeSpec {
    TensorType::<Bf16ProfileType, 1>::new([len])
        .with_static_layout("contiguous")
        .erase()
}

fn launch_spec(kernel: &'static str, config: LaunchConfig) -> CudaLaunchSpec {
    CudaLaunchSpec::new(
        kernel,
        config.grid_dim,
        config.block_dim,
        config.shared_mem_bytes,
    )
}

fn checked_u32(value: usize, context: &str) -> u32 {
    u32::try_from(value).expect(context)
}

fn launch_for_elems(value: usize, context: &str) -> LaunchConfig {
    LaunchConfig::for_num_elems(checked_u32(value, context))
}

pub fn qwen_rmsnorm_bf16_plan(len: usize) -> QwenLaunchPlan {
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (DefaultRmsNormPlan::block_threads(), 1, 1),
        shared_mem_bytes: 0,
    };
    let operation = TypedOperationSpec::new(
        "qwen-rmsnorm-bf16",
        OperationKind::RmsNorm,
        OperationRoute::CudaKernel,
    )
    .with_input(contiguous_f32(len))
    .with_input(contiguous_bf16(len))
    .with_output(contiguous_f32(len))
    .with_launch(launch_spec("qwen_rmsnorm_bf16_kernel", config));
    QwenLaunchPlan { config, operation }
}

pub fn qwen_rmsnorm_batched_bf16_plan(batch: usize, dim: usize) -> QwenLaunchPlan {
    batch
        .checked_mul(dim)
        .expect("batched Qwen RMSNorm plan shape overflow");
    let config = LaunchConfig {
        grid_dim: (
            checked_u32(batch, "batched Qwen RMSNorm grid overflow"),
            1,
            1,
        ),
        block_dim: (DefaultRmsNormPlan::block_threads(), 1, 1),
        shared_mem_bytes: 0,
    };
    let operation = TypedOperationSpec::new(
        "qwen-rmsnorm-batched-bf16",
        OperationKind::RmsNorm,
        OperationRoute::CudaKernel,
    )
    .with_input(
        TensorType::<F32ProfileType, 2>::new([batch, dim])
            .with_static_layout("row-major")
            .erase(),
    )
    .with_input(contiguous_bf16(dim))
    .with_output(
        TensorType::<F32ProfileType, 2>::new([batch, dim])
            .with_static_layout("row-major")
            .erase(),
    )
    .with_launch(launch_spec("qwen_rmsnorm_batched_bf16_kernel", config));
    QwenLaunchPlan { config, operation }
}

pub fn qwen_split_query_gate_plan(n_heads: usize, head_dim: usize) -> QwenLaunchPlan {
    let q_len = n_heads
        .checked_mul(head_dim)
        .expect("Qwen query/gate plan shape overflow");
    let q_gate_len = q_len
        .checked_mul(2)
        .expect("Qwen query/gate input shape overflow");
    let config = launch_for_elems(q_len, "Qwen query/gate launch length overflow");
    let operation = TypedOperationSpec::new(
        "qwen-split-query-gate",
        OperationKind::Attention,
        OperationRoute::CudaKernel,
    )
    .with_input(contiguous_f32(q_gate_len))
    .with_output(contiguous_f32(q_len))
    .with_output(contiguous_f32(q_len))
    .with_launch(launch_spec("qwen_split_query_gate_kernel", config));
    QwenLaunchPlan { config, operation }
}

pub fn qwen_linear_conv_silu_step_plan(channels: usize, kernel_size: usize) -> QwenLaunchPlan {
    let conv_state_len = channels
        .checked_mul(kernel_size)
        .expect("Qwen conv plan state shape overflow");
    let config = launch_for_elems(channels, "Qwen conv launch length overflow");
    let operation = TypedOperationSpec::new(
        "qwen-linear-conv-silu-step",
        OperationKind::Attention,
        OperationRoute::CudaKernel,
    )
    .with_input(contiguous_f32(channels))
    .with_input(contiguous_bf16(conv_state_len))
    .with_input(contiguous_f32(conv_state_len))
    .with_output(contiguous_f32(conv_state_len))
    .with_output(contiguous_f32(channels))
    .with_launch(launch_spec("qwen_linear_conv_silu_step_kernel", config));
    QwenLaunchPlan { config, operation }
}

pub fn qwen_split_linear_qkv_plan(
    value_heads: usize,
    key_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> QwenLaunchPlan {
    assert!(key_heads > 0, "Qwen linear key head count must be nonzero");
    assert!(
        value_heads % key_heads == 0,
        "Qwen linear value heads must be divisible by key heads"
    );
    let key_len = key_heads
        .checked_mul(key_dim)
        .expect("Qwen linear key plan shape overflow");
    let query_len = value_heads
        .checked_mul(key_dim)
        .expect("Qwen linear query plan shape overflow");
    let value_len = value_heads
        .checked_mul(value_dim)
        .expect("Qwen linear value plan shape overflow");
    let qkv_len = key_len
        .checked_mul(2)
        .and_then(|len| len.checked_add(value_len))
        .expect("Qwen linear qkv plan shape overflow");
    let config = launch_for_elems(
        query_len.max(value_len),
        "Qwen linear qkv launch length overflow",
    );
    let operation = TypedOperationSpec::new(
        "qwen-split-linear-qkv",
        OperationKind::Attention,
        OperationRoute::CudaKernel,
    )
    .with_input(contiguous_f32(qkv_len))
    .with_output(contiguous_f32(query_len))
    .with_output(contiguous_f32(query_len))
    .with_output(contiguous_f32(value_len))
    .with_launch(launch_spec("qwen_split_linear_qkv_kernel", config));
    QwenLaunchPlan { config, operation }
}

pub fn qwen_gated_delta_decay_plan(len: usize) -> QwenLaunchPlan {
    let config = launch_for_elems(len, "Qwen gated delta decay launch length overflow");
    let operation = TypedOperationSpec::new(
        "qwen-gated-delta-decay",
        OperationKind::Attention,
        OperationRoute::CudaKernel,
    )
    .with_input(contiguous_f32(len))
    .with_input(contiguous_bf16(len))
    .with_input(contiguous_bf16(len))
    .with_output(contiguous_f32(len))
    .with_launch(launch_spec("qwen_gated_delta_decay_kernel", config));
    QwenLaunchPlan { config, operation }
}

pub fn qwen_gated_delta_step_plan(
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> QwenLaunchPlan {
    let key_len = value_heads
        .checked_mul(key_dim)
        .expect("Qwen gated delta key plan shape overflow");
    let value_len = value_heads
        .checked_mul(value_dim)
        .expect("Qwen gated delta value plan shape overflow");
    let state_len = key_len
        .checked_mul(value_dim)
        .expect("Qwen gated delta state plan shape overflow");
    let config = launch_for_elems(value_len, "Qwen gated delta step launch length overflow");
    let operation = TypedOperationSpec::new(
        "qwen-gated-delta-step",
        OperationKind::Attention,
        OperationRoute::CudaKernel,
    )
    .with_input(contiguous_f32(key_len))
    .with_input(contiguous_f32(key_len))
    .with_input(contiguous_f32(value_len))
    .with_input(contiguous_f32(value_heads))
    .with_input(contiguous_f32(value_heads))
    .with_input(contiguous_f32(state_len))
    .with_output(contiguous_f32(value_len))
    .with_output(contiguous_f32(state_len))
    .with_launch(launch_spec("qwen_gated_delta_step_kernel", config));
    QwenLaunchPlan { config, operation }
}

pub fn qwen_gated_rmsnorm_bf16_plan(batch: usize, dim: usize) -> QwenLaunchPlan {
    batch
        .checked_mul(dim)
        .expect("Qwen gated RMSNorm plan shape overflow");
    let config = LaunchConfig {
        grid_dim: (checked_u32(batch, "Qwen gated RMSNorm grid overflow"), 1, 1),
        block_dim: (DefaultRmsNormPlan::block_threads(), 1, 1),
        shared_mem_bytes: 0,
    };
    let operation = TypedOperationSpec::new(
        "qwen-gated-rmsnorm-bf16",
        OperationKind::RmsNorm,
        OperationRoute::CudaKernel,
    )
    .with_input(
        TensorType::<F32ProfileType, 2>::new([batch, dim])
            .with_static_layout("row-major")
            .erase(),
    )
    .with_input(
        TensorType::<F32ProfileType, 2>::new([batch, dim])
            .with_static_layout("row-major")
            .erase(),
    )
    .with_input(contiguous_bf16(dim))
    .with_output(
        TensorType::<F32ProfileType, 2>::new([batch, dim])
            .with_static_layout("row-major")
            .erase(),
    )
    .with_launch(launch_spec("qwen_gated_rmsnorm_bf16_kernel", config));
    QwenLaunchPlan { config, operation }
}

pub fn qwen_rmsnorm_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    eps: f32,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(
        input.len(),
        weight.len(),
        "Qwen RMSNorm weight length mismatch"
    );
    assert_eq!(
        input.len(),
        output.len(),
        "Qwen RMSNorm output length mismatch"
    );
    let plan = qwen_rmsnorm_bf16_plan(input.len());

    cuda_launch! {
        kernel: qwen_rmsnorm_bf16_kernel,
        stream: stream,
        module: module,
        config: plan.config,
        args: [
            slice(*input),
            slice(*weight),
            eps,
            slice_mut(*output)
        ]
    }
}

pub fn qwen_rmsnorm_batched_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    batch: usize,
    dim: usize,
    eps: f32,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let active_len = batch
        .checked_mul(dim)
        .expect("batched Qwen RMSNorm input shape overflow");
    assert!(
        input.len() >= active_len,
        "batched Qwen RMSNorm input too short: {} < {}",
        input.len(),
        active_len
    );
    assert_eq!(
        weight.len(),
        dim,
        "batched Qwen RMSNorm weight length mismatch"
    );
    assert!(
        output.len() >= active_len,
        "batched Qwen RMSNorm output too short: {} < {}",
        output.len(),
        active_len
    );
    let plan = qwen_rmsnorm_batched_bf16_plan(batch, dim);

    cuda_launch! {
        kernel: qwen_rmsnorm_batched_bf16_kernel,
        stream: stream,
        module: module,
        config: plan.config,
        args: [
            slice(*input),
            slice(*weight),
            batch as u32,
            dim as u32,
            eps,
            slice_mut(*output)
        ]
    }
}

pub fn qwen_split_query_gate(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    q_gate: &DeviceBuffer<f32>,
    n_heads: usize,
    head_dim: usize,
    query: &mut DeviceBuffer<f32>,
    gate: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let q_len = n_heads
        .checked_mul(head_dim)
        .expect("Qwen query/gate shape overflow");
    assert_eq!(q_gate.len(), q_len * 2, "Qwen q_gate length mismatch");
    assert_eq!(query.len(), q_len, "Qwen query length mismatch");
    assert_eq!(gate.len(), q_len, "Qwen gate length mismatch");
    let plan = qwen_split_query_gate_plan(n_heads, head_dim);

    cuda_launch! {
        kernel: qwen_split_query_gate_kernel,
        stream: stream,
        module: module,
        config: plan.config,
        args: [
            slice(*q_gate),
            n_heads as u32,
            head_dim as u32,
            slice_mut(*query),
            slice_mut(*gate)
        ]
    }
}

pub fn qwen_linear_conv_silu_step(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    conv_state_in: &DeviceBuffer<f32>,
    channels: usize,
    kernel_size: usize,
    conv_state_out: &mut DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(input.len(), channels, "Qwen conv input length mismatch");
    assert_eq!(
        weight.len(),
        channels
            .checked_mul(kernel_size)
            .expect("Qwen conv weight shape overflow"),
        "Qwen conv weight length mismatch"
    );
    assert_eq!(
        conv_state_in.len(),
        channels
            .checked_mul(kernel_size)
            .expect("Qwen conv state shape overflow"),
        "Qwen conv input state length mismatch"
    );
    assert_eq!(
        conv_state_out.len(),
        channels
            .checked_mul(kernel_size)
            .expect("Qwen conv state shape overflow"),
        "Qwen conv output state length mismatch"
    );
    assert_eq!(output.len(), channels, "Qwen conv output length mismatch");
    let plan = qwen_linear_conv_silu_step_plan(channels, kernel_size);

    cuda_launch! {
        kernel: qwen_linear_conv_silu_step_kernel,
        stream: stream,
        module: module,
        config: plan.config,
        args: [
            slice(*input),
            slice(*weight),
            slice(*conv_state_in),
            channels as u32,
            kernel_size as u32,
            slice_mut(*conv_state_out),
            slice_mut(*output)
        ]
    }
}

pub fn qwen_split_linear_qkv(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    qkv: &DeviceBuffer<f32>,
    value_heads: usize,
    key_heads: usize,
    key_dim: usize,
    value_dim: usize,
    query: &mut DeviceBuffer<f32>,
    key: &mut DeviceBuffer<f32>,
    value: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert!(key_heads > 0, "Qwen linear key head count must be nonzero");
    assert!(
        value_heads % key_heads == 0,
        "Qwen linear value heads must be divisible by key heads"
    );
    let key_len = key_heads
        .checked_mul(key_dim)
        .expect("Qwen linear key shape overflow");
    let query_len = value_heads
        .checked_mul(key_dim)
        .expect("Qwen linear query shape overflow");
    let value_len = value_heads
        .checked_mul(value_dim)
        .expect("Qwen linear value shape overflow");
    assert_eq!(
        qkv.len(),
        key_len * 2 + value_len,
        "Qwen qkv length mismatch"
    );
    assert_eq!(
        query.len(),
        query_len,
        "Qwen repeated query length mismatch"
    );
    assert_eq!(key.len(), query_len, "Qwen repeated key length mismatch");
    assert_eq!(value.len(), value_len, "Qwen value length mismatch");
    let plan = qwen_split_linear_qkv_plan(value_heads, key_heads, key_dim, value_dim);

    cuda_launch! {
        kernel: qwen_split_linear_qkv_kernel,
        stream: stream,
        module: module,
        config: plan.config,
        args: [
            slice(*qkv),
            value_heads as u32,
            key_heads as u32,
            key_dim as u32,
            value_dim as u32,
            slice_mut(*query),
            slice_mut(*key),
            slice_mut(*value)
        ]
    }
}

pub fn qwen_gated_delta_decay(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    a: &DeviceBuffer<f32>,
    a_log: &DeviceBuffer<Bf16>,
    dt_bias: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(a.len(), a_log.len(), "Qwen decay A_log length mismatch");
    assert_eq!(a.len(), dt_bias.len(), "Qwen decay dt_bias length mismatch");
    assert_eq!(a.len(), output.len(), "Qwen decay output length mismatch");
    let plan = qwen_gated_delta_decay_plan(output.len());

    cuda_launch! {
        kernel: qwen_gated_delta_decay_kernel,
        stream: stream,
        module: module,
        config: plan.config,
        args: [slice(*a), slice(*a_log), slice(*dt_bias), slice_mut(*output)]
    }
}

pub fn qwen_gated_delta_step(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    query: &DeviceBuffer<f32>,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    g: &DeviceBuffer<f32>,
    beta_logits: &DeviceBuffer<f32>,
    state_in: &DeviceBuffer<f32>,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    output: &mut DeviceBuffer<f32>,
    state_out: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let value_len = value_heads
        .checked_mul(value_dim)
        .expect("Qwen gated delta value shape overflow");
    let state_len = value_heads
        .checked_mul(key_dim)
        .and_then(|len| len.checked_mul(value_dim))
        .expect("Qwen gated delta state shape overflow");
    assert_eq!(
        query.len(),
        value_heads * key_dim,
        "Qwen query length mismatch"
    );
    assert_eq!(key.len(), value_heads * key_dim, "Qwen key length mismatch");
    assert_eq!(value.len(), value_len, "Qwen value length mismatch");
    assert_eq!(g.len(), value_heads, "Qwen decay length mismatch");
    assert_eq!(
        beta_logits.len(),
        value_heads,
        "Qwen beta logits length mismatch"
    );
    assert_eq!(
        state_in.len(),
        state_len,
        "Qwen input state length mismatch"
    );
    assert_eq!(output.len(), value_len, "Qwen delta output length mismatch");
    assert_eq!(
        state_out.len(),
        state_len,
        "Qwen output state length mismatch"
    );
    let plan = qwen_gated_delta_step_plan(value_heads, key_dim, value_dim);

    cuda_launch! {
        kernel: qwen_gated_delta_step_kernel,
        stream: stream,
        module: module,
        config: plan.config,
        args: [
            slice(*query),
            slice(*key),
            slice(*value),
            slice(*g),
            slice(*beta_logits),
            slice(*state_in),
            value_heads as u32,
            key_dim as u32,
            value_dim as u32,
            slice_mut(*output),
            slice_mut(*state_out)
        ]
    }
}

pub fn qwen_gated_rmsnorm_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    batch: usize,
    dim: usize,
    eps: f32,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let active_len = batch
        .checked_mul(dim)
        .expect("Qwen gated RMSNorm shape overflow");
    assert_eq!(input.len(), active_len, "Qwen gated RMSNorm input mismatch");
    assert_eq!(gate.len(), active_len, "Qwen gated RMSNorm gate mismatch");
    assert_eq!(weight.len(), dim, "Qwen gated RMSNorm weight mismatch");
    assert_eq!(
        output.len(),
        active_len,
        "Qwen gated RMSNorm output mismatch"
    );
    let plan = qwen_gated_rmsnorm_bf16_plan(batch, dim);

    cuda_launch! {
        kernel: qwen_gated_rmsnorm_bf16_kernel,
        stream: stream,
        module: module,
        config: plan.config,
        args: [
            slice(*input),
            slice(*gate),
            slice(*weight),
            batch as u32,
            dim as u32,
            eps,
            slice_mut(*output)
        ]
    }
}

#[cfg(test)]
mod tests {
    use nn_rust_profiling::{NumericKind, OperationKind, OperationRoute};

    use super::*;

    #[test]
    fn rmsnorm_plan_carries_qwen_launch_and_types() {
        let plan = qwen_rmsnorm_bf16_plan(5120);
        assert_eq!(plan.config.grid_dim, (1, 1, 1));
        assert_eq!(
            plan.operation.launch.as_ref().unwrap().kernel,
            "qwen_rmsnorm_bf16_kernel"
        );
        assert_eq!(plan.operation.kind, OperationKind::RmsNorm);
        assert_eq!(plan.operation.route, OperationRoute::CudaKernel);
        assert_eq!(plan.operation.inputs[0].dtype, NumericKind::F32);
        assert_eq!(plan.operation.inputs[1].dtype, NumericKind::Bf16);
        assert_eq!(plan.operation.outputs[0].accumulator, NumericKind::F32);
    }

    #[test]
    fn split_linear_qkv_plan_tracks_repeated_key_shape() {
        let plan = qwen_split_linear_qkv_plan(48, 8, 128, 128);
        assert_eq!(plan.config.grid_dim, (24, 1, 1));
        assert_eq!(
            plan.operation.launch.as_ref().unwrap().kernel,
            "qwen_split_linear_qkv_kernel"
        );
        assert_eq!(plan.operation.inputs[0].shape, vec![8192]);
        assert_eq!(plan.operation.outputs[0].shape, vec![6144]);
        assert_eq!(plan.operation.outputs[1].shape, vec![6144]);
        assert_eq!(plan.operation.outputs[2].shape, vec![6144]);
    }

    #[test]
    fn gated_delta_step_plan_tracks_state_shape() {
        let plan = qwen_gated_delta_step_plan(48, 128, 128);
        assert_eq!(plan.config.grid_dim, (24, 1, 1));
        assert_eq!(
            plan.operation.launch.as_ref().unwrap().kernel,
            "qwen_gated_delta_step_kernel"
        );
        assert_eq!(plan.operation.inputs[0].shape, vec![6144]);
        assert_eq!(plan.operation.inputs[5].shape, vec![786432]);
        assert_eq!(plan.operation.outputs[0].shape, vec![6144]);
        assert_eq!(plan.operation.outputs[1].shape, vec![786432]);
    }
}
