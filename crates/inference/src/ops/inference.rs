use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_host::cuda_launch;

use crate::{
    dtypes::{Bf16, TensorElement},
    layout::{
        AddOp, ArgmaxProblem, AttentionGeometry, AttentionScoresProblem, BinaryElementwiseProblem,
        BlockLogitSelection256Plan, BlockRmsNorm256Plan, ColumnMajor, CudaEmbeddingRoute,
        CudaRmsNormRoute, DeviceEmbeddingTable, DeviceMatrix, DeviceMatrixMut,
        DeviceRowwiseScaledMatrix, DeviceTensor3, DeviceTensor3Mut, DeviceVector, DeviceVectorMut,
        EmbeddingLookupProblem, KvCacheGeometry, KvCacheWriteProblem, LogitSelectionKernelPlan,
        MatvecKernelPlan, MatvecTile, RmsNormKernelPlan, RmsNormOperation, RmsNormProblem,
        RowMajor, RowMajorWarpRowMatvecPlan, RowMajorWarpRows2MatvecPlan,
        RowMajorWarpRows4MatvecPlan, RowMajorWarpRows8MatvecPlan, RowwiseScaledLinearProblem,
        SeqHeadDimMajor, SiluMulOp, SoftmaxValueProblem, TopKProblem,
    },
};
// cuda_launch! resolves cuda-oxide's generated __*_CudaKernel marker types by
// bare name, so keep the wildcard import for kernels launched from this module.
use crate::kernels::inference::*;
use crate::rowwise_scaled::DeviceRowwiseScaledI8Matrix;

type DefaultMatvecPlan = RowMajorWarpRows4MatvecPlan;
type DefaultRmsNormPlan = BlockRmsNorm256Plan;
type DefaultLogitSelectionPlan = BlockLogitSelection256Plan;

const ATTENTION_SCORES_PER_BLOCK: u32 = 4;
const ATTENTION_SCORES_BLOCK_THREADS: u32 = 32 * ATTENTION_SCORES_PER_BLOCK;
const ATTENTION_SOFTMAX_BLOCK_THREADS: u32 = 256;
pub const SINGLE_QUERY_ATTENTION_MAX_SEQ: usize = 4096;
const SINGLE_QUERY_ATTENTION_BLOCK_THREADS: u32 = 256;

fn assert_rope_frequency_len(freqs_len: usize, head_dim: usize) {
    assert!(freqs_len > 0, "RoPE frequency length must be nonzero");
    assert!(
        freqs_len <= head_dim / 2,
        "RoPE frequency length mismatch: got {freqs_len}, expected at most {}",
        head_dim / 2
    );
}

pub trait CudaEmbeddingWeight: Sized {
    fn launch_embedding(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        weight: &DeviceBuffer<Self>,
        token_id: u32,
        dim: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError>;
}

pub trait CudaRmsNormWeight: Sized {
    fn launch_rmsnorm<Plan>(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        operation: RmsNormOperation<Plan, f32>,
        input: &DeviceVector<'_, f32>,
        weight: &DeviceVector<'_, Self>,
        output: &mut DeviceVectorMut<'_, f32>,
    ) -> Result<(), DriverError>
    where
        Plan: RmsNormKernelPlan;
}

pub trait CudaMatvecWeight<L, Tile>: Sized
where
    Tile: MatvecTile,
{
    fn launch_matvec(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        input: &DeviceVector<'_, f32>,
        weight: &DeviceMatrix<'_, Self, L>,
        output: &mut DeviceVectorMut<'_, f32>,
    ) -> Result<(), DriverError>;
}

pub trait CudaLinearWeight {
    fn launch_linear(
        &self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError>;
}

impl CudaEmbeddingWeight for Bf16 {
    fn launch_embedding(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        weight: &DeviceBuffer<Self>,
        token_id: u32,
        dim: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        cuda_launch! {
            kernel: embedding_bf16_kernel,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(output.len() as u32),
            args: [slice(*weight), token_id, dim as u32, slice_mut(*output)]
        }
    }
}

impl CudaRmsNormWeight for Bf16 {
    fn launch_rmsnorm<Plan>(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        operation: RmsNormOperation<Plan, f32>,
        input: &DeviceVector<'_, f32>,
        weight: &DeviceVector<'_, Self>,
        output: &mut DeviceVectorMut<'_, f32>,
    ) -> Result<(), DriverError>
    where
        Plan: RmsNormKernelPlan,
    {
        cuda_launch! {
            kernel: rmsnorm_bf16_kernel,
            stream: stream,
            module: module,
            config: LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (operation.block_threads(), 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(*input.buffer()),
                slice(*weight.buffer()),
                operation.epsilon_f32(),
                slice_mut(*output.buffer_mut())
            ]
        }
    }
}

impl<Tile> CudaMatvecWeight<RowMajor, Tile> for Bf16
where
    Tile: MatvecTile,
{
    fn launch_matvec(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        input: &DeviceVector<'_, f32>,
        weight: &DeviceMatrix<'_, Self, RowMajor>,
        output: &mut DeviceVectorMut<'_, f32>,
    ) -> Result<(), DriverError> {
        let layout = weight.layout();
        let shape = layout.shape();
        let stride = layout.stride();

        cuda_launch! {
            kernel: matvec_bf16_kernel,
            stream: stream,
            module: module,
            config: LaunchConfig {
                grid_dim: (Tile::grid_rows(shape.rows), 1, 1),
                block_dim: (Tile::block_threads(), 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(*input.buffer()),
                slice(*weight.buffer()),
                shape.rows as u32,
                shape.cols as u32,
                stride.row as u32,
                stride.col as u32,
                Tile::ROWS_PER_BLOCK,
                slice_mut(*output.buffer_mut())
            ]
        }
    }
}

impl CudaLinearWeight for DeviceBuffer<Bf16> {
    fn launch_linear(
        &self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        let input = DeviceVector::new(input);
        let weight = DeviceMatrix::<Bf16, <DefaultMatvecPlan as MatvecKernelPlan>::Layout>::packed(
            self,
            output.len(),
            input.len(),
        )
        .expect("linear BF16 dense weight shape mismatch");
        let output = DeviceVectorMut::new(output);
        let mut output = output;

        <Bf16 as CudaMatvecWeight<
            <DefaultMatvecPlan as MatvecKernelPlan>::Layout,
            <DefaultMatvecPlan as MatvecKernelPlan>::Tile,
        >>::launch_matvec(stream, module, &input, &weight, &mut output)
    }
}

impl CudaLinearWeight for DeviceRowwiseScaledI8Matrix {
    fn launch_linear(
        &self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        let input = DeviceVector::new(input);
        let weight = DeviceRowwiseScaledMatrix::<i8, f32, RowMajor>::packed(
            &self.values,
            &self.scales,
            self.rows,
            self.cols,
        )
        .expect("linear rowwise scaled i8 value shape mismatch");
        let output = DeviceVectorMut::new(output);
        let problem =
            RowwiseScaledLinearProblem::<f32, i8, f32, f32, RowMajor>::new(input, weight, output)
                .expect("linear rowwise scaled i8 operand shape mismatch");
        let (input, weight, mut output) = problem.into_parts();
        let layout = weight.layout().values();
        let shape = layout.shape();
        let stride = layout.stride();

        cuda_launch! {
            kernel: matvec_i8_scaled_kernel,
            stream: stream,
            module: module,
            config: LaunchConfig {
                grid_dim: (DefaultMatvecPlan::grid_rows(shape.rows), 1, 1),
                block_dim: (DefaultMatvecPlan::block_threads(), 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(*input.buffer()),
                slice(*weight.values().buffer()),
                slice(*weight.scales().buffer()),
                shape.rows as u32,
                shape.cols as u32,
                stride.row as u32,
                stride.col as u32,
                DefaultMatvecPlan::rows_per_block(),
                slice_mut(*output.buffer_mut())
            ]
        }
    }
}

pub fn linear_top1_bf16_partial_count(rows: usize) -> usize {
    DefaultMatvecPlan::grid_rows(rows) as usize
}

pub fn linear_top1_i8_scaled_partial_count(rows: usize) -> usize {
    DefaultMatvecPlan::grid_rows(rows) as usize
}

fn linear_top1_bf16_partial_count_with_plan<Plan>(rows: usize) -> usize
where
    Plan: MatvecKernelPlan<Layout = RowMajor>,
{
    Plan::grid_rows(rows) as usize
}

pub fn linear_top1_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    partial_tokens: &mut DeviceBuffer<u32>,
    partial_logits: &mut DeviceBuffer<f32>,
    packed_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    linear_top1_bf16_with_plan::<DefaultMatvecPlan>(
        stream,
        module,
        input,
        weight,
        partial_tokens,
        partial_logits,
        packed_out,
    )
}

pub fn linear_top1_bf16_rows1(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    partial_tokens: &mut DeviceBuffer<u32>,
    partial_logits: &mut DeviceBuffer<f32>,
    packed_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    linear_top1_bf16_with_plan::<RowMajorWarpRowMatvecPlan>(
        stream,
        module,
        input,
        weight,
        partial_tokens,
        partial_logits,
        packed_out,
    )
}

pub fn linear_top1_bf16_rows2(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    partial_tokens: &mut DeviceBuffer<u32>,
    partial_logits: &mut DeviceBuffer<f32>,
    packed_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    linear_top1_bf16_with_plan::<RowMajorWarpRows2MatvecPlan>(
        stream,
        module,
        input,
        weight,
        partial_tokens,
        partial_logits,
        packed_out,
    )
}

pub fn linear_top1_bf16_rows8(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    partial_tokens: &mut DeviceBuffer<u32>,
    partial_logits: &mut DeviceBuffer<f32>,
    packed_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    linear_top1_bf16_with_plan::<RowMajorWarpRows8MatvecPlan>(
        stream,
        module,
        input,
        weight,
        partial_tokens,
        partial_logits,
        packed_out,
    )
}

fn linear_top1_bf16_with_plan<Plan>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    partial_tokens: &mut DeviceBuffer<u32>,
    partial_logits: &mut DeviceBuffer<f32>,
    packed_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError>
where
    Plan: MatvecKernelPlan<Layout = RowMajor>,
{
    assert!(!input.is_empty(), "linear top1 input must be nonempty");
    assert!(
        weight.len() % input.len() == 0,
        "linear top1 BF16 weight length must be divisible by input length"
    );
    assert!(
        !packed_out.is_empty(),
        "linear top1 packed output must have at least one element"
    );

    let rows = weight.len() / input.len();
    assert!(rows > 0, "linear top1 BF16 weight must have rows");
    let partial_count = linear_top1_bf16_partial_count_with_plan::<Plan>(rows);
    assert!(
        partial_tokens.len() >= partial_count,
        "linear top1 partial token buffer too short: {} < {}",
        partial_tokens.len(),
        partial_count
    );
    assert!(
        partial_logits.len() >= partial_count,
        "linear top1 partial logit buffer too short: {} < {}",
        partial_logits.len(),
        partial_count
    );

    let input = DeviceVector::new(input);
    let weight =
        DeviceMatrix::<Bf16, <Plan as MatvecKernelPlan>::Layout>::packed(weight, rows, input.len())
            .expect("linear top1 BF16 weight shape mismatch");
    let shape = weight.layout().shape();
    let stride = weight.layout().stride();

    cuda_launch! {
        kernel: matvec_top1_bf16_stage_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (partial_count as u32, 1, 1),
            block_dim: (Plan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*input.buffer()),
            slice(*weight.buffer()),
            shape.rows as u32,
            shape.cols as u32,
            stride.row as u32,
            stride.col as u32,
            Plan::rows_per_block(),
            slice_mut(*partial_tokens),
            slice_mut(*partial_logits)
        ]
    }?;

    cuda_launch! {
        kernel: argmax_pairs_f32_packed_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (DefaultLogitSelectionPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*partial_tokens),
            slice(*partial_logits),
            partial_count as u32,
            slice_mut(*packed_out)
        ]
    }
}

pub fn linear_top1_i8_scaled(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceRowwiseScaledI8Matrix,
    partial_tokens: &mut DeviceBuffer<u32>,
    partial_logits: &mut DeviceBuffer<f32>,
    packed_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    assert!(
        !input.is_empty(),
        "linear top1 scaled-i8 input must be nonempty"
    );
    assert!(
        !packed_out.is_empty(),
        "linear top1 scaled-i8 packed output must have at least one element"
    );
    assert_eq!(
        weight.cols,
        input.len(),
        "linear top1 scaled-i8 input dim mismatch"
    );
    assert!(
        weight.rows > 0,
        "linear top1 scaled-i8 weight must have rows"
    );
    assert_eq!(
        weight.values.len(),
        weight
            .rows
            .checked_mul(weight.cols)
            .expect("linear top1 scaled-i8 weight shape overflow"),
        "linear top1 scaled-i8 value length mismatch"
    );
    assert_eq!(
        weight.scales.len(),
        weight.rows,
        "linear top1 scaled-i8 scale length mismatch"
    );

    let partial_count = linear_top1_i8_scaled_partial_count(weight.rows);
    assert!(
        partial_tokens.len() >= partial_count,
        "linear top1 scaled-i8 partial token buffer too short: {} < {}",
        partial_tokens.len(),
        partial_count
    );
    assert!(
        partial_logits.len() >= partial_count,
        "linear top1 scaled-i8 partial logit buffer too short: {} < {}",
        partial_logits.len(),
        partial_count
    );

    let values = &weight.values;
    let scales = &weight.scales;
    let weight = DeviceRowwiseScaledMatrix::<i8, f32, RowMajor>::packed(
        values,
        scales,
        weight.rows,
        weight.cols,
    )
    .expect("linear top1 scaled-i8 weight shape mismatch");
    let layout = weight.layout().values();
    let shape = layout.shape();
    let stride = layout.stride();

    cuda_launch! {
        kernel: matvec_top1_i8_scaled_stage_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (partial_count as u32, 1, 1),
            block_dim: (DefaultMatvecPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*input),
            slice(*weight.values().buffer()),
            slice(*weight.scales().buffer()),
            shape.rows as u32,
            shape.cols as u32,
            stride.row as u32,
            stride.col as u32,
            DefaultMatvecPlan::rows_per_block(),
            slice_mut(*partial_tokens),
            slice_mut(*partial_logits)
        ]
    }?;

    cuda_launch! {
        kernel: argmax_pairs_f32_packed_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (DefaultLogitSelectionPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*partial_tokens),
            slice(*partial_logits),
            partial_count as u32,
            slice_mut(*packed_out)
        ]
    }
}

pub fn embedding<W>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weight: &DeviceBuffer<W>,
    token_id: u32,
    dim: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError>
where
    W: CudaEmbeddingWeight + TensorElement + CudaEmbeddingRoute<Output = f32>,
{
    let weight =
        DeviceEmbeddingTable::<W>::packed(weight, dim).expect("embedding table shape mismatch");
    let output = DeviceVectorMut::new(output);
    let problem = EmbeddingLookupProblem::<W, f32>::new(weight, token_id, output)
        .expect("embedding lookup shape mismatch");
    let (weight, token_id, mut output) = problem.into_parts();

    W::launch_embedding(
        stream,
        module,
        weight.buffer(),
        token_id,
        weight.dim(),
        output.buffer_mut(),
    )
}

pub fn embedding_tokens_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weight: &DeviceBuffer<Bf16>,
    tokens: &DeviceBuffer<u32>,
    token_count: usize,
    dim: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert!(
        tokens.len() >= token_count,
        "batched embedding token buffer too short: {} < {}",
        tokens.len(),
        token_count
    );
    assert!(
        output.len() >= token_count * dim,
        "batched embedding output too short: {} < {}",
        output.len(),
        token_count * dim
    );
    assert!(
        dim > 0 && weight.len() % dim == 0,
        "batched embedding weight shape mismatch"
    );

    cuda_launch! {
        kernel: embedding_tokens_bf16_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems((token_count * dim) as u32),
        args: [
            slice(*weight),
            slice(*tokens),
            token_count as u32,
            dim as u32,
            slice_mut(*output)
        ]
    }
}

pub fn rmsnorm<W>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<W>,
    eps: f32,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError>
where
    W: CudaRmsNormWeight + TensorElement,
    f32: CudaRmsNormRoute<W, Accumulator = f32, Output = f32>,
{
    let input = DeviceVector::new(input);
    let weight = DeviceVector::new(weight);
    let output = DeviceVectorMut::new(output);
    let operation = RmsNormOperation::<DefaultRmsNormPlan, f32>::from_f32_epsilon(eps);
    let problem =
        RmsNormProblem::<f32, W, f32, DefaultRmsNormPlan>::new(operation, input, weight, output)
            .expect("RMSNorm operand shape mismatch");
    let (operation, input, weight, mut output) = problem.into_parts();

    W::launch_rmsnorm(stream, module, operation, &input, &weight, &mut output)
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

    cuda_launch! {
        kernel: qwen_rmsnorm_bf16_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (DefaultRmsNormPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*input),
            slice(*weight),
            eps,
            slice_mut(*output)
        ]
    }
}

pub fn rmsnorm_batched_bf16(
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
        .expect("batched RMSNorm input shape overflow");
    assert!(
        input.len() >= active_len,
        "batched RMSNorm input too short: {} < {}",
        input.len(),
        active_len
    );
    assert_eq!(weight.len(), dim, "batched RMSNorm weight length mismatch");
    assert!(
        output.len() >= active_len,
        "batched RMSNorm output too short: {} < {}",
        output.len(),
        active_len
    );

    cuda_launch! {
        kernel: rmsnorm_batched_bf16_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (DefaultRmsNormPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
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

    cuda_launch! {
        kernel: qwen_rmsnorm_batched_bf16_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (DefaultRmsNormPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
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

pub fn linear<W: CudaLinearWeight>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &W,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    weight.launch_linear(stream, module, input, output)
}

pub fn linear_pair_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight_a: &DeviceBuffer<Bf16>,
    weight_b: &DeviceBuffer<Bf16>,
    output_a: &mut DeviceBuffer<f32>,
    output_b: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(
        output_a.len(),
        output_b.len(),
        "paired BF16 linear output length mismatch"
    );
    assert_eq!(
        weight_a.len(),
        weight_b.len(),
        "paired BF16 linear weight length mismatch"
    );

    let input = DeviceVector::new(input);
    let weight_a = DeviceMatrix::<Bf16, <DefaultMatvecPlan as MatvecKernelPlan>::Layout>::packed(
        weight_a,
        output_a.len(),
        input.len(),
    )
    .expect("paired BF16 linear weight_a shape mismatch");
    let weight_b = DeviceMatrix::<Bf16, <DefaultMatvecPlan as MatvecKernelPlan>::Layout>::packed(
        weight_b,
        output_b.len(),
        input.len(),
    )
    .expect("paired BF16 linear weight_b shape mismatch");
    let output_a = DeviceVectorMut::new(output_a);
    let output_b = DeviceVectorMut::new(output_b);
    let mut output_a = output_a;
    let mut output_b = output_b;
    let layout = weight_a.layout();
    let shape = layout.shape();
    let stride = layout.stride();

    cuda_launch! {
        kernel: matvec_pair_bf16_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (DefaultMatvecPlan::grid_rows(shape.rows), 1, 1),
            block_dim: (DefaultMatvecPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*input.buffer()),
            slice(*weight_a.buffer()),
            slice(*weight_b.buffer()),
            shape.rows as u32,
            shape.cols as u32,
            stride.row as u32,
            stride.col as u32,
            DefaultMatvecPlan::rows_per_block(),
            slice_mut(*output_a.buffer_mut()),
            slice_mut(*output_b.buffer_mut())
        ]
    }
}

pub fn silu_gate_up_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    gate_weight: &DeviceBuffer<Bf16>,
    up_weight: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    silu_gate_up_bf16_with_plan::<DefaultMatvecPlan>(
        stream,
        module,
        input,
        gate_weight,
        up_weight,
        output,
    )
}

pub fn silu_gate_up_bf16_rows8(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    gate_weight: &DeviceBuffer<Bf16>,
    up_weight: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    silu_gate_up_bf16_with_plan::<RowMajorWarpRows8MatvecPlan>(
        stream,
        module,
        input,
        gate_weight,
        up_weight,
        output,
    )
}

fn silu_gate_up_bf16_with_plan<Plan>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    gate_weight: &DeviceBuffer<Bf16>,
    up_weight: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError>
where
    Plan: MatvecKernelPlan,
{
    assert_eq!(
        gate_weight.len(),
        up_weight.len(),
        "SiLU gate/up BF16 weight length mismatch"
    );

    let input = DeviceVector::new(input);
    let gate_weight = DeviceMatrix::<Bf16, <Plan as MatvecKernelPlan>::Layout>::packed(
        gate_weight,
        output.len(),
        input.len(),
    )
    .expect("SiLU gate BF16 weight shape mismatch");
    let up_weight = DeviceMatrix::<Bf16, <Plan as MatvecKernelPlan>::Layout>::packed(
        up_weight,
        output.len(),
        input.len(),
    )
    .expect("SiLU up BF16 weight shape mismatch");
    let output = DeviceVectorMut::new(output);
    let mut output = output;
    let layout = gate_weight.layout();
    let shape = layout.shape();
    let stride = layout.stride();

    cuda_launch! {
        kernel: matvec_silu_gate_up_bf16_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (Plan::grid_rows(shape.rows), 1, 1),
            block_dim: (Plan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*input.buffer()),
            slice(*gate_weight.buffer()),
            slice(*up_weight.buffer()),
            shape.rows as u32,
            shape.cols as u32,
            stride.row as u32,
            stride.col as u32,
            Plan::rows_per_block(),
            slice_mut(*output.buffer_mut())
        ]
    }
}

pub fn linear_residual_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    residual: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(
        residual.len(),
        output.len(),
        "linear residual output length mismatch"
    );

    let input = DeviceVector::new(input);
    let weight = DeviceMatrix::<Bf16, <DefaultMatvecPlan as MatvecKernelPlan>::Layout>::packed(
        weight,
        output.len(),
        input.len(),
    )
    .expect("residual BF16 linear weight shape mismatch");
    let residual = DeviceVector::with_len(residual, output.len())
        .expect("residual BF16 linear residual shape mismatch");
    let output = DeviceVectorMut::new(output);
    let mut output = output;
    let layout = weight.layout();
    let shape = layout.shape();
    let stride = layout.stride();

    cuda_launch! {
        kernel: matvec_residual_bf16_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (DefaultMatvecPlan::grid_rows(shape.rows), 1, 1),
            block_dim: (DefaultMatvecPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*input.buffer()),
            slice(*weight.buffer()),
            slice(*residual.buffer()),
            shape.rows as u32,
            shape.cols as u32,
            stride.row as u32,
            stride.col as u32,
            DefaultMatvecPlan::rows_per_block(),
            slice_mut(*output.buffer_mut())
        ]
    }
}

pub fn linear_triple_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight_a: &DeviceBuffer<Bf16>,
    weight_b: &DeviceBuffer<Bf16>,
    weight_c: &DeviceBuffer<Bf16>,
    output_a: &mut DeviceBuffer<f32>,
    output_b: &mut DeviceBuffer<f32>,
    output_c: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let input = DeviceVector::new(input);
    let weight_a = DeviceMatrix::<Bf16, <DefaultMatvecPlan as MatvecKernelPlan>::Layout>::packed(
        weight_a,
        output_a.len(),
        input.len(),
    )
    .expect("triple BF16 linear weight_a shape mismatch");
    let weight_b = DeviceMatrix::<Bf16, <DefaultMatvecPlan as MatvecKernelPlan>::Layout>::packed(
        weight_b,
        output_b.len(),
        input.len(),
    )
    .expect("triple BF16 linear weight_b shape mismatch");
    let weight_c = DeviceMatrix::<Bf16, <DefaultMatvecPlan as MatvecKernelPlan>::Layout>::packed(
        weight_c,
        output_c.len(),
        input.len(),
    )
    .expect("triple BF16 linear weight_c shape mismatch");
    let output_a = DeviceVectorMut::new(output_a);
    let output_b = DeviceVectorMut::new(output_b);
    let output_c = DeviceVectorMut::new(output_c);
    let mut output_a = output_a;
    let mut output_b = output_b;
    let mut output_c = output_c;
    let layout = weight_a.layout();
    let shape = layout.shape();
    let stride = layout.stride();
    let rows = output_a.len().max(output_b.len()).max(output_c.len());

    cuda_launch! {
        kernel: matvec_triple_bf16_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (DefaultMatvecPlan::grid_rows(rows), 1, 1),
            block_dim: (DefaultMatvecPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*input.buffer()),
            slice(*weight_a.buffer()),
            slice(*weight_b.buffer()),
            slice(*weight_c.buffer()),
            output_a.len() as u32,
            output_b.len() as u32,
            output_c.len() as u32,
            shape.cols as u32,
            stride.row as u32,
            stride.col as u32,
            DefaultMatvecPlan::rows_per_block(),
            slice_mut(*output_a.buffer_mut()),
            slice_mut(*output_b.buffer_mut()),
            slice_mut(*output_c.buffer_mut())
        ]
    }
}

pub fn linear_batched_bf16(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    batch: usize,
    input_dim: usize,
    output_dim: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let input_len = batch
        .checked_mul(input_dim)
        .expect("batched linear input shape overflow");
    let output_len = batch
        .checked_mul(output_dim)
        .expect("batched linear output shape overflow");
    assert!(
        input.len() >= input_len,
        "batched linear input too short: {} < {}",
        input.len(),
        input_len
    );
    assert_eq!(
        weight.len(),
        output_dim
            .checked_mul(input_dim)
            .expect("batched linear weight shape overflow"),
        "batched linear weight length mismatch"
    );
    assert!(
        output.len() >= output_len,
        "batched linear output too short: {} < {}",
        output.len(),
        output_len
    );

    super::matmul::gemm_f32_bf16_prefix::<RowMajor, ColumnMajor, RowMajor>(
        stream, module, input, weight, output, batch, output_dim, input_dim, 1.0, 0.0,
    )
}

pub fn silu_mul(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let gate = DeviceVector::new(gate);
    let up = DeviceVector::new(up);
    let output = DeviceVectorMut::new(output);
    let problem = BinaryElementwiseProblem::<f32, f32, f32, SiluMulOp>::new(gate, up, output)
        .expect("silu_mul operand shape mismatch");
    let (gate, up, mut output) = problem.into_parts();

    cuda_launch! {
        kernel: silu_mul_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(output.len() as u32),
        args: [slice(*gate.buffer()), slice(*up.buffer()), slice_mut(*output.buffer_mut())]
    }
}

pub fn sigmoid_mul(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(gate.len(), up.len(), "sigmoid_mul operand length mismatch");
    assert_eq!(
        gate.len(),
        output.len(),
        "sigmoid_mul output length mismatch"
    );

    cuda_launch! {
        kernel: sigmoid_mul_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(output.len() as u32),
        args: [slice(*gate), slice(*up), slice_mut(*output)]
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

    cuda_launch! {
        kernel: qwen_split_query_gate_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(q_len as u32),
        args: [
            slice(*q_gate),
            n_heads as u32,
            head_dim as u32,
            slice_mut(*query),
            slice_mut(*gate)
        ]
    }
}

pub fn silu_mul_prefix(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    count: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert!(
        gate.len() >= count && up.len() >= count && output.len() >= count,
        "silu_mul prefix length mismatch"
    );

    cuda_launch! {
        kernel: silu_mul_n_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(count as u32),
        args: [slice(*gate), slice(*up), count as u32, slice_mut(*output)]
    }
}

pub fn add(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    lhs: &DeviceBuffer<f32>,
    rhs: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let lhs = DeviceVector::new(lhs);
    let rhs = DeviceVector::new(rhs);
    let output = DeviceVectorMut::new(output);
    let problem = BinaryElementwiseProblem::<f32, f32, f32, AddOp>::new(lhs, rhs, output)
        .expect("add operand shape mismatch");
    let (lhs, rhs, mut output) = problem.into_parts();

    cuda_launch! {
        kernel: add_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(output.len() as u32),
        args: [slice(*lhs.buffer()), slice(*rhs.buffer()), slice_mut(*output.buffer_mut())]
    }
}

pub fn add_prefix(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    lhs: &DeviceBuffer<f32>,
    rhs: &DeviceBuffer<f32>,
    count: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert!(
        lhs.len() >= count && rhs.len() >= count && output.len() >= count,
        "add prefix length mismatch"
    );

    cuda_launch! {
        kernel: add_n_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(count as u32),
        args: [slice(*lhs), slice(*rhs), count as u32, slice_mut(*output)]
    }
}

pub fn copy_matrix_row_to_vector(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    matrix: &DeviceBuffer<f32>,
    row: usize,
    cols: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert!(
        matrix.len() >= (row + 1) * cols,
        "matrix row copy input length mismatch"
    );
    assert_eq!(output.len(), cols, "matrix row copy output length mismatch");

    cuda_launch! {
        kernel: copy_matrix_row_to_vector_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(cols as u32),
        args: [slice(*matrix), row as u32, cols as u32, slice_mut(*output)]
    }
}

pub fn copy_vector_to_matrix_row(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    row: usize,
    cols: usize,
    matrix: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(input.len(), cols, "matrix row write input length mismatch");
    assert!(
        matrix.len() >= (row + 1) * cols,
        "matrix row write output length mismatch"
    );

    cuda_launch! {
        kernel: copy_vector_to_matrix_row_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(cols as u32),
        args: [slice(*input), row as u32, cols as u32, slice_mut(*matrix)]
    }
}

pub fn prepare_prefill_attention_batch(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    query_batch: &DeviceBuffer<f32>,
    key_batch: &DeviceBuffer<f32>,
    value_batch: &DeviceBuffer<f32>,
    freqs: &DeviceBuffer<f32>,
    prompt_len: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    query_rot_batch: &mut DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<f32>,
    value_cache: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert!(prompt_len > 0, "prefill prompt length must be nonzero");
    assert!(head_dim > 0, "RoPE head_dim must be nonzero");
    assert_eq!(head_dim % 2, 0, "RoPE head_dim must be even");
    assert_eq!(q_len % head_dim, 0, "query row shape mismatch");
    assert_eq!(kv_len % head_dim, 0, "KV row shape mismatch");
    assert_rope_frequency_len(freqs.len(), head_dim);
    assert!(
        query_batch.len() >= prompt_len * q_len,
        "prefill query batch too short"
    );
    assert!(
        key_batch.len() >= prompt_len * kv_len,
        "prefill key batch too short"
    );
    assert!(
        value_batch.len() >= prompt_len * kv_len,
        "prefill value batch too short"
    );
    assert!(
        query_rot_batch.len() >= prompt_len * q_len,
        "prefill rotated query batch too short"
    );
    assert!(
        key_cache.len() >= prompt_len * kv_len,
        "key cache too short for prefill prompt"
    );
    assert!(
        value_cache.len() >= prompt_len * kv_len,
        "value cache too short for prefill prompt"
    );

    let width = q_len.max(kv_len);
    let count = prompt_len
        .checked_mul(width)
        .expect("prefill attention batch launch shape overflow");
    cuda_launch! {
        kernel: prepare_prefill_attention_batch_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(count as u32),
        args: [
            slice(*query_batch),
            slice(*key_batch),
            slice(*value_batch),
            slice(*freqs),
            prompt_len as u32,
            q_len as u32,
            kv_len as u32,
            head_dim as u32,
            slice_mut(*query_rot_batch),
            slice_mut(*key_cache),
            slice_mut(*value_cache)
        ]
    }
}

pub fn prepare_incremental_attention(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    query: &DeviceBuffer<f32>,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    freqs: &DeviceBuffer<f32>,
    position: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_rot: &mut DeviceBuffer<f32>,
    key_cache: &mut DeviceBuffer<f32>,
    value_cache: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert!(head_dim > 0, "RoPE head_dim must be nonzero");
    assert_eq!(head_dim % 2, 0, "RoPE head_dim must be even");
    assert!(n_kv_heads > 0, "n_kv_heads must be nonzero");
    assert_eq!(
        n_heads % n_kv_heads,
        0,
        "n_heads must be divisible by n_kv_heads"
    );
    let q_len = n_heads
        .checked_mul(head_dim)
        .expect("incremental query shape overflow");
    let kv_len = n_kv_heads
        .checked_mul(head_dim)
        .expect("incremental KV shape overflow");
    let expected_cache_len = max_seq_len
        .checked_mul(kv_len)
        .expect("incremental KV cache shape overflow");
    assert_eq!(query.len(), q_len, "query shape mismatch");
    assert_eq!(key.len(), kv_len, "key shape mismatch");
    assert_eq!(value.len(), kv_len, "value shape mismatch");
    assert_eq!(query_rot.len(), q_len, "query RoPE output shape mismatch");
    assert_rope_frequency_len(freqs.len(), head_dim);
    assert!(
        position < max_seq_len,
        "incremental attention position exceeds max_seq_len"
    );
    assert_eq!(
        key_cache.len(),
        expected_cache_len,
        "key cache shape mismatch"
    );
    assert_eq!(
        value_cache.len(),
        expected_cache_len,
        "value cache shape mismatch"
    );

    cuda_launch! {
        kernel: prepare_incremental_attention_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(q_len.max(kv_len) as u32),
        args: [
            slice(*query),
            slice(*key),
            slice(*value),
            slice(*freqs),
            position as u32,
            q_len as u32,
            kv_len as u32,
            head_dim as u32,
            slice_mut(*query_rot),
            slice_mut(*key_cache),
            slice_mut(*value_cache)
        ]
    }
}

pub fn single_token_gqa(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    value: &DeviceBuffer<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert!(n_kv_heads > 0, "n_kv_heads must be nonzero");
    assert_eq!(
        n_heads % n_kv_heads,
        0,
        "n_heads must be divisible by n_kv_heads"
    );
    assert_eq!(
        value.len(),
        n_kv_heads * head_dim,
        "single-token GQA value length mismatch"
    );
    assert_eq!(
        output.len(),
        n_heads * head_dim,
        "single-token GQA output length mismatch"
    );

    cuda_launch! {
        kernel: single_token_gqa_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(output.len() as u32),
        args: [slice(*value), n_heads as u32, n_kv_heads as u32, head_dim as u32, slice_mut(*output)]
    }
}

pub fn apply_rope(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    freqs: &DeviceBuffer<f32>,
    position: usize,
    head_dim: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert_eq!(input.len(), output.len(), "RoPE output length mismatch");
    assert_eq!(head_dim % 2, 0, "RoPE head_dim must be even");
    assert_eq!(input.len() % head_dim, 0, "RoPE input shape mismatch");
    assert_rope_frequency_len(freqs.len(), head_dim);

    cuda_launch! {
        kernel: apply_rope_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(output.len() as u32),
        args: [slice(*input), slice(*freqs), position as u32, head_dim as u32, slice_mut(*output)]
    }
}

pub fn apply_rope_write_kv_cache(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    freqs: &DeviceBuffer<f32>,
    position: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cache: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let geometry = KvCacheGeometry::new(max_seq_len, n_kv_heads, head_dim);
    let input = DeviceVector::with_len(input, geometry.token_len())
        .expect("RoPE KV cache write token shape mismatch");
    let cache =
        DeviceTensor3Mut::<f32, SeqHeadDimMajor>::packed(cache, max_seq_len, n_kv_heads, head_dim)
            .expect("RoPE KV cache buffer shape mismatch");
    let problem = KvCacheWriteProblem::new(input, cache, position, geometry)
        .expect("RoPE KV cache operand shape mismatch");
    assert_eq!(head_dim % 2, 0, "RoPE head_dim must be even");
    assert_rope_frequency_len(freqs.len(), head_dim);
    let position = problem.position();
    let (input, mut cache) = problem.into_parts();

    cuda_launch! {
        kernel: apply_rope_write_kv_cache_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(input.len() as u32),
        args: [
            slice(*input.buffer()),
            slice(*freqs),
            position as u32,
            head_dim as u32,
            slice_mut(*cache.buffer_mut())
        ]
    }
}

pub fn write_kv_cache(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    position: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cache: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let geometry = KvCacheGeometry::new(max_seq_len, n_kv_heads, head_dim);
    let input = DeviceVector::with_len(input, geometry.token_len())
        .expect("KV cache write token shape mismatch");
    let cache =
        DeviceTensor3Mut::<f32, SeqHeadDimMajor>::packed(cache, max_seq_len, n_kv_heads, head_dim)
            .expect("KV cache buffer shape mismatch");
    let problem = KvCacheWriteProblem::new(input, cache, position, geometry)
        .expect("KV cache write operand shape mismatch");
    let position = problem.position();
    let (input, mut cache) = problem.into_parts();

    cuda_launch! {
        kernel: write_kv_cache_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(input.len() as u32),
        args: [slice(*input.buffer()), position as u32, slice_mut(*cache.buffer_mut())]
    }
}

pub fn attention_scores(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    seq_len: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scores: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let geometry = AttentionGeometry::new(seq_len, max_seq_len, n_heads, n_kv_heads, head_dim)
        .expect("attention geometry mismatch");
    let query = DeviceVector::with_len(query, geometry.query_len()).expect("query shape mismatch");
    let key_cache =
        DeviceTensor3::<f32, SeqHeadDimMajor>::packed(key_cache, max_seq_len, n_kv_heads, head_dim)
            .expect("key cache shape mismatch");
    let scores = DeviceMatrixMut::<f32, RowMajor>::packed(scores, n_heads, max_seq_len)
        .expect("attention score shape mismatch");
    let problem = AttentionScoresProblem::new(geometry, query, key_cache, scores)
        .expect("attention score operand shape mismatch");
    let (geometry, query, key_cache, mut scores) = problem.into_parts();
    let active_scores = geometry
        .n_heads
        .checked_mul(geometry.seq_len)
        .expect("attention active score count overflow");

    cuda_launch! {
        kernel: attention_scores_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (
                (active_scores as u32).div_ceil(ATTENTION_SCORES_PER_BLOCK),
                1,
                1,
            ),
            block_dim: (ATTENTION_SCORES_BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*query.buffer()),
            slice(*key_cache.buffer()),
            geometry.seq_len as u32,
            geometry.max_seq_len as u32,
            geometry.n_heads as u32,
            geometry.n_kv_heads as u32,
            geometry.head_dim as u32,
            ATTENTION_SCORES_PER_BLOCK,
            slice_mut(*scores.buffer_mut())
        ]
    }
}

pub fn attention_scores_from_matrix_row(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    query_batch: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    query_row: usize,
    seq_len: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scores: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let geometry = AttentionGeometry::new(seq_len, max_seq_len, n_heads, n_kv_heads, head_dim)
        .expect("attention geometry mismatch");
    let query_len = geometry.query_len();
    assert!(
        query_batch.len() >= (query_row + 1) * query_len,
        "query matrix row shape mismatch"
    );
    let key_cache =
        DeviceTensor3::<f32, SeqHeadDimMajor>::packed(key_cache, max_seq_len, n_kv_heads, head_dim)
            .expect("key cache shape mismatch");
    let scores = DeviceMatrixMut::<f32, RowMajor>::packed(scores, n_heads, max_seq_len)
        .expect("attention score shape mismatch");
    let active_scores = geometry
        .n_heads
        .checked_mul(geometry.seq_len)
        .expect("attention active score count overflow");
    let mut scores = scores;

    cuda_launch! {
        kernel: attention_scores_from_matrix_row_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (
                (active_scores as u32).div_ceil(ATTENTION_SCORES_PER_BLOCK),
                1,
                1,
            ),
            block_dim: (ATTENTION_SCORES_BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*query_batch),
            slice(*key_cache.buffer()),
            query_row as u32,
            geometry.seq_len as u32,
            geometry.max_seq_len as u32,
            geometry.n_heads as u32,
            geometry.n_kv_heads as u32,
            geometry.head_dim as u32,
            ATTENTION_SCORES_PER_BLOCK,
            slice_mut(*scores.buffer_mut())
        ]
    }
}

pub fn softmax_value(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    scores: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    seq_len: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let geometry = AttentionGeometry::new(seq_len, max_seq_len, n_heads, n_kv_heads, head_dim)
        .expect("attention geometry mismatch");
    let scores = DeviceMatrix::<f32, RowMajor>::packed(scores, n_heads, max_seq_len)
        .expect("attention score shape mismatch");
    let value_cache = DeviceTensor3::<f32, SeqHeadDimMajor>::packed(
        value_cache,
        max_seq_len,
        n_kv_heads,
        head_dim,
    )
    .expect("value cache shape mismatch");
    let output = DeviceVectorMut::with_len(output, geometry.output_len())
        .expect("attention output shape mismatch");
    let problem = SoftmaxValueProblem::new(geometry, scores, value_cache, output)
        .expect("attention value operand shape mismatch");
    let (geometry, scores, value_cache, mut output) = problem.into_parts();

    cuda_launch! {
        kernel: softmax_value_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (geometry.n_heads as u32, 1, 1),
            block_dim: (ATTENTION_SOFTMAX_BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*scores.buffer()),
            slice(*value_cache.buffer()),
            geometry.seq_len as u32,
            geometry.max_seq_len as u32,
            geometry.n_heads as u32,
            geometry.n_kv_heads as u32,
            geometry.head_dim as u32,
            slice_mut(*output.buffer_mut())
        ]
    }
}

pub fn single_query_attention(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    seq_len: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert!(
        seq_len <= SINGLE_QUERY_ATTENTION_MAX_SEQ,
        "fused single-query attention sequence length exceeds kernel capacity"
    );
    let geometry = AttentionGeometry::new(seq_len, max_seq_len, n_heads, n_kv_heads, head_dim)
        .expect("attention geometry mismatch");
    let query = DeviceVector::with_len(query, geometry.query_len()).expect("query shape mismatch");
    let key_cache =
        DeviceTensor3::<f32, SeqHeadDimMajor>::packed(key_cache, max_seq_len, n_kv_heads, head_dim)
            .expect("key cache shape mismatch");
    let value_cache = DeviceTensor3::<f32, SeqHeadDimMajor>::packed(
        value_cache,
        max_seq_len,
        n_kv_heads,
        head_dim,
    )
    .expect("value cache shape mismatch");
    let output = DeviceVectorMut::with_len(output, geometry.output_len())
        .expect("attention output shape mismatch");
    let mut output = output;

    cuda_launch! {
        kernel: single_query_attention_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (geometry.n_heads as u32, 1, 1),
            block_dim: (SINGLE_QUERY_ATTENTION_BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*query.buffer()),
            slice(*key_cache.buffer()),
            slice(*value_cache.buffer()),
            geometry.seq_len as u32,
            geometry.n_heads as u32,
            geometry.n_kv_heads as u32,
            geometry.head_dim as u32,
            slice_mut(*output.buffer_mut())
        ]
    }
}

pub fn prefill_causal_attention(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    query_batch: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    prompt_len: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    assert!(
        prompt_len <= SINGLE_QUERY_ATTENTION_MAX_SEQ,
        "fused prefill attention prompt length exceeds kernel capacity"
    );
    let geometry = AttentionGeometry::new(prompt_len, max_seq_len, n_heads, n_kv_heads, head_dim)
        .expect("attention geometry mismatch");
    let q_len = geometry.query_len();
    assert!(
        query_batch.len() >= prompt_len * q_len,
        "prefill attention query matrix shape mismatch"
    );
    let key_cache =
        DeviceTensor3::<f32, SeqHeadDimMajor>::packed(key_cache, max_seq_len, n_kv_heads, head_dim)
            .expect("key cache shape mismatch");
    let value_cache = DeviceTensor3::<f32, SeqHeadDimMajor>::packed(
        value_cache,
        max_seq_len,
        n_kv_heads,
        head_dim,
    )
    .expect("value cache shape mismatch");
    assert!(
        output.len() >= prompt_len * q_len,
        "prefill attention output matrix shape mismatch"
    );
    let mut output = DeviceVectorMut::new(output);

    cuda_launch! {
        kernel: prefill_causal_attention_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (geometry.n_heads as u32, geometry.seq_len as u32, 1),
            block_dim: (SINGLE_QUERY_ATTENTION_BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*query_batch),
            slice(*key_cache.buffer()),
            slice(*value_cache.buffer()),
            geometry.seq_len as u32,
            geometry.n_heads as u32,
            geometry.n_kv_heads as u32,
            geometry.head_dim as u32,
            slice_mut(*output.buffer_mut())
        ]
    }
}

pub fn softmax_value_to_matrix_row(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    scores: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    seq_len: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    output_row: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let geometry = AttentionGeometry::new(seq_len, max_seq_len, n_heads, n_kv_heads, head_dim)
        .expect("attention geometry mismatch");
    let scores = DeviceMatrix::<f32, RowMajor>::packed(scores, n_heads, max_seq_len)
        .expect("attention score shape mismatch");
    let value_cache = DeviceTensor3::<f32, SeqHeadDimMajor>::packed(
        value_cache,
        max_seq_len,
        n_kv_heads,
        head_dim,
    )
    .expect("value cache shape mismatch");
    let output_cols = n_heads
        .checked_mul(head_dim)
        .expect("attention output row shape overflow");
    assert!(
        output.len() >= (output_row + 1) * output_cols,
        "attention output matrix row shape mismatch"
    );
    let mut output = DeviceVectorMut::new(output);

    cuda_launch! {
        kernel: softmax_value_to_matrix_row_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (geometry.n_heads as u32, 1, 1),
            block_dim: (ATTENTION_SOFTMAX_BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*scores.buffer()),
            slice(*value_cache.buffer()),
            geometry.seq_len as u32,
            geometry.max_seq_len as u32,
            geometry.n_heads as u32,
            geometry.n_kv_heads as u32,
            geometry.head_dim as u32,
            output_row as u32,
            slice_mut(*output.buffer_mut())
        ]
    }
}

pub fn argmax_f32(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    logits: &DeviceBuffer<f32>,
    token_out: &mut DeviceBuffer<u32>,
    logit_out: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let logits = DeviceVector::new(logits);
    let token_out = DeviceVectorMut::new(token_out);
    let logit_out = DeviceVectorMut::new(logit_out);
    let problem =
        ArgmaxProblem::new(logits, token_out, logit_out).expect("argmax operand shape mismatch");
    let (logits, mut token_out, mut logit_out) = problem.into_parts();

    cuda_launch! {
        kernel: argmax_f32_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (DefaultLogitSelectionPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [slice(*logits.buffer()), slice_mut(*token_out.buffer_mut()), slice_mut(*logit_out.buffer_mut())]
    }
}

pub fn argmax_f32_packed(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    logits: &DeviceBuffer<f32>,
    packed_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    assert!(
        !packed_out.is_empty(),
        "packed argmax output must have at least one element"
    );
    let logits = DeviceVector::new(logits);
    let mut packed_out = DeviceVectorMut::new(packed_out);

    cuda_launch! {
        kernel: argmax_f32_packed_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (DefaultLogitSelectionPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [slice(*logits.buffer()), slice_mut(*packed_out.buffer_mut())]
    }
}

pub fn top_k_f32(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    logits: &DeviceBuffer<f32>,
    k: usize,
    token_out: &mut DeviceBuffer<u32>,
    logit_out: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    let logits = DeviceVector::new(logits);
    let token_out = DeviceVectorMut::new(token_out);
    let logit_out = DeviceVectorMut::new(logit_out);
    let problem =
        TopKProblem::new(logits, k, token_out, logit_out).expect("top-k operand shape mismatch");
    let k = problem.k();
    let (logits, mut token_out, mut logit_out) = problem.into_parts();

    cuda_launch! {
        kernel: top_k_f32_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (DefaultLogitSelectionPlan::block_threads(), 1, 1),
            shared_mem_bytes: 0,
        },
        args: [
            slice(*logits.buffer()),
            k as u32,
            slice_mut(*token_out.buffer_mut()),
            slice_mut(*logit_out.buffer_mut())
        ]
    }
}
