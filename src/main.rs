use std::{
    collections::HashSet,
    env,
    error::Error,
    fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
    process,
    sync::Arc,
    time::Instant,
};

mod cuda_worker;

use cuda_core::{CudaContext, CudaModule, CudaStream, DeviceBuffer};
use cuda_worker::{CudaWorkerPool, SMOKE_LAUNCH_TAPE};
use nn_rust_inference::{
    dtypes::{Bf16, DType},
    inference::{
        self, ChatBackendComparisonResult, ChatForcedLogitsTraceOptions, ChatInferenceOptions,
        ChatLogitsOptions, ChatLogitsResult, ChatLogitsTraceOptions, ChatLogitsTraceResult,
        ChatTraceComparisonSuiteOptions, GenerationBackendComparisonSuiteSummary,
        GenerationBackendComparisonSummary, GenerationBackendLogitsComparisonSuiteSummary,
        GenerationComparisonDriver, GenerationFinishReason, InferenceBackend, SamplingOptions,
        SystemPrompt, TokenLogit,
    },
    kernels::activation::swiglu_reference,
    layout::{ColumnMajor, Layout2D, MatrixLayout, RowMajor},
    model::{
        Bf16Top1Plan, GreedyGenerationStep, MinistralAllLinearInt8Runtime, MinistralTextRuntime,
        Qwen35AttentionWeightLayout, RuntimeMemoryStats, TextConfig, TextLayerKind, TextModelKind,
        qwen35_full_layer_smoke, qwen35_linear_layer_smoke, qwen35_load_layer_smoke,
        qwen35_prefix_layers_smoke, qwen35_weight_layout_report,
    },
    ops, runtime,
    safetensors::{ModelTensor, ModelWeights, TensorInfo, model_tensor_alias},
    tokenizer::TekkenTokenizer,
};
use nn_rust_quantization::RowwiseScaledI8Matrix;

type AppResult<T> = std::result::Result<T, Box<dyn Error>>;

const N: usize = 1024;
const DEFAULT_MINISTRAL_DIR: &str = "models/Ministral-3-8B-Reasoning-2512";
const DEFAULT_QWEN3_6_27B_DIR: &str = "models/Qwen3.6-27B";

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

async fn run() -> AppResult<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    run_cli_async(args).await
}

async fn run_cli_async(args: Vec<String>) -> AppResult<()> {
    let mut args = args;
    let command = if args.is_empty() {
        "smoke".to_string()
    } else {
        args.remove(0)
    };

    if matches!(command.as_str(), "smoke-workers" | "worker-smoke") {
        return run_smoke_workers(&args).await;
    }

    let result = tokio::task::spawn_blocking(move || {
        run_cli_command(command, args).map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| io::Error::other(format!("async CLI task failed: {error}")))?;
    result.map_err(|error| io::Error::other(error).into())
}

fn run_cli_command(command: String, args: Vec<String>) -> AppResult<()> {
    match command.as_str() {
        "smoke" => run_smoke(),
        "gemm-stress" | "matmul-stress" => run_gemm_stress(&args),
        "ministral-gemm-stress" | "ministral-matmul-stress" => run_ministral_gemm_stress(&args),
        "decode-matvec-bench" | "matvec-bench" => run_decode_matvec_bench(&args),
        "logit-stress" | "logits-stress" => run_logit_stress(&args),
        "attention-stress" => run_attention_stress(&args),
        "ministral-bf16-prefill-bench" | "ministral-prefill-bench" => {
            run_ministral_bf16_prefill_bench(&args)
        }
        "ministral-bf16-decode-bench" | "ministral-decode-bench" => {
            run_ministral_bf16_decode_bench(&args)
        }
        "ministral-exported-decode-bench" | "ministral-all-linear-int8-decode-bench" => {
            run_ministral_exported_decode_bench(&args)
        }
        "ministral-exported-prefill-compare" | "ministral-all-linear-int8-prefill-compare" => {
            run_ministral_exported_prefill_compare(&args)
        }
        "ministral-bf16-prefill-compare" | "ministral-prefill-compare" => {
            run_ministral_bf16_prefill_compare(&args)
        }
        "ministral-bf16-generation-compare" | "ministral-generation-compare" => {
            run_ministral_bf16_generation_compare(&args)
        }
        "ministral-stop-smoke" => run_ministral_stop_smoke(&args),
        "ministral-chat-backend-smoke" => run_ministral_chat_backend_smoke(&args),
        "ministral-chat-all-linear-int8-smoke" | "ministral-chat-all-linear-int8-session-smoke" => {
            run_ministral_chat_all_linear_int8_smoke(&args)
        }
        "ministral-quantization-manifest" | "ministral-quantize-plan" => {
            run_ministral_quantization_manifest(&args)
        }
        "ministral-quantize-export" | "ministral-quantization-export" => {
            run_ministral_quantization_export(&args)
        }
        "ministral-weight-source-compare" | "ministral-weights-source-compare" => {
            run_ministral_weight_source_compare(&args)
        }
        "ministral-quantize-validate" | "ministral-quantization-validate" => {
            run_ministral_quantization_validate(&args)
        }
        "ministral-quantize-eval" | "ministral-quantization-eval" => {
            run_ministral_quantize_eval(&args)
        }
        "ministral-quantization-suite" | "ministral-quantization-drift-suite" => {
            run_ministral_quantization_suite(&args)
        }
        "ministral-output-projection-export-probe" => {
            run_ministral_output_projection_export_probe(&args)
        }
        "qwen-weight-smoke" | "qwen3-5-weight-smoke" => run_qwen_weight_smoke(&args),
        "qwen-layer-load-smoke" | "qwen3-5-layer-load-smoke" => run_qwen_layer_load_smoke(&args),
        "qwen-full-layer-smoke" | "qwen3-5-full-layer-smoke" => run_qwen_full_layer_smoke(&args),
        "qwen-linear-layer-smoke" | "qwen3-5-linear-layer-smoke" => {
            run_qwen_linear_layer_smoke(&args)
        }
        "qwen-prefix-layers-smoke" | "qwen3-5-prefix-layers-smoke" => {
            run_qwen_prefix_layers_smoke(&args)
        }
        "ministral-eval" => run_ministral_eval(&args),
        "ministral-eval-exported" => run_ministral_eval_exported(&args),
        "ministral-chat" | "ministral-chat-suite" => run_ministral_chat_suite(&args),
        "ministral-chat-exported" | "ministral-chat-exported-generate" => {
            run_ministral_chat_exported_suite(&args)
        }
        "ministral-text-generate" | "ministral-text-suite" => run_ministral_text_suite(&args),
        "ministral-text-exported-generate" | "ministral-text-exported-suite" => {
            run_ministral_text_exported_suite(&args)
        }
        "ministral-text-logits" => run_ministral_text_logits_suite(&args),
        "ministral-text-exported-logits" | "ministral-text-logits-exported" => {
            run_ministral_text_exported_logits_suite(&args)
        }
        "ministral-text-trace" => run_ministral_text_trace_suite(&args),
        "ministral-text-exported-trace" | "ministral-text-trace-exported" => {
            run_ministral_text_exported_trace_suite(&args)
        }
        "ministral-text-logits-compare" => run_ministral_text_logits_compare(&args),
        "ministral-text-exported-logits-compare" | "ministral-text-logits-exported-compare" => {
            run_ministral_text_exported_logits_compare(&args)
        }
        "ministral-text-forced-compare" => run_ministral_text_forced_compare(&args),
        "ministral-text-forced-target-compare" => run_ministral_text_forced_target_compare(&args),
        "ministral-text-forced-target-exported-compare" => {
            run_ministral_text_forced_target_exported_compare(&args)
        }
        "ministral-text-exported-compare" => run_ministral_text_exported_compare(&args),
        "ministral-text-compare" => run_ministral_text_compare(&args),
        "ministral-tokens-generate" | "ministral-tokens-suite" => run_ministral_tokens_suite(&args),
        "qwen-tokens-generate" | "qwen-tokens-suite" => run_qwen_tokens_suite(&args),
        "ministral-tokens-exported-generate" | "ministral-tokens-exported-suite" => {
            run_ministral_tokens_exported_suite(&args)
        }
        "ministral-tokens-logits" => run_ministral_tokens_logits_suite(&args),
        "ministral-tokens-exported-logits" | "ministral-tokens-logits-exported" => {
            run_ministral_tokens_exported_logits_suite(&args)
        }
        "ministral-tokens-trace" => run_ministral_tokens_trace_suite(&args),
        "ministral-tokens-exported-trace" | "ministral-tokens-trace-exported" => {
            run_ministral_tokens_exported_trace_suite(&args)
        }
        "ministral-tokens-eval" => run_ministral_tokens_eval(&args),
        "ministral-tokens-eval-exported" | "ministral-tokens-exported-eval" => {
            run_ministral_tokens_eval_exported(&args)
        }
        "ministral-tokens-forced-compare" => run_ministral_tokens_forced_compare(&args),
        "ministral-tokens-exported-forced-compare" => {
            run_ministral_tokens_exported_forced_compare(&args)
        }
        "ministral-tokens-logits-compare" => run_ministral_tokens_logits_compare(&args),
        "ministral-tokens-exported-logits-compare" => {
            run_ministral_tokens_exported_logits_compare(&args)
        }
        "ministral-tokens-exported-compare" => run_ministral_tokens_exported_compare(&args),
        "ministral-tokens-compare" => run_ministral_tokens_compare(&args),
        "ministral-chat-logits-smoke" | "ministral-chat-logits" => {
            run_ministral_chat_logits_smoke(&args)
        }
        "ministral-chat-exported-logits" | "ministral-chat-logits-exported" => {
            run_ministral_chat_exported_logits_suite(&args)
        }
        "ministral-chat-logits-compare" => run_ministral_chat_logits_compare(&args),
        "ministral-chat-exported-logits-compare" | "ministral-chat-logits-exported-compare" => {
            run_ministral_chat_exported_logits_compare(&args)
        }
        "ministral-chat-trace" | "ministral-chat-logits-trace-smoke" => {
            run_ministral_chat_trace_suite(&args)
        }
        "ministral-chat-exported-trace" | "ministral-chat-trace-exported" => {
            run_ministral_chat_exported_trace_suite(&args)
        }
        "ministral-chat-forced-trace-smoke" => run_ministral_chat_forced_trace_smoke(&args),
        "ministral-chat-forced-trace-suite-smoke" => {
            run_ministral_chat_forced_trace_suite_smoke(&args)
        }
        "ministral-chat-forced-target-compare" => run_ministral_chat_forced_target_compare(&args),
        "ministral-chat-forced-target-exported-compare" => {
            run_ministral_chat_forced_target_exported_compare(&args)
        }
        "ministral-chat-exported-compare" => run_ministral_chat_exported_compare(&args),
        "ministral-chat-compare" => run_ministral_chat_compare(&args),
        other => Err(invalid_input(format!(
            "unknown command {other:?}; expected `smoke`, `smoke-workers`, `gemm-stress`, `ministral-gemm-stress`, `decode-matvec-bench`, `logit-stress`, `attention-stress`, \
             `ministral-bf16-prefill-bench`, `ministral-bf16-decode-bench`, \
             `ministral-exported-decode-bench`, `ministral-exported-prefill-compare`, \
             `ministral-bf16-prefill-compare`, \
             `ministral-bf16-generation-compare`, \
             `ministral-stop-smoke`, \
             `ministral-chat-backend-smoke`, \
             `ministral-chat-all-linear-int8-smoke`, `ministral-chat`, \
             `ministral-quantization-manifest`, `ministral-quantize-export`, \
             `ministral-weight-source-compare`, \
             `ministral-quantize-validate`, `ministral-quantize-eval`, \
             `ministral-quantization-suite`, \
             `ministral-output-projection-export-probe`, `ministral-eval`, \
             `ministral-eval-exported`, \
             `ministral-chat-exported`, `ministral-text-generate`, \
             `ministral-text-exported-generate`, `ministral-text-logits`, `ministral-text-trace`, \
             `ministral-text-exported-logits`, `ministral-text-exported-trace`, \
             `ministral-text-logits-compare`, `ministral-text-exported-logits-compare`, \
             `ministral-text-forced-compare`, \
             `ministral-text-forced-target-compare`, \
             `ministral-text-forced-target-exported-compare`, `ministral-text-exported-compare`, \
             `ministral-text-compare`, \
             `ministral-tokens-generate`, `ministral-tokens-exported-generate`, \
             `qwen-weight-smoke`, `qwen-layer-load-smoke`, `qwen-full-layer-smoke`, \
             `qwen-linear-layer-smoke`, `qwen-prefix-layers-smoke`, `qwen-tokens-generate`, \
             `ministral-tokens-logits`, `ministral-tokens-exported-logits`, \
             `ministral-tokens-trace`, `ministral-tokens-exported-trace`, \
             `ministral-tokens-eval`, `ministral-tokens-eval-exported`, \
             `ministral-tokens-forced-compare`, `ministral-tokens-exported-forced-compare`, \
             `ministral-tokens-logits-compare`, `ministral-tokens-exported-logits-compare`, \
             `ministral-tokens-exported-compare`, \
             `ministral-tokens-compare`, `ministral-chat-logits-smoke`, \
             `ministral-chat-exported-logits`, `ministral-chat-logits-compare`, \
             `ministral-chat-exported-logits-compare`, \
             `ministral-chat-trace`, `ministral-chat-exported-trace`, \
             `ministral-chat-forced-trace-smoke`, \
             `ministral-chat-forced-trace-suite-smoke`, \
             `ministral-chat-forced-target-compare`, \
             `ministral-chat-forced-target-exported-compare`, `ministral-chat-exported-compare`, \
             or `ministral-chat-compare`"
        ))),
    }
}

fn run_smoke() -> AppResult<()> {
    let (stream, module) = cuda_handles()?;

    run_relu(&stream, &module)?;
    run_swiglu(&stream, &module)?;
    run_vecadd(&stream, &module)?;

    println!("all tests passed");
    Ok(())
}

async fn run_smoke_workers(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let worker_count = parse_optional_usize(args, &mut index, 3, "worker_count")?;
    let queue_depth = parse_optional_usize(args, &mut index, 8, "queue_depth")?;
    if index != args.len() {
        return Err(invalid_input(
            "smoke-workers accepts at most [worker_count] [queue_depth]",
        ));
    }
    if worker_count == 0 {
        return Err(invalid_input("smoke-workers worker_count must be nonzero"));
    }
    if queue_depth == 0 {
        return Err(invalid_input("smoke-workers queue_depth must be nonzero"));
    }

    let device_index = cuda_device_index_from_env()?;
    let pool = CudaWorkerPool::new(worker_count, queue_depth, device_index)?;
    let [relu_descriptor, swiglu_descriptor, vecadd_descriptor] = SMOKE_LAUNCH_TAPE;
    let relu = pool.submit(relu_descriptor);
    let swiglu = pool.submit(swiglu_descriptor);
    let vecadd = pool.submit(vecadd_descriptor);
    tokio::try_join!(relu, swiglu, vecadd)?;

    println!(
        "worker smoke passed: workers={} queue_depth={queue_depth}",
        pool.worker_count()
    );
    Ok(())
}

fn run_relu(stream: &Arc<CudaStream>, module: &Arc<CudaModule>) -> AppResult<()> {
    println!("Test 1: ReLU kernel");

    let data: Vec<f32> = (0..N).map(|i| (i as f32 - 512.0) / 256.0).collect();
    let input = DeviceBuffer::from_host(stream, &data)?;
    let mut output = DeviceBuffer::<f32>::zeroed(stream, N)?;

    ops::relu(stream, module, &input, &mut output)?;

    let result = output.to_host_vec(stream)?;
    assert_close(result[0], 0.0, 1e-5, "relu[0]")?;
    assert_close(result[768], 1.0, 1e-5, "relu[768]")?;

    println!("  ReLU passed");
    Ok(())
}

fn run_swiglu(stream: &Arc<CudaStream>, module: &Arc<CudaModule>) -> AppResult<()> {
    println!("Test 2: SwiGLU kernel");

    let data: Vec<f32> = (0..N).map(|i| (i as f32 - 512.0) / 256.0).collect();
    let input = DeviceBuffer::from_host(stream, &data)?;
    let mut output = DeviceBuffer::<f32>::zeroed(stream, N)?;

    ops::swiglu(stream, module, &input, &mut output)?;

    let result = output.to_host_vec(stream)?;
    for (i, (&actual, &x)) in result.iter().zip(data.iter()).enumerate() {
        assert_close(actual, swiglu_reference(x), 1e-3, &format!("swiglu[{i}]"))?;
    }

    println!("  SwiGLU passed");
    Ok(())
}

fn run_vecadd(stream: &Arc<CudaStream>, module: &Arc<CudaModule>) -> AppResult<()> {
    println!("Test 3: Vector addition kernel");

    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
    let dev_a = DeviceBuffer::from_host(stream, &a)?;
    let dev_b = DeviceBuffer::from_host(stream, &b)?;
    let mut dev_out = DeviceBuffer::<f32>::zeroed(stream, N)?;

    ops::vecadd(stream, module, &dev_a, &dev_b, &mut dev_out)?;

    let result = dev_out.to_host_vec(stream)?;
    for i in 0..N {
        assert_close(result[i], a[i] + b[i], 1e-5, &format!("vecadd[{i}]"))?;
    }

    println!("  Vector addition passed");
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct GemmStressStats {
    element_count: usize,
    max_abs_diff: f32,
    sum_abs_diff: f64,
}

fn run_gemm_stress(args: &[String]) -> AppResult<()> {
    let repeat_count = match args {
        [] => 1,
        [value] => value.parse::<usize>().map_err(|error| {
            invalid_input(format!(
                "gemm-stress repeat count must be a positive integer, got {value:?}: {error}"
            ))
        })?,
        _ => {
            return Err(invalid_input(
                "gemm-stress accepts at most one optional repeat count",
            ));
        }
    };
    if repeat_count == 0 {
        return Err(invalid_input("gemm-stress repeat count must be nonzero"));
    }

    let (stream, module) = cuda_handles()?;
    let shapes = [
        (1, 1, 1),
        (3, 5, 7),
        (16, 16, 16),
        (17, 19, 23),
        (31, 33, 29),
        (64, 64, 64),
        (65, 33, 17),
        (128, 96, 64),
    ];

    let start = Instant::now();
    let mut case_count = 0usize;
    let mut element_count = 0usize;
    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;

    for repeat in 0..repeat_count {
        for (shape_index, &(m, n, k)) in shapes.iter().enumerate() {
            let seed = ((repeat as u64) << 32) ^ shape_index as u64 ^ 0x9e37_79b9_7f4a_7c15;
            accumulate_gemm_stats(
                run_gemm_stress_case::<RowMajor, RowMajor, RowMajor>(
                    &stream, &module, m, n, k, seed, "rrr",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_gemm_stress_case::<RowMajor, ColumnMajor, RowMajor>(
                    &stream,
                    &module,
                    m,
                    n,
                    k,
                    seed ^ 0x11,
                    "rcr",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_gemm_stress_case::<ColumnMajor, RowMajor, RowMajor>(
                    &stream,
                    &module,
                    m,
                    n,
                    k,
                    seed ^ 0x22,
                    "crr",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_gemm_stress_case::<RowMajor, RowMajor, ColumnMajor>(
                    &stream,
                    &module,
                    m,
                    n,
                    k,
                    seed ^ 0x33,
                    "rrc",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_gemm_bf16_stress_case::<RowMajor, RowMajor, RowMajor>(
                    &stream,
                    &module,
                    m,
                    n,
                    k,
                    seed ^ 0x44,
                    "bf16-rrr",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_gemm_bf16_stress_case::<RowMajor, ColumnMajor, RowMajor>(
                    &stream,
                    &module,
                    m,
                    n,
                    k,
                    seed ^ 0x55,
                    "bf16-rcr",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_gemm_bf16_stress_case::<ColumnMajor, RowMajor, RowMajor>(
                    &stream,
                    &module,
                    m,
                    n,
                    k,
                    seed ^ 0x66,
                    "bf16-crr",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_gemm_bf16_stress_case::<RowMajor, RowMajor, ColumnMajor>(
                    &stream,
                    &module,
                    m,
                    n,
                    k,
                    seed ^ 0x77,
                    "bf16-rrc",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_linear_batched_bf16_stress_case(
                    &stream,
                    &module,
                    m,
                    n,
                    k,
                    seed ^ 0x88,
                    "linear-bf16",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_linear_batched_i8_scaled_stress_case(
                    &stream,
                    &module,
                    m,
                    n,
                    k,
                    seed ^ 0x89,
                    "linear-i8-scaled",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_linear_qkv_batched_bf16_stress_case(
                    &stream,
                    &module,
                    m,
                    n,
                    n.div_ceil(2),
                    k,
                    seed ^ 0x9a,
                    "linear-qkv-bf16",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_silu_gate_up_via_gemm_stress_case(
                    &stream,
                    &module,
                    m,
                    n,
                    k,
                    seed ^ 0x9b,
                    "silu-gate-up-bf16",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_silu_gate_up_bf16_stress_case(
                    &stream,
                    &module,
                    k,
                    n,
                    seed ^ 0x9c,
                    "silu-gate-up-decode-bf16",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
            accumulate_gemm_stats(
                run_linear_residual_bf16_stress_case(
                    &stream,
                    &module,
                    k,
                    n,
                    seed ^ 0x99,
                    "linear-residual-bf16",
                )?,
                &mut case_count,
                &mut element_count,
                &mut max_abs_diff,
                &mut sum_abs_diff,
            );
        }
    }

    let mean_abs_diff = if element_count == 0 {
        0.0
    } else {
        sum_abs_diff / element_count as f64
    };
    println!(
        "GEMM stress passed: cases={} elements={} max_abs_diff={:.8} mean_abs_diff={:.12} elapsed_seconds={:.6}",
        case_count,
        element_count,
        max_abs_diff,
        mean_abs_diff,
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn run_ministral_gemm_stress(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = if index < args.len()
        && !args[index].starts_with("--")
        && args[index].parse::<usize>().is_err()
    {
        let path = PathBuf::from(&args[index]);
        index += 1;
        path
    } else {
        PathBuf::from(DEFAULT_MINISTRAL_DIR)
    };
    let batch = parse_optional_usize(args, &mut index, 1, "batch")?;
    let repeat_count = parse_optional_usize(args, &mut index, 1, "repeat_count")?;
    if index != args.len() {
        return Err(invalid_input(format!(
            "ministral-gemm-stress accepts at most [model_dir] [batch] [repeat_count], got extra argument {:?}",
            args[index]
        )));
    }
    if batch == 0 {
        return Err(invalid_input("ministral-gemm-stress batch must be nonzero"));
    }
    if repeat_count == 0 {
        return Err(invalid_input(
            "ministral-gemm-stress repeat_count must be nonzero",
        ));
    }

    let config = TextConfig::from_model_dir(&model_dir)?;
    let q_len = config.n_heads * config.head_dim;
    let kv_len = config.n_kv_heads * config.head_dim;
    let (stream, module) = cuda_handles()?;

    println!(
        "Ministral GEMM stress: model_dir={} batch={} repeats={} dim={} hidden_dim={} q_len={} kv_len={}",
        model_dir.display(),
        batch,
        repeat_count,
        config.dim,
        config.hidden_dim,
        q_len,
        kv_len
    );

    let start = Instant::now();
    let mut case_count = 0usize;
    let mut element_count = 0usize;
    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;

    for repeat in 0..repeat_count {
        let seed = ((repeat as u64) << 32) ^ 0x6d69_6e69_7374_7261;
        accumulate_gemm_stats(
            run_linear_qkv_batched_bf16_stress_case(
                &stream,
                &module,
                batch,
                q_len,
                kv_len,
                config.dim,
                seed ^ 0x11,
                "ministral-qkv-bf16",
            )?,
            &mut case_count,
            &mut element_count,
            &mut max_abs_diff,
            &mut sum_abs_diff,
        );
        accumulate_gemm_stats(
            run_linear_batched_bf16_stress_case(
                &stream,
                &module,
                batch,
                config.dim,
                q_len,
                seed ^ 0x22,
                "ministral-wo-bf16",
            )?,
            &mut case_count,
            &mut element_count,
            &mut max_abs_diff,
            &mut sum_abs_diff,
        );
        accumulate_gemm_stats(
            run_silu_gate_up_via_gemm_stress_case(
                &stream,
                &module,
                batch,
                config.hidden_dim,
                config.dim,
                seed ^ 0x33,
                "ministral-ffn-gate-up-bf16",
            )?,
            &mut case_count,
            &mut element_count,
            &mut max_abs_diff,
            &mut sum_abs_diff,
        );
        accumulate_gemm_stats(
            run_linear_batched_bf16_stress_case(
                &stream,
                &module,
                batch,
                config.dim,
                config.hidden_dim,
                seed ^ 0x44,
                "ministral-ffn-down-bf16",
            )?,
            &mut case_count,
            &mut element_count,
            &mut max_abs_diff,
            &mut sum_abs_diff,
        );
    }

    let mean_abs_diff = if element_count == 0 {
        0.0
    } else {
        sum_abs_diff / element_count as f64
    };
    println!(
        "Ministral GEMM stress passed: cases={} elements={} max_abs_diff={:.8} mean_abs_diff={:.12} elapsed_seconds={:.6}",
        case_count,
        element_count,
        max_abs_diff,
        mean_abs_diff,
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn run_decode_matvec_bench(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = if index < args.len()
        && !args[index].starts_with("--")
        && args[index].parse::<usize>().is_err()
    {
        let path = PathBuf::from(&args[index]);
        index += 1;
        path
    } else {
        PathBuf::from(DEFAULT_MINISTRAL_DIR)
    };
    let repeat_count = parse_optional_usize(args, &mut index, 10, "repeat_count")?;
    if index != args.len() {
        return Err(invalid_input(format!(
            "decode-matvec-bench accepts at most [model_dir] [repeat_count], got extra argument {:?}",
            args[index]
        )));
    }
    if repeat_count == 0 {
        return Err(invalid_input(
            "decode-matvec-bench repeat count must be nonzero",
        ));
    }

    let (stream, module) = cuda_handles()?;
    let config = TextConfig::from_model_dir(&model_dir)?;
    let q_len = config.n_heads * config.head_dim;
    let kv_len = config.n_kv_heads * config.head_dim;

    println!(
        "Decode matvec bench: model_dir={} repeats={} dim={} hidden_dim={} q_len={} kv_len={}",
        model_dir.display(),
        repeat_count,
        config.dim,
        config.hidden_dim,
        q_len,
        kv_len
    );

    run_decode_linear_triple_bf16_bench(
        &stream,
        &module,
        repeat_count,
        config.dim,
        q_len,
        kv_len,
        0x243f_6a88_85a3_08d3,
    )?;
    run_decode_silu_gate_up_bf16_bench(
        &stream,
        &module,
        repeat_count,
        config.dim,
        config.hidden_dim,
        0x1319_8a2e_0370_7344,
    )?;
    run_decode_linear_residual_bf16_bench(
        &stream,
        &module,
        repeat_count,
        config.hidden_dim,
        config.dim,
        0xa409_3822_299f_31d0,
    )?;
    run_decode_linear_top1_bf16_bench(
        &stream,
        &module,
        repeat_count,
        config.dim,
        config.vocab_size,
        0x9e37_79b9_7f4a_7c15,
    )?;

    Ok(())
}

fn run_decode_linear_triple_bf16_bench(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    repeat_count: usize,
    input_dim: usize,
    q_output_dim: usize,
    kv_output_dim: usize,
    mut seed: u64,
) -> AppResult<()> {
    let input_layout = MatrixLayout::<RowMajor>::packed(1, input_dim);
    let q_weight_layout = MatrixLayout::<RowMajor>::packed(q_output_dim, input_dim);
    let kv_weight_layout = MatrixLayout::<RowMajor>::packed(kv_output_dim, input_dim);
    let mut input = vec![0.0_f32; input_layout.capacity()];
    let mut wq = vec![Bf16::from_bits(0); q_weight_layout.capacity()];
    let mut wk = vec![Bf16::from_bits(0); kv_weight_layout.capacity()];
    let mut wv = vec![Bf16::from_bits(0); kv_weight_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input, &input_layout, 1, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut wq,
        &q_weight_layout,
        q_output_dim,
        input_dim,
        &mut seed,
    );
    fill_bf16_matrix::<RowMajor>(
        &mut wk,
        &kv_weight_layout,
        kv_output_dim,
        input_dim,
        &mut seed,
    );
    fill_bf16_matrix::<RowMajor>(
        &mut wv,
        &kv_weight_layout,
        kv_output_dim,
        input_dim,
        &mut seed,
    );

    let dev_input = DeviceBuffer::from_host(stream, &input)?;
    let dev_wq = DeviceBuffer::from_host(stream, &wq)?;
    let dev_wk = DeviceBuffer::from_host(stream, &wk)?;
    let dev_wv = DeviceBuffer::from_host(stream, &wv)?;
    let mut dev_q = DeviceBuffer::<f32>::zeroed(stream, q_output_dim)?;
    let mut dev_k = DeviceBuffer::<f32>::zeroed(stream, kv_output_dim)?;
    let mut dev_v = DeviceBuffer::<f32>::zeroed(stream, kv_output_dim)?;

    ops::linear_triple_bf16(
        stream, module, &dev_input, &dev_wq, &dev_wk, &dev_wv, &mut dev_q, &mut dev_k, &mut dev_v,
    )?;
    stream.synchronize()?;
    let q = dev_q.to_host_vec(stream)?;
    let k = dev_k.to_host_vec(stream)?;
    let v = dev_v.to_host_vec(stream)?;
    verify_linear_samples(
        "decode-q",
        &input,
        &wq,
        &q_weight_layout,
        &q,
        q_output_dim,
        input_dim,
    )?;
    verify_linear_samples(
        "decode-k",
        &input,
        &wk,
        &kv_weight_layout,
        &k,
        kv_output_dim,
        input_dim,
    )?;
    verify_linear_samples(
        "decode-v",
        &input,
        &wv,
        &kv_weight_layout,
        &v,
        kv_output_dim,
        input_dim,
    )?;

    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat_count {
        ops::linear_triple_bf16(
            stream, module, &dev_input, &dev_wq, &dev_wk, &dev_wv, &mut dev_q, &mut dev_k,
            &mut dev_v,
        )?;
    }
    stream.synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    let madds = input_dim
        .checked_mul(q_output_dim + 2 * kv_output_dim)
        .ok_or_else(|| invalid_input("decode triple BF16 benchmark operation count overflow"))?;
    print_decode_matvec_bench_result("linear-triple-bf16", repeat_count, madds, seconds);
    Ok(())
}

fn run_decode_silu_gate_up_bf16_bench(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    repeat_count: usize,
    input_dim: usize,
    output_dim: usize,
    mut seed: u64,
) -> AppResult<()> {
    let input_layout = MatrixLayout::<RowMajor>::packed(1, input_dim);
    let weight_layout = MatrixLayout::<RowMajor>::packed(output_dim, input_dim);
    let mut input = vec![0.0_f32; input_layout.capacity()];
    let mut gate_weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    let mut up_weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input, &input_layout, 1, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut gate_weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );
    fill_bf16_matrix::<RowMajor>(
        &mut up_weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );

    let dev_input = DeviceBuffer::from_host(stream, &input)?;
    let dev_gate_weight = DeviceBuffer::from_host(stream, &gate_weight)?;
    let dev_up_weight = DeviceBuffer::from_host(stream, &up_weight)?;
    let mut dev_gate = DeviceBuffer::<f32>::zeroed(stream, output_dim)?;
    let mut dev_up = DeviceBuffer::<f32>::zeroed(stream, output_dim)?;
    let mut dev_output = DeviceBuffer::<f32>::zeroed(stream, output_dim)?;
    let mut dev_output_rows8 = DeviceBuffer::<f32>::zeroed(stream, output_dim)?;

    ops::linear_pair_bf16(
        stream,
        module,
        &dev_input,
        &dev_gate_weight,
        &dev_up_weight,
        &mut dev_gate,
        &mut dev_up,
    )?;
    stream.synchronize()?;
    let gate = dev_gate.to_host_vec(stream)?;
    let up = dev_up.to_host_vec(stream)?;
    verify_linear_samples(
        "decode-gate-pair",
        &input,
        &gate_weight,
        &weight_layout,
        &gate,
        output_dim,
        input_dim,
    )?;
    verify_linear_samples(
        "decode-up-pair",
        &input,
        &up_weight,
        &weight_layout,
        &up,
        output_dim,
        input_dim,
    )?;

    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat_count {
        ops::linear_pair_bf16(
            stream,
            module,
            &dev_input,
            &dev_gate_weight,
            &dev_up_weight,
            &mut dev_gate,
            &mut dev_up,
        )?;
    }
    stream.synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    let madds = 2usize
        .checked_mul(output_dim)
        .and_then(|n| n.checked_mul(input_dim))
        .ok_or_else(|| invalid_input("decode paired BF16 benchmark operation count overflow"))?;
    print_decode_matvec_bench_result("linear-pair-bf16", repeat_count, madds, seconds);

    ops::silu_gate_up_bf16(
        stream,
        module,
        &dev_input,
        &dev_gate_weight,
        &dev_up_weight,
        &mut dev_output,
    )?;
    stream.synchronize()?;
    let output = dev_output.to_host_vec(stream)?;
    verify_silu_gate_up_samples(
        "decode-silu-gate-up",
        &input,
        &gate_weight,
        &up_weight,
        &weight_layout,
        &output,
        output_dim,
        input_dim,
    )?;

    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat_count {
        ops::silu_gate_up_bf16(
            stream,
            module,
            &dev_input,
            &dev_gate_weight,
            &dev_up_weight,
            &mut dev_output,
        )?;
    }
    stream.synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    let madds = 2usize
        .checked_mul(output_dim)
        .and_then(|n| n.checked_mul(input_dim))
        .ok_or_else(|| invalid_input("decode SiLU gate/up benchmark operation count overflow"))?;
    print_decode_matvec_bench_result("silu-gate-up-bf16", repeat_count, madds, seconds);

    ops::silu_gate_up_bf16_rows8(
        stream,
        module,
        &dev_input,
        &dev_gate_weight,
        &dev_up_weight,
        &mut dev_output_rows8,
    )?;
    stream.synchronize()?;
    let output_rows8 = dev_output_rows8.to_host_vec(stream)?;
    verify_silu_gate_up_samples(
        "decode-silu-gate-up-rows8",
        &input,
        &gate_weight,
        &up_weight,
        &weight_layout,
        &output_rows8,
        output_dim,
        input_dim,
    )?;

    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat_count {
        ops::silu_gate_up_bf16_rows8(
            stream,
            module,
            &dev_input,
            &dev_gate_weight,
            &dev_up_weight,
            &mut dev_output_rows8,
        )?;
    }
    stream.synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    print_decode_matvec_bench_result("silu-gate-up-bf16-rows8", repeat_count, madds, seconds);
    Ok(())
}

fn run_decode_linear_residual_bf16_bench(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    repeat_count: usize,
    input_dim: usize,
    output_dim: usize,
    mut seed: u64,
) -> AppResult<()> {
    let input_layout = MatrixLayout::<RowMajor>::packed(1, input_dim);
    let weight_layout = MatrixLayout::<RowMajor>::packed(output_dim, input_dim);
    let output_layout = MatrixLayout::<RowMajor>::packed(1, output_dim);
    let mut input = vec![0.0_f32; input_layout.capacity()];
    let mut weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    let mut residual = vec![0.0_f32; output_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input, &input_layout, 1, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );
    fill_matrix::<RowMajor>(&mut residual, &output_layout, 1, output_dim, &mut seed);

    let dev_input = DeviceBuffer::from_host(stream, &input)?;
    let dev_weight = DeviceBuffer::from_host(stream, &weight)?;
    let dev_residual = DeviceBuffer::from_host(stream, &residual)?;
    let mut dev_output = DeviceBuffer::<f32>::zeroed(stream, output_dim)?;

    ops::linear_residual_bf16(
        stream,
        module,
        &dev_input,
        &dev_weight,
        &dev_residual,
        &mut dev_output,
    )?;
    stream.synchronize()?;
    let output = dev_output.to_host_vec(stream)?;
    verify_linear_residual_samples(
        "decode-linear-residual",
        &input,
        &weight,
        &weight_layout,
        &residual,
        &output,
        output_dim,
        input_dim,
    )?;

    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat_count {
        ops::linear_residual_bf16(
            stream,
            module,
            &dev_input,
            &dev_weight,
            &dev_residual,
            &mut dev_output,
        )?;
    }
    stream.synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    let madds = output_dim
        .checked_mul(input_dim)
        .ok_or_else(|| invalid_input("decode residual BF16 benchmark operation count overflow"))?;
    print_decode_matvec_bench_result("linear-residual-bf16", repeat_count, madds, seconds);
    Ok(())
}

fn run_decode_linear_top1_bf16_bench(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    repeat_count: usize,
    input_dim: usize,
    output_dim: usize,
    mut seed: u64,
) -> AppResult<()> {
    let input_layout = MatrixLayout::<RowMajor>::packed(1, input_dim);
    let weight_layout = MatrixLayout::<RowMajor>::packed(output_dim, input_dim);
    let mut input = vec![0.0_f32; input_layout.capacity()];
    let mut weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input, &input_layout, 1, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );

    let forced_row = if output_dim == 1 { 0 } else { output_dim / 3 };
    let mut expected_logit = 0.0_f32;
    for col in 0..input_dim {
        let input_value = input[input_layout.offset(0, col)];
        let forced_weight = if input_value >= 0.0 { 4.0 } else { -4.0 };
        weight[weight_layout.offset(forced_row, col)] = Bf16::from_f32(forced_weight);
        expected_logit += input_value * forced_weight;
    }

    let dev_input = DeviceBuffer::from_host(stream, &input)?;
    let dev_weight = DeviceBuffer::from_host(stream, &weight)?;
    let partial_count = output_dim;
    let mut dev_partial_tokens = DeviceBuffer::<u32>::zeroed(stream, partial_count)?;
    let mut dev_partial_logits = DeviceBuffer::<f32>::zeroed(stream, partial_count)?;
    let mut dev_packed = DeviceBuffer::<u64>::zeroed(stream, 1)?;

    ops::linear_top1_bf16_rows1(
        stream,
        module,
        &dev_input,
        &dev_weight,
        &mut dev_partial_tokens,
        &mut dev_partial_logits,
        &mut dev_packed,
    )?;
    stream.synchronize()?;
    let rows1_packed = dev_packed.to_host_vec(stream)?[0];
    let rows1_token = rows1_packed as u32;
    let rows1_logit = f32::from_bits((rows1_packed >> 32) as u32);
    let tolerance =
        (1.0e-4_f32 * (input_dim.max(1) as f32).sqrt()).max(expected_logit.abs() * 2.0e-5);
    if rows1_token != forced_row as u32 || (rows1_logit - expected_logit).abs() > tolerance {
        return Err(invalid_data(format!(
            "decode linear top1 rows1 BF16 benchmark verification failed: expected=({}, {:.8}) got=({}, {:.8}) tolerance={:.8}",
            forced_row, expected_logit, rows1_token, rows1_logit, tolerance
        )));
    }

    ops::linear_top1_bf16_rows2(
        stream,
        module,
        &dev_input,
        &dev_weight,
        &mut dev_partial_tokens,
        &mut dev_partial_logits,
        &mut dev_packed,
    )?;
    stream.synchronize()?;
    let rows2_packed = dev_packed.to_host_vec(stream)?[0];
    let rows2_token = rows2_packed as u32;
    let rows2_logit = f32::from_bits((rows2_packed >> 32) as u32);
    if rows2_token != forced_row as u32 || (rows2_logit - expected_logit).abs() > tolerance {
        return Err(invalid_data(format!(
            "decode linear top1 rows2 BF16 benchmark verification failed: expected=({}, {:.8}) got=({}, {:.8}) tolerance={:.8}",
            forced_row, expected_logit, rows2_token, rows2_logit, tolerance
        )));
    }

    ops::linear_top1_bf16(
        stream,
        module,
        &dev_input,
        &dev_weight,
        &mut dev_partial_tokens,
        &mut dev_partial_logits,
        &mut dev_packed,
    )?;
    stream.synchronize()?;
    let packed = dev_packed.to_host_vec(stream)?[0];
    let token = packed as u32;
    let logit = f32::from_bits((packed >> 32) as u32);
    if token != forced_row as u32 || (logit - expected_logit).abs() > tolerance {
        return Err(invalid_data(format!(
            "decode linear top1 BF16 benchmark verification failed: expected=({}, {:.8}) got=({}, {:.8}) tolerance={:.8}",
            forced_row, expected_logit, token, logit, tolerance
        )));
    }

    ops::linear_top1_bf16_rows8(
        stream,
        module,
        &dev_input,
        &dev_weight,
        &mut dev_partial_tokens,
        &mut dev_partial_logits,
        &mut dev_packed,
    )?;
    stream.synchronize()?;
    let rows8_packed = dev_packed.to_host_vec(stream)?[0];
    let rows8_token = rows8_packed as u32;
    let rows8_logit = f32::from_bits((rows8_packed >> 32) as u32);
    if rows8_token != forced_row as u32 || (rows8_logit - expected_logit).abs() > tolerance {
        return Err(invalid_data(format!(
            "decode linear top1 rows8 BF16 benchmark verification failed: expected=({}, {:.8}) got=({}, {:.8}) tolerance={:.8}",
            forced_row, expected_logit, rows8_token, rows8_logit, tolerance
        )));
    }

    let madds = output_dim
        .checked_mul(input_dim)
        .ok_or_else(|| invalid_input("decode top1 BF16 benchmark operation count overflow"))?;

    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat_count {
        ops::linear_top1_bf16_rows1(
            stream,
            module,
            &dev_input,
            &dev_weight,
            &mut dev_partial_tokens,
            &mut dev_partial_logits,
            &mut dev_packed,
        )?;
    }
    stream.synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    print_decode_matvec_bench_result("linear-top1-bf16-rows1", repeat_count, madds, seconds);

    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat_count {
        ops::linear_top1_bf16_rows2(
            stream,
            module,
            &dev_input,
            &dev_weight,
            &mut dev_partial_tokens,
            &mut dev_partial_logits,
            &mut dev_packed,
        )?;
    }
    stream.synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    print_decode_matvec_bench_result("linear-top1-bf16-rows2", repeat_count, madds, seconds);

    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat_count {
        ops::linear_top1_bf16(
            stream,
            module,
            &dev_input,
            &dev_weight,
            &mut dev_partial_tokens,
            &mut dev_partial_logits,
            &mut dev_packed,
        )?;
    }
    stream.synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    print_decode_matvec_bench_result("linear-top1-bf16", repeat_count, madds, seconds);

    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeat_count {
        ops::linear_top1_bf16_rows8(
            stream,
            module,
            &dev_input,
            &dev_weight,
            &mut dev_partial_tokens,
            &mut dev_partial_logits,
            &mut dev_packed,
        )?;
    }
    stream.synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    print_decode_matvec_bench_result("linear-top1-bf16-rows8", repeat_count, madds, seconds);

    run_decode_linear_top1_bf16_interleaved_bench(
        stream,
        module,
        repeat_count,
        madds,
        &dev_input,
        &dev_weight,
        &mut dev_partial_tokens,
        &mut dev_partial_logits,
        &mut dev_packed,
    )?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum DecodeTop1BenchVariant {
    Rows1,
    Rows2,
    Rows4,
    Rows8,
}

impl DecodeTop1BenchVariant {
    const ALL: [Self; 4] = [Self::Rows1, Self::Rows2, Self::Rows4, Self::Rows8];

    fn label(self) -> &'static str {
        match self {
            Self::Rows1 => "linear-top1-bf16-rows1-interleaved",
            Self::Rows2 => "linear-top1-bf16-rows2-interleaved",
            Self::Rows4 => "linear-top1-bf16-interleaved",
            Self::Rows8 => "linear-top1-bf16-rows8-interleaved",
        }
    }
}

fn run_decode_linear_top1_bf16_interleaved_bench(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    repeat_count: usize,
    madds_per_launch: usize,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    partial_tokens: &mut DeviceBuffer<u32>,
    partial_logits: &mut DeviceBuffer<f32>,
    packed: &mut DeviceBuffer<u64>,
) -> AppResult<()> {
    let variants = DecodeTop1BenchVariant::ALL;
    let mut samples = vec![Vec::with_capacity(repeat_count); variants.len()];

    stream.synchronize()?;
    for round in 0..repeat_count {
        for offset in 0..variants.len() {
            let variant_index = (round + offset) % variants.len();
            let variant = variants[variant_index];
            let start = Instant::now();
            launch_decode_linear_top1_bf16_variant(
                variant,
                stream,
                module,
                input,
                weight,
                partial_tokens,
                partial_logits,
                packed,
            )?;
            stream.synchronize()?;
            samples[variant_index].push(start.elapsed().as_secs_f64());
        }
    }

    for (variant, variant_samples) in variants.iter().copied().zip(samples.iter()) {
        print_decode_interleaved_bench_result(variant.label(), madds_per_launch, variant_samples);
    }

    Ok(())
}

fn launch_decode_linear_top1_bf16_variant(
    variant: DecodeTop1BenchVariant,
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<Bf16>,
    partial_tokens: &mut DeviceBuffer<u32>,
    partial_logits: &mut DeviceBuffer<f32>,
    packed: &mut DeviceBuffer<u64>,
) -> AppResult<()> {
    match variant {
        DecodeTop1BenchVariant::Rows1 => ops::linear_top1_bf16_rows1(
            stream,
            module,
            input,
            weight,
            partial_tokens,
            partial_logits,
            packed,
        )?,
        DecodeTop1BenchVariant::Rows2 => ops::linear_top1_bf16_rows2(
            stream,
            module,
            input,
            weight,
            partial_tokens,
            partial_logits,
            packed,
        )?,
        DecodeTop1BenchVariant::Rows4 => ops::linear_top1_bf16(
            stream,
            module,
            input,
            weight,
            partial_tokens,
            partial_logits,
            packed,
        )?,
        DecodeTop1BenchVariant::Rows8 => ops::linear_top1_bf16_rows8(
            stream,
            module,
            input,
            weight,
            partial_tokens,
            partial_logits,
            packed,
        )?,
    }
    Ok(())
}

fn print_decode_matvec_bench_result(
    label: &str,
    repeat_count: usize,
    madds_per_launch: usize,
    seconds: f64,
) {
    let seconds_per_launch = seconds / repeat_count as f64;
    let gmadds_per_second = (madds_per_launch as f64 * repeat_count as f64) / seconds / 1.0e9;
    println!(
        "  {label}: launches={} madds_per_launch={} seconds={:.6} seconds_per_launch={:.6} giga_madds_per_second={:.3}",
        repeat_count, madds_per_launch, seconds, seconds_per_launch, gmadds_per_second
    );
}

fn print_decode_interleaved_bench_result(
    label: &str,
    madds_per_launch: usize,
    seconds_samples: &[f64],
) {
    let launch_count = seconds_samples.len();
    let total_seconds: f64 = seconds_samples.iter().sum();
    let mean_seconds_per_launch = total_seconds / launch_count as f64;
    let median_seconds_per_launch = median_f64(seconds_samples);
    let best_seconds_per_launch = seconds_samples
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let gmadds_per_second = (madds_per_launch as f64 * launch_count as f64) / total_seconds / 1.0e9;
    println!(
        "  {label}: interleaved_launches={} madds_per_launch={} total_seconds={:.6} mean_seconds_per_launch={:.6} median_seconds_per_launch={:.6} best_seconds_per_launch={:.6} giga_madds_per_second={:.3}",
        launch_count,
        madds_per_launch,
        total_seconds,
        mean_seconds_per_launch,
        median_seconds_per_launch,
        best_seconds_per_launch,
        gmadds_per_second
    );
}

fn run_logit_stress(args: &[String]) -> AppResult<()> {
    let repeat_count = match args {
        [] => 1,
        [value] => value.parse::<usize>().map_err(|error| {
            invalid_input(format!(
                "logit-stress repeat count must be a positive integer, got {value:?}: {error}"
            ))
        })?,
        _ => {
            return Err(invalid_input(
                "logit-stress accepts at most one optional repeat count",
            ));
        }
    };
    if repeat_count == 0 {
        return Err(invalid_input("logit-stress repeat count must be nonzero"));
    }

    let (stream, module) = cuda_handles()?;
    let lengths = [1usize, 2, 3, 17, 255, 256, 257, 4096, 32768, 65537, 131072];

    let start = Instant::now();
    let mut case_count = 0usize;
    let mut element_count = 0usize;

    for repeat in 0..repeat_count {
        for (shape_index, &len) in lengths.iter().enumerate() {
            let seed = ((repeat as u64) << 32) ^ shape_index as u64 ^ 0x6a09_e667_f3bc_c909;
            element_count += run_argmax_packed_stress_case(&stream, &module, len, seed)?;
            case_count += 2;
        }

        let top1_shapes = [
            (1usize, 1usize),
            (3, 5),
            (16, 16),
            (17, 31),
            (255, 64),
            (257, 65),
            (4097, 96),
            (8192, 128),
        ];
        for (shape_index, &(output_dim, input_dim)) in top1_shapes.iter().enumerate() {
            let seed = ((repeat as u64) << 32) ^ shape_index as u64 ^ 0xd1b5_4a32_d192_ed03;
            element_count +=
                run_linear_top1_bf16_stress_case(&stream, &module, input_dim, output_dim, seed)?;
            element_count += run_linear_top1_i8_scaled_stress_case(
                &stream,
                &module,
                input_dim,
                output_dim,
                seed ^ 0x88,
            )?;
            case_count += 1;
        }
    }

    println!(
        "Logit stress passed: cases={} elements={} elapsed_seconds={:.6}",
        case_count,
        element_count,
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn run_attention_stress(args: &[String]) -> AppResult<()> {
    let repeat_count = match args {
        [] => 1,
        [value] => value.parse::<usize>().map_err(|error| {
            invalid_input(format!(
                "attention-stress repeat count must be a positive integer, got {value:?}: {error}"
            ))
        })?,
        _ => {
            return Err(invalid_input(
                "attention-stress accepts at most one optional repeat count",
            ));
        }
    };
    if repeat_count == 0 {
        return Err(invalid_input(
            "attention-stress repeat count must be nonzero",
        ));
    }

    let (stream, module) = cuda_handles()?;
    let shapes = [
        (1usize, 1usize, 16usize, 1usize, 1usize),
        (2, 1, 16, 3, 5),
        (4, 2, 16, 8, 11),
        (8, 2, 32, 17, 19),
        (8, 1, 64, 64, 67),
        (16, 4, 64, 129, 131),
        (32, 8, 128, 257, 263),
    ];

    let start = Instant::now();
    let mut case_count = 0usize;
    let mut element_count = 0usize;
    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;

    for repeat in 0..repeat_count {
        for (shape_index, &(n_heads, n_kv_heads, head_dim, seq_len, max_seq_len)) in
            shapes.iter().enumerate()
        {
            let seed = ((repeat as u64) << 32) ^ shape_index as u64 ^ 0xa5a5_63c1_9e37_11d7;
            let stats = run_single_query_attention_stress_case(
                &stream,
                &module,
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                max_seq_len,
                seed,
            )?;
            case_count += 1;
            element_count += stats.element_count;
            max_abs_diff = max_abs_diff.max(stats.max_abs_diff);
            sum_abs_diff += stats.sum_abs_diff;

            let stats = run_prefill_causal_attention_stress_case(
                &stream,
                &module,
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                max_seq_len,
                seed ^ 0x9df9_7f4c_51a2_67b1,
            )?;
            case_count += 1;
            element_count += stats.element_count;
            max_abs_diff = max_abs_diff.max(stats.max_abs_diff);
            sum_abs_diff += stats.sum_abs_diff;

            let stats = run_incremental_attention_stress_case(
                &stream,
                &module,
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                max_seq_len,
                seed ^ 0x5eed_1a77_2b2b_4311,
            )?;
            case_count += 1;
            element_count += stats.element_count;
            max_abs_diff = max_abs_diff.max(stats.max_abs_diff);
            sum_abs_diff += stats.sum_abs_diff;
        }
    }

    let mean_abs_diff = if element_count == 0 {
        0.0
    } else {
        sum_abs_diff / element_count as f64
    };
    println!(
        "Attention stress passed: cases={} elements={} max_abs_diff={:.8} mean_abs_diff={:.12} elapsed_seconds={:.6}",
        case_count,
        element_count,
        max_abs_diff,
        mean_abs_diff,
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn run_ministral_bf16_prefill_bench(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let max_prompt_len = parse_optional_usize(args, &mut index, 64, "max_prompt_len")?;
    let repeat_count = parse_optional_usize(args, &mut index, 3, "repeat_count")?;
    if index != args.len() {
        return Err(invalid_input(
            "ministral-bf16-prefill-bench accepts at most: [model_dir] [max_prompt_len] [repeat_count]",
        ));
    }
    if max_prompt_len == 0 {
        return Err(invalid_input("max_prompt_len must be nonzero"));
    }
    if repeat_count == 0 {
        return Err(invalid_input("repeat_count must be nonzero"));
    }

    let config = TextConfig::from_model_dir(&model_dir)?;
    if max_prompt_len > config.max_position_embeddings {
        return Err(invalid_input(format!(
            "max_prompt_len {max_prompt_len} exceeds model max_position_embeddings {}",
            config.max_position_embeddings
        )));
    }

    let lengths = prefill_bench_lengths(max_prompt_len);
    let (stream, module) = cuda_handles()?;
    let mut runtime = MinistralTextRuntime::new(stream, module, &model_dir, max_prompt_len)?;
    let memory = runtime.memory_stats();

    println!(
        "Ministral BF16 prefill bench: model_dir={} max_prompt_len={} repeats={} lengths={:?}",
        model_dir.display(),
        max_prompt_len,
        repeat_count,
        lengths
    );
    println!(
        "  memory_bytes weights={} kv_cache={} scratch={} total={}",
        memory.weights_bytes,
        memory.kv_cache_bytes,
        memory.scratch_bytes,
        memory.total_resident_bytes
    );

    for length in lengths {
        let prompt = prefill_bench_prompt(length, config.vocab_size);
        let mut total_seconds = 0.0_f64;
        let mut best_seconds = f64::INFINITY;
        let mut top_token = None;
        let mut top_logit = None;

        for _ in 0..repeat_count {
            runtime.reset_sequence();
            let start = Instant::now();
            runtime.prefill(&prompt)?;
            let elapsed = start.elapsed().as_secs_f64();
            total_seconds += elapsed;
            best_seconds = best_seconds.min(elapsed);

            let top = runtime.next_top_logits(1)?;
            if let Some(&(token, logit)) = top.first() {
                top_token = Some(token);
                top_logit = Some(logit);
            }
        }

        let mean_seconds = total_seconds / repeat_count as f64;
        println!(
            "  len={} mean_prefill_seconds={:.6} best_prefill_seconds={:.6} mean_tokens_per_second={:.3} best_tokens_per_second={:.3} top_token={} top_logit={:.8}",
            length,
            mean_seconds,
            best_seconds,
            length as f64 / mean_seconds,
            length as f64 / best_seconds,
            top_token
                .map(|token| token.to_string())
                .unwrap_or_else(|| "none".to_string()),
            top_logit.unwrap_or(f32::NAN)
        );
    }

    Ok(())
}

fn run_ministral_bf16_decode_bench(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 16, "max_new_tokens")?;
    let repeat_count = parse_optional_usize(args, &mut index, 5, "repeat_count")?;
    let top_k = parse_optional_usize(args, &mut index, 3, "top_k")?.max(1);
    let mut top1_plan = Bf16Top1Plan::default();
    if max_new_tokens == 0 {
        return Err(invalid_input("max_new_tokens must be nonzero"));
    }
    if repeat_count == 0 {
        return Err(invalid_input("repeat_count must be nonzero"));
    }
    let mut prompt_words = Vec::new();
    while index < args.len() {
        match args[index].as_str() {
            "--top1-plan" => {
                let value = parse_required_flag_value(args, &mut index, "--top1-plan")?;
                top1_plan = parse_bf16_top1_plan(value)?;
            }
            value => {
                prompt_words.push(value.to_string());
                index += 1;
            }
        }
    }
    let prompt = if prompt_words.is_empty() {
        "hello".to_string()
    } else {
        prompt_words.join(" ")
    };

    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let prompt_tokens = tokenizer.encode_lossy(&prompt, true)?;
    let max_seq_len = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| invalid_input("ministral decode bench sequence length overflow"))?;
    let config = TextConfig::from_model_dir(&model_dir)?;
    if max_seq_len > config.max_position_embeddings {
        return Err(invalid_input(format!(
            "requested sequence length {max_seq_len} exceeds model max_position_embeddings {}",
            config.max_position_embeddings
        )));
    }

    let stop_token_id = tokenizer.eos_token_id();
    let (stream, module) = cuda_handles()?;
    let mut runtime =
        MinistralTextRuntime::new(stream.clone(), module.clone(), &model_dir, max_seq_len)?;
    runtime.set_top1_plan(top1_plan);
    let memory = runtime.memory_stats();

    println!(
        "Ministral BF16 decode bench: model_dir={} prompt={:?} max_new_tokens={} repeats={} top_k={} top1_plan={}",
        model_dir.display(),
        prompt,
        max_new_tokens,
        repeat_count,
        top_k,
        runtime.top1_plan().label()
    );
    print_token_window("prompt_tokens", &prompt_tokens);
    println!("  eos_token_id={stop_token_id:?}");
    println!(
        "  memory_bytes weights={} kv_cache={} scratch={} total={}",
        memory.weights_bytes,
        memory.kv_cache_bytes,
        memory.scratch_bytes,
        memory.total_resident_bytes
    );

    let mut expected_generated_tokens = None::<Vec<u32>>;
    let mut total_prefill_seconds = 0.0_f64;
    let mut total_decode_seconds = 0.0_f64;
    let mut total_generated_tokens = 0usize;
    let mut best_prefill_seconds = f64::INFINITY;
    let mut best_decode_seconds = f64::INFINITY;
    let mut best_decode_tokens_per_second = 0.0_f64;
    let mut prefill_seconds_samples = Vec::with_capacity(repeat_count);
    let mut decode_seconds_samples = Vec::with_capacity(repeat_count);
    let mut decode_tokens_per_second_samples = Vec::with_capacity(repeat_count);

    for repeat in 0..repeat_count {
        runtime.reset_sequence();

        let prefill_start = Instant::now();
        runtime.prefill(&prompt_tokens)?;
        runtime.synchronize()?;
        let prefill_seconds = prefill_start.elapsed().as_secs_f64();

        let decode_start = Instant::now();
        let steps = runtime.generate_greedy_until(max_new_tokens, top_k, stop_token_id)?;
        runtime.synchronize()?;
        let decode_seconds = decode_start.elapsed().as_secs_f64();

        let generated_tokens = runtime.tokens()[prompt_tokens.len()..].to_vec();
        if generated_tokens.len() != steps.len() {
            return Err(invalid_data(format!(
                "decode bench repeat {repeat} generated {} tokens but returned {} steps",
                generated_tokens.len(),
                steps.len()
            )));
        }
        if let Some(expected) = &expected_generated_tokens {
            if expected != &generated_tokens {
                return Err(invalid_data(format!(
                    "decode bench generated tokens diverged at repeat {repeat}: expected {:?}, got {:?}",
                    expected, generated_tokens
                )));
            }
        } else {
            expected_generated_tokens = Some(generated_tokens.clone());
        }

        let generated_count = generated_tokens.len();
        let decode_tokens_per_second = tokens_per_second(generated_count, decode_seconds);
        prefill_seconds_samples.push(prefill_seconds);
        decode_seconds_samples.push(decode_seconds);
        decode_tokens_per_second_samples.push(decode_tokens_per_second);
        total_prefill_seconds += prefill_seconds;
        total_decode_seconds += decode_seconds;
        total_generated_tokens += generated_count;
        best_prefill_seconds = best_prefill_seconds.min(prefill_seconds);
        best_decode_seconds = best_decode_seconds.min(decode_seconds);
        best_decode_tokens_per_second = best_decode_tokens_per_second.max(decode_tokens_per_second);

        let finish_reason = match generated_tokens.last().copied() {
            Some(token) if Some(token) == stop_token_id => format!("stop-token({token})"),
            _ if generated_count == max_new_tokens => "max-new-tokens".to_string(),
            _ => "early-return-without-stop-token".to_string(),
        };
        println!(
            "  repeat={} generated_tokens={} prefill_seconds={:.6} decode_seconds={:.6} decode_tokens_per_second={:.3} finish_reason={}",
            repeat,
            generated_count,
            prefill_seconds,
            decode_seconds,
            decode_tokens_per_second,
            finish_reason
        );
    }

    let mean_prefill_seconds = total_prefill_seconds / repeat_count as f64;
    let mean_decode_seconds = total_decode_seconds / repeat_count as f64;
    let median_prefill_seconds = median_f64(&prefill_seconds_samples);
    let median_decode_seconds = median_f64(&decode_seconds_samples);
    let median_decode_tokens_per_second = median_f64(&decode_tokens_per_second_samples);
    let aggregate_decode_tokens_per_second =
        tokens_per_second(total_generated_tokens, total_decode_seconds);
    let aggregate_total_tokens_per_second = tokens_per_second(
        total_generated_tokens,
        total_prefill_seconds + total_decode_seconds,
    );
    let generated_tokens = expected_generated_tokens.unwrap_or_default();

    println!("  generated_tokens={generated_tokens:?}");
    println!(
        "  generated_text={:?}",
        tokenizer.decode_lossy(&generated_tokens)?
    );
    println!(
        "  summary mean_prefill_seconds={:.6} median_prefill_seconds={:.6} best_prefill_seconds={:.6} mean_decode_seconds={:.6} median_decode_seconds={:.6} best_decode_seconds={:.6} aggregate_decode_tokens_per_second={:.3} median_decode_tokens_per_second={:.3} best_decode_tokens_per_second={:.3} aggregate_total_tokens_per_second={:.3}",
        mean_prefill_seconds,
        median_prefill_seconds,
        best_prefill_seconds,
        mean_decode_seconds,
        median_decode_seconds,
        best_decode_seconds,
        aggregate_decode_tokens_per_second,
        median_decode_tokens_per_second,
        best_decode_tokens_per_second,
        aggregate_total_tokens_per_second
    );

    Ok(())
}

fn run_ministral_exported_decode_bench(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 16, "max_new_tokens")?;
    let repeat_count = parse_optional_usize(args, &mut index, 5, "repeat_count")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?.max(1);
    if max_new_tokens == 0 {
        return Err(invalid_input("max_new_tokens must be nonzero"));
    }
    if repeat_count == 0 {
        return Err(invalid_input("repeat_count must be nonzero"));
    }
    let prompt = if index < args.len() {
        args[index..].join(" ")
    } else {
        "hello".to_string()
    };

    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let prompt_tokens = tokenizer.encode_lossy(&prompt, true)?;
    let max_seq_len = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| invalid_input("ministral exported decode bench sequence length overflow"))?;
    let config = TextConfig::from_model_dir(&model_dir)?;
    if max_seq_len > config.max_position_embeddings {
        return Err(invalid_input(format!(
            "requested sequence length {max_seq_len} exceeds model max_position_embeddings {}",
            config.max_position_embeddings
        )));
    }

    let stop_token_id = tokenizer.eos_token_id();
    let (stream, module) = cuda_handles()?;
    let mut runtime = MinistralAllLinearInt8Runtime::new_from_export(
        stream.clone(),
        module.clone(),
        &model_dir,
        &export_dir,
        max_seq_len,
    )?;
    let memory = runtime.memory_stats();

    println!(
        "Ministral exported all-linear-int8 decode bench: model_dir={} export_dir={} prompt={:?} max_new_tokens={} repeats={} top_k={}",
        model_dir.display(),
        export_dir.display(),
        prompt,
        max_new_tokens,
        repeat_count,
        top_k
    );
    print_token_window("prompt_tokens", &prompt_tokens);
    println!("  eos_token_id={stop_token_id:?}");
    println!(
        "  memory_bytes weights={} kv_cache={} scratch={} total={}",
        memory.weights_bytes,
        memory.kv_cache_bytes,
        memory.scratch_bytes,
        memory.total_resident_bytes
    );

    let mut expected_generated_tokens = None::<Vec<u32>>;
    let mut total_prefill_seconds = 0.0_f64;
    let mut total_decode_seconds = 0.0_f64;
    let mut total_generated_tokens = 0usize;
    let mut best_prefill_seconds = f64::INFINITY;
    let mut best_decode_seconds = f64::INFINITY;
    let mut best_decode_tokens_per_second = 0.0_f64;
    let mut prefill_seconds_samples = Vec::with_capacity(repeat_count);
    let mut decode_seconds_samples = Vec::with_capacity(repeat_count);
    let mut decode_tokens_per_second_samples = Vec::with_capacity(repeat_count);

    for repeat in 0..repeat_count {
        runtime.reset_sequence();

        let prefill_start = Instant::now();
        runtime.prefill(&prompt_tokens)?;
        runtime.synchronize()?;
        let prefill_seconds = prefill_start.elapsed().as_secs_f64();

        let decode_start = Instant::now();
        let steps = runtime.generate_greedy_until(max_new_tokens, top_k, stop_token_id)?;
        runtime.synchronize()?;
        let decode_seconds = decode_start.elapsed().as_secs_f64();

        let generated_tokens = runtime.tokens()[prompt_tokens.len()..].to_vec();
        if generated_tokens.len() != steps.len() {
            return Err(invalid_data(format!(
                "exported decode bench repeat {repeat} generated {} tokens but returned {} steps",
                generated_tokens.len(),
                steps.len()
            )));
        }
        if let Some(expected) = &expected_generated_tokens {
            if expected != &generated_tokens {
                return Err(invalid_data(format!(
                    "exported decode bench generated tokens diverged at repeat {repeat}: expected {:?}, got {:?}",
                    expected, generated_tokens
                )));
            }
        } else {
            expected_generated_tokens = Some(generated_tokens.clone());
        }

        let generated_count = generated_tokens.len();
        let decode_tokens_per_second = tokens_per_second(generated_count, decode_seconds);
        prefill_seconds_samples.push(prefill_seconds);
        decode_seconds_samples.push(decode_seconds);
        decode_tokens_per_second_samples.push(decode_tokens_per_second);
        total_prefill_seconds += prefill_seconds;
        total_decode_seconds += decode_seconds;
        total_generated_tokens += generated_count;
        best_prefill_seconds = best_prefill_seconds.min(prefill_seconds);
        best_decode_seconds = best_decode_seconds.min(decode_seconds);
        best_decode_tokens_per_second = best_decode_tokens_per_second.max(decode_tokens_per_second);

        let finish_reason = match generated_tokens.last().copied() {
            Some(token) if Some(token) == stop_token_id => format!("stop-token({token})"),
            _ if generated_count == max_new_tokens => "max-new-tokens".to_string(),
            _ => "early-return-without-stop-token".to_string(),
        };
        println!(
            "  repeat={} generated_tokens={} prefill_seconds={:.6} decode_seconds={:.6} decode_tokens_per_second={:.3} finish_reason={}",
            repeat,
            generated_count,
            prefill_seconds,
            decode_seconds,
            decode_tokens_per_second,
            finish_reason
        );
    }

    let mean_prefill_seconds = total_prefill_seconds / repeat_count as f64;
    let mean_decode_seconds = total_decode_seconds / repeat_count as f64;
    let median_prefill_seconds = median_f64(&prefill_seconds_samples);
    let median_decode_seconds = median_f64(&decode_seconds_samples);
    let median_decode_tokens_per_second = median_f64(&decode_tokens_per_second_samples);
    let aggregate_decode_tokens_per_second =
        tokens_per_second(total_generated_tokens, total_decode_seconds);
    let aggregate_total_tokens_per_second = tokens_per_second(
        total_generated_tokens,
        total_prefill_seconds + total_decode_seconds,
    );
    let generated_tokens = expected_generated_tokens.unwrap_or_default();

    println!("  generated_tokens={generated_tokens:?}");
    println!(
        "  generated_text={:?}",
        tokenizer.decode_lossy(&generated_tokens)?
    );
    println!(
        "  summary mean_prefill_seconds={:.6} median_prefill_seconds={:.6} best_prefill_seconds={:.6} mean_decode_seconds={:.6} median_decode_seconds={:.6} best_decode_seconds={:.6} aggregate_decode_tokens_per_second={:.3} median_decode_tokens_per_second={:.3} best_decode_tokens_per_second={:.3} aggregate_total_tokens_per_second={:.3}",
        mean_prefill_seconds,
        median_prefill_seconds,
        best_prefill_seconds,
        mean_decode_seconds,
        median_decode_seconds,
        best_decode_seconds,
        aggregate_decode_tokens_per_second,
        median_decode_tokens_per_second,
        best_decode_tokens_per_second,
        aggregate_total_tokens_per_second
    );

    Ok(())
}

fn run_ministral_exported_prefill_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let top_k = parse_optional_usize(args, &mut index, 3, "top_k")?.max(1);
    let cli = parse_chat_cli(args, index)?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let (stream, module) = cuda_handles()?;

    println!(
        "Ministral exported all-linear-int8 prefill compare: model_dir={} export_dir={} prompts_len={} top_k={}",
        model_dir.display(),
        export_dir.display(),
        cli.prompts.len(),
        top_k
    );

    let mut token_match_count = 0usize;
    let mut top_tokens_match = true;
    let mut kl_sum = 0.0_f64;
    let mut max_kl = 0.0_f64;
    let mut max_abs_diff = 0.0_f32;

    for (prompt_index, prompt) in cli.prompts.iter().enumerate() {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let max_seq_len = prompt_tokens.len().max(1);
        let mut reference_runtime = MinistralAllLinearInt8Runtime::new_from_export(
            stream.clone(),
            module.clone(),
            &model_dir,
            &export_dir,
            max_seq_len,
        )?;
        let mut batched_runtime = MinistralAllLinearInt8Runtime::new_from_export(
            stream.clone(),
            module.clone(),
            &model_dir,
            &export_dir,
            max_seq_len,
        )?;

        reference_runtime.prefill_token_loop_reference(&prompt_tokens)?;
        batched_runtime.prefill(&prompt_tokens)?;
        reference_runtime.synchronize()?;
        batched_runtime.synchronize()?;

        let reference_logits = reference_runtime.next_logits_to_host()?;
        let batched_logits = batched_runtime.next_logits_to_host()?;
        let kl = logits_kl_divergence(&reference_logits, &batched_logits)?;
        let prompt_max_abs_diff = logits_max_abs_diff(&reference_logits, &batched_logits)?;
        let mean_abs_diff = logits_mean_abs_diff(&reference_logits, &batched_logits)?;
        let reference_top = logits_top_k(&reference_logits, top_k);
        let batched_top = logits_top_k(&batched_logits, top_k);
        let prompt_top_tokens_match = reference_top.first().map(|(token, _)| token)
            == batched_top.first().map(|(token, _)| token);
        if prompt_top_tokens_match {
            token_match_count += 1;
        }
        top_tokens_match &= prompt_top_tokens_match;
        kl_sum += kl;
        max_kl = max_kl.max(kl);
        max_abs_diff = max_abs_diff.max(prompt_max_abs_diff);

        println!("  prompt[{prompt_index}]={prompt:?}");
        print_token_window("prompt_tokens", &prompt_tokens);
        println!(
            "    top_tokens_match={} kl_divergence={:.12} max_abs_diff={:.8} mean_abs_diff={:.8}",
            prompt_top_tokens_match, kl, prompt_max_abs_diff, mean_abs_diff
        );
        println!("    token_loop_top_logits={reference_top:?}");
        println!("    batched_top_logits={batched_top:?}");
    }

    let mean_kl = if cli.prompts.is_empty() {
        0.0
    } else {
        kl_sum / cli.prompts.len() as f64
    };
    println!(
        "  summary token_match_count={}/{} top_tokens_match={} mean_kl_divergence={:.12} max_kl_divergence={:.12} max_abs_diff={:.8}",
        token_match_count,
        cli.prompts.len(),
        top_tokens_match,
        mean_kl,
        max_kl,
        max_abs_diff
    );

    Ok(())
}

fn run_ministral_bf16_prefill_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let max_prompt_len = parse_optional_usize(args, &mut index, 16, "max_prompt_len")?;
    let top_k = parse_optional_usize(args, &mut index, 3, "top_k")?;
    if index != args.len() {
        return Err(invalid_input(
            "ministral-bf16-prefill-compare accepts at most: [model_dir] [max_prompt_len] [top_k]",
        ));
    }
    if max_prompt_len == 0 {
        return Err(invalid_input("max_prompt_len must be nonzero"));
    }
    if top_k == 0 {
        return Err(invalid_input("top_k must be nonzero"));
    }

    let config = TextConfig::from_model_dir(&model_dir)?;
    if max_prompt_len > config.max_position_embeddings {
        return Err(invalid_input(format!(
            "max_prompt_len {max_prompt_len} exceeds model max_position_embeddings {}",
            config.max_position_embeddings
        )));
    }

    let lengths = prefill_bench_lengths(max_prompt_len);
    let (stream, module) = cuda_handles()?;
    let mut runtime = MinistralTextRuntime::new(stream, module, &model_dir, max_prompt_len)?;

    println!(
        "Ministral BF16 prefill compare: model_dir={} max_prompt_len={} top_k={} lengths={:?}",
        model_dir.display(),
        max_prompt_len,
        top_k,
        lengths
    );

    let mut all_top_tokens_match = true;
    let mut max_observed_abs_diff = 0.0_f32;
    let mut max_observed_kl = 0.0_f64;

    for length in lengths {
        let prompt = prefill_bench_prompt(length, config.vocab_size);

        runtime.reset_sequence();
        let reference_start = Instant::now();
        runtime.prefill_token_loop_reference(&prompt)?;
        let reference_seconds = reference_start.elapsed().as_secs_f64();
        let reference_logits = runtime.next_logits_to_host()?;

        runtime.reset_sequence();
        let candidate_start = Instant::now();
        runtime.prefill(&prompt)?;
        let candidate_seconds = candidate_start.elapsed().as_secs_f64();
        let candidate_logits = runtime.next_logits_to_host()?;

        let max_abs_diff = logits_max_abs_diff(&reference_logits, &candidate_logits)?;
        let mean_abs_diff = logits_mean_abs_diff(&reference_logits, &candidate_logits)?;
        let kl = logits_kl_divergence(&reference_logits, &candidate_logits)?;
        let reference_top = logits_top_k(&reference_logits, top_k);
        let candidate_top = logits_top_k(&candidate_logits, top_k);
        let top_tokens_match = reference_top.first().map(|(token, _)| token)
            == candidate_top.first().map(|(token, _)| token);

        all_top_tokens_match &= top_tokens_match;
        max_observed_abs_diff = max_observed_abs_diff.max(max_abs_diff);
        max_observed_kl = max_observed_kl.max(kl);

        println!(
            "  len={} top_tokens_match={} reference_seconds={:.6} candidate_seconds={:.6} speedup={:.3} kl_divergence={:.12} max_abs_diff={:.8} mean_abs_diff={:.8}",
            length,
            top_tokens_match,
            reference_seconds,
            candidate_seconds,
            reference_seconds / candidate_seconds,
            kl,
            max_abs_diff,
            mean_abs_diff
        );
        println!("    reference_top_logits={:?}", reference_top);
        println!("    candidate_top_logits={:?}", candidate_top);
    }

    println!(
        "  summary top_tokens_match={} max_kl_divergence={:.12} max_abs_diff={:.8}",
        all_top_tokens_match, max_observed_kl, max_observed_abs_diff
    );
    if !all_top_tokens_match {
        return Err(invalid_data(
            "batched BF16 prefill top token diverged from token-loop reference",
        ));
    }
    if max_observed_kl > 1.0e-4 {
        return Err(invalid_data(format!(
            "batched BF16 prefill KL divergence {max_observed_kl:.12} exceeded 1e-4"
        )));
    }
    if max_observed_abs_diff > 1.0e-2 {
        return Err(invalid_data(format!(
            "batched BF16 prefill max_abs_diff {max_observed_abs_diff:.8} exceeded 1e-2"
        )));
    }

    Ok(())
}

fn run_ministral_stop_smoke(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 4, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 3, "top_k")?.max(1);
    if max_new_tokens == 0 {
        return Err(invalid_input(
            "ministral stop smoke requires max_new_tokens greater than zero",
        ));
    }
    let prompt = if index < args.len() {
        args[index..].join(" ")
    } else {
        "hello".to_string()
    };

    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let prompt_tokens = tokenizer.encode_lossy(&prompt, true)?;
    let max_seq_len = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| invalid_input("ministral stop smoke sequence length overflow"))?;
    let (stream, module) = cuda_handles()?;
    let mut runtime =
        MinistralTextRuntime::new(stream.clone(), module.clone(), &model_dir, max_seq_len)?;

    runtime.prefill(&prompt_tokens)?;
    runtime.synchronize()?;
    let initial_top_logits = runtime.next_top_logits(top_k)?;
    let &(stop_token_id, stop_logit) = initial_top_logits
        .first()
        .ok_or_else(|| invalid_data("ministral stop smoke produced no next-token logits"))?;
    let eos_token_id = tokenizer.eos_token_id();

    let steps = runtime.generate_greedy_until(max_new_tokens, top_k, Some(stop_token_id))?;
    runtime.synchronize()?;
    if steps.len() != 1 {
        return Err(invalid_data(format!(
            "ministral stop smoke expected exactly one step after forced stop, got {}",
            steps.len()
        )));
    }
    if steps[0].token_id != stop_token_id {
        return Err(invalid_data(format!(
            "ministral stop smoke expected stop token {stop_token_id}, got {}",
            steps[0].token_id
        )));
    }

    let eos_stop_steps_len = if let Some(eos_token_id) = eos_token_id {
        let mut eos_runtime = MinistralTextRuntime::new(stream, module, &model_dir, max_seq_len)?;
        eos_runtime.prefill(&prompt_tokens)?;
        eos_runtime.synchronize()?;
        let eos_steps = eos_runtime.generate_greedy_until_from_top1(
            max_new_tokens,
            Some(eos_token_id),
            (eos_token_id, 0.0),
        )?;
        eos_runtime.synchronize()?;
        if eos_steps.len() != 1 {
            return Err(invalid_data(format!(
                "ministral stop smoke expected exactly one step after forced EOS stop, got {}",
                eos_steps.len()
            )));
        }
        if eos_steps[0].token_id != eos_token_id {
            return Err(invalid_data(format!(
                "ministral stop smoke expected EOS stop token {eos_token_id}, got {}",
                eos_steps[0].token_id
            )));
        }
        Some(eos_steps.len())
    } else {
        None
    };

    println!(
        "Ministral stop smoke: model_dir={} prompt={:?} max_new_tokens={} top_k={}",
        model_dir.display(),
        prompt,
        max_new_tokens,
        top_k
    );
    print_token_window("prompt_tokens", &prompt_tokens);
    println!("  eos_token_id={eos_token_id:?}");
    println!("  forced_stop_token_id={stop_token_id}");
    println!("  forced_stop_logit={stop_logit:.8}");
    println!(
        "  generated_tokens={:?}",
        &runtime.tokens()[prompt_tokens.len()..]
    );
    println!(
        "  generated_text={:?}",
        tokenizer.decode_lossy(&runtime.tokens()[prompt_tokens.len()..])?
    );
    println!("  steps_len={}", steps.len());
    println!("  stop_condition_observed=true");
    println!("  eos_stop_steps_len={eos_stop_steps_len:?}");
    println!(
        "  eos_stop_condition_observed={}",
        eos_stop_steps_len == Some(1)
    );

    Ok(())
}

#[derive(Debug)]
struct Bf16GenerationTrace {
    prefill_seconds: f64,
    decode_seconds: f64,
    generated_tokens: Vec<u32>,
    all_tokens: Vec<u32>,
    steps: Vec<Bf16GenerationTraceStep>,
    final_logits: Vec<f32>,
}

#[derive(Debug)]
struct Bf16GenerationTraceStep {
    step: usize,
    token_id: u32,
    logit: f32,
    top_logits: Vec<(u32, f32)>,
    logits: Vec<f32>,
}

#[derive(Debug)]
struct Bf16FastTop1GenerationTrace {
    prefill_seconds: f64,
    decode_seconds: f64,
    generated_tokens: Vec<u32>,
    all_tokens: Vec<u32>,
    steps: Vec<GreedyGenerationStep>,
    final_logits: Vec<f32>,
}

fn run_ministral_bf16_generation_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 4, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 3, "top_k")?.max(1);
    if max_new_tokens == 0 {
        return Err(invalid_input(
            "ministral-bf16-generation-compare requires max_new_tokens greater than zero",
        ));
    }
    let prompt = if index < args.len() {
        args[index..].join(" ")
    } else {
        "hello".to_string()
    };

    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let prompt_tokens = tokenizer.encode_lossy(&prompt, true)?;
    let max_seq_len = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| invalid_input("ministral generation comparison sequence length overflow"))?;
    let config = TextConfig::from_model_dir(&model_dir)?;
    if max_seq_len > config.max_position_embeddings {
        return Err(invalid_input(format!(
            "requested sequence length {max_seq_len} exceeds model max_position_embeddings {}",
            config.max_position_embeddings
        )));
    }

    let stop_token_id = tokenizer.eos_token_id();
    let (stream, module) = cuda_handles()?;
    let mut runtime = MinistralTextRuntime::new(stream, module, &model_dir, max_seq_len)?;

    let reference = trace_ministral_bf16_generation(
        &mut runtime,
        &prompt_tokens,
        max_new_tokens,
        top_k,
        stop_token_id,
        true,
    )?;
    runtime.reset_sequence();
    let candidate = trace_ministral_bf16_generation(
        &mut runtime,
        &prompt_tokens,
        max_new_tokens,
        top_k,
        stop_token_id,
        false,
    )?;
    let fast_top1_candidate = if top_k == 1 {
        runtime.reset_sequence();
        Some(trace_ministral_bf16_fast_top1_generation(
            &mut runtime,
            &prompt_tokens,
            max_new_tokens,
            stop_token_id,
        )?)
    } else {
        None
    };

    let token_ids_match = reference.generated_tokens == candidate.generated_tokens;
    let step_count_match = reference.steps.len() == candidate.steps.len();
    let mut max_observed_abs_diff = 0.0_f32;
    let mut max_observed_mean_abs_diff = 0.0_f32;
    let mut max_observed_kl = 0.0_f64;
    let mut all_top_tokens_match = true;

    println!(
        "Ministral BF16 generation compare: model_dir={} prompt={:?} max_new_tokens={} top_k={}",
        model_dir.display(),
        prompt,
        max_new_tokens,
        top_k
    );
    print_token_window("prompt_tokens", &prompt_tokens);
    println!("  eos_token_id={stop_token_id:?}");
    println!(
        "  reference_prefill=token-loop candidate_prefill=batched-gemm reference_prefill_seconds={:.6} candidate_prefill_seconds={:.6} reference_decode_seconds={:.6} candidate_decode_seconds={:.6}",
        reference.prefill_seconds,
        candidate.prefill_seconds,
        reference.decode_seconds,
        candidate.decode_seconds
    );
    println!(
        "  reference_generated_tokens={:?}",
        reference.generated_tokens
    );
    println!(
        "  candidate_generated_tokens={:?}",
        candidate.generated_tokens
    );
    println!(
        "  reference_generated_text={:?}",
        tokenizer.decode_lossy(&reference.generated_tokens)?
    );
    println!(
        "  candidate_generated_text={:?}",
        tokenizer.decode_lossy(&candidate.generated_tokens)?
    );
    if let Some(fast) = &fast_top1_candidate {
        println!(
            "  fast_top1_prefill=batched-gemm fast_top1_decode=projection-top1 fast_top1_prefill_seconds={:.6} fast_top1_decode_seconds={:.6}",
            fast.prefill_seconds, fast.decode_seconds
        );
        println!("  fast_top1_generated_tokens={:?}", fast.generated_tokens);
        println!(
            "  fast_top1_generated_text={:?}",
            tokenizer.decode_lossy(&fast.generated_tokens)?
        );
        println!(
            "  fast_top1_all_text_skip_special={:?}",
            tokenizer.decode_lossy(&fast.all_tokens)?
        );
    }
    println!(
        "  reference_all_text_skip_special={:?}",
        tokenizer.decode_lossy(&reference.all_tokens)?
    );
    println!(
        "  candidate_all_text_skip_special={:?}",
        tokenizer.decode_lossy(&candidate.all_tokens)?
    );

    for (reference_step, candidate_step) in reference.steps.iter().zip(candidate.steps.iter()) {
        let max_abs_diff = logits_max_abs_diff(&reference_step.logits, &candidate_step.logits)?;
        let mean_abs_diff = logits_mean_abs_diff(&reference_step.logits, &candidate_step.logits)?;
        let kl = logits_kl_divergence(&reference_step.logits, &candidate_step.logits)?;
        let top_tokens_match = reference_step.top_logits.first().map(|(token, _)| token)
            == candidate_step.top_logits.first().map(|(token, _)| token);
        all_top_tokens_match &= top_tokens_match;
        max_observed_abs_diff = max_observed_abs_diff.max(max_abs_diff);
        max_observed_mean_abs_diff = max_observed_mean_abs_diff.max(mean_abs_diff);
        max_observed_kl = max_observed_kl.max(kl);

        println!(
            "  step={} reference_token={} candidate_token={} token_matches={} reference_logit={:.8} candidate_logit={:.8} top_tokens_match={} kl_divergence={:.12} max_abs_diff={:.8} mean_abs_diff={:.8}",
            reference_step.step,
            reference_step.token_id,
            candidate_step.token_id,
            reference_step.token_id == candidate_step.token_id,
            reference_step.logit,
            candidate_step.logit,
            top_tokens_match,
            kl,
            max_abs_diff,
            mean_abs_diff
        );
        println!("    reference_top_logits={:?}", reference_step.top_logits);
        println!("    candidate_top_logits={:?}", candidate_step.top_logits);
    }

    println!(
        "  summary step_count_match={} token_ids_match={} top_tokens_match={} max_kl_divergence={:.12} max_abs_diff={:.8} max_mean_abs_diff={:.8}",
        step_count_match,
        token_ids_match,
        all_top_tokens_match,
        max_observed_kl,
        max_observed_abs_diff,
        max_observed_mean_abs_diff
    );

    if let Some(fast) = &fast_top1_candidate {
        let fast_step_count_match = candidate.steps.len() == fast.steps.len();
        let fast_token_ids_match = candidate.generated_tokens == fast.generated_tokens;
        let mut max_fast_top1_logit_diff = 0.0_f32;
        let mut fast_top_tokens_match = true;
        for (candidate_step, fast_step) in candidate.steps.iter().zip(fast.steps.iter()) {
            fast_top_tokens_match &= candidate_step.token_id == fast_step.token_id;
            max_fast_top1_logit_diff =
                max_fast_top1_logit_diff.max((candidate_step.logit - fast_step.logit).abs());
        }
        let fast_final_abs_diff = logits_max_abs_diff(&candidate.final_logits, &fast.final_logits)?;
        let fast_final_mean_abs_diff =
            logits_mean_abs_diff(&candidate.final_logits, &fast.final_logits)?;
        let fast_final_kl = logits_kl_divergence(&candidate.final_logits, &fast.final_logits)?;
        println!(
            "  fast_top1_summary step_count_match={} token_ids_match={} top_tokens_match={} max_top1_logit_diff={:.8} final_kl_divergence={:.12} final_max_abs_diff={:.8} final_mean_abs_diff={:.8}",
            fast_step_count_match,
            fast_token_ids_match,
            fast_top_tokens_match,
            max_fast_top1_logit_diff,
            fast_final_kl,
            fast_final_abs_diff,
            fast_final_mean_abs_diff
        );
        if !fast_step_count_match {
            return Err(invalid_data(format!(
                "BF16 fast top1 step count diverged: reference={} fast={}",
                candidate.steps.len(),
                fast.steps.len()
            )));
        }
        if !fast_token_ids_match {
            return Err(invalid_data("BF16 fast top1 token ids diverged"));
        }
        if !fast_top_tokens_match {
            return Err(invalid_data("BF16 fast top1 top tokens diverged"));
        }
        if max_fast_top1_logit_diff > 1.0e-2 {
            return Err(invalid_data(format!(
                "BF16 fast top1 logit diff {max_fast_top1_logit_diff:.8} exceeded 1e-2"
            )));
        }
        if fast_final_kl > 1.0e-4 {
            return Err(invalid_data(format!(
                "BF16 fast top1 final KL divergence {fast_final_kl:.12} exceeded 1e-4"
            )));
        }
        if fast_final_abs_diff > 1.0e-2 {
            return Err(invalid_data(format!(
                "BF16 fast top1 final max_abs_diff {fast_final_abs_diff:.8} exceeded 1e-2"
            )));
        }
    }

    if !step_count_match {
        return Err(invalid_data(format!(
            "BF16 generation comparison step count diverged: reference={} candidate={}",
            reference.steps.len(),
            candidate.steps.len()
        )));
    }
    if !token_ids_match {
        return Err(invalid_data(
            "BF16 generation comparison token ids diverged",
        ));
    }
    if !all_top_tokens_match {
        return Err(invalid_data(
            "BF16 generation comparison top tokens diverged",
        ));
    }
    if max_observed_kl > 1.0e-4 {
        return Err(invalid_data(format!(
            "BF16 generation comparison KL divergence {max_observed_kl:.12} exceeded 1e-4"
        )));
    }
    if max_observed_abs_diff > 1.0e-2 {
        return Err(invalid_data(format!(
            "BF16 generation comparison max_abs_diff {max_observed_abs_diff:.8} exceeded 1e-2"
        )));
    }

    Ok(())
}

fn trace_ministral_bf16_generation(
    runtime: &mut MinistralTextRuntime,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
    token_loop_prefill: bool,
) -> AppResult<Bf16GenerationTrace> {
    let prefill_start = Instant::now();
    if token_loop_prefill {
        runtime.prefill_token_loop_reference(prompt_tokens)?;
    } else {
        runtime.prefill(prompt_tokens)?;
    }
    runtime.synchronize()?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    let decode_start = Instant::now();
    let mut steps = Vec::with_capacity(max_new_tokens);
    for step in 0..max_new_tokens {
        let logits = runtime.next_logits_to_host()?;
        let top_logits = runtime.next_top_logits(top_k)?;
        let &(token_id, logit) = top_logits
            .first()
            .ok_or_else(|| invalid_data("BF16 generation trace produced no top logits"))?;
        runtime.advance_with_token(token_id)?;
        steps.push(Bf16GenerationTraceStep {
            step,
            token_id,
            logit,
            top_logits,
            logits,
        });
        if stop_token_id == Some(token_id) {
            break;
        }
    }
    runtime.synchronize()?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let final_logits = runtime.next_logits_to_host()?;
    let all_tokens = runtime.tokens().to_vec();
    let generated_tokens = all_tokens[prompt_tokens.len()..].to_vec();

    Ok(Bf16GenerationTrace {
        prefill_seconds,
        decode_seconds,
        generated_tokens,
        all_tokens,
        steps,
        final_logits,
    })
}

fn trace_ministral_bf16_fast_top1_generation(
    runtime: &mut MinistralTextRuntime,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    stop_token_id: Option<u32>,
) -> AppResult<Bf16FastTop1GenerationTrace> {
    let prefill_start = Instant::now();
    runtime.prefill(prompt_tokens)?;
    runtime.synchronize()?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    let decode_start = Instant::now();
    let steps = runtime.generate_greedy_until(max_new_tokens, 1, stop_token_id)?;
    runtime.synchronize()?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let final_logits = runtime.next_logits_to_host()?;
    let all_tokens = runtime.tokens().to_vec();
    let generated_tokens = all_tokens[prompt_tokens.len()..].to_vec();

    Ok(Bf16FastTop1GenerationTrace {
        prefill_seconds,
        decode_seconds,
        generated_tokens,
        all_tokens,
        steps,
        final_logits,
    })
}

fn prefill_bench_lengths(max_prompt_len: usize) -> Vec<usize> {
    let mut lengths = Vec::new();
    for candidate in [1usize, 3, 8, 16, 32, 64, 128, 256, 512] {
        if candidate <= max_prompt_len {
            lengths.push(candidate);
        }
    }
    if lengths.last().copied() != Some(max_prompt_len) {
        lengths.push(max_prompt_len);
    }
    lengths
}

fn prefill_bench_prompt(len: usize, vocab_size: usize) -> Vec<u32> {
    let usable_vocab = vocab_size.saturating_sub(1).max(1);
    (0..len).map(|i| 1 + (i % usable_vocab) as u32).collect()
}

fn logits_max_abs_diff(reference: &[f32], candidate: &[f32]) -> AppResult<f32> {
    if reference.len() != candidate.len() {
        return Err(invalid_data(format!(
            "logit length mismatch: reference={} candidate={}",
            reference.len(),
            candidate.len()
        )));
    }

    Ok(reference
        .iter()
        .zip(candidate.iter())
        .map(|(reference, candidate)| (reference - candidate).abs())
        .fold(0.0_f32, f32::max))
}

fn logits_mean_abs_diff(reference: &[f32], candidate: &[f32]) -> AppResult<f32> {
    if reference.len() != candidate.len() {
        return Err(invalid_data(format!(
            "logit length mismatch: reference={} candidate={}",
            reference.len(),
            candidate.len()
        )));
    }
    if reference.is_empty() {
        return Ok(0.0);
    }

    let sum = reference
        .iter()
        .zip(candidate.iter())
        .map(|(reference, candidate)| (reference - candidate).abs() as f64)
        .sum::<f64>();
    Ok((sum / reference.len() as f64) as f32)
}

fn logits_kl_divergence(reference: &[f32], candidate: &[f32]) -> AppResult<f64> {
    if reference.len() != candidate.len() {
        return Err(invalid_data(format!(
            "logit length mismatch: reference={} candidate={}",
            reference.len(),
            candidate.len()
        )));
    }

    let reference_logsumexp = logits_logsumexp(reference);
    let candidate_logsumexp = logits_logsumexp(candidate);
    let mut kl = 0.0_f64;
    for (&reference, &candidate) in reference.iter().zip(candidate.iter()) {
        let reference_logprob = reference as f64 - reference_logsumexp;
        let candidate_logprob = candidate as f64 - candidate_logsumexp;
        let probability = reference_logprob.exp();
        kl += probability * (reference_logprob - candidate_logprob);
    }
    Ok(kl.max(0.0))
}

fn logits_logsumexp(logits: &[f32]) -> f64 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    if !max_logit.is_finite() {
        return max_logit;
    }

    let sum = logits
        .iter()
        .map(|&logit| (logit as f64 - max_logit).exp())
        .sum::<f64>();
    max_logit + sum.ln()
}

fn logits_top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut top = Vec::<(u32, f32)>::new();
    for (token_id, &logit) in logits.iter().enumerate() {
        let token_id = token_id as u32;
        let insert_at = top
            .iter()
            .position(|&(_, current)| logit > current)
            .unwrap_or(top.len());
        if insert_at < k {
            top.insert(insert_at, (token_id, logit));
            if top.len() > k {
                top.pop();
            }
        }
    }
    top
}

fn accumulate_gemm_stats(
    stats: GemmStressStats,
    case_count: &mut usize,
    element_count: &mut usize,
    max_abs_diff: &mut f32,
    sum_abs_diff: &mut f64,
) {
    *case_count += 1;
    *element_count += stats.element_count;
    *max_abs_diff = max_abs_diff.max(stats.max_abs_diff);
    *sum_abs_diff += stats.sum_abs_diff;
}

fn tokens_per_second(token_count: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        token_count as f64 / seconds
    } else {
        f64::INFINITY
    }
}

fn median_f64(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "median requires at least one value");
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn run_single_query_attention_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    max_seq_len: usize,
    seed: u64,
) -> AppResult<GemmStressStats> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(invalid_input(format!(
            "invalid attention stress heads: n_heads={n_heads} n_kv_heads={n_kv_heads}"
        )));
    }
    if seq_len == 0 || seq_len > max_seq_len {
        return Err(invalid_input(format!(
            "invalid attention stress sequence shape: seq_len={seq_len} max_seq_len={max_seq_len}"
        )));
    }
    if seq_len > ops::SINGLE_QUERY_ATTENTION_MAX_SEQ {
        return Err(invalid_input(format!(
            "attention stress seq_len {seq_len} exceeds fused kernel capacity {}",
            ops::SINGLE_QUERY_ATTENTION_MAX_SEQ
        )));
    }

    let q_len = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid_input("attention stress query shape overflow"))?;
    let kv_token_len = n_kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid_input("attention stress KV token shape overflow"))?;
    let kv_cache_len = max_seq_len
        .checked_mul(kv_token_len)
        .ok_or_else(|| invalid_input("attention stress KV cache shape overflow"))?;
    let scores_len = n_heads
        .checked_mul(max_seq_len)
        .ok_or_else(|| invalid_input("attention stress score shape overflow"))?;

    let mut seed = seed;
    let mut query = vec![0.0_f32; q_len];
    let mut key_cache = vec![0.0_f32; kv_cache_len];
    let mut value_cache = vec![0.0_f32; kv_cache_len];
    fill_stress_slice(&mut query, &mut seed, 0.25);
    fill_stress_slice(&mut key_cache[..seq_len * kv_token_len], &mut seed, 0.25);
    fill_stress_slice(&mut value_cache[..seq_len * kv_token_len], &mut seed, 0.25);

    let dev_query = DeviceBuffer::from_host(stream, &query)?;
    let dev_key_cache = DeviceBuffer::from_host(stream, &key_cache)?;
    let dev_value_cache = DeviceBuffer::from_host(stream, &value_cache)?;
    let mut dev_scores = DeviceBuffer::<f32>::zeroed(stream, scores_len)?;
    let mut dev_split_out = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut dev_fused_out = DeviceBuffer::<f32>::zeroed(stream, q_len)?;

    ops::attention_scores(
        stream,
        module,
        &dev_query,
        &dev_key_cache,
        seq_len,
        max_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        &mut dev_scores,
    )?;
    ops::softmax_value(
        stream,
        module,
        &dev_scores,
        &dev_value_cache,
        seq_len,
        max_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        &mut dev_split_out,
    )?;
    ops::single_query_attention(
        stream,
        module,
        &dev_query,
        &dev_key_cache,
        &dev_value_cache,
        seq_len,
        max_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        &mut dev_fused_out,
    )?;

    let split = dev_split_out.to_host_vec(stream)?;
    let fused = dev_fused_out.to_host_vec(stream)?;
    compare_attention_output(
        &split,
        &fused,
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        max_seq_len,
    )
}

fn run_prefill_causal_attention_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    max_seq_len: usize,
    seed: u64,
) -> AppResult<GemmStressStats> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(invalid_input(format!(
            "invalid prefill attention stress heads: n_heads={n_heads} n_kv_heads={n_kv_heads}"
        )));
    }
    if seq_len == 0 || seq_len > max_seq_len {
        return Err(invalid_input(format!(
            "invalid prefill attention stress sequence shape: seq_len={seq_len} max_seq_len={max_seq_len}"
        )));
    }
    if seq_len > ops::SINGLE_QUERY_ATTENTION_MAX_SEQ {
        return Err(invalid_input(format!(
            "prefill attention stress seq_len {seq_len} exceeds fused kernel capacity {}",
            ops::SINGLE_QUERY_ATTENTION_MAX_SEQ
        )));
    }

    let q_len = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid_input("prefill attention stress query shape overflow"))?;
    let kv_token_len = n_kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid_input("prefill attention stress KV token shape overflow"))?;
    let query_batch_len = seq_len
        .checked_mul(q_len)
        .ok_or_else(|| invalid_input("prefill attention stress query batch shape overflow"))?;
    let kv_cache_len = max_seq_len
        .checked_mul(kv_token_len)
        .ok_or_else(|| invalid_input("prefill attention stress KV cache shape overflow"))?;
    let scores_len = n_heads
        .checked_mul(max_seq_len)
        .ok_or_else(|| invalid_input("prefill attention stress score shape overflow"))?;

    let mut seed = seed;
    let mut query_batch = vec![0.0_f32; query_batch_len];
    let mut key_cache = vec![0.0_f32; kv_cache_len];
    let mut value_cache = vec![0.0_f32; kv_cache_len];
    fill_stress_slice(&mut query_batch, &mut seed, 0.25);
    fill_stress_slice(&mut key_cache[..seq_len * kv_token_len], &mut seed, 0.25);
    fill_stress_slice(&mut value_cache[..seq_len * kv_token_len], &mut seed, 0.25);

    let dev_query_batch = DeviceBuffer::from_host(stream, &query_batch)?;
    let dev_key_cache = DeviceBuffer::from_host(stream, &key_cache)?;
    let dev_value_cache = DeviceBuffer::from_host(stream, &value_cache)?;
    let mut dev_scores = DeviceBuffer::<f32>::zeroed(stream, scores_len)?;
    let mut dev_split_out = DeviceBuffer::<f32>::zeroed(stream, query_batch_len)?;
    let mut dev_fused_out = DeviceBuffer::<f32>::zeroed(stream, query_batch_len)?;

    for position in 0..seq_len {
        ops::attention_scores_from_matrix_row(
            stream,
            module,
            &dev_query_batch,
            &dev_key_cache,
            position,
            position + 1,
            max_seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            &mut dev_scores,
        )?;
        ops::softmax_value_to_matrix_row(
            stream,
            module,
            &dev_scores,
            &dev_value_cache,
            position + 1,
            max_seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            position,
            &mut dev_split_out,
        )?;
    }
    ops::prefill_causal_attention(
        stream,
        module,
        &dev_query_batch,
        &dev_key_cache,
        &dev_value_cache,
        seq_len,
        max_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        &mut dev_fused_out,
    )?;

    let split = dev_split_out.to_host_vec(stream)?;
    let fused = dev_fused_out.to_host_vec(stream)?;
    compare_attention_output(
        &split,
        &fused,
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        max_seq_len,
    )
}

fn run_incremental_attention_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    max_seq_len: usize,
    seed: u64,
) -> AppResult<GemmStressStats> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(invalid_input(format!(
            "invalid incremental attention stress heads: n_heads={n_heads} n_kv_heads={n_kv_heads}"
        )));
    }
    if seq_len == 0 || seq_len > max_seq_len {
        return Err(invalid_input(format!(
            "invalid incremental attention stress sequence shape: seq_len={seq_len} max_seq_len={max_seq_len}"
        )));
    }

    let position = seq_len - 1;
    let q_len = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid_input("incremental attention stress query shape overflow"))?;
    let kv_token_len = n_kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid_input("incremental attention stress KV token shape overflow"))?;
    let kv_cache_len = max_seq_len
        .checked_mul(kv_token_len)
        .ok_or_else(|| invalid_input("incremental attention stress KV cache shape overflow"))?;

    let mut seed = seed;
    let mut query = vec![0.0_f32; q_len];
    let mut key = vec![0.0_f32; kv_token_len];
    let mut value = vec![0.0_f32; kv_token_len];
    let mut freqs = vec![0.0_f32; head_dim / 2];
    fill_stress_slice(&mut query, &mut seed, 0.25);
    fill_stress_slice(&mut key, &mut seed, 0.25);
    fill_stress_slice(&mut value, &mut seed, 0.25);
    fill_stress_slice(&mut freqs, &mut seed, 0.01);

    let dev_query = DeviceBuffer::from_host(stream, &query)?;
    let dev_key = DeviceBuffer::from_host(stream, &key)?;
    let dev_value = DeviceBuffer::from_host(stream, &value)?;
    let dev_freqs = DeviceBuffer::from_host(stream, &freqs)?;
    let mut split_query_rot = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut split_key_cache = DeviceBuffer::<f32>::zeroed(stream, kv_cache_len)?;
    let mut split_value_cache = DeviceBuffer::<f32>::zeroed(stream, kv_cache_len)?;
    let mut fused_query_rot = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut fused_key_cache = DeviceBuffer::<f32>::zeroed(stream, kv_cache_len)?;
    let mut fused_value_cache = DeviceBuffer::<f32>::zeroed(stream, kv_cache_len)?;

    ops::apply_rope(
        stream,
        module,
        &dev_query,
        &dev_freqs,
        position,
        head_dim,
        &mut split_query_rot,
    )?;
    ops::apply_rope_write_kv_cache(
        stream,
        module,
        &dev_key,
        &dev_freqs,
        position,
        max_seq_len,
        n_kv_heads,
        head_dim,
        &mut split_key_cache,
    )?;
    ops::write_kv_cache(
        stream,
        module,
        &dev_value,
        position,
        max_seq_len,
        n_kv_heads,
        head_dim,
        &mut split_value_cache,
    )?;
    ops::prepare_incremental_attention(
        stream,
        module,
        &dev_query,
        &dev_key,
        &dev_value,
        &dev_freqs,
        position,
        max_seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        &mut fused_query_rot,
        &mut fused_key_cache,
        &mut fused_value_cache,
    )?;

    let split_query_rot = split_query_rot.to_host_vec(stream)?;
    let split_key_cache = split_key_cache.to_host_vec(stream)?;
    let split_value_cache = split_value_cache.to_host_vec(stream)?;
    let fused_query_rot = fused_query_rot.to_host_vec(stream)?;
    let fused_key_cache = fused_key_cache.to_host_vec(stream)?;
    let fused_value_cache = fused_value_cache.to_host_vec(stream)?;
    compare_incremental_attention_staging(
        &split_query_rot,
        &fused_query_rot,
        &split_key_cache,
        &fused_key_cache,
        &split_value_cache,
        &fused_value_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        position,
        max_seq_len,
    )
}

fn run_gemm_stress_case<ALayout, BLayout, CLayout>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    m: usize,
    n: usize,
    k: usize,
    seed: u64,
    label: &str,
) -> AppResult<GemmStressStats>
where
    ALayout: Layout2D,
    BLayout: Layout2D,
    CLayout: Layout2D,
{
    let alpha = 0.75_f32;
    let beta = 0.25_f32;
    let a_layout = MatrixLayout::<ALayout>::packed(m, k);
    let b_layout = MatrixLayout::<BLayout>::packed(k, n);
    let c_layout = MatrixLayout::<CLayout>::packed(m, n);

    let mut seed = seed;
    let mut a = vec![0.0_f32; a_layout.capacity()];
    let mut b = vec![0.0_f32; b_layout.capacity()];
    let mut c = vec![0.0_f32; c_layout.capacity()];
    fill_matrix::<ALayout>(&mut a, &a_layout, m, k, &mut seed);
    fill_matrix::<BLayout>(&mut b, &b_layout, k, n, &mut seed);
    fill_matrix::<CLayout>(&mut c, &c_layout, m, n, &mut seed);

    let c_initial = c.clone();
    let expected = cpu_gemm_reference(
        &a, &a_layout, &b, &b_layout, &c_initial, &c_layout, m, n, k, alpha, beta,
    );

    let dev_a = DeviceBuffer::from_host(stream, &a)?;
    let dev_b = DeviceBuffer::from_host(stream, &b)?;
    let mut dev_c = DeviceBuffer::from_host(stream, &c)?;
    ops::gemm_f32::<ALayout, BLayout, CLayout>(
        stream, module, &dev_a, &dev_b, &mut dev_c, m, n, k, alpha, beta,
    )?;
    let actual = dev_c.to_host_vec(stream)?;

    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;
    let mut element_count = 0usize;
    for row in 0..m {
        for col in 0..n {
            let offset = c_layout.offset(row, col);
            let diff = (actual[offset] - expected[offset]).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff as f64;
            element_count += 1;
        }
    }

    let tolerance = 1.0e-4_f32 * (k.max(1) as f32).sqrt();
    if max_abs_diff > tolerance {
        return Err(invalid_data(format!(
            "GEMM stress {label} {m}x{k} * {k}x{n} failed: max_abs_diff={max_abs_diff:.8}, tolerance={tolerance:.8}, layouts A={} B={} C={}",
            ALayout::NAME,
            BLayout::NAME,
            CLayout::NAME
        )));
    }

    Ok(GemmStressStats {
        element_count,
        max_abs_diff,
        sum_abs_diff,
    })
}

fn run_gemm_bf16_stress_case<ALayout, BLayout, CLayout>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    m: usize,
    n: usize,
    k: usize,
    seed: u64,
    label: &str,
) -> AppResult<GemmStressStats>
where
    ALayout: Layout2D,
    BLayout: Layout2D,
    CLayout: Layout2D,
{
    let alpha = 0.75_f32;
    let beta = 0.25_f32;
    let a_layout = MatrixLayout::<ALayout>::packed(m, k);
    let b_layout = MatrixLayout::<BLayout>::packed(k, n);
    let c_layout = MatrixLayout::<CLayout>::packed(m, n);

    let mut seed = seed;
    let mut a = vec![0.0_f32; a_layout.capacity()];
    let mut b = vec![Bf16::from_bits(0); b_layout.capacity()];
    let mut c = vec![0.0_f32; c_layout.capacity()];
    fill_matrix::<ALayout>(&mut a, &a_layout, m, k, &mut seed);
    fill_bf16_matrix::<BLayout>(&mut b, &b_layout, k, n, &mut seed);
    fill_matrix::<CLayout>(&mut c, &c_layout, m, n, &mut seed);

    let c_initial = c.clone();
    let expected = cpu_gemm_bf16_reference(
        &a, &a_layout, &b, &b_layout, &c_initial, &c_layout, m, n, k, alpha, beta,
    );

    let dev_a = DeviceBuffer::from_host(stream, &a)?;
    let dev_b = DeviceBuffer::from_host(stream, &b)?;
    let mut dev_c = DeviceBuffer::from_host(stream, &c)?;
    ops::gemm_f32_bf16::<ALayout, BLayout, CLayout>(
        stream, module, &dev_a, &dev_b, &mut dev_c, m, n, k, alpha, beta,
    )?;
    let actual = dev_c.to_host_vec(stream)?;

    compare_gemm_output(label, &actual, &expected, &c_layout, m, n, k)
}

fn run_linear_batched_bf16_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    batch: usize,
    output_dim: usize,
    input_dim: usize,
    seed: u64,
    label: &str,
) -> AppResult<GemmStressStats> {
    let input_layout = MatrixLayout::<RowMajor>::packed(batch, input_dim);
    let weight_layout = MatrixLayout::<RowMajor>::packed(output_dim, input_dim);
    let output_layout = MatrixLayout::<RowMajor>::packed(batch, output_dim);

    let mut seed = seed;
    let mut input = vec![0.0_f32; input_layout.capacity()];
    let mut weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    let output = vec![0.0_f32; output_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input, &input_layout, batch, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );

    let expected = cpu_batched_linear_bf16_reference(
        &input,
        &input_layout,
        &weight,
        &weight_layout,
        batch,
        output_dim,
        input_dim,
    );

    let dev_input = DeviceBuffer::from_host(stream, &input)?;
    let dev_weight = DeviceBuffer::from_host(stream, &weight)?;
    let mut dev_output = DeviceBuffer::from_host(stream, &output)?;
    ops::linear_batched_bf16(
        stream,
        module,
        &dev_input,
        &dev_weight,
        batch,
        input_dim,
        output_dim,
        &mut dev_output,
    )?;
    let actual = dev_output.to_host_vec(stream)?;

    compare_gemm_output(
        label,
        &actual,
        &expected,
        &output_layout,
        batch,
        output_dim,
        input_dim,
    )
}

fn run_linear_batched_i8_scaled_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    batch: usize,
    output_dim: usize,
    input_dim: usize,
    seed: u64,
    label: &str,
) -> AppResult<GemmStressStats> {
    let input_layout = MatrixLayout::<RowMajor>::packed(batch, input_dim);
    let weight_layout = MatrixLayout::<RowMajor>::packed(output_dim, input_dim);
    let output_layout = MatrixLayout::<RowMajor>::packed(batch, output_dim);

    let mut seed = seed;
    let mut input = vec![0.0_f32; input_layout.capacity()];
    let mut weight_bf16 = vec![Bf16::from_bits(0); weight_layout.capacity()];
    let output = vec![0.0_f32; output_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input, &input_layout, batch, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut weight_bf16,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );
    let weight =
        RowwiseScaledI8Matrix::from_bf16_rows_symmetric(&weight_bf16, output_dim, input_dim);

    let expected = cpu_batched_linear_i8_scaled_reference(
        &input,
        &input_layout,
        &weight,
        batch,
        output_dim,
        input_dim,
    );

    let dev_input = DeviceBuffer::from_host(stream, &input)?;
    let dev_weight = weight.to_device(stream)?;
    let mut dev_output = DeviceBuffer::from_host(stream, &output)?;
    ops::linear_batched_i8_scaled(
        stream,
        module,
        &dev_input,
        &dev_weight,
        batch,
        input_dim,
        output_dim,
        &mut dev_output,
    )?;
    let actual = dev_output.to_host_vec(stream)?;

    compare_gemm_output(
        label,
        &actual,
        &expected,
        &output_layout,
        batch,
        output_dim,
        input_dim,
    )
}

fn run_linear_qkv_batched_bf16_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    batch: usize,
    q_output_dim: usize,
    kv_output_dim: usize,
    input_dim: usize,
    seed: u64,
    label: &str,
) -> AppResult<GemmStressStats> {
    let input_layout = MatrixLayout::<RowMajor>::packed(batch, input_dim);
    let q_weight_layout = MatrixLayout::<RowMajor>::packed(q_output_dim, input_dim);
    let kv_weight_layout = MatrixLayout::<RowMajor>::packed(kv_output_dim, input_dim);
    let q_output_layout = MatrixLayout::<RowMajor>::packed(batch, q_output_dim);
    let kv_output_layout = MatrixLayout::<RowMajor>::packed(batch, kv_output_dim);

    let mut seed = seed;
    let mut input = vec![0.0_f32; input_layout.capacity()];
    let mut wq = vec![Bf16::from_bits(0); q_weight_layout.capacity()];
    let mut wk = vec![Bf16::from_bits(0); kv_weight_layout.capacity()];
    let mut wv = vec![Bf16::from_bits(0); kv_weight_layout.capacity()];
    let query = vec![0.0_f32; q_output_layout.capacity()];
    let key = vec![0.0_f32; kv_output_layout.capacity()];
    let value = vec![0.0_f32; kv_output_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input, &input_layout, batch, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut wq,
        &q_weight_layout,
        q_output_dim,
        input_dim,
        &mut seed,
    );
    fill_bf16_matrix::<RowMajor>(
        &mut wk,
        &kv_weight_layout,
        kv_output_dim,
        input_dim,
        &mut seed,
    );
    fill_bf16_matrix::<RowMajor>(
        &mut wv,
        &kv_weight_layout,
        kv_output_dim,
        input_dim,
        &mut seed,
    );

    let expected_query = cpu_batched_linear_bf16_reference(
        &input,
        &input_layout,
        &wq,
        &q_weight_layout,
        batch,
        q_output_dim,
        input_dim,
    );
    let expected_key = cpu_batched_linear_bf16_reference(
        &input,
        &input_layout,
        &wk,
        &kv_weight_layout,
        batch,
        kv_output_dim,
        input_dim,
    );
    let expected_value = cpu_batched_linear_bf16_reference(
        &input,
        &input_layout,
        &wv,
        &kv_weight_layout,
        batch,
        kv_output_dim,
        input_dim,
    );

    let dev_input = DeviceBuffer::from_host(stream, &input)?;
    let dev_wq = DeviceBuffer::from_host(stream, &wq)?;
    let dev_wk = DeviceBuffer::from_host(stream, &wk)?;
    let dev_wv = DeviceBuffer::from_host(stream, &wv)?;
    let mut dev_query = DeviceBuffer::from_host(stream, &query)?;
    let mut dev_key = DeviceBuffer::from_host(stream, &key)?;
    let mut dev_value = DeviceBuffer::from_host(stream, &value)?;
    ops::linear_qkv_batched_bf16(
        stream,
        module,
        &dev_input,
        &dev_wq,
        &dev_wk,
        &dev_wv,
        batch,
        input_dim,
        q_output_dim,
        kv_output_dim,
        &mut dev_query,
        &mut dev_key,
        &mut dev_value,
    )?;
    let actual_query = dev_query.to_host_vec(stream)?;
    let actual_key = dev_key.to_host_vec(stream)?;
    let actual_value = dev_value.to_host_vec(stream)?;

    let q_stats = compare_gemm_output(
        &format!("{label}-q"),
        &actual_query,
        &expected_query,
        &q_output_layout,
        batch,
        q_output_dim,
        input_dim,
    )?;
    let k_stats = compare_gemm_output(
        &format!("{label}-k"),
        &actual_key,
        &expected_key,
        &kv_output_layout,
        batch,
        kv_output_dim,
        input_dim,
    )?;
    let v_stats = compare_gemm_output(
        &format!("{label}-v"),
        &actual_value,
        &expected_value,
        &kv_output_layout,
        batch,
        kv_output_dim,
        input_dim,
    )?;

    Ok(GemmStressStats {
        element_count: q_stats.element_count + k_stats.element_count + v_stats.element_count,
        max_abs_diff: q_stats
            .max_abs_diff
            .max(k_stats.max_abs_diff)
            .max(v_stats.max_abs_diff),
        sum_abs_diff: q_stats.sum_abs_diff + k_stats.sum_abs_diff + v_stats.sum_abs_diff,
    })
}

fn run_silu_gate_up_via_gemm_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    batch: usize,
    output_dim: usize,
    input_dim: usize,
    seed: u64,
    label: &str,
) -> AppResult<GemmStressStats> {
    let input_layout = MatrixLayout::<RowMajor>::packed(batch, input_dim);
    let weight_layout = MatrixLayout::<RowMajor>::packed(output_dim, input_dim);
    let output_layout = MatrixLayout::<RowMajor>::packed(batch, output_dim);

    let mut seed = seed;
    let mut input = vec![0.0_f32; input_layout.capacity()];
    let mut gate_weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    let mut up_weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    let output = vec![0.0_f32; output_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input, &input_layout, batch, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut gate_weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );
    fill_bf16_matrix::<RowMajor>(
        &mut up_weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );

    let expected = cpu_batched_silu_gate_up_bf16_reference(
        &input,
        &input_layout,
        &gate_weight,
        &up_weight,
        &weight_layout,
        batch,
        output_dim,
        input_dim,
    );

    let dev_input = DeviceBuffer::from_host(stream, &input)?;
    let dev_gate_weight = DeviceBuffer::from_host(stream, &gate_weight)?;
    let dev_up_weight = DeviceBuffer::from_host(stream, &up_weight)?;
    let mut dev_gate = DeviceBuffer::<f32>::zeroed(stream, output_layout.capacity())?;
    let mut dev_up = DeviceBuffer::<f32>::zeroed(stream, output_layout.capacity())?;
    let mut dev_output = DeviceBuffer::from_host(stream, &output)?;
    ops::linear_batched_bf16(
        stream,
        module,
        &dev_input,
        &dev_gate_weight,
        batch,
        input_dim,
        output_dim,
        &mut dev_gate,
    )?;
    ops::linear_batched_bf16(
        stream,
        module,
        &dev_input,
        &dev_up_weight,
        batch,
        input_dim,
        output_dim,
        &mut dev_up,
    )?;
    ops::silu_mul_prefix(
        stream,
        module,
        &dev_gate,
        &dev_up,
        output_layout.capacity(),
        &mut dev_output,
    )?;
    let actual = dev_output.to_host_vec(stream)?;

    compare_gemm_output(
        label,
        &actual,
        &expected,
        &output_layout,
        batch,
        output_dim,
        input_dim,
    )
}

fn run_silu_gate_up_bf16_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input_dim: usize,
    output_dim: usize,
    seed: u64,
    label: &str,
) -> AppResult<GemmStressStats> {
    let input_layout = MatrixLayout::<RowMajor>::packed(1, input_dim);
    let weight_layout = MatrixLayout::<RowMajor>::packed(output_dim, input_dim);
    let output_layout = MatrixLayout::<RowMajor>::packed(1, output_dim);

    let mut seed = seed;
    let mut input_matrix = vec![0.0_f32; input_layout.capacity()];
    let mut gate_weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    let mut up_weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    let output = vec![0.0_f32; output_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input_matrix, &input_layout, 1, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut gate_weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );
    fill_bf16_matrix::<RowMajor>(
        &mut up_weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );

    let expected = cpu_batched_silu_gate_up_bf16_reference(
        &input_matrix,
        &input_layout,
        &gate_weight,
        &up_weight,
        &weight_layout,
        1,
        output_dim,
        input_dim,
    );
    let dev_input = DeviceBuffer::from_host(stream, &input_matrix)?;
    let dev_gate_weight = DeviceBuffer::from_host(stream, &gate_weight)?;
    let dev_up_weight = DeviceBuffer::from_host(stream, &up_weight)?;
    let mut dev_output = DeviceBuffer::from_host(stream, &output)?;
    ops::silu_gate_up_bf16(
        stream,
        module,
        &dev_input,
        &dev_gate_weight,
        &dev_up_weight,
        &mut dev_output,
    )?;
    let actual = dev_output.to_host_vec(stream)?;

    compare_gemm_output(
        label,
        &actual,
        &expected,
        &output_layout,
        1,
        output_dim,
        input_dim,
    )
}

fn run_linear_residual_bf16_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input_dim: usize,
    output_dim: usize,
    seed: u64,
    label: &str,
) -> AppResult<GemmStressStats> {
    let input_layout = MatrixLayout::<RowMajor>::packed(1, input_dim);
    let weight_layout = MatrixLayout::<RowMajor>::packed(output_dim, input_dim);
    let output_layout = MatrixLayout::<RowMajor>::packed(1, output_dim);

    let mut seed = seed;
    let mut input_matrix = vec![0.0_f32; input_layout.capacity()];
    let mut weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    let mut residual = vec![0.0_f32; output_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input_matrix, &input_layout, 1, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );
    fill_matrix::<RowMajor>(&mut residual, &output_layout, 1, output_dim, &mut seed);

    let expected = cpu_linear_residual_bf16_reference(
        &input_matrix,
        &input_layout,
        &weight,
        &weight_layout,
        &residual,
        &output_layout,
        output_dim,
        input_dim,
    );
    let output = vec![0.0_f32; output_layout.capacity()];
    let dev_input = DeviceBuffer::from_host(stream, &input_matrix)?;
    let dev_weight = DeviceBuffer::from_host(stream, &weight)?;
    let dev_residual = DeviceBuffer::from_host(stream, &residual)?;
    let mut dev_output = DeviceBuffer::from_host(stream, &output)?;
    ops::linear_residual_bf16(
        stream,
        module,
        &dev_input,
        &dev_weight,
        &dev_residual,
        &mut dev_output,
    )?;
    let actual = dev_output.to_host_vec(stream)?;

    compare_gemm_output(
        label,
        &actual,
        &expected,
        &output_layout,
        1,
        output_dim,
        input_dim,
    )
}

fn run_argmax_packed_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    len: usize,
    seed: u64,
) -> AppResult<usize> {
    let mut seed = seed;
    let mut logits = vec![0.0_f32; len];
    fill_stress_slice(&mut logits, &mut seed, 10.0);

    let expected_token = if len == 1 { 0 } else { len / 3 };
    let tied_token = len - 1;
    let forced_logit = 12345.0_f32;
    logits[expected_token] = forced_logit;
    logits[tied_token] = forced_logit;

    let dev_logits = DeviceBuffer::from_host(stream, &logits)?;
    let mut dev_packed = DeviceBuffer::<u64>::zeroed(stream, 1)?;
    ops::argmax_f32_packed(stream, module, &dev_logits, &mut dev_packed)?;
    let packed = dev_packed.to_host_vec(stream)?[0];
    let token = packed as u32;
    let logit = f32::from_bits((packed >> 32) as u32);

    if token != expected_token as u32 || logit.to_bits() != forced_logit.to_bits() {
        return Err(invalid_data(format!(
            "packed argmax stress failed: len={len} expected=({expected_token}, {forced_logit:.8}) got=({token}, {logit:.8})"
        )));
    }

    Ok(len)
}

fn run_linear_top1_bf16_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input_dim: usize,
    output_dim: usize,
    seed: u64,
) -> AppResult<usize> {
    let input_layout = MatrixLayout::<RowMajor>::packed(1, input_dim);
    let weight_layout = MatrixLayout::<RowMajor>::packed(output_dim, input_dim);

    let mut seed = seed;
    let mut input_matrix = vec![0.0_f32; input_layout.capacity()];
    let mut weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input_matrix, &input_layout, 1, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut weight,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );

    let forced_row = if output_dim == 1 { 0 } else { output_dim / 3 };
    for col in 0..input_dim {
        let input_value = input_matrix[input_layout.offset(0, col)];
        let sign = if input_value >= 0.0 { 1.0 } else { -1.0 };
        weight[weight_layout.offset(forced_row, col)] = Bf16::from_f32(sign * 4.0);
    }

    let (expected_token, expected_logit) = cpu_linear_top1_bf16_reference(
        &input_matrix,
        &input_layout,
        &weight,
        &weight_layout,
        output_dim,
        input_dim,
    );

    let dev_input = DeviceBuffer::from_host(stream, &input_matrix)?;
    let dev_weight = DeviceBuffer::from_host(stream, &weight)?;
    let partial_count = ops::linear_top1_bf16_partial_count(output_dim);
    let mut dev_partial_tokens = DeviceBuffer::<u32>::zeroed(stream, partial_count)?;
    let mut dev_partial_logits = DeviceBuffer::<f32>::zeroed(stream, partial_count)?;
    let mut dev_packed = DeviceBuffer::<u64>::zeroed(stream, 1)?;
    ops::linear_top1_bf16(
        stream,
        module,
        &dev_input,
        &dev_weight,
        &mut dev_partial_tokens,
        &mut dev_partial_logits,
        &mut dev_packed,
    )?;
    let packed = dev_packed.to_host_vec(stream)?[0];
    let token = packed as u32;
    let logit = f32::from_bits((packed >> 32) as u32);

    let tolerance = 1.0e-4_f32 * (input_dim.max(1) as f32).sqrt();
    if token != expected_token || (logit - expected_logit).abs() > tolerance {
        return Err(invalid_data(format!(
            "linear top1 BF16 stress failed: output_dim={output_dim} input_dim={input_dim} expected=({expected_token}, {expected_logit:.8}) got=({token}, {logit:.8}) tolerance={tolerance:.8}"
        )));
    }

    Ok(output_dim * input_dim)
}

fn run_linear_top1_i8_scaled_stress_case(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    input_dim: usize,
    output_dim: usize,
    seed: u64,
) -> AppResult<usize> {
    let input_layout = MatrixLayout::<RowMajor>::packed(1, input_dim);
    let weight_layout = MatrixLayout::<RowMajor>::packed(output_dim, input_dim);

    let mut seed = seed;
    let mut input_matrix = vec![0.0_f32; input_layout.capacity()];
    let mut weight_bf16 = vec![Bf16::from_bits(0); weight_layout.capacity()];
    fill_matrix::<RowMajor>(&mut input_matrix, &input_layout, 1, input_dim, &mut seed);
    fill_bf16_matrix::<RowMajor>(
        &mut weight_bf16,
        &weight_layout,
        output_dim,
        input_dim,
        &mut seed,
    );

    let forced_row = if output_dim == 1 { 0 } else { output_dim / 3 };
    for col in 0..input_dim {
        let input_value = input_matrix[input_layout.offset(0, col)];
        let sign = if input_value >= 0.0 { 1.0 } else { -1.0 };
        weight_bf16[weight_layout.offset(forced_row, col)] = Bf16::from_f32(sign * 4.0);
    }

    let weight =
        RowwiseScaledI8Matrix::from_bf16_rows_symmetric(&weight_bf16, output_dim, input_dim);
    let logits = weight.matvec_cpu(&input_matrix[..input_dim]);
    let (expected_token, expected_logit) = top1_from_logits(&logits)
        .ok_or_else(|| invalid_data("scaled-i8 top1 stress produced no CPU logits"))?;

    let dev_input = DeviceBuffer::from_host(stream, &input_matrix)?;
    let dev_weight = weight.to_device(stream)?;
    let partial_count = ops::linear_top1_i8_scaled_partial_count(output_dim);
    let mut dev_partial_tokens = DeviceBuffer::<u32>::zeroed(stream, partial_count)?;
    let mut dev_partial_logits = DeviceBuffer::<f32>::zeroed(stream, partial_count)?;
    let mut dev_packed = DeviceBuffer::<u64>::zeroed(stream, 1)?;
    ops::linear_top1_i8_scaled(
        stream,
        module,
        &dev_input,
        &dev_weight,
        &mut dev_partial_tokens,
        &mut dev_partial_logits,
        &mut dev_packed,
    )?;
    let packed = dev_packed.to_host_vec(stream)?[0];
    let token = packed as u32;
    let logit = f32::from_bits((packed >> 32) as u32);

    let tolerance = 1.0e-4_f32 * (input_dim.max(1) as f32).sqrt();
    if token != expected_token || (logit - expected_logit).abs() > tolerance {
        return Err(invalid_data(format!(
            "linear top1 scaled-i8 stress failed: output_dim={output_dim} input_dim={input_dim} expected=({expected_token}, {expected_logit:.8}) got=({token}, {logit:.8}) tolerance={tolerance:.8}"
        )));
    }

    Ok(output_dim * input_dim)
}

fn top1_from_logits(logits: &[f32]) -> Option<(u32, f32)> {
    let mut best = None::<(u32, f32)>;
    for (index, &logit) in logits.iter().enumerate() {
        let token = index as u32;
        if best
            .map(|(best_token, best_logit)| {
                logit > best_logit || (logit == best_logit && token < best_token)
            })
            .unwrap_or(true)
        {
            best = Some((token, logit));
        }
    }
    best
}

fn sample_rows(row_count: usize) -> Vec<usize> {
    let mut rows = vec![0];
    if row_count > 1 {
        rows.push(row_count / 3);
        rows.push(row_count - 1);
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

fn cpu_linear_row_bf16_reference(
    input: &[f32],
    weight: &[Bf16],
    weight_layout: &MatrixLayout<RowMajor>,
    row: usize,
    input_dim: usize,
) -> f32 {
    let mut acc = 0.0_f32;
    for col in 0..input_dim {
        acc += input[col] * weight[weight_layout.offset(row, col)].to_f32();
    }
    acc
}

fn verify_linear_samples(
    label: &str,
    input: &[f32],
    weight: &[Bf16],
    weight_layout: &MatrixLayout<RowMajor>,
    actual: &[f32],
    output_dim: usize,
    input_dim: usize,
) -> AppResult<()> {
    let tolerance = 1.0e-4_f32 * (input_dim.max(1) as f32).sqrt();
    for row in sample_rows(output_dim) {
        let expected = cpu_linear_row_bf16_reference(input, weight, weight_layout, row, input_dim);
        assert_close(actual[row], expected, tolerance, &format!("{label}[{row}]"))?;
    }
    Ok(())
}

fn verify_silu_gate_up_samples(
    label: &str,
    input: &[f32],
    gate_weight: &[Bf16],
    up_weight: &[Bf16],
    weight_layout: &MatrixLayout<RowMajor>,
    actual: &[f32],
    output_dim: usize,
    input_dim: usize,
) -> AppResult<()> {
    let tolerance = 1.0e-4_f32 * (input_dim.max(1) as f32).sqrt();
    for row in sample_rows(output_dim) {
        let gate = cpu_linear_row_bf16_reference(input, gate_weight, weight_layout, row, input_dim);
        let up = cpu_linear_row_bf16_reference(input, up_weight, weight_layout, row, input_dim);
        let expected = gate / (1.0 + (-gate).exp()) * up;
        assert_close(actual[row], expected, tolerance, &format!("{label}[{row}]"))?;
    }
    Ok(())
}

fn verify_linear_residual_samples(
    label: &str,
    input: &[f32],
    weight: &[Bf16],
    weight_layout: &MatrixLayout<RowMajor>,
    residual: &[f32],
    actual: &[f32],
    output_dim: usize,
    input_dim: usize,
) -> AppResult<()> {
    let tolerance = 1.0e-4_f32 * (input_dim.max(1) as f32).sqrt();
    for row in sample_rows(output_dim) {
        let expected = residual[row]
            + cpu_linear_row_bf16_reference(input, weight, weight_layout, row, input_dim);
        assert_close(actual[row], expected, tolerance, &format!("{label}[{row}]"))?;
    }
    Ok(())
}

fn compare_gemm_output<L: Layout2D>(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    c_layout: &MatrixLayout<L>,
    m: usize,
    n: usize,
    k: usize,
) -> AppResult<GemmStressStats> {
    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;
    let mut element_count = 0usize;
    for row in 0..m {
        for col in 0..n {
            let offset = c_layout.offset(row, col);
            let diff = (actual[offset] - expected[offset]).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff as f64;
            element_count += 1;
        }
    }

    let tolerance = 1.0e-4_f32 * (k.max(1) as f32).sqrt();
    if max_abs_diff > tolerance {
        return Err(invalid_data(format!(
            "GEMM stress {label} {m}x{k} * {k}x{n} failed: max_abs_diff={max_abs_diff:.8}, tolerance={tolerance:.8}"
        )));
    }

    Ok(GemmStressStats {
        element_count,
        max_abs_diff,
        sum_abs_diff,
    })
}

fn compare_attention_output(
    split: &[f32],
    fused: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    max_seq_len: usize,
) -> AppResult<GemmStressStats> {
    if split.len() != fused.len() {
        return Err(invalid_data(format!(
            "attention stress output length mismatch: split={} fused={}",
            split.len(),
            fused.len()
        )));
    }

    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;
    for (&split, &fused) in split.iter().zip(fused.iter()) {
        let diff = (split - fused).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff as f64;
    }

    let tolerance = 5.0e-5_f32;
    if max_abs_diff > tolerance {
        return Err(invalid_data(format!(
            "attention stress failed: n_heads={n_heads} n_kv_heads={n_kv_heads} head_dim={head_dim} seq_len={seq_len} max_seq_len={max_seq_len} max_abs_diff={max_abs_diff:.8} tolerance={tolerance:.8}"
        )));
    }

    Ok(GemmStressStats {
        element_count: split.len(),
        max_abs_diff,
        sum_abs_diff,
    })
}

fn compare_incremental_attention_staging(
    split_query_rot: &[f32],
    fused_query_rot: &[f32],
    split_key_cache: &[f32],
    fused_key_cache: &[f32],
    split_value_cache: &[f32],
    fused_value_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: usize,
    max_seq_len: usize,
) -> AppResult<GemmStressStats> {
    let mut element_count = 0usize;
    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;
    accumulate_slice_diff(
        "query_rot",
        split_query_rot,
        fused_query_rot,
        &mut element_count,
        &mut max_abs_diff,
        &mut sum_abs_diff,
    )?;
    accumulate_slice_diff(
        "key_cache",
        split_key_cache,
        fused_key_cache,
        &mut element_count,
        &mut max_abs_diff,
        &mut sum_abs_diff,
    )?;
    accumulate_slice_diff(
        "value_cache",
        split_value_cache,
        fused_value_cache,
        &mut element_count,
        &mut max_abs_diff,
        &mut sum_abs_diff,
    )?;

    let tolerance = 1.0e-6_f32;
    if max_abs_diff > tolerance {
        return Err(invalid_data(format!(
            "incremental attention staging stress failed: n_heads={n_heads} n_kv_heads={n_kv_heads} head_dim={head_dim} position={position} max_seq_len={max_seq_len} max_abs_diff={max_abs_diff:.8} tolerance={tolerance:.8}"
        )));
    }

    Ok(GemmStressStats {
        element_count,
        max_abs_diff,
        sum_abs_diff,
    })
}

fn accumulate_slice_diff(
    label: &str,
    reference: &[f32],
    candidate: &[f32],
    element_count: &mut usize,
    max_abs_diff: &mut f32,
    sum_abs_diff: &mut f64,
) -> AppResult<()> {
    if reference.len() != candidate.len() {
        return Err(invalid_data(format!(
            "{label} length mismatch: reference={} candidate={}",
            reference.len(),
            candidate.len()
        )));
    }
    for (&reference, &candidate) in reference.iter().zip(candidate.iter()) {
        let diff = (reference - candidate).abs();
        *max_abs_diff = (*max_abs_diff).max(diff);
        *sum_abs_diff += diff as f64;
        *element_count += 1;
    }
    Ok(())
}

fn fill_matrix<L: Layout2D>(
    values: &mut [f32],
    layout: &MatrixLayout<L>,
    rows: usize,
    cols: usize,
    seed: &mut u64,
) {
    for row in 0..rows {
        for col in 0..cols {
            values[layout.offset(row, col)] = next_stress_value(seed);
        }
    }
}

fn fill_stress_slice(values: &mut [f32], seed: &mut u64, scale: f32) {
    for value in values {
        *value = next_stress_value(seed) * scale;
    }
}

fn fill_bf16_matrix<L: Layout2D>(
    values: &mut [Bf16],
    layout: &MatrixLayout<L>,
    rows: usize,
    cols: usize,
    seed: &mut u64,
) {
    for row in 0..rows {
        for col in 0..cols {
            values[layout.offset(row, col)] = Bf16::from_f32(next_stress_value(seed));
        }
    }
}

fn cpu_gemm_reference<ALayout, BLayout, CLayout>(
    a: &[f32],
    a_layout: &MatrixLayout<ALayout>,
    b: &[f32],
    b_layout: &MatrixLayout<BLayout>,
    c_initial: &[f32],
    c_layout: &MatrixLayout<CLayout>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut out = c_initial.to_vec();
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[a_layout.offset(row, kk)] * b[b_layout.offset(kk, col)];
            }
            let c_offset = c_layout.offset(row, col);
            out[c_offset] = alpha * acc + beta * c_initial[c_offset];
        }
    }
    out
}

fn cpu_gemm_bf16_reference<ALayout, BLayout, CLayout>(
    a: &[f32],
    a_layout: &MatrixLayout<ALayout>,
    b: &[Bf16],
    b_layout: &MatrixLayout<BLayout>,
    c_initial: &[f32],
    c_layout: &MatrixLayout<CLayout>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut out = c_initial.to_vec();
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[a_layout.offset(row, kk)] * b[b_layout.offset(kk, col)].to_f32();
            }
            let c_offset = c_layout.offset(row, col);
            out[c_offset] = alpha * acc + beta * c_initial[c_offset];
        }
    }
    out
}

fn cpu_batched_linear_bf16_reference(
    input: &[f32],
    input_layout: &MatrixLayout<RowMajor>,
    weight: &[Bf16],
    weight_layout: &MatrixLayout<RowMajor>,
    batch: usize,
    output_dim: usize,
    input_dim: usize,
) -> Vec<f32> {
    let output_layout = MatrixLayout::<RowMajor>::packed(batch, output_dim);
    let mut out = vec![0.0_f32; output_layout.capacity()];
    for row in 0..batch {
        for col in 0..output_dim {
            let mut acc = 0.0_f32;
            for kk in 0..input_dim {
                acc += input[input_layout.offset(row, kk)]
                    * weight[weight_layout.offset(col, kk)].to_f32();
            }
            out[output_layout.offset(row, col)] = acc;
        }
    }
    out
}

fn cpu_batched_linear_i8_scaled_reference(
    input: &[f32],
    input_layout: &MatrixLayout<RowMajor>,
    weight: &RowwiseScaledI8Matrix,
    batch: usize,
    output_dim: usize,
    input_dim: usize,
) -> Vec<f32> {
    let output_layout = MatrixLayout::<RowMajor>::packed(batch, output_dim);
    let mut out = vec![0.0_f32; output_layout.capacity()];
    for row in 0..batch {
        for col in 0..output_dim {
            let mut acc = 0.0_f32;
            for kk in 0..input_dim {
                acc += input[input_layout.offset(row, kk)] * weight.scaled_value(col, kk);
            }
            out[output_layout.offset(row, col)] = acc;
        }
    }
    out
}

fn cpu_batched_silu_gate_up_bf16_reference(
    input: &[f32],
    input_layout: &MatrixLayout<RowMajor>,
    gate_weight: &[Bf16],
    up_weight: &[Bf16],
    weight_layout: &MatrixLayout<RowMajor>,
    batch: usize,
    output_dim: usize,
    input_dim: usize,
) -> Vec<f32> {
    let output_layout = MatrixLayout::<RowMajor>::packed(batch, output_dim);
    let mut out = vec![0.0_f32; output_layout.capacity()];
    for row in 0..batch {
        for col in 0..output_dim {
            let mut gate_acc = 0.0_f32;
            let mut up_acc = 0.0_f32;
            for kk in 0..input_dim {
                let input_value = input[input_layout.offset(row, kk)];
                gate_acc += input_value * gate_weight[weight_layout.offset(col, kk)].to_f32();
                up_acc += input_value * up_weight[weight_layout.offset(col, kk)].to_f32();
            }
            out[output_layout.offset(row, col)] = gate_acc / (1.0 + (-gate_acc).exp()) * up_acc;
        }
    }
    out
}

fn cpu_linear_residual_bf16_reference(
    input: &[f32],
    input_layout: &MatrixLayout<RowMajor>,
    weight: &[Bf16],
    weight_layout: &MatrixLayout<RowMajor>,
    residual: &[f32],
    output_layout: &MatrixLayout<RowMajor>,
    output_dim: usize,
    input_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; output_layout.capacity()];
    for row in 0..output_dim {
        let mut acc = 0.0_f32;
        for col in 0..input_dim {
            acc += input[input_layout.offset(0, col)]
                * weight[weight_layout.offset(row, col)].to_f32();
        }
        let out_offset = output_layout.offset(0, row);
        out[out_offset] = residual[out_offset] + acc;
    }
    out
}

fn cpu_linear_top1_bf16_reference(
    input: &[f32],
    input_layout: &MatrixLayout<RowMajor>,
    weight: &[Bf16],
    weight_layout: &MatrixLayout<RowMajor>,
    output_dim: usize,
    input_dim: usize,
) -> (u32, f32) {
    let mut best_token = 0u32;
    let mut best_logit = f32::NEG_INFINITY;
    for row in 0..output_dim {
        let mut acc = 0.0_f32;
        for col in 0..input_dim {
            acc += input[input_layout.offset(0, col)]
                * weight[weight_layout.offset(row, col)].to_f32();
        }
        let token = row as u32;
        if acc > best_logit || (acc == best_logit && token < best_token) {
            best_token = token;
            best_logit = acc;
        }
    }

    (best_token, best_logit)
}

fn next_stress_value(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bucket = ((*seed >> 32) & 0xffff) as f32 / 65535.0;
    bucket * 2.0 - 1.0
}

fn run_ministral_chat_backend_smoke(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let backend = parse_optional_backend(args, &mut index, InferenceBackend::Bf16)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let prompt = first_prompt(&cli);
    let (stream, module) = cuda_handles()?;
    let options = ChatInferenceOptions {
        max_new_tokens,
        top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let result = inference::run_ministral_single_turn_chat_with_backend(
        stream, module, &model_dir, prompt, backend, &options,
    )?;

    println!(
        "Ministral chat backend: backend={} user_prompt={:?}",
        backend_label(backend),
        prompt
    );
    print_chat_result(&result, &cli.system_prompt);
    if let Some(path) = &cli.report_path {
        write_chat_generation_report(
            path,
            &[(prompt.to_string(), result)],
            &InferenceReportMetadata::new(&model_dir, backend)
                .with_decode_strategy("greedy")
                .with_system_prompt(&cli.system_prompt)
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
        )?;
        println!("  report={}", path.display());
    }

    Ok(())
}

fn run_ministral_chat_all_linear_int8_smoke(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let prompt = first_prompt(&cli);
    let (stream, module) = cuda_handles()?;
    let options = ChatInferenceOptions {
        max_new_tokens,
        top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let result = inference::run_ministral_all_linear_int8_chat_comparison(
        stream, module, &model_dir, prompt, &options,
    )?;

    println!("Ministral all-linear int8 chat: user_prompt={prompt:?}");
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    print_token_window("prompt_tokens", &result.prompt_tokens);
    println!("  generated_tokens={:?}", result.generated_tokens);
    println!("  generated_text={:?}", result.generated_text);
    println!(
        "  reference_memory_bytes={} int8_memory_bytes={}",
        result.reference_memory_stats.total_resident_bytes,
        result.int8_memory_stats.total_resident_bytes
    );
    for step in &result.steps {
        println!(
            "  step={} reference_token={} int8_token={} token_matches={} kl_divergence={:.12} max_abs_diff={:.8}",
            step.step,
            step.reference_token_id,
            step.int8_token_id,
            step.token_matches,
            step.kl_divergence,
            step.max_abs_diff
        );
    }

    Ok(())
}

#[derive(Debug)]
struct QuantizationManifestCli {
    report_path: Option<PathBuf>,
}

#[derive(Debug)]
struct QuantizationExportCli {
    output_dir: PathBuf,
    limit_tensors: Option<usize>,
    tensor_names: Vec<String>,
    tensor_indices: Vec<usize>,
}

#[derive(Debug)]
struct QuantizationValidateCli {
    export_dir: PathBuf,
    report_path: Option<PathBuf>,
}

#[derive(Debug)]
struct WeightSourceCompareCli {
    element_start: usize,
    element_limit: usize,
    tensor_names: Vec<String>,
}

#[derive(Debug)]
struct TensorPrefixComparison {
    compared_elements: usize,
    exact_prefix_match: bool,
    max_abs_diff: f32,
    mean_abs_diff: f32,
    first_mismatch: Option<usize>,
}

#[derive(Debug)]
struct QuantizationTensorPlan {
    name: String,
    role: String,
    dtype: DType,
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
    source_bytes: u64,
    quantized_value_bytes: u64,
    quantized_scale_bytes: u64,
    quantized_bytes: u64,
}

#[derive(Debug)]
struct PreservedTensorPlan {
    name: String,
    role: String,
    dtype: DType,
    shape: Vec<usize>,
    source_bytes: u64,
}

#[derive(Debug)]
struct QuantizationManifestSummary {
    quantized_tensor_count: usize,
    preserved_tensor_count: usize,
    source_quantized_bytes: u64,
    projected_quantized_bytes: u64,
    projected_quantized_value_bytes: u64,
    projected_quantized_scale_bytes: u64,
    preserved_source_bytes: u64,
    source_total_bytes: u64,
    projected_total_bytes: u64,
}

impl QuantizationManifestSummary {
    fn quantized_ratio(&self) -> f64 {
        ratio_u64(self.projected_quantized_bytes, self.source_quantized_bytes)
    }

    fn projected_total_ratio(&self) -> f64 {
        ratio_u64(self.projected_total_bytes, self.source_total_bytes)
    }
}

#[derive(Debug)]
struct QuantizationManifest {
    model_dir: PathBuf,
    weights_path: PathBuf,
    n_layers: usize,
    quantized_tensors: Vec<QuantizationTensorPlan>,
    preserved_tensors: Vec<PreservedTensorPlan>,
    summary: QuantizationManifestSummary,
}

#[derive(Debug)]
struct QuantizedTensorExport {
    index: usize,
    name: String,
    role: String,
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
    values_path: PathBuf,
    scales_path: PathBuf,
    values_bytes: u64,
    scales_bytes: u64,
}

#[derive(Debug)]
struct QuantizationExportManifest {
    model_dir: PathBuf,
    weights_path: PathBuf,
    output_dir: PathBuf,
    planned_tensor_count: usize,
    exported_tensors: Vec<QuantizedTensorExport>,
    complete_export: bool,
    summary: QuantizationManifestSummary,
}

fn run_ministral_quantization_manifest(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let cli = parse_quantization_manifest_cli(args, index)?;
    let manifest = build_all_linear_quantization_manifest(&model_dir)?;

    println!(
        "Ministral all-linear int8 quantization manifest: tensors={} preserved={} layers={}",
        manifest.summary.quantized_tensor_count,
        manifest.summary.preserved_tensor_count,
        manifest.n_layers
    );
    println!("  model_dir={}", manifest.model_dir.display());
    println!("  weights_path={}", manifest.weights_path.display());
    println!(
        "  source_quantized_bytes={}",
        manifest.summary.source_quantized_bytes
    );
    println!(
        "  projected_quantized_bytes={}",
        manifest.summary.projected_quantized_bytes
    );
    println!(
        "  quantized_weight_ratio={:.6}",
        manifest.summary.quantized_ratio()
    );
    println!(
        "  source_total_bytes={}",
        manifest.summary.source_total_bytes
    );
    println!(
        "  projected_total_bytes={}",
        manifest.summary.projected_total_bytes
    );
    println!(
        "  projected_total_ratio={:.6}",
        manifest.summary.projected_total_ratio()
    );

    if let Some(path) = &cli.report_path {
        write_quantization_manifest_report(path, &manifest)?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_quantization_export(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let cli = parse_quantization_export_cli(args, index)?;
    let export = export_all_linear_quantization_archive(&model_dir, &cli)?;

    println!(
        "Ministral all-linear int8 quantization export: exported={}/{} complete={}",
        export.exported_tensors.len(),
        export.planned_tensor_count,
        export.complete_export
    );
    println!("  model_dir={}", export.model_dir.display());
    println!("  weights_path={}", export.weights_path.display());
    println!("  output_dir={}", export.output_dir.display());
    println!(
        "  projected_quantized_bytes={}",
        export.summary.projected_quantized_bytes
    );
    println!(
        "  manifest={}",
        export.output_dir.join("manifest.json").display()
    );

    Ok(())
}

fn run_ministral_quantization_validate(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let cli = parse_quantization_validate_cli(args, index)?;
    let manifest = build_all_linear_quantization_manifest(&model_dir)?;
    let validation = nn_rust_quantization::validate_ministral_all_linear_int8_export_archive(
        &model_dir,
        &cli.export_dir,
    );

    match validation {
        Ok(()) => {
            println!("Ministral all-linear int8 export validation: valid=true");
            println!("  model_dir={}", model_dir.display());
            println!("  export_dir={}", cli.export_dir.display());
            println!(
                "  expected_quantized_tensor_count={}",
                manifest.summary.quantized_tensor_count
            );
            if let Some(path) = &cli.report_path {
                write_quantization_validation_report(
                    path,
                    &model_dir,
                    &cli.export_dir,
                    &manifest,
                    true,
                    None,
                )?;
                println!("report={}", path.display());
            }
            Ok(())
        }
        Err(error) => {
            let error_message = error.to_string();
            println!("Ministral all-linear int8 export validation: valid=false");
            println!("  model_dir={}", model_dir.display());
            println!("  export_dir={}", cli.export_dir.display());
            println!(
                "  expected_quantized_tensor_count={}",
                manifest.summary.quantized_tensor_count
            );
            println!("  error={error_message}");
            if let Some(path) = &cli.report_path {
                write_quantization_validation_report(
                    path,
                    &model_dir,
                    &cli.export_dir,
                    &manifest,
                    false,
                    Some(&error_message),
                )?;
                println!("report={}", path.display());
            }
            Err(invalid_data(format!(
                "all-linear int8 export archive {} is invalid: {error_message}",
                cli.export_dir.display()
            )))
        }
    }
}

fn run_ministral_quantize_eval(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;

    match nn_rust_quantization::validate_ministral_all_linear_int8_export_archive(
        &model_dir,
        &export_dir,
    ) {
        Ok(()) => {
            println!(
                "Ministral quantize+eval: reusing complete all-linear-int8 export at {}",
                export_dir.display()
            );
        }
        Err(error) => {
            println!(
                "Ministral quantize+eval: export_dir={} is not a complete archive: {error}",
                export_dir.display()
            );
            let export_cli = QuantizationExportCli {
                output_dir: export_dir.clone(),
                limit_tensors: None,
                tensor_names: Vec::new(),
                tensor_indices: Vec::new(),
            };
            let export = export_all_linear_quantization_archive(&model_dir, &export_cli)?;
            println!(
                "Ministral quantize+eval export: exported={}/{} complete={}",
                export.exported_tensors.len(),
                export.planned_tensor_count,
                export.complete_export
            );
            println!(
                "  manifest={}",
                export.output_dir.join("manifest.json").display()
            );
            if !export.complete_export {
                return Err(invalid_data(format!(
                    "quantize+eval requires a complete archive, exported {}/{} tensors",
                    export.exported_tensors.len(),
                    export.planned_tensor_count
                )));
            }
        }
    }

    run_ministral_exported_eval_with_cli(
        &model_dir,
        &export_dir,
        max_new_tokens,
        top_k,
        logits_top_k,
        &cli,
        "Ministral quantize+eval all-linear-int8",
    )
}

#[derive(Debug)]
struct QuantizationVariantSummary {
    variant: &'static str,
    prompt_count: usize,
    token_match_count: usize,
    kl_sum: f64,
    max_kl: f64,
    max_abs_diff: f32,
}

impl QuantizationVariantSummary {
    fn new(variant: &'static str) -> Self {
        Self {
            variant,
            prompt_count: 0,
            token_match_count: 0,
            kl_sum: 0.0,
            max_kl: 0.0,
            max_abs_diff: 0.0,
        }
    }

    fn update(&mut self, result: &nn_rust_quantization::QuantizationSuiteVariantProbe) {
        self.prompt_count += 1;
        if result.token_matches {
            self.token_match_count += 1;
        }
        self.kl_sum += result.kl_divergence;
        self.max_kl = self.max_kl.max(result.kl_divergence);
        self.max_abs_diff = self.max_abs_diff.max(result.max_abs_diff);
    }

    fn mean_kl(&self) -> f64 {
        if self.prompt_count == 0 {
            0.0
        } else {
            self.kl_sum / self.prompt_count as f64
        }
    }
}

fn run_ministral_quantization_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let top_k = parse_optional_usize(args, &mut index, 3, "top_k")?.max(1);
    let cli = parse_chat_cli(args, index)?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let prompts: Vec<Vec<u32>> = cli
        .prompts
        .iter()
        .map(|prompt| tokenizer.encode_lossy(prompt, true))
        .collect::<AppResult<_>>()?;
    let (stream, module) = cuda_handles()?;
    let suite = nn_rust_quantization::run_ministral_quantization_suite_probe(
        &stream, &module, &model_dir, &prompts, top_k,
    )?;

    println!(
        "Ministral quantization suite: model_dir={} prompts_len={} top_k={}",
        model_dir.display(),
        suite.prompts.len(),
        top_k
    );

    let mut summaries = Vec::<QuantizationVariantSummary>::new();
    for (prompt_index, prompt_result) in suite.prompts.iter().enumerate() {
        let prompt_text = cli
            .prompts
            .get(prompt_index)
            .map(String::as_str)
            .unwrap_or("<tokens>");
        println!("  prompt[{prompt_index}]={prompt_text:?}");
        print_token_window("prompt_tokens", &prompt_result.tokens);

        for variant in &prompt_result.variants {
            if let Some(summary) = summaries
                .iter_mut()
                .find(|summary| summary.variant == variant.variant)
            {
                summary.update(variant);
            } else {
                let mut summary = QuantizationVariantSummary::new(variant.variant);
                summary.update(variant);
                summaries.push(summary);
            }

            println!(
                "    variant={} token_matches={} reference_token={} int8_token={} reference_logit={:.8} int8_logit={:.8} kl_divergence={:.12} max_abs_diff={:.8} mean_abs_diff={:.8}",
                variant.variant,
                variant.token_matches,
                variant.reference_token_id,
                variant.int8_token_id,
                variant.reference_logit,
                variant.int8_logit,
                variant.kl_divergence,
                variant.max_abs_diff,
                variant.mean_abs_diff
            );
            println!(
                "      reference_top_logits={:?}",
                variant.reference_top_logits
            );
            println!("      int8_top_logits={:?}", variant.int8_top_logits);
        }
    }

    println!("  summary:");
    for summary in summaries {
        println!(
            "    variant={} token_match_count={}/{} mean_kl_divergence={:.12} max_kl_divergence={:.12} max_abs_diff={:.8}",
            summary.variant,
            summary.token_match_count,
            summary.prompt_count,
            summary.mean_kl(),
            summary.max_kl,
            summary.max_abs_diff
        );
    }

    Ok(())
}

fn run_ministral_output_projection_export_probe(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = if index < args.len() && !args[index].starts_with("--") {
        let path = PathBuf::from(&args[index]);
        index += 1;
        path
    } else {
        PathBuf::from("runs/ministral_chat_trace/all_linear_int8_export_smoke")
    };
    let top_k = parse_optional_usize(args, &mut index, 3, "top_k")?;
    let tokens = parse_remaining_token_ids(args, index)?;
    let (stream, module) = cuda_handles()?;
    let result = nn_rust_quantization::run_ministral_output_projection_export_quantization_probe(
        &stream,
        &module,
        &model_dir,
        &export_dir,
        &tokens,
        top_k,
    )?;

    println!("Ministral output projection export probe:");
    println!("  model_dir={}", model_dir.display());
    println!("  export_dir={}", export_dir.display());
    println!("  tokens={:?}", result.tokens);
    println!("  rows={} cols={}", result.rows, result.cols);
    println!(
        "  top_token_matches={}",
        result.reference_top_logits.first().map(|(token, _)| token)
            == result.int8_top_logits.first().map(|(token, _)| token)
    );
    println!("  kl_divergence={:.12}", result.kl_divergence);
    println!("  max_abs_diff={:.8}", result.max_abs_diff);
    println!("  mean_abs_diff={:.8}", result.mean_abs_diff);
    println!("  reference_top_logits={:?}", result.reference_top_logits);
    println!("  int8_top_logits={:?}", result.int8_top_logits);

    Ok(())
}

fn parse_quantization_manifest_cli(
    args: &[String],
    start: usize,
) -> AppResult<QuantizationManifestCli> {
    let mut report_path = None;
    let mut index = start;

    while index < args.len() {
        match args[index].as_str() {
            "--report" => {
                let value = parse_required_flag_value(args, &mut index, "--report")?;
                report_path = Some(PathBuf::from(value));
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "unknown quantization manifest flag {flag:?}"
                )));
            }
            value => {
                return Err(invalid_input(format!(
                    "unexpected quantization manifest argument {value:?}"
                )));
            }
        }
    }

    Ok(QuantizationManifestCli { report_path })
}

fn parse_quantization_export_cli(
    args: &[String],
    start: usize,
) -> AppResult<QuantizationExportCli> {
    let mut output_dir = PathBuf::from("runs/ministral_chat_trace/all_linear_int8_export");
    let mut limit_tensors = None;
    let mut tensor_names = Vec::new();
    let mut tensor_indices = Vec::new();
    let mut index = start;

    if index < args.len() && !args[index].starts_with("--") {
        output_dir = PathBuf::from(&args[index]);
        index += 1;
    }

    while index < args.len() {
        match args[index].as_str() {
            "--limit-tensors" => {
                let value = parse_required_flag_value(args, &mut index, "--limit-tensors")?;
                let value = value.parse::<usize>()?;
                if value == 0 {
                    return Err(invalid_input("--limit-tensors must be greater than zero"));
                }
                limit_tensors = Some(value);
            }
            "--tensor" => {
                let value = parse_required_flag_value(args, &mut index, "--tensor")?;
                if value.is_empty() {
                    return Err(invalid_input("--tensor must not be empty"));
                }
                tensor_names.push(value.to_string());
            }
            "--tensor-index" => {
                let value = parse_required_flag_value(args, &mut index, "--tensor-index")?;
                tensor_indices.push(value.parse::<usize>()?);
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "unknown quantization export flag {flag:?}"
                )));
            }
            value => {
                return Err(invalid_input(format!(
                    "unexpected quantization export argument {value:?}"
                )));
            }
        }
    }

    if limit_tensors.is_some() && (!tensor_names.is_empty() || !tensor_indices.is_empty()) {
        return Err(invalid_input(
            "--limit-tensors cannot be combined with --tensor or --tensor-index",
        ));
    }

    Ok(QuantizationExportCli {
        output_dir,
        limit_tensors,
        tensor_names,
        tensor_indices,
    })
}

fn parse_weight_source_compare_cli(
    args: &[String],
    start: usize,
) -> AppResult<WeightSourceCompareCli> {
    let mut element_start = 0;
    let mut element_limit = 64;
    let mut tensor_names = Vec::new();
    let mut index = start;

    while index < args.len() {
        match args[index].as_str() {
            "--start" => {
                let value = parse_required_flag_value(args, &mut index, "--start")?;
                element_start = value.parse::<usize>()?;
            }
            "--limit" | "--prefix-elements" => {
                let value = parse_required_flag_value(args, &mut index, "--limit")?;
                element_limit = value.parse::<usize>()?;
                if element_limit == 0 {
                    return Err(invalid_input("--limit must be greater than zero"));
                }
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "unknown weight source compare flag {flag:?}"
                )));
            }
            tensor_name => {
                tensor_names.push(tensor_name.to_string());
                index += 1;
            }
        }
    }

    Ok(WeightSourceCompareCli {
        element_start,
        element_limit,
        tensor_names,
    })
}

fn default_weight_source_compare_tensor_names(config: &TextConfig) -> Vec<String> {
    let last_layer = config.n_layers.saturating_sub(1);
    vec![
        "tok_embeddings.weight".to_string(),
        "norm.weight".to_string(),
        "output.weight".to_string(),
        "layers.0.attention_norm.weight".to_string(),
        "layers.0.attention.wq.weight".to_string(),
        "layers.0.feed_forward.w1.weight".to_string(),
        format!("layers.{last_layer}.ffn_norm.weight"),
        format!("layers.{last_layer}.attention.wo.weight"),
        format!("layers.{last_layer}.feed_forward.w2.weight"),
    ]
}

fn parse_quantization_validate_cli(
    args: &[String],
    start: usize,
) -> AppResult<QuantizationValidateCli> {
    let mut export_dir = PathBuf::from("runs/ministral_chat_trace/all_linear_int8_export");
    let mut report_path = None;
    let mut index = start;

    if index < args.len() && !args[index].starts_with("--") {
        export_dir = PathBuf::from(&args[index]);
        index += 1;
    }

    while index < args.len() {
        match args[index].as_str() {
            "--report" => {
                let value = parse_required_flag_value(args, &mut index, "--report")?;
                report_path = Some(PathBuf::from(value));
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "unknown quantization validate flag {flag:?}"
                )));
            }
            value => {
                return Err(invalid_input(format!(
                    "unexpected quantization validate argument {value:?}"
                )));
            }
        }
    }

    Ok(QuantizationValidateCli {
        export_dir,
        report_path,
    })
}

fn build_all_linear_quantization_manifest(model_dir: &Path) -> AppResult<QuantizationManifest> {
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    let weights_path = weights.source_path().to_path_buf();
    let mut quantized_tensors = Vec::new();
    let q_len = config.n_heads * config.head_dim;
    let kv_len = config.n_kv_heads * config.head_dim;

    push_quantized_tensor_plan(
        &weights,
        &mut quantized_tensors,
        "output.weight",
        "output",
        &[config.vocab_size, config.dim],
    )?;
    for layer in 0..config.n_layers {
        push_quantized_tensor_plan(
            &weights,
            &mut quantized_tensors,
            &format!("layers.{layer}.attention.wq.weight"),
            "attention.wq",
            &[q_len, config.dim],
        )?;
        push_quantized_tensor_plan(
            &weights,
            &mut quantized_tensors,
            &format!("layers.{layer}.attention.wk.weight"),
            "attention.wk",
            &[kv_len, config.dim],
        )?;
        push_quantized_tensor_plan(
            &weights,
            &mut quantized_tensors,
            &format!("layers.{layer}.attention.wv.weight"),
            "attention.wv",
            &[kv_len, config.dim],
        )?;
        push_quantized_tensor_plan(
            &weights,
            &mut quantized_tensors,
            &format!("layers.{layer}.attention.wo.weight"),
            "attention.wo",
            &[config.dim, q_len],
        )?;
        push_quantized_tensor_plan(
            &weights,
            &mut quantized_tensors,
            &format!("layers.{layer}.feed_forward.w1.weight"),
            "feed_forward.w1",
            &[config.hidden_dim, config.dim],
        )?;
        push_quantized_tensor_plan(
            &weights,
            &mut quantized_tensors,
            &format!("layers.{layer}.feed_forward.w3.weight"),
            "feed_forward.w3",
            &[config.hidden_dim, config.dim],
        )?;
        push_quantized_tensor_plan(
            &weights,
            &mut quantized_tensors,
            &format!("layers.{layer}.feed_forward.w2.weight"),
            "feed_forward.w2",
            &[config.dim, config.hidden_dim],
        )?;
    }

    let mut quantized_names = HashSet::<String>::new();
    for tensor in &quantized_tensors {
        quantized_names.insert(tensor.name.clone());
        if let Some(alias) = model_tensor_alias(&tensor.name) {
            quantized_names.insert(alias);
        }
    }
    let mut preserved_tensors = Vec::new();
    for tensor in weights.tensors() {
        if !quantized_names.contains(tensor.name.as_str()) {
            preserved_tensors.push(PreservedTensorPlan {
                name: tensor.name.clone(),
                role: preserved_tensor_role(&tensor.name).to_string(),
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                source_bytes: tensor.byte_len(),
            });
        }
    }
    preserved_tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let source_quantized_bytes = quantized_tensors
        .iter()
        .map(|tensor| tensor.source_bytes)
        .sum();
    let projected_quantized_value_bytes = quantized_tensors
        .iter()
        .map(|tensor| tensor.quantized_value_bytes)
        .sum();
    let projected_quantized_scale_bytes = quantized_tensors
        .iter()
        .map(|tensor| tensor.quantized_scale_bytes)
        .sum();
    let projected_quantized_bytes = quantized_tensors
        .iter()
        .map(|tensor| tensor.quantized_bytes)
        .sum();
    let preserved_source_bytes = preserved_tensors
        .iter()
        .map(|tensor| tensor.source_bytes)
        .sum();
    let source_total_bytes = weights
        .tensors()
        .iter()
        .map(|tensor| tensor.byte_len())
        .sum();
    let projected_total_bytes = preserved_source_bytes + projected_quantized_bytes;
    let quantized_tensor_count = quantized_tensors.len();
    let preserved_tensor_count = preserved_tensors.len();

    Ok(QuantizationManifest {
        model_dir: model_dir.to_path_buf(),
        weights_path,
        n_layers: config.n_layers,
        quantized_tensors,
        preserved_tensors,
        summary: QuantizationManifestSummary {
            quantized_tensor_count,
            preserved_tensor_count,
            source_quantized_bytes,
            projected_quantized_bytes,
            projected_quantized_value_bytes,
            projected_quantized_scale_bytes,
            preserved_source_bytes,
            source_total_bytes,
            projected_total_bytes,
        },
    })
}

fn export_all_linear_quantization_archive(
    model_dir: &Path,
    cli: &QuantizationExportCli,
) -> AppResult<QuantizationExportManifest> {
    let manifest = build_all_linear_quantization_manifest(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    fs::create_dir_all(&cli.output_dir)?;

    let selected_plans = selected_quantization_export_plans(&manifest, cli)?;
    let mut exported_tensors = Vec::with_capacity(selected_plans.len());

    for (plan_index, plan) in selected_plans {
        let tensor = weights.tensor(&plan.name)?;
        validate_bf16_matrix_tensor(&tensor, &plan.shape)?;
        let weight = weights.read_bf16_tensor(&tensor)?;
        let quantized = nn_rust_quantization::RowwiseScaledI8Matrix::from_bf16_rows_symmetric(
            &weight, plan.rows, plan.cols,
        );
        let (values_path, scales_path) = nn_rust_quantization::rowwise_scaled_i8_export_file_paths(
            &cli.output_dir,
            plan_index,
            &plan.name,
        );

        write_i8_file(&values_path, &quantized.values)?;
        write_f32_file(&scales_path, &quantized.scales)?;

        exported_tensors.push(QuantizedTensorExport {
            index: plan_index,
            name: plan.name.clone(),
            role: plan.role.clone(),
            shape: plan.shape.clone(),
            rows: plan.rows,
            cols: plan.cols,
            values_bytes: quantized.values.len() as u64,
            scales_bytes: (quantized.scales.len() * f32_dtype_size_in_bytes()) as u64,
            values_path,
            scales_path,
        });
    }

    let export = QuantizationExportManifest {
        model_dir: manifest.model_dir,
        weights_path: manifest.weights_path,
        output_dir: cli.output_dir.clone(),
        planned_tensor_count: manifest.quantized_tensors.len(),
        complete_export: exported_tensors.len() == manifest.quantized_tensors.len(),
        exported_tensors,
        summary: manifest.summary,
    };
    write_quantization_export_manifest(&export.output_dir.join("manifest.json"), &export)?;

    Ok(export)
}

fn selected_quantization_export_plans<'a>(
    manifest: &'a QuantizationManifest,
    cli: &QuantizationExportCli,
) -> AppResult<Vec<(usize, &'a QuantizationTensorPlan)>> {
    if cli.tensor_names.is_empty() && cli.tensor_indices.is_empty() {
        let tensors_to_export = cli
            .limit_tensors
            .map(|limit| limit.min(manifest.quantized_tensors.len()))
            .unwrap_or(manifest.quantized_tensors.len());
        return Ok(manifest
            .quantized_tensors
            .iter()
            .take(tensors_to_export)
            .enumerate()
            .collect());
    }

    let mut seen = HashSet::new();
    let mut selected = Vec::new();

    for &plan_index in &cli.tensor_indices {
        let Some(plan) = manifest.quantized_tensors.get(plan_index) else {
            return Err(invalid_input(format!(
                "quantization export tensor index {plan_index} is out of range; planned_tensor_count={}",
                manifest.quantized_tensors.len()
            )));
        };
        if seen.insert(plan_index) {
            selected.push((plan_index, plan));
        }
    }

    for name in &cli.tensor_names {
        let Some(plan_index) = manifest
            .quantized_tensors
            .iter()
            .position(|plan| plan.name.as_str() == name.as_str())
        else {
            return Err(invalid_input(format!(
                "quantization export tensor {name:?} is not an all-linear int8 tensor"
            )));
        };
        if seen.insert(plan_index) {
            selected.push((plan_index, &manifest.quantized_tensors[plan_index]));
        }
    }

    Ok(selected)
}

fn run_ministral_weight_source_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let mut cli = parse_weight_source_compare_cli(args, index)?;
    if cli.tensor_names.is_empty() {
        let config = TextConfig::from_model_dir(&model_dir)?;
        cli.tensor_names = default_weight_source_compare_tensor_names(&config);
    }

    let consolidated = ModelWeights::open_consolidated_model_dir(&model_dir)?;
    let sharded = ModelWeights::open_sharded_model_dir(&model_dir)?;

    println!(
        "Ministral weight source compare: tensors={}",
        cli.tensor_names.len()
    );
    println!("  model_dir={}", model_dir.display());
    println!(
        "  consolidated_source={}",
        consolidated.source_path().display()
    );
    println!("  sharded_source={}", sharded.source_path().display());
    println!("  start_element={}", cli.element_start);
    println!("  prefix_elements={}", cli.element_limit);

    for name in &cli.tensor_names {
        let consolidated_tensor = consolidated.tensor(name)?;
        let sharded_tensor = sharded.tensor(name)?;
        let dtype_matches = consolidated_tensor.dtype == sharded_tensor.dtype;
        let shape_matches = consolidated_tensor.shape == sharded_tensor.shape;
        let byte_len_matches = consolidated_tensor.byte_len() == sharded_tensor.byte_len();

        println!("  tensor={name}");
        println!("    consolidated_name={}", consolidated_tensor.name);
        println!("    sharded_name={}", sharded_tensor.name);
        println!(
            "    dtype={} shape={:?} bytes={}",
            consolidated_tensor.dtype.safetensors_name(),
            consolidated_tensor.shape,
            consolidated_tensor.byte_len()
        );
        println!(
            "    dtype_matches={} shape_matches={} byte_len_matches={}",
            dtype_matches, shape_matches, byte_len_matches
        );

        if !(dtype_matches && shape_matches) {
            continue;
        }

        let prefix = compare_weight_source_tensor_prefix(
            &consolidated,
            &consolidated_tensor,
            &sharded,
            &sharded_tensor,
            cli.element_start,
            cli.element_limit,
        )?;
        println!(
            "    compared_elements={} exact_prefix_match={} max_abs_diff={:.8} mean_abs_diff={:.8}",
            prefix.compared_elements,
            prefix.exact_prefix_match,
            prefix.max_abs_diff,
            prefix.mean_abs_diff
        );
        if let Some(index) = prefix.first_mismatch {
            println!("    first_mismatch={index}");
        }
    }

    Ok(())
}

fn compare_weight_source_tensor_prefix(
    consolidated: &ModelWeights,
    consolidated_tensor: &ModelTensor<'_>,
    sharded: &ModelWeights,
    sharded_tensor: &ModelTensor<'_>,
    element_start: usize,
    element_limit: usize,
) -> AppResult<TensorPrefixComparison> {
    match consolidated_tensor.dtype {
        DType::Bf16 => {
            let element_count = consolidated_tensor.typed_element_count::<Bf16>()?;
            if element_start >= element_count {
                return Err(invalid_input(format!(
                    "start element {element_start} is out of range for tensor {} with {element_count} elements",
                    consolidated_tensor.name
                )));
            }
            let compared_elements = element_limit.min(element_count - element_start);
            let consolidated_values = consolidated.read_bf16_range(
                consolidated_tensor,
                element_start,
                compared_elements,
            )?;
            let sharded_values =
                sharded.read_bf16_range(sharded_tensor, element_start, compared_elements)?;
            Ok(compare_bf16_prefix_values(
                &consolidated_values,
                &sharded_values,
            ))
        }
        DType::F32 => {
            let element_count = consolidated_tensor.typed_element_count::<f32>()?;
            if element_start >= element_count {
                return Err(invalid_input(format!(
                    "start element {element_start} is out of range for tensor {} with {element_count} elements",
                    consolidated_tensor.name
                )));
            }
            let compared_elements = element_limit.min(element_count - element_start);
            let consolidated_values = consolidated.read_tensor_range::<f32>(
                consolidated_tensor,
                element_start,
                compared_elements,
            )?;
            let sharded_values = sharded.read_tensor_range::<f32>(
                sharded_tensor,
                element_start,
                compared_elements,
            )?;
            Ok(compare_f32_prefix_values(
                &consolidated_values,
                &sharded_values,
            ))
        }
        dtype => Err(invalid_input(format!(
            "weight source compare does not support {} tensor {}",
            dtype.safetensors_name(),
            consolidated_tensor.name
        ))),
    }
}

fn compare_bf16_prefix_values(lhs: &[Bf16], rhs: &[Bf16]) -> TensorPrefixComparison {
    compare_prefix_values(lhs.len(), |index| {
        (
            lhs[index].to_bits() == rhs[index].to_bits(),
            (lhs[index].to_f32() - rhs[index].to_f32()).abs(),
        )
    })
}

fn compare_f32_prefix_values(lhs: &[f32], rhs: &[f32]) -> TensorPrefixComparison {
    compare_prefix_values(lhs.len(), |index| {
        (
            lhs[index].to_bits() == rhs[index].to_bits(),
            (lhs[index] - rhs[index]).abs(),
        )
    })
}

fn compare_prefix_values<F>(len: usize, mut diff_at: F) -> TensorPrefixComparison
where
    F: FnMut(usize) -> (bool, f32),
{
    let mut exact_prefix_match = true;
    let mut first_mismatch = None;
    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f32;

    for index in 0..len {
        let (exact_match, abs_diff) = diff_at(index);
        if !exact_match && first_mismatch.is_none() {
            first_mismatch = Some(index);
        }
        exact_prefix_match &= exact_match;
        max_abs_diff = max_abs_diff.max(abs_diff);
        sum_abs_diff += abs_diff;
    }

    TensorPrefixComparison {
        compared_elements: len,
        exact_prefix_match,
        max_abs_diff,
        mean_abs_diff: if len == 0 {
            0.0
        } else {
            sum_abs_diff / len as f32
        },
        first_mismatch,
    }
}

fn write_i8_file(path: &Path, values: &[i8]) -> AppResult<()> {
    let bytes: Vec<u8> = values.iter().map(|value| *value as u8).collect();
    fs::write(path, bytes)?;
    Ok(())
}

fn write_f32_file(path: &Path, values: &[f32]) -> AppResult<()> {
    let mut bytes = Vec::with_capacity(values.len() * f32_dtype_size_in_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn push_quantized_tensor_plan(
    weights: &ModelWeights,
    out: &mut Vec<QuantizationTensorPlan>,
    name: &str,
    role: &str,
    expected_shape: &[usize],
) -> AppResult<()> {
    let tensor = weights.tensor(name)?;
    validate_bf16_matrix_tensor(&tensor, expected_shape)?;
    let rows = expected_shape[0];
    let cols = expected_shape[1];
    let quantized_value_bytes = tensor.element_count() as u64;
    let quantized_scale_bytes = rows as u64 * f32_dtype_size_in_bytes() as u64;

    out.push(QuantizationTensorPlan {
        name: name.to_string(),
        role: role.to_string(),
        dtype: tensor.dtype,
        shape: tensor.shape.clone(),
        rows,
        cols,
        source_bytes: tensor.byte_len(),
        quantized_value_bytes,
        quantized_scale_bytes,
        quantized_bytes: quantized_value_bytes + quantized_scale_bytes,
    });
    Ok(())
}

fn validate_bf16_matrix_tensor(tensor: &TensorInfo, expected_shape: &[usize]) -> AppResult<()> {
    if tensor.dtype != DType::Bf16 {
        return Err(invalid_data(format!(
            "tensor {} has dtype {}, expected BF16",
            tensor.name,
            tensor.dtype.safetensors_name()
        )));
    }
    if tensor.shape != expected_shape {
        return Err(invalid_data(format!(
            "tensor {} has shape {:?}, expected {:?}",
            tensor.name, tensor.shape, expected_shape
        )));
    }
    if tensor.shape.len() != 2 {
        return Err(invalid_data(format!(
            "tensor {} is not a matrix: {:?}",
            tensor.name, tensor.shape
        )));
    }
    Ok(())
}

fn preserved_tensor_role(name: &str) -> &'static str {
    if name == "tok_embeddings.weight" || name == "language_model.model.embed_tokens.weight" {
        "token_embedding"
    } else if name == "norm.weight" || name == "language_model.model.norm.weight" {
        "final_norm"
    } else if name.ends_with(".attention_norm.weight") || name.ends_with(".input_layernorm.weight")
    {
        "attention_norm"
    } else if name.ends_with(".ffn_norm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
    {
        "ffn_norm"
    } else if name.starts_with("vision_encoder.") {
        "vision_encoder_preserved_text_runtime"
    } else if name.starts_with("patch_merger.") {
        "vision_patch_merger_preserved_text_runtime"
    } else if name == "pre_mm_projector_norm.weight" {
        "vision_projector_norm_preserved_text_runtime"
    } else if name.starts_with("vision_language_adapter.") {
        "vision_language_adapter_preserved_text_runtime"
    } else {
        "preserved_other"
    }
}

fn ratio_u64(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn run_ministral_eval(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let trace_options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let text_eval_suite = inference::run_ministral_text_generation_backend_eval_suite_with_driver(
        stream.clone(),
        module.clone(),
        &model_dir,
        &cli.prompts,
        candidate_backend,
        &trace_options,
        cli.driver,
    )?;
    let chat_trace_options = ChatTraceComparisonSuiteOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let chat_eval_suite = inference::run_ministral_single_turn_chat_backend_eval_suite(
        stream.clone(),
        module.clone(),
        &model_dir,
        &cli.prompts,
        candidate_backend,
        &chat_trace_options,
        cli.driver,
    )?;
    let text_forced_target_suite = if cli.forced_target_pairs.is_empty() {
        None
    } else {
        Some(
            inference::run_ministral_text_generation_backend_forced_target_comparison_suite(
                stream.clone(),
                module.clone(),
                &model_dir,
                &cli.forced_target_pairs,
                candidate_backend,
                logits_top_k,
            )?,
        )
    };
    let chat_forced_target_options = ChatForcedLogitsTraceOptions {
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
    };
    let chat_forced_target_suite = if cli.forced_target_pairs.is_empty() {
        None
    } else {
        Some(
            inference::run_ministral_single_turn_chat_backend_forced_target_comparison_suite(
                stream,
                module,
                &model_dir,
                &cli.forced_target_pairs,
                candidate_backend,
                &chat_forced_target_options,
            )?,
        )
    };
    let text_generation_summary =
        chat_compare_summary_from_generation_suite(&text_eval_suite.generation.summary);
    let chat_generation_summary =
        chat_compare_summary_from_generation_suite(&chat_eval_suite.generation.summary);
    let text_forced_target_summary = text_forced_target_suite
        .as_ref()
        .map(|suite| chat_compare_summary_from_forced_suite(&suite.summary));
    let chat_forced_target_summary = chat_forced_target_suite
        .as_ref()
        .map(|suite| chat_compare_summary_from_forced_suite(&suite.summary));

    println!(
        "Ministral eval: candidate={} driver={} prompts_len={} forced_target_pairs_len={}",
        backend_label(candidate_backend),
        driver_label(cli.driver),
        cli.prompts.len(),
        cli.forced_target_pairs.len()
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    print_logits_summary("text_logits", &text_eval_suite.logits.summary);
    print_compare_suite_summary("text_generation", &text_generation_summary);
    print_logits_summary("chat_logits", &chat_eval_suite.logits.summary);
    print_compare_suite_summary("chat_generation", &chat_generation_summary);
    if let (Some(suite), Some(summary)) = (
        text_forced_target_suite.as_ref(),
        text_forced_target_summary.as_ref(),
    ) {
        print_compare_suite_summary("text_forced_target", summary);
        print_forced_suite_extra_summary("text_forced_target", &suite.summary);
    }
    if let (Some(suite), Some(summary)) = (
        chat_forced_target_suite.as_ref(),
        chat_forced_target_summary.as_ref(),
    ) {
        print_compare_suite_summary("chat_forced_target", summary);
        print_forced_suite_extra_summary("chat_forced_target", &suite.summary);
    }

    if let Some(path) = &cli.report_path {
        write_ministral_eval_report(
            path,
            &model_dir,
            candidate_backend,
            None,
            cli.driver,
            &cli.system_prompt,
            max_new_tokens,
            top_k,
            logits_top_k,
            &cli.prompt_file_paths,
            &cli.pair_file_paths,
            &cli.thresholds,
            &text_eval_suite,
            &text_generation_summary,
            &chat_eval_suite,
            &chat_generation_summary,
            text_forced_target_suite.as_ref(),
            chat_forced_target_suite.as_ref(),
        )?;
        println!("report={}", path.display());
    }

    enforce_token_logits_compare_thresholds(&text_eval_suite.logits.summary, &cli.thresholds)?;
    enforce_chat_compare_thresholds(&text_generation_summary, &cli.thresholds)?;
    enforce_token_logits_compare_thresholds(&chat_eval_suite.logits.summary, &cli.thresholds)?;
    enforce_chat_compare_thresholds(&chat_generation_summary, &cli.thresholds)?;
    if let Some(summary) = &text_forced_target_summary {
        enforce_chat_compare_thresholds(summary, &cli.thresholds)?;
    }
    if let Some(summary) = &chat_forced_target_summary {
        enforce_chat_compare_thresholds(summary, &cli.thresholds)?;
    }

    Ok(())
}

fn run_ministral_eval_exported(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;

    run_ministral_exported_eval_with_cli(
        &model_dir,
        &export_dir,
        max_new_tokens,
        top_k,
        logits_top_k,
        &cli,
        "Ministral exported all-linear-int8 eval",
    )
}

fn run_ministral_exported_eval_with_cli(
    model_dir: &Path,
    export_dir: &Path,
    max_new_tokens: usize,
    top_k: usize,
    logits_top_k: usize,
    cli: &ChatCli,
    label: &str,
) -> AppResult<()> {
    let (stream, module) = cuda_handles()?;
    let trace_options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let text_eval_suite =
        inference::run_ministral_text_generation_exported_all_linear_int8_eval_suite_with_driver(
            stream.clone(),
            module.clone(),
            &model_dir,
            &export_dir,
            &cli.prompts,
            &trace_options,
            cli.driver,
        )?;
    let chat_trace_options = ChatTraceComparisonSuiteOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let chat_eval_suite =
        inference::run_ministral_single_turn_chat_exported_all_linear_int8_eval_suite(
            stream.clone(),
            module.clone(),
            &model_dir,
            &export_dir,
            &cli.prompts,
            &chat_trace_options,
            cli.driver,
        )?;
    let text_forced_target_suite = if cli.forced_target_pairs.is_empty() {
        None
    } else {
        Some(
            inference::run_ministral_text_generation_exported_all_linear_int8_forced_target_comparison_suite(
                stream.clone(),
                module.clone(),
                &model_dir,
                &export_dir,
                &cli.forced_target_pairs,
                logits_top_k,
            )?,
        )
    };
    let chat_forced_target_options = ChatForcedLogitsTraceOptions {
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
    };
    let chat_forced_target_suite = if cli.forced_target_pairs.is_empty() {
        None
    } else {
        Some(
            inference::run_ministral_single_turn_chat_exported_all_linear_int8_forced_target_comparison_suite(
                stream,
                module,
                &model_dir,
                &export_dir,
                &cli.forced_target_pairs,
                &chat_forced_target_options,
            )?,
        )
    };
    let text_generation_summary =
        chat_compare_summary_from_generation_suite(&text_eval_suite.generation.summary);
    let chat_generation_summary =
        chat_compare_summary_from_generation_suite(&chat_eval_suite.generation.summary);
    let text_forced_target_summary = text_forced_target_suite
        .as_ref()
        .map(|suite| chat_compare_summary_from_forced_suite(&suite.summary));
    let chat_forced_target_summary = chat_forced_target_suite
        .as_ref()
        .map(|suite| chat_compare_summary_from_forced_suite(&suite.summary));

    println!(
        "{label}: export_dir={} driver={} prompts_len={} forced_target_pairs_len={}",
        export_dir.display(),
        driver_label(cli.driver),
        cli.prompts.len(),
        cli.forced_target_pairs.len()
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    print_logits_summary("text_logits", &text_eval_suite.logits.summary);
    print_compare_suite_summary("text_generation", &text_generation_summary);
    print_logits_summary("chat_logits", &chat_eval_suite.logits.summary);
    print_compare_suite_summary("chat_generation", &chat_generation_summary);
    if let (Some(suite), Some(summary)) = (
        text_forced_target_suite.as_ref(),
        text_forced_target_summary.as_ref(),
    ) {
        print_compare_suite_summary("text_forced_target", summary);
        print_forced_suite_extra_summary("text_forced_target", &suite.summary);
    }
    if let (Some(suite), Some(summary)) = (
        chat_forced_target_suite.as_ref(),
        chat_forced_target_summary.as_ref(),
    ) {
        print_compare_suite_summary("chat_forced_target", summary);
        print_forced_suite_extra_summary("chat_forced_target", &suite.summary);
    }

    if let Some(path) = &cli.report_path {
        write_ministral_eval_report(
            path,
            &model_dir,
            InferenceBackend::AllLinearInt8,
            Some(&export_dir),
            cli.driver,
            &cli.system_prompt,
            max_new_tokens,
            top_k,
            logits_top_k,
            &cli.prompt_file_paths,
            &cli.pair_file_paths,
            &cli.thresholds,
            &text_eval_suite,
            &text_generation_summary,
            &chat_eval_suite,
            &chat_generation_summary,
            text_forced_target_suite.as_ref(),
            chat_forced_target_suite.as_ref(),
        )?;
        println!("report={}", path.display());
    }

    enforce_token_logits_compare_thresholds(&text_eval_suite.logits.summary, &cli.thresholds)?;
    enforce_chat_compare_thresholds(&text_generation_summary, &cli.thresholds)?;
    enforce_token_logits_compare_thresholds(&chat_eval_suite.logits.summary, &cli.thresholds)?;
    enforce_chat_compare_thresholds(&chat_generation_summary, &cli.thresholds)?;
    if let Some(summary) = &text_forced_target_summary {
        enforce_chat_compare_thresholds(summary, &cli.thresholds)?;
    }
    if let Some(summary) = &chat_forced_target_summary {
        enforce_chat_compare_thresholds(summary, &cli.thresholds)?;
    }

    Ok(())
}

fn run_ministral_chat_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let backend = parse_optional_backend(args, &mut index, InferenceBackend::Bf16)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatInferenceOptions {
        max_new_tokens,
        top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite = if let Some(sampling) = cli.sampling {
        inference::run_ministral_single_turn_chat_sampled_suite_with_backend(
            stream,
            module,
            &model_dir,
            &cli.prompts,
            backend,
            &options,
            sampling,
        )?
    } else {
        inference::run_ministral_single_turn_chat_suite_with_backend(
            stream,
            module,
            &model_dir,
            &cli.prompts,
            backend,
            &options,
        )?
    };

    println!(
        "Ministral chat suite: backend={} decode_strategy={} prompts_len={}",
        backend_label(backend),
        decode_strategy_label(cli.sampling),
        suite.prompts.len()
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    for (index, prompt_result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", prompt_result.user_prompt);
        print_chat_result(&prompt_result.result, &cli.system_prompt);
    }

    if let Some(path) = &cli.report_path {
        let report_results: Vec<_> = suite
            .prompts
            .into_iter()
            .map(|prompt| (prompt.user_prompt, prompt.result))
            .collect();
        write_chat_generation_report(
            path,
            &report_results,
            &InferenceReportMetadata::new(&model_dir, backend)
                .with_decode_strategy(decode_strategy_label(cli.sampling))
                .with_system_prompt(&cli.system_prompt)
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_chat_exported_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatInferenceOptions {
        max_new_tokens,
        top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite = if let Some(sampling) = cli.sampling {
        inference::run_ministral_single_turn_chat_exported_all_linear_int8_sampled_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            &options,
            sampling,
        )?
    } else {
        inference::run_ministral_single_turn_chat_exported_all_linear_int8_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            &options,
        )?
    };

    println!(
        "Ministral chat exported all-linear-int8 suite: export_dir={} decode_strategy={} prompts_len={}",
        export_dir.display(),
        decode_strategy_label(cli.sampling),
        suite.prompts.len()
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    for (index, prompt_result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", prompt_result.user_prompt);
        print_chat_result(&prompt_result.result, &cli.system_prompt);
    }

    if let Some(path) = &cli.report_path {
        let report_results: Vec<_> = suite
            .prompts
            .into_iter()
            .map(|prompt| (prompt.user_prompt, prompt.result))
            .collect();
        write_chat_generation_report(
            path,
            &report_results,
            &InferenceReportMetadata::new(&model_dir, InferenceBackend::AllLinearInt8)
                .with_export_dir(&export_dir)
                .with_decode_strategy(decode_strategy_label(cli.sampling))
                .with_system_prompt(&cli.system_prompt)
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_text_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let backend = parse_optional_backend(args, &mut index, InferenceBackend::Bf16)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = if let Some(sampling) = cli.sampling {
        inference::run_ministral_text_generation_sampled_suite_with_backend(
            stream,
            module,
            &model_dir,
            &cli.prompts,
            backend,
            max_new_tokens,
            top_k,
            model_eos_token_id(&model_dir)?,
            sampling,
        )?
    } else {
        inference::run_ministral_text_generation_suite_with_backend(
            stream,
            module,
            &model_dir,
            &cli.prompts,
            backend,
            max_new_tokens,
            top_k,
            model_eos_token_id(&model_dir)?,
        )?
    };

    println!(
        "Ministral text suite: backend={} decode_strategy={} prompts_len={}",
        backend_label(backend),
        decode_strategy_label(cli.sampling),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_text);
        print_text_result(result);
    }

    if let Some(path) = &cli.report_path {
        write_text_generation_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, backend)
                .with_decode_strategy(decode_strategy_label(cli.sampling))
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_text_exported_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = if let Some(sampling) = cli.sampling {
        inference::run_ministral_text_generation_exported_all_linear_int8_sampled_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            max_new_tokens,
            top_k,
            model_eos_token_id(&model_dir)?,
            sampling,
        )?
    } else {
        inference::run_ministral_text_generation_exported_all_linear_int8_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            max_new_tokens,
            top_k,
            model_eos_token_id(&model_dir)?,
        )?
    };

    println!(
        "Ministral text exported all-linear-int8 suite: export_dir={} decode_strategy={} prompts_len={}",
        export_dir.display(),
        decode_strategy_label(cli.sampling),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_text);
        print_text_result(result);
    }

    if let Some(path) = &cli.report_path {
        write_text_generation_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, InferenceBackend::AllLinearInt8)
                .with_export_dir(&export_dir)
                .with_decode_strategy(decode_strategy_label(cli.sampling))
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_text_logits_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let backend = parse_optional_backend(args, &mut index, InferenceBackend::Bf16)?;
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = inference::run_ministral_text_generation_logits_suite_with_backend(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        backend,
        top_k,
    )?;

    println!(
        "Ministral text logits suite: backend={} prompts_len={}",
        backend_label(backend),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_text);
        print_text_logits_result(result);
    }

    if let Some(path) = &cli.report_path {
        write_text_logits_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, backend).with_top_k(top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_text_exported_logits_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = inference::run_ministral_text_generation_exported_all_linear_int8_logits_suite(
        stream,
        module,
        &model_dir,
        &export_dir,
        &cli.prompts,
        top_k,
    )?;

    println!(
        "Ministral text exported all-linear-int8 logits suite: export_dir={} prompts_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_text);
        print_text_logits_result(result);
    }

    if let Some(path) = &cli.report_path {
        write_text_logits_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, InferenceBackend::AllLinearInt8)
                .with_export_dir(&export_dir)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_text_trace_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let backend = parse_optional_backend(args, &mut index, InferenceBackend::Bf16)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite = inference::run_ministral_text_generation_logits_trace_suite_with_backend(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        backend,
        &options,
    )?;

    println!(
        "Ministral text trace suite: backend={} prompts_len={}",
        backend_label(backend),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_text);
        print_text_trace_result(result);
    }

    if let Some(path) = &cli.report_path {
        write_text_logits_trace_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, backend)
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k)
                .with_logits_top_k(logits_top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_text_exported_trace_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite =
        inference::run_ministral_text_generation_exported_all_linear_int8_logits_trace_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            &options,
        )?;

    println!(
        "Ministral text exported all-linear-int8 trace suite: export_dir={} prompts_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_text);
        print_text_trace_result(result);
    }

    if let Some(path) = &cli.report_path {
        write_text_logits_trace_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, InferenceBackend::AllLinearInt8)
                .with_export_dir(&export_dir)
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k)
                .with_logits_top_k(logits_top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_text_logits_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = inference::run_ministral_text_generation_backend_next_logits_comparison_suite(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        candidate_backend,
        top_k,
    )?;

    println!(
        "Ministral text logits comparison: candidate={} prompts_len={}",
        backend_label(candidate_backend),
        suite.prompts.len()
    );
    println!(
        "  token_match_count={}/{}",
        suite.summary.token_match_count, suite.summary.prompt_count
    );
    println!("  top_tokens_match={}", suite.summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", suite.summary.mean_kl());
    println!("  max_kl_divergence={:.12}", suite.summary.max_kl);
    println!("  max_abs_diff={:.8}", suite.summary.max_abs_diff);
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_text);
        print_text_logits_comparison_result(result);
    }

    if let Some(path) = &cli.report_path {
        write_text_logits_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
        )?;
        println!("report={}", path.display());
    }
    enforce_token_logits_compare_thresholds(&suite.summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_text_exported_logits_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite =
        inference::run_ministral_text_generation_exported_all_linear_int8_next_logits_comparison_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            top_k,
        )?;

    println!(
        "Ministral text exported all-linear-int8 logits comparison: export_dir={} prompts_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    println!(
        "  token_match_count={}/{}",
        suite.summary.token_match_count, suite.summary.prompt_count
    );
    println!("  top_tokens_match={}", suite.summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", suite.summary.mean_kl());
    println!("  max_kl_divergence={:.12}", suite.summary.max_kl);
    println!("  max_abs_diff={:.8}", suite.summary.max_abs_diff);
    println!(
        "  reference_memory_bytes={} candidate_memory_bytes={}",
        suite.reference_memory_stats.total_resident_bytes,
        suite.candidate_memory_stats.total_resident_bytes
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_text);
        print_text_logits_comparison_result(result);
    }

    if let Some(path) = &cli.report_path {
        write_text_logits_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
        )?;
        println!("report={}", path.display());
    }
    enforce_token_logits_compare_thresholds(&suite.summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_text_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite = inference::run_ministral_text_generation_backend_comparison_suite_with_driver(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        candidate_backend,
        &options,
        cli.driver,
    )?;
    let mut suite_summary = ChatCompareSuiteSummary::default();
    for result in &suite.prompts {
        suite_summary.observe(&result.comparison.summary());
    }

    println!(
        "Ministral text generation comparison: candidate={} driver={} prompts_len={}",
        backend_label(candidate_backend),
        driver_label(cli.driver),
        suite.prompts.len()
    );
    println!(
        "  token_match_count={}/{}",
        suite_summary.token_match_count, suite_summary.step_count
    );
    println!("  top_tokens_match={}", suite_summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", suite_summary.mean_kl());
    println!("  max_kl_divergence={:.12}", suite_summary.max_kl);
    println!("  max_abs_diff={:.8}", suite_summary.max_abs_diff);
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_text);
        print_text_comparison_result(result);
    }

    if let Some(path) = &cli.report_path {
        write_text_compare_report(
            path,
            &suite.prompts,
            &suite_summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_text_exported_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite =
        inference::run_ministral_text_generation_exported_all_linear_int8_comparison_suite_with_driver(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            &options,
            cli.driver,
        )?;
    let mut suite_summary = ChatCompareSuiteSummary::default();
    for result in &suite.prompts {
        suite_summary.observe(&result.comparison.summary());
    }

    println!(
        "Ministral text exported all-linear-int8 comparison: export_dir={} driver={} prompts_len={}",
        export_dir.display(),
        driver_label(cli.driver),
        suite.prompts.len()
    );
    println!(
        "  token_match_count={}/{}",
        suite_summary.token_match_count, suite_summary.step_count
    );
    println!("  top_tokens_match={}", suite_summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", suite_summary.mean_kl());
    println!("  max_kl_divergence={:.12}", suite_summary.max_kl);
    println!("  max_abs_diff={:.8}", suite_summary.max_abs_diff);
    println!(
        "  reference_memory_bytes={} candidate_memory_bytes={}",
        suite.reference_memory_stats.total_resident_bytes,
        suite.candidate_memory_stats.total_resident_bytes
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_text);
        print_text_comparison_result(result);
    }

    if let Some(path) = &cli.report_path {
        write_text_compare_report(
            path,
            &suite.prompts,
            &suite_summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_text_forced_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let logits_top_k = parse_optional_usize(args, &mut index, 8, "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = inference::run_ministral_text_generation_backend_forced_comparison_suite(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        candidate_backend,
        logits_top_k,
    )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let suite_summary = chat_compare_summary_from_forced_suite(&suite.summary);

    println!(
        "Ministral text forced comparison: candidate={} sequences_len={}",
        backend_label(candidate_backend),
        suite.prompts.len()
    );
    print_compare_suite_summary("text_forced", &suite_summary);
    println!(
        "  forced_reference_match_count={}/{}",
        suite.summary.summary.forced_reference_match_count, suite.summary.summary.step_count
    );
    println!(
        "  forced_candidate_match_count={}/{}",
        suite.summary.summary.forced_candidate_match_count, suite.summary.summary.step_count
    );
    println!(
        "  mean_forced_logprob_abs_diff={:.12}",
        suite.summary.summary.mean_forced_logprob_abs_diff()
    );
    println!(
        "  max_forced_logprob_abs_diff={:.8}",
        suite.summary.summary.max_forced_logprob_abs_diff
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  sequence[{index}]={:?}", result.sequence_text);
        print_text_forced_comparison_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_text_forced_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_text_forced_target_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let logits_top_k = parse_optional_usize(args, &mut index, 8, "logits_top_k")?;
    let cli = parse_text_forced_target_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = inference::run_ministral_text_generation_backend_forced_target_comparison_suite(
        stream,
        module,
        &model_dir,
        &cli.pairs,
        candidate_backend,
        logits_top_k,
    )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let suite_summary = chat_compare_summary_from_forced_suite(&suite.summary);

    println!(
        "Ministral text forced target comparison: candidate={} pairs_len={}",
        backend_label(candidate_backend),
        suite.prompts.len()
    );
    print_compare_suite_summary("text_forced_target", &suite_summary);
    println!(
        "  forced_reference_match_count={}/{}",
        suite.summary.summary.forced_reference_match_count, suite.summary.summary.step_count
    );
    println!(
        "  forced_candidate_match_count={}/{}",
        suite.summary.summary.forced_candidate_match_count, suite.summary.summary.step_count
    );
    println!(
        "  mean_forced_logprob_abs_diff={:.12}",
        suite.summary.summary.mean_forced_logprob_abs_diff()
    );
    println!(
        "  max_forced_logprob_abs_diff={:.8}",
        suite.summary.summary.max_forced_logprob_abs_diff
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  pair[{index}] prompt={:?}", result.prompt_text);
        println!("  pair[{index}] target={:?}", result.target_text);
        print_text_forced_target_comparison_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_text_forced_target_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.pair_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_text_forced_target_exported_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let logits_top_k = parse_optional_usize(args, &mut index, 8, "logits_top_k")?;
    let cli = parse_text_forced_target_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite =
        inference::run_ministral_text_generation_exported_all_linear_int8_forced_target_comparison_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.pairs,
            logits_top_k,
        )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let suite_summary = chat_compare_summary_from_forced_suite(&suite.summary);

    println!(
        "Ministral text exported all-linear-int8 forced target comparison: export_dir={} pairs_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    print_compare_suite_summary("text_forced_target", &suite_summary);
    println!(
        "  forced_reference_match_count={}/{}",
        suite.summary.summary.forced_reference_match_count, suite.summary.summary.step_count
    );
    println!(
        "  forced_candidate_match_count={}/{}",
        suite.summary.summary.forced_candidate_match_count, suite.summary.summary.step_count
    );
    println!(
        "  mean_forced_logprob_abs_diff={:.12}",
        suite.summary.summary.mean_forced_logprob_abs_diff()
    );
    println!(
        "  max_forced_logprob_abs_diff={:.8}",
        suite.summary.summary.max_forced_logprob_abs_diff
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  pair[{index}] prompt={:?}", result.prompt_text);
        println!("  pair[{index}] target={:?}", result.target_text);
        print_text_forced_target_comparison_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_text_forced_target_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.pair_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_chat_forced_target_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let logits_top_k = parse_optional_usize(args, &mut index, 8, "logits_top_k")?;
    let cli = parse_chat_forced_target_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatForcedLogitsTraceOptions {
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
    };
    let suite = inference::run_ministral_single_turn_chat_backend_forced_target_comparison_suite(
        stream,
        module,
        &model_dir,
        &cli.pairs,
        candidate_backend,
        &options,
    )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let suite_summary = chat_compare_summary_from_forced_suite(&suite.summary);

    println!(
        "Ministral chat forced target comparison: candidate={} pairs_len={} system_prompt_mode={}",
        backend_label(candidate_backend),
        suite.prompts.len(),
        system_prompt_mode_label(&cli.system_prompt)
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    print_compare_suite_summary("chat_forced_target", &suite_summary);
    println!(
        "  forced_reference_match_count={}/{}",
        suite.summary.summary.forced_reference_match_count, suite.summary.summary.step_count
    );
    println!(
        "  forced_candidate_match_count={}/{}",
        suite.summary.summary.forced_candidate_match_count, suite.summary.summary.step_count
    );
    println!(
        "  mean_forced_logprob_abs_diff={:.12}",
        suite.summary.summary.mean_forced_logprob_abs_diff()
    );
    println!(
        "  max_forced_logprob_abs_diff={:.8}",
        suite.summary.summary.max_forced_logprob_abs_diff
    );
    for (index, prompt_result) in suite.prompts.iter().enumerate() {
        println!(
            "  pair[{index}] user_prompt={:?}",
            prompt_result.user_prompt
        );
        println!(
            "  pair[{index}] target={:?}",
            prompt_result.result.target_text
        );
        print_chat_forced_target_comparison_result(&prompt_result.result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_chat_forced_target_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.pair_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_chat_forced_target_exported_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let logits_top_k = parse_optional_usize(args, &mut index, 8, "logits_top_k")?;
    let cli = parse_chat_forced_target_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatForcedLogitsTraceOptions {
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
    };
    let suite =
        inference::run_ministral_single_turn_chat_exported_all_linear_int8_forced_target_comparison_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.pairs,
            &options,
        )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let suite_summary = chat_compare_summary_from_forced_suite(&suite.summary);

    println!(
        "Ministral chat exported all-linear-int8 forced target comparison: export_dir={} pairs_len={} system_prompt_mode={}",
        export_dir.display(),
        suite.prompts.len(),
        system_prompt_mode_label(&cli.system_prompt)
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    print_compare_suite_summary("chat_forced_target", &suite_summary);
    println!(
        "  forced_reference_match_count={}/{}",
        suite.summary.summary.forced_reference_match_count, suite.summary.summary.step_count
    );
    println!(
        "  forced_candidate_match_count={}/{}",
        suite.summary.summary.forced_candidate_match_count, suite.summary.summary.step_count
    );
    println!(
        "  mean_forced_logprob_abs_diff={:.12}",
        suite.summary.summary.mean_forced_logprob_abs_diff()
    );
    println!(
        "  max_forced_logprob_abs_diff={:.8}",
        suite.summary.summary.max_forced_logprob_abs_diff
    );
    for (index, prompt_result) in suite.prompts.iter().enumerate() {
        println!(
            "  pair[{index}] user_prompt={:?}",
            prompt_result.user_prompt
        );
        println!(
            "  pair[{index}] target={:?}",
            prompt_result.result.target_text
        );
        print_chat_forced_target_comparison_result(&prompt_result.result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_chat_forced_target_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.pair_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_tokens_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let backend = parse_optional_backend(args, &mut index, InferenceBackend::Bf16)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let cli = parse_token_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = if let Some(sampling) = cli.sampling {
        inference::run_ministral_generation_sampled_suite_with_backend(
            stream,
            module,
            &model_dir,
            &cli.prompts,
            backend,
            max_new_tokens,
            top_k,
            model_eos_token_id(&model_dir)?,
            sampling,
        )?
    } else {
        inference::run_ministral_generation_suite_with_backend(
            stream,
            module,
            &model_dir,
            &cli.prompts,
            backend,
            max_new_tokens,
            top_k,
            model_eos_token_id(&model_dir)?,
        )?
    };
    let tokenizer = TekkenTokenizer::open(&model_dir)?;

    println!(
        "Ministral token suite: backend={} decode_strategy={} prompts_len={}",
        backend_label(backend),
        decode_strategy_label(cli.sampling),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_generation_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, backend)
                .with_decode_strategy(decode_strategy_label(cli.sampling))
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_qwen_weight_smoke(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_optional_non_numeric_model_dir(args, &mut index, DEFAULT_QWEN3_6_27B_DIR);
    if index != args.len() {
        return Err(invalid_input(
            "qwen-weight-smoke accepts at most [model_dir]",
        ));
    }

    let report = qwen35_weight_layout_report(&model_dir)?;
    let config = report.config;
    println!(
        "Qwen3.5 weight layout smoke passed: model_dir={} source={}",
        model_dir.display(),
        report.source_path.display()
    );
    println!(
        "  dim={} hidden_dim={} layers={} full_attention_layers={} linear_attention_layers={} heads={} kv_heads={} head_dim={} rotary_dim={} vocab={} eos={:?}",
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.layer_kinds.full_attention_count(),
        config.layer_kinds.linear_attention_count(),
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.rotary_dim,
        config.vocab_size,
        config.eos_token_id
    );
    if let Some(params) = config.qwen3_5 {
        println!(
            "  linear_attention key_heads={} value_heads={} key_head_dim={} value_head_dim={} qkv_dim={} value_dim={} conv_kernel={}",
            params.linear_num_key_heads,
            params.linear_num_value_heads,
            params.linear_key_head_dim,
            params.linear_value_head_dim,
            params.qkv_dim(),
            params.value_dim(),
            params.linear_conv_kernel_dim
        );
    }
    println!(
        "  embedding {}",
        qwen35_tensor_layout_label(&report.embedding)
    );
    println!("  norm {}", qwen35_tensor_layout_label(&report.norm));
    println!("  output {}", qwen35_tensor_layout_label(&report.output));

    if let Some(layer) = report
        .layers
        .iter()
        .find(|layer| matches!(layer.attention, Qwen35AttentionWeightLayout::Linear(_)))
    {
        println!("  first_linear_layer={}", layer.layer);
        if let Qwen35AttentionWeightLayout::Linear(attn) = &layer.attention {
            println!("    {}", qwen35_tensor_layout_label(&attn.in_proj_qkv));
            println!("    {}", qwen35_tensor_layout_label(&attn.conv1d));
            println!("    {}", qwen35_tensor_layout_label(&attn.out_proj));
        }
    }

    if let Some(layer) = report
        .layers
        .iter()
        .find(|layer| matches!(layer.attention, Qwen35AttentionWeightLayout::Full(_)))
    {
        println!("  first_full_attention_layer={}", layer.layer);
        if let Qwen35AttentionWeightLayout::Full(attn) = &layer.attention {
            println!("    {}", qwen35_tensor_layout_label(&attn.q_proj));
            println!("    {}", qwen35_tensor_layout_label(&attn.k_proj));
            println!("    {}", qwen35_tensor_layout_label(&attn.o_proj));
        }
    }

    Ok(())
}

fn qwen35_tensor_layout_label(tensor: &nn_rust_inference::model::Qwen35TensorLayout) -> String {
    format!(
        "{} dtype={} shape={:?} bytes={}",
        tensor.name,
        tensor.dtype.safetensors_name(),
        tensor.shape,
        tensor.byte_len
    )
}

fn run_qwen_layer_load_smoke(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_optional_non_numeric_model_dir(args, &mut index, DEFAULT_QWEN3_6_27B_DIR);
    let layer = parse_optional_usize(args, &mut index, 0, "layer")?;
    let (stream, module) = cuda_handles().map_err(|error| {
        invalid_input(format!(
            "qwen-layer-load-smoke CUDA initialization failed: {error}"
        ))
    })?;
    let mut execute_full = false;
    let mut token_id = 248044_u32;
    let mut prefix_len = 8_usize;
    while index < args.len() {
        match args[index].as_str() {
            "runfull" | "execute-full" | "--execute-full" => {
                execute_full = true;
                index += 1;
            }
            "--token" => {
                let value = parse_required_flag_value(args, &mut index, "--token")?;
                token_id = value.parse::<u32>().map_err(|error| {
                    invalid_input(format!(
                        "qwen-layer-load-smoke --token must be a u32, got {value:?}: {error}"
                    ))
                })?;
            }
            "--prefix" => {
                let value = parse_required_flag_value(args, &mut index, "--prefix")?;
                prefix_len = value.parse::<usize>().map_err(|error| {
                    invalid_input(format!(
                        "qwen-layer-load-smoke --prefix must be a usize, got {value:?}: {error}"
                    ))
                })?;
            }
            other => {
                return Err(invalid_input(format!(
                    "unexpected qwen-layer-load-smoke argument {other:?}; expected runfull, --token, or --prefix"
                )));
            }
        }
    }

    if execute_full {
        let smoke =
            qwen35_full_layer_smoke(&stream, &module, &model_dir, layer, token_id, 0, prefix_len)?;
        println!(
            "Qwen3.5 full layer smoke passed: model_dir={} layer={} token_id={} position={} output_max_abs={:.8}",
            model_dir.display(),
            smoke.layer,
            smoke.token_id,
            smoke.position,
            smoke.output_max_abs
        );
        println!("  output_prefix={:?}", smoke.output_prefix);
        return Ok(());
    }

    let smoke = qwen35_load_layer_smoke(&stream, &model_dir, layer)?;
    println!(
        "Qwen3.5 layer load smoke passed: model_dir={} layer={} kind={} host_weight_bytes={} device_weight_bytes={}",
        model_dir.display(),
        smoke.layer,
        text_layer_kind_label(smoke.kind),
        smoke.host_weight_bytes,
        smoke.device_weight_bytes
    );
    Ok(())
}

fn run_qwen_full_layer_smoke(args: &[String]) -> AppResult<()> {
    let (stream, module) = cuda_handles().map_err(|error| {
        invalid_input(format!(
            "qwen-full-layer-smoke CUDA initialization failed: {error}"
        ))
    })?;
    let mut index = 0;
    let model_dir = parse_optional_non_numeric_model_dir(args, &mut index, DEFAULT_QWEN3_6_27B_DIR);
    let layer = parse_optional_usize(args, &mut index, 3, "layer")?;
    let token_id = parse_optional_usize(args, &mut index, 248044, "token_id")?;
    let token_id = u32::try_from(token_id)
        .map_err(|_| invalid_input("qwen-full-layer-smoke token_id must fit u32"))?;
    let prefix_len = parse_optional_usize(args, &mut index, 8, "prefix_len")?;
    if index != args.len() {
        return Err(invalid_input(
            "qwen-full-layer-smoke accepts at most [model_dir] [layer] [token_id] [prefix_len]",
        ));
    }

    let smoke =
        qwen35_full_layer_smoke(&stream, &module, &model_dir, layer, token_id, 0, prefix_len)?;
    println!(
        "Qwen3.5 full layer smoke passed: model_dir={} layer={} token_id={} position={} output_max_abs={:.8}",
        model_dir.display(),
        smoke.layer,
        smoke.token_id,
        smoke.position,
        smoke.output_max_abs
    );
    println!("  output_prefix={:?}", smoke.output_prefix);
    Ok(())
}

fn run_qwen_linear_layer_smoke(args: &[String]) -> AppResult<()> {
    let (stream, module) = cuda_handles().map_err(|error| {
        invalid_input(format!(
            "qwen-linear-layer-smoke CUDA initialization failed: {error}"
        ))
    })?;
    let mut index = 0;
    let model_dir = parse_optional_non_numeric_model_dir(args, &mut index, DEFAULT_QWEN3_6_27B_DIR);
    let layer = parse_optional_usize(args, &mut index, 0, "layer")?;
    let token_id = parse_optional_usize(args, &mut index, 248044, "token_id")?;
    let token_id = u32::try_from(token_id)
        .map_err(|_| invalid_input("qwen-linear-layer-smoke token_id must fit u32"))?;
    let prefix_len = parse_optional_usize(args, &mut index, 8, "prefix_len")?;
    if index != args.len() {
        return Err(invalid_input(
            "qwen-linear-layer-smoke accepts at most [model_dir] [layer] [token_id] [prefix_len]",
        ));
    }

    let smoke =
        qwen35_linear_layer_smoke(&stream, &module, &model_dir, layer, token_id, 0, prefix_len)?;
    println!(
        "Qwen3.5 linear layer smoke passed: model_dir={} layer={} token_id={} position={} output_max_abs={:.8} recurrent_state_max_abs={:.8}",
        model_dir.display(),
        smoke.layer,
        smoke.token_id,
        smoke.position,
        smoke.output_max_abs,
        smoke.recurrent_state_max_abs
    );
    println!("  output_prefix={:?}", smoke.output_prefix);
    Ok(())
}

fn run_qwen_prefix_layers_smoke(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_optional_non_numeric_model_dir(args, &mut index, DEFAULT_QWEN3_6_27B_DIR);
    let layer_count = parse_optional_usize(args, &mut index, 4, "layer_count")?;
    let token_id = parse_optional_usize(args, &mut index, 248044, "token_id")?;
    let token_id = u32::try_from(token_id)
        .map_err(|_| invalid_input("qwen-prefix-layers-smoke token_id must fit u32"))?;
    let prefix_len = parse_optional_usize(args, &mut index, 8, "prefix_len")?;
    if index != args.len() {
        return Err(invalid_input(
            "qwen-prefix-layers-smoke accepts at most [model_dir] [layer_count] [token_id] [prefix_len]",
        ));
    }

    let (stream, module) = cuda_handles().map_err(|error| {
        invalid_input(format!(
            "qwen-prefix-layers-smoke CUDA initialization failed: {error}"
        ))
    })?;
    let smoke = qwen35_prefix_layers_smoke(
        &stream,
        &module,
        &model_dir,
        layer_count,
        token_id,
        0,
        prefix_len,
    )?;
    println!(
        "Qwen3.5 prefix layer smoke passed: model_dir={} layer_count={} linear_layers={} full_layers={} token_id={} position={} output_max_abs={:.8}",
        model_dir.display(),
        smoke.layer_count,
        smoke.linear_layer_count,
        smoke.full_layer_count,
        smoke.token_id,
        smoke.position,
        smoke.output_max_abs
    );
    println!("  output_prefix={:?}", smoke.output_prefix);
    Ok(())
}

fn text_layer_kind_label(kind: TextLayerKind) -> &'static str {
    match kind {
        TextLayerKind::FullAttention => "full_attention",
        TextLayerKind::LinearAttention => "linear_attention",
    }
}

fn run_qwen_tokens_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_optional_non_numeric_model_dir(args, &mut index, DEFAULT_QWEN3_6_27B_DIR);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let cli = parse_token_cli(args, index)?;
    if cli.report_path.is_some() {
        return Err(invalid_input(
            "qwen token reports need a Qwen tokenizer implementation; omit --report for now",
        ));
    }

    let config = TextConfig::from_model_dir(&model_dir)?;
    if config.model_kind == TextModelKind::Qwen35Text {
        let report = qwen35_weight_layout_report(&model_dir)?;
        return Err(invalid_input(format!(
            "Qwen3.5 text layout is recognized and {} layer tensor layouts validate ({} linear attention, {} full attention), but Qwen3.5 execution kernels are not implemented yet",
            report.layers.len(),
            report.config.layer_kinds.linear_attention_count(),
            report.config.layer_kinds.full_attention_count()
        )));
    }

    let stop_token_id = config.eos_token_id;
    let (stream, module) = cuda_handles()?;
    let suite = if let Some(sampling) = cli.sampling {
        inference::run_ministral_generation_sampled_suite_with_backend(
            stream,
            module,
            &model_dir,
            &cli.prompts,
            InferenceBackend::Bf16,
            max_new_tokens,
            top_k,
            stop_token_id,
            sampling,
        )?
    } else {
        inference::run_ministral_generation_suite_with_backend(
            stream,
            module,
            &model_dir,
            &cli.prompts,
            InferenceBackend::Bf16,
            max_new_tokens,
            top_k,
            stop_token_id,
        )?
    };

    println!(
        "Qwen token suite: backend=bf16 decode_strategy={} prompts_len={} eos_token_id={:?}",
        decode_strategy_label(cli.sampling),
        suite.prompts.len(),
        stop_token_id
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_result(result, None)?;
    }

    Ok(())
}

fn run_ministral_tokens_exported_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let cli = parse_token_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = if let Some(sampling) = cli.sampling {
        inference::run_ministral_generation_exported_all_linear_int8_sampled_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            max_new_tokens,
            top_k,
            model_eos_token_id(&model_dir)?,
            sampling,
        )?
    } else {
        inference::run_ministral_generation_exported_all_linear_int8_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            max_new_tokens,
            top_k,
            model_eos_token_id(&model_dir)?,
        )?
    };
    let tokenizer = TekkenTokenizer::open(&model_dir)?;

    println!(
        "Ministral token exported all-linear-int8 suite: export_dir={} decode_strategy={} prompts_len={}",
        export_dir.display(),
        decode_strategy_label(cli.sampling),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_generation_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, InferenceBackend::AllLinearInt8)
                .with_export_dir(&export_dir)
                .with_decode_strategy(decode_strategy_label(cli.sampling))
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_tokens_logits_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let backend = parse_optional_backend(args, &mut index, InferenceBackend::Bf16)?;
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_token_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = inference::run_ministral_generation_logits_suite_with_backend(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        backend,
        top_k,
    )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;

    println!(
        "Ministral token logits suite: backend={} prompts_len={}",
        backend_label(backend),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_logits_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_logits_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, backend).with_top_k(top_k),
            &cli.prompt_file_paths,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_tokens_exported_logits_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_token_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = inference::run_ministral_generation_exported_all_linear_int8_logits_suite(
        stream,
        module,
        &model_dir,
        &export_dir,
        &cli.prompts,
        top_k,
    )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;

    println!(
        "Ministral token exported all-linear-int8 logits suite: export_dir={} prompts_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_logits_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_logits_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, InferenceBackend::AllLinearInt8)
                .with_export_dir(&export_dir)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_tokens_trace_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let backend = parse_optional_backend(args, &mut index, InferenceBackend::Bf16)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_token_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite = inference::run_ministral_generation_logits_trace_suite_with_backend(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        backend,
        &options,
    )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;

    println!(
        "Ministral token trace suite: backend={} prompts_len={}",
        backend_label(backend),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_trace_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_logits_trace_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, backend)
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k)
                .with_logits_top_k(logits_top_k),
            &cli.prompt_file_paths,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_tokens_exported_trace_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_token_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite = inference::run_ministral_generation_exported_all_linear_int8_logits_trace_suite(
        stream,
        module,
        &model_dir,
        &export_dir,
        &cli.prompts,
        &options,
    )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;

    println!(
        "Ministral token exported all-linear-int8 trace suite: export_dir={} prompts_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_trace_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_logits_trace_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, InferenceBackend::AllLinearInt8)
                .with_export_dir(&export_dir)
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k)
                .with_logits_top_k(logits_top_k),
            &cli.prompt_file_paths,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_tokens_eval(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_token_compare_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let eval_suite = inference::run_ministral_generation_backend_eval_suite_with_driver(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        candidate_backend,
        &options,
        cli.driver,
    )?;
    let generation_summary =
        chat_compare_summary_from_generation_suite(&eval_suite.generation.summary);

    println!(
        "Ministral token eval: candidate={} driver={} prompts_len={}",
        backend_label(candidate_backend),
        driver_label(cli.driver),
        cli.prompts.len()
    );
    print_logits_summary("token_logits", &eval_suite.logits.summary);
    print_compare_suite_summary("token_generation", &generation_summary);

    if let Some(path) = &cli.report_path {
        let tokenizer = TekkenTokenizer::open(&model_dir)?;
        write_ministral_token_eval_report(
            path,
            &model_dir,
            candidate_backend,
            None,
            cli.driver,
            max_new_tokens,
            top_k,
            logits_top_k,
            &cli.prompt_file_paths,
            &cli.thresholds,
            &generation_summary,
            &eval_suite,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }

    enforce_token_logits_compare_thresholds(&eval_suite.logits.summary, &cli.thresholds)?;
    enforce_chat_compare_thresholds(&generation_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_tokens_eval_exported(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_token_compare_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let eval_suite =
        inference::run_ministral_generation_exported_all_linear_int8_eval_suite_with_driver(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            &options,
            cli.driver,
        )?;
    let generation_summary =
        chat_compare_summary_from_generation_suite(&eval_suite.generation.summary);

    println!(
        "Ministral token exported all-linear-int8 eval: export_dir={} driver={} prompts_len={}",
        export_dir.display(),
        driver_label(cli.driver),
        cli.prompts.len()
    );
    print_logits_summary("token_logits", &eval_suite.logits.summary);
    print_compare_suite_summary("token_generation", &generation_summary);
    println!(
        "  logits_reference_memory_bytes={} logits_candidate_memory_bytes={}",
        eval_suite
            .logits
            .reference_memory_stats
            .total_resident_bytes,
        eval_suite
            .logits
            .candidate_memory_stats
            .total_resident_bytes
    );
    println!(
        "  generation_reference_memory_bytes={} generation_candidate_memory_bytes={}",
        eval_suite
            .generation
            .reference_memory_stats
            .total_resident_bytes,
        eval_suite
            .generation
            .candidate_memory_stats
            .total_resident_bytes
    );

    if let Some(path) = &cli.report_path {
        let tokenizer = TekkenTokenizer::open(&model_dir)?;
        write_ministral_token_eval_report(
            path,
            &model_dir,
            InferenceBackend::AllLinearInt8,
            Some(&export_dir),
            cli.driver,
            max_new_tokens,
            top_k,
            logits_top_k,
            &cli.prompt_file_paths,
            &cli.thresholds,
            &generation_summary,
            &eval_suite,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }

    enforce_token_logits_compare_thresholds(&eval_suite.logits.summary, &cli.thresholds)?;
    enforce_chat_compare_thresholds(&generation_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_tokens_forced_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let logits_top_k = parse_optional_usize(args, &mut index, 8, "logits_top_k")?;
    let cli = parse_token_compare_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = inference::run_ministral_generation_backend_forced_comparison_suite(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        candidate_backend,
        logits_top_k,
    )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let suite_summary = chat_compare_summary_from_forced_suite(&suite.summary);

    println!(
        "Ministral token forced comparison: candidate={} sequences_len={}",
        backend_label(candidate_backend),
        suite.prompts.len()
    );
    print_compare_suite_summary("token_forced", &suite_summary);
    println!(
        "  forced_reference_match_count={}/{}",
        suite.summary.summary.forced_reference_match_count, suite.summary.summary.step_count
    );
    println!(
        "  forced_candidate_match_count={}/{}",
        suite.summary.summary.forced_candidate_match_count, suite.summary.summary.step_count
    );
    println!(
        "  mean_forced_logprob_abs_diff={:.12}",
        suite.summary.summary.mean_forced_logprob_abs_diff()
    );
    println!(
        "  max_forced_logprob_abs_diff={:.8}",
        suite.summary.summary.max_forced_logprob_abs_diff
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  sequence[{index}]={:?}", result.sequence_tokens);
        print_generation_forced_comparison_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_forced_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_tokens_exported_forced_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let logits_top_k = parse_optional_usize(args, &mut index, 8, "logits_top_k")?;
    let cli = parse_token_compare_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite =
        inference::run_ministral_generation_exported_all_linear_int8_forced_comparison_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            logits_top_k,
        )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let suite_summary = chat_compare_summary_from_forced_suite(&suite.summary);

    println!(
        "Ministral token exported all-linear-int8 forced comparison: export_dir={} sequences_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    print_compare_suite_summary("token_forced", &suite_summary);
    println!(
        "  forced_reference_match_count={}/{}",
        suite.summary.summary.forced_reference_match_count, suite.summary.summary.step_count
    );
    println!(
        "  forced_candidate_match_count={}/{}",
        suite.summary.summary.forced_candidate_match_count, suite.summary.summary.step_count
    );
    println!(
        "  mean_forced_logprob_abs_diff={:.12}",
        suite.summary.summary.mean_forced_logprob_abs_diff()
    );
    println!(
        "  max_forced_logprob_abs_diff={:.8}",
        suite.summary.summary.max_forced_logprob_abs_diff
    );
    println!(
        "  reference_memory_bytes={} candidate_memory_bytes={}",
        suite.reference_memory_stats.total_resident_bytes,
        suite.candidate_memory_stats.total_resident_bytes
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  sequence[{index}]={:?}", result.sequence_tokens);
        print_generation_forced_comparison_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_forced_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_tokens_logits_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_token_compare_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite = inference::run_ministral_generation_backend_next_logits_comparison_suite(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        candidate_backend,
        top_k,
    )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;

    println!(
        "Ministral token logits comparison: candidate={} prompts_len={}",
        backend_label(candidate_backend),
        suite.prompts.len()
    );
    println!(
        "  token_match_count={}/{}",
        suite.summary.token_match_count, suite.summary.prompt_count
    );
    println!("  top_tokens_match={}", suite.summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", suite.summary.mean_kl());
    println!("  max_kl_divergence={:.12}", suite.summary.max_kl);
    println!("  max_abs_diff={:.8}", suite.summary.max_abs_diff);
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_logits_comparison_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_logits_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_token_logits_compare_thresholds(&suite.summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_tokens_exported_logits_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_token_compare_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let suite =
        inference::run_ministral_generation_exported_all_linear_int8_next_logits_comparison_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            top_k,
        )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;

    println!(
        "Ministral token exported all-linear-int8 logits comparison: export_dir={} prompts_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    println!(
        "  token_match_count={}/{}",
        suite.summary.token_match_count, suite.summary.prompt_count
    );
    println!("  top_tokens_match={}", suite.summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", suite.summary.mean_kl());
    println!("  max_kl_divergence={:.12}", suite.summary.max_kl);
    println!("  max_abs_diff={:.8}", suite.summary.max_abs_diff);
    println!(
        "  reference_memory_bytes={} candidate_memory_bytes={}",
        suite.reference_memory_stats.total_resident_bytes,
        suite.candidate_memory_stats.total_resident_bytes
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_logits_comparison_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_logits_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_token_logits_compare_thresholds(&suite.summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_tokens_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_token_compare_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite = inference::run_ministral_generation_backend_comparison_suite_with_driver(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        candidate_backend,
        &options,
        cli.driver,
    )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let mut suite_summary = ChatCompareSuiteSummary::default();
    for result in &suite.prompts {
        suite_summary.observe(&result.summary());
    }

    println!(
        "Ministral token generation comparison: candidate={} driver={} prompts_len={}",
        backend_label(candidate_backend),
        driver_label(cli.driver),
        suite.prompts.len()
    );
    println!(
        "  token_match_count={}/{}",
        suite_summary.token_match_count, suite_summary.step_count
    );
    println!("  top_tokens_match={}", suite_summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", suite_summary.mean_kl());
    println!("  max_kl_divergence={:.12}", suite_summary.max_kl);
    println!("  max_abs_diff={:.8}", suite_summary.max_abs_diff);
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_comparison_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_compare_report(
            path,
            &suite.prompts,
            &suite_summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_tokens_exported_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_token_compare_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = inference::GenerationLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite =
        inference::run_ministral_generation_exported_all_linear_int8_comparison_suite_with_driver(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            &options,
            cli.driver,
        )?;
    let tokenizer = TekkenTokenizer::open(&model_dir)?;
    let mut suite_summary = ChatCompareSuiteSummary::default();
    for result in &suite.prompts {
        suite_summary.observe(&result.summary());
    }

    println!(
        "Ministral token exported all-linear-int8 comparison: export_dir={} driver={} prompts_len={}",
        export_dir.display(),
        driver_label(cli.driver),
        suite.prompts.len()
    );
    println!(
        "  token_match_count={}/{}",
        suite_summary.token_match_count, suite_summary.step_count
    );
    println!("  top_tokens_match={}", suite_summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", suite_summary.mean_kl());
    println!("  max_kl_divergence={:.12}", suite_summary.max_kl);
    println!("  max_abs_diff={:.8}", suite_summary.max_abs_diff);
    println!(
        "  reference_memory_bytes={} candidate_memory_bytes={}",
        suite.reference_memory_stats.total_resident_bytes,
        suite.candidate_memory_stats.total_resident_bytes
    );
    for (index, result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", result.prompt_tokens);
        print_generation_comparison_result(result, Some(&tokenizer))?;
    }

    if let Some(path) = &cli.report_path {
        write_token_compare_report(
            path,
            &suite.prompts,
            &suite_summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
            &tokenizer,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_chat_logits_smoke(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let backend = parse_optional_backend(args, &mut index, InferenceBackend::Bf16)?;
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatLogitsOptions {
        top_k,
        system_prompt: cli.system_prompt.clone(),
    };
    let suite = inference::run_ministral_single_turn_chat_logits_suite_with_backend(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        backend,
        &options,
    )?;

    for prompt_result in &suite.prompts {
        println!(
            "Ministral chat logits: backend={} user_prompt={:?}",
            backend_label(backend),
            prompt_result.user_prompt
        );
        print_chat_logits_result(&prompt_result.result, &cli.system_prompt);
    }

    if let Some(path) = &cli.report_path {
        let report_results: Vec<_> = suite
            .prompts
            .into_iter()
            .map(|prompt| (prompt.user_prompt, prompt.result))
            .collect();
        write_chat_logits_report(
            path,
            &report_results,
            &InferenceReportMetadata::new(&model_dir, backend)
                .with_system_prompt(&cli.system_prompt)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_chat_exported_logits_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatLogitsOptions {
        top_k,
        system_prompt: cli.system_prompt.clone(),
    };
    let suite = inference::run_ministral_single_turn_chat_exported_all_linear_int8_logits_suite(
        stream,
        module,
        &model_dir,
        &export_dir,
        &cli.prompts,
        &options,
    )?;

    println!(
        "Ministral chat exported all-linear-int8 logits suite: export_dir={} prompts_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    for (index, prompt_result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", prompt_result.user_prompt);
        print_chat_logits_result(&prompt_result.result, &cli.system_prompt);
    }

    if let Some(path) = &cli.report_path {
        let report_results: Vec<_> = suite
            .prompts
            .into_iter()
            .map(|prompt| (prompt.user_prompt, prompt.result))
            .collect();
        write_chat_logits_report(
            path,
            &report_results,
            &InferenceReportMetadata::new(&model_dir, InferenceBackend::AllLinearInt8)
                .with_export_dir(&export_dir)
                .with_system_prompt(&cli.system_prompt)
                .with_top_k(top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_chat_logits_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatLogitsOptions {
        top_k,
        system_prompt: cli.system_prompt.clone(),
    };
    let suite = inference::run_ministral_single_turn_chat_logits_comparison_suite(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        candidate_backend,
        &options,
    )?;

    println!(
        "Ministral chat logits comparison: candidate={} prompts_len={}",
        backend_label(candidate_backend),
        suite.prompts.len()
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    println!(
        "  token_match_count={}/{}",
        suite.summary.token_match_count, suite.summary.prompt_count
    );
    println!("  top_tokens_match={}", suite.summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", suite.summary.mean_kl());
    println!("  max_kl_divergence={:.12}", suite.summary.max_kl);
    println!("  max_abs_diff={:.8}", suite.summary.max_abs_diff);
    for (index, prompt_result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", prompt_result.user_prompt);
        print_chat_logits_comparison_result(&prompt_result.result, &cli.system_prompt);
    }

    if let Some(path) = &cli.report_path {
        write_chat_logits_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
        )?;
        println!("report={}", path.display());
    }
    enforce_token_logits_compare_thresholds(&suite.summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_chat_exported_logits_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let top_k = parse_optional_usize(args, &mut index, 8, "top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatLogitsOptions {
        top_k,
        system_prompt: cli.system_prompt.clone(),
    };
    let suite =
        inference::run_ministral_single_turn_chat_exported_all_linear_int8_logits_comparison_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            &options,
        )?;

    println!(
        "Ministral chat exported all-linear-int8 logits comparison: export_dir={} prompts_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    println!(
        "  token_match_count={}/{}",
        suite.summary.token_match_count, suite.summary.prompt_count
    );
    println!("  top_tokens_match={}", suite.summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", suite.summary.mean_kl());
    println!("  max_kl_divergence={:.12}", suite.summary.max_kl);
    println!("  max_abs_diff={:.8}", suite.summary.max_abs_diff);
    println!(
        "  reference_memory_bytes={} candidate_memory_bytes={}",
        suite.reference_memory_stats.total_resident_bytes,
        suite.candidate_memory_stats.total_resident_bytes
    );
    for (index, prompt_result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", prompt_result.user_prompt);
        print_chat_logits_comparison_result(&prompt_result.result, &cli.system_prompt);
    }

    if let Some(path) = &cli.report_path {
        write_chat_logits_compare_report(
            path,
            &suite.prompts,
            &suite.summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
        )?;
        println!("report={}", path.display());
    }
    enforce_token_logits_compare_thresholds(&suite.summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_chat_trace_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let backend = parse_optional_backend(args, &mut index, InferenceBackend::Bf16)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite = inference::run_ministral_single_turn_chat_logits_trace_suite_with_backend(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        backend,
        &options,
    )?;

    println!(
        "Ministral chat trace suite: backend={} prompts_len={}",
        backend_label(backend),
        suite.prompts.len()
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    for (index, prompt_result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", prompt_result.user_prompt);
        print_chat_trace_result(&prompt_result.result, &cli.system_prompt);
    }

    if let Some(path) = &cli.report_path {
        write_chat_logits_trace_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, backend)
                .with_system_prompt(&cli.system_prompt)
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k)
                .with_logits_top_k(logits_top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_chat_exported_trace_suite(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite =
        inference::run_ministral_single_turn_chat_exported_all_linear_int8_logits_trace_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            &options,
        )?;

    println!(
        "Ministral chat exported all-linear-int8 trace suite: export_dir={} prompts_len={}",
        export_dir.display(),
        suite.prompts.len()
    );
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    for (index, prompt_result) in suite.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", prompt_result.user_prompt);
        print_chat_trace_result(&prompt_result.result, &cli.system_prompt);
    }

    if let Some(path) = &cli.report_path {
        write_chat_logits_trace_report(
            path,
            &suite.prompts,
            &InferenceReportMetadata::new(&model_dir, InferenceBackend::AllLinearInt8)
                .with_export_dir(&export_dir)
                .with_system_prompt(&cli.system_prompt)
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k)
                .with_logits_top_k(logits_top_k),
            &cli.prompt_file_paths,
        )?;
        println!("report={}", path.display());
    }

    Ok(())
}

fn run_ministral_chat_forced_trace_smoke(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let prompt = first_prompt(&cli);
    let (stream, module) = cuda_handles()?;
    let trace_options = ChatLogitsTraceOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let generated_trace = inference::run_ministral_single_turn_chat_logits_trace(
        stream.clone(),
        module.clone(),
        &model_dir,
        prompt,
        &trace_options,
    )?;
    let forced_options = ChatForcedLogitsTraceOptions {
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
    };
    let forced_trace = inference::run_ministral_single_turn_chat_forced_logits_trace(
        stream,
        module,
        &model_dir,
        prompt,
        &generated_trace.generated_tokens,
        &forced_options,
    )?;
    let comparison = inference::compare_logits_traces(&generated_trace.steps, &forced_trace.steps)?;

    println!("Ministral chat forced trace: user_prompt={prompt:?}");
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    print_token_window("prompt_tokens", &generated_trace.prompt_tokens);
    println!("  generated_tokens={:?}", generated_trace.generated_tokens);
    println!("  forced_tokens={:?}", forced_trace.generated_tokens);
    println!("  token_ids_match={}", comparison.token_ids_match);
    println!("  mean_kl_divergence={:.12}", comparison.mean_kl_divergence);
    println!("  max_kl_divergence={:.12}", comparison.max_kl_divergence);
    println!("  max_abs_diff={:.8}", comparison.max_abs_diff);
    for step in &comparison.steps {
        println!(
            "  step={} reference_token={} candidate_token={} kl_divergence={:.12} max_abs_diff={:.8}",
            step.step,
            step.reference_token_id,
            step.candidate_token_id,
            step.kl_divergence,
            step.max_abs_diff
        );
    }

    if let Some(path) = &cli.report_path {
        write_forced_trace_report(
            path,
            prompt,
            &generated_trace,
            &forced_trace,
            &InferenceReportMetadata::new(&model_dir, InferenceBackend::Bf16)
                .with_system_prompt(&cli.system_prompt)
                .with_max_new_tokens(max_new_tokens)
                .with_top_k(top_k)
                .with_logits_top_k(logits_top_k),
            &cli.prompt_file_paths,
        )?;
        println!("  report={}", path.display());
    }

    Ok(())
}

fn run_ministral_chat_forced_trace_suite_smoke(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatTraceComparisonSuiteOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let result = inference::run_ministral_chat_forced_trace_suite(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        &options,
    )?;

    println!("Ministral chat forced trace suite:");
    println!(
        "  use_default_system={}",
        matches!(cli.system_prompt, SystemPrompt::DefaultFromModel)
    );
    println!("  prompts_len={}", result.prompts.len());
    println!("  total_steps={}", result.total_steps);
    println!("  all_token_ids_match={}", result.all_token_ids_match);
    println!("  mean_kl_divergence={:.12}", result.mean_kl_divergence);
    println!("  max_kl_divergence={:.12}", result.max_kl_divergence);
    println!("  max_abs_diff={:.8}", result.max_abs_diff);
    for (index, prompt_result) in result.prompts.iter().enumerate() {
        println!("  prompt[{index}]={:?}", prompt_result.prompt);
        println!(
            "    prompt_tokens_len={}",
            prompt_result.prompt_tokens.len()
        );
        println!("    generated_tokens={:?}", prompt_result.generated_tokens);
        println!("    generated_text={:?}", prompt_result.generated_text);
        println!("    steps={}", prompt_result.comparison.steps.len());
        println!(
            "    token_ids_match={}",
            prompt_result.comparison.token_ids_match
        );
        println!(
            "    mean_kl_divergence={:.12}",
            prompt_result.comparison.mean_kl_divergence
        );
        println!(
            "    max_kl_divergence={:.12}",
            prompt_result.comparison.max_kl_divergence
        );
        println!(
            "    max_abs_diff={:.8}",
            prompt_result.comparison.max_abs_diff
        );
    }

    Ok(())
}

fn run_ministral_chat_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let candidate_backend =
        parse_optional_backend(args, &mut index, InferenceBackend::AllLinearInt8)?;
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatTraceComparisonSuiteOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite = inference::run_ministral_single_turn_chat_backend_comparison_suite(
        stream,
        module,
        &model_dir,
        &cli.prompts,
        candidate_backend,
        &options,
        cli.driver,
    )?;
    let mut suite_summary = ChatCompareSuiteSummary::default();

    for prompt_result in &suite.prompts {
        let summary = prompt_result.result.comparison.summary();
        suite_summary.observe(&summary);
        print_chat_compare_result(
            &prompt_result.user_prompt,
            candidate_backend,
            cli.driver,
            &prompt_result.result,
            &summary,
        );
    }

    if let Some(path) = &cli.report_path {
        let results: Vec<_> = suite
            .prompts
            .into_iter()
            .map(|prompt| (prompt.user_prompt, prompt.result))
            .collect();
        write_chat_compare_report(
            path,
            &results,
            &suite_summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn run_ministral_chat_exported_compare(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let model_dir = parse_model_dir(args, &mut index);
    let export_dir = parse_optional_export_dir(args, &mut index);
    let max_new_tokens = parse_optional_usize(args, &mut index, 1, "max_new_tokens")?;
    let top_k = parse_optional_usize(args, &mut index, 1, "top_k")?;
    let logits_top_k = parse_optional_usize(args, &mut index, top_k.max(3), "logits_top_k")?;
    let cli = parse_chat_cli(args, index)?;
    let (stream, module) = cuda_handles()?;
    let options = ChatTraceComparisonSuiteOptions {
        max_new_tokens,
        top_k,
        logits_top_k,
        system_prompt: cli.system_prompt.clone(),
        stop_token_id: model_eos_token_id(&model_dir)?,
    };
    let suite =
        inference::run_ministral_single_turn_chat_exported_all_linear_int8_comparison_suite(
            stream,
            module,
            &model_dir,
            &export_dir,
            &cli.prompts,
            &options,
            cli.driver,
        )?;
    let mut suite_summary = ChatCompareSuiteSummary::default();

    println!(
        "Ministral chat exported all-linear-int8 comparison: export_dir={} driver={} prompts_len={}",
        export_dir.display(),
        driver_label(cli.driver),
        suite.prompts.len()
    );
    println!(
        "  reference_memory_bytes={} candidate_memory_bytes={}",
        suite.reference_memory_stats.total_resident_bytes,
        suite.candidate_memory_stats.total_resident_bytes
    );
    for prompt_result in &suite.prompts {
        let summary = prompt_result.result.comparison.summary();
        suite_summary.observe(&summary);
        print_chat_compare_result(
            &prompt_result.user_prompt,
            InferenceBackend::AllLinearInt8,
            cli.driver,
            &prompt_result.result,
            &summary,
        );
    }

    if let Some(path) = &cli.report_path {
        let results: Vec<_> = suite
            .prompts
            .into_iter()
            .map(|prompt| (prompt.user_prompt, prompt.result))
            .collect();
        write_chat_compare_report(
            path,
            &results,
            &suite_summary,
            &cli.prompt_file_paths,
            &cli.thresholds,
        )?;
        println!("report={}", path.display());
    }
    enforce_chat_compare_thresholds(&suite_summary, &cli.thresholds)?;

    Ok(())
}

fn cuda_handles() -> AppResult<(Arc<CudaStream>, Arc<CudaModule>)> {
    let device_index = cuda_device_index_from_env()?;
    cuda_handles_for_device(device_index)
}

pub(crate) fn cuda_handles_for_device(
    device_index: usize,
) -> AppResult<(Arc<CudaStream>, Arc<CudaModule>)> {
    let ctx = CudaContext::new(device_index)?;
    let stream = ctx.default_stream();
    let module = runtime::load_default_module(&ctx)?;
    Ok((stream, module))
}

pub(crate) fn cuda_worker_handles_for_device(
    device_index: usize,
) -> AppResult<(Arc<CudaStream>, Arc<CudaModule>)> {
    let ctx = CudaContext::new(device_index)?;
    let stream = ctx.new_stream()?;
    let module = runtime::load_default_module(&ctx)?;
    Ok((stream, module))
}

fn cuda_device_index_from_env() -> AppResult<usize> {
    match env::var("NN_RUST_CUDA_DEVICE") {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            invalid_input(format!(
                "NN_RUST_CUDA_DEVICE must be a nonnegative CUDA device index, got {value:?}: {error}"
            ))
        }),
        Err(env::VarError::NotPresent) => Ok(0),
        Err(error) => Err(invalid_input(format!(
            "could not read NN_RUST_CUDA_DEVICE: {error}"
        ))),
    }
}

fn assert_close(actual: f32, expected: f32, tolerance: f32, context: &str) -> AppResult<()> {
    if (actual - expected).abs() <= tolerance {
        Ok(())
    } else {
        Err(invalid_data(format!(
            "{context} mismatch: expected {expected}, got {actual}"
        )))
    }
}

fn parse_model_dir(args: &[String], index: &mut usize) -> PathBuf {
    if *index < args.len() && !args[*index].starts_with("--") {
        let path = PathBuf::from(&args[*index]);
        *index += 1;
        path
    } else {
        PathBuf::from(DEFAULT_MINISTRAL_DIR)
    }
}

fn parse_optional_non_numeric_model_dir(
    args: &[String],
    index: &mut usize,
    default: &str,
) -> PathBuf {
    if *index < args.len()
        && !args[*index].starts_with("--")
        && args[*index].parse::<usize>().is_err()
    {
        let path = PathBuf::from(&args[*index]);
        *index += 1;
        path
    } else {
        PathBuf::from(default)
    }
}

fn parse_optional_export_dir(args: &[String], index: &mut usize) -> PathBuf {
    if *index < args.len()
        && !args[*index].starts_with("--")
        && args[*index].parse::<usize>().is_err()
    {
        let path = PathBuf::from(&args[*index]);
        *index += 1;
        path
    } else {
        PathBuf::from("runs/ministral_chat_trace/all_linear_int8_export")
    }
}

fn parse_optional_backend(
    args: &[String],
    index: &mut usize,
    default: InferenceBackend,
) -> AppResult<InferenceBackend> {
    if *index >= args.len() || args[*index].starts_with("--") {
        return Ok(default);
    }

    let Some(backend) = parse_backend(&args[*index]) else {
        return Ok(default);
    };
    *index += 1;
    Ok(backend)
}

fn parse_backend(value: &str) -> Option<InferenceBackend> {
    match value {
        "bf16" => Some(InferenceBackend::Bf16),
        "all-linear-int8" | "int8" => Some(InferenceBackend::AllLinearInt8),
        _ => None,
    }
}

fn parse_bf16_top1_plan(value: &str) -> AppResult<Bf16Top1Plan> {
    match value {
        "rows1" | "row1" | "1" => Ok(Bf16Top1Plan::Rows1),
        "rows2" | "row2" | "2" => Ok(Bf16Top1Plan::Rows2),
        "rows4" | "row4" | "4" | "default" => Ok(Bf16Top1Plan::Rows4),
        "rows8" | "row8" | "8" => Ok(Bf16Top1Plan::Rows8),
        other => Err(invalid_input(format!(
            "unknown BF16 top1 plan {other:?}; expected rows1, rows2, rows4, or rows8"
        ))),
    }
}

fn parse_optional_usize(
    args: &[String],
    index: &mut usize,
    default: usize,
    _name: &str,
) -> AppResult<usize> {
    if *index >= args.len() || args[*index].starts_with("--") || args[*index] == "--next" {
        return Ok(default);
    }

    match args[*index].parse::<usize>() {
        Ok(value) => {
            *index += 1;
            Ok(value)
        }
        Err(_) => Ok(default),
    }
}

fn parse_remaining_token_ids(args: &[String], start: usize) -> AppResult<Vec<u32>> {
    let mut tokens = Vec::new();
    for value in &args[start..] {
        if value.starts_with("--") {
            return Err(invalid_input(format!(
                "unexpected token probe flag {value:?}"
            )));
        }
        tokens.push(
            value.parse::<u32>().map_err(|error| {
                invalid_input(format!("token id {value:?} is not a u32: {error}"))
            })?,
        );
    }
    if tokens.is_empty() {
        return Err(invalid_input(
            "output projection export probe requires at least one token",
        ));
    }
    Ok(tokens)
}

fn parse_required_flag_value<'a>(
    args: &'a [String],
    index: &mut usize,
    flag: &str,
) -> AppResult<&'a str> {
    *index += 1;
    args.get(*index)
        .map(|value| {
            *index += 1;
            value.as_str()
        })
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")))
}

#[derive(Debug)]
struct ChatCli {
    system_prompt: SystemPrompt,
    prompts: Vec<String>,
    forced_target_pairs: Vec<(String, String)>,
    prompt_file_paths: Vec<PathBuf>,
    pair_file_paths: Vec<PathBuf>,
    report_path: Option<PathBuf>,
    driver: GenerationComparisonDriver,
    thresholds: ChatCompareThresholds,
    sampling: Option<SamplingOptions>,
}

#[derive(Debug, Default)]
struct ChatCompareThresholds {
    require_token_match: bool,
    require_top_tokens_match: bool,
    max_mean_kl: Option<f64>,
    max_max_kl: Option<f64>,
    max_max_abs_diff: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
struct InferenceReportMetadata<'a> {
    model_dir: &'a Path,
    backend: InferenceBackend,
    export_dir: Option<&'a Path>,
    decode_strategy: Option<&'static str>,
    system_prompt: Option<&'a SystemPrompt>,
    max_new_tokens: Option<usize>,
    top_k: Option<usize>,
    logits_top_k: Option<usize>,
}

impl<'a> InferenceReportMetadata<'a> {
    fn new(model_dir: &'a Path, backend: InferenceBackend) -> Self {
        Self {
            model_dir,
            backend,
            export_dir: None,
            decode_strategy: None,
            system_prompt: None,
            max_new_tokens: None,
            top_k: None,
            logits_top_k: None,
        }
    }

    fn with_export_dir(mut self, export_dir: &'a Path) -> Self {
        self.export_dir = Some(export_dir);
        self
    }

    fn with_decode_strategy(mut self, decode_strategy: &'static str) -> Self {
        self.decode_strategy = Some(decode_strategy);
        self
    }

    fn with_system_prompt(mut self, system_prompt: &'a SystemPrompt) -> Self {
        self.system_prompt = Some(system_prompt);
        self
    }

    fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = Some(max_new_tokens);
        self
    }

    fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    fn with_logits_top_k(mut self, logits_top_k: usize) -> Self {
        self.logits_top_k = Some(logits_top_k);
        self
    }
}

#[derive(Debug)]
struct TextForcedTargetCli {
    pairs: Vec<(String, String)>,
    pair_file_paths: Vec<PathBuf>,
    report_path: Option<PathBuf>,
    thresholds: ChatCompareThresholds,
}

#[derive(Debug)]
struct ChatForcedTargetCli {
    system_prompt: SystemPrompt,
    pairs: Vec<(String, String)>,
    pair_file_paths: Vec<PathBuf>,
    report_path: Option<PathBuf>,
    thresholds: ChatCompareThresholds,
}

fn parse_chat_cli(args: &[String], start: usize) -> AppResult<ChatCli> {
    let mut system_prompt = SystemPrompt::DefaultFromModel;
    let mut report_path = None;
    let mut driver = GenerationComparisonDriver::Candidate;
    let mut thresholds = ChatCompareThresholds::default();
    let mut sampling = None;
    let mut prompts = Vec::new();
    let mut forced_target_pairs = Vec::new();
    let mut prompt_file_paths = Vec::new();
    let mut pair_file_paths = Vec::new();
    let mut current_prompt = Vec::new();
    let mut index = start;

    while index < args.len() {
        match args[index].as_str() {
            "--no-system" => {
                system_prompt = SystemPrompt::None;
                index += 1;
            }
            "--system" => {
                let value = parse_required_flag_value(args, &mut index, "--system")?;
                system_prompt = SystemPrompt::Custom(value.to_string());
            }
            "--report" => {
                let value = parse_required_flag_value(args, &mut index, "--report")?;
                report_path = Some(PathBuf::from(value));
            }
            "--prompts-file" => {
                let value = parse_required_flag_value(args, &mut index, "--prompts-file")?;
                prompt_file_paths.push(PathBuf::from(value));
                prompts.extend(read_prompt_file_lines(value)?);
            }
            "--prompt" => {
                push_prompt_words(&mut prompts, &mut current_prompt);
                let value = parse_required_flag_value(args, &mut index, "--prompt")?;
                prompts.push(value.to_string());
            }
            "--pairs-file" => {
                let value = parse_required_flag_value(args, &mut index, "--pairs-file")?;
                pair_file_paths.push(PathBuf::from(value));
                forced_target_pairs
                    .extend(read_forced_target_pair_file(value, "eval forced target")?);
            }
            "--drive-reference" => {
                driver = GenerationComparisonDriver::Reference;
                index += 1;
            }
            "--drive-candidate" => {
                driver = GenerationComparisonDriver::Candidate;
                index += 1;
            }
            "--require-token-match" => {
                thresholds.require_token_match = true;
                index += 1;
            }
            "--require-top-tokens-match" => {
                thresholds.require_top_tokens_match = true;
                index += 1;
            }
            "--max-mean-kl" => {
                let value = parse_required_flag_value(args, &mut index, "--max-mean-kl")?;
                thresholds.max_mean_kl = Some(value.parse()?);
            }
            "--max-max-kl" => {
                let value = parse_required_flag_value(args, &mut index, "--max-max-kl")?;
                thresholds.max_max_kl = Some(value.parse()?);
            }
            "--max-max-abs-diff" => {
                let value = parse_required_flag_value(args, &mut index, "--max-max-abs-diff")?;
                thresholds.max_max_abs_diff = Some(value.parse()?);
            }
            "--sample-temperature" => {
                let value = parse_required_flag_value(args, &mut index, "--sample-temperature")?;
                sampling
                    .get_or_insert_with(SamplingOptions::default)
                    .temperature = value.parse()?;
            }
            "--sample-seed" => {
                let value = parse_required_flag_value(args, &mut index, "--sample-seed")?;
                sampling.get_or_insert_with(SamplingOptions::default).seed = value.parse()?;
            }
            "--next" => {
                push_prompt_words(&mut prompts, &mut current_prompt);
                index += 1;
            }
            value => {
                current_prompt.push(value.to_string());
                index += 1;
            }
        }
    }

    push_prompt_words(&mut prompts, &mut current_prompt);
    if prompts.is_empty() {
        prompts.push("Hello".to_string());
    }

    Ok(ChatCli {
        system_prompt,
        prompts,
        forced_target_pairs,
        prompt_file_paths,
        pair_file_paths,
        report_path,
        driver,
        thresholds,
        sampling,
    })
}

fn parse_text_forced_target_cli(args: &[String], start: usize) -> AppResult<TextForcedTargetCli> {
    let mut report_path = None;
    let mut thresholds = ChatCompareThresholds::default();
    let mut pairs = Vec::new();
    let mut pair_file_paths = Vec::new();
    let mut current_prompt = Vec::new();
    let mut current_target = Vec::new();
    let mut parsing_target = false;
    let mut index = start;

    while index < args.len() {
        match args[index].as_str() {
            "--report" => {
                let value = parse_required_flag_value(args, &mut index, "--report")?;
                report_path = Some(PathBuf::from(value));
            }
            "--pairs-file" => {
                let value = parse_required_flag_value(args, &mut index, "--pairs-file")?;
                pair_file_paths.push(PathBuf::from(value));
                pairs.extend(read_forced_target_pair_file(value, "text forced target")?);
            }
            "--require-token-match" => {
                thresholds.require_token_match = true;
                index += 1;
            }
            "--require-top-tokens-match" => {
                thresholds.require_top_tokens_match = true;
                index += 1;
            }
            "--max-mean-kl" => {
                let value = parse_required_flag_value(args, &mut index, "--max-mean-kl")?;
                thresholds.max_mean_kl = Some(value.parse()?);
            }
            "--max-max-kl" => {
                let value = parse_required_flag_value(args, &mut index, "--max-max-kl")?;
                thresholds.max_max_kl = Some(value.parse()?);
            }
            "--max-max-abs-diff" => {
                let value = parse_required_flag_value(args, &mut index, "--max-max-abs-diff")?;
                thresholds.max_max_abs_diff = Some(value.parse()?);
            }
            "--target" => {
                if parsing_target {
                    return Err(invalid_input("duplicate --target in forced target pair"));
                }
                parsing_target = true;
                index += 1;
            }
            "--next" => {
                push_forced_target_pair(
                    &mut pairs,
                    &mut current_prompt,
                    &mut current_target,
                    "text forced target",
                )?;
                parsing_target = false;
                index += 1;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "unknown text forced target flag {flag:?}"
                )));
            }
            value => {
                if parsing_target {
                    current_target.push(value.to_string());
                } else {
                    current_prompt.push(value.to_string());
                }
                index += 1;
            }
        }
    }

    if !current_prompt.is_empty() || !current_target.is_empty() {
        push_forced_target_pair(
            &mut pairs,
            &mut current_prompt,
            &mut current_target,
            "text forced target",
        )?;
    }
    if pairs.is_empty() {
        pairs.push(("Hello".to_string(), ",".to_string()));
    }

    Ok(TextForcedTargetCli {
        pairs,
        pair_file_paths,
        report_path,
        thresholds,
    })
}

fn parse_chat_forced_target_cli(args: &[String], start: usize) -> AppResult<ChatForcedTargetCli> {
    let mut system_prompt = SystemPrompt::DefaultFromModel;
    let mut report_path = None;
    let mut thresholds = ChatCompareThresholds::default();
    let mut pairs = Vec::new();
    let mut pair_file_paths = Vec::new();
    let mut current_prompt = Vec::new();
    let mut current_target = Vec::new();
    let mut parsing_target = false;
    let mut index = start;

    while index < args.len() {
        match args[index].as_str() {
            "--no-system" => {
                system_prompt = SystemPrompt::None;
                index += 1;
            }
            "--system" => {
                let value = parse_required_flag_value(args, &mut index, "--system")?;
                system_prompt = SystemPrompt::Custom(value.to_string());
            }
            "--report" => {
                let value = parse_required_flag_value(args, &mut index, "--report")?;
                report_path = Some(PathBuf::from(value));
            }
            "--pairs-file" => {
                let value = parse_required_flag_value(args, &mut index, "--pairs-file")?;
                pair_file_paths.push(PathBuf::from(value));
                pairs.extend(read_forced_target_pair_file(value, "chat forced target")?);
            }
            "--require-token-match" => {
                thresholds.require_token_match = true;
                index += 1;
            }
            "--require-top-tokens-match" => {
                thresholds.require_top_tokens_match = true;
                index += 1;
            }
            "--max-mean-kl" => {
                let value = parse_required_flag_value(args, &mut index, "--max-mean-kl")?;
                thresholds.max_mean_kl = Some(value.parse()?);
            }
            "--max-max-kl" => {
                let value = parse_required_flag_value(args, &mut index, "--max-max-kl")?;
                thresholds.max_max_kl = Some(value.parse()?);
            }
            "--max-max-abs-diff" => {
                let value = parse_required_flag_value(args, &mut index, "--max-max-abs-diff")?;
                thresholds.max_max_abs_diff = Some(value.parse()?);
            }
            "--target" => {
                if parsing_target {
                    return Err(invalid_input(
                        "duplicate --target in chat forced target pair",
                    ));
                }
                parsing_target = true;
                index += 1;
            }
            "--next" => {
                push_forced_target_pair(
                    &mut pairs,
                    &mut current_prompt,
                    &mut current_target,
                    "chat forced target",
                )?;
                parsing_target = false;
                index += 1;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "unknown chat forced target flag {flag:?}"
                )));
            }
            value => {
                if parsing_target {
                    current_target.push(value.to_string());
                } else {
                    current_prompt.push(value.to_string());
                }
                index += 1;
            }
        }
    }

    if !current_prompt.is_empty() || !current_target.is_empty() {
        push_forced_target_pair(
            &mut pairs,
            &mut current_prompt,
            &mut current_target,
            "chat forced target",
        )?;
    }
    if pairs.is_empty() {
        pairs.push(("Hello".to_string(), ",".to_string()));
    }

    Ok(ChatForcedTargetCli {
        system_prompt,
        pairs,
        pair_file_paths,
        report_path,
        thresholds,
    })
}

#[derive(Debug)]
struct TokenCli {
    prompts: Vec<Vec<u32>>,
    prompt_file_paths: Vec<PathBuf>,
    report_path: Option<PathBuf>,
    sampling: Option<SamplingOptions>,
}

#[derive(Debug)]
struct TokenCompareCli {
    prompts: Vec<Vec<u32>>,
    prompt_file_paths: Vec<PathBuf>,
    report_path: Option<PathBuf>,
    driver: GenerationComparisonDriver,
    thresholds: ChatCompareThresholds,
}

fn parse_token_cli(args: &[String], start: usize) -> AppResult<TokenCli> {
    let mut report_path = None;
    let mut sampling = None;
    let mut prompts = Vec::new();
    let mut prompt_file_paths = Vec::new();
    let mut current_prompt = Vec::new();
    let mut index = start;

    while index < args.len() {
        match args[index].as_str() {
            "--report" => {
                let value = parse_required_flag_value(args, &mut index, "--report")?;
                report_path = Some(PathBuf::from(value));
            }
            "--prompts-file" => {
                let value = parse_required_flag_value(args, &mut index, "--prompts-file")?;
                prompt_file_paths.push(PathBuf::from(value));
                prompts.extend(read_token_prompt_file(value)?);
            }
            "--sample-temperature" => {
                let value = parse_required_flag_value(args, &mut index, "--sample-temperature")?;
                sampling
                    .get_or_insert_with(SamplingOptions::default)
                    .temperature = value.parse()?;
            }
            "--sample-seed" => {
                let value = parse_required_flag_value(args, &mut index, "--sample-seed")?;
                sampling.get_or_insert_with(SamplingOptions::default).seed = value.parse()?;
            }
            "--next" => {
                push_token_prompt(&mut prompts, &mut current_prompt)?;
                index += 1;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "unknown token generation flag {flag:?}"
                )));
            }
            value => {
                current_prompt.push(value.parse::<u32>().map_err(|error| {
                    invalid_input(format!("token id {value:?} is not a u32: {error}"))
                })?);
                index += 1;
            }
        }
    }

    if !current_prompt.is_empty() {
        push_token_prompt(&mut prompts, &mut current_prompt)?;
    }
    if prompts.is_empty() {
        return Err(invalid_input(
            "token generation requires at least one prompt token",
        ));
    }

    Ok(TokenCli {
        prompts,
        prompt_file_paths,
        report_path,
        sampling,
    })
}

fn parse_token_compare_cli(args: &[String], start: usize) -> AppResult<TokenCompareCli> {
    let mut report_path = None;
    let mut driver = GenerationComparisonDriver::Candidate;
    let mut thresholds = ChatCompareThresholds::default();
    let mut prompts = Vec::new();
    let mut prompt_file_paths = Vec::new();
    let mut current_prompt = Vec::new();
    let mut index = start;

    while index < args.len() {
        match args[index].as_str() {
            "--report" => {
                let value = parse_required_flag_value(args, &mut index, "--report")?;
                report_path = Some(PathBuf::from(value));
            }
            "--prompts-file" => {
                let value = parse_required_flag_value(args, &mut index, "--prompts-file")?;
                prompt_file_paths.push(PathBuf::from(value));
                prompts.extend(read_token_prompt_file(value)?);
            }
            "--drive-reference" => {
                driver = GenerationComparisonDriver::Reference;
                index += 1;
            }
            "--drive-candidate" => {
                driver = GenerationComparisonDriver::Candidate;
                index += 1;
            }
            "--require-token-match" => {
                thresholds.require_token_match = true;
                index += 1;
            }
            "--require-top-tokens-match" => {
                thresholds.require_top_tokens_match = true;
                index += 1;
            }
            "--max-mean-kl" => {
                let value = parse_required_flag_value(args, &mut index, "--max-mean-kl")?;
                thresholds.max_mean_kl = Some(value.parse()?);
            }
            "--max-max-kl" => {
                let value = parse_required_flag_value(args, &mut index, "--max-max-kl")?;
                thresholds.max_max_kl = Some(value.parse()?);
            }
            "--max-max-abs-diff" => {
                let value = parse_required_flag_value(args, &mut index, "--max-max-abs-diff")?;
                thresholds.max_max_abs_diff = Some(value.parse()?);
            }
            "--next" => {
                push_token_prompt(&mut prompts, &mut current_prompt)?;
                index += 1;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "unknown token comparison flag {flag:?}"
                )));
            }
            value => {
                current_prompt.push(value.parse::<u32>().map_err(|error| {
                    invalid_input(format!("token id {value:?} is not a u32: {error}"))
                })?);
                index += 1;
            }
        }
    }

    if !current_prompt.is_empty() {
        push_token_prompt(&mut prompts, &mut current_prompt)?;
    }
    if prompts.is_empty() {
        return Err(invalid_input(
            "token comparison requires at least one prompt token",
        ));
    }

    Ok(TokenCompareCli {
        prompts,
        prompt_file_paths,
        report_path,
        driver,
        thresholds,
    })
}

fn push_token_prompt(prompts: &mut Vec<Vec<u32>>, current_prompt: &mut Vec<u32>) -> AppResult<()> {
    if current_prompt.is_empty() {
        return Err(invalid_input("token prompt segment cannot be empty"));
    }

    prompts.push(std::mem::take(current_prompt));
    Ok(())
}

fn push_forced_target_pair(
    pairs: &mut Vec<(String, String)>,
    current_prompt: &mut Vec<String>,
    current_target: &mut Vec<String>,
    context: &str,
) -> AppResult<()> {
    if current_prompt.is_empty() {
        return Err(invalid_input(format!(
            "{context} pair requires a nonempty prompt before --target"
        )));
    }
    if current_target.is_empty() {
        return Err(invalid_input(format!(
            "{context} pair requires a nonempty target after --target"
        )));
    }

    pairs.push((current_prompt.join(" "), current_target.join(" ")));
    current_prompt.clear();
    current_target.clear();
    Ok(())
}

fn push_prompt_words(prompts: &mut Vec<String>, current_prompt: &mut Vec<String>) {
    if !current_prompt.is_empty() {
        prompts.push(current_prompt.join(" "));
        current_prompt.clear();
    }
}

fn read_prompt_file_lines(path: impl AsRef<Path>) -> AppResult<Vec<String>> {
    let contents = fs::read_to_string(path)?;
    Ok(contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect())
}

fn read_token_prompt_file(path: impl AsRef<Path>) -> AppResult<Vec<Vec<u32>>> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)?;
    let mut prompts = Vec::new();

    for (line_index, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut prompt = Vec::new();
        for token in line.split_whitespace() {
            prompt.push(token.parse::<u32>().map_err(|error| {
                invalid_input(format!(
                    "token prompts file {} line {} token {token:?} is not a u32: {error}",
                    path.display(),
                    line_index + 1
                ))
            })?);
        }
        prompts.push(prompt);
    }

    if prompts.is_empty() {
        return Err(invalid_input(format!(
            "token prompts file {} did not contain any prompts",
            path.display()
        )));
    }

    Ok(prompts)
}

fn read_forced_target_pair_file(
    path: impl AsRef<Path>,
    context: &str,
) -> AppResult<Vec<(String, String)>> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)?;
    let mut pairs = Vec::new();

    for (line_index, line) in contents.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let Some((prompt, target)) = line.split_once('\t') else {
            return Err(invalid_input(format!(
                "{context} pairs file {} line {} must contain a tab-separated prompt and target",
                path.display(),
                line_index + 1
            )));
        };
        if prompt.is_empty() {
            return Err(invalid_input(format!(
                "{context} pairs file {} line {} has an empty prompt",
                path.display(),
                line_index + 1
            )));
        }
        if target.is_empty() {
            return Err(invalid_input(format!(
                "{context} pairs file {} line {} has an empty target",
                path.display(),
                line_index + 1
            )));
        }
        pairs.push((prompt.to_string(), target.to_string()));
    }

    if pairs.is_empty() {
        return Err(invalid_input(format!(
            "{context} pairs file {} did not contain any pairs",
            path.display()
        )));
    }

    Ok(pairs)
}

fn first_prompt(cli: &ChatCli) -> &str {
    cli.prompts
        .first()
        .map(String::as_str)
        .expect("chat cli always has at least one prompt")
}

fn model_eos_token_id(model_dir: &Path) -> AppResult<Option<u32>> {
    Ok(TekkenTokenizer::open(model_dir)?.eos_token_id())
}

fn print_chat_result(result: &inference::ChatInferenceResult, system_prompt: &SystemPrompt) {
    println!(
        "  use_default_system={}",
        matches!(system_prompt, SystemPrompt::DefaultFromModel)
    );
    print_token_window("prompt_tokens", &result.prompt_tokens);
    println!("  generated_tokens={:?}", result.generated_tokens);
    println!("  generated_text={:?}", result.generated_text);
    print_generation_timings(result.timings, result.generated_tokens.len());
    println!(
        "  finish_reason={}",
        finish_reason_label(result.finish_reason)
    );
    for step in &result.steps {
        println!(
            "  step={} token={} logit={:.8} top_logits={:?}",
            step.step, step.token_id, step.logit, step.top_logits
        );
    }
}

fn print_text_result(result: &inference::TextGenerationResult) {
    print_token_window("prompt_tokens", &result.prompt_tokens);
    println!("  generated_tokens={:?}", result.generated_tokens);
    println!("  generated_text={:?}", result.generated_text);
    print_generation_timings(result.timings, result.generated_tokens.len());
    println!(
        "  finish_reason={}",
        finish_reason_label(result.finish_reason)
    );
    for step in &result.steps {
        println!(
            "  step={} token={} logit={:.8} top_logits={:?}",
            step.step, step.token_id, step.logit, step.top_logits
        );
    }
}

fn print_text_logits_result(result: &inference::TextGenerationLogitsResult) {
    print_token_window("prompt_tokens", &result.result.prompt_tokens);
    println!("  backend={}", backend_label(result.result.backend));
    println!("  logits_len={}", result.result.logits.len());
    println!("  logsumexp={:.8}", result.result.logsumexp);
    for (rank, token) in result.result.top_logits.iter().enumerate() {
        println!(
            "  top[{rank}] token={} logit={:.8} logprob={:.8}",
            token.token_id, token.logit, token.logprob
        );
    }
}

fn print_text_trace_result(result: &inference::TextGenerationBackendLogitsTraceResult) {
    print_token_window("prompt_tokens", &result.trace.prompt_tokens);
    println!("  backend={}", backend_label(result.trace.backend));
    println!("  generated_tokens={:?}", result.trace.generated_tokens);
    println!("  generated_text={:?}", result.generated_text);
    println!("  all_text_skip_special={:?}", result.all_text_skip_special);
    println!(
        "  finish_reason={}",
        finish_reason_label(result.trace.finish_reason)
    );
    for step in &result.trace.steps {
        println!(
            "  step={} token={} logit={:.8} logsumexp={:.8}",
            step.step, step.token_id, step.logit, step.logsumexp
        );
        for (rank, token) in step.top_logits.iter().enumerate() {
            println!(
                "    top[{rank}] token={} logit={:.8} logprob={:.8}",
                token.token_id, token.logit, token.logprob
            );
        }
    }
}

fn print_text_logits_comparison_result(
    result: &inference::TextGenerationBackendLogitsComparisonResult,
) {
    print_token_window("prompt_tokens", &result.comparison.prompt_tokens);
    println!(
        "  reference={} candidate={}",
        backend_label(result.comparison.reference.backend),
        backend_label(result.comparison.candidate.backend)
    );
    println!(
        "  reference_token={} candidate_token={} token_matches={}",
        result.comparison.reference_token_id,
        result.comparison.candidate_token_id,
        result.comparison.token_matches
    );
    println!(
        "  top_tokens_match={}",
        result.comparison.top_tokens_match()
    );
    println!("  kl_divergence={:.12}", result.comparison.kl_divergence);
    println!("  max_abs_diff={:.8}", result.comparison.max_abs_diff);
    println!("  mean_abs_diff={:.8}", result.comparison.mean_abs_diff);
    for (rank, token) in result.comparison.reference.top_logits.iter().enumerate() {
        println!(
            "  reference_top[{rank}] token={} logit={:.8} logprob={:.8}",
            token.token_id, token.logit, token.logprob
        );
    }
    for (rank, token) in result.comparison.candidate.top_logits.iter().enumerate() {
        println!(
            "  candidate_top[{rank}] token={} logit={:.8} logprob={:.8}",
            token.token_id, token.logit, token.logprob
        );
    }
}

fn print_text_comparison_result(result: &inference::TextGenerationBackendComparisonResult) {
    let summary = result.comparison.summary();
    print_token_window("prompt_tokens", &result.comparison.prompt_tokens);
    println!(
        "  reference={} candidate={} driver={}",
        backend_label(result.comparison.reference_backend),
        backend_label(result.comparison.candidate_backend),
        driver_label(result.comparison.driver)
    );
    println!(
        "  generated_tokens={:?}",
        result.comparison.generated_tokens
    );
    println!("  generated_text={:?}", result.generated_text);
    println!("  all_text_skip_special={:?}", result.all_text_skip_special);
    println!(
        "  token_match_count={}/{}",
        summary.token_match_count, summary.step_count
    );
    println!("  top_tokens_match={}", summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", summary.mean_kl());
    println!("  max_kl_divergence={:.12}", summary.max_kl);
    println!("  max_abs_diff={:.8}", summary.max_abs_diff);
    for step in &result.comparison.steps {
        println!(
            "  step={} prefix_len={} reference_token={} candidate_token={} token_matches={} kl_divergence={:.12} max_abs_diff={:.8}",
            step.step,
            step.prefix_len,
            step.reference_token_id,
            step.candidate_token_id,
            step.token_matches,
            step.kl_divergence,
            step.max_abs_diff
        );
    }
}

fn print_text_forced_comparison_result(
    result: &inference::TextGenerationBackendForcedComparisonResult,
    tokenizer: Option<&TekkenTokenizer>,
) -> AppResult<()> {
    println!("  sequence_text={:?}", result.sequence_text);
    print_generation_forced_comparison_result(&result.comparison, tokenizer)
}

fn print_text_forced_target_comparison_result(
    result: &inference::TextGenerationBackendForcedTargetComparisonResult,
    tokenizer: Option<&TekkenTokenizer>,
) -> AppResult<()> {
    println!("  prompt_text={:?}", result.prompt_text);
    println!("  target_text={:?}", result.target_text);
    print_generation_forced_comparison_result(&result.comparison, tokenizer)
}

fn print_chat_forced_target_comparison_result(
    result: &inference::ChatBackendForcedTargetComparisonResult,
    tokenizer: Option<&TekkenTokenizer>,
) -> AppResult<()> {
    println!("  formatted_prompt={:?}", result.formatted_prompt);
    println!("  target_text={:?}", result.target_text);
    print_generation_forced_comparison_result(&result.comparison, tokenizer)
}

fn print_generation_result(
    result: &inference::GenerationResult,
    tokenizer: Option<&TekkenTokenizer>,
) -> AppResult<()> {
    print_token_window("prompt_tokens", &result.prompt_tokens);
    println!("  generated_tokens={:?}", result.generated_tokens);
    if let Some(tokenizer) = tokenizer {
        println!(
            "  generated_text={:?}",
            tokenizer.decode_lossy(&result.generated_tokens)?
        );
        println!(
            "  all_text_skip_special={:?}",
            tokenizer.decode_lossy(&result.all_tokens)?
        );
    }
    print_generation_timings(result.timings, result.generated_tokens.len());
    println!(
        "  finish_reason={}",
        finish_reason_label(result.finish_reason)
    );
    for step in &result.steps {
        println!(
            "  step={} token={} logit={:.8} top_logits={:?}",
            step.step, step.token_id, step.logit, step.top_logits
        );
    }
    Ok(())
}

fn print_generation_timings(timings: inference::GenerationTimings, generated_token_count: usize) {
    println!("  prefill_seconds={:.6}", timings.prefill_seconds);
    println!("  decode_seconds={:.6}", timings.decode_seconds);
    println!(
        "  decode_tokens_per_second={:.3}",
        timings
            .decode_tokens_per_second(generated_token_count)
            .unwrap_or(0.0)
    );
    println!(
        "  total_tokens_per_second={:.3}",
        timings
            .total_tokens_per_second(generated_token_count)
            .unwrap_or(0.0)
    );
}

fn print_generation_logits_result(
    result: &inference::GenerationLogitsResult,
    tokenizer: Option<&TekkenTokenizer>,
) -> AppResult<()> {
    print_token_window("prompt_tokens", &result.prompt_tokens);
    if let Some(tokenizer) = tokenizer {
        println!(
            "  prompt_text_skip_special={:?}",
            tokenizer.decode_lossy(&result.prompt_tokens)?
        );
    }
    println!("  logits_len={}", result.logits.len());
    println!("  logsumexp={:.8}", result.logsumexp);
    for (rank, token) in result.top_logits.iter().enumerate() {
        println!(
            "  top[{rank}] token={} logit={:.8} logprob={:.8}",
            token.token_id, token.logit, token.logprob
        );
    }
    Ok(())
}

fn print_generation_trace_result(
    result: &inference::GenerationLogitsTraceResult,
    tokenizer: Option<&TekkenTokenizer>,
) -> AppResult<()> {
    print_token_window("prompt_tokens", &result.prompt_tokens);
    println!("  generated_tokens={:?}", result.generated_tokens);
    if let Some(tokenizer) = tokenizer {
        println!(
            "  generated_text={:?}",
            tokenizer.decode_lossy(&result.generated_tokens)?
        );
        println!(
            "  all_text_skip_special={:?}",
            tokenizer.decode_lossy(&result.all_tokens)?
        );
    }
    println!(
        "  finish_reason={}",
        finish_reason_label(result.finish_reason)
    );
    for step in &result.steps {
        println!(
            "  step={} token={} logit={:.8} logsumexp={:.8}",
            step.step, step.token_id, step.logit, step.logsumexp
        );
        for (rank, token) in step.top_logits.iter().enumerate() {
            println!(
                "    top[{rank}] token={} logit={:.8} logprob={:.8}",
                token.token_id, token.logit, token.logprob
            );
        }
    }
    Ok(())
}

fn print_generation_logits_comparison_result(
    result: &inference::GenerationBackendLogitsComparisonResult,
    tokenizer: Option<&TekkenTokenizer>,
) -> AppResult<()> {
    print_token_window("prompt_tokens", &result.prompt_tokens);
    if let Some(tokenizer) = tokenizer {
        println!(
            "  prompt_text_skip_special={:?}",
            tokenizer.decode_lossy(&result.prompt_tokens)?
        );
    }
    println!(
        "  reference={} candidate={}",
        backend_label(result.reference.backend),
        backend_label(result.candidate.backend)
    );
    println!(
        "  reference_token={} candidate_token={} token_matches={}",
        result.reference_token_id, result.candidate_token_id, result.token_matches
    );
    println!("  top_tokens_match={}", result.top_tokens_match());
    println!("  kl_divergence={:.12}", result.kl_divergence);
    println!("  max_abs_diff={:.8}", result.max_abs_diff);
    println!("  mean_abs_diff={:.8}", result.mean_abs_diff);
    for (rank, token) in result.reference.top_logits.iter().enumerate() {
        println!(
            "  reference_top[{rank}] token={} logit={:.8} logprob={:.8}",
            token.token_id, token.logit, token.logprob
        );
    }
    for (rank, token) in result.candidate.top_logits.iter().enumerate() {
        println!(
            "  candidate_top[{rank}] token={} logit={:.8} logprob={:.8}",
            token.token_id, token.logit, token.logprob
        );
    }
    Ok(())
}

fn print_generation_comparison_result(
    result: &inference::GenerationBackendComparisonResult,
    tokenizer: Option<&TekkenTokenizer>,
) -> AppResult<()> {
    let summary = result.summary();
    print_token_window("prompt_tokens", &result.prompt_tokens);
    println!(
        "  reference={} candidate={} driver={}",
        backend_label(result.reference_backend),
        backend_label(result.candidate_backend),
        driver_label(result.driver)
    );
    println!("  generated_tokens={:?}", result.generated_tokens);
    if let Some(tokenizer) = tokenizer {
        println!(
            "  generated_text={:?}",
            tokenizer.decode_lossy(&result.generated_tokens)?
        );
        println!(
            "  all_text_skip_special={:?}",
            tokenizer.decode_lossy(&result.all_tokens)?
        );
    }
    println!(
        "  token_match_count={}/{}",
        summary.token_match_count, summary.step_count
    );
    println!("  top_tokens_match={}", summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", summary.mean_kl());
    println!("  max_kl_divergence={:.12}", summary.max_kl);
    println!("  max_abs_diff={:.8}", summary.max_abs_diff);
    for step in &result.steps {
        println!(
            "  step={} prefix_len={} reference_token={} candidate_token={} token_matches={} kl_divergence={:.12} max_abs_diff={:.8}",
            step.step,
            step.prefix_len,
            step.reference_token_id,
            step.candidate_token_id,
            step.token_matches,
            step.kl_divergence,
            step.max_abs_diff
        );
    }
    Ok(())
}

fn print_generation_forced_comparison_result(
    result: &inference::GenerationBackendForcedComparisonResult,
    tokenizer: Option<&TekkenTokenizer>,
) -> AppResult<()> {
    let summary = result.summary();
    print_token_window("sequence_tokens", &result.sequence_tokens);
    println!(
        "  reference={} candidate={}",
        backend_label(result.reference_backend),
        backend_label(result.candidate_backend)
    );
    println!("  forced_tokens={:?}", result.forced_tokens);
    if let Some(tokenizer) = tokenizer {
        println!(
            "  sequence_text_skip_special={:?}",
            tokenizer.decode_lossy(&result.sequence_tokens)?
        );
        println!(
            "  forced_text_skip_special={:?}",
            tokenizer.decode_lossy(&result.forced_tokens)?
        );
    }
    println!(
        "  token_match_count={}/{}",
        summary.token_match_count, summary.step_count
    );
    println!(
        "  forced_reference_match_count={}/{}",
        summary.forced_reference_match_count, summary.step_count
    );
    println!(
        "  forced_candidate_match_count={}/{}",
        summary.forced_candidate_match_count, summary.step_count
    );
    println!("  top_tokens_match={}", summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", summary.mean_kl());
    println!("  max_kl_divergence={:.12}", summary.max_kl);
    println!("  max_abs_diff={:.8}", summary.max_abs_diff);
    println!(
        "  mean_forced_logprob_abs_diff={:.12}",
        summary.mean_forced_logprob_abs_diff()
    );
    println!(
        "  max_forced_logprob_abs_diff={:.8}",
        summary.max_forced_logprob_abs_diff
    );
    for step in &result.steps {
        println!(
            "  step={} prefix_len={} forced_token={} reference_token={} candidate_token={} token_matches={} forced_reference_match={} forced_candidate_match={} kl_divergence={:.12} max_abs_diff={:.8} reference_forced_logprob={:.8} candidate_forced_logprob={:.8}",
            step.step,
            step.prefix_len,
            step.forced_token_id,
            step.reference_token_id,
            step.candidate_token_id,
            step.token_matches,
            step.forced_token_matches_reference,
            step.forced_token_matches_candidate,
            step.kl_divergence,
            step.max_abs_diff,
            step.reference_forced_logprob,
            step.candidate_forced_logprob
        );
    }
    Ok(())
}

fn print_chat_logits_result(result: &ChatLogitsResult, system_prompt: &SystemPrompt) {
    println!(
        "  use_default_system={}",
        matches!(system_prompt, SystemPrompt::DefaultFromModel)
    );
    print_token_window("prompt_tokens", &result.prompt_tokens);
    println!("  logits_len={}", result.logits.len());
    println!("  logsumexp={:.8}", result.logsumexp);
    for (rank, token) in result.top_logits.iter().enumerate() {
        println!(
            "  top[{rank}] token={} logit={:.8} logprob={:.8}",
            token.token_id, token.logit, token.logprob
        );
    }
}

fn print_chat_logits_comparison_result(
    result: &inference::ChatBackendLogitsComparisonResult,
    system_prompt: &SystemPrompt,
) {
    println!(
        "  use_default_system={}",
        matches!(system_prompt, SystemPrompt::DefaultFromModel)
    );
    print_token_window("prompt_tokens", &result.comparison.prompt_tokens);
    println!(
        "  reference={} candidate={}",
        backend_label(result.comparison.reference.backend),
        backend_label(result.comparison.candidate.backend)
    );
    println!(
        "  reference_token={} candidate_token={} token_matches={}",
        result.comparison.reference_token_id,
        result.comparison.candidate_token_id,
        result.comparison.token_matches
    );
    println!(
        "  top_tokens_match={}",
        result.comparison.top_tokens_match()
    );
    println!("  kl_divergence={:.12}", result.comparison.kl_divergence);
    println!("  max_abs_diff={:.8}", result.comparison.max_abs_diff);
    println!("  mean_abs_diff={:.8}", result.comparison.mean_abs_diff);
    for (rank, token) in result.comparison.reference.top_logits.iter().enumerate() {
        println!(
            "  reference_top[{rank}] token={} logit={:.8} logprob={:.8}",
            token.token_id, token.logit, token.logprob
        );
    }
    for (rank, token) in result.comparison.candidate.top_logits.iter().enumerate() {
        println!(
            "  candidate_top[{rank}] token={} logit={:.8} logprob={:.8}",
            token.token_id, token.logit, token.logprob
        );
    }
}

fn print_chat_trace_result(
    result: &inference::ChatBackendLogitsTraceResult,
    system_prompt: &SystemPrompt,
) {
    println!(
        "  use_default_system={}",
        matches!(system_prompt, SystemPrompt::DefaultFromModel)
    );
    print_token_window("prompt_tokens", &result.trace.prompt_tokens);
    println!("  generated_tokens={:?}", result.trace.generated_tokens);
    println!("  generated_text={:?}", result.generated_text);
    println!(
        "  finish_reason={}",
        finish_reason_label(result.trace.finish_reason)
    );
    for step in &result.trace.steps {
        println!(
            "  step={} token={} logit={:.8} logsumexp={:.8}",
            step.step, step.token_id, step.logit, step.logsumexp
        );
        for (rank, token) in step.top_logits.iter().enumerate() {
            println!(
                "    top[{rank}] token={} logit={:.8} logprob={:.8}",
                token.token_id, token.logit, token.logprob
            );
        }
    }
}

fn print_chat_compare_result(
    prompt: &str,
    candidate_backend: InferenceBackend,
    driver: GenerationComparisonDriver,
    result: &ChatBackendComparisonResult,
    summary: &GenerationBackendComparisonSummary,
) {
    println!(
        "Ministral chat comparison: candidate={} driver={} user_prompt={prompt:?}",
        backend_label(candidate_backend),
        driver_label(driver)
    );
    print_token_window("prompt_tokens", &result.comparison.prompt_tokens);
    println!(
        "  generated_tokens={:?}",
        result.comparison.generated_tokens
    );
    println!("  generated_text={:?}", result.generated_text);
    println!(
        "  step_count={} token_match_count={} top_tokens_match={} mean_kl={:.12} max_kl={:.12} max_abs_diff={:.8}",
        summary.step_count,
        summary.token_match_count,
        summary.top_tokens_match,
        summary.mean_kl(),
        summary.max_kl,
        summary.max_abs_diff
    );
    for step in &result.comparison.steps {
        println!(
            "  step={} reference_token={} candidate_token={} token_matches={} top_tokens_match={} kl_divergence={:.12} max_abs_diff={:.8}",
            step.step,
            step.reference_token_id,
            step.candidate_token_id,
            step.token_matches,
            step.top_tokens_match(),
            step.kl_divergence,
            step.max_abs_diff
        );
    }
}

fn print_token_window(label: &str, tokens: &[u32]) {
    let prefix: Vec<u32> = tokens.iter().take(8).copied().collect();
    let suffix_start = tokens.len().saturating_sub(8);
    let suffix: Vec<u32> = tokens.iter().skip(suffix_start).copied().collect();
    println!("  {label}_len={}", tokens.len());
    println!("  {label}_prefix={prefix:?}");
    println!("  {label}_suffix={suffix:?}");
}

fn print_logits_summary(label: &str, summary: &GenerationBackendLogitsComparisonSuiteSummary) {
    println!("{label}:");
    println!(
        "  token_match_count={}/{}",
        summary.token_match_count, summary.prompt_count
    );
    println!("  top_tokens_match={}", summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", summary.mean_kl());
    println!("  max_kl_divergence={:.12}", summary.max_kl);
    println!("  max_abs_diff={:.8}", summary.max_abs_diff);
}

fn print_compare_suite_summary(label: &str, summary: &ChatCompareSuiteSummary) {
    println!("{label}:");
    println!(
        "  token_match_count={}/{}",
        summary.token_match_count, summary.step_count
    );
    println!("  top_tokens_match={}", summary.top_tokens_match);
    println!("  mean_kl_divergence={:.12}", summary.mean_kl());
    println!("  max_kl_divergence={:.12}", summary.max_kl);
    println!("  max_abs_diff={:.8}", summary.max_abs_diff);
}

fn print_forced_suite_extra_summary(
    label: &str,
    summary: &inference::GenerationBackendForcedComparisonSuiteSummary,
) {
    println!(
        "  {label}_forced_reference_match_count={}/{}",
        summary.summary.forced_reference_match_count, summary.summary.step_count
    );
    println!(
        "  {label}_forced_candidate_match_count={}/{}",
        summary.summary.forced_candidate_match_count, summary.summary.step_count
    );
    println!(
        "  {label}_mean_forced_logprob_abs_diff={:.12}",
        summary.summary.mean_forced_logprob_abs_diff()
    );
    println!(
        "  {label}_max_forced_logprob_abs_diff={:.8}",
        summary.summary.max_forced_logprob_abs_diff
    );
}

#[derive(Debug, Default)]
struct ChatCompareSuiteSummary {
    prompt_count: usize,
    step_count: usize,
    token_match_count: usize,
    kl_sum: f64,
    max_kl: f64,
    max_abs_diff: f32,
    top_tokens_match: bool,
}

impl ChatCompareSuiteSummary {
    fn observe(&mut self, summary: &GenerationBackendComparisonSummary) {
        if self.prompt_count == 0 {
            self.top_tokens_match = true;
        }
        self.prompt_count += 1;
        self.step_count += summary.step_count;
        self.token_match_count += summary.token_match_count;
        self.kl_sum += summary.kl_sum;
        self.max_kl = self.max_kl.max(summary.max_kl);
        self.max_abs_diff = self.max_abs_diff.max(summary.max_abs_diff);
        self.top_tokens_match &= summary.top_tokens_match;
    }

    fn mean_kl(&self) -> f64 {
        if self.step_count == 0 {
            0.0
        } else {
            self.kl_sum / self.step_count as f64
        }
    }

    fn token_ids_match(&self) -> bool {
        self.token_match_count == self.step_count
    }
}

fn chat_compare_summary_from_generation_suite(
    summary: &GenerationBackendComparisonSuiteSummary,
) -> ChatCompareSuiteSummary {
    ChatCompareSuiteSummary {
        prompt_count: summary.prompt_count,
        step_count: summary.summary.step_count,
        token_match_count: summary.summary.token_match_count,
        kl_sum: summary.summary.kl_sum,
        max_kl: summary.summary.max_kl,
        max_abs_diff: summary.summary.max_abs_diff,
        top_tokens_match: summary.summary.top_tokens_match,
    }
}

fn chat_compare_summary_from_forced_suite(
    summary: &inference::GenerationBackendForcedComparisonSuiteSummary,
) -> ChatCompareSuiteSummary {
    ChatCompareSuiteSummary {
        prompt_count: summary.prompt_count,
        step_count: summary.summary.step_count,
        token_match_count: summary.summary.token_match_count,
        kl_sum: summary.summary.kl_sum,
        max_kl: summary.summary.max_kl,
        max_abs_diff: summary.summary.max_abs_diff,
        top_tokens_match: summary.summary.top_tokens_match,
    }
}

fn enforce_chat_compare_thresholds(
    summary: &ChatCompareSuiteSummary,
    thresholds: &ChatCompareThresholds,
) -> AppResult<()> {
    if thresholds.require_token_match && !summary.token_ids_match() {
        return Err(invalid_data(format!(
            "token match threshold failed: {}/{} steps matched",
            summary.token_match_count, summary.step_count
        )));
    }
    if thresholds.require_top_tokens_match && !summary.top_tokens_match {
        return Err(invalid_data("top token ids threshold failed"));
    }
    if let Some(max_mean_kl) = thresholds.max_mean_kl {
        if summary.mean_kl() > max_mean_kl {
            return Err(invalid_data(format!(
                "mean KL threshold failed: {:.12} > {:.12}",
                summary.mean_kl(),
                max_mean_kl
            )));
        }
    }
    if let Some(max_max_kl) = thresholds.max_max_kl {
        if summary.max_kl > max_max_kl {
            return Err(invalid_data(format!(
                "max KL threshold failed: {:.12} > {:.12}",
                summary.max_kl, max_max_kl
            )));
        }
    }
    if let Some(max_max_abs_diff) = thresholds.max_max_abs_diff {
        if summary.max_abs_diff > max_max_abs_diff {
            return Err(invalid_data(format!(
                "max abs diff threshold failed: {:.8} > {:.8}",
                summary.max_abs_diff, max_max_abs_diff
            )));
        }
    }

    Ok(())
}

fn enforce_token_logits_compare_thresholds(
    summary: &GenerationBackendLogitsComparisonSuiteSummary,
    thresholds: &ChatCompareThresholds,
) -> AppResult<()> {
    if thresholds.require_token_match && summary.token_match_count != summary.prompt_count {
        return Err(invalid_data(format!(
            "token match threshold failed: {}/{} prompts matched",
            summary.token_match_count, summary.prompt_count
        )));
    }
    if thresholds.require_top_tokens_match && !summary.top_tokens_match {
        return Err(invalid_data("top token ids threshold failed"));
    }
    if let Some(max_mean_kl) = thresholds.max_mean_kl {
        if summary.mean_kl() > max_mean_kl {
            return Err(invalid_data(format!(
                "mean KL threshold failed: {:.12} > {:.12}",
                summary.mean_kl(),
                max_mean_kl
            )));
        }
    }
    if let Some(max_max_kl) = thresholds.max_max_kl {
        if summary.max_kl > max_max_kl {
            return Err(invalid_data(format!(
                "max KL threshold failed: {:.12} > {:.12}",
                summary.max_kl, max_max_kl
            )));
        }
    }
    if let Some(max_max_abs_diff) = thresholds.max_max_abs_diff {
        if summary.max_abs_diff > max_max_abs_diff {
            return Err(invalid_data(format!(
                "max abs diff threshold failed: {:.8} > {:.8}",
                summary.max_abs_diff, max_max_abs_diff
            )));
        }
    }

    Ok(())
}

fn write_chat_generation_report(
    path: &Path,
    results: &[(String, inference::ChatInferenceResult)],
    metadata: &InferenceReportMetadata<'_>,
    prompt_file_paths: &[PathBuf],
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_inference_report_metadata(&mut out, metadata);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, (prompt, result)) in results.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "user_prompt", prompt, 6, true);
        push_json_field_string(&mut out, "backend", backend_label(result.backend), 6, true);
        push_json_field_string(
            &mut out,
            "formatted_prompt",
            &result.formatted_prompt,
            6,
            true,
        );
        push_json_field_u32_array(&mut out, "prompt_tokens", &result.prompt_tokens, 6, true);
        push_json_field_u32_array(
            &mut out,
            "generated_tokens",
            &result.generated_tokens,
            6,
            true,
        );
        push_json_field_u32_array(&mut out, "all_tokens", &result.all_tokens, 6, true);
        push_json_field_runtime_memory_stats(&mut out, "memory", &result.memory_stats, 6, true);
        push_json_field_generation_timings(
            &mut out,
            "timings",
            result.timings,
            result.generated_tokens.len(),
            6,
            true,
        );
        push_json_field_string(&mut out, "generated_text", &result.generated_text, 6, true);
        push_json_field_string(
            &mut out,
            "all_text_skip_special",
            &result.all_text_skip_special,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "finish_reason",
            &finish_reason_label(result.finish_reason),
            6,
            true,
        );
        push_json_field_generation_steps(&mut out, "steps", &result.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_ministral_eval_report(
    path: &Path,
    model_dir: &Path,
    candidate_backend: InferenceBackend,
    candidate_export_dir: Option<&Path>,
    driver: GenerationComparisonDriver,
    system_prompt: &SystemPrompt,
    max_new_tokens: usize,
    top_k: usize,
    logits_top_k: usize,
    prompt_file_paths: &[PathBuf],
    pair_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
    text_eval_suite: &inference::TextGenerationBackendEvalSuiteResult,
    text_generation_summary: &ChatCompareSuiteSummary,
    chat_eval_suite: &inference::ChatBackendEvalSuiteResult,
    chat_generation_summary: &ChatCompareSuiteSummary,
    text_forced_target_suite: Option<
        &inference::TextGenerationBackendForcedTargetComparisonSuiteResult,
    >,
    chat_forced_target_suite: Option<&inference::ChatBackendForcedTargetComparisonSuiteResult>,
) -> AppResult<()> {
    let tokenizer = if text_forced_target_suite.is_some() || chat_forced_target_suite.is_some() {
        Some(TekkenTokenizer::open(model_dir)?)
    } else {
        None
    };
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_string(
        &mut out,
        "model_dir",
        &model_dir.display().to_string(),
        2,
        true,
    );
    push_json_field_string(
        &mut out,
        "candidate_backend",
        backend_label(candidate_backend),
        2,
        true,
    );
    if let Some(candidate_export_dir) = candidate_export_dir {
        push_json_field_string(
            &mut out,
            "candidate_export_dir",
            &candidate_export_dir.display().to_string(),
            2,
            true,
        );
    }
    push_json_field_string(&mut out, "driver", driver_label(driver), 2, true);
    push_json_field_usize(&mut out, "max_new_tokens", max_new_tokens, 2, true);
    push_json_field_usize(&mut out, "top_k", top_k, 2, true);
    push_json_field_usize(&mut out, "logits_top_k", logits_top_k, 2, true);
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, true);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    push_json_field_path_array(
        &mut out,
        "forced_target_pair_files",
        pair_file_paths,
        2,
        true,
    );
    push_json_field_usize(
        &mut out,
        "prompt_count",
        text_eval_suite.logits.prompts.len(),
        2,
        true,
    );
    push_json_field_usize(
        &mut out,
        "forced_target_pair_count",
        text_forced_target_suite
            .as_ref()
            .map(|suite| suite.prompts.len())
            .or_else(|| {
                chat_forced_target_suite
                    .as_ref()
                    .map(|suite| suite.prompts.len())
            })
            .unwrap_or(0),
        2,
        true,
    );
    push_json_field_string(
        &mut out,
        "system_prompt_mode",
        system_prompt_mode_label(system_prompt),
        2,
        true,
    );
    push_json_field_bool(
        &mut out,
        "use_default_system",
        matches!(system_prompt, SystemPrompt::DefaultFromModel),
        2,
        true,
    );
    push_indent(&mut out, 2);
    out.push_str("\"text_logits\": ");
    push_eval_text_logits_section(&mut out, &text_eval_suite.logits)?;
    out.push_str(",\n");
    push_indent(&mut out, 2);
    out.push_str("\"text_generation\": ");
    push_eval_text_generation_section(
        &mut out,
        &text_eval_suite.generation,
        text_generation_summary,
    )?;
    out.push_str(",\n");
    push_indent(&mut out, 2);
    out.push_str("\"chat_logits\": ");
    push_eval_chat_logits_section(&mut out, &chat_eval_suite.logits)?;
    out.push_str(",\n");
    push_indent(&mut out, 2);
    out.push_str("\"chat_generation\": ");
    push_eval_chat_generation_section(
        &mut out,
        &chat_eval_suite.generation,
        chat_generation_summary,
    )?;
    if let Some(suite) = text_forced_target_suite {
        out.push_str(",\n");
        push_eval_text_forced_target_section(
            &mut out,
            suite,
            tokenizer
                .as_ref()
                .expect("tokenizer is available for forced target eval report"),
        )?;
    }
    if let Some(suite) = chat_forced_target_suite {
        out.push_str(",\n");
        push_eval_chat_forced_target_section(
            &mut out,
            suite,
            tokenizer
                .as_ref()
                .expect("tokenizer is available for forced target eval report"),
        )?;
    }
    out.push_str("\n}\n");
    write_report(path, out)
}

fn push_eval_text_logits_section(
    out: &mut String,
    suite: &inference::TextGenerationBackendLogitsComparisonSuiteResult,
) -> AppResult<()> {
    out.push_str("{\"summary\": ");
    push_generation_logits_compare_suite_summary(out, &suite.summary);
    out.push_str(", \"memory\": ");
    push_memory_stats_summary(
        out,
        &suite.reference_memory_stats,
        &suite.candidate_memory_stats,
    );
    out.push_str(", \"prompts\": [\n");
    for (i, result) in suite.prompts.iter().enumerate() {
        comma_line(out, i, "    ");
        push_text_logits_comparison_prompt_json(out, result, 4)?;
    }
    out.push_str("\n  ]}");
    Ok(())
}

fn push_eval_text_generation_section(
    out: &mut String,
    suite: &inference::TextGenerationBackendComparisonSuiteResult,
    suite_summary: &ChatCompareSuiteSummary,
) -> AppResult<()> {
    out.push_str("{\"summary\": ");
    push_chat_compare_suite_summary(out, suite_summary);
    out.push_str(", \"memory\": ");
    push_memory_stats_summary(
        out,
        &suite.reference_memory_stats,
        &suite.candidate_memory_stats,
    );
    out.push_str(", \"prompts\": [\n");
    for (i, result) in suite.prompts.iter().enumerate() {
        comma_line(out, i, "    ");
        push_text_generation_comparison_prompt_json(out, result, 4)?;
    }
    out.push_str("\n  ]}");
    Ok(())
}

fn push_eval_chat_logits_section(
    out: &mut String,
    suite: &inference::ChatBackendLogitsComparisonSuiteResult,
) -> AppResult<()> {
    out.push_str("{\"summary\": ");
    push_generation_logits_compare_suite_summary(out, &suite.summary);
    out.push_str(", \"memory\": ");
    push_memory_stats_summary(
        out,
        &suite.reference_memory_stats,
        &suite.candidate_memory_stats,
    );
    out.push_str(", \"prompts\": [\n");
    for (i, prompt_result) in suite.prompts.iter().enumerate() {
        comma_line(out, i, "    ");
        push_chat_logits_comparison_prompt_json(out, prompt_result, 4)?;
    }
    out.push_str("\n  ]}");
    Ok(())
}

fn push_eval_chat_generation_section(
    out: &mut String,
    suite: &inference::ChatBackendComparisonSuiteResult,
    suite_summary: &ChatCompareSuiteSummary,
) -> AppResult<()> {
    out.push_str("{\"summary\": ");
    push_chat_compare_suite_summary(out, suite_summary);
    out.push_str(", \"memory\": ");
    push_memory_stats_summary(
        out,
        &suite.reference_memory_stats,
        &suite.candidate_memory_stats,
    );
    out.push_str(", \"prompts\": [\n");
    for (i, prompt_result) in suite.prompts.iter().enumerate() {
        comma_line(out, i, "    ");
        push_chat_generation_comparison_prompt_json(out, prompt_result, 4)?;
    }
    out.push_str("\n  ]}");
    Ok(())
}

fn push_text_logits_comparison_prompt_json(
    out: &mut String,
    result: &inference::TextGenerationBackendLogitsComparisonResult,
    indent: usize,
) -> AppResult<()> {
    let comparison = &result.comparison;
    out.push_str("{\n");
    push_json_field_string(out, "prompt_text", &result.prompt_text, indent + 2, true);
    push_json_field_u32_array(
        out,
        "prompt_tokens",
        &comparison.prompt_tokens,
        indent + 2,
        true,
    );
    push_generation_logits_comparison_fields(out, comparison, indent + 2);
    out.push('\n');
    push_indent(out, indent);
    out.push('}');
    Ok(())
}

fn push_text_generation_comparison_prompt_json(
    out: &mut String,
    result: &inference::TextGenerationBackendComparisonResult,
    indent: usize,
) -> AppResult<()> {
    let comparison = &result.comparison;
    let summary = comparison.summary();
    out.push_str("{\n");
    push_json_field_string(out, "prompt_text", &result.prompt_text, indent + 2, true);
    push_json_field_u32_array(
        out,
        "prompt_tokens",
        &comparison.prompt_tokens,
        indent + 2,
        true,
    );
    push_json_field_u32_array(
        out,
        "generated_tokens",
        &comparison.generated_tokens,
        indent + 2,
        true,
    );
    push_json_field_u32_array(out, "all_tokens", &comparison.all_tokens, indent + 2, true);
    push_json_field_string(
        out,
        "generated_text",
        &result.generated_text,
        indent + 2,
        true,
    );
    push_json_field_string(
        out,
        "all_text_skip_special",
        &result.all_text_skip_special,
        indent + 2,
        true,
    );
    push_generation_comparison_fields(out, comparison, &summary, indent + 2);
    out.push('\n');
    push_indent(out, indent);
    out.push('}');
    Ok(())
}

fn push_chat_logits_comparison_prompt_json(
    out: &mut String,
    prompt_result: &inference::ChatBackendLogitsComparisonPromptResult,
    indent: usize,
) -> AppResult<()> {
    let result = &prompt_result.result;
    let comparison = &result.comparison;
    out.push_str("{\n");
    push_json_field_string(
        out,
        "user_prompt",
        &prompt_result.user_prompt,
        indent + 2,
        true,
    );
    push_json_field_string(
        out,
        "formatted_prompt",
        &result.formatted_prompt,
        indent + 2,
        true,
    );
    push_json_field_u32_array(
        out,
        "prompt_tokens",
        &comparison.prompt_tokens,
        indent + 2,
        true,
    );
    push_generation_logits_comparison_fields(out, comparison, indent + 2);
    out.push('\n');
    push_indent(out, indent);
    out.push('}');
    Ok(())
}

fn push_chat_generation_comparison_prompt_json(
    out: &mut String,
    prompt_result: &inference::ChatBackendComparisonPromptResult,
    indent: usize,
) -> AppResult<()> {
    let result = &prompt_result.result;
    let comparison = &result.comparison;
    let summary = comparison.summary();
    out.push_str("{\n");
    push_json_field_string(
        out,
        "user_prompt",
        &prompt_result.user_prompt,
        indent + 2,
        true,
    );
    push_json_field_string(
        out,
        "formatted_prompt",
        &result.formatted_prompt,
        indent + 2,
        true,
    );
    push_json_field_u32_array(
        out,
        "prompt_tokens",
        &comparison.prompt_tokens,
        indent + 2,
        true,
    );
    push_json_field_u32_array(
        out,
        "generated_tokens",
        &comparison.generated_tokens,
        indent + 2,
        true,
    );
    push_json_field_u32_array(out, "all_tokens", &comparison.all_tokens, indent + 2, true);
    push_json_field_string(
        out,
        "generated_text",
        &result.generated_text,
        indent + 2,
        true,
    );
    push_json_field_string(
        out,
        "all_text_skip_special",
        &result.all_text_skip_special,
        indent + 2,
        true,
    );
    push_generation_comparison_fields(out, comparison, &summary, indent + 2);
    out.push('\n');
    push_indent(out, indent);
    out.push('}');
    Ok(())
}

fn push_generation_logits_comparison_fields(
    out: &mut String,
    comparison: &inference::GenerationBackendLogitsComparisonResult,
    indent: usize,
) {
    push_json_field_string(
        out,
        "reference_backend",
        backend_label(comparison.reference.backend),
        indent,
        true,
    );
    push_json_field_string(
        out,
        "candidate_backend",
        backend_label(comparison.candidate.backend),
        indent,
        true,
    );
    push_json_field_u32(
        out,
        "reference_token_id",
        comparison.reference_token_id,
        indent,
        true,
    );
    push_json_field_u32(
        out,
        "candidate_token_id",
        comparison.candidate_token_id,
        indent,
        true,
    );
    push_json_field_bool(out, "token_matches", comparison.token_matches, indent, true);
    push_json_field_bool(
        out,
        "top_tokens_match",
        comparison.top_tokens_match(),
        indent,
        true,
    );
    push_json_field_f64(out, "kl_divergence", comparison.kl_divergence, indent, true);
    push_json_field_f32(out, "max_abs_diff", comparison.max_abs_diff, indent, true);
    push_json_field_f32(out, "mean_abs_diff", comparison.mean_abs_diff, indent, true);
    push_json_field_token_logits(
        out,
        "reference_top_logits",
        &comparison.reference.top_logits,
        indent,
        true,
    );
    push_json_field_token_logits(
        out,
        "candidate_top_logits",
        &comparison.candidate.top_logits,
        indent,
        false,
    );
}

fn push_generation_comparison_fields(
    out: &mut String,
    comparison: &inference::GenerationBackendComparisonResult,
    summary: &GenerationBackendComparisonSummary,
    indent: usize,
) {
    push_json_field_string(out, "driver", driver_label(comparison.driver), indent, true);
    push_json_field_string(
        out,
        "reference_backend",
        backend_label(comparison.reference_backend),
        indent,
        true,
    );
    push_json_field_string(
        out,
        "candidate_backend",
        backend_label(comparison.candidate_backend),
        indent,
        true,
    );
    push_json_field_chat_compare_summary(out, "summary", summary, indent, true);
    push_json_field_chat_compare_steps(out, "steps", &comparison.steps, indent, false);
}

fn push_eval_text_forced_target_section(
    out: &mut String,
    suite: &inference::TextGenerationBackendForcedTargetComparisonSuiteResult,
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    push_indent(out, 2);
    out.push_str("\"text_forced_target\": {\"summary\": ");
    push_generation_forced_compare_suite_summary(out, &suite.summary);
    out.push_str(", \"memory\": ");
    push_memory_stats_summary(
        out,
        &suite.reference_memory_stats,
        &suite.candidate_memory_stats,
    );
    out.push_str(", \"prompts\": [\n");
    for (i, result) in suite.prompts.iter().enumerate() {
        let comparison = &result.comparison;
        let summary = comparison.summary();
        comma_line(out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(out, "prompt_text", &result.prompt_text, 6, true);
        push_json_field_string(out, "target_text", &result.target_text, 6, true);
        push_eval_forced_target_comparison_fields(out, comparison, &summary, tokenizer, 6)?;
        out.push_str("\n    }");
    }
    out.push_str("\n  ]}");

    Ok(())
}

fn push_eval_chat_forced_target_section(
    out: &mut String,
    suite: &inference::ChatBackendForcedTargetComparisonSuiteResult,
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    push_indent(out, 2);
    out.push_str("\"chat_forced_target\": {\"summary\": ");
    push_generation_forced_compare_suite_summary(out, &suite.summary);
    out.push_str(", \"memory\": ");
    push_memory_stats_summary(
        out,
        &suite.reference_memory_stats,
        &suite.candidate_memory_stats,
    );
    out.push_str(", \"prompts\": [\n");
    for (i, prompt_result) in suite.prompts.iter().enumerate() {
        let result = &prompt_result.result;
        let comparison = &result.comparison;
        let summary = comparison.summary();
        comma_line(out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(out, "user_prompt", &prompt_result.user_prompt, 6, true);
        push_json_field_string(out, "formatted_prompt", &result.formatted_prompt, 6, true);
        push_json_field_string(out, "target_text", &result.target_text, 6, true);
        push_eval_forced_target_comparison_fields(out, comparison, &summary, tokenizer, 6)?;
        out.push_str("\n    }");
    }
    out.push_str("\n  ]}");

    Ok(())
}

fn push_eval_forced_target_comparison_fields(
    out: &mut String,
    comparison: &inference::GenerationBackendForcedComparisonResult,
    summary: &inference::GenerationBackendForcedComparisonSummary,
    tokenizer: &TekkenTokenizer,
    indent: usize,
) -> AppResult<()> {
    push_json_field_u32_array(
        out,
        "sequence_tokens",
        &comparison.sequence_tokens,
        indent,
        true,
    );
    push_json_field_string(
        out,
        "sequence_text_skip_special",
        &tokenizer.decode_lossy(&comparison.sequence_tokens)?,
        indent,
        true,
    );
    push_json_field_u32_array(
        out,
        "prompt_tokens",
        &comparison.prompt_tokens,
        indent,
        true,
    );
    push_json_field_string(
        out,
        "prompt_text_skip_special",
        &tokenizer.decode_lossy(&comparison.prompt_tokens)?,
        indent,
        true,
    );
    push_json_field_u32_array(
        out,
        "forced_tokens",
        &comparison.forced_tokens,
        indent,
        true,
    );
    push_json_field_string(
        out,
        "forced_text_skip_special",
        &tokenizer.decode_lossy(&comparison.forced_tokens)?,
        indent,
        true,
    );
    push_json_field_string(
        out,
        "reference_backend",
        backend_label(comparison.reference_backend),
        indent,
        true,
    );
    push_json_field_string(
        out,
        "candidate_backend",
        backend_label(comparison.candidate_backend),
        indent,
        true,
    );
    push_json_field_forced_compare_summary(out, "summary", summary, indent, true);
    push_json_field_forced_compare_steps(out, "steps", &comparison.steps, indent, false);
    Ok(())
}

fn write_quantization_manifest_report(
    path: &Path,
    manifest: &QuantizationManifest,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_string(
        &mut out,
        "model_dir",
        &manifest.model_dir.display().to_string(),
        2,
        true,
    );
    push_json_field_string(
        &mut out,
        "weights_path",
        &manifest.weights_path.display().to_string(),
        2,
        true,
    );
    push_json_field_string(
        &mut out,
        "quantization",
        "row-symmetric-int8-with-f32-row-scales",
        2,
        true,
    );
    push_indent(&mut out, 2);
    out.push_str(&format!("\"n_layers\": {},\n", manifest.n_layers));
    push_indent(&mut out, 2);
    out.push_str("\"summary\": ");
    push_quantization_manifest_summary(&mut out, &manifest.summary);
    out.push_str(",\n");
    push_indent(&mut out, 2);
    out.push_str("\"quantized_tensors\": [\n");
    for (index, tensor) in manifest.quantized_tensors.iter().enumerate() {
        comma_line(&mut out, index, "    ");
        push_quantized_tensor_plan_json(&mut out, tensor, 4);
    }
    out.push_str("\n  ],\n");
    push_indent(&mut out, 2);
    out.push_str("\"preserved_tensors\": [\n");
    for (index, tensor) in manifest.preserved_tensors.iter().enumerate() {
        comma_line(&mut out, index, "    ");
        push_preserved_tensor_plan(&mut out, tensor, 4);
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_quantization_export_manifest(
    path: &Path,
    export: &QuantizationExportManifest,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_string(
        &mut out,
        "model_dir",
        &export.model_dir.display().to_string(),
        2,
        true,
    );
    push_json_field_string(
        &mut out,
        "weights_path",
        &export.weights_path.display().to_string(),
        2,
        true,
    );
    push_json_field_string(
        &mut out,
        "output_dir",
        &export.output_dir.display().to_string(),
        2,
        true,
    );
    push_json_field_string(
        &mut out,
        "quantization",
        "row-symmetric-int8-with-f32-row-scales",
        2,
        true,
    );
    push_indent(&mut out, 2);
    out.push_str(&format!(
        "\"planned_tensor_count\": {},\n",
        export.planned_tensor_count
    ));
    push_indent(&mut out, 2);
    out.push_str(&format!(
        "\"exported_tensor_count\": {},\n",
        export.exported_tensors.len()
    ));
    push_json_field_bool(&mut out, "complete_export", export.complete_export, 2, true);
    push_indent(&mut out, 2);
    out.push_str("\"summary\": ");
    push_quantization_manifest_summary(&mut out, &export.summary);
    out.push_str(",\n");
    push_indent(&mut out, 2);
    out.push_str("\"tensors\": [\n");
    for (index, tensor) in export.exported_tensors.iter().enumerate() {
        if index > 0 {
            out.push_str(",\n");
        }
        push_quantized_tensor_export_json(&mut out, tensor, 4);
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_quantization_validation_report(
    path: &Path,
    model_dir: &Path,
    export_dir: &Path,
    manifest: &QuantizationManifest,
    valid: bool,
    error_message: Option<&str>,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_string(
        &mut out,
        "model_dir",
        &model_dir.display().to_string(),
        2,
        true,
    );
    push_json_field_string(
        &mut out,
        "export_dir",
        &export_dir.display().to_string(),
        2,
        true,
    );
    push_json_field_string(
        &mut out,
        "quantization",
        "row-symmetric-int8-with-f32-row-scales",
        2,
        true,
    );
    push_json_field_usize(
        &mut out,
        "expected_quantized_tensor_count",
        manifest.summary.quantized_tensor_count,
        2,
        true,
    );
    push_json_field_bool(&mut out, "valid", valid, 2, error_message.is_some());
    if let Some(error_message) = error_message {
        push_json_field_string(&mut out, "error", error_message, 2, false);
    }
    out.push_str("}\n");
    write_report(path, out)
}

fn write_ministral_token_eval_report(
    path: &Path,
    model_dir: &Path,
    candidate_backend: InferenceBackend,
    candidate_export_dir: Option<&Path>,
    driver: GenerationComparisonDriver,
    max_new_tokens: usize,
    top_k: usize,
    logits_top_k: usize,
    prompt_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
    generation_summary: &ChatCompareSuiteSummary,
    eval_suite: &inference::GenerationBackendEvalSuiteResult,
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_string(
        &mut out,
        "model_dir",
        &model_dir.display().to_string(),
        2,
        true,
    );
    push_json_field_string(
        &mut out,
        "candidate_backend",
        backend_label(candidate_backend),
        2,
        true,
    );
    if let Some(candidate_export_dir) = candidate_export_dir {
        push_json_field_string(
            &mut out,
            "candidate_export_dir",
            &candidate_export_dir.display().to_string(),
            2,
            true,
        );
    }
    push_json_field_string(&mut out, "driver", driver_label(driver), 2, true);
    push_json_field_usize(&mut out, "max_new_tokens", max_new_tokens, 2, true);
    push_json_field_usize(&mut out, "top_k", top_k, 2, true);
    push_json_field_usize(&mut out, "logits_top_k", logits_top_k, 2, true);
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, true);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    push_json_field_usize(
        &mut out,
        "prompt_count",
        eval_suite.logits.prompts.len(),
        2,
        true,
    );
    push_indent(&mut out, 2);
    out.push_str("\"token_logits\": {\"summary\": ");
    push_generation_logits_compare_suite_summary(&mut out, &eval_suite.logits.summary);
    out.push_str(", \"memory\": ");
    push_memory_stats_summary(
        &mut out,
        &eval_suite.logits.reference_memory_stats,
        &eval_suite.logits.candidate_memory_stats,
    );
    out.push_str(", \"prompts\": [\n");
    for (i, result) in eval_suite.logits.prompts.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        push_token_logits_comparison_prompt_json(&mut out, result, tokenizer, 4)?;
    }
    out.push_str("\n  ]},\n");
    push_indent(&mut out, 2);
    out.push_str("\"token_generation\": {\"summary\": ");
    push_chat_compare_suite_summary(&mut out, generation_summary);
    out.push_str(", \"memory\": ");
    push_memory_stats_summary(
        &mut out,
        &eval_suite.generation.reference_memory_stats,
        &eval_suite.generation.candidate_memory_stats,
    );
    out.push_str(", \"prompts\": [\n");
    for (i, result) in eval_suite.generation.prompts.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        push_token_generation_comparison_prompt_json(&mut out, result, tokenizer, 4)?;
    }
    out.push_str("\n  ]}\n}\n");
    write_report(path, out)
}

fn push_token_logits_comparison_prompt_json(
    out: &mut String,
    result: &inference::GenerationBackendLogitsComparisonResult,
    tokenizer: &TekkenTokenizer,
    indent: usize,
) -> AppResult<()> {
    out.push_str("{\n");
    push_json_field_u32_array(
        out,
        "prompt_tokens",
        &result.prompt_tokens,
        indent + 2,
        true,
    );
    push_json_field_string(
        out,
        "prompt_text_skip_special",
        &tokenizer.decode_lossy(&result.prompt_tokens)?,
        indent + 2,
        true,
    );
    push_json_field_string(
        out,
        "reference_backend",
        backend_label(result.reference.backend),
        indent + 2,
        true,
    );
    push_json_field_string(
        out,
        "candidate_backend",
        backend_label(result.candidate.backend),
        indent + 2,
        true,
    );
    push_json_field_u32(
        out,
        "reference_token_id",
        result.reference_token_id,
        indent + 2,
        true,
    );
    push_json_field_u32(
        out,
        "candidate_token_id",
        result.candidate_token_id,
        indent + 2,
        true,
    );
    push_json_field_bool(out, "token_matches", result.token_matches, indent + 2, true);
    push_json_field_bool(
        out,
        "top_tokens_match",
        result.top_tokens_match(),
        indent + 2,
        true,
    );
    push_json_field_f64(out, "kl_divergence", result.kl_divergence, indent + 2, true);
    push_json_field_f32(out, "max_abs_diff", result.max_abs_diff, indent + 2, true);
    push_json_field_f32(out, "mean_abs_diff", result.mean_abs_diff, indent + 2, true);
    push_json_field_token_logits(
        out,
        "reference_top_logits",
        &result.reference.top_logits,
        indent + 2,
        true,
    );
    push_json_field_token_logits(
        out,
        "candidate_top_logits",
        &result.candidate.top_logits,
        indent + 2,
        false,
    );
    out.push('\n');
    push_indent(out, indent);
    out.push('}');
    Ok(())
}

fn push_token_generation_comparison_prompt_json(
    out: &mut String,
    result: &inference::GenerationBackendComparisonResult,
    tokenizer: &TekkenTokenizer,
    indent: usize,
) -> AppResult<()> {
    let summary = result.summary();
    out.push_str("{\n");
    push_json_field_u32_array(
        out,
        "prompt_tokens",
        &result.prompt_tokens,
        indent + 2,
        true,
    );
    push_json_field_string(
        out,
        "prompt_text_skip_special",
        &tokenizer.decode_lossy(&result.prompt_tokens)?,
        indent + 2,
        true,
    );
    push_json_field_u32_array(
        out,
        "generated_tokens",
        &result.generated_tokens,
        indent + 2,
        true,
    );
    push_json_field_u32_array(out, "all_tokens", &result.all_tokens, indent + 2, true);
    push_json_field_string(
        out,
        "generated_text",
        &tokenizer.decode_lossy(&result.generated_tokens)?,
        indent + 2,
        true,
    );
    push_json_field_string(
        out,
        "all_text_skip_special",
        &tokenizer.decode_lossy(&result.all_tokens)?,
        indent + 2,
        true,
    );
    push_json_field_string(out, "driver", driver_label(result.driver), indent + 2, true);
    push_json_field_string(
        out,
        "reference_backend",
        backend_label(result.reference_backend),
        indent + 2,
        true,
    );
    push_json_field_string(
        out,
        "candidate_backend",
        backend_label(result.candidate_backend),
        indent + 2,
        true,
    );
    push_json_field_chat_compare_summary(out, "summary", &summary, indent + 2, true);
    push_json_field_chat_compare_steps(out, "steps", &result.steps, indent + 2, false);
    out.push('\n');
    push_indent(out, indent);
    out.push('}');
    Ok(())
}

fn write_text_generation_report(
    path: &Path,
    results: &[inference::TextGenerationResult],
    metadata: &InferenceReportMetadata<'_>,
    prompt_file_paths: &[PathBuf],
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_inference_report_metadata(&mut out, metadata);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "backend", backend_label(result.backend), 6, true);
        push_json_field_string(&mut out, "prompt_text", &result.prompt_text, 6, true);
        push_json_field_u32_array(&mut out, "prompt_tokens", &result.prompt_tokens, 6, true);
        push_json_field_u32_array(
            &mut out,
            "generated_tokens",
            &result.generated_tokens,
            6,
            true,
        );
        push_json_field_u32_array(&mut out, "all_tokens", &result.all_tokens, 6, true);
        push_json_field_runtime_memory_stats(&mut out, "memory", &result.memory_stats, 6, true);
        push_json_field_generation_timings(
            &mut out,
            "timings",
            result.timings,
            result.generated_tokens.len(),
            6,
            true,
        );
        push_json_field_string(&mut out, "generated_text", &result.generated_text, 6, true);
        push_json_field_string(
            &mut out,
            "all_text_skip_special",
            &result.all_text_skip_special,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "finish_reason",
            &finish_reason_label(result.finish_reason),
            6,
            true,
        );
        push_json_field_generation_steps(&mut out, "steps", &result.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_text_logits_report(
    path: &Path,
    results: &[inference::TextGenerationLogitsResult],
    metadata: &InferenceReportMetadata<'_>,
    prompt_file_paths: &[PathBuf],
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_inference_report_metadata(&mut out, metadata);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "prompt_text", &result.prompt_text, 6, true);
        push_json_field_string(
            &mut out,
            "backend",
            backend_label(result.result.backend),
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "prompt_tokens",
            &result.result.prompt_tokens,
            6,
            true,
        );
        push_json_field_f32(&mut out, "logsumexp", result.result.logsumexp, 6, true);
        push_json_field_token_logits(&mut out, "top_logits", &result.result.top_logits, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_text_logits_trace_report(
    path: &Path,
    results: &[inference::TextGenerationBackendLogitsTraceResult],
    metadata: &InferenceReportMetadata<'_>,
    prompt_file_paths: &[PathBuf],
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_inference_report_metadata(&mut out, metadata);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "prompt_text", &result.prompt_text, 6, true);
        push_json_field_string(
            &mut out,
            "backend",
            backend_label(result.trace.backend),
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "prompt_tokens",
            &result.trace.prompt_tokens,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "generated_tokens",
            &result.trace.generated_tokens,
            6,
            true,
        );
        push_json_field_u32_array(&mut out, "all_tokens", &result.trace.all_tokens, 6, true);
        push_json_field_string(&mut out, "generated_text", &result.generated_text, 6, true);
        push_json_field_string(
            &mut out,
            "all_text_skip_special",
            &result.all_text_skip_special,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "finish_reason",
            &finish_reason_label(result.trace.finish_reason),
            6,
            true,
        );
        push_json_field_chat_trace_steps(&mut out, "steps", &result.trace.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_text_logits_compare_report(
    path: &Path,
    results: &[inference::TextGenerationBackendLogitsComparisonResult],
    summary: &GenerationBackendLogitsComparisonSuiteSummary,
    prompt_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        let comparison = &result.comparison;
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "prompt_text", &result.prompt_text, 6, true);
        push_json_field_u32_array(
            &mut out,
            "prompt_tokens",
            &comparison.prompt_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "reference_backend",
            backend_label(comparison.reference.backend),
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "candidate_backend",
            backend_label(comparison.candidate.backend),
            6,
            true,
        );
        push_json_field_u32(
            &mut out,
            "reference_token_id",
            comparison.reference_token_id,
            6,
            true,
        );
        push_json_field_u32(
            &mut out,
            "candidate_token_id",
            comparison.candidate_token_id,
            6,
            true,
        );
        push_json_field_bool(&mut out, "token_matches", comparison.token_matches, 6, true);
        push_json_field_bool(
            &mut out,
            "top_tokens_match",
            comparison.top_tokens_match(),
            6,
            true,
        );
        push_json_field_f64(&mut out, "kl_divergence", comparison.kl_divergence, 6, true);
        push_json_field_f32(&mut out, "max_abs_diff", comparison.max_abs_diff, 6, true);
        push_json_field_f32(&mut out, "mean_abs_diff", comparison.mean_abs_diff, 6, true);
        push_json_field_token_logits(
            &mut out,
            "reference_top_logits",
            &comparison.reference.top_logits,
            6,
            true,
        );
        push_json_field_token_logits(
            &mut out,
            "candidate_top_logits",
            &comparison.candidate.top_logits,
            6,
            false,
        );
        out.push_str("\n    }");
    }
    out.push_str("\n  ],\n  \"summary\": ");
    push_generation_logits_compare_suite_summary(&mut out, summary);
    out.push_str(",\n");
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, false);
    out.push_str("}\n");
    write_report(path, out)
}

fn write_text_compare_report(
    path: &Path,
    results: &[inference::TextGenerationBackendComparisonResult],
    suite_summary: &ChatCompareSuiteSummary,
    prompt_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        let comparison = &result.comparison;
        let summary = comparison.summary();
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "prompt_text", &result.prompt_text, 6, true);
        push_json_field_u32_array(
            &mut out,
            "prompt_tokens",
            &comparison.prompt_tokens,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "generated_tokens",
            &comparison.generated_tokens,
            6,
            true,
        );
        push_json_field_u32_array(&mut out, "all_tokens", &comparison.all_tokens, 6, true);
        push_json_field_string(&mut out, "generated_text", &result.generated_text, 6, true);
        push_json_field_string(
            &mut out,
            "all_text_skip_special",
            &result.all_text_skip_special,
            6,
            true,
        );
        push_json_field_string(&mut out, "driver", driver_label(comparison.driver), 6, true);
        push_json_field_string(
            &mut out,
            "reference_backend",
            backend_label(comparison.reference_backend),
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "candidate_backend",
            backend_label(comparison.candidate_backend),
            6,
            true,
        );
        push_json_field_chat_compare_summary(&mut out, "summary", &summary, 6, true);
        push_json_field_chat_compare_steps(&mut out, "steps", &comparison.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ],\n  \"summary\": ");
    push_chat_compare_suite_summary(&mut out, suite_summary);
    out.push_str(",\n");
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, false);
    out.push_str("}\n");
    write_report(path, out)
}

fn write_text_forced_compare_report(
    path: &Path,
    results: &[inference::TextGenerationBackendForcedComparisonResult],
    suite_summary: &inference::GenerationBackendForcedComparisonSuiteSummary,
    prompt_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        let comparison = &result.comparison;
        let summary = comparison.summary();
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "sequence_text", &result.sequence_text, 6, true);
        push_json_field_u32_array(
            &mut out,
            "sequence_tokens",
            &comparison.sequence_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "sequence_text_skip_special",
            &tokenizer.decode_lossy(&comparison.sequence_tokens)?,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "prompt_tokens",
            &comparison.prompt_tokens,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "forced_tokens",
            &comparison.forced_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "forced_text_skip_special",
            &tokenizer.decode_lossy(&comparison.forced_tokens)?,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "reference_backend",
            backend_label(comparison.reference_backend),
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "candidate_backend",
            backend_label(comparison.candidate_backend),
            6,
            true,
        );
        push_json_field_forced_compare_summary(&mut out, "summary", &summary, 6, true);
        push_json_field_forced_compare_steps(&mut out, "steps", &comparison.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ],\n  \"summary\": ");
    push_generation_forced_compare_suite_summary(&mut out, suite_summary);
    out.push_str(",\n");
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, false);
    out.push_str("}\n");
    write_report(path, out)
}

fn write_text_forced_target_compare_report(
    path: &Path,
    results: &[inference::TextGenerationBackendForcedTargetComparisonResult],
    suite_summary: &inference::GenerationBackendForcedComparisonSuiteSummary,
    pair_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_path_array(
        &mut out,
        "forced_target_pair_files",
        pair_file_paths,
        2,
        true,
    );
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        let comparison = &result.comparison;
        let summary = comparison.summary();
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "prompt_text", &result.prompt_text, 6, true);
        push_json_field_string(&mut out, "target_text", &result.target_text, 6, true);
        push_json_field_u32_array(
            &mut out,
            "sequence_tokens",
            &comparison.sequence_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "sequence_text_skip_special",
            &tokenizer.decode_lossy(&comparison.sequence_tokens)?,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "prompt_tokens",
            &comparison.prompt_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "prompt_text_skip_special",
            &tokenizer.decode_lossy(&comparison.prompt_tokens)?,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "forced_tokens",
            &comparison.forced_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "forced_text_skip_special",
            &tokenizer.decode_lossy(&comparison.forced_tokens)?,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "reference_backend",
            backend_label(comparison.reference_backend),
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "candidate_backend",
            backend_label(comparison.candidate_backend),
            6,
            true,
        );
        push_json_field_forced_compare_summary(&mut out, "summary", &summary, 6, true);
        push_json_field_forced_compare_steps(&mut out, "steps", &comparison.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ],\n  \"summary\": ");
    push_generation_forced_compare_suite_summary(&mut out, suite_summary);
    out.push_str(",\n");
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, false);
    out.push_str("}\n");
    write_report(path, out)
}

fn write_chat_forced_target_compare_report(
    path: &Path,
    results: &[inference::ChatBackendForcedTargetComparisonPromptResult],
    suite_summary: &inference::GenerationBackendForcedComparisonSuiteSummary,
    pair_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_path_array(
        &mut out,
        "forced_target_pair_files",
        pair_file_paths,
        2,
        true,
    );
    out.push_str("  \"prompts\": [\n");
    for (i, prompt_result) in results.iter().enumerate() {
        let result = &prompt_result.result;
        let comparison = &result.comparison;
        let summary = comparison.summary();
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "user_prompt", &prompt_result.user_prompt, 6, true);
        push_json_field_string(
            &mut out,
            "formatted_prompt",
            &result.formatted_prompt,
            6,
            true,
        );
        push_json_field_string(&mut out, "target_text", &result.target_text, 6, true);
        push_json_field_u32_array(
            &mut out,
            "sequence_tokens",
            &comparison.sequence_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "sequence_text_skip_special",
            &tokenizer.decode_lossy(&comparison.sequence_tokens)?,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "prompt_tokens",
            &comparison.prompt_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "prompt_text_skip_special",
            &tokenizer.decode_lossy(&comparison.prompt_tokens)?,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "forced_tokens",
            &comparison.forced_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "forced_text_skip_special",
            &tokenizer.decode_lossy(&comparison.forced_tokens)?,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "reference_backend",
            backend_label(comparison.reference_backend),
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "candidate_backend",
            backend_label(comparison.candidate_backend),
            6,
            true,
        );
        push_json_field_forced_compare_summary(&mut out, "summary", &summary, 6, true);
        push_json_field_forced_compare_steps(&mut out, "steps", &comparison.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ],\n  \"summary\": ");
    push_generation_forced_compare_suite_summary(&mut out, suite_summary);
    out.push_str(",\n");
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, false);
    out.push_str("}\n");
    write_report(path, out)
}

fn write_token_generation_report(
    path: &Path,
    results: &[inference::GenerationResult],
    metadata: &InferenceReportMetadata<'_>,
    prompt_file_paths: &[PathBuf],
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_inference_report_metadata(&mut out, metadata);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "backend", backend_label(result.backend), 6, true);
        push_json_field_u32_array(&mut out, "prompt_tokens", &result.prompt_tokens, 6, true);
        push_json_field_u32_array(
            &mut out,
            "generated_tokens",
            &result.generated_tokens,
            6,
            true,
        );
        push_json_field_u32_array(&mut out, "all_tokens", &result.all_tokens, 6, true);
        push_json_field_generation_timings(
            &mut out,
            "timings",
            result.timings,
            result.generated_tokens.len(),
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "generated_text",
            &tokenizer.decode_lossy(&result.generated_tokens)?,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "all_text_skip_special",
            &tokenizer.decode_lossy(&result.all_tokens)?,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "finish_reason",
            &finish_reason_label(result.finish_reason),
            6,
            true,
        );
        push_json_field_generation_steps(&mut out, "steps", &result.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_token_logits_report(
    path: &Path,
    results: &[inference::GenerationLogitsResult],
    metadata: &InferenceReportMetadata<'_>,
    prompt_file_paths: &[PathBuf],
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_inference_report_metadata(&mut out, metadata);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "backend", backend_label(result.backend), 6, true);
        push_json_field_u32_array(&mut out, "prompt_tokens", &result.prompt_tokens, 6, true);
        push_json_field_string(
            &mut out,
            "prompt_text_skip_special",
            &tokenizer.decode_lossy(&result.prompt_tokens)?,
            6,
            true,
        );
        push_json_field_f32(&mut out, "logsumexp", result.logsumexp, 6, true);
        push_json_field_token_logits(&mut out, "top_logits", &result.top_logits, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_token_logits_trace_report(
    path: &Path,
    results: &[inference::GenerationLogitsTraceResult],
    metadata: &InferenceReportMetadata<'_>,
    prompt_file_paths: &[PathBuf],
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_inference_report_metadata(&mut out, metadata);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "backend", backend_label(result.backend), 6, true);
        push_json_field_u32_array(&mut out, "prompt_tokens", &result.prompt_tokens, 6, true);
        push_json_field_u32_array(
            &mut out,
            "generated_tokens",
            &result.generated_tokens,
            6,
            true,
        );
        push_json_field_u32_array(&mut out, "all_tokens", &result.all_tokens, 6, true);
        push_json_field_string(
            &mut out,
            "generated_text",
            &tokenizer.decode_lossy(&result.generated_tokens)?,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "all_text_skip_special",
            &tokenizer.decode_lossy(&result.all_tokens)?,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "finish_reason",
            &finish_reason_label(result.finish_reason),
            6,
            true,
        );
        push_json_field_chat_trace_steps(&mut out, "steps", &result.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_token_logits_compare_report(
    path: &Path,
    results: &[inference::GenerationBackendLogitsComparisonResult],
    summary: &GenerationBackendLogitsComparisonSuiteSummary,
    prompt_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_u32_array(&mut out, "prompt_tokens", &result.prompt_tokens, 6, true);
        push_json_field_string(
            &mut out,
            "prompt_text_skip_special",
            &tokenizer.decode_lossy(&result.prompt_tokens)?,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "reference_backend",
            backend_label(result.reference.backend),
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "candidate_backend",
            backend_label(result.candidate.backend),
            6,
            true,
        );
        push_json_field_u32(
            &mut out,
            "reference_token_id",
            result.reference_token_id,
            6,
            true,
        );
        push_json_field_u32(
            &mut out,
            "candidate_token_id",
            result.candidate_token_id,
            6,
            true,
        );
        push_json_field_bool(&mut out, "token_matches", result.token_matches, 6, true);
        push_json_field_bool(
            &mut out,
            "top_tokens_match",
            result.top_tokens_match(),
            6,
            true,
        );
        push_json_field_f64(&mut out, "kl_divergence", result.kl_divergence, 6, true);
        push_json_field_f32(&mut out, "max_abs_diff", result.max_abs_diff, 6, true);
        push_json_field_f32(&mut out, "mean_abs_diff", result.mean_abs_diff, 6, true);
        push_json_field_token_logits(
            &mut out,
            "reference_top_logits",
            &result.reference.top_logits,
            6,
            true,
        );
        push_json_field_token_logits(
            &mut out,
            "candidate_top_logits",
            &result.candidate.top_logits,
            6,
            false,
        );
        out.push_str("\n    }");
    }
    out.push_str("\n  ],\n  \"summary\": ");
    push_generation_logits_compare_suite_summary(&mut out, summary);
    out.push_str(",\n");
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, false);
    out.push_str("}\n");
    write_report(path, out)
}

fn write_token_compare_report(
    path: &Path,
    results: &[inference::GenerationBackendComparisonResult],
    suite_summary: &ChatCompareSuiteSummary,
    prompt_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        let summary = result.summary();
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_u32_array(&mut out, "prompt_tokens", &result.prompt_tokens, 6, true);
        push_json_field_string(
            &mut out,
            "prompt_text_skip_special",
            &tokenizer.decode_lossy(&result.prompt_tokens)?,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "generated_tokens",
            &result.generated_tokens,
            6,
            true,
        );
        push_json_field_u32_array(&mut out, "all_tokens", &result.all_tokens, 6, true);
        push_json_field_string(
            &mut out,
            "generated_text",
            &tokenizer.decode_lossy(&result.generated_tokens)?,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "all_text_skip_special",
            &tokenizer.decode_lossy(&result.all_tokens)?,
            6,
            true,
        );
        push_json_field_string(&mut out, "driver", driver_label(result.driver), 6, true);
        push_json_field_string(
            &mut out,
            "reference_backend",
            backend_label(result.reference_backend),
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "candidate_backend",
            backend_label(result.candidate_backend),
            6,
            true,
        );
        push_json_field_chat_compare_summary(&mut out, "summary", &summary, 6, true);
        push_json_field_chat_compare_steps(&mut out, "steps", &result.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ],\n  \"summary\": ");
    push_chat_compare_suite_summary(&mut out, suite_summary);
    out.push_str(",\n");
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, false);
    out.push_str("}\n");
    write_report(path, out)
}

fn write_token_forced_compare_report(
    path: &Path,
    results: &[inference::GenerationBackendForcedComparisonResult],
    suite_summary: &inference::GenerationBackendForcedComparisonSuiteSummary,
    prompt_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
    tokenizer: &TekkenTokenizer,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, result) in results.iter().enumerate() {
        let summary = result.summary();
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_u32_array(
            &mut out,
            "sequence_tokens",
            &result.sequence_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "sequence_text_skip_special",
            &tokenizer.decode_lossy(&result.sequence_tokens)?,
            6,
            true,
        );
        push_json_field_u32_array(&mut out, "prompt_tokens", &result.prompt_tokens, 6, true);
        push_json_field_u32_array(&mut out, "forced_tokens", &result.forced_tokens, 6, true);
        push_json_field_string(
            &mut out,
            "forced_text_skip_special",
            &tokenizer.decode_lossy(&result.forced_tokens)?,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "reference_backend",
            backend_label(result.reference_backend),
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "candidate_backend",
            backend_label(result.candidate_backend),
            6,
            true,
        );
        push_json_field_forced_compare_summary(&mut out, "summary", &summary, 6, true);
        push_json_field_forced_compare_steps(&mut out, "steps", &result.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ],\n  \"summary\": ");
    push_generation_forced_compare_suite_summary(&mut out, suite_summary);
    out.push_str(",\n");
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, false);
    out.push_str("}\n");
    write_report(path, out)
}

fn write_chat_logits_report(
    path: &Path,
    results: &[(String, ChatLogitsResult)],
    metadata: &InferenceReportMetadata<'_>,
    prompt_file_paths: &[PathBuf],
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_inference_report_metadata(&mut out, metadata);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, (prompt, result)) in results.iter().enumerate() {
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "user_prompt", prompt, 6, true);
        push_json_field_string(
            &mut out,
            "formatted_prompt",
            &result.formatted_prompt,
            6,
            true,
        );
        push_json_field_u32_array(&mut out, "prompt_tokens", &result.prompt_tokens, 6, true);
        push_json_field_f32(&mut out, "logsumexp", result.logsumexp, 6, true);
        push_json_field_token_logits(&mut out, "top_logits", &result.top_logits, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_chat_logits_compare_report(
    path: &Path,
    results: &[inference::ChatBackendLogitsComparisonPromptResult],
    summary: &GenerationBackendLogitsComparisonSuiteSummary,
    prompt_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, prompt_result) in results.iter().enumerate() {
        let result = &prompt_result.result;
        let comparison = &result.comparison;
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "user_prompt", &prompt_result.user_prompt, 6, true);
        push_json_field_string(
            &mut out,
            "formatted_prompt",
            &result.formatted_prompt,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "prompt_tokens",
            &comparison.prompt_tokens,
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "reference_backend",
            backend_label(comparison.reference.backend),
            6,
            true,
        );
        push_json_field_string(
            &mut out,
            "candidate_backend",
            backend_label(comparison.candidate.backend),
            6,
            true,
        );
        push_json_field_u32(
            &mut out,
            "reference_token_id",
            comparison.reference_token_id,
            6,
            true,
        );
        push_json_field_u32(
            &mut out,
            "candidate_token_id",
            comparison.candidate_token_id,
            6,
            true,
        );
        push_json_field_bool(&mut out, "token_matches", comparison.token_matches, 6, true);
        push_json_field_bool(
            &mut out,
            "top_tokens_match",
            comparison.top_tokens_match(),
            6,
            true,
        );
        push_json_field_f64(&mut out, "kl_divergence", comparison.kl_divergence, 6, true);
        push_json_field_f32(&mut out, "max_abs_diff", comparison.max_abs_diff, 6, true);
        push_json_field_f32(&mut out, "mean_abs_diff", comparison.mean_abs_diff, 6, true);
        push_json_field_token_logits(
            &mut out,
            "reference_top_logits",
            &comparison.reference.top_logits,
            6,
            true,
        );
        push_json_field_token_logits(
            &mut out,
            "candidate_top_logits",
            &comparison.candidate.top_logits,
            6,
            false,
        );
        out.push_str("\n    }");
    }
    out.push_str("\n  ],\n  \"summary\": ");
    push_generation_logits_compare_suite_summary(&mut out, summary);
    out.push_str(",\n");
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, false);
    out.push_str("}\n");
    write_report(path, out)
}

fn write_chat_logits_trace_report(
    path: &Path,
    results: &[inference::ChatBackendLogitsTracePromptResult],
    metadata: &InferenceReportMetadata<'_>,
    prompt_file_paths: &[PathBuf],
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_inference_report_metadata(&mut out, metadata);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, prompt_result) in results.iter().enumerate() {
        let result = &prompt_result.result;
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "user_prompt", &prompt_result.user_prompt, 6, true);
        push_json_field_string(
            &mut out,
            "formatted_prompt",
            &result.formatted_prompt,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "prompt_tokens",
            &result.trace.prompt_tokens,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "generated_tokens",
            &result.trace.generated_tokens,
            6,
            true,
        );
        push_json_field_string(&mut out, "generated_text", &result.generated_text, 6, true);
        push_json_field_string(
            &mut out,
            "finish_reason",
            &finish_reason_label(result.trace.finish_reason),
            6,
            true,
        );
        push_json_field_chat_trace_steps(&mut out, "steps", &result.trace.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ]\n}\n");
    write_report(path, out)
}

fn write_forced_trace_report(
    path: &Path,
    prompt: &str,
    generated_trace: &ChatLogitsTraceResult,
    forced_trace: &ChatLogitsTraceResult,
    metadata: &InferenceReportMetadata<'_>,
    prompt_file_paths: &[PathBuf],
) -> AppResult<()> {
    let comparison = inference::compare_logits_traces(&generated_trace.steps, &forced_trace.steps)?;
    let mut out = String::new();
    out.push_str("{\n");
    push_inference_report_metadata(&mut out, metadata);
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n    {\n");
    push_json_field_string(&mut out, "user_prompt", prompt, 6, true);
    push_json_field_u32_array(
        &mut out,
        "prompt_tokens",
        &generated_trace.prompt_tokens,
        6,
        true,
    );
    push_json_field_u32_array(
        &mut out,
        "generated_tokens",
        &generated_trace.generated_tokens,
        6,
        true,
    );
    push_json_field_string(
        &mut out,
        "generated_text",
        &generated_trace.generated_text,
        6,
        true,
    );
    push_json_field_trace_comparison(&mut out, "summary", &comparison, 6, false);
    out.push_str("\n    }\n  ]\n}\n");
    write_report(path, out)
}

fn write_chat_compare_report(
    path: &Path,
    results: &[(String, ChatBackendComparisonResult)],
    suite_summary: &ChatCompareSuiteSummary,
    prompt_file_paths: &[PathBuf],
    thresholds: &ChatCompareThresholds,
) -> AppResult<()> {
    let mut out = String::new();
    out.push_str("{\n");
    push_json_field_path_array(&mut out, "prompt_files", prompt_file_paths, 2, true);
    out.push_str("  \"prompts\": [\n");
    for (i, (prompt, result)) in results.iter().enumerate() {
        let summary = result.comparison.summary();
        comma_line(&mut out, i, "    ");
        out.push_str("    {\n");
        push_json_field_string(&mut out, "user_prompt", prompt, 6, true);
        push_json_field_string(
            &mut out,
            "formatted_prompt",
            &result.formatted_prompt,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "prompt_tokens",
            &result.comparison.prompt_tokens,
            6,
            true,
        );
        push_json_field_u32_array(
            &mut out,
            "generated_tokens",
            &result.comparison.generated_tokens,
            6,
            true,
        );
        push_json_field_string(&mut out, "generated_text", &result.generated_text, 6, true);
        push_json_field_chat_compare_summary(&mut out, "summary", &summary, 6, true);
        push_json_field_chat_compare_steps(&mut out, "steps", &result.comparison.steps, 6, false);
        out.push_str("\n    }");
    }
    out.push_str("\n  ],\n  \"summary\": ");
    push_chat_compare_suite_summary(&mut out, suite_summary);
    out.push_str(",\n");
    push_json_field_thresholds(&mut out, "thresholds", thresholds, 2, false);
    out.push_str("}\n");
    write_report(path, out)
}

fn write_report(path: &Path, contents: String) -> AppResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, contents)?;
    Ok(())
}

fn push_inference_report_metadata(out: &mut String, metadata: &InferenceReportMetadata<'_>) {
    let model_dir = metadata.model_dir.display().to_string();
    push_json_field_string(out, "model_dir", &model_dir, 2, true);
    push_json_field_string(out, "backend", backend_label(metadata.backend), 2, true);
    if let Some(export_dir) = metadata.export_dir {
        let export_dir = export_dir.display().to_string();
        push_json_field_string(out, "export_dir", &export_dir, 2, true);
    }
    if let Some(decode_strategy) = metadata.decode_strategy {
        push_json_field_string(out, "decode_strategy", decode_strategy, 2, true);
    }
    if let Some(system_prompt) = metadata.system_prompt {
        push_json_field_string(
            out,
            "system_prompt_mode",
            system_prompt_mode_label(system_prompt),
            2,
            true,
        );
    }
    if let Some(max_new_tokens) = metadata.max_new_tokens {
        push_json_field_usize(out, "max_new_tokens", max_new_tokens, 2, true);
    }
    if let Some(top_k) = metadata.top_k {
        push_json_field_usize(out, "top_k", top_k, 2, true);
    }
    if let Some(logits_top_k) = metadata.logits_top_k {
        push_json_field_usize(out, "logits_top_k", logits_top_k, 2, true);
    }
}

fn push_json_field_string(out: &mut String, name: &str, value: &str, indent: usize, comma: bool) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": ");
    push_json_string(out, value);
    push_optional_comma(out, comma);
}

fn push_json_field_f32(out: &mut String, name: &str, value: f32, indent: usize, comma: bool) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(&format!(": {:.8}", value));
    push_optional_comma(out, comma);
}

fn push_json_field_f64(out: &mut String, name: &str, value: f64, indent: usize, comma: bool) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(&format!(": {:.12}", value));
    push_optional_comma(out, comma);
}

fn push_json_field_u32(out: &mut String, name: &str, value: u32, indent: usize, comma: bool) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(&format!(": {value}"));
    push_optional_comma(out, comma);
}

fn push_json_field_usize(out: &mut String, name: &str, value: usize, indent: usize, comma: bool) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(&format!(": {value}"));
    push_optional_comma(out, comma);
}

fn push_json_field_bool(out: &mut String, name: &str, value: bool, indent: usize, comma: bool) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(if value { ": true" } else { ": false" });
    push_optional_comma(out, comma);
}

fn push_json_field_generation_timings(
    out: &mut String,
    name: &str,
    timings: inference::GenerationTimings,
    generated_token_count: usize,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": {\n");
    push_json_field_f64(
        out,
        "prefill_seconds",
        timings.prefill_seconds,
        indent + 2,
        true,
    );
    push_json_field_f64(
        out,
        "decode_seconds",
        timings.decode_seconds,
        indent + 2,
        true,
    );
    push_json_field_f64(
        out,
        "total_seconds",
        timings.total_seconds(),
        indent + 2,
        true,
    );
    push_json_field_f64(
        out,
        "decode_tokens_per_second",
        timings
            .decode_tokens_per_second(generated_token_count)
            .unwrap_or(0.0),
        indent + 2,
        true,
    );
    push_json_field_f64(
        out,
        "total_tokens_per_second",
        timings
            .total_tokens_per_second(generated_token_count)
            .unwrap_or(0.0),
        indent + 2,
        false,
    );
    out.push('\n');
    push_indent(out, indent);
    out.push('}');
    push_optional_comma(out, comma);
}

fn push_json_field_u32_array(
    out: &mut String,
    name: &str,
    values: &[u32],
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": ");
    push_u32_array(out, values);
    push_optional_comma(out, comma);
}

fn push_json_field_path_array(
    out: &mut String,
    name: &str,
    values: &[PathBuf],
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": [");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        push_json_string(out, &value.display().to_string());
    }
    out.push(']');
    push_optional_comma(out, comma);
}

fn push_json_field_thresholds(
    out: &mut String,
    name: &str,
    thresholds: &ChatCompareThresholds,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": {");
    out.push_str(&format!(
        "\"require_token_match\": {}, \"require_top_tokens_match\": {}, ",
        thresholds.require_token_match, thresholds.require_top_tokens_match
    ));
    out.push_str("\"max_mean_kl\": ");
    push_optional_f64_value(out, thresholds.max_mean_kl);
    out.push_str(", \"max_max_kl\": ");
    push_optional_f64_value(out, thresholds.max_max_kl);
    out.push_str(", \"max_max_abs_diff\": ");
    push_optional_f32_value(out, thresholds.max_max_abs_diff);
    out.push('}');
    push_optional_comma(out, comma);
}

fn push_json_field_token_logits(
    out: &mut String,
    name: &str,
    values: &[TokenLogit],
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": [");
    push_token_logits_array_body(out, values);
    out.push(']');
    push_optional_comma(out, comma);
}

fn push_token_logits_array(out: &mut String, values: &[TokenLogit]) {
    out.push('[');
    push_token_logits_array_body(out, values);
    out.push(']');
}

fn push_token_logits_array_body(out: &mut String, values: &[TokenLogit]) {
    for (i, token) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "{{\"token_id\": {}, \"logit\": {:.8}, \"logprob\": {:.8}}}",
            token.token_id, token.logit, token.logprob
        ));
    }
}

fn push_json_field_trace_comparison(
    out: &mut String,
    name: &str,
    comparison: &inference::LogitsTraceComparison,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(&format!(
        ": {{\"step_count\": {}, \"token_ids_match\": {}, \"mean_kl_divergence\": {:.12}, \"max_kl_divergence\": {:.12}, \"max_abs_diff\": {:.8}}}",
        comparison.steps.len(),
        comparison.token_ids_match,
        comparison.mean_kl_divergence,
        comparison.max_kl_divergence,
        comparison.max_abs_diff
    ));
    push_optional_comma(out, comma);
}

fn push_json_field_chat_compare_summary(
    out: &mut String,
    name: &str,
    summary: &GenerationBackendComparisonSummary,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": ");
    push_chat_compare_summary(out, summary);
    push_optional_comma(out, comma);
}

fn push_json_field_forced_compare_summary(
    out: &mut String,
    name: &str,
    summary: &inference::GenerationBackendForcedComparisonSummary,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": ");
    push_forced_compare_summary(out, summary);
    push_optional_comma(out, comma);
}

fn push_json_field_chat_compare_steps(
    out: &mut String,
    name: &str,
    steps: &[inference::GenerationBackendComparisonStep],
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": [");
    for (i, step) in steps.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "{{\"step\": {}, \"prefix_len\": {}, \"reference_token_id\": {}, \"reference_logit\": {:.8}, \"candidate_token_id\": {}, \"candidate_logit\": {:.8}, \"token_matches\": {}, \"top_tokens_match\": {}, \"kl_divergence\": {:.12}, \"max_abs_diff\": {:.8}, \"mean_abs_diff\": {:.8}, \"reference_top_logits\": ",
            step.step,
            step.prefix_len,
            step.reference_token_id,
            step.reference_logit,
            step.candidate_token_id,
            step.candidate_logit,
            step.token_matches,
            step.top_tokens_match(),
            step.kl_divergence,
            step.max_abs_diff,
            step.mean_abs_diff
        ));
        push_token_logits_array(out, &step.reference_top_logits);
        out.push_str(", \"candidate_top_logits\": ");
        push_token_logits_array(out, &step.candidate_top_logits);
        out.push('}');
    }
    out.push(']');
    push_optional_comma(out, comma);
}

fn push_json_field_forced_compare_steps(
    out: &mut String,
    name: &str,
    steps: &[inference::GenerationBackendForcedComparisonStep],
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": [");
    for (i, step) in steps.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "{{\"step\": {}, \"prefix_len\": {}, \"forced_token_id\": {}, \"reference_token_id\": {}, \"candidate_token_id\": {}, \"token_matches\": {}, \"forced_token_matches_reference\": {}, \"forced_token_matches_candidate\": {}, \"top_tokens_match\": {}, \"kl_divergence\": {:.12}, \"max_abs_diff\": {:.8}, \"mean_abs_diff\": {:.8}, \"reference_logit\": {:.8}, \"candidate_logit\": {:.8}, \"reference_forced_logit\": {:.8}, \"candidate_forced_logit\": {:.8}, \"reference_forced_logprob\": {:.8}, \"candidate_forced_logprob\": {:.8}, \"forced_logprob_abs_diff\": {:.8}, \"reference_top_logits\": ",
            step.step,
            step.prefix_len,
            step.forced_token_id,
            step.reference_token_id,
            step.candidate_token_id,
            step.token_matches,
            step.forced_token_matches_reference,
            step.forced_token_matches_candidate,
            step.top_tokens_match(),
            step.kl_divergence,
            step.max_abs_diff,
            step.mean_abs_diff,
            step.reference_logit,
            step.candidate_logit,
            step.reference_forced_logit,
            step.candidate_forced_logit,
            step.reference_forced_logprob,
            step.candidate_forced_logprob,
            step.forced_logprob_abs_diff()
        ));
        push_token_logits_array(out, &step.reference_top_logits);
        out.push_str(", \"candidate_top_logits\": ");
        push_token_logits_array(out, &step.candidate_top_logits);
        out.push('}');
    }
    out.push(']');
    push_optional_comma(out, comma);
}

fn push_json_field_chat_trace_steps(
    out: &mut String,
    name: &str,
    steps: &[inference::ChatLogitsTraceStep],
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": [");
    for (i, step) in steps.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "{{\"step\": {}, \"token_id\": {}, \"logit\": {:.8}, \"logsumexp\": {:.8}, \"top_logits\": [",
            step.step, step.token_id, step.logit, step.logsumexp
        ));
        for (rank, token) in step.top_logits.iter().enumerate() {
            if rank > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!(
                "{{\"token_id\": {}, \"logit\": {:.8}, \"logprob\": {:.8}}}",
                token.token_id, token.logit, token.logprob
            ));
        }
        out.push_str("]}");
    }
    out.push(']');
    push_optional_comma(out, comma);
}

fn push_json_field_generation_steps(
    out: &mut String,
    name: &str,
    steps: &[nn_rust_inference::model::GreedyGenerationStep],
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": [");
    for (i, step) in steps.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "{{\"step\": {}, \"token_id\": {}, \"logit\": {:.8}, \"top_logits\": [",
            step.step, step.token_id, step.logit
        ));
        for (rank, (token_id, logit)) in step.top_logits.iter().enumerate() {
            if rank > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!(
                "{{\"token_id\": {}, \"logit\": {:.8}}}",
                token_id, logit
            ));
        }
        out.push_str("]}");
    }
    out.push(']');
    push_optional_comma(out, comma);
}

fn push_chat_compare_summary(out: &mut String, summary: &GenerationBackendComparisonSummary) {
    out.push_str(&format!(
        "{{\"step_count\": {}, \"token_match_count\": {}, \"top_tokens_match\": {}, \"mean_kl_divergence\": {:.12}, \"max_kl_divergence\": {:.12}, \"max_abs_diff\": {:.8}}}",
        summary.step_count,
        summary.token_match_count,
        summary.top_tokens_match,
        summary.mean_kl(),
        summary.max_kl,
        summary.max_abs_diff
    ));
}

fn push_forced_compare_summary(
    out: &mut String,
    summary: &inference::GenerationBackendForcedComparisonSummary,
) {
    out.push_str(&format!(
        "{{\"step_count\": {}, \"token_match_count\": {}, \"forced_reference_match_count\": {}, \"forced_candidate_match_count\": {}, \"top_tokens_match\": {}, \"mean_kl_divergence\": {:.12}, \"max_kl_divergence\": {:.12}, \"max_abs_diff\": {:.8}, \"mean_forced_logprob_abs_diff\": {:.12}, \"max_forced_logprob_abs_diff\": {:.8}}}",
        summary.step_count,
        summary.token_match_count,
        summary.forced_reference_match_count,
        summary.forced_candidate_match_count,
        summary.top_tokens_match,
        summary.mean_kl(),
        summary.max_kl,
        summary.max_abs_diff,
        summary.mean_forced_logprob_abs_diff(),
        summary.max_forced_logprob_abs_diff
    ));
}

fn push_generation_logits_compare_suite_summary(
    out: &mut String,
    summary: &GenerationBackendLogitsComparisonSuiteSummary,
) {
    out.push_str(&format!(
        "{{\"prompt_count\": {}, \"token_match_count\": {}, \"token_ids_match\": {}, \"top_tokens_match\": {}, \"mean_kl_divergence\": {:.12}, \"max_kl_divergence\": {:.12}, \"max_abs_diff\": {:.8}}}",
        summary.prompt_count,
        summary.token_match_count,
        summary.token_match_count == summary.prompt_count,
        summary.top_tokens_match,
        summary.mean_kl(),
        summary.max_kl,
        summary.max_abs_diff
    ));
}

fn push_generation_forced_compare_suite_summary(
    out: &mut String,
    summary: &inference::GenerationBackendForcedComparisonSuiteSummary,
) {
    out.push_str(&format!(
        "{{\"prompt_count\": {}, \"step_count\": {}, \"token_match_count\": {}, \"token_ids_match\": {}, \"forced_reference_match_count\": {}, \"forced_candidate_match_count\": {}, \"top_tokens_match\": {}, \"mean_kl_divergence\": {:.12}, \"max_kl_divergence\": {:.12}, \"max_abs_diff\": {:.8}, \"mean_forced_logprob_abs_diff\": {:.12}, \"max_forced_logprob_abs_diff\": {:.8}}}",
        summary.prompt_count,
        summary.summary.step_count,
        summary.summary.token_match_count,
        summary.summary.token_match_count == summary.summary.step_count,
        summary.summary.forced_reference_match_count,
        summary.summary.forced_candidate_match_count,
        summary.summary.top_tokens_match,
        summary.summary.mean_kl(),
        summary.summary.max_kl,
        summary.summary.max_abs_diff,
        summary.summary.mean_forced_logprob_abs_diff(),
        summary.summary.max_forced_logprob_abs_diff
    ));
}

fn push_memory_stats_summary(
    out: &mut String,
    reference: &RuntimeMemoryStats,
    candidate: &RuntimeMemoryStats,
) {
    let total_ratio = if reference.total_resident_bytes == 0 {
        0.0
    } else {
        candidate.total_resident_bytes as f64 / reference.total_resident_bytes as f64
    };
    let weight_ratio = if reference.weights_bytes == 0 {
        0.0
    } else {
        candidate.weights_bytes as f64 / reference.weights_bytes as f64
    };

    out.push_str(&format!(
        "{{\"reference_weights_bytes\": {}, \"candidate_weights_bytes\": {}, \"weights_ratio\": {:.6}, \"reference_kv_cache_bytes\": {}, \"candidate_kv_cache_bytes\": {}, \"reference_scratch_bytes\": {}, \"candidate_scratch_bytes\": {}, \"reference_total_resident_bytes\": {}, \"candidate_total_resident_bytes\": {}, \"total_resident_ratio\": {:.6}}}",
        reference.weights_bytes,
        candidate.weights_bytes,
        weight_ratio,
        reference.kv_cache_bytes,
        candidate.kv_cache_bytes,
        reference.scratch_bytes,
        candidate.scratch_bytes,
        reference.total_resident_bytes,
        candidate.total_resident_bytes,
        total_ratio
    ));
}

fn push_json_field_runtime_memory_stats(
    out: &mut String,
    name: &str,
    memory: &RuntimeMemoryStats,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": ");
    push_runtime_memory_stats(out, memory);
    push_optional_comma(out, comma);
}

fn push_runtime_memory_stats(out: &mut String, memory: &RuntimeMemoryStats) {
    out.push_str(&format!(
        "{{\"weights_bytes\": {}, \"kv_cache_bytes\": {}, \"scratch_bytes\": {}, \"total_resident_bytes\": {}}}",
        memory.weights_bytes,
        memory.kv_cache_bytes,
        memory.scratch_bytes,
        memory.total_resident_bytes
    ));
}

fn push_quantization_manifest_summary(out: &mut String, summary: &QuantizationManifestSummary) {
    out.push_str(&format!(
        "{{\"quantized_tensor_count\": {}, \"preserved_tensor_count\": {}, \"source_quantized_bytes\": {}, \"projected_quantized_bytes\": {}, \"projected_quantized_value_bytes\": {}, \"projected_quantized_scale_bytes\": {}, \"quantized_weight_ratio\": {:.6}, \"preserved_source_bytes\": {}, \"source_total_bytes\": {}, \"projected_total_bytes\": {}, \"projected_total_ratio\": {:.6}}}",
        summary.quantized_tensor_count,
        summary.preserved_tensor_count,
        summary.source_quantized_bytes,
        summary.projected_quantized_bytes,
        summary.projected_quantized_value_bytes,
        summary.projected_quantized_scale_bytes,
        summary.quantized_ratio(),
        summary.preserved_source_bytes,
        summary.source_total_bytes,
        summary.projected_total_bytes,
        summary.projected_total_ratio()
    ));
}

fn push_quantized_tensor_plan_json(
    out: &mut String,
    tensor: &QuantizationTensorPlan,
    indent: usize,
) {
    push_indent(out, indent);
    out.push_str("{\"name\": ");
    push_json_string(out, &tensor.name);
    out.push_str(", \"role\": ");
    push_json_string(out, &tensor.role);
    out.push_str(", \"dtype\": ");
    push_json_string(out, tensor.dtype.safetensors_name());
    out.push_str(", \"shape\": ");
    push_usize_array(out, &tensor.shape);
    out.push_str(&format!(
        ", \"rows\": {}, \"cols\": {}, \"source_bytes\": {}, \"quantized_value_bytes\": {}, \"quantized_scale_bytes\": {}, \"quantized_bytes\": {}, \"ratio\": {:.6}}}",
        tensor.rows,
        tensor.cols,
        tensor.source_bytes,
        tensor.quantized_value_bytes,
        tensor.quantized_scale_bytes,
        tensor.quantized_bytes,
        ratio_u64(tensor.quantized_bytes, tensor.source_bytes)
    ));
}

fn push_quantized_tensor_export_json(
    out: &mut String,
    tensor: &QuantizedTensorExport,
    indent: usize,
) {
    push_indent(out, indent);
    out.push_str(&format!("{{\"index\": {}, \"name\": ", tensor.index));
    push_json_string(out, &tensor.name);
    out.push_str(", \"role\": ");
    push_json_string(out, &tensor.role);
    out.push_str(", \"shape\": ");
    push_usize_array(out, &tensor.shape);
    out.push_str(&format!(
        ", \"rows\": {}, \"cols\": {}, \"values_bytes\": {}, \"scales_bytes\": {}",
        tensor.rows, tensor.cols, tensor.values_bytes, tensor.scales_bytes
    ));
    out.push_str(", \"values_path\": ");
    push_json_string(out, &tensor.values_path.display().to_string());
    out.push_str(", \"scales_path\": ");
    push_json_string(out, &tensor.scales_path.display().to_string());
    out.push('}');
}

fn push_preserved_tensor_plan(out: &mut String, tensor: &PreservedTensorPlan, indent: usize) {
    push_indent(out, indent);
    out.push_str("{\"name\": ");
    push_json_string(out, &tensor.name);
    out.push_str(", \"role\": ");
    push_json_string(out, &tensor.role);
    out.push_str(", \"dtype\": ");
    push_json_string(out, tensor.dtype.safetensors_name());
    out.push_str(", \"shape\": ");
    push_usize_array(out, &tensor.shape);
    out.push_str(&format!(", \"source_bytes\": {}}}", tensor.source_bytes));
}

fn push_chat_compare_suite_summary(out: &mut String, summary: &ChatCompareSuiteSummary) {
    out.push_str(&format!(
        "{{\"prompt_count\": {}, \"step_count\": {}, \"token_match_count\": {}, \"token_ids_match\": {}, \"top_tokens_match\": {}, \"mean_kl_divergence\": {:.12}, \"max_kl_divergence\": {:.12}, \"max_abs_diff\": {:.8}}}",
        summary.prompt_count,
        summary.step_count,
        summary.token_match_count,
        summary.token_ids_match(),
        summary.top_tokens_match,
        summary.mean_kl(),
        summary.max_kl,
        summary.max_abs_diff
    ));
}

fn push_u32_array(out: &mut String, values: &[u32]) {
    out.push('[');
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
}

fn push_usize_array(out: &mut String, values: &[usize]) {
    out.push('[');
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
}

fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
}

fn push_optional_f64_value(out: &mut String, value: Option<f64>) {
    if let Some(value) = value {
        out.push_str(&format!("{value:.12}"));
    } else {
        out.push_str("null");
    }
}

fn push_optional_f32_value(out: &mut String, value: Option<f32>) {
    if let Some(value) = value {
        out.push_str(&format!("{value:.8}"));
    } else {
        out.push_str("null");
    }
}

fn comma_line(out: &mut String, index: usize, indent: &str) {
    if index > 0 {
        out.push_str(",\n");
        out.push_str(indent);
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push(' ');
    }
}

fn push_optional_comma(out: &mut String, comma: bool) {
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn backend_label(backend: InferenceBackend) -> &'static str {
    match backend {
        InferenceBackend::Bf16 => "bf16",
        InferenceBackend::AllLinearInt8 => "all-linear-int8",
    }
}

fn driver_label(driver: GenerationComparisonDriver) -> &'static str {
    match driver {
        GenerationComparisonDriver::Candidate => "candidate",
        GenerationComparisonDriver::Reference => "reference",
    }
}

fn system_prompt_mode_label(system_prompt: &SystemPrompt) -> &'static str {
    match system_prompt {
        SystemPrompt::DefaultFromModel => "default-from-model",
        SystemPrompt::None => "none",
        SystemPrompt::Custom(_) => "custom",
    }
}

fn decode_strategy_label(sampling: Option<SamplingOptions>) -> &'static str {
    if sampling.is_some() {
        "sampled"
    } else {
        "greedy"
    }
}

fn finish_reason_label(reason: GenerationFinishReason) -> String {
    match reason {
        GenerationFinishReason::MaxNewTokens => "max-new-tokens".to_string(),
        GenerationFinishReason::StopToken(token) => format!("stop-token({token})"),
    }
}

fn f32_dtype_size_in_bytes() -> usize {
    DType::F32
        .size_in_bytes()
        .expect("F32 dtype must be byte-aligned")
}

fn invalid_input(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(ErrorKind::InvalidInput, message.into()))
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn chat_cli_accepts_explicit_prompt_flag() {
        let cli = parse_chat_cli(&args(&["--prompt", "hello"]), 0).unwrap();
        assert_eq!(cli.prompts, vec!["hello"]);
    }

    #[test]
    fn chat_cli_keeps_positional_prompt_behavior() {
        let cli = parse_chat_cli(&args(&["hello"]), 0).unwrap();
        assert_eq!(cli.prompts, vec!["hello"]);
    }

    #[test]
    fn chat_cli_prompt_flag_flushes_positional_prompt() {
        let cli = parse_chat_cli(&args(&["hello", "--prompt", "goodbye"]), 0).unwrap();
        assert_eq!(cli.prompts, vec!["hello", "goodbye"]);
    }
}
