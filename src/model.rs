use std::{fs, io, mem::size_of, path::Path, sync::Arc};

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};

use crate::{
    dtypes::Bf16,
    ops,
    quantization::{DeviceQuantizedI8Matrix, QuantizedI8Matrix, quantized_export_file_paths},
    safetensors::{ModelWeights, Result, TensorInfo},
};

#[derive(Debug, Clone, Copy)]
pub struct TextConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rope_yarn: Option<YarnRopeConfig>,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct YarnRopeConfig {
    pub factor: f32,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub attention_factor: f32,
}

impl TextConfig {
    pub const MINISTRAL_3_8B_REASONING_2512: Self = Self {
        dim: 4096,
        hidden_dim: 14336,
        n_layers: 34,
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 128,
        rope_theta: 1_000_000.0,
        rope_yarn: Some(YarnRopeConfig {
            factor: 16.0,
            original_max_position_embeddings: 16_384,
            beta_fast: 32.0,
            beta_slow: 1.0,
            // The model has mscale=1 and mscale_all_dim=1 / apply_scale=false,
            // so the YaRN magnitude correction is neutral.
            attention_factor: 1.0,
        }),
        max_position_embeddings: 262_144,
        vocab_size: 131_072,
        norm_eps: 1e-5,
    };

    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let config_json = fs::read_to_string(model_dir.as_ref().join("config.json"))?;
        Self::from_config_json(&config_json)
    }

    fn from_config_json(json: &str) -> Result<Self> {
        let text_config = parse_json_object_field(json, "text_config")?;
        let rope_parameters = parse_json_object_field(text_config, "rope_parameters").ok();
        let rope_yarn = rope_parameters
            .and_then(|rope| {
                let rope_type = parse_json_string_field(rope, "rope_type")
                    .or_else(|_| parse_json_string_field(rope, "type"))
                    .ok()?;
                (rope_type == "yarn").then_some(rope)
            })
            .map(|rope| -> Result<YarnRopeConfig> {
                Ok(YarnRopeConfig {
                    factor: parse_json_f32_field(rope, "factor")?,
                    original_max_position_embeddings: parse_json_usize_field(
                        rope,
                        "original_max_position_embeddings",
                    )?,
                    beta_fast: parse_json_f32_field(rope, "beta_fast")?,
                    beta_slow: parse_json_f32_field(rope, "beta_slow")?,
                    attention_factor: parse_optional_json_f32_field(rope, "mscale").unwrap_or(1.0),
                })
            })
            .transpose()?;

        let config = Self {
            dim: parse_json_usize_field(text_config, "hidden_size")?,
            hidden_dim: parse_json_usize_field(text_config, "intermediate_size")?,
            n_layers: parse_json_usize_field(text_config, "num_hidden_layers")?,
            n_heads: parse_json_usize_field(text_config, "num_attention_heads")?,
            n_kv_heads: parse_json_usize_field(text_config, "num_key_value_heads")?,
            head_dim: parse_json_usize_field(text_config, "head_dim")?,
            rope_theta: rope_parameters
                .and_then(|rope| parse_optional_json_f32_field(rope, "rope_theta"))
                .unwrap_or_else(|| {
                    parse_optional_json_f32_field(text_config, "rope_theta").unwrap_or(10_000.0)
                }),
            rope_yarn,
            max_position_embeddings: parse_json_usize_field(
                text_config,
                "max_position_embeddings",
            )?,
            vocab_size: parse_json_usize_field(text_config, "vocab_size")?,
            norm_eps: parse_json_f32_field(text_config, "rms_norm_eps")?,
        };
        config.validate_for_inference()?;
        Ok(config)
    }

    pub fn validate_for_inference(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(invalid_data("text config hidden_size must be nonzero"));
        }
        if self.hidden_dim == 0 {
            return Err(invalid_data(
                "text config intermediate_size must be nonzero",
            ));
        }
        if self.n_layers == 0 {
            return Err(invalid_data(
                "text config num_hidden_layers must be nonzero",
            ));
        }
        if self.n_heads == 0 {
            return Err(invalid_data(
                "text config num_attention_heads must be nonzero",
            ));
        }
        if self.n_kv_heads == 0 {
            return Err(invalid_data(
                "text config num_key_value_heads must be nonzero",
            ));
        }
        if self.head_dim == 0 || self.head_dim % 2 != 0 {
            return Err(invalid_data(format!(
                "text config head_dim must be nonzero and even, got {}",
                self.head_dim
            )));
        }
        if self.dim != self.n_heads * self.head_dim {
            return Err(invalid_data(format!(
                "text config hidden_size {} must equal num_attention_heads {} * head_dim {}",
                self.dim, self.n_heads, self.head_dim
            )));
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(invalid_data(format!(
                "text config num_attention_heads {} must be divisible by num_key_value_heads {}",
                self.n_heads, self.n_kv_heads
            )));
        }
        if self.max_position_embeddings == 0 {
            return Err(invalid_data(
                "text config max_position_embeddings must be nonzero",
            ));
        }
        if self.vocab_size == 0 {
            return Err(invalid_data("text config vocab_size must be nonzero"));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(invalid_data(format!(
                "text config rope_theta must be finite and positive, got {}",
                self.rope_theta
            )));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(invalid_data(format!(
                "text config rms_norm_eps must be finite and positive, got {}",
                self.norm_eps
            )));
        }
        if let Some(yarn) = self.rope_yarn {
            yarn.validate_for_inference()?;
        }

        Ok(())
    }
}

impl YarnRopeConfig {
    fn validate_for_inference(&self) -> Result<()> {
        if !self.factor.is_finite() || self.factor <= 0.0 {
            return Err(invalid_data(format!(
                "YaRN factor must be finite and positive, got {}",
                self.factor
            )));
        }
        if self.original_max_position_embeddings == 0 {
            return Err(invalid_data(
                "YaRN original_max_position_embeddings must be nonzero",
            ));
        }
        if !self.beta_fast.is_finite() || self.beta_fast <= 0.0 {
            return Err(invalid_data(format!(
                "YaRN beta_fast must be finite and positive, got {}",
                self.beta_fast
            )));
        }
        if !self.beta_slow.is_finite() || self.beta_slow <= 0.0 {
            return Err(invalid_data(format!(
                "YaRN beta_slow must be finite and positive, got {}",
                self.beta_slow
            )));
        }
        if !self.attention_factor.is_finite() || self.attention_factor <= 0.0 {
            return Err(invalid_data(format!(
                "YaRN attention_factor must be finite and positive, got {}",
                self.attention_factor
            )));
        }
        if (self.attention_factor - 1.0).abs() > f32::EPSILON {
            return Err(invalid_data(format!(
                "non-neutral YaRN attention_factor {} is not supported yet",
                self.attention_factor
            )));
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct FfnProbe {
    pub layer: usize,
    pub token_id: u32,
    pub output_prefix: Vec<f32>,
    pub l2_norm: f32,
    pub max_abs: f32,
}

#[derive(Debug)]
pub struct BlockProbe {
    pub layer: usize,
    pub token_id: u32,
    pub query_prefix: Vec<f32>,
    pub key_prefix: Vec<f32>,
    pub attention_prefix: Vec<f32>,
    pub output_prefix: Vec<f32>,
    pub l2_norm: f32,
    pub max_abs: f32,
}

#[derive(Debug)]
pub struct PromptBlockProbe {
    pub layer: usize,
    pub tokens: Vec<u32>,
    pub query_prefix: Vec<f32>,
    pub key_prefix: Vec<f32>,
    pub score_prefix: Vec<f32>,
    pub attention_prefix: Vec<f32>,
    pub output_prefix: Vec<f32>,
    pub l2_norm: f32,
    pub max_abs: f32,
}

#[derive(Debug)]
pub struct PromptBlockReferenceProbe {
    pub layer: usize,
    pub tokens: Vec<u32>,
    pub query_max_abs_diff: f32,
    pub key_max_abs_diff: f32,
    pub score_max_abs_diff: f32,
    pub attention_max_abs_diff: f32,
    pub output_max_abs_diff: f32,
    pub output_mean_abs_diff: f32,
    pub gpu_output_prefix: Vec<f32>,
    pub reference_output_prefix: Vec<f32>,
}

#[derive(Debug)]
pub struct BlockReferenceProbe {
    pub layer: usize,
    pub token_id: u32,
    pub query_max_abs_diff: f32,
    pub key_max_abs_diff: f32,
    pub attention_max_abs_diff: f32,
    pub output_max_abs_diff: f32,
    pub output_mean_abs_diff: f32,
    pub gpu_output_prefix: Vec<f32>,
    pub reference_output_prefix: Vec<f32>,
}

#[derive(Debug)]
pub struct TextForwardProbe {
    pub tokens: Vec<u32>,
    pub layers_run: usize,
    pub final_hidden_prefix: Vec<f32>,
    pub logits_prefix: Vec<f32>,
    pub top_logits: Vec<(u32, f32)>,
    pub final_hidden_l2_norm: f32,
    pub logits_max_abs: f32,
}

#[derive(Debug)]
pub struct OutputProjectionQuantizationProbe {
    pub tokens: Vec<u32>,
    pub rows: usize,
    pub cols: usize,
    pub reference_top_logits: Vec<(u32, f32)>,
    pub quantized_top_logits: Vec<(u32, f32)>,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub reference_logits_prefix: Vec<f32>,
    pub quantized_logits_prefix: Vec<f32>,
}

#[derive(Debug)]
pub struct FfnQuantizationProbe {
    pub tokens: Vec<u32>,
    pub layers_run: usize,
    pub reference_top_logits: Vec<(u32, f32)>,
    pub quantized_top_logits: Vec<(u32, f32)>,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub reference_logits_prefix: Vec<f32>,
    pub quantized_logits_prefix: Vec<f32>,
}

#[derive(Debug)]
pub struct AttentionQuantizationProbe {
    pub tokens: Vec<u32>,
    pub layers_run: usize,
    pub reference_top_logits: Vec<(u32, f32)>,
    pub quantized_top_logits: Vec<(u32, f32)>,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub reference_logits_prefix: Vec<f32>,
    pub quantized_logits_prefix: Vec<f32>,
}

#[derive(Debug)]
pub struct AttentionFfnQuantizationProbe {
    pub tokens: Vec<u32>,
    pub layers_run: usize,
    pub reference_top_logits: Vec<(u32, f32)>,
    pub quantized_top_logits: Vec<(u32, f32)>,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub reference_logits_prefix: Vec<f32>,
    pub quantized_logits_prefix: Vec<f32>,
}

#[derive(Debug)]
pub struct AllLinearQuantizationProbe {
    pub tokens: Vec<u32>,
    pub layers_run: usize,
    pub output_rows: usize,
    pub output_cols: usize,
    pub reference_top_logits: Vec<(u32, f32)>,
    pub quantized_top_logits: Vec<(u32, f32)>,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub reference_logits_prefix: Vec<f32>,
    pub quantized_logits_prefix: Vec<f32>,
}

#[derive(Debug)]
pub struct QuantizationSuiteVariantProbe {
    pub variant: &'static str,
    pub reference_token_id: u32,
    pub reference_logit: f32,
    pub quantized_token_id: u32,
    pub quantized_logit: f32,
    pub token_matches: bool,
    pub reference_top_logits: Vec<(u32, f32)>,
    pub quantized_top_logits: Vec<(u32, f32)>,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

#[derive(Debug)]
pub struct QuantizationSuitePromptProbe {
    pub tokens: Vec<u32>,
    pub variants: Vec<QuantizationSuiteVariantProbe>,
}

#[derive(Debug)]
pub struct QuantizationSuiteProbe {
    pub prompts: Vec<QuantizationSuitePromptProbe>,
}

#[derive(Debug)]
pub struct QuantizationGenerationSuiteStepProbe {
    pub step: usize,
    pub prefix_len: usize,
    pub reference_token_id: u32,
    pub reference_logit: f32,
    pub variants: Vec<QuantizationSuiteVariantProbe>,
}

#[derive(Debug)]
pub struct QuantizationGenerationSuitePromptProbe {
    pub initial_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub steps: Vec<QuantizationGenerationSuiteStepProbe>,
}

#[derive(Debug)]
pub struct QuantizationGenerationSuiteProbe {
    pub prompts: Vec<QuantizationGenerationSuitePromptProbe>,
    pub total_steps: usize,
}

#[derive(Debug)]
pub struct AllLinearFreeGenerationStepProbe {
    pub step: usize,
    pub prefix_len: usize,
    pub reference_token_id: u32,
    pub reference_logit: f32,
    pub quantized_token_id: u32,
    pub quantized_logit: f32,
    pub token_matches: bool,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub reference_top_logits: Vec<(u32, f32)>,
    pub quantized_top_logits: Vec<(u32, f32)>,
}

#[derive(Debug)]
pub struct AllLinearFreeGenerationProbe {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub steps: Vec<AllLinearFreeGenerationStepProbe>,
}

#[derive(Debug)]
pub struct AllLinearQuantizedRuntimeComparisonStepProbe {
    pub step: usize,
    pub prefix_len: usize,
    pub reference_token_id: u32,
    pub reference_logit: f32,
    pub quantized_token_id: u32,
    pub quantized_logit: f32,
    pub token_matches: bool,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub reference_top_logits: Vec<(u32, f32)>,
    pub quantized_top_logits: Vec<(u32, f32)>,
}

#[derive(Debug)]
pub struct AllLinearQuantizedRuntimeComparisonProbe {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub quantized_memory_stats: RuntimeMemoryStats,
    pub steps: Vec<AllLinearQuantizedRuntimeComparisonStepProbe>,
}

#[derive(Debug)]
pub struct GreedyGenerationStep {
    pub step: usize,
    pub token_id: u32,
    pub logit: f32,
    pub top_logits: Vec<(u32, f32)>,
}

#[derive(Debug)]
pub struct GreedyGenerationProbe {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub steps: Vec<GreedyGenerationStep>,
}

#[derive(Debug)]
pub struct GreedyGenerationComparisonStep {
    pub step: usize,
    pub full_token_id: u32,
    pub incremental_token_id: u32,
    pub token_matches: bool,
    pub winning_logit_abs_diff: f32,
    pub top_tokens_match: bool,
    pub top_logits_max_abs_diff: f32,
}

#[derive(Debug)]
pub struct GreedyGenerationComparisonProbe {
    pub prompt_tokens: Vec<u32>,
    pub full: GreedyGenerationProbe,
    pub incremental: GreedyGenerationProbe,
    pub generated_tokens_match: bool,
    pub max_winning_logit_abs_diff: f32,
    pub max_top_logits_abs_diff: f32,
    pub steps: Vec<GreedyGenerationComparisonStep>,
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeMemoryStats {
    pub weights_bytes: usize,
    pub kv_cache_bytes: usize,
    pub scratch_bytes: usize,
    pub total_resident_bytes: usize,
}

struct LayerHostWeights {
    attention_norm: Vec<Bf16>,
    wq: Vec<Bf16>,
    wk: Vec<Bf16>,
    wv: Vec<Bf16>,
    wo: Vec<Bf16>,
    ffn_norm: Vec<Bf16>,
    w1: Vec<Bf16>,
    w3: Vec<Bf16>,
    w2: Vec<Bf16>,
}

struct LayerDeviceWeights {
    attention_norm: DeviceBuffer<Bf16>,
    wq: DeviceBuffer<Bf16>,
    wk: DeviceBuffer<Bf16>,
    wv: DeviceBuffer<Bf16>,
    wo: DeviceBuffer<Bf16>,
    ffn_norm: DeviceBuffer<Bf16>,
    w1: DeviceBuffer<Bf16>,
    w3: DeviceBuffer<Bf16>,
    w2: DeviceBuffer<Bf16>,
}

struct QuantizedFfnLayerDeviceWeights {
    attention_norm: DeviceBuffer<Bf16>,
    wq: DeviceBuffer<Bf16>,
    wk: DeviceBuffer<Bf16>,
    wv: DeviceBuffer<Bf16>,
    wo: DeviceBuffer<Bf16>,
    ffn_norm: DeviceBuffer<Bf16>,
    w1: DeviceQuantizedI8Matrix,
    w3: DeviceQuantizedI8Matrix,
    w2: DeviceQuantizedI8Matrix,
}

struct QuantizedAttentionLayerDeviceWeights {
    attention_norm: DeviceBuffer<Bf16>,
    wq: DeviceQuantizedI8Matrix,
    wk: DeviceQuantizedI8Matrix,
    wv: DeviceQuantizedI8Matrix,
    wo: DeviceQuantizedI8Matrix,
    ffn_norm: DeviceBuffer<Bf16>,
    w1: DeviceBuffer<Bf16>,
    w3: DeviceBuffer<Bf16>,
    w2: DeviceBuffer<Bf16>,
}

struct QuantizedAttentionFfnLayerDeviceWeights {
    attention_norm: DeviceBuffer<Bf16>,
    wq: DeviceQuantizedI8Matrix,
    wk: DeviceQuantizedI8Matrix,
    wv: DeviceQuantizedI8Matrix,
    wo: DeviceQuantizedI8Matrix,
    ffn_norm: DeviceBuffer<Bf16>,
    w1: DeviceQuantizedI8Matrix,
    w3: DeviceQuantizedI8Matrix,
    w2: DeviceQuantizedI8Matrix,
}

trait AttentionLayerWeights {
    type Wq: ops::CudaLinearWeight;
    type Wk: ops::CudaLinearWeight;
    type Wv: ops::CudaLinearWeight;
    type Wo: ops::CudaLinearWeight;

    fn attention_norm(&self) -> &DeviceBuffer<Bf16>;
    fn wq(&self) -> &Self::Wq;
    fn wk(&self) -> &Self::Wk;
    fn wv(&self) -> &Self::Wv;
    fn wo(&self) -> &Self::Wo;
}

trait FfnLayerWeights {
    type W1: ops::CudaLinearWeight;
    type W3: ops::CudaLinearWeight;
    type W2: ops::CudaLinearWeight;

    fn ffn_norm(&self) -> &DeviceBuffer<Bf16>;
    fn w1(&self) -> &Self::W1;
    fn w3(&self) -> &Self::W3;
    fn w2(&self) -> &Self::W2;
}

impl AttentionLayerWeights for LayerDeviceWeights {
    type Wq = DeviceBuffer<Bf16>;
    type Wk = DeviceBuffer<Bf16>;
    type Wv = DeviceBuffer<Bf16>;
    type Wo = DeviceBuffer<Bf16>;

    fn attention_norm(&self) -> &DeviceBuffer<Bf16> {
        &self.attention_norm
    }

    fn wq(&self) -> &Self::Wq {
        &self.wq
    }

    fn wk(&self) -> &Self::Wk {
        &self.wk
    }

    fn wv(&self) -> &Self::Wv {
        &self.wv
    }

    fn wo(&self) -> &Self::Wo {
        &self.wo
    }
}

impl FfnLayerWeights for LayerDeviceWeights {
    type W1 = DeviceBuffer<Bf16>;
    type W3 = DeviceBuffer<Bf16>;
    type W2 = DeviceBuffer<Bf16>;

    fn ffn_norm(&self) -> &DeviceBuffer<Bf16> {
        &self.ffn_norm
    }

    fn w1(&self) -> &Self::W1 {
        &self.w1
    }

    fn w3(&self) -> &Self::W3 {
        &self.w3
    }

    fn w2(&self) -> &Self::W2 {
        &self.w2
    }
}

impl AttentionLayerWeights for QuantizedFfnLayerDeviceWeights {
    type Wq = DeviceBuffer<Bf16>;
    type Wk = DeviceBuffer<Bf16>;
    type Wv = DeviceBuffer<Bf16>;
    type Wo = DeviceBuffer<Bf16>;

    fn attention_norm(&self) -> &DeviceBuffer<Bf16> {
        &self.attention_norm
    }

    fn wq(&self) -> &Self::Wq {
        &self.wq
    }

    fn wk(&self) -> &Self::Wk {
        &self.wk
    }

    fn wv(&self) -> &Self::Wv {
        &self.wv
    }

    fn wo(&self) -> &Self::Wo {
        &self.wo
    }
}

impl FfnLayerWeights for QuantizedFfnLayerDeviceWeights {
    type W1 = DeviceQuantizedI8Matrix;
    type W3 = DeviceQuantizedI8Matrix;
    type W2 = DeviceQuantizedI8Matrix;

    fn ffn_norm(&self) -> &DeviceBuffer<Bf16> {
        &self.ffn_norm
    }

    fn w1(&self) -> &Self::W1 {
        &self.w1
    }

    fn w3(&self) -> &Self::W3 {
        &self.w3
    }

    fn w2(&self) -> &Self::W2 {
        &self.w2
    }
}

impl AttentionLayerWeights for QuantizedAttentionLayerDeviceWeights {
    type Wq = DeviceQuantizedI8Matrix;
    type Wk = DeviceQuantizedI8Matrix;
    type Wv = DeviceQuantizedI8Matrix;
    type Wo = DeviceQuantizedI8Matrix;

    fn attention_norm(&self) -> &DeviceBuffer<Bf16> {
        &self.attention_norm
    }

    fn wq(&self) -> &Self::Wq {
        &self.wq
    }

    fn wk(&self) -> &Self::Wk {
        &self.wk
    }

    fn wv(&self) -> &Self::Wv {
        &self.wv
    }

    fn wo(&self) -> &Self::Wo {
        &self.wo
    }
}

impl FfnLayerWeights for QuantizedAttentionLayerDeviceWeights {
    type W1 = DeviceBuffer<Bf16>;
    type W3 = DeviceBuffer<Bf16>;
    type W2 = DeviceBuffer<Bf16>;

    fn ffn_norm(&self) -> &DeviceBuffer<Bf16> {
        &self.ffn_norm
    }

    fn w1(&self) -> &Self::W1 {
        &self.w1
    }

    fn w3(&self) -> &Self::W3 {
        &self.w3
    }

    fn w2(&self) -> &Self::W2 {
        &self.w2
    }
}

impl AttentionLayerWeights for QuantizedAttentionFfnLayerDeviceWeights {
    type Wq = DeviceQuantizedI8Matrix;
    type Wk = DeviceQuantizedI8Matrix;
    type Wv = DeviceQuantizedI8Matrix;
    type Wo = DeviceQuantizedI8Matrix;

    fn attention_norm(&self) -> &DeviceBuffer<Bf16> {
        &self.attention_norm
    }

    fn wq(&self) -> &Self::Wq {
        &self.wq
    }

    fn wk(&self) -> &Self::Wk {
        &self.wk
    }

    fn wv(&self) -> &Self::Wv {
        &self.wv
    }

    fn wo(&self) -> &Self::Wo {
        &self.wo
    }
}

impl FfnLayerWeights for QuantizedAttentionFfnLayerDeviceWeights {
    type W1 = DeviceQuantizedI8Matrix;
    type W3 = DeviceQuantizedI8Matrix;
    type W2 = DeviceQuantizedI8Matrix;

    fn ffn_norm(&self) -> &DeviceBuffer<Bf16> {
        &self.ffn_norm
    }

    fn w1(&self) -> &Self::W1 {
        &self.w1
    }

    fn w3(&self) -> &Self::W3 {
        &self.w3
    }

    fn w2(&self) -> &Self::W2 {
        &self.w2
    }
}

trait DeviceWeightBytes {
    fn device_weight_bytes(&self) -> usize;
}

impl DeviceWeightBytes for DeviceBuffer<Bf16> {
    fn device_weight_bytes(&self) -> usize {
        self.len() * size_of::<Bf16>()
    }
}

impl DeviceWeightBytes for DeviceQuantizedI8Matrix {
    fn device_weight_bytes(&self) -> usize {
        quantized_matrix_device_bytes(self)
    }
}

impl DeviceWeightBytes for LayerDeviceWeights {
    fn device_weight_bytes(&self) -> usize {
        layer_device_weight_bytes(self)
    }
}

impl DeviceWeightBytes for QuantizedAttentionFfnLayerDeviceWeights {
    fn device_weight_bytes(&self) -> usize {
        quantized_attention_ffn_layer_device_weight_bytes(self)
    }
}

struct LayerKvCache {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
}

struct RuntimeScratch {
    hidden_a: DeviceBuffer<f32>,
    hidden_b: DeviceBuffer<f32>,
    prompt_a: Vec<DeviceBuffer<f32>>,
    prompt_b: Vec<DeviceBuffer<f32>>,
    prompt_a_batch: DeviceBuffer<f32>,
    prompt_b_batch: DeviceBuffer<f32>,
    layer: LayerScratch,
    logits_normed: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    argmax_packed: DeviceBuffer<u64>,
    output_top1_tokens: DeviceBuffer<u32>,
    output_top1_logits: DeviceBuffer<f32>,
    topk_tokens: DeviceBuffer<u32>,
    topk_logits: DeviceBuffer<f32>,
}

struct LayerScratch {
    attention_normed: DeviceBuffer<f32>,
    attention_normed_batch: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    query_batch: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    key_batch: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    value_batch: DeviceBuffer<f32>,
    query_rot: DeviceBuffer<f32>,
    query_rot_batch: DeviceBuffer<f32>,
    key_rot: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    attention_heads: DeviceBuffer<f32>,
    attention_heads_batch: DeviceBuffer<f32>,
    attention_delta: DeviceBuffer<f32>,
    attention_delta_batch: DeviceBuffer<f32>,
    attention_residual: DeviceBuffer<f32>,
    attention_residual_batch: DeviceBuffer<f32>,
    ffn: FfnScratch,
}

struct FfnScratch {
    normed: DeviceBuffer<f32>,
    normed_batch: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    gate_batch: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    up_batch: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    activated_batch: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    down_batch: DeviceBuffer<f32>,
}

const DEVICE_TOP_K_CAPACITY: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bf16Top1Plan {
    Rows1,
    Rows2,
    Rows4,
    Rows8,
}

impl Default for Bf16Top1Plan {
    fn default() -> Self {
        Self::Rows4
    }
}

impl Bf16Top1Plan {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Rows1 => "rows1",
            Self::Rows2 => "rows2",
            Self::Rows4 => "rows4",
            Self::Rows8 => "rows8",
        }
    }
}

struct SingleTokenBlockOutput {
    query: Vec<f32>,
    key: Vec<f32>,
    attention_residual: Vec<f32>,
    output: Vec<f32>,
}

struct PromptBlockOutput {
    query: Vec<f32>,
    key: Vec<f32>,
    scores: Vec<f32>,
    attention_residual: Vec<f32>,
    output: Vec<f32>,
}

pub struct MinistralTextRuntime {
    core: RuntimeCore<LayerDeviceWeights, DeviceBuffer<Bf16>>,
}

pub struct MinistralAllLinearQuantizedRuntime {
    core: RuntimeCore<QuantizedAttentionFfnLayerDeviceWeights, DeviceQuantizedI8Matrix>,
}

struct RuntimeCore<L, O> {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    config: TextConfig,
    dev_tok_embeddings: DeviceBuffer<Bf16>,
    dev_rope_freqs: DeviceBuffer<f32>,
    dev_norm_weight: DeviceBuffer<Bf16>,
    dev_output_weight: O,
    layers: Vec<L>,
    caches: Vec<LayerKvCache>,
    scratch: RuntimeScratch,
    max_seq_len: usize,
    tokens: Vec<u32>,
    next_logits_ready: bool,
    top1_plan: Bf16Top1Plan,
}

impl<L, O> RuntimeCore<L, O>
where
    L: AttentionLayerWeights + FfnLayerWeights + DeviceWeightBytes,
    O: ops::CudaLinearWeight + DeviceWeightBytes,
{
    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        if !self.tokens.is_empty() {
            return Err(invalid_data("runtime has already been prefilled"));
        }
        if prompt_tokens.is_empty() {
            return Err(invalid_data("runtime prefill requires at least one token"));
        }
        if prompt_tokens.len() > self.max_seq_len {
            return Err(invalid_data(format!(
                "prompt length {} exceeds runtime max_seq_len {}",
                prompt_tokens.len(),
                self.max_seq_len
            )));
        }
        for &token in prompt_tokens {
            validate_token(&self.config, token)?;
        }

        write_prompt_embeddings_device(
            &self.stream,
            &self.module,
            &self.config,
            &self.dev_tok_embeddings,
            prompt_tokens,
            &mut self.scratch.prompt_a,
        )?;

        let mut hidden_is_a = true;
        let prompt_len = prompt_tokens.len();
        for (layer, cache) in self.layers.iter().zip(self.caches.iter_mut()) {
            let RuntimeScratch {
                prompt_a,
                prompt_b,
                layer: layer_scratch,
                ..
            } = &mut self.scratch;

            if hidden_is_a {
                run_prefill_layer_device_with_cache(
                    &self.stream,
                    &self.module,
                    &self.config,
                    layer,
                    &self.dev_rope_freqs,
                    self.max_seq_len,
                    cache,
                    layer_scratch,
                    &prompt_a[..prompt_len],
                    &mut prompt_b[..prompt_len],
                )?;
            } else {
                run_prefill_layer_device_with_cache(
                    &self.stream,
                    &self.module,
                    &self.config,
                    layer,
                    &self.dev_rope_freqs,
                    self.max_seq_len,
                    cache,
                    layer_scratch,
                    &prompt_b[..prompt_len],
                    &mut prompt_a[..prompt_len],
                )?;
            }
            hidden_is_a = !hidden_is_a;
        }

        let final_hidden = if hidden_is_a {
            &self.scratch.prompt_a[prompt_len - 1]
        } else {
            &self.scratch.prompt_b[prompt_len - 1]
        };

        logits_from_hidden_device_into(
            &self.stream,
            &self.module,
            &self.config,
            final_hidden,
            &self.dev_norm_weight,
            &self.dev_output_weight,
            &mut self.scratch.logits_normed,
            &mut self.scratch.logits,
        )?;
        self.stream.synchronize()?;
        self.next_logits_ready = true;
        self.tokens = prompt_tokens.to_vec();

        Ok(())
    }

    fn generate_greedy_until(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        if !self.next_logits_ready {
            return Err(invalid_data("runtime must be prefilled before generation"));
        }

        let mut steps = Vec::with_capacity(max_new_tokens);
        let reported_top_k = top_k.max(1);

        for step in 0..max_new_tokens {
            let top_logits = {
                let RuntimeScratch {
                    logits,
                    argmax_packed,
                    topk_tokens,
                    topk_logits,
                    ..
                } = &mut self.scratch;
                top_k_logits_device(
                    &self.stream,
                    &self.module,
                    logits,
                    reported_top_k,
                    argmax_packed,
                    topk_tokens,
                    topk_logits,
                )?
            };
            let &(token_id, logit) = top_logits
                .first()
                .ok_or_else(|| invalid_data("runtime produced no top logits"))?;

            self.accept_token(token_id)?;
            steps.push(GreedyGenerationStep {
                step,
                token_id,
                logit,
                top_logits,
            });
            if stop_token_id == Some(token_id) {
                break;
            }
        }

        Ok(steps)
    }

    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn next_logits_to_host(&self) -> Result<Vec<f32>> {
        if !self.next_logits_ready {
            return Err(invalid_data("runtime has no next-token logits ready"));
        }

        Ok(self.scratch.logits.to_host_vec(&self.stream)?)
    }

    fn next_top_logits(&mut self, top_k: usize) -> Result<Vec<(u32, f32)>> {
        if !self.next_logits_ready {
            return Err(invalid_data("runtime has no next-token logits ready"));
        }

        let RuntimeScratch {
            logits,
            argmax_packed,
            topk_tokens,
            topk_logits,
            ..
        } = &mut self.scratch;
        top_k_logits_device(
            &self.stream,
            &self.module,
            logits,
            top_k.max(1),
            argmax_packed,
            topk_tokens,
            topk_logits,
        )
    }

    fn advance_with_token(&mut self, token_id: u32) -> Result<()> {
        if !self.next_logits_ready {
            return Err(invalid_data(
                "runtime must have next-token logits ready before advancing",
            ));
        }

        self.accept_token(token_id)
    }

    fn memory_stats(&self) -> RuntimeMemoryStats {
        let weights_bytes = self.dev_tok_embeddings.len() * size_of::<Bf16>()
            + self.dev_rope_freqs.len() * size_of::<f32>()
            + self.dev_norm_weight.len() * size_of::<Bf16>()
            + self.dev_output_weight.device_weight_bytes()
            + self
                .layers
                .iter()
                .map(DeviceWeightBytes::device_weight_bytes)
                .sum::<usize>();
        let kv_cache_bytes = self
            .caches
            .iter()
            .map(|cache| (cache.key.len() + cache.value.len()) * size_of::<f32>())
            .sum();
        let scratch_bytes = self.scratch.bytes();

        RuntimeMemoryStats {
            weights_bytes,
            kv_cache_bytes,
            scratch_bytes,
            total_resident_bytes: weights_bytes + kv_cache_bytes + scratch_bytes,
        }
    }

    fn reset_sequence(&mut self) {
        self.tokens.clear();
        self.next_logits_ready = false;
    }

    fn accept_token(&mut self, token_id: u32) -> Result<()> {
        let position = self.tokens.len();
        if position >= self.max_seq_len {
            return Err(invalid_data(format!(
                "cannot append token at position {position}; runtime max_seq_len is {}",
                self.max_seq_len
            )));
        }

        let final_hidden_is_a = run_incremental_token_forward_into(
            &self.stream,
            &self.module,
            &self.config,
            &self.dev_rope_freqs,
            &self.dev_tok_embeddings,
            &self.layers,
            &mut self.caches,
            &mut self.scratch,
            self.max_seq_len,
            position,
            token_id,
        )?;
        let RuntimeScratch {
            hidden_a,
            hidden_b,
            logits_normed,
            logits,
            ..
        } = &mut self.scratch;
        let hidden = if final_hidden_is_a {
            &*hidden_a
        } else {
            &*hidden_b
        };
        logits_from_hidden_device_into(
            &self.stream,
            &self.module,
            &self.config,
            hidden,
            &self.dev_norm_weight,
            &self.dev_output_weight,
            logits_normed,
            logits,
        )?;
        self.next_logits_ready = true;
        self.tokens.push(token_id);

        Ok(())
    }
}

impl RuntimeCore<LayerDeviceWeights, DeviceBuffer<Bf16>> {
    fn prefill_batched_bf16_hidden(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        if !self.tokens.is_empty() {
            return Err(invalid_data("runtime has already been prefilled"));
        }
        if prompt_tokens.is_empty() {
            return Err(invalid_data("runtime prefill requires at least one token"));
        }
        if prompt_tokens.len() > self.max_seq_len {
            return Err(invalid_data(format!(
                "prompt length {} exceeds runtime max_seq_len {}",
                prompt_tokens.len(),
                self.max_seq_len
            )));
        }
        for &token in prompt_tokens {
            validate_token(&self.config, token)?;
        }

        let prompt_tokens_dev = DeviceBuffer::from_host(&self.stream, prompt_tokens)?;
        ops::embedding_tokens_bf16(
            &self.stream,
            &self.module,
            &self.dev_tok_embeddings,
            &prompt_tokens_dev,
            prompt_tokens.len(),
            self.config.dim,
            &mut self.scratch.prompt_a_batch,
        )?;

        let mut hidden_is_a = true;
        let prompt_len = prompt_tokens.len();
        for (layer, cache) in self.layers.iter().zip(self.caches.iter_mut()) {
            let RuntimeScratch {
                prompt_a_batch,
                prompt_b_batch,
                layer: layer_scratch,
                ..
            } = &mut self.scratch;

            if hidden_is_a {
                run_prefill_bf16_layer_device_with_cache(
                    &self.stream,
                    &self.module,
                    &self.config,
                    layer,
                    &self.dev_rope_freqs,
                    self.max_seq_len,
                    prompt_len,
                    cache,
                    layer_scratch,
                    prompt_a_batch,
                    prompt_b_batch,
                )?;
            } else {
                run_prefill_bf16_layer_device_with_cache(
                    &self.stream,
                    &self.module,
                    &self.config,
                    layer,
                    &self.dev_rope_freqs,
                    self.max_seq_len,
                    prompt_len,
                    cache,
                    layer_scratch,
                    prompt_b_batch,
                    prompt_a_batch,
                )?;
            }
            hidden_is_a = !hidden_is_a;
        }

        let RuntimeScratch {
            prompt_a_batch,
            prompt_b_batch,
            hidden_a,
            ..
        } = &mut self.scratch;
        let final_hidden_batch = if hidden_is_a {
            &*prompt_a_batch
        } else {
            &*prompt_b_batch
        };
        ops::copy_matrix_row_to_vector(
            &self.stream,
            &self.module,
            final_hidden_batch,
            prompt_len - 1,
            self.config.dim,
            hidden_a,
        )?;

        self.next_logits_ready = false;
        self.tokens = prompt_tokens.to_vec();
        Ok(())
    }

    fn prefill_batched_bf16(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        self.prefill_batched_bf16_hidden(prompt_tokens)?;

        let RuntimeScratch {
            hidden_a,
            logits_normed,
            logits,
            ..
        } = &mut self.scratch;
        logits_from_hidden_device_into(
            &self.stream,
            &self.module,
            &self.config,
            hidden_a,
            &self.dev_norm_weight,
            &self.dev_output_weight,
            logits_normed,
            logits,
        )?;
        self.stream.synchronize()?;
        self.next_logits_ready = true;

        Ok(())
    }

    fn prefill_batched_bf16_top1(&mut self, prompt_tokens: &[u32]) -> Result<(u32, f32)> {
        self.prefill_batched_bf16_hidden(prompt_tokens)?;

        let RuntimeScratch {
            hidden_a,
            logits_normed,
            argmax_packed,
            output_top1_tokens,
            output_top1_logits,
            ..
        } = &mut self.scratch;
        let top1 = output_top1_from_hidden_bf16_into(
            &self.stream,
            &self.module,
            &self.config,
            self.top1_plan,
            hidden_a,
            &self.dev_norm_weight,
            &self.dev_output_weight,
            logits_normed,
            output_top1_tokens,
            output_top1_logits,
            argmax_packed,
        )?;
        self.next_logits_ready = false;
        Ok(top1)
    }

    fn generate_greedy_bf16(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
    ) -> Result<Vec<GreedyGenerationStep>> {
        self.generate_greedy_until_bf16(max_new_tokens, top_k, None)
    }

    fn generate_greedy_until_bf16(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        if !self.next_logits_ready {
            return Err(invalid_data("runtime must be prefilled before generation"));
        }

        let mut steps = Vec::with_capacity(max_new_tokens);
        let reported_top_k = top_k.max(1);
        if reported_top_k == 1 {
            return self.generate_greedy_until_bf16_top1(max_new_tokens, stop_token_id);
        }

        for step in 0..max_new_tokens {
            let top_logits = {
                let RuntimeScratch {
                    logits,
                    argmax_packed,
                    topk_tokens,
                    topk_logits,
                    ..
                } = &mut self.scratch;
                top_k_logits_device(
                    &self.stream,
                    &self.module,
                    logits,
                    reported_top_k,
                    argmax_packed,
                    topk_tokens,
                    topk_logits,
                )?
            };
            let &(token_id, logit) = top_logits
                .first()
                .ok_or_else(|| invalid_data("runtime produced no top logits"))?;

            self.accept_token_bf16(token_id)?;
            steps.push(GreedyGenerationStep {
                step,
                token_id,
                logit,
                top_logits,
            });
            if stop_token_id == Some(token_id) {
                break;
            }
        }

        Ok(steps)
    }

    fn generate_greedy_until_bf16_top1(
        &mut self,
        max_new_tokens: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }

        let next_top = {
            let RuntimeScratch {
                logits,
                argmax_packed,
                topk_tokens,
                topk_logits,
                ..
            } = &mut self.scratch;
            let top_logits = top_k_logits_device(
                &self.stream,
                &self.module,
                logits,
                1,
                argmax_packed,
                topk_tokens,
                topk_logits,
            )?;
            *top_logits
                .first()
                .ok_or_else(|| invalid_data("runtime produced no top logits"))?
        };

        self.generate_greedy_until_bf16_top1_from_next(max_new_tokens, stop_token_id, next_top)
    }

    fn generate_greedy_until_bf16_top1_from_next(
        &mut self,
        max_new_tokens: usize,
        stop_token_id: Option<u32>,
        mut next_top: (u32, f32),
    ) -> Result<Vec<GreedyGenerationStep>> {
        let mut steps = Vec::with_capacity(max_new_tokens);
        for step in 0..max_new_tokens {
            let (token_id, logit) = next_top;
            let step_top_logits = vec![next_top];
            let final_hidden_is_a = self.run_bf16_token_forward(token_id)?;
            let should_finish = stop_token_id == Some(token_id) || step + 1 == max_new_tokens;

            let computed_next_top = if should_finish {
                let RuntimeScratch {
                    hidden_a,
                    hidden_b,
                    logits_normed,
                    logits,
                    ..
                } = &mut self.scratch;
                let hidden = if final_hidden_is_a {
                    &*hidden_a
                } else {
                    &*hidden_b
                };
                logits_from_hidden_device_into(
                    &self.stream,
                    &self.module,
                    &self.config,
                    hidden,
                    &self.dev_norm_weight,
                    &self.dev_output_weight,
                    logits_normed,
                    logits,
                )?;
                self.next_logits_ready = true;
                None
            } else {
                let RuntimeScratch {
                    hidden_a,
                    hidden_b,
                    logits_normed,
                    argmax_packed,
                    output_top1_tokens,
                    output_top1_logits,
                    ..
                } = &mut self.scratch;
                let hidden = if final_hidden_is_a {
                    &*hidden_a
                } else {
                    &*hidden_b
                };
                let top1 = output_top1_from_hidden_bf16_into(
                    &self.stream,
                    &self.module,
                    &self.config,
                    self.top1_plan,
                    hidden,
                    &self.dev_norm_weight,
                    &self.dev_output_weight,
                    logits_normed,
                    output_top1_tokens,
                    output_top1_logits,
                    argmax_packed,
                )?;
                self.next_logits_ready = false;
                Some(top1)
            };

            self.tokens.push(token_id);
            steps.push(GreedyGenerationStep {
                step,
                token_id,
                logit,
                top_logits: step_top_logits,
            });
            if should_finish {
                break;
            }
            next_top = computed_next_top.ok_or_else(|| {
                invalid_data("BF16 top-1 generation did not produce continuation logits")
            })?;
        }

        Ok(steps)
    }

    fn advance_with_token_bf16(&mut self, token_id: u32) -> Result<()> {
        if !self.next_logits_ready {
            return Err(invalid_data(
                "runtime must have next-token logits ready before advancing",
            ));
        }

        self.accept_token_bf16(token_id)
    }

    fn run_bf16_token_forward(&mut self, token_id: u32) -> Result<bool> {
        let position = self.tokens.len();
        if position >= self.max_seq_len {
            return Err(invalid_data(format!(
                "cannot append token at position {position}; runtime max_seq_len is {}",
                self.max_seq_len
            )));
        }

        run_incremental_bf16_token_forward_into(
            &self.stream,
            &self.module,
            &self.config,
            &self.dev_rope_freqs,
            &self.dev_tok_embeddings,
            &self.layers,
            &mut self.caches,
            &mut self.scratch,
            self.max_seq_len,
            position,
            token_id,
        )
    }

    fn accept_token_bf16(&mut self, token_id: u32) -> Result<()> {
        let final_hidden_is_a = self.run_bf16_token_forward(token_id)?;
        let RuntimeScratch {
            hidden_a,
            hidden_b,
            logits_normed,
            logits,
            ..
        } = &mut self.scratch;
        let hidden = if final_hidden_is_a {
            &*hidden_a
        } else {
            &*hidden_b
        };
        logits_from_hidden_device_into(
            &self.stream,
            &self.module,
            &self.config,
            hidden,
            &self.dev_norm_weight,
            &self.dev_output_weight,
            logits_normed,
            logits,
        )?;
        self.next_logits_ready = true;
        self.tokens.push(token_id);

        Ok(())
    }
}

impl RuntimeCore<QuantizedAttentionFfnLayerDeviceWeights, DeviceQuantizedI8Matrix> {
    fn prefill_batched_i8_hidden(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        if !self.tokens.is_empty() {
            return Err(invalid_data("runtime has already been prefilled"));
        }
        if prompt_tokens.is_empty() {
            return Err(invalid_data("runtime prefill requires at least one token"));
        }
        if prompt_tokens.len() > self.max_seq_len {
            return Err(invalid_data(format!(
                "prompt length {} exceeds runtime max_seq_len {}",
                prompt_tokens.len(),
                self.max_seq_len
            )));
        }
        for &token in prompt_tokens {
            validate_token(&self.config, token)?;
        }

        let prompt_tokens_dev = DeviceBuffer::from_host(&self.stream, prompt_tokens)?;
        ops::embedding_tokens_bf16(
            &self.stream,
            &self.module,
            &self.dev_tok_embeddings,
            &prompt_tokens_dev,
            prompt_tokens.len(),
            self.config.dim,
            &mut self.scratch.prompt_a_batch,
        )?;

        let mut hidden_is_a = true;
        let prompt_len = prompt_tokens.len();
        for (layer, cache) in self.layers.iter().zip(self.caches.iter_mut()) {
            let RuntimeScratch {
                prompt_a_batch,
                prompt_b_batch,
                layer: layer_scratch,
                ..
            } = &mut self.scratch;

            if hidden_is_a {
                run_prefill_quantized_layer_device_with_cache(
                    &self.stream,
                    &self.module,
                    &self.config,
                    layer,
                    &self.dev_rope_freqs,
                    self.max_seq_len,
                    prompt_len,
                    cache,
                    layer_scratch,
                    prompt_a_batch,
                    prompt_b_batch,
                )?;
            } else {
                run_prefill_quantized_layer_device_with_cache(
                    &self.stream,
                    &self.module,
                    &self.config,
                    layer,
                    &self.dev_rope_freqs,
                    self.max_seq_len,
                    prompt_len,
                    cache,
                    layer_scratch,
                    prompt_b_batch,
                    prompt_a_batch,
                )?;
            }
            hidden_is_a = !hidden_is_a;
        }

        let RuntimeScratch {
            prompt_a_batch,
            prompt_b_batch,
            hidden_a,
            ..
        } = &mut self.scratch;
        let final_hidden_batch = if hidden_is_a {
            &*prompt_a_batch
        } else {
            &*prompt_b_batch
        };
        ops::copy_matrix_row_to_vector(
            &self.stream,
            &self.module,
            final_hidden_batch,
            prompt_len - 1,
            self.config.dim,
            hidden_a,
        )?;

        self.next_logits_ready = false;
        self.tokens = prompt_tokens.to_vec();
        Ok(())
    }

    fn prefill_batched_i8(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        self.prefill_batched_i8_hidden(prompt_tokens)?;

        let RuntimeScratch {
            hidden_a,
            logits_normed,
            logits,
            ..
        } = &mut self.scratch;
        logits_from_hidden_device_into(
            &self.stream,
            &self.module,
            &self.config,
            hidden_a,
            &self.dev_norm_weight,
            &self.dev_output_weight,
            logits_normed,
            logits,
        )?;
        self.stream.synchronize()?;
        self.next_logits_ready = true;

        Ok(())
    }

    fn generate_greedy_until_i8(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        if !self.next_logits_ready {
            return Err(invalid_data("runtime must be prefilled before generation"));
        }

        let reported_top_k = top_k.max(1);
        if reported_top_k != 1 {
            return self.generate_greedy_until(max_new_tokens, reported_top_k, stop_token_id);
        }
        self.generate_greedy_until_i8_top1(max_new_tokens, stop_token_id)
    }

    fn generate_greedy_until_i8_top1(
        &mut self,
        max_new_tokens: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }

        let next_top = {
            let RuntimeScratch {
                logits,
                argmax_packed,
                topk_tokens,
                topk_logits,
                ..
            } = &mut self.scratch;
            let top_logits = top_k_logits_device(
                &self.stream,
                &self.module,
                logits,
                1,
                argmax_packed,
                topk_tokens,
                topk_logits,
            )?;
            *top_logits
                .first()
                .ok_or_else(|| invalid_data("runtime produced no top logits"))?
        };

        self.generate_greedy_until_i8_top1_from_next(max_new_tokens, stop_token_id, next_top)
    }

    fn generate_greedy_until_i8_top1_from_next(
        &mut self,
        max_new_tokens: usize,
        stop_token_id: Option<u32>,
        mut next_top: (u32, f32),
    ) -> Result<Vec<GreedyGenerationStep>> {
        let mut steps = Vec::with_capacity(max_new_tokens);
        for step in 0..max_new_tokens {
            let (token_id, logit) = next_top;
            let step_top_logits = vec![next_top];
            let final_hidden_is_a = self.run_i8_token_forward(token_id)?;
            let should_finish = stop_token_id == Some(token_id) || step + 1 == max_new_tokens;

            let computed_next_top = if should_finish {
                let RuntimeScratch {
                    hidden_a,
                    hidden_b,
                    logits_normed,
                    logits,
                    ..
                } = &mut self.scratch;
                let hidden = if final_hidden_is_a {
                    &*hidden_a
                } else {
                    &*hidden_b
                };
                logits_from_hidden_device_into(
                    &self.stream,
                    &self.module,
                    &self.config,
                    hidden,
                    &self.dev_norm_weight,
                    &self.dev_output_weight,
                    logits_normed,
                    logits,
                )?;
                self.next_logits_ready = true;
                None
            } else {
                let RuntimeScratch {
                    hidden_a,
                    hidden_b,
                    logits_normed,
                    argmax_packed,
                    output_top1_tokens,
                    output_top1_logits,
                    ..
                } = &mut self.scratch;
                let hidden = if final_hidden_is_a {
                    &*hidden_a
                } else {
                    &*hidden_b
                };
                let top1 = output_top1_from_hidden_i8_into(
                    &self.stream,
                    &self.module,
                    &self.config,
                    hidden,
                    &self.dev_norm_weight,
                    &self.dev_output_weight,
                    logits_normed,
                    output_top1_tokens,
                    output_top1_logits,
                    argmax_packed,
                )?;
                self.next_logits_ready = false;
                Some(top1)
            };

            self.tokens.push(token_id);
            steps.push(GreedyGenerationStep {
                step,
                token_id,
                logit,
                top_logits: step_top_logits,
            });
            if should_finish {
                break;
            }
            next_top = computed_next_top.ok_or_else(|| {
                invalid_data("scaled-i8 top-1 generation did not produce continuation logits")
            })?;
        }

        Ok(steps)
    }

    fn run_i8_token_forward(&mut self, token_id: u32) -> Result<bool> {
        let position = self.tokens.len();
        if position >= self.max_seq_len {
            return Err(invalid_data(format!(
                "cannot append token at position {position}; runtime max_seq_len is {}",
                self.max_seq_len
            )));
        }

        run_incremental_token_forward_into(
            &self.stream,
            &self.module,
            &self.config,
            &self.dev_rope_freqs,
            &self.dev_tok_embeddings,
            &self.layers,
            &mut self.caches,
            &mut self.scratch,
            self.max_seq_len,
            position,
            token_id,
        )
    }
}

impl MinistralTextRuntime {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(invalid_data("runtime max_seq_len must be nonzero"));
        }

        let model_dir = model_dir.as_ref();
        let config = TextConfig::from_model_dir(model_dir)?;
        validate_runtime_max_seq_len(&config, max_seq_len)?;
        let weights = ModelWeights::open_model_dir(model_dir)?;
        let token_embedding = read_embedding_weight(&weights, &config)?;
        let rope_freqs = rope_frequencies(&config);
        let norm_weight = read_model_norm_weight(&weights, &config)?;
        let output_weight = read_output_weight(&weights, &config)?;
        let dev_tok_embeddings = DeviceBuffer::from_host(&stream, &token_embedding)?;
        let dev_rope_freqs = DeviceBuffer::from_host(&stream, &rope_freqs)?;
        let dev_norm_weight = DeviceBuffer::from_host(&stream, &norm_weight)?;
        let dev_output_weight = DeviceBuffer::from_host(&stream, &output_weight)?;
        drop(token_embedding);
        let layers = load_layer_device_weights(&stream, &weights, &config)?;
        let caches = allocate_layer_kv_caches(&stream, &config, max_seq_len)?;
        let scratch = RuntimeScratch::new(&stream, &config, max_seq_len)?;
        stream.synchronize()?;

        Ok(Self {
            core: RuntimeCore {
                stream,
                module,
                config,
                dev_tok_embeddings,
                dev_rope_freqs,
                dev_norm_weight,
                dev_output_weight,
                layers,
                caches,
                scratch,
                max_seq_len,
                tokens: Vec::new(),
                next_logits_ready: false,
                top1_plan: Bf16Top1Plan::default(),
            },
        })
    }

    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        self.core.prefill_batched_bf16(prompt_tokens)
    }

    pub fn prefill_token_loop_reference(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        self.core.prefill(prompt_tokens)
    }

    pub fn prefill_top1(&mut self, prompt_tokens: &[u32]) -> Result<(u32, f32)> {
        self.core.prefill_batched_bf16_top1(prompt_tokens)
    }

    pub fn top1_plan(&self) -> Bf16Top1Plan {
        self.core.top1_plan
    }

    pub fn set_top1_plan(&mut self, plan: Bf16Top1Plan) {
        self.core.top1_plan = plan;
    }

    pub fn generate_greedy(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
    ) -> Result<Vec<GreedyGenerationStep>> {
        self.core.generate_greedy_bf16(max_new_tokens, top_k)
    }

    pub fn generate_greedy_until(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        self.core
            .generate_greedy_until_bf16(max_new_tokens, top_k, stop_token_id)
    }

    pub fn generate_greedy_until_from_top1(
        &mut self,
        max_new_tokens: usize,
        stop_token_id: Option<u32>,
        initial_top1: (u32, f32),
    ) -> Result<Vec<GreedyGenerationStep>> {
        self.core.generate_greedy_until_bf16_top1_from_next(
            max_new_tokens,
            stop_token_id,
            initial_top1,
        )
    }

    pub fn tokens(&self) -> &[u32] {
        self.core.tokens()
    }

    pub fn max_seq_len(&self) -> usize {
        self.core.max_seq_len()
    }

    pub fn next_logits_to_host(&self) -> Result<Vec<f32>> {
        self.core.next_logits_to_host()
    }

    pub fn next_top_logits(&mut self, top_k: usize) -> Result<Vec<(u32, f32)>> {
        self.core.next_top_logits(top_k)
    }

    pub fn advance_with_token(&mut self, token_id: u32) -> Result<()> {
        self.core.advance_with_token_bf16(token_id)
    }

    pub fn memory_stats(&self) -> RuntimeMemoryStats {
        self.core.memory_stats()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.core.synchronize()
    }

    pub fn reset_sequence(&mut self) {
        self.core.reset_sequence();
    }
}

impl Drop for MinistralTextRuntime {
    fn drop(&mut self) {
        let _ = self.core.stream.synchronize();
    }
}

impl MinistralAllLinearQuantizedRuntime {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(invalid_data("runtime max_seq_len must be nonzero"));
        }

        let model_dir = model_dir.as_ref();
        let config = TextConfig::from_model_dir(model_dir)?;
        validate_runtime_max_seq_len(&config, max_seq_len)?;
        let weights = ModelWeights::open_model_dir(model_dir)?;
        let token_embedding = read_embedding_weight(&weights, &config)?;
        let rope_freqs = rope_frequencies(&config);
        let norm_weight = read_model_norm_weight(&weights, &config)?;
        let output_weight = read_output_weight(&weights, &config)?;
        let quantized_output_weight = QuantizedI8Matrix::from_bf16_rows_symmetric(
            &output_weight,
            config.vocab_size,
            config.dim,
        );
        let dev_tok_embeddings = DeviceBuffer::from_host(&stream, &token_embedding)?;
        let dev_rope_freqs = DeviceBuffer::from_host(&stream, &rope_freqs)?;
        let dev_norm_weight = DeviceBuffer::from_host(&stream, &norm_weight)?;
        let dev_output_weight = quantized_output_weight.to_device(&stream)?;
        drop((token_embedding, output_weight, quantized_output_weight));
        let layers = load_quantized_attention_ffn_layer_device_weights(&stream, &weights, &config)?;
        let caches = allocate_layer_kv_caches(&stream, &config, max_seq_len)?;
        let scratch = RuntimeScratch::new(&stream, &config, max_seq_len)?;
        stream.synchronize()?;

        Ok(Self {
            core: RuntimeCore {
                stream,
                module,
                config,
                dev_tok_embeddings,
                dev_rope_freqs,
                dev_norm_weight,
                dev_output_weight,
                layers,
                caches,
                scratch,
                max_seq_len,
                tokens: Vec::new(),
                next_logits_ready: false,
                top1_plan: Bf16Top1Plan::default(),
            },
        })
    }

    pub fn new_from_export(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        export_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(invalid_data("runtime max_seq_len must be nonzero"));
        }

        let model_dir = model_dir.as_ref();
        let export_dir = export_dir.as_ref();
        let config = TextConfig::from_model_dir(model_dir)?;
        validate_runtime_max_seq_len(&config, max_seq_len)?;
        let weights = ModelWeights::open_model_dir(model_dir)?;
        validate_all_linear_int8_export_archive(export_dir, &config, Some(weights.source_path()))?;
        let token_embedding = read_embedding_weight(&weights, &config)?;
        let rope_freqs = rope_frequencies(&config);
        let norm_weight = read_model_norm_weight(&weights, &config)?;
        let quantized_output_weight = QuantizedI8Matrix::read_exported(
            export_dir,
            0,
            "output.weight",
            config.vocab_size,
            config.dim,
        )?;
        let dev_tok_embeddings = DeviceBuffer::from_host(&stream, &token_embedding)?;
        let dev_rope_freqs = DeviceBuffer::from_host(&stream, &rope_freqs)?;
        let dev_norm_weight = DeviceBuffer::from_host(&stream, &norm_weight)?;
        let dev_output_weight = quantized_output_weight.to_device(&stream)?;
        drop((token_embedding, norm_weight, quantized_output_weight));
        let layers = load_exported_quantized_attention_ffn_layer_device_weights(
            &stream, &weights, &config, export_dir,
        )?;
        let caches = allocate_layer_kv_caches(&stream, &config, max_seq_len)?;
        let scratch = RuntimeScratch::new(&stream, &config, max_seq_len)?;
        stream.synchronize()?;

        Ok(Self {
            core: RuntimeCore {
                stream,
                module,
                config,
                dev_tok_embeddings,
                dev_rope_freqs,
                dev_norm_weight,
                dev_output_weight,
                layers,
                caches,
                scratch,
                max_seq_len,
                tokens: Vec::new(),
                next_logits_ready: false,
                top1_plan: Bf16Top1Plan::default(),
            },
        })
    }

    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        self.core.prefill_batched_i8(prompt_tokens)
    }

    pub fn prefill_token_loop_reference(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        self.core.prefill(prompt_tokens)
    }

    pub fn generate_greedy(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
    ) -> Result<Vec<GreedyGenerationStep>> {
        self.core
            .generate_greedy_until_i8(max_new_tokens, top_k, None)
    }

    pub fn generate_greedy_until(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        self.core
            .generate_greedy_until_i8(max_new_tokens, top_k, stop_token_id)
    }

    pub fn tokens(&self) -> &[u32] {
        self.core.tokens()
    }

    pub fn max_seq_len(&self) -> usize {
        self.core.max_seq_len()
    }

    pub fn next_logits_to_host(&self) -> Result<Vec<f32>> {
        self.core.next_logits_to_host()
    }

    pub fn next_top_logits(&mut self, top_k: usize) -> Result<Vec<(u32, f32)>> {
        self.core.next_top_logits(top_k)
    }

    pub fn advance_with_token(&mut self, token_id: u32) -> Result<()> {
        self.core.advance_with_token(token_id)
    }

    pub fn memory_stats(&self) -> RuntimeMemoryStats {
        self.core.memory_stats()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.core.synchronize()
    }

    pub fn reset_sequence(&mut self) {
        self.core.reset_sequence();
    }
}

impl Drop for MinistralAllLinearQuantizedRuntime {
    fn drop(&mut self) {
        let _ = self.core.stream.synchronize();
    }
}

pub fn run_ministral_ffn_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    layer: usize,
    token_id: u32,
) -> Result<FfnProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    validate_layer_token(&config, layer, token_id)?;

    let weights = ModelWeights::open_model_dir(model_dir)?;

    let token_row = read_embedding_row(&weights, &config, token_id)?;
    let ffn_norm = read_bf16_shape(
        &weights,
        &format!("layers.{layer}.ffn_norm.weight"),
        &[config.dim],
    )?;
    let w1 = read_bf16_shape(
        &weights,
        &format!("layers.{layer}.feed_forward.w1.weight"),
        &[config.hidden_dim, config.dim],
    )?;
    let w3 = read_bf16_shape(
        &weights,
        &format!("layers.{layer}.feed_forward.w3.weight"),
        &[config.hidden_dim, config.dim],
    )?;
    let w2 = read_bf16_shape(
        &weights,
        &format!("layers.{layer}.feed_forward.w2.weight"),
        &[config.dim, config.hidden_dim],
    )?;

    let dev_token_row = DeviceBuffer::from_host(stream, &token_row)?;
    let dev_ffn_norm = DeviceBuffer::from_host(stream, &ffn_norm)?;
    let dev_w1 = DeviceBuffer::from_host(stream, &w1)?;
    let dev_w3 = DeviceBuffer::from_host(stream, &w3)?;
    let dev_w2 = DeviceBuffer::from_host(stream, &w2)?;
    stream.synchronize()?;
    drop((token_row, ffn_norm, w1, w3, w2));

    let mut hidden = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    ops::embedding(stream, module, &dev_token_row, 0, config.dim, &mut hidden)?;
    let output = run_ffn_from_hidden(
        stream,
        module,
        &config,
        &hidden,
        &dev_ffn_norm,
        &dev_w1,
        &dev_w3,
        &dev_w2,
    )?;

    let output_host = output.to_host_vec(stream)?;
    let l2_norm = output_host.iter().map(|x| x * x).sum::<f32>().sqrt();
    let max_abs = output_host.iter().map(|x| x.abs()).fold(0.0, f32::max);

    Ok(FfnProbe {
        layer,
        token_id,
        output_prefix: output_host.into_iter().take(8).collect(),
        l2_norm,
        max_abs,
    })
}

pub fn run_ministral_block_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    layer: usize,
    token_id: u32,
) -> Result<BlockProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    validate_layer_token(&config, layer, token_id)?;

    let weights = ModelWeights::open_model_dir(model_dir)?;

    let token_row = read_embedding_row(&weights, &config, token_id)?;
    let layer_host = read_layer_host_weights(&weights, &config, layer)?;
    let layer_dev = upload_layer_weights(stream, &layer_host)?;
    stream.synchronize()?;
    let block = run_single_token_block_gpu(stream, module, &config, &token_row, &layer_dev)?;
    drop(layer_host);

    let l2_norm = block.output.iter().map(|x| x * x).sum::<f32>().sqrt();
    let max_abs = block.output.iter().map(|x| x.abs()).fold(0.0, f32::max);

    Ok(BlockProbe {
        layer,
        token_id,
        query_prefix: block.query.into_iter().take(4).collect(),
        key_prefix: block.key.into_iter().take(4).collect(),
        attention_prefix: block.attention_residual.into_iter().take(8).collect(),
        output_prefix: block.output.into_iter().take(8).collect(),
        l2_norm,
        max_abs,
    })
}

pub fn run_ministral_block_reference_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    layer: usize,
    token_id: u32,
) -> Result<BlockReferenceProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    validate_layer_token(&config, layer, token_id)?;

    let weights = ModelWeights::open_model_dir(model_dir)?;
    let token_row = read_embedding_row(&weights, &config, token_id)?;
    let layer_host = read_layer_host_weights(&weights, &config, layer)?;
    let layer_dev = upload_layer_weights(stream, &layer_host)?;
    stream.synchronize()?;

    let gpu = run_single_token_block_gpu(stream, module, &config, &token_row, &layer_dev)?;
    let reference = run_single_token_block_cpu(&config, &token_row, &layer_host);

    Ok(BlockReferenceProbe {
        layer,
        token_id,
        query_max_abs_diff: max_abs_diff(&gpu.query, &reference.query),
        key_max_abs_diff: max_abs_diff(&gpu.key, &reference.key),
        attention_max_abs_diff: max_abs_diff(
            &gpu.attention_residual,
            &reference.attention_residual,
        ),
        output_max_abs_diff: max_abs_diff(&gpu.output, &reference.output),
        output_mean_abs_diff: mean_abs_diff(&gpu.output, &reference.output),
        gpu_output_prefix: gpu.output.into_iter().take(8).collect(),
        reference_output_prefix: reference.output.into_iter().take(8).collect(),
    })
}

pub fn run_ministral_prompt_block_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    layer: usize,
    tokens: &[u32],
) -> Result<PromptBlockProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    if tokens.is_empty() {
        return Err(invalid_data(
            "prompt block probe requires at least one token",
        ));
    }
    for &token in tokens {
        validate_layer_token(&config, layer, token)?;
    }

    let weights = ModelWeights::open_model_dir(model_dir)?;
    let token_rows = tokens
        .iter()
        .map(|&token| read_embedding_row(&weights, &config, token))
        .collect::<Result<Vec<_>>>()?;
    let layer_host = read_layer_host_weights(&weights, &config, layer)?;
    let rope_freqs = rope_frequencies(&config);
    let layer_dev = upload_layer_weights(stream, &layer_host)?;
    let dev_rope_freqs = DeviceBuffer::from_host(stream, &rope_freqs)?;
    stream.synchronize()?;
    let block = run_prompt_block_gpu(
        stream,
        module,
        &config,
        &token_rows,
        &layer_dev,
        &dev_rope_freqs,
    )?;
    let l2_norm = block.output.iter().map(|x| x * x).sum::<f32>().sqrt();
    let max_abs = block.output.iter().map(|x| x.abs()).fold(0.0, f32::max);

    Ok(PromptBlockProbe {
        layer,
        tokens: tokens.to_vec(),
        query_prefix: block.query.into_iter().take(4).collect(),
        key_prefix: block.key.into_iter().take(4).collect(),
        score_prefix: block.scores.into_iter().take(tokens.len().min(8)).collect(),
        attention_prefix: block.attention_residual.into_iter().take(8).collect(),
        output_prefix: block.output.into_iter().take(8).collect(),
        l2_norm,
        max_abs,
    })
}

pub fn run_ministral_prompt_block_reference_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    layer: usize,
    tokens: &[u32],
) -> Result<PromptBlockReferenceProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    if tokens.is_empty() {
        return Err(invalid_data(
            "prompt block reference probe requires at least one token",
        ));
    }
    for &token in tokens {
        validate_layer_token(&config, layer, token)?;
    }

    let weights = ModelWeights::open_model_dir(model_dir)?;
    let token_rows = tokens
        .iter()
        .map(|&token| read_embedding_row(&weights, &config, token))
        .collect::<Result<Vec<_>>>()?;
    let layer_host = read_layer_host_weights(&weights, &config, layer)?;
    let rope_freqs = rope_frequencies(&config);
    let layer_dev = upload_layer_weights(stream, &layer_host)?;
    let dev_rope_freqs = DeviceBuffer::from_host(stream, &rope_freqs)?;
    stream.synchronize()?;

    let gpu = run_prompt_block_gpu(
        stream,
        module,
        &config,
        &token_rows,
        &layer_dev,
        &dev_rope_freqs,
    )?;
    let reference = run_prompt_block_cpu(&config, &token_rows, &layer_host, &rope_freqs);

    Ok(PromptBlockReferenceProbe {
        layer,
        tokens: tokens.to_vec(),
        query_max_abs_diff: max_abs_diff(&gpu.query, &reference.query),
        key_max_abs_diff: max_abs_diff(&gpu.key, &reference.key),
        score_max_abs_diff: max_abs_diff(&gpu.scores, &reference.scores),
        attention_max_abs_diff: max_abs_diff(
            &gpu.attention_residual,
            &reference.attention_residual,
        ),
        output_max_abs_diff: max_abs_diff(&gpu.output, &reference.output),
        output_mean_abs_diff: mean_abs_diff(&gpu.output, &reference.output),
        gpu_output_prefix: gpu.output.into_iter().take(8).collect(),
        reference_output_prefix: reference.output.into_iter().take(8).collect(),
    })
}

pub fn run_ministral_text_forward_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    tokens: &[u32],
    top_k: usize,
) -> Result<TextForwardProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    run_ministral_text_forward_with_weights(stream, module, &weights, &config, tokens, top_k)
}

pub fn run_ministral_output_projection_quantization_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    tokens: &[u32],
    top_k: usize,
) -> Result<OutputProjectionQuantizationProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    run_ministral_output_projection_quantization_with_weights(
        stream, module, &weights, &config, tokens, top_k,
    )
}

pub fn run_ministral_output_projection_export_quantization_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    tokens: &[u32],
    top_k: usize,
) -> Result<OutputProjectionQuantizationProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    run_ministral_output_projection_export_quantization_with_weights(
        stream,
        module,
        &weights,
        &config,
        export_dir.as_ref(),
        tokens,
        top_k,
    )
}

pub fn run_ministral_ffn_quantization_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    tokens: &[u32],
    top_k: usize,
) -> Result<FfnQuantizationProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    run_ministral_ffn_quantization_with_weights(stream, module, &weights, &config, tokens, top_k)
}

pub fn run_ministral_attention_quantization_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    tokens: &[u32],
    top_k: usize,
) -> Result<AttentionQuantizationProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    run_ministral_attention_quantization_with_weights(
        stream, module, &weights, &config, tokens, top_k,
    )
}

pub fn run_ministral_attention_ffn_quantization_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    tokens: &[u32],
    top_k: usize,
) -> Result<AttentionFfnQuantizationProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    run_ministral_attention_ffn_quantization_with_weights(
        stream, module, &weights, &config, tokens, top_k,
    )
}

pub fn run_ministral_all_linear_quantization_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    tokens: &[u32],
    top_k: usize,
) -> Result<AllLinearQuantizationProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    run_ministral_all_linear_quantization_with_weights(
        stream, module, &weights, &config, tokens, top_k,
    )
}

pub fn run_ministral_quantization_suite_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    top_k: usize,
) -> Result<QuantizationSuiteProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    run_ministral_quantization_suite_with_weights(stream, module, &weights, &config, prompts, top_k)
}

pub fn run_ministral_quantization_generation_suite_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<QuantizationGenerationSuiteProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    run_ministral_quantization_generation_suite_with_weights(
        stream,
        module,
        &weights,
        &config,
        prompts,
        max_new_tokens,
        top_k,
        stop_token_id,
    )
}

pub fn validate_ministral_all_linear_int8_export_archive(
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
) -> Result<()> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    validate_all_linear_int8_export_archive(
        export_dir.as_ref(),
        &config,
        Some(weights.source_path()),
    )
}

pub fn run_ministral_all_linear_free_generation_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<AllLinearFreeGenerationProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    let weights = ModelWeights::open_model_dir(model_dir)?;
    run_ministral_all_linear_free_generation_with_weights(
        stream,
        module,
        &weights,
        &config,
        prompt_tokens,
        max_new_tokens,
        top_k,
        stop_token_id,
    )
}

pub fn run_ministral_greedy_generation_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
) -> Result<GreedyGenerationProbe> {
    let model_dir = model_dir.as_ref();
    let config = TextConfig::from_model_dir(model_dir)?;
    if prompt_tokens.is_empty() {
        return Err(invalid_data(
            "generation requires at least one prompt token",
        ));
    }
    for &token in prompt_tokens {
        validate_token(&config, token)?;
    }

    let weights = ModelWeights::open_model_dir(model_dir)?;
    let mut all_tokens = prompt_tokens.to_vec();
    let mut steps = Vec::with_capacity(max_new_tokens);
    let reported_top_k = top_k.max(1);

    for step in 0..max_new_tokens {
        let forward = run_ministral_text_forward_with_weights(
            stream,
            module,
            &weights,
            &config,
            &all_tokens,
            reported_top_k,
        )?;
        let &(token_id, logit) = forward
            .top_logits
            .first()
            .ok_or_else(|| invalid_data("text forward produced no top logits"))?;
        all_tokens.push(token_id);
        steps.push(GreedyGenerationStep {
            step,
            token_id,
            logit,
            top_logits: forward.top_logits,
        });
    }

    Ok(GreedyGenerationProbe {
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens: all_tokens[prompt_tokens.len()..].to_vec(),
        all_tokens,
        steps,
    })
}

pub fn run_ministral_incremental_greedy_generation_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
) -> Result<GreedyGenerationProbe> {
    run_ministral_incremental_greedy_generation_until(
        stream,
        module,
        model_dir,
        prompt_tokens,
        max_new_tokens,
        top_k,
        None,
    )
}

pub fn run_ministral_generation_comparison_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
) -> Result<GreedyGenerationComparisonProbe> {
    let model_dir = model_dir.as_ref();
    let full = run_ministral_greedy_generation_probe(
        stream,
        module,
        model_dir,
        prompt_tokens,
        max_new_tokens,
        top_k,
    )?;
    let incremental = run_ministral_incremental_greedy_generation_probe(
        stream,
        module,
        model_dir,
        prompt_tokens,
        max_new_tokens,
        top_k,
    )?;

    let mut steps = Vec::with_capacity(full.steps.len().min(incremental.steps.len()));
    let mut max_winning_logit_abs_diff = 0.0;
    let mut max_top_logits_abs_diff = 0.0;
    for (full_step, incremental_step) in full.steps.iter().zip(incremental.steps.iter()) {
        let winning_logit_abs_diff = (full_step.logit - incremental_step.logit).abs();
        let top_logits_max_abs_diff =
            paired_top_logits_max_abs_diff(&full_step.top_logits, &incremental_step.top_logits);
        max_winning_logit_abs_diff = f32::max(max_winning_logit_abs_diff, winning_logit_abs_diff);
        max_top_logits_abs_diff = f32::max(max_top_logits_abs_diff, top_logits_max_abs_diff);

        steps.push(GreedyGenerationComparisonStep {
            step: full_step.step,
            full_token_id: full_step.token_id,
            incremental_token_id: incremental_step.token_id,
            token_matches: full_step.token_id == incremental_step.token_id,
            winning_logit_abs_diff,
            top_tokens_match: top_logit_tokens_match(
                &full_step.top_logits,
                &incremental_step.top_logits,
            ),
            top_logits_max_abs_diff,
        });
    }

    Ok(GreedyGenerationComparisonProbe {
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens_match: full.generated_tokens == incremental.generated_tokens,
        full,
        incremental,
        max_winning_logit_abs_diff,
        max_top_logits_abs_diff,
        steps,
    })
}

pub fn run_ministral_incremental_greedy_generation_until(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<GreedyGenerationProbe> {
    if prompt_tokens.is_empty() {
        return Err(invalid_data(
            "generation requires at least one prompt token",
        ));
    }

    let max_seq_len = prompt_tokens.len() + max_new_tokens;
    let mut runtime = MinistralTextRuntime::new(
        Arc::clone(stream),
        Arc::clone(module),
        model_dir,
        max_seq_len,
    )?;
    runtime.prefill(prompt_tokens)?;
    let steps = runtime.generate_greedy_until(max_new_tokens, top_k, stop_token_id)?;
    let all_tokens = runtime.tokens().to_vec();

    Ok(GreedyGenerationProbe {
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens: all_tokens[prompt_tokens.len()..].to_vec(),
        all_tokens,
        steps,
    })
}

pub fn run_ministral_all_linear_quantized_runtime_generation_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<GreedyGenerationProbe> {
    if prompt_tokens.is_empty() {
        return Err(invalid_data(
            "all-linear quantized runtime generation requires at least one prompt token",
        ));
    }

    let max_seq_len = prompt_tokens.len() + max_new_tokens;
    let mut runtime = MinistralAllLinearQuantizedRuntime::new(
        Arc::clone(stream),
        Arc::clone(module),
        model_dir,
        max_seq_len,
    )?;
    runtime.prefill(prompt_tokens)?;
    let steps = runtime.generate_greedy_until(max_new_tokens, top_k, stop_token_id)?;
    let all_tokens = runtime.tokens().to_vec();

    Ok(GreedyGenerationProbe {
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens: all_tokens[prompt_tokens.len()..].to_vec(),
        all_tokens,
        steps,
    })
}

pub fn run_ministral_all_linear_quantized_runtime_comparison_probe(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<AllLinearQuantizedRuntimeComparisonProbe> {
    if prompt_tokens.is_empty() {
        return Err(invalid_data(
            "all-linear quantized runtime comparison requires at least one prompt token",
        ));
    }

    let max_seq_len = prompt_tokens.len() + max_new_tokens;
    let model_dir = model_dir.as_ref();
    let mut reference_runtime = MinistralTextRuntime::new(
        Arc::clone(stream),
        Arc::clone(module),
        model_dir,
        max_seq_len,
    )?;
    let mut quantized_runtime = MinistralAllLinearQuantizedRuntime::new(
        Arc::clone(stream),
        Arc::clone(module),
        model_dir,
        max_seq_len,
    )?;
    reference_runtime.prefill(prompt_tokens)?;
    quantized_runtime.prefill(prompt_tokens)?;
    let reference_memory_stats = reference_runtime.memory_stats();
    let quantized_memory_stats = quantized_runtime.memory_stats();
    let top_k = top_k.max(1);
    let mut generated_tokens = Vec::new();
    let mut steps = Vec::with_capacity(max_new_tokens);

    for step in 0..max_new_tokens {
        let prefix_len = quantized_runtime.tokens().len();
        let reference_logits = reference_runtime.next_logits_to_host()?;
        let quantized_logits = quantized_runtime.next_logits_to_host()?;
        let kl_divergence = kl_divergence_from_logits(&reference_logits, &quantized_logits)?;
        let max_abs_diff = max_abs_diff(&reference_logits, &quantized_logits);
        let mean_abs_diff = mean_abs_diff(&reference_logits, &quantized_logits);
        let reference_top_logits = top_k_logits(&reference_logits, top_k);
        let quantized_top_logits = top_k_logits(&quantized_logits, top_k);
        let &(reference_token_id, reference_logit) = reference_top_logits
            .first()
            .ok_or_else(|| invalid_data("reference runtime produced no top logits"))?;
        let &(quantized_token_id, quantized_logit) = quantized_top_logits
            .first()
            .ok_or_else(|| invalid_data("quantized runtime produced no top logits"))?;
        let stop = Some(quantized_token_id) == stop_token_id;

        steps.push(AllLinearQuantizedRuntimeComparisonStepProbe {
            step,
            prefix_len,
            reference_token_id,
            reference_logit,
            quantized_token_id,
            quantized_logit,
            token_matches: reference_token_id == quantized_token_id,
            kl_divergence,
            max_abs_diff,
            mean_abs_diff,
            reference_top_logits,
            quantized_top_logits,
        });
        reference_runtime.advance_with_token(quantized_token_id)?;
        quantized_runtime.advance_with_token(quantized_token_id)?;
        generated_tokens.push(quantized_token_id);

        if stop {
            break;
        }
    }

    Ok(AllLinearQuantizedRuntimeComparisonProbe {
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens,
        all_tokens: quantized_runtime.tokens().to_vec(),
        reference_memory_stats,
        quantized_memory_stats,
        steps,
    })
}

fn run_ministral_text_forward_with_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
    top_k: usize,
) -> Result<TextForwardProbe> {
    if tokens.is_empty() {
        return Err(invalid_data(
            "text forward probe requires at least one token",
        ));
    }
    for &token in tokens {
        validate_token(config, token)?;
    }

    let final_hidden = run_prompt_to_final_hidden(stream, module, weights, config, tokens)?;
    let norm_weight = read_model_norm_weight(weights, config)?;
    let output_weight = read_output_weight(weights, config)?;
    let dev_final_hidden = DeviceBuffer::from_host(stream, &final_hidden)?;
    let dev_norm_weight = DeviceBuffer::from_host(stream, &norm_weight)?;
    let dev_output_weight = DeviceBuffer::from_host(stream, &output_weight)?;
    stream.synchronize()?;
    drop((norm_weight, output_weight));

    let mut normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;
    ops::rmsnorm(
        stream,
        module,
        &dev_final_hidden,
        &dev_norm_weight,
        config.norm_eps,
        &mut normed,
    )?;
    ops::linear(stream, module, &normed, &dev_output_weight, &mut logits)?;
    let logits_host = logits.to_host_vec(stream)?;

    let final_hidden_l2_norm = final_hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    let logits_max_abs = logits_host.iter().map(|x| x.abs()).fold(0.0, f32::max);

    Ok(TextForwardProbe {
        tokens: tokens.to_vec(),
        layers_run: config.n_layers,
        final_hidden_prefix: final_hidden.iter().copied().take(8).collect(),
        logits_prefix: logits_host.iter().copied().take(8).collect(),
        top_logits: top_k_logits(&logits_host, top_k),
        final_hidden_l2_norm,
        logits_max_abs,
    })
}

fn run_ministral_output_projection_quantization_with_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
    top_k: usize,
) -> Result<OutputProjectionQuantizationProbe> {
    if tokens.is_empty() {
        return Err(invalid_data(
            "output projection quantization probe requires at least one token",
        ));
    }
    for &token in tokens {
        validate_token(config, token)?;
    }

    let final_hidden = run_prompt_to_final_hidden(stream, module, weights, config, tokens)?;
    let norm_weight = read_model_norm_weight(weights, config)?;
    let output_weight = read_output_weight(weights, config)?;
    let quantized_output_weight =
        QuantizedI8Matrix::from_bf16_rows_symmetric(&output_weight, config.vocab_size, config.dim);
    let dev_final_hidden = DeviceBuffer::from_host(stream, &final_hidden)?;
    let dev_norm_weight = DeviceBuffer::from_host(stream, &norm_weight)?;
    let dev_output_weight = DeviceBuffer::from_host(stream, &output_weight)?;
    let dev_quantized_output_weight = quantized_output_weight.to_device(stream)?;
    stream.synchronize()?;
    drop((norm_weight, output_weight, quantized_output_weight));

    let mut normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut reference_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;
    let mut quantized_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;
    ops::rmsnorm(
        stream,
        module,
        &dev_final_hidden,
        &dev_norm_weight,
        config.norm_eps,
        &mut normed,
    )?;
    ops::linear(
        stream,
        module,
        &normed,
        &dev_output_weight,
        &mut reference_logits,
    )?;
    ops::linear(
        stream,
        module,
        &normed,
        &dev_quantized_output_weight,
        &mut quantized_logits,
    )?;

    let reference_logits_host = reference_logits.to_host_vec(stream)?;
    let quantized_logits_host = quantized_logits.to_host_vec(stream)?;
    let kl_divergence = kl_divergence_from_logits(&reference_logits_host, &quantized_logits_host)?;
    let max_abs_diff = max_abs_diff(&reference_logits_host, &quantized_logits_host);
    let mean_abs_diff = mean_abs_diff(&reference_logits_host, &quantized_logits_host);

    Ok(OutputProjectionQuantizationProbe {
        tokens: tokens.to_vec(),
        rows: config.vocab_size,
        cols: config.dim,
        reference_top_logits: top_k_logits(&reference_logits_host, top_k),
        quantized_top_logits: top_k_logits(&quantized_logits_host, top_k),
        kl_divergence,
        max_abs_diff,
        mean_abs_diff,
        reference_logits_prefix: reference_logits_host.iter().copied().take(8).collect(),
        quantized_logits_prefix: quantized_logits_host.iter().copied().take(8).collect(),
    })
}

fn run_ministral_output_projection_export_quantization_with_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    export_dir: &Path,
    tokens: &[u32],
    top_k: usize,
) -> Result<OutputProjectionQuantizationProbe> {
    if tokens.is_empty() {
        return Err(invalid_data(
            "exported output projection quantization probe requires at least one token",
        ));
    }
    for &token in tokens {
        validate_token(config, token)?;
    }

    let final_hidden = run_prompt_to_final_hidden(stream, module, weights, config, tokens)?;
    let norm_weight = read_model_norm_weight(weights, config)?;
    let output_weight = read_output_weight(weights, config)?;
    let quantized_output_weight = QuantizedI8Matrix::read_exported(
        export_dir,
        0,
        "output.weight",
        config.vocab_size,
        config.dim,
    )?;
    let dev_final_hidden = DeviceBuffer::from_host(stream, &final_hidden)?;
    let dev_norm_weight = DeviceBuffer::from_host(stream, &norm_weight)?;
    let dev_output_weight = DeviceBuffer::from_host(stream, &output_weight)?;
    let dev_quantized_output_weight = quantized_output_weight.to_device(stream)?;
    stream.synchronize()?;
    drop((norm_weight, output_weight, quantized_output_weight));

    let mut normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut reference_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;
    let mut quantized_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;
    ops::rmsnorm(
        stream,
        module,
        &dev_final_hidden,
        &dev_norm_weight,
        config.norm_eps,
        &mut normed,
    )?;
    ops::linear(
        stream,
        module,
        &normed,
        &dev_output_weight,
        &mut reference_logits,
    )?;
    ops::linear(
        stream,
        module,
        &normed,
        &dev_quantized_output_weight,
        &mut quantized_logits,
    )?;

    let reference_logits_host = reference_logits.to_host_vec(stream)?;
    let quantized_logits_host = quantized_logits.to_host_vec(stream)?;
    let kl_divergence = kl_divergence_from_logits(&reference_logits_host, &quantized_logits_host)?;
    let max_abs_diff = max_abs_diff(&reference_logits_host, &quantized_logits_host);
    let mean_abs_diff = mean_abs_diff(&reference_logits_host, &quantized_logits_host);

    Ok(OutputProjectionQuantizationProbe {
        tokens: tokens.to_vec(),
        rows: config.vocab_size,
        cols: config.dim,
        reference_top_logits: top_k_logits(&reference_logits_host, top_k),
        quantized_top_logits: top_k_logits(&quantized_logits_host, top_k),
        kl_divergence,
        max_abs_diff,
        mean_abs_diff,
        reference_logits_prefix: reference_logits_host.iter().copied().take(8).collect(),
        quantized_logits_prefix: quantized_logits_host.iter().copied().take(8).collect(),
    })
}

fn run_ministral_ffn_quantization_with_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
    top_k: usize,
) -> Result<FfnQuantizationProbe> {
    if tokens.is_empty() {
        return Err(invalid_data(
            "FFN quantization probe requires at least one token",
        ));
    }
    for &token in tokens {
        validate_token(config, token)?;
    }

    let reference_hidden = run_prompt_to_final_hidden(stream, module, weights, config, tokens)?;
    let quantized_hidden =
        run_prompt_to_final_hidden_quantized_ffn(stream, module, weights, config, tokens)?;
    let norm_weight = read_model_norm_weight(weights, config)?;
    let output_weight = read_output_weight(weights, config)?;
    let dev_norm_weight = DeviceBuffer::from_host(stream, &norm_weight)?;
    let dev_output_weight = DeviceBuffer::from_host(stream, &output_weight)?;
    stream.synchronize()?;
    drop((norm_weight, output_weight));

    let dev_reference_hidden = DeviceBuffer::from_host(stream, &reference_hidden)?;
    let dev_quantized_hidden = DeviceBuffer::from_host(stream, &quantized_hidden)?;
    let mut reference_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut quantized_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut reference_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;
    let mut quantized_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;

    logits_from_hidden_device_into(
        stream,
        module,
        config,
        &dev_reference_hidden,
        &dev_norm_weight,
        &dev_output_weight,
        &mut reference_normed,
        &mut reference_logits,
    )?;
    logits_from_hidden_device_into(
        stream,
        module,
        config,
        &dev_quantized_hidden,
        &dev_norm_weight,
        &dev_output_weight,
        &mut quantized_normed,
        &mut quantized_logits,
    )?;

    let reference_logits_host = reference_logits.to_host_vec(stream)?;
    let quantized_logits_host = quantized_logits.to_host_vec(stream)?;
    let kl_divergence = kl_divergence_from_logits(&reference_logits_host, &quantized_logits_host)?;
    let max_abs_diff = max_abs_diff(&reference_logits_host, &quantized_logits_host);
    let mean_abs_diff = mean_abs_diff(&reference_logits_host, &quantized_logits_host);

    Ok(FfnQuantizationProbe {
        tokens: tokens.to_vec(),
        layers_run: config.n_layers,
        reference_top_logits: top_k_logits(&reference_logits_host, top_k),
        quantized_top_logits: top_k_logits(&quantized_logits_host, top_k),
        kl_divergence,
        max_abs_diff,
        mean_abs_diff,
        reference_logits_prefix: reference_logits_host.iter().copied().take(8).collect(),
        quantized_logits_prefix: quantized_logits_host.iter().copied().take(8).collect(),
    })
}

fn run_ministral_attention_quantization_with_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
    top_k: usize,
) -> Result<AttentionQuantizationProbe> {
    if tokens.is_empty() {
        return Err(invalid_data(
            "attention quantization probe requires at least one token",
        ));
    }
    for &token in tokens {
        validate_token(config, token)?;
    }

    let reference_hidden = run_prompt_to_final_hidden(stream, module, weights, config, tokens)?;
    let quantized_hidden =
        run_prompt_to_final_hidden_quantized_attention(stream, module, weights, config, tokens)?;
    let norm_weight = read_model_norm_weight(weights, config)?;
    let output_weight = read_output_weight(weights, config)?;
    let dev_norm_weight = DeviceBuffer::from_host(stream, &norm_weight)?;
    let dev_output_weight = DeviceBuffer::from_host(stream, &output_weight)?;
    stream.synchronize()?;
    drop((norm_weight, output_weight));

    let dev_reference_hidden = DeviceBuffer::from_host(stream, &reference_hidden)?;
    let dev_quantized_hidden = DeviceBuffer::from_host(stream, &quantized_hidden)?;
    let mut reference_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut quantized_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut reference_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;
    let mut quantized_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;

    logits_from_hidden_device_into(
        stream,
        module,
        config,
        &dev_reference_hidden,
        &dev_norm_weight,
        &dev_output_weight,
        &mut reference_normed,
        &mut reference_logits,
    )?;
    logits_from_hidden_device_into(
        stream,
        module,
        config,
        &dev_quantized_hidden,
        &dev_norm_weight,
        &dev_output_weight,
        &mut quantized_normed,
        &mut quantized_logits,
    )?;

    let reference_logits_host = reference_logits.to_host_vec(stream)?;
    let quantized_logits_host = quantized_logits.to_host_vec(stream)?;
    let kl_divergence = kl_divergence_from_logits(&reference_logits_host, &quantized_logits_host)?;
    let max_abs_diff = max_abs_diff(&reference_logits_host, &quantized_logits_host);
    let mean_abs_diff = mean_abs_diff(&reference_logits_host, &quantized_logits_host);

    Ok(AttentionQuantizationProbe {
        tokens: tokens.to_vec(),
        layers_run: config.n_layers,
        reference_top_logits: top_k_logits(&reference_logits_host, top_k),
        quantized_top_logits: top_k_logits(&quantized_logits_host, top_k),
        kl_divergence,
        max_abs_diff,
        mean_abs_diff,
        reference_logits_prefix: reference_logits_host.iter().copied().take(8).collect(),
        quantized_logits_prefix: quantized_logits_host.iter().copied().take(8).collect(),
    })
}

fn run_ministral_attention_ffn_quantization_with_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
    top_k: usize,
) -> Result<AttentionFfnQuantizationProbe> {
    if tokens.is_empty() {
        return Err(invalid_data(
            "attention+FFN quantization probe requires at least one token",
        ));
    }
    for &token in tokens {
        validate_token(config, token)?;
    }

    let reference_hidden = run_prompt_to_final_hidden(stream, module, weights, config, tokens)?;
    let quantized_hidden = run_prompt_to_final_hidden_quantized_attention_ffn(
        stream, module, weights, config, tokens,
    )?;
    let norm_weight = read_model_norm_weight(weights, config)?;
    let output_weight = read_output_weight(weights, config)?;
    let dev_norm_weight = DeviceBuffer::from_host(stream, &norm_weight)?;
    let dev_output_weight = DeviceBuffer::from_host(stream, &output_weight)?;
    stream.synchronize()?;
    drop((norm_weight, output_weight));

    let dev_reference_hidden = DeviceBuffer::from_host(stream, &reference_hidden)?;
    let dev_quantized_hidden = DeviceBuffer::from_host(stream, &quantized_hidden)?;
    let mut reference_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut quantized_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut reference_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;
    let mut quantized_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;

    logits_from_hidden_device_into(
        stream,
        module,
        config,
        &dev_reference_hidden,
        &dev_norm_weight,
        &dev_output_weight,
        &mut reference_normed,
        &mut reference_logits,
    )?;
    logits_from_hidden_device_into(
        stream,
        module,
        config,
        &dev_quantized_hidden,
        &dev_norm_weight,
        &dev_output_weight,
        &mut quantized_normed,
        &mut quantized_logits,
    )?;

    let reference_logits_host = reference_logits.to_host_vec(stream)?;
    let quantized_logits_host = quantized_logits.to_host_vec(stream)?;
    let kl_divergence = kl_divergence_from_logits(&reference_logits_host, &quantized_logits_host)?;
    let max_abs_diff = max_abs_diff(&reference_logits_host, &quantized_logits_host);
    let mean_abs_diff = mean_abs_diff(&reference_logits_host, &quantized_logits_host);

    Ok(AttentionFfnQuantizationProbe {
        tokens: tokens.to_vec(),
        layers_run: config.n_layers,
        reference_top_logits: top_k_logits(&reference_logits_host, top_k),
        quantized_top_logits: top_k_logits(&quantized_logits_host, top_k),
        kl_divergence,
        max_abs_diff,
        mean_abs_diff,
        reference_logits_prefix: reference_logits_host.iter().copied().take(8).collect(),
        quantized_logits_prefix: quantized_logits_host.iter().copied().take(8).collect(),
    })
}

fn run_ministral_all_linear_quantization_with_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
    top_k: usize,
) -> Result<AllLinearQuantizationProbe> {
    if tokens.is_empty() {
        return Err(invalid_data(
            "all-linear quantization probe requires at least one token",
        ));
    }
    for &token in tokens {
        validate_token(config, token)?;
    }

    let reference_hidden = run_prompt_to_final_hidden(stream, module, weights, config, tokens)?;
    let quantized_hidden = run_prompt_to_final_hidden_quantized_attention_ffn(
        stream, module, weights, config, tokens,
    )?;
    let norm_weight = read_model_norm_weight(weights, config)?;
    let output_weight = read_output_weight(weights, config)?;
    let quantized_output_weight =
        QuantizedI8Matrix::from_bf16_rows_symmetric(&output_weight, config.vocab_size, config.dim);
    let dev_norm_weight = DeviceBuffer::from_host(stream, &norm_weight)?;
    let dev_output_weight = DeviceBuffer::from_host(stream, &output_weight)?;
    let dev_quantized_output_weight = quantized_output_weight.to_device(stream)?;
    stream.synchronize()?;
    drop((norm_weight, output_weight, quantized_output_weight));

    let dev_reference_hidden = DeviceBuffer::from_host(stream, &reference_hidden)?;
    let dev_quantized_hidden = DeviceBuffer::from_host(stream, &quantized_hidden)?;
    let mut reference_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut quantized_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut reference_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;
    let mut quantized_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;

    logits_from_hidden_device_into(
        stream,
        module,
        config,
        &dev_reference_hidden,
        &dev_norm_weight,
        &dev_output_weight,
        &mut reference_normed,
        &mut reference_logits,
    )?;
    ops::rmsnorm(
        stream,
        module,
        &dev_quantized_hidden,
        &dev_norm_weight,
        config.norm_eps,
        &mut quantized_normed,
    )?;
    ops::linear(
        stream,
        module,
        &quantized_normed,
        &dev_quantized_output_weight,
        &mut quantized_logits,
    )?;

    let reference_logits_host = reference_logits.to_host_vec(stream)?;
    let quantized_logits_host = quantized_logits.to_host_vec(stream)?;
    let kl_divergence = kl_divergence_from_logits(&reference_logits_host, &quantized_logits_host)?;
    let max_abs_diff = max_abs_diff(&reference_logits_host, &quantized_logits_host);
    let mean_abs_diff = mean_abs_diff(&reference_logits_host, &quantized_logits_host);

    Ok(AllLinearQuantizationProbe {
        tokens: tokens.to_vec(),
        layers_run: config.n_layers,
        output_rows: config.vocab_size,
        output_cols: config.dim,
        reference_top_logits: top_k_logits(&reference_logits_host, top_k),
        quantized_top_logits: top_k_logits(&quantized_logits_host, top_k),
        kl_divergence,
        max_abs_diff,
        mean_abs_diff,
        reference_logits_prefix: reference_logits_host.iter().copied().take(8).collect(),
        quantized_logits_prefix: quantized_logits_host.iter().copied().take(8).collect(),
    })
}

fn run_ministral_quantization_suite_with_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    prompts: &[Vec<u32>],
    top_k: usize,
) -> Result<QuantizationSuiteProbe> {
    if prompts.is_empty() {
        return Err(invalid_data(
            "quantization suite requires at least one prompt",
        ));
    }

    let output_weights = load_quantization_output_weights(stream, weights, config)?;
    let mut prompt_results = Vec::with_capacity(prompts.len());
    for tokens in prompts {
        prompt_results.push(run_quantization_suite_prompt_with_outputs(
            stream,
            module,
            weights,
            config,
            tokens,
            &output_weights,
            top_k,
        )?);
    }

    Ok(QuantizationSuiteProbe {
        prompts: prompt_results,
    })
}

fn run_ministral_quantization_generation_suite_with_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    prompts: &[Vec<u32>],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<QuantizationGenerationSuiteProbe> {
    if prompts.is_empty() {
        return Err(invalid_data(
            "quantization generation suite requires at least one prompt",
        ));
    }

    let output_weights = load_quantization_output_weights(stream, weights, config)?;
    let mut prompt_results = Vec::with_capacity(prompts.len());
    let mut total_steps = 0;
    for tokens in prompts {
        if tokens.is_empty() {
            return Err(invalid_data(
                "quantization generation suite prompt cannot be empty",
            ));
        }
        let initial_tokens = tokens.clone();
        let mut current_tokens = tokens.clone();
        let mut generated_tokens = Vec::new();
        let mut steps = Vec::new();

        for step in 0..max_new_tokens {
            let prefix_len = current_tokens.len();
            let prompt_result = run_quantization_suite_prompt_with_outputs(
                stream,
                module,
                weights,
                config,
                &current_tokens,
                &output_weights,
                top_k,
            )?;
            let first_variant = prompt_result
                .variants
                .first()
                .ok_or_else(|| invalid_data("quantization suite returned no variants"))?;
            let &(reference_token_id, reference_logit) =
                first_variant.reference_top_logits.first().ok_or_else(|| {
                    invalid_data("quantization suite reference top logits were empty")
                })?;

            current_tokens.push(reference_token_id);
            generated_tokens.push(reference_token_id);
            steps.push(QuantizationGenerationSuiteStepProbe {
                step,
                prefix_len,
                reference_token_id,
                reference_logit,
                variants: prompt_result.variants,
            });
            total_steps += 1;

            if Some(reference_token_id) == stop_token_id {
                break;
            }
        }

        prompt_results.push(QuantizationGenerationSuitePromptProbe {
            initial_tokens,
            generated_tokens,
            steps,
        });
    }

    Ok(QuantizationGenerationSuiteProbe {
        prompts: prompt_results,
        total_steps,
    })
}

fn run_ministral_all_linear_free_generation_with_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<AllLinearFreeGenerationProbe> {
    if prompt_tokens.is_empty() {
        return Err(invalid_data(
            "all-linear free generation requires at least one prompt token",
        ));
    }

    let output_weights = load_quantization_output_weights(stream, weights, config)?;
    let top_k = top_k.max(1);
    let mut current_tokens = prompt_tokens.to_vec();
    let mut generated_tokens = Vec::new();
    let mut steps = Vec::new();

    for step in 0..max_new_tokens {
        let prefix_len = current_tokens.len();
        let variant = run_all_linear_quantization_variant_with_outputs(
            stream,
            module,
            weights,
            config,
            &current_tokens,
            &output_weights,
            top_k,
        )?;

        let quantized_token_id = variant.quantized_token_id;
        let stop = Some(quantized_token_id) == stop_token_id;
        current_tokens.push(quantized_token_id);
        generated_tokens.push(quantized_token_id);
        steps.push(AllLinearFreeGenerationStepProbe {
            step,
            prefix_len,
            reference_token_id: variant.reference_token_id,
            reference_logit: variant.reference_logit,
            quantized_token_id,
            quantized_logit: variant.quantized_logit,
            token_matches: variant.token_matches,
            kl_divergence: variant.kl_divergence,
            max_abs_diff: variant.max_abs_diff,
            mean_abs_diff: variant.mean_abs_diff,
            reference_top_logits: variant.reference_top_logits,
            quantized_top_logits: variant.quantized_top_logits,
        });

        if stop {
            break;
        }
    }

    Ok(AllLinearFreeGenerationProbe {
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens,
        all_tokens: current_tokens,
        steps,
    })
}

struct QuantizationOutputWeights {
    norm: DeviceBuffer<Bf16>,
    output: DeviceBuffer<Bf16>,
    quantized_output: DeviceQuantizedI8Matrix,
}

fn load_quantization_output_weights(
    stream: &Arc<CudaStream>,
    weights: &ModelWeights,
    config: &TextConfig,
) -> Result<QuantizationOutputWeights> {
    let norm_weight = read_model_norm_weight(weights, config)?;
    let output_weight = read_output_weight(weights, config)?;
    let quantized_output_weight =
        QuantizedI8Matrix::from_bf16_rows_symmetric(&output_weight, config.vocab_size, config.dim);
    let dev_norm_weight = DeviceBuffer::from_host(stream, &norm_weight)?;
    let dev_output_weight = DeviceBuffer::from_host(stream, &output_weight)?;
    let dev_quantized_output_weight = quantized_output_weight.to_device(stream)?;
    stream.synchronize()?;
    drop((norm_weight, output_weight, quantized_output_weight));

    Ok(QuantizationOutputWeights {
        norm: dev_norm_weight,
        output: dev_output_weight,
        quantized_output: dev_quantized_output_weight,
    })
}

fn run_quantization_suite_prompt_with_outputs(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
    output_weights: &QuantizationOutputWeights,
    top_k: usize,
) -> Result<QuantizationSuitePromptProbe> {
    if tokens.is_empty() {
        return Err(invalid_data("quantization suite prompt cannot be empty"));
    }
    for &token in tokens {
        validate_token(config, token)?;
    }

    let reference_hidden = run_prompt_to_final_hidden(stream, module, weights, config, tokens)?;
    let ffn_hidden =
        run_prompt_to_final_hidden_quantized_ffn(stream, module, weights, config, tokens)?;
    let attention_hidden =
        run_prompt_to_final_hidden_quantized_attention(stream, module, weights, config, tokens)?;
    let attention_ffn_hidden = run_prompt_to_final_hidden_quantized_attention_ffn(
        stream, module, weights, config, tokens,
    )?;

    let variants = vec![
        compare_hidden_logits_variant(
            stream,
            module,
            config,
            "output",
            &reference_hidden,
            &reference_hidden,
            &output_weights.norm,
            &output_weights.output,
            OutputProjection::QuantizedI8(&output_weights.quantized_output),
            top_k,
        )?,
        compare_hidden_logits_variant(
            stream,
            module,
            config,
            "ffn",
            &reference_hidden,
            &ffn_hidden,
            &output_weights.norm,
            &output_weights.output,
            OutputProjection::Bf16(&output_weights.output),
            top_k,
        )?,
        compare_hidden_logits_variant(
            stream,
            module,
            config,
            "attention",
            &reference_hidden,
            &attention_hidden,
            &output_weights.norm,
            &output_weights.output,
            OutputProjection::Bf16(&output_weights.output),
            top_k,
        )?,
        compare_hidden_logits_variant(
            stream,
            module,
            config,
            "attention+ffn",
            &reference_hidden,
            &attention_ffn_hidden,
            &output_weights.norm,
            &output_weights.output,
            OutputProjection::Bf16(&output_weights.output),
            top_k,
        )?,
        compare_hidden_logits_variant(
            stream,
            module,
            config,
            "all-linear",
            &reference_hidden,
            &attention_ffn_hidden,
            &output_weights.norm,
            &output_weights.output,
            OutputProjection::QuantizedI8(&output_weights.quantized_output),
            top_k,
        )?,
    ];

    Ok(QuantizationSuitePromptProbe {
        tokens: tokens.to_vec(),
        variants,
    })
}

fn run_all_linear_quantization_variant_with_outputs(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
    output_weights: &QuantizationOutputWeights,
    top_k: usize,
) -> Result<QuantizationSuiteVariantProbe> {
    if tokens.is_empty() {
        return Err(invalid_data(
            "all-linear quantization prompt cannot be empty",
        ));
    }
    for &token in tokens {
        validate_token(config, token)?;
    }

    let reference_hidden = run_prompt_to_final_hidden(stream, module, weights, config, tokens)?;
    let attention_ffn_hidden = run_prompt_to_final_hidden_quantized_attention_ffn(
        stream, module, weights, config, tokens,
    )?;

    compare_hidden_logits_variant(
        stream,
        module,
        config,
        "all-linear",
        &reference_hidden,
        &attention_ffn_hidden,
        &output_weights.norm,
        &output_weights.output,
        OutputProjection::QuantizedI8(&output_weights.quantized_output),
        top_k,
    )
}

enum OutputProjection<'a> {
    Bf16(&'a DeviceBuffer<Bf16>),
    QuantizedI8(&'a DeviceQuantizedI8Matrix),
}

impl ops::CudaLinearWeight for OutputProjection<'_> {
    fn launch_linear(
        &self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> std::result::Result<(), DriverError> {
        match self {
            Self::Bf16(weight) => ops::linear(stream, module, input, *weight, output),
            Self::QuantizedI8(weight) => ops::linear(stream, module, input, *weight, output),
        }
    }
}

fn compare_hidden_logits_variant(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    variant: &'static str,
    reference_hidden: &[f32],
    candidate_hidden: &[f32],
    norm_weight: &DeviceBuffer<Bf16>,
    reference_output_weight: &DeviceBuffer<Bf16>,
    candidate_output_weight: OutputProjection<'_>,
    top_k: usize,
) -> Result<QuantizationSuiteVariantProbe> {
    let dev_reference_hidden = DeviceBuffer::from_host(stream, reference_hidden)?;
    let dev_candidate_hidden = DeviceBuffer::from_host(stream, candidate_hidden)?;
    let mut reference_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut candidate_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut reference_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;
    let mut candidate_logits = DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?;

    logits_from_hidden_device_into(
        stream,
        module,
        config,
        &dev_reference_hidden,
        norm_weight,
        reference_output_weight,
        &mut reference_normed,
        &mut reference_logits,
    )?;
    ops::rmsnorm(
        stream,
        module,
        &dev_candidate_hidden,
        norm_weight,
        config.norm_eps,
        &mut candidate_normed,
    )?;
    ops::linear(
        stream,
        module,
        &candidate_normed,
        &candidate_output_weight,
        &mut candidate_logits,
    )?;

    let reference_logits_host = reference_logits.to_host_vec(stream)?;
    let candidate_logits_host = candidate_logits.to_host_vec(stream)?;
    let kl_divergence = kl_divergence_from_logits(&reference_logits_host, &candidate_logits_host)?;
    let max_abs_diff = max_abs_diff(&reference_logits_host, &candidate_logits_host);
    let mean_abs_diff = mean_abs_diff(&reference_logits_host, &candidate_logits_host);
    let reference_top_logits = top_k_logits(&reference_logits_host, top_k);
    let quantized_top_logits = top_k_logits(&candidate_logits_host, top_k);
    let &(reference_token_id, reference_logit) = reference_top_logits
        .first()
        .ok_or_else(|| invalid_data("reference top logits were empty"))?;
    let &(quantized_token_id, quantized_logit) = quantized_top_logits
        .first()
        .ok_or_else(|| invalid_data("candidate top logits were empty"))?;

    Ok(QuantizationSuiteVariantProbe {
        variant,
        reference_token_id,
        reference_logit,
        quantized_token_id,
        quantized_logit,
        token_matches: reference_token_id == quantized_token_id,
        reference_top_logits,
        quantized_top_logits,
        kl_divergence,
        max_abs_diff,
        mean_abs_diff,
    })
}

fn run_prompt_to_final_hidden(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    let mut hidden_states =
        read_prompt_embeddings_from_weights(stream, module, weights, config, tokens)?;
    let rope_freqs = rope_frequencies(config);
    let dev_rope_freqs = DeviceBuffer::from_host(stream, &rope_freqs)?;
    stream.synchronize()?;
    drop(rope_freqs);

    for layer in 0..config.n_layers {
        let layer_host = read_layer_host_weights(weights, config, layer)?;
        let layer_dev = upload_layer_weights(stream, &layer_host)?;
        stream.synchronize()?;
        drop(layer_host);

        hidden_states = run_prompt_layer_from_hidden_states(
            stream,
            module,
            config,
            &layer_dev,
            &dev_rope_freqs,
            &hidden_states,
        )?;
        stream.synchronize()?;
    }

    hidden_states
        .last()
        .cloned()
        .ok_or_else(|| invalid_data("missing final hidden state"))
}

fn run_prompt_to_final_hidden_quantized_attention_ffn(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    let mut hidden_states =
        read_prompt_embeddings_from_weights(stream, module, weights, config, tokens)?;
    let rope_freqs = rope_frequencies(config);
    let dev_rope_freqs = DeviceBuffer::from_host(stream, &rope_freqs)?;
    stream.synchronize()?;
    drop(rope_freqs);

    for layer in 0..config.n_layers {
        let layer_host = read_layer_host_weights(weights, config, layer)?;
        let layer_dev = upload_quantized_attention_ffn_layer_weights(stream, config, &layer_host)?;
        stream.synchronize()?;
        drop(layer_host);

        hidden_states = run_prompt_layer_quantized_attention_ffn_from_hidden_states(
            stream,
            module,
            config,
            &layer_dev,
            &dev_rope_freqs,
            &hidden_states,
        )?;
        stream.synchronize()?;
    }

    hidden_states
        .last()
        .cloned()
        .ok_or_else(|| invalid_data("missing final hidden state"))
}

fn run_prompt_to_final_hidden_quantized_attention(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    let mut hidden_states =
        read_prompt_embeddings_from_weights(stream, module, weights, config, tokens)?;
    let rope_freqs = rope_frequencies(config);
    let dev_rope_freqs = DeviceBuffer::from_host(stream, &rope_freqs)?;
    stream.synchronize()?;
    drop(rope_freqs);

    for layer in 0..config.n_layers {
        let layer_host = read_layer_host_weights(weights, config, layer)?;
        let layer_dev = upload_quantized_attention_layer_weights(stream, config, &layer_host)?;
        stream.synchronize()?;
        drop(layer_host);

        hidden_states = run_prompt_layer_quantized_attention_from_hidden_states(
            stream,
            module,
            config,
            &layer_dev,
            &dev_rope_freqs,
            &hidden_states,
        )?;
        stream.synchronize()?;
    }

    hidden_states
        .last()
        .cloned()
        .ok_or_else(|| invalid_data("missing final hidden state"))
}

fn run_prompt_to_final_hidden_quantized_ffn(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    let mut hidden_states =
        read_prompt_embeddings_from_weights(stream, module, weights, config, tokens)?;
    let rope_freqs = rope_frequencies(config);
    let dev_rope_freqs = DeviceBuffer::from_host(stream, &rope_freqs)?;
    stream.synchronize()?;
    drop(rope_freqs);

    for layer in 0..config.n_layers {
        let layer_host = read_layer_host_weights(weights, config, layer)?;
        let layer_dev = upload_quantized_ffn_layer_weights(stream, config, &layer_host)?;
        stream.synchronize()?;
        drop(layer_host);

        hidden_states = run_prompt_layer_quantized_ffn_from_hidden_states(
            stream,
            module,
            config,
            &layer_dev,
            &dev_rope_freqs,
            &hidden_states,
        )?;
        stream.synchronize()?;
    }

    hidden_states
        .last()
        .cloned()
        .ok_or_else(|| invalid_data("missing final hidden state"))
}

fn read_embedding_row(
    weights: &ModelWeights,
    config: &TextConfig,
    token_id: u32,
) -> Result<Vec<Bf16>> {
    let tensor = weights.tensor("tok_embeddings.weight")?;
    expect_shape(&tensor, &[config.vocab_size, config.dim])?;
    weights.read_bf16_range(&tensor, token_id as usize * config.dim, config.dim)
}

fn read_embedding_weight(weights: &ModelWeights, config: &TextConfig) -> Result<Vec<Bf16>> {
    read_bf16_shape(
        weights,
        "tok_embeddings.weight",
        &[config.vocab_size, config.dim],
    )
}

fn write_prompt_embeddings_device(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    embedding_weight: &DeviceBuffer<Bf16>,
    tokens: &[u32],
    out: &mut [DeviceBuffer<f32>],
) -> Result<()> {
    if out.len() < tokens.len() {
        return Err(invalid_data(format!(
            "prompt embedding output has {} buffers, expected at least {}",
            out.len(),
            tokens.len()
        )));
    }

    for (i, (&token, hidden)) in tokens.iter().zip(out.iter_mut()).enumerate() {
        if hidden.len() != config.dim {
            return Err(invalid_data(format!(
                "prompt embedding output {i} has length {}, expected {}",
                hidden.len(),
                config.dim
            )));
        }
        ops::embedding(stream, module, embedding_weight, token, config.dim, hidden)?;
    }

    Ok(())
}

fn read_prompt_embeddings_from_weights(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    weights: &ModelWeights,
    config: &TextConfig,
    tokens: &[u32],
) -> Result<Vec<Vec<f32>>> {
    let mut hidden_states = Vec::with_capacity(tokens.len());

    for &token in tokens {
        let token_row = read_embedding_row(weights, config, token)?;
        let dev_token_row = DeviceBuffer::from_host(stream, &token_row)?;
        let mut hidden = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
        ops::embedding(stream, module, &dev_token_row, 0, config.dim, &mut hidden)?;
        hidden_states.push(hidden.to_host_vec(stream)?);
    }

    Ok(hidden_states)
}

fn run_single_token_block_gpu(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    token_row: &[Bf16],
    layer: &LayerDeviceWeights,
) -> Result<SingleTokenBlockOutput> {
    let dev_token_row = DeviceBuffer::from_host(stream, token_row)?;
    let mut hidden = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut attention_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut query = DeviceBuffer::<f32>::zeroed(stream, config.n_heads * config.head_dim)?;
    let mut key = DeviceBuffer::<f32>::zeroed(stream, config.n_kv_heads * config.head_dim)?;
    let mut value = DeviceBuffer::<f32>::zeroed(stream, config.n_kv_heads * config.head_dim)?;
    let mut attention_heads =
        DeviceBuffer::<f32>::zeroed(stream, config.n_heads * config.head_dim)?;
    let mut attention_delta = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut attention_residual = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;

    ops::embedding(stream, module, &dev_token_row, 0, config.dim, &mut hidden)?;
    ops::rmsnorm(
        stream,
        module,
        &hidden,
        &layer.attention_norm,
        config.norm_eps,
        &mut attention_normed,
    )?;
    ops::linear(stream, module, &attention_normed, &layer.wq, &mut query)?;
    ops::linear(stream, module, &attention_normed, &layer.wk, &mut key)?;
    ops::linear(stream, module, &attention_normed, &layer.wv, &mut value)?;
    ops::single_token_gqa(
        stream,
        module,
        &value,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        &mut attention_heads,
    )?;
    ops::linear(
        stream,
        module,
        &attention_heads,
        &layer.wo,
        &mut attention_delta,
    )?;
    ops::add(
        stream,
        module,
        &hidden,
        &attention_delta,
        &mut attention_residual,
    )?;

    let query_host = query.to_host_vec(stream)?;
    let key_host = key.to_host_vec(stream)?;
    let attention_host = attention_residual.to_host_vec(stream)?;
    let output = run_ffn_from_hidden(
        stream,
        module,
        config,
        &attention_residual,
        &layer.ffn_norm,
        &layer.w1,
        &layer.w3,
        &layer.w2,
    )?;
    let output_host = output.to_host_vec(stream)?;

    Ok(SingleTokenBlockOutput {
        query: query_host,
        key: key_host,
        attention_residual: attention_host,
        output: output_host,
    })
}

fn run_single_token_block_cpu(
    config: &TextConfig,
    token_row: &[Bf16],
    layer: &LayerHostWeights,
) -> SingleTokenBlockOutput {
    let hidden = storage_to_f32_vec(token_row);
    let attention_normed = cpu_rmsnorm_bf16(&hidden, &layer.attention_norm, config.norm_eps);
    let query = cpu_matvec_bf16(
        &attention_normed,
        &layer.wq,
        config.n_heads * config.head_dim,
        config.dim,
    );
    let key = cpu_matvec_bf16(
        &attention_normed,
        &layer.wk,
        config.n_kv_heads * config.head_dim,
        config.dim,
    );
    let value = cpu_matvec_bf16(
        &attention_normed,
        &layer.wv,
        config.n_kv_heads * config.head_dim,
        config.dim,
    );
    let attention_heads =
        cpu_single_token_gqa(&value, config.n_heads, config.n_kv_heads, config.head_dim);
    let attention_delta = cpu_matvec_bf16(
        &attention_heads,
        &layer.wo,
        config.dim,
        config.n_heads * config.head_dim,
    );
    let attention_residual = cpu_add(&hidden, &attention_delta);
    let output = cpu_ffn_from_hidden(config, &attention_residual, layer);

    SingleTokenBlockOutput {
        query,
        key,
        attention_residual,
        output,
    }
}

fn run_prompt_block_gpu(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    token_rows: &[Vec<Bf16>],
    layer: &LayerDeviceWeights,
    rope_freqs: &DeviceBuffer<f32>,
) -> Result<PromptBlockOutput> {
    let max_seq_len = token_rows.len();
    let kv_len = config.n_kv_heads * config.head_dim;
    let q_len = config.n_heads * config.head_dim;
    let mut key_cache = DeviceBuffer::<f32>::zeroed(stream, max_seq_len * kv_len)?;
    let mut value_cache = DeviceBuffer::<f32>::zeroed(stream, max_seq_len * kv_len)?;
    let mut scores = DeviceBuffer::<f32>::zeroed(stream, config.n_heads * max_seq_len)?;
    let mut hidden = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut attention_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut query = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut key = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut value = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut query_rot = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut key_rot = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut attention_heads = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut attention_delta = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut attention_residual = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;

    let mut final_attention = Vec::new();
    let mut final_output = Vec::new();
    for (position, token_row) in token_rows.iter().enumerate() {
        let dev_token_row = DeviceBuffer::from_host(stream, token_row)?;
        ops::embedding(stream, module, &dev_token_row, 0, config.dim, &mut hidden)?;
        ops::rmsnorm(
            stream,
            module,
            &hidden,
            &layer.attention_norm,
            config.norm_eps,
            &mut attention_normed,
        )?;
        ops::linear(stream, module, &attention_normed, &layer.wq, &mut query)?;
        ops::linear(stream, module, &attention_normed, &layer.wk, &mut key)?;
        ops::linear(stream, module, &attention_normed, &layer.wv, &mut value)?;
        ops::apply_rope(
            stream,
            module,
            &query,
            rope_freqs,
            position,
            config.head_dim,
            &mut query_rot,
        )?;
        ops::apply_rope(
            stream,
            module,
            &key,
            rope_freqs,
            position,
            config.head_dim,
            &mut key_rot,
        )?;
        ops::write_kv_cache(
            stream,
            module,
            &key_rot,
            position,
            max_seq_len,
            config.n_kv_heads,
            config.head_dim,
            &mut key_cache,
        )?;
        ops::write_kv_cache(
            stream,
            module,
            &value,
            position,
            max_seq_len,
            config.n_kv_heads,
            config.head_dim,
            &mut value_cache,
        )?;
        ops::attention_scores(
            stream,
            module,
            &query_rot,
            &key_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scores,
        )?;
        ops::softmax_value(
            stream,
            module,
            &scores,
            &value_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut attention_heads,
        )?;
        ops::linear(
            stream,
            module,
            &attention_heads,
            &layer.wo,
            &mut attention_delta,
        )?;
        ops::add(
            stream,
            module,
            &hidden,
            &attention_delta,
            &mut attention_residual,
        )?;

        if position + 1 == max_seq_len {
            let output = run_ffn_from_hidden(
                stream,
                module,
                config,
                &attention_residual,
                &layer.ffn_norm,
                &layer.w1,
                &layer.w3,
                &layer.w2,
            )?;
            final_attention = attention_residual.to_host_vec(stream)?;
            final_output = output.to_host_vec(stream)?;
        }
    }

    Ok(PromptBlockOutput {
        query: query_rot.to_host_vec(stream)?,
        key: key_rot.to_host_vec(stream)?,
        scores: scores.to_host_vec(stream)?,
        attention_residual: final_attention,
        output: final_output,
    })
}

fn run_prompt_block_cpu(
    config: &TextConfig,
    token_rows: &[Vec<Bf16>],
    layer: &LayerHostWeights,
    rope_freqs: &[f32],
) -> PromptBlockOutput {
    let max_seq_len = token_rows.len();
    let kv_len = config.n_kv_heads * config.head_dim;
    let q_len = config.n_heads * config.head_dim;
    let mut key_cache = vec![0.0; max_seq_len * kv_len];
    let mut value_cache = vec![0.0; max_seq_len * kv_len];
    let mut final_query = vec![0.0; q_len];
    let mut final_key = vec![0.0; kv_len];
    let mut final_scores = vec![0.0; config.n_heads * max_seq_len];
    let mut final_attention = Vec::new();
    let mut final_output = Vec::new();

    for (position, token_row) in token_rows.iter().enumerate() {
        let hidden = storage_to_f32_vec(token_row);
        let attention_normed = cpu_rmsnorm_bf16(&hidden, &layer.attention_norm, config.norm_eps);
        let query = cpu_matvec_bf16(
            &attention_normed,
            &layer.wq,
            config.n_heads * config.head_dim,
            config.dim,
        );
        let key = cpu_matvec_bf16(
            &attention_normed,
            &layer.wk,
            config.n_kv_heads * config.head_dim,
            config.dim,
        );
        let value = cpu_matvec_bf16(
            &attention_normed,
            &layer.wv,
            config.n_kv_heads * config.head_dim,
            config.dim,
        );
        let query_rot = cpu_apply_rope(&query, rope_freqs, position, config.head_dim);
        let key_rot = cpu_apply_rope(&key, rope_freqs, position, config.head_dim);
        let cache_offset = position * kv_len;
        key_cache[cache_offset..cache_offset + kv_len].copy_from_slice(&key_rot);
        value_cache[cache_offset..cache_offset + kv_len].copy_from_slice(&value);

        let scores = cpu_attention_scores(
            &query_rot,
            &key_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
        );
        let attention_heads = cpu_softmax_value(
            &scores,
            &value_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
        );
        let attention_delta = cpu_matvec_bf16(
            &attention_heads,
            &layer.wo,
            config.dim,
            config.n_heads * config.head_dim,
        );
        let attention_residual = cpu_add(&hidden, &attention_delta);

        if position + 1 == max_seq_len {
            final_query = query_rot;
            final_key = key_rot;
            final_scores = scores;
            final_attention = attention_residual;
            final_output = cpu_ffn_from_hidden(config, &final_attention, layer);
        }
    }

    PromptBlockOutput {
        query: final_query,
        key: final_key,
        scores: final_scores,
        attention_residual: final_attention,
        output: final_output,
    }
}

fn read_layer_host_weights(
    weights: &ModelWeights,
    config: &TextConfig,
    layer: usize,
) -> Result<LayerHostWeights> {
    Ok(LayerHostWeights {
        attention_norm: read_bf16_shape(
            weights,
            &format!("layers.{layer}.attention_norm.weight"),
            &[config.dim],
        )?,
        wq: read_bf16_shape(
            weights,
            &format!("layers.{layer}.attention.wq.weight"),
            &[config.n_heads * config.head_dim, config.dim],
        )?,
        wk: read_bf16_shape(
            weights,
            &format!("layers.{layer}.attention.wk.weight"),
            &[config.n_kv_heads * config.head_dim, config.dim],
        )?,
        wv: read_bf16_shape(
            weights,
            &format!("layers.{layer}.attention.wv.weight"),
            &[config.n_kv_heads * config.head_dim, config.dim],
        )?,
        wo: read_bf16_shape(
            weights,
            &format!("layers.{layer}.attention.wo.weight"),
            &[config.dim, config.n_heads * config.head_dim],
        )?,
        ffn_norm: read_bf16_shape(
            weights,
            &format!("layers.{layer}.ffn_norm.weight"),
            &[config.dim],
        )?,
        w1: read_bf16_shape(
            weights,
            &format!("layers.{layer}.feed_forward.w1.weight"),
            &[config.hidden_dim, config.dim],
        )?,
        w3: read_bf16_shape(
            weights,
            &format!("layers.{layer}.feed_forward.w3.weight"),
            &[config.hidden_dim, config.dim],
        )?,
        w2: read_bf16_shape(
            weights,
            &format!("layers.{layer}.feed_forward.w2.weight"),
            &[config.dim, config.hidden_dim],
        )?,
    })
}

fn read_model_norm_weight(weights: &ModelWeights, config: &TextConfig) -> Result<Vec<Bf16>> {
    read_bf16_shape(weights, "norm.weight", &[config.dim])
}

fn read_output_weight(weights: &ModelWeights, config: &TextConfig) -> Result<Vec<Bf16>> {
    read_bf16_shape(weights, "output.weight", &[config.vocab_size, config.dim])
}

fn upload_layer_weights(
    stream: &Arc<CudaStream>,
    weights: &LayerHostWeights,
) -> Result<LayerDeviceWeights> {
    Ok(LayerDeviceWeights {
        attention_norm: DeviceBuffer::from_host(stream, &weights.attention_norm)?,
        wq: DeviceBuffer::from_host(stream, &weights.wq)?,
        wk: DeviceBuffer::from_host(stream, &weights.wk)?,
        wv: DeviceBuffer::from_host(stream, &weights.wv)?,
        wo: DeviceBuffer::from_host(stream, &weights.wo)?,
        ffn_norm: DeviceBuffer::from_host(stream, &weights.ffn_norm)?,
        w1: DeviceBuffer::from_host(stream, &weights.w1)?,
        w3: DeviceBuffer::from_host(stream, &weights.w3)?,
        w2: DeviceBuffer::from_host(stream, &weights.w2)?,
    })
}

fn upload_quantized_ffn_layer_weights(
    stream: &Arc<CudaStream>,
    config: &TextConfig,
    weights: &LayerHostWeights,
) -> Result<QuantizedFfnLayerDeviceWeights> {
    let w1 =
        QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.w1, config.hidden_dim, config.dim);
    let w3 =
        QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.w3, config.hidden_dim, config.dim);
    let w2 =
        QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.w2, config.dim, config.hidden_dim);

    Ok(QuantizedFfnLayerDeviceWeights {
        attention_norm: DeviceBuffer::from_host(stream, &weights.attention_norm)?,
        wq: DeviceBuffer::from_host(stream, &weights.wq)?,
        wk: DeviceBuffer::from_host(stream, &weights.wk)?,
        wv: DeviceBuffer::from_host(stream, &weights.wv)?,
        wo: DeviceBuffer::from_host(stream, &weights.wo)?,
        ffn_norm: DeviceBuffer::from_host(stream, &weights.ffn_norm)?,
        w1: w1.to_device(stream)?,
        w3: w3.to_device(stream)?,
        w2: w2.to_device(stream)?,
    })
}

fn upload_quantized_attention_layer_weights(
    stream: &Arc<CudaStream>,
    config: &TextConfig,
    weights: &LayerHostWeights,
) -> Result<QuantizedAttentionLayerDeviceWeights> {
    let q_len = config.n_heads * config.head_dim;
    let kv_len = config.n_kv_heads * config.head_dim;
    let wq = QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.wq, q_len, config.dim);
    let wk = QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.wk, kv_len, config.dim);
    let wv = QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.wv, kv_len, config.dim);
    let wo = QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.wo, config.dim, q_len);

    Ok(QuantizedAttentionLayerDeviceWeights {
        attention_norm: DeviceBuffer::from_host(stream, &weights.attention_norm)?,
        wq: wq.to_device(stream)?,
        wk: wk.to_device(stream)?,
        wv: wv.to_device(stream)?,
        wo: wo.to_device(stream)?,
        ffn_norm: DeviceBuffer::from_host(stream, &weights.ffn_norm)?,
        w1: DeviceBuffer::from_host(stream, &weights.w1)?,
        w3: DeviceBuffer::from_host(stream, &weights.w3)?,
        w2: DeviceBuffer::from_host(stream, &weights.w2)?,
    })
}

fn upload_quantized_attention_ffn_layer_weights(
    stream: &Arc<CudaStream>,
    config: &TextConfig,
    weights: &LayerHostWeights,
) -> Result<QuantizedAttentionFfnLayerDeviceWeights> {
    let q_len = config.n_heads * config.head_dim;
    let kv_len = config.n_kv_heads * config.head_dim;
    let wq = QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.wq, q_len, config.dim);
    let wk = QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.wk, kv_len, config.dim);
    let wv = QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.wv, kv_len, config.dim);
    let wo = QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.wo, config.dim, q_len);
    let w1 =
        QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.w1, config.hidden_dim, config.dim);
    let w3 =
        QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.w3, config.hidden_dim, config.dim);
    let w2 =
        QuantizedI8Matrix::from_bf16_rows_symmetric(&weights.w2, config.dim, config.hidden_dim);

    Ok(QuantizedAttentionFfnLayerDeviceWeights {
        attention_norm: DeviceBuffer::from_host(stream, &weights.attention_norm)?,
        wq: wq.to_device(stream)?,
        wk: wk.to_device(stream)?,
        wv: wv.to_device(stream)?,
        wo: wo.to_device(stream)?,
        ffn_norm: DeviceBuffer::from_host(stream, &weights.ffn_norm)?,
        w1: w1.to_device(stream)?,
        w3: w3.to_device(stream)?,
        w2: w2.to_device(stream)?,
    })
}

fn upload_exported_quantized_attention_ffn_layer_weights(
    stream: &Arc<CudaStream>,
    config: &TextConfig,
    weights: &ModelWeights,
    layer: usize,
    export_dir: &Path,
) -> Result<QuantizedAttentionFfnLayerDeviceWeights> {
    let q_len = config.n_heads * config.head_dim;
    let kv_len = config.n_kv_heads * config.head_dim;
    let index = |offset| exported_quantized_layer_tensor_index(layer, offset);
    let wq = QuantizedI8Matrix::read_exported(
        export_dir,
        index(0),
        &format!("layers.{layer}.attention.wq.weight"),
        q_len,
        config.dim,
    )?;
    let wk = QuantizedI8Matrix::read_exported(
        export_dir,
        index(1),
        &format!("layers.{layer}.attention.wk.weight"),
        kv_len,
        config.dim,
    )?;
    let wv = QuantizedI8Matrix::read_exported(
        export_dir,
        index(2),
        &format!("layers.{layer}.attention.wv.weight"),
        kv_len,
        config.dim,
    )?;
    let wo = QuantizedI8Matrix::read_exported(
        export_dir,
        index(3),
        &format!("layers.{layer}.attention.wo.weight"),
        config.dim,
        q_len,
    )?;
    let w1 = QuantizedI8Matrix::read_exported(
        export_dir,
        index(4),
        &format!("layers.{layer}.feed_forward.w1.weight"),
        config.hidden_dim,
        config.dim,
    )?;
    let w3 = QuantizedI8Matrix::read_exported(
        export_dir,
        index(5),
        &format!("layers.{layer}.feed_forward.w3.weight"),
        config.hidden_dim,
        config.dim,
    )?;
    let w2 = QuantizedI8Matrix::read_exported(
        export_dir,
        index(6),
        &format!("layers.{layer}.feed_forward.w2.weight"),
        config.dim,
        config.hidden_dim,
    )?;

    Ok(QuantizedAttentionFfnLayerDeviceWeights {
        attention_norm: DeviceBuffer::from_host(
            stream,
            &read_bf16_shape(
                weights,
                &format!("layers.{layer}.attention_norm.weight"),
                &[config.dim],
            )?,
        )?,
        wq: wq.to_device(stream)?,
        wk: wk.to_device(stream)?,
        wv: wv.to_device(stream)?,
        wo: wo.to_device(stream)?,
        ffn_norm: DeviceBuffer::from_host(
            stream,
            &read_bf16_shape(
                weights,
                &format!("layers.{layer}.ffn_norm.weight"),
                &[config.dim],
            )?,
        )?,
        w1: w1.to_device(stream)?,
        w3: w3.to_device(stream)?,
        w2: w2.to_device(stream)?,
    })
}

#[derive(Debug)]
struct ExportedQuantizedTensorSpec {
    index: usize,
    name: String,
    rows: usize,
    cols: usize,
}

fn validate_all_linear_int8_export_archive(
    export_dir: &Path,
    config: &TextConfig,
    expected_weights_path: Option<&Path>,
) -> Result<()> {
    let export_metadata = fs::metadata(export_dir).map_err(|error| {
        invalid_data(format!(
            "all-linear int8 export directory {} is not readable: {error}",
            export_dir.display()
        ))
    })?;
    if !export_metadata.is_dir() {
        return Err(invalid_data(format!(
            "all-linear int8 export path {} is not a directory",
            export_dir.display()
        )));
    }

    let manifest_path = export_dir.join("manifest.json");
    validate_export_regular_file_len(&manifest_path, None, "all-linear int8 export manifest")?;
    validate_quantized_export_manifest_metadata(&manifest_path, expected_weights_path)?;

    let expected_tensors = all_linear_int8_export_tensor_specs(config);
    for tensor in &expected_tensors {
        let (values_path, scales_path) =
            quantized_export_file_paths(export_dir, tensor.index, &tensor.name);
        let values_bytes = checked_export_file_len(
            tensor.rows,
            tensor.cols,
            "quantized values byte count",
            &tensor.name,
        )?;
        let scales_bytes = checked_export_file_len(
            tensor.rows,
            size_of::<f32>(),
            "quantized scales byte count",
            &tensor.name,
        )?;
        validate_export_regular_file_len(
            &values_path,
            Some(values_bytes as u64),
            "all-linear int8 values file",
        )?;
        validate_export_regular_file_len(
            &scales_path,
            Some(scales_bytes as u64),
            "all-linear int8 scales file",
        )?;
    }

    Ok(())
}

fn validate_quantized_export_manifest_metadata(
    manifest_path: &Path,
    expected_weights_path: Option<&Path>,
) -> Result<()> {
    let manifest = fs::read_to_string(manifest_path).map_err(|error| {
        invalid_data(format!(
            "all-linear int8 export manifest {} is not readable: {error}",
            manifest_path.display()
        ))
    })?;
    let quantization = parse_json_string_field(&manifest, "quantization")?;
    if quantization != "row-symmetric-int8-with-f32-row-scales" {
        return Err(invalid_data(format!(
            "all-linear int8 export manifest {} has unsupported quantization {quantization:?}",
            manifest_path.display()
        )));
    }

    let Some(expected_weights_path) = expected_weights_path else {
        return Ok(());
    };
    let manifest_weights_path = parse_json_string_field(&manifest, "weights_path")?;
    if paths_refer_to_same_file(Path::new(&manifest_weights_path), expected_weights_path) {
        return Ok(());
    }

    Err(invalid_data(format!(
        "all-linear int8 export manifest {} was generated from weights_path {}, but current model weights source is {}; regenerate the export archive",
        manifest_path.display(),
        manifest_weights_path,
        expected_weights_path.display()
    )))
}

fn paths_refer_to_same_file(lhs: &Path, rhs: &Path) -> bool {
    if lhs == rhs {
        return true;
    }

    match (fs::canonicalize(lhs), fs::canonicalize(rhs)) {
        (Ok(lhs), Ok(rhs)) => lhs == rhs,
        _ => false,
    }
}

fn all_linear_int8_export_tensor_specs(config: &TextConfig) -> Vec<ExportedQuantizedTensorSpec> {
    let q_len = config.n_heads * config.head_dim;
    let kv_len = config.n_kv_heads * config.head_dim;
    let mut specs = Vec::with_capacity(1 + config.n_layers * 7);

    specs.push(ExportedQuantizedTensorSpec {
        index: 0,
        name: "output.weight".to_string(),
        rows: config.vocab_size,
        cols: config.dim,
    });
    for layer in 0..config.n_layers {
        let index = |offset| exported_quantized_layer_tensor_index(layer, offset);
        specs.push(ExportedQuantizedTensorSpec {
            index: index(0),
            name: format!("layers.{layer}.attention.wq.weight"),
            rows: q_len,
            cols: config.dim,
        });
        specs.push(ExportedQuantizedTensorSpec {
            index: index(1),
            name: format!("layers.{layer}.attention.wk.weight"),
            rows: kv_len,
            cols: config.dim,
        });
        specs.push(ExportedQuantizedTensorSpec {
            index: index(2),
            name: format!("layers.{layer}.attention.wv.weight"),
            rows: kv_len,
            cols: config.dim,
        });
        specs.push(ExportedQuantizedTensorSpec {
            index: index(3),
            name: format!("layers.{layer}.attention.wo.weight"),
            rows: config.dim,
            cols: q_len,
        });
        specs.push(ExportedQuantizedTensorSpec {
            index: index(4),
            name: format!("layers.{layer}.feed_forward.w1.weight"),
            rows: config.hidden_dim,
            cols: config.dim,
        });
        specs.push(ExportedQuantizedTensorSpec {
            index: index(5),
            name: format!("layers.{layer}.feed_forward.w3.weight"),
            rows: config.hidden_dim,
            cols: config.dim,
        });
        specs.push(ExportedQuantizedTensorSpec {
            index: index(6),
            name: format!("layers.{layer}.feed_forward.w2.weight"),
            rows: config.dim,
            cols: config.hidden_dim,
        });
    }

    specs
}

fn checked_export_file_len(
    lhs: usize,
    rhs: usize,
    context: &str,
    tensor_name: &str,
) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        invalid_data(format!(
            "{context} overflow for exported tensor {tensor_name}"
        ))
    })
}

fn validate_export_regular_file_len(
    path: &Path,
    expected_bytes: Option<u64>,
    context: &str,
) -> Result<()> {
    let metadata = fs::metadata(path).map_err(|error| {
        invalid_data(format!(
            "{context} {} is not readable: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(invalid_data(format!(
            "{context} {} is not a regular file",
            path.display()
        )));
    }
    if let Some(expected_bytes) = expected_bytes {
        let actual_bytes = metadata.len();
        if actual_bytes != expected_bytes {
            return Err(invalid_data(format!(
                "{context} {} has {actual_bytes} bytes, expected {expected_bytes}",
                path.display()
            )));
        }
    }

    Ok(())
}

fn exported_quantized_layer_tensor_index(layer: usize, offset: usize) -> usize {
    1 + layer * 7 + offset
}

fn load_layer_device_weights(
    stream: &Arc<CudaStream>,
    weights: &ModelWeights,
    config: &TextConfig,
) -> Result<Vec<LayerDeviceWeights>> {
    let mut layers = Vec::with_capacity(config.n_layers);

    for layer_idx in 0..config.n_layers {
        let layer_host = read_layer_host_weights(weights, config, layer_idx)?;
        layers.push(upload_layer_weights(stream, &layer_host)?);
    }

    Ok(layers)
}

fn load_quantized_attention_ffn_layer_device_weights(
    stream: &Arc<CudaStream>,
    weights: &ModelWeights,
    config: &TextConfig,
) -> Result<Vec<QuantizedAttentionFfnLayerDeviceWeights>> {
    let mut layers = Vec::with_capacity(config.n_layers);

    for layer_idx in 0..config.n_layers {
        let layer_host = read_layer_host_weights(weights, config, layer_idx)?;
        layers.push(upload_quantized_attention_ffn_layer_weights(
            stream,
            config,
            &layer_host,
        )?);
    }

    Ok(layers)
}

fn load_exported_quantized_attention_ffn_layer_device_weights(
    stream: &Arc<CudaStream>,
    weights: &ModelWeights,
    config: &TextConfig,
    export_dir: &Path,
) -> Result<Vec<QuantizedAttentionFfnLayerDeviceWeights>> {
    let mut layers = Vec::with_capacity(config.n_layers);

    for layer_idx in 0..config.n_layers {
        layers.push(upload_exported_quantized_attention_ffn_layer_weights(
            stream, config, weights, layer_idx, export_dir,
        )?);
    }

    Ok(layers)
}

fn rope_frequencies(config: &TextConfig) -> Vec<f32> {
    let half = config.head_dim / 2;
    let default_freqs: Vec<f32> = (0..half)
        .map(|i| {
            1.0 / config
                .rope_theta
                .powf((2 * i) as f32 / config.head_dim as f32)
        })
        .collect();

    if let Some(yarn) = config.rope_yarn {
        return yarn_rope_frequencies(config, yarn, &default_freqs);
    }

    default_freqs
}

fn yarn_rope_frequencies(
    config: &TextConfig,
    yarn: YarnRopeConfig,
    default_freqs: &[f32],
) -> Vec<f32> {
    let low = yarn_correction_dim(
        yarn.beta_fast,
        config.head_dim,
        config.rope_theta,
        yarn.original_max_position_embeddings,
    )
    .floor()
    .max(0.0);
    let high = yarn_correction_dim(
        yarn.beta_slow,
        config.head_dim,
        config.rope_theta,
        yarn.original_max_position_embeddings,
    )
    .ceil()
    .min((config.head_dim - 1) as f32);

    default_freqs
        .iter()
        .enumerate()
        .map(|(i, &extrapolated)| {
            let interpolation = extrapolated / yarn.factor;
            let extrapolation_weight = 1.0 - yarn_linear_ramp(low, high, i as f32);
            interpolation * (1.0 - extrapolation_weight) + extrapolated * extrapolation_weight
        })
        .collect()
}

fn yarn_correction_dim(
    num_rotations: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> f32 {
    (dim as f32
        * ((max_position_embeddings as f32) / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * base.ln())
}

fn yarn_linear_ramp(low: f32, mut high: f32, dim: f32) -> f32 {
    if (low - high).abs() < f32::EPSILON {
        high += 0.001;
    }
    ((dim - low) / (high - low)).clamp(0.0, 1.0)
}

fn allocate_layer_kv_caches(
    stream: &Arc<CudaStream>,
    config: &TextConfig,
    max_seq_len: usize,
) -> Result<Vec<LayerKvCache>> {
    let kv_cache_len = max_seq_len * config.n_kv_heads * config.head_dim;
    let mut caches = Vec::with_capacity(config.n_layers);

    for _ in 0..config.n_layers {
        caches.push(LayerKvCache {
            key: DeviceBuffer::<f32>::zeroed(stream, kv_cache_len)?,
            value: DeviceBuffer::<f32>::zeroed(stream, kv_cache_len)?,
        });
    }

    Ok(caches)
}

fn allocate_hidden_state_buffers(
    stream: &Arc<CudaStream>,
    count: usize,
    dim: usize,
) -> Result<Vec<DeviceBuffer<f32>>> {
    let mut buffers = Vec::with_capacity(count);
    for _ in 0..count {
        buffers.push(DeviceBuffer::<f32>::zeroed(stream, dim)?);
    }

    Ok(buffers)
}

impl RuntimeScratch {
    fn new(stream: &Arc<CudaStream>, config: &TextConfig, max_seq_len: usize) -> Result<Self> {
        let prompt_batch_len = max_seq_len
            .checked_mul(config.dim)
            .ok_or_else(|| invalid_data("prompt batch scratch shape overflow"))?;
        let output_top1_partials = config.vocab_size;
        Ok(Self {
            hidden_a: DeviceBuffer::<f32>::zeroed(stream, config.dim)?,
            hidden_b: DeviceBuffer::<f32>::zeroed(stream, config.dim)?,
            prompt_a: allocate_hidden_state_buffers(stream, max_seq_len, config.dim)?,
            prompt_b: allocate_hidden_state_buffers(stream, max_seq_len, config.dim)?,
            prompt_a_batch: DeviceBuffer::<f32>::zeroed(stream, prompt_batch_len)?,
            prompt_b_batch: DeviceBuffer::<f32>::zeroed(stream, prompt_batch_len)?,
            layer: LayerScratch::new(stream, config, max_seq_len)?,
            logits_normed: DeviceBuffer::<f32>::zeroed(stream, config.dim)?,
            logits: DeviceBuffer::<f32>::zeroed(stream, config.vocab_size)?,
            argmax_packed: DeviceBuffer::<u64>::zeroed(stream, 1)?,
            output_top1_tokens: DeviceBuffer::<u32>::zeroed(stream, output_top1_partials)?,
            output_top1_logits: DeviceBuffer::<f32>::zeroed(stream, output_top1_partials)?,
            topk_tokens: DeviceBuffer::<u32>::zeroed(stream, DEVICE_TOP_K_CAPACITY)?,
            topk_logits: DeviceBuffer::<f32>::zeroed(stream, DEVICE_TOP_K_CAPACITY)?,
        })
    }

    fn bytes(&self) -> usize {
        let prompt_bytes = self
            .prompt_a
            .iter()
            .chain(self.prompt_b.iter())
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>();

        (self.hidden_a.len()
            + self.hidden_b.len()
            + self.prompt_a_batch.len()
            + self.prompt_b_batch.len()
            + self.logits_normed.len()
            + self.logits.len()
            + self.output_top1_logits.len()
            + self.topk_logits.len())
            * size_of::<f32>()
            + self.argmax_packed.len() * size_of::<u64>()
            + self.output_top1_tokens.len() * size_of::<u32>()
            + self.topk_tokens.len() * size_of::<u32>()
            + prompt_bytes
            + self.layer.bytes()
    }
}

impl LayerScratch {
    fn new(stream: &Arc<CudaStream>, config: &TextConfig, max_seq_len: usize) -> Result<Self> {
        let kv_len = config.n_kv_heads * config.head_dim;
        let q_len = config.n_heads * config.head_dim;
        let attention_normed_batch_len = max_seq_len
            .checked_mul(config.dim)
            .ok_or_else(|| invalid_data("attention norm batch scratch shape overflow"))?;
        let q_batch_len = max_seq_len
            .checked_mul(q_len)
            .ok_or_else(|| invalid_data("query batch scratch shape overflow"))?;
        let kv_batch_len = max_seq_len
            .checked_mul(kv_len)
            .ok_or_else(|| invalid_data("KV batch scratch shape overflow"))?;

        Ok(Self {
            attention_normed: DeviceBuffer::<f32>::zeroed(stream, config.dim)?,
            attention_normed_batch: DeviceBuffer::<f32>::zeroed(
                stream,
                attention_normed_batch_len,
            )?,
            query: DeviceBuffer::<f32>::zeroed(stream, q_len)?,
            query_batch: DeviceBuffer::<f32>::zeroed(stream, q_batch_len)?,
            key: DeviceBuffer::<f32>::zeroed(stream, kv_len)?,
            key_batch: DeviceBuffer::<f32>::zeroed(stream, kv_batch_len)?,
            value: DeviceBuffer::<f32>::zeroed(stream, kv_len)?,
            value_batch: DeviceBuffer::<f32>::zeroed(stream, kv_batch_len)?,
            query_rot: DeviceBuffer::<f32>::zeroed(stream, q_len)?,
            query_rot_batch: DeviceBuffer::<f32>::zeroed(stream, q_batch_len)?,
            key_rot: DeviceBuffer::<f32>::zeroed(stream, kv_len)?,
            scores: DeviceBuffer::<f32>::zeroed(stream, config.n_heads * max_seq_len)?,
            attention_heads: DeviceBuffer::<f32>::zeroed(stream, q_len)?,
            attention_heads_batch: DeviceBuffer::<f32>::zeroed(stream, q_batch_len)?,
            attention_delta: DeviceBuffer::<f32>::zeroed(stream, config.dim)?,
            attention_delta_batch: DeviceBuffer::<f32>::zeroed(stream, attention_normed_batch_len)?,
            attention_residual: DeviceBuffer::<f32>::zeroed(stream, config.dim)?,
            attention_residual_batch: DeviceBuffer::<f32>::zeroed(
                stream,
                attention_normed_batch_len,
            )?,
            ffn: FfnScratch::new(stream, config, max_seq_len)?,
        })
    }

    fn bytes(&self) -> usize {
        (self.attention_normed.len()
            + self.attention_normed_batch.len()
            + self.query.len()
            + self.query_batch.len()
            + self.key.len()
            + self.key_batch.len()
            + self.value.len()
            + self.value_batch.len()
            + self.query_rot.len()
            + self.query_rot_batch.len()
            + self.key_rot.len()
            + self.scores.len()
            + self.attention_heads.len()
            + self.attention_heads_batch.len()
            + self.attention_delta.len()
            + self.attention_delta_batch.len()
            + self.attention_residual.len()
            + self.attention_residual_batch.len())
            * size_of::<f32>()
            + self.ffn.bytes()
    }
}

impl FfnScratch {
    fn new(stream: &Arc<CudaStream>, config: &TextConfig, max_seq_len: usize) -> Result<Self> {
        let hidden_batch_len = max_seq_len
            .checked_mul(config.dim)
            .ok_or_else(|| invalid_data("FFN hidden batch scratch shape overflow"))?;
        let intermediate_batch_len = max_seq_len
            .checked_mul(config.hidden_dim)
            .ok_or_else(|| invalid_data("FFN intermediate batch scratch shape overflow"))?;
        Ok(Self {
            normed: DeviceBuffer::<f32>::zeroed(stream, config.dim)?,
            normed_batch: DeviceBuffer::<f32>::zeroed(stream, hidden_batch_len)?,
            gate: DeviceBuffer::<f32>::zeroed(stream, config.hidden_dim)?,
            gate_batch: DeviceBuffer::<f32>::zeroed(stream, intermediate_batch_len)?,
            up: DeviceBuffer::<f32>::zeroed(stream, config.hidden_dim)?,
            up_batch: DeviceBuffer::<f32>::zeroed(stream, intermediate_batch_len)?,
            activated: DeviceBuffer::<f32>::zeroed(stream, config.hidden_dim)?,
            activated_batch: DeviceBuffer::<f32>::zeroed(stream, intermediate_batch_len)?,
            down: DeviceBuffer::<f32>::zeroed(stream, config.dim)?,
            down_batch: DeviceBuffer::<f32>::zeroed(stream, hidden_batch_len)?,
        })
    }

    fn bytes(&self) -> usize {
        (self.normed.len()
            + self.normed_batch.len()
            + self.gate.len()
            + self.gate_batch.len()
            + self.up.len()
            + self.up_batch.len()
            + self.activated.len()
            + self.activated_batch.len()
            + self.down.len()
            + self.down_batch.len())
            * size_of::<f32>()
    }
}

fn layer_device_weight_bytes(layer: &LayerDeviceWeights) -> usize {
    (layer.attention_norm.len()
        + layer.wq.len()
        + layer.wk.len()
        + layer.wv.len()
        + layer.wo.len()
        + layer.ffn_norm.len()
        + layer.w1.len()
        + layer.w3.len()
        + layer.w2.len())
        * size_of::<Bf16>()
}

fn quantized_matrix_device_bytes(matrix: &DeviceQuantizedI8Matrix) -> usize {
    matrix.values.len() * size_of::<i8>() + matrix.scales.len() * size_of::<f32>()
}

fn quantized_attention_ffn_layer_device_weight_bytes(
    layer: &QuantizedAttentionFfnLayerDeviceWeights,
) -> usize {
    (layer.attention_norm.len() + layer.ffn_norm.len()) * size_of::<Bf16>()
        + quantized_matrix_device_bytes(&layer.wq)
        + quantized_matrix_device_bytes(&layer.wk)
        + quantized_matrix_device_bytes(&layer.wv)
        + quantized_matrix_device_bytes(&layer.wo)
        + quantized_matrix_device_bytes(&layer.w1)
        + quantized_matrix_device_bytes(&layer.w3)
        + quantized_matrix_device_bytes(&layer.w2)
}

fn run_prefill_layer_device_with_cache<L>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    layer: &L,
    rope_freqs: &DeviceBuffer<f32>,
    max_seq_len: usize,
    cache: &mut LayerKvCache,
    scratch: &mut LayerScratch,
    hidden_states: &[DeviceBuffer<f32>],
    next_hidden_states: &mut [DeviceBuffer<f32>],
) -> Result<()>
where
    L: AttentionLayerWeights + FfnLayerWeights,
{
    validate_device_hidden_states(config, hidden_states)?;
    validate_device_hidden_states(config, next_hidden_states)?;
    if next_hidden_states.len() != hidden_states.len() {
        return Err(invalid_data(format!(
            "prefill output has {} hidden states, expected {}",
            next_hidden_states.len(),
            hidden_states.len()
        )));
    }

    for (position, (hidden, output)) in hidden_states
        .iter()
        .zip(next_hidden_states.iter_mut())
        .enumerate()
    {
        run_incremental_layer_token_into(
            stream,
            module,
            config,
            layer,
            rope_freqs,
            max_seq_len,
            position,
            hidden,
            cache,
            output,
            scratch,
        )?;
    }

    Ok(())
}

fn run_prefill_bf16_layer_device_with_cache(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    layer: &LayerDeviceWeights,
    rope_freqs: &DeviceBuffer<f32>,
    max_seq_len: usize,
    prompt_len: usize,
    cache: &mut LayerKvCache,
    scratch: &mut LayerScratch,
    hidden_states: &DeviceBuffer<f32>,
    next_hidden_states: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if prompt_len == 0 || prompt_len > max_seq_len {
        return Err(invalid_data(format!(
            "prefill prompt length {prompt_len} is invalid for max_seq_len {max_seq_len}"
        )));
    }

    let dim = config.dim;
    let q_len = config.n_heads * config.head_dim;
    let kv_len = config.n_kv_heads * config.head_dim;
    let hidden_batch_len = max_seq_len
        .checked_mul(dim)
        .ok_or_else(|| invalid_data("hidden batch shape overflow"))?;
    let q_batch_len = max_seq_len
        .checked_mul(q_len)
        .ok_or_else(|| invalid_data("query batch shape overflow"))?;
    let kv_batch_len = max_seq_len
        .checked_mul(kv_len)
        .ok_or_else(|| invalid_data("KV batch shape overflow"))?;
    let expected_cache_len = max_seq_len * kv_len;

    if hidden_states.len() != hidden_batch_len || next_hidden_states.len() != hidden_batch_len {
        return Err(invalid_data(format!(
            "prefill hidden batch length mismatch: input={}, output={}, expected {}",
            hidden_states.len(),
            next_hidden_states.len(),
            hidden_batch_len
        )));
    }
    if cache.key.len() != expected_cache_len || cache.value.len() != expected_cache_len {
        return Err(invalid_data(format!(
            "KV cache length mismatch: key={}, value={}, expected {}",
            cache.key.len(),
            cache.value.len(),
            expected_cache_len
        )));
    }

    ops::rmsnorm_batched_bf16(
        stream,
        module,
        hidden_states,
        &layer.attention_norm,
        prompt_len,
        dim,
        config.norm_eps,
        &mut scratch.attention_normed_batch,
    )?;
    ops::linear_qkv_batched_bf16(
        stream,
        module,
        &scratch.attention_normed_batch,
        &layer.wq,
        &layer.wk,
        &layer.wv,
        prompt_len,
        dim,
        q_len,
        kv_len,
        &mut scratch.query_batch,
        &mut scratch.key_batch,
        &mut scratch.value_batch,
    )?;

    if scratch.query_batch.len() != q_batch_len
        || scratch.key_batch.len() != kv_batch_len
        || scratch.value_batch.len() != kv_batch_len
        || scratch.query_rot_batch.len() != q_batch_len
    {
        return Err(invalid_data(
            "prefill attention batch scratch length mismatch",
        ));
    }

    ops::prepare_prefill_attention_batch(
        stream,
        module,
        &scratch.query_batch,
        &scratch.key_batch,
        &scratch.value_batch,
        rope_freqs,
        prompt_len,
        q_len,
        kv_len,
        config.head_dim,
        &mut scratch.query_rot_batch,
        &mut cache.key,
        &mut cache.value,
    )?;

    if prompt_len <= ops::SINGLE_QUERY_ATTENTION_MAX_SEQ {
        ops::prefill_causal_attention(
            stream,
            module,
            &scratch.query_rot_batch,
            &cache.key,
            &cache.value,
            prompt_len,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scratch.attention_heads_batch,
        )?;
    } else {
        for position in 0..prompt_len {
            ops::attention_scores_from_matrix_row(
                stream,
                module,
                &scratch.query_rot_batch,
                &cache.key,
                position,
                position + 1,
                max_seq_len,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                &mut scratch.scores,
            )?;
            ops::softmax_value_to_matrix_row(
                stream,
                module,
                &scratch.scores,
                &cache.value,
                position + 1,
                max_seq_len,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                position,
                &mut scratch.attention_heads_batch,
            )?;
        }
    }

    ops::linear_batched_bf16(
        stream,
        module,
        &scratch.attention_heads_batch,
        &layer.wo,
        prompt_len,
        q_len,
        dim,
        &mut scratch.attention_delta_batch,
    )?;
    ops::add_prefix(
        stream,
        module,
        hidden_states,
        &scratch.attention_delta_batch,
        prompt_len
            .checked_mul(dim)
            .ok_or_else(|| invalid_data("prefill attention residual shape overflow"))?,
        &mut scratch.attention_residual_batch,
    )?;

    run_ffn_batched_bf16_from_hidden_into(
        stream,
        module,
        config,
        &scratch.attention_residual_batch,
        &layer.ffn_norm,
        &layer.w1,
        &layer.w3,
        &layer.w2,
        prompt_len,
        next_hidden_states,
        &mut scratch.ffn,
    )
}

fn run_prefill_quantized_layer_device_with_cache(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    layer: &QuantizedAttentionFfnLayerDeviceWeights,
    rope_freqs: &DeviceBuffer<f32>,
    max_seq_len: usize,
    prompt_len: usize,
    cache: &mut LayerKvCache,
    scratch: &mut LayerScratch,
    hidden_states: &DeviceBuffer<f32>,
    next_hidden_states: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if prompt_len == 0 || prompt_len > max_seq_len {
        return Err(invalid_data(format!(
            "prefill prompt length {prompt_len} is invalid for max_seq_len {max_seq_len}"
        )));
    }

    let dim = config.dim;
    let q_len = config.n_heads * config.head_dim;
    let kv_len = config.n_kv_heads * config.head_dim;
    let hidden_batch_len = max_seq_len
        .checked_mul(dim)
        .ok_or_else(|| invalid_data("hidden batch shape overflow"))?;
    let q_batch_len = max_seq_len
        .checked_mul(q_len)
        .ok_or_else(|| invalid_data("query batch shape overflow"))?;
    let kv_batch_len = max_seq_len
        .checked_mul(kv_len)
        .ok_or_else(|| invalid_data("KV batch shape overflow"))?;
    let expected_cache_len = max_seq_len * kv_len;

    if hidden_states.len() != hidden_batch_len || next_hidden_states.len() != hidden_batch_len {
        return Err(invalid_data(format!(
            "prefill hidden batch length mismatch: input={}, output={}, expected {}",
            hidden_states.len(),
            next_hidden_states.len(),
            hidden_batch_len
        )));
    }
    if cache.key.len() != expected_cache_len || cache.value.len() != expected_cache_len {
        return Err(invalid_data(format!(
            "KV cache length mismatch: key={}, value={}, expected {}",
            cache.key.len(),
            cache.value.len(),
            expected_cache_len
        )));
    }

    ops::rmsnorm_batched_bf16(
        stream,
        module,
        hidden_states,
        &layer.attention_norm,
        prompt_len,
        dim,
        config.norm_eps,
        &mut scratch.attention_normed_batch,
    )?;
    ops::linear_qkv_batched_i8_scaled(
        stream,
        module,
        &scratch.attention_normed_batch,
        &layer.wq,
        &layer.wk,
        &layer.wv,
        prompt_len,
        dim,
        q_len,
        kv_len,
        &mut scratch.query_batch,
        &mut scratch.key_batch,
        &mut scratch.value_batch,
    )?;

    if scratch.query_batch.len() != q_batch_len
        || scratch.key_batch.len() != kv_batch_len
        || scratch.value_batch.len() != kv_batch_len
        || scratch.query_rot_batch.len() != q_batch_len
    {
        return Err(invalid_data(
            "prefill attention batch scratch length mismatch",
        ));
    }

    ops::prepare_prefill_attention_batch(
        stream,
        module,
        &scratch.query_batch,
        &scratch.key_batch,
        &scratch.value_batch,
        rope_freqs,
        prompt_len,
        q_len,
        kv_len,
        config.head_dim,
        &mut scratch.query_rot_batch,
        &mut cache.key,
        &mut cache.value,
    )?;

    if prompt_len <= ops::SINGLE_QUERY_ATTENTION_MAX_SEQ {
        ops::prefill_causal_attention(
            stream,
            module,
            &scratch.query_rot_batch,
            &cache.key,
            &cache.value,
            prompt_len,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scratch.attention_heads_batch,
        )?;
    } else {
        for position in 0..prompt_len {
            ops::attention_scores_from_matrix_row(
                stream,
                module,
                &scratch.query_rot_batch,
                &cache.key,
                position,
                position + 1,
                max_seq_len,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                &mut scratch.scores,
            )?;
            ops::softmax_value_to_matrix_row(
                stream,
                module,
                &scratch.scores,
                &cache.value,
                position + 1,
                max_seq_len,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                position,
                &mut scratch.attention_heads_batch,
            )?;
        }
    }

    ops::linear_batched_i8_scaled(
        stream,
        module,
        &scratch.attention_heads_batch,
        &layer.wo,
        prompt_len,
        q_len,
        dim,
        &mut scratch.attention_delta_batch,
    )?;
    ops::add_prefix(
        stream,
        module,
        hidden_states,
        &scratch.attention_delta_batch,
        prompt_len
            .checked_mul(dim)
            .ok_or_else(|| invalid_data("prefill attention residual shape overflow"))?,
        &mut scratch.attention_residual_batch,
    )?;

    run_ffn_batched_i8_from_hidden_into(
        stream,
        module,
        config,
        &scratch.attention_residual_batch,
        &layer.ffn_norm,
        &layer.w1,
        &layer.w3,
        &layer.w2,
        prompt_len,
        next_hidden_states,
        &mut scratch.ffn,
    )
}

fn run_incremental_token_forward_into<L>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    rope_freqs: &DeviceBuffer<f32>,
    embedding_weight: &DeviceBuffer<Bf16>,
    layers: &[L],
    caches: &mut [LayerKvCache],
    scratch: &mut RuntimeScratch,
    max_seq_len: usize,
    position: usize,
    token_id: u32,
) -> Result<bool>
where
    L: AttentionLayerWeights + FfnLayerWeights,
{
    if position >= max_seq_len {
        return Err(invalid_data(format!(
            "position {position} exceeds max sequence length {max_seq_len}"
        )));
    }
    if caches.len() != config.n_layers {
        return Err(invalid_data(format!(
            "KV cache has {} layers, expected {}",
            caches.len(),
            config.n_layers
        )));
    }
    if layers.len() != config.n_layers {
        return Err(invalid_data(format!(
            "resident layer weights have {} layers, expected {}",
            layers.len(),
            config.n_layers
        )));
    }
    validate_token(config, token_id)?;

    ops::embedding(
        stream,
        module,
        embedding_weight,
        token_id,
        config.dim,
        &mut scratch.hidden_a,
    )?;

    let mut hidden_is_a = true;
    for (layer, cache) in layers.iter().zip(caches.iter_mut()) {
        if hidden_is_a {
            run_incremental_layer_token_into(
                stream,
                module,
                config,
                layer,
                rope_freqs,
                max_seq_len,
                position,
                &scratch.hidden_a,
                cache,
                &mut scratch.hidden_b,
                &mut scratch.layer,
            )?;
        } else {
            run_incremental_layer_token_into(
                stream,
                module,
                config,
                layer,
                rope_freqs,
                max_seq_len,
                position,
                &scratch.hidden_b,
                cache,
                &mut scratch.hidden_a,
                &mut scratch.layer,
            )?;
        }
        hidden_is_a = !hidden_is_a;
    }

    Ok(hidden_is_a)
}

fn run_incremental_layer_token_into<L>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    layer: &L,
    rope_freqs: &DeviceBuffer<f32>,
    max_seq_len: usize,
    position: usize,
    hidden: &DeviceBuffer<f32>,
    cache: &mut LayerKvCache,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut LayerScratch,
) -> Result<()>
where
    L: AttentionLayerWeights + FfnLayerWeights,
{
    let kv_len = config.n_kv_heads * config.head_dim;
    let expected_cache_len = max_seq_len * kv_len;
    if cache.key.len() != expected_cache_len || cache.value.len() != expected_cache_len {
        return Err(invalid_data(format!(
            "KV cache length mismatch: key={}, value={}, expected {}",
            cache.key.len(),
            cache.value.len(),
            expected_cache_len
        )));
    }
    if output.len() != config.dim {
        return Err(invalid_data(format!(
            "incremental output has length {}, expected {}",
            output.len(),
            config.dim
        )));
    }

    ops::rmsnorm(
        stream,
        module,
        hidden,
        layer.attention_norm(),
        config.norm_eps,
        &mut scratch.attention_normed,
    )?;
    ops::linear(
        stream,
        module,
        &scratch.attention_normed,
        layer.wq(),
        &mut scratch.query,
    )?;
    ops::linear(
        stream,
        module,
        &scratch.attention_normed,
        layer.wk(),
        &mut scratch.key,
    )?;
    ops::linear(
        stream,
        module,
        &scratch.attention_normed,
        layer.wv(),
        &mut scratch.value,
    )?;
    ops::apply_rope(
        stream,
        module,
        &scratch.query,
        rope_freqs,
        position,
        config.head_dim,
        &mut scratch.query_rot,
    )?;
    ops::apply_rope(
        stream,
        module,
        &scratch.key,
        rope_freqs,
        position,
        config.head_dim,
        &mut scratch.key_rot,
    )?;
    ops::write_kv_cache(
        stream,
        module,
        &scratch.key_rot,
        position,
        max_seq_len,
        config.n_kv_heads,
        config.head_dim,
        &mut cache.key,
    )?;
    ops::write_kv_cache(
        stream,
        module,
        &scratch.value,
        position,
        max_seq_len,
        config.n_kv_heads,
        config.head_dim,
        &mut cache.value,
    )?;
    if position < ops::SINGLE_QUERY_ATTENTION_MAX_SEQ {
        ops::single_query_attention(
            stream,
            module,
            &scratch.query_rot,
            &cache.key,
            &cache.value,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scratch.attention_heads,
        )?;
    } else {
        ops::attention_scores(
            stream,
            module,
            &scratch.query_rot,
            &cache.key,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scratch.scores,
        )?;
        ops::softmax_value(
            stream,
            module,
            &scratch.scores,
            &cache.value,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scratch.attention_heads,
        )?;
    }
    ops::linear(
        stream,
        module,
        &scratch.attention_heads,
        layer.wo(),
        &mut scratch.attention_delta,
    )?;
    ops::add(
        stream,
        module,
        hidden,
        &scratch.attention_delta,
        &mut scratch.attention_residual,
    )?;

    run_ffn_from_hidden_into(
        stream,
        module,
        config,
        &scratch.attention_residual,
        layer.ffn_norm(),
        layer.w1(),
        layer.w3(),
        layer.w2(),
        output,
        &mut scratch.ffn,
    )
}

fn run_incremental_bf16_token_forward_into(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    rope_freqs: &DeviceBuffer<f32>,
    embedding_weight: &DeviceBuffer<Bf16>,
    layers: &[LayerDeviceWeights],
    caches: &mut [LayerKvCache],
    scratch: &mut RuntimeScratch,
    max_seq_len: usize,
    position: usize,
    token_id: u32,
) -> Result<bool> {
    if position >= max_seq_len {
        return Err(invalid_data(format!(
            "position {position} exceeds max sequence length {max_seq_len}"
        )));
    }
    if caches.len() != config.n_layers {
        return Err(invalid_data(format!(
            "KV cache has {} layers, expected {}",
            caches.len(),
            config.n_layers
        )));
    }
    if layers.len() != config.n_layers {
        return Err(invalid_data(format!(
            "resident layer weights have {} layers, expected {}",
            layers.len(),
            config.n_layers
        )));
    }
    validate_token(config, token_id)?;

    ops::embedding(
        stream,
        module,
        embedding_weight,
        token_id,
        config.dim,
        &mut scratch.hidden_a,
    )?;

    let mut hidden_is_a = true;
    for (layer, cache) in layers.iter().zip(caches.iter_mut()) {
        if hidden_is_a {
            run_incremental_bf16_layer_token_into(
                stream,
                module,
                config,
                layer,
                rope_freqs,
                max_seq_len,
                position,
                &scratch.hidden_a,
                cache,
                &mut scratch.hidden_b,
                &mut scratch.layer,
            )?;
        } else {
            run_incremental_bf16_layer_token_into(
                stream,
                module,
                config,
                layer,
                rope_freqs,
                max_seq_len,
                position,
                &scratch.hidden_b,
                cache,
                &mut scratch.hidden_a,
                &mut scratch.layer,
            )?;
        }
        hidden_is_a = !hidden_is_a;
    }

    Ok(hidden_is_a)
}

fn run_incremental_bf16_layer_token_into(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    layer: &LayerDeviceWeights,
    rope_freqs: &DeviceBuffer<f32>,
    max_seq_len: usize,
    position: usize,
    hidden: &DeviceBuffer<f32>,
    cache: &mut LayerKvCache,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut LayerScratch,
) -> Result<()> {
    let kv_len = config.n_kv_heads * config.head_dim;
    let expected_cache_len = max_seq_len * kv_len;
    if cache.key.len() != expected_cache_len || cache.value.len() != expected_cache_len {
        return Err(invalid_data(format!(
            "KV cache length mismatch: key={}, value={}, expected {}",
            cache.key.len(),
            cache.value.len(),
            expected_cache_len
        )));
    }
    if output.len() != config.dim {
        return Err(invalid_data(format!(
            "incremental output has length {}, expected {}",
            output.len(),
            config.dim
        )));
    }

    ops::rmsnorm(
        stream,
        module,
        hidden,
        &layer.attention_norm,
        config.norm_eps,
        &mut scratch.attention_normed,
    )?;
    ops::linear_triple_bf16(
        stream,
        module,
        &scratch.attention_normed,
        &layer.wq,
        &layer.wk,
        &layer.wv,
        &mut scratch.query,
        &mut scratch.key,
        &mut scratch.value,
    )?;
    ops::prepare_incremental_attention(
        stream,
        module,
        &scratch.query,
        &scratch.key,
        &scratch.value,
        rope_freqs,
        position,
        max_seq_len,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        &mut scratch.query_rot,
        &mut cache.key,
        &mut cache.value,
    )?;
    if position < ops::SINGLE_QUERY_ATTENTION_MAX_SEQ {
        ops::single_query_attention(
            stream,
            module,
            &scratch.query_rot,
            &cache.key,
            &cache.value,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scratch.attention_heads,
        )?;
    } else {
        ops::attention_scores(
            stream,
            module,
            &scratch.query_rot,
            &cache.key,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scratch.scores,
        )?;
        ops::softmax_value(
            stream,
            module,
            &scratch.scores,
            &cache.value,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scratch.attention_heads,
        )?;
    }
    ops::linear_residual_bf16(
        stream,
        module,
        &scratch.attention_heads,
        &layer.wo,
        hidden,
        &mut scratch.attention_residual,
    )?;

    run_ffn_bf16_from_hidden_into(
        stream,
        module,
        config,
        &scratch.attention_residual,
        &layer.ffn_norm,
        &layer.w1,
        &layer.w3,
        &layer.w2,
        output,
        &mut scratch.ffn,
    )
}

fn logits_from_hidden_device_into<W>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    hidden: &DeviceBuffer<f32>,
    norm_weight: &DeviceBuffer<Bf16>,
    output_weight: &W,
    normed: &mut DeviceBuffer<f32>,
    logits: &mut DeviceBuffer<f32>,
) -> Result<()>
where
    W: ops::CudaLinearWeight,
{
    if normed.len() != config.dim {
        return Err(invalid_data(format!(
            "logit norm scratch has length {}, expected {}",
            normed.len(),
            config.dim
        )));
    }
    if logits.len() != config.vocab_size {
        return Err(invalid_data(format!(
            "logit scratch has length {}, expected {}",
            logits.len(),
            config.vocab_size
        )));
    }

    ops::rmsnorm(stream, module, hidden, norm_weight, config.norm_eps, normed)?;
    ops::linear(stream, module, normed, output_weight, logits)?;
    Ok(())
}

fn output_top1_from_hidden_bf16_into(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    top1_plan: Bf16Top1Plan,
    hidden: &DeviceBuffer<f32>,
    norm_weight: &DeviceBuffer<Bf16>,
    output_weight: &DeviceBuffer<Bf16>,
    normed: &mut DeviceBuffer<f32>,
    partial_tokens: &mut DeviceBuffer<u32>,
    partial_logits: &mut DeviceBuffer<f32>,
    packed_out: &mut DeviceBuffer<u64>,
) -> Result<(u32, f32)> {
    if normed.len() != config.dim {
        return Err(invalid_data(format!(
            "logit norm scratch has length {}, expected {}",
            normed.len(),
            config.dim
        )));
    }
    let expected_weight_len = config
        .vocab_size
        .checked_mul(config.dim)
        .ok_or_else(|| invalid_data("output projection weight shape overflow"))?;
    if output_weight.len() != expected_weight_len {
        return Err(invalid_data(format!(
            "output projection weight has length {}, expected {}",
            output_weight.len(),
            expected_weight_len
        )));
    }

    ops::rmsnorm(stream, module, hidden, norm_weight, config.norm_eps, normed)?;
    match top1_plan {
        Bf16Top1Plan::Rows1 => ops::linear_top1_bf16_rows1(
            stream,
            module,
            normed,
            output_weight,
            partial_tokens,
            partial_logits,
            packed_out,
        )?,
        Bf16Top1Plan::Rows2 => ops::linear_top1_bf16_rows2(
            stream,
            module,
            normed,
            output_weight,
            partial_tokens,
            partial_logits,
            packed_out,
        )?,
        Bf16Top1Plan::Rows4 => ops::linear_top1_bf16(
            stream,
            module,
            normed,
            output_weight,
            partial_tokens,
            partial_logits,
            packed_out,
        )?,
        Bf16Top1Plan::Rows8 => ops::linear_top1_bf16_rows8(
            stream,
            module,
            normed,
            output_weight,
            partial_tokens,
            partial_logits,
            packed_out,
        )?,
    }
    let packed = packed_out.to_host_vec(stream)?[0];
    Ok(unpack_packed_top_logit(packed))
}

fn output_top1_from_hidden_i8_into(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    hidden: &DeviceBuffer<f32>,
    norm_weight: &DeviceBuffer<Bf16>,
    output_weight: &DeviceQuantizedI8Matrix,
    normed: &mut DeviceBuffer<f32>,
    partial_tokens: &mut DeviceBuffer<u32>,
    partial_logits: &mut DeviceBuffer<f32>,
    packed_out: &mut DeviceBuffer<u64>,
) -> Result<(u32, f32)> {
    if normed.len() != config.dim {
        return Err(invalid_data(format!(
            "logit norm scratch has length {}, expected {}",
            normed.len(),
            config.dim
        )));
    }
    if output_weight.rows != config.vocab_size || output_weight.cols != config.dim {
        return Err(invalid_data(format!(
            "scaled-i8 output projection shape is [{}, {}], expected [{}, {}]",
            output_weight.rows, output_weight.cols, config.vocab_size, config.dim
        )));
    }

    ops::rmsnorm(stream, module, hidden, norm_weight, config.norm_eps, normed)?;
    ops::linear_top1_i8_scaled(
        stream,
        module,
        normed,
        output_weight,
        partial_tokens,
        partial_logits,
        packed_out,
    )?;
    let packed = packed_out.to_host_vec(stream)?[0];
    Ok(unpack_packed_top_logit(packed))
}

fn run_prompt_layer_from_hidden_states(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    layer: &LayerDeviceWeights,
    rope_freqs: &DeviceBuffer<f32>,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    validate_hidden_states(config, hidden_states)?;

    let max_seq_len = hidden_states.len();
    let kv_len = config.n_kv_heads * config.head_dim;
    let q_len = config.n_heads * config.head_dim;
    let mut key_cache = DeviceBuffer::<f32>::zeroed(stream, max_seq_len * kv_len)?;
    let mut value_cache = DeviceBuffer::<f32>::zeroed(stream, max_seq_len * kv_len)?;
    let mut scores = DeviceBuffer::<f32>::zeroed(stream, config.n_heads * max_seq_len)?;
    let mut attention_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut query = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut key = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut value = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut query_rot = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut key_rot = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut attention_heads = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut attention_delta = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut attention_residual = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut next_hidden_states = Vec::with_capacity(hidden_states.len());

    for (position, hidden_host) in hidden_states.iter().enumerate() {
        let hidden = DeviceBuffer::from_host(stream, hidden_host)?;
        ops::rmsnorm(
            stream,
            module,
            &hidden,
            &layer.attention_norm,
            config.norm_eps,
            &mut attention_normed,
        )?;
        ops::linear(stream, module, &attention_normed, &layer.wq, &mut query)?;
        ops::linear(stream, module, &attention_normed, &layer.wk, &mut key)?;
        ops::linear(stream, module, &attention_normed, &layer.wv, &mut value)?;
        ops::apply_rope(
            stream,
            module,
            &query,
            rope_freqs,
            position,
            config.head_dim,
            &mut query_rot,
        )?;
        ops::apply_rope(
            stream,
            module,
            &key,
            rope_freqs,
            position,
            config.head_dim,
            &mut key_rot,
        )?;
        ops::write_kv_cache(
            stream,
            module,
            &key_rot,
            position,
            max_seq_len,
            config.n_kv_heads,
            config.head_dim,
            &mut key_cache,
        )?;
        ops::write_kv_cache(
            stream,
            module,
            &value,
            position,
            max_seq_len,
            config.n_kv_heads,
            config.head_dim,
            &mut value_cache,
        )?;
        ops::attention_scores(
            stream,
            module,
            &query_rot,
            &key_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scores,
        )?;
        ops::softmax_value(
            stream,
            module,
            &scores,
            &value_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut attention_heads,
        )?;
        ops::linear(
            stream,
            module,
            &attention_heads,
            &layer.wo,
            &mut attention_delta,
        )?;
        ops::add(
            stream,
            module,
            &hidden,
            &attention_delta,
            &mut attention_residual,
        )?;

        let output = run_ffn_from_hidden(
            stream,
            module,
            config,
            &attention_residual,
            &layer.ffn_norm,
            &layer.w1,
            &layer.w3,
            &layer.w2,
        )?;
        next_hidden_states.push(output.to_host_vec(stream)?);
    }

    Ok(next_hidden_states)
}

fn run_prompt_layer_quantized_ffn_from_hidden_states(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    layer: &QuantizedFfnLayerDeviceWeights,
    rope_freqs: &DeviceBuffer<f32>,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    validate_hidden_states(config, hidden_states)?;

    let max_seq_len = hidden_states.len();
    let kv_len = config.n_kv_heads * config.head_dim;
    let q_len = config.n_heads * config.head_dim;
    let mut key_cache = DeviceBuffer::<f32>::zeroed(stream, max_seq_len * kv_len)?;
    let mut value_cache = DeviceBuffer::<f32>::zeroed(stream, max_seq_len * kv_len)?;
    let mut scores = DeviceBuffer::<f32>::zeroed(stream, config.n_heads * max_seq_len)?;
    let mut attention_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut query = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut key = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut value = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut query_rot = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut key_rot = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut attention_heads = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut attention_delta = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut attention_residual = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut next_hidden_states = Vec::with_capacity(hidden_states.len());

    for (position, hidden_host) in hidden_states.iter().enumerate() {
        let hidden = DeviceBuffer::from_host(stream, hidden_host)?;
        ops::rmsnorm(
            stream,
            module,
            &hidden,
            &layer.attention_norm,
            config.norm_eps,
            &mut attention_normed,
        )?;
        ops::linear(stream, module, &attention_normed, &layer.wq, &mut query)?;
        ops::linear(stream, module, &attention_normed, &layer.wk, &mut key)?;
        ops::linear(stream, module, &attention_normed, &layer.wv, &mut value)?;
        ops::apply_rope(
            stream,
            module,
            &query,
            rope_freqs,
            position,
            config.head_dim,
            &mut query_rot,
        )?;
        ops::apply_rope(
            stream,
            module,
            &key,
            rope_freqs,
            position,
            config.head_dim,
            &mut key_rot,
        )?;
        ops::write_kv_cache(
            stream,
            module,
            &key_rot,
            position,
            max_seq_len,
            config.n_kv_heads,
            config.head_dim,
            &mut key_cache,
        )?;
        ops::write_kv_cache(
            stream,
            module,
            &value,
            position,
            max_seq_len,
            config.n_kv_heads,
            config.head_dim,
            &mut value_cache,
        )?;
        ops::attention_scores(
            stream,
            module,
            &query_rot,
            &key_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scores,
        )?;
        ops::softmax_value(
            stream,
            module,
            &scores,
            &value_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut attention_heads,
        )?;
        ops::linear(
            stream,
            module,
            &attention_heads,
            &layer.wo,
            &mut attention_delta,
        )?;
        ops::add(
            stream,
            module,
            &hidden,
            &attention_delta,
            &mut attention_residual,
        )?;

        let output = run_ffn_from_hidden(
            stream,
            module,
            config,
            &attention_residual,
            &layer.ffn_norm,
            &layer.w1,
            &layer.w3,
            &layer.w2,
        )?;
        next_hidden_states.push(output.to_host_vec(stream)?);
    }

    Ok(next_hidden_states)
}

fn run_prompt_layer_quantized_attention_from_hidden_states(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    layer: &QuantizedAttentionLayerDeviceWeights,
    rope_freqs: &DeviceBuffer<f32>,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    validate_hidden_states(config, hidden_states)?;

    let max_seq_len = hidden_states.len();
    let kv_len = config.n_kv_heads * config.head_dim;
    let q_len = config.n_heads * config.head_dim;
    let mut key_cache = DeviceBuffer::<f32>::zeroed(stream, max_seq_len * kv_len)?;
    let mut value_cache = DeviceBuffer::<f32>::zeroed(stream, max_seq_len * kv_len)?;
    let mut scores = DeviceBuffer::<f32>::zeroed(stream, config.n_heads * max_seq_len)?;
    let mut attention_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut query = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut key = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut value = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut query_rot = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut key_rot = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut attention_heads = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut attention_delta = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut attention_residual = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut next_hidden_states = Vec::with_capacity(hidden_states.len());

    for (position, hidden_host) in hidden_states.iter().enumerate() {
        let hidden = DeviceBuffer::from_host(stream, hidden_host)?;
        ops::rmsnorm(
            stream,
            module,
            &hidden,
            &layer.attention_norm,
            config.norm_eps,
            &mut attention_normed,
        )?;
        ops::linear(stream, module, &attention_normed, &layer.wq, &mut query)?;
        ops::linear(stream, module, &attention_normed, &layer.wk, &mut key)?;
        ops::linear(stream, module, &attention_normed, &layer.wv, &mut value)?;
        ops::apply_rope(
            stream,
            module,
            &query,
            rope_freqs,
            position,
            config.head_dim,
            &mut query_rot,
        )?;
        ops::apply_rope(
            stream,
            module,
            &key,
            rope_freqs,
            position,
            config.head_dim,
            &mut key_rot,
        )?;
        ops::write_kv_cache(
            stream,
            module,
            &key_rot,
            position,
            max_seq_len,
            config.n_kv_heads,
            config.head_dim,
            &mut key_cache,
        )?;
        ops::write_kv_cache(
            stream,
            module,
            &value,
            position,
            max_seq_len,
            config.n_kv_heads,
            config.head_dim,
            &mut value_cache,
        )?;
        ops::attention_scores(
            stream,
            module,
            &query_rot,
            &key_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scores,
        )?;
        ops::softmax_value(
            stream,
            module,
            &scores,
            &value_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut attention_heads,
        )?;
        ops::linear(
            stream,
            module,
            &attention_heads,
            &layer.wo,
            &mut attention_delta,
        )?;
        ops::add(
            stream,
            module,
            &hidden,
            &attention_delta,
            &mut attention_residual,
        )?;

        let output = run_ffn_from_hidden(
            stream,
            module,
            config,
            &attention_residual,
            &layer.ffn_norm,
            &layer.w1,
            &layer.w3,
            &layer.w2,
        )?;
        next_hidden_states.push(output.to_host_vec(stream)?);
    }

    Ok(next_hidden_states)
}

fn run_prompt_layer_quantized_attention_ffn_from_hidden_states(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    layer: &QuantizedAttentionFfnLayerDeviceWeights,
    rope_freqs: &DeviceBuffer<f32>,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    validate_hidden_states(config, hidden_states)?;

    let max_seq_len = hidden_states.len();
    let kv_len = config.n_kv_heads * config.head_dim;
    let q_len = config.n_heads * config.head_dim;
    let mut key_cache = DeviceBuffer::<f32>::zeroed(stream, max_seq_len * kv_len)?;
    let mut value_cache = DeviceBuffer::<f32>::zeroed(stream, max_seq_len * kv_len)?;
    let mut scores = DeviceBuffer::<f32>::zeroed(stream, config.n_heads * max_seq_len)?;
    let mut attention_normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut query = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut key = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut value = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut query_rot = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut key_rot = DeviceBuffer::<f32>::zeroed(stream, kv_len)?;
    let mut attention_heads = DeviceBuffer::<f32>::zeroed(stream, q_len)?;
    let mut attention_delta = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut attention_residual = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut next_hidden_states = Vec::with_capacity(hidden_states.len());

    for (position, hidden_host) in hidden_states.iter().enumerate() {
        let hidden = DeviceBuffer::from_host(stream, hidden_host)?;
        ops::rmsnorm(
            stream,
            module,
            &hidden,
            &layer.attention_norm,
            config.norm_eps,
            &mut attention_normed,
        )?;
        ops::linear(stream, module, &attention_normed, &layer.wq, &mut query)?;
        ops::linear(stream, module, &attention_normed, &layer.wk, &mut key)?;
        ops::linear(stream, module, &attention_normed, &layer.wv, &mut value)?;
        ops::apply_rope(
            stream,
            module,
            &query,
            rope_freqs,
            position,
            config.head_dim,
            &mut query_rot,
        )?;
        ops::apply_rope(
            stream,
            module,
            &key,
            rope_freqs,
            position,
            config.head_dim,
            &mut key_rot,
        )?;
        ops::write_kv_cache(
            stream,
            module,
            &key_rot,
            position,
            max_seq_len,
            config.n_kv_heads,
            config.head_dim,
            &mut key_cache,
        )?;
        ops::write_kv_cache(
            stream,
            module,
            &value,
            position,
            max_seq_len,
            config.n_kv_heads,
            config.head_dim,
            &mut value_cache,
        )?;
        ops::attention_scores(
            stream,
            module,
            &query_rot,
            &key_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut scores,
        )?;
        ops::softmax_value(
            stream,
            module,
            &scores,
            &value_cache,
            position + 1,
            max_seq_len,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            &mut attention_heads,
        )?;
        ops::linear(
            stream,
            module,
            &attention_heads,
            &layer.wo,
            &mut attention_delta,
        )?;
        ops::add(
            stream,
            module,
            &hidden,
            &attention_delta,
            &mut attention_residual,
        )?;

        let output = run_ffn_from_hidden(
            stream,
            module,
            config,
            &attention_residual,
            &layer.ffn_norm,
            &layer.w1,
            &layer.w3,
            &layer.w2,
        )?;
        next_hidden_states.push(output.to_host_vec(stream)?);
    }

    Ok(next_hidden_states)
}

fn run_ffn_from_hidden<W1, W3, W2>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    hidden: &DeviceBuffer<f32>,
    ffn_norm: &DeviceBuffer<Bf16>,
    w1: &W1,
    w3: &W3,
    w2: &W2,
) -> Result<DeviceBuffer<f32>>
where
    W1: ops::CudaLinearWeight,
    W3: ops::CudaLinearWeight,
    W2: ops::CudaLinearWeight,
{
    let mut normed = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut gate = DeviceBuffer::<f32>::zeroed(stream, config.hidden_dim)?;
    let mut up = DeviceBuffer::<f32>::zeroed(stream, config.hidden_dim)?;
    let mut activated = DeviceBuffer::<f32>::zeroed(stream, config.hidden_dim)?;
    let mut down = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;
    let mut output = DeviceBuffer::<f32>::zeroed(stream, config.dim)?;

    ops::rmsnorm(
        stream,
        module,
        hidden,
        ffn_norm,
        config.norm_eps,
        &mut normed,
    )?;
    ops::linear(stream, module, &normed, w1, &mut gate)?;
    ops::linear(stream, module, &normed, w3, &mut up)?;
    ops::silu_mul(stream, module, &gate, &up, &mut activated)?;
    ops::linear(stream, module, &activated, w2, &mut down)?;
    ops::add(stream, module, hidden, &down, &mut output)?;
    stream.synchronize()?;

    Ok(output)
}

fn run_ffn_batched_bf16_from_hidden_into(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    hidden: &DeviceBuffer<f32>,
    ffn_norm: &DeviceBuffer<Bf16>,
    w1: &DeviceBuffer<Bf16>,
    w3: &DeviceBuffer<Bf16>,
    w2: &DeviceBuffer<Bf16>,
    batch: usize,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut FfnScratch,
) -> Result<()> {
    let hidden_batch_len = batch
        .checked_mul(config.dim)
        .ok_or_else(|| invalid_data("batched FFN hidden shape overflow"))?;
    let intermediate_batch_len = batch
        .checked_mul(config.hidden_dim)
        .ok_or_else(|| invalid_data("batched FFN intermediate shape overflow"))?;

    if hidden.len() < hidden_batch_len || output.len() < hidden_batch_len {
        return Err(invalid_data(format!(
            "batched FFN hidden buffer too short: input={}, output={}, expected at least {}",
            hidden.len(),
            output.len(),
            hidden_batch_len
        )));
    }
    if scratch.gate_batch.len() < intermediate_batch_len
        || scratch.up_batch.len() < intermediate_batch_len
        || scratch.activated_batch.len() < intermediate_batch_len
        || scratch.down_batch.len() < hidden_batch_len
    {
        return Err(invalid_data("batched FFN scratch buffer too short"));
    }

    ops::rmsnorm_batched_bf16(
        stream,
        module,
        hidden,
        ffn_norm,
        batch,
        config.dim,
        config.norm_eps,
        &mut scratch.normed_batch,
    )?;
    ops::linear_batched_bf16(
        stream,
        module,
        &scratch.normed_batch,
        w1,
        batch,
        config.dim,
        config.hidden_dim,
        &mut scratch.gate_batch,
    )?;
    ops::linear_batched_bf16(
        stream,
        module,
        &scratch.normed_batch,
        w3,
        batch,
        config.dim,
        config.hidden_dim,
        &mut scratch.up_batch,
    )?;
    ops::silu_mul_prefix(
        stream,
        module,
        &scratch.gate_batch,
        &scratch.up_batch,
        intermediate_batch_len,
        &mut scratch.activated_batch,
    )?;
    ops::linear_batched_bf16(
        stream,
        module,
        &scratch.activated_batch,
        w2,
        batch,
        config.hidden_dim,
        config.dim,
        &mut scratch.down_batch,
    )?;
    ops::add_prefix(
        stream,
        module,
        hidden,
        &scratch.down_batch,
        hidden_batch_len,
        output,
    )?;
    Ok(())
}

fn run_ffn_batched_i8_from_hidden_into(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    hidden: &DeviceBuffer<f32>,
    ffn_norm: &DeviceBuffer<Bf16>,
    w1: &DeviceQuantizedI8Matrix,
    w3: &DeviceQuantizedI8Matrix,
    w2: &DeviceQuantizedI8Matrix,
    batch: usize,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut FfnScratch,
) -> Result<()> {
    let hidden_batch_len = batch
        .checked_mul(config.dim)
        .ok_or_else(|| invalid_data("batched quantized FFN hidden shape overflow"))?;
    let intermediate_batch_len = batch
        .checked_mul(config.hidden_dim)
        .ok_or_else(|| invalid_data("batched quantized FFN intermediate shape overflow"))?;

    if hidden.len() < hidden_batch_len || output.len() < hidden_batch_len {
        return Err(invalid_data(format!(
            "batched quantized FFN hidden buffer too short: input={}, output={}, expected at least {}",
            hidden.len(),
            output.len(),
            hidden_batch_len
        )));
    }
    if scratch.gate_batch.len() < intermediate_batch_len
        || scratch.up_batch.len() < intermediate_batch_len
        || scratch.activated_batch.len() < intermediate_batch_len
        || scratch.down_batch.len() < hidden_batch_len
    {
        return Err(invalid_data(
            "batched quantized FFN scratch buffer too short",
        ));
    }

    ops::rmsnorm_batched_bf16(
        stream,
        module,
        hidden,
        ffn_norm,
        batch,
        config.dim,
        config.norm_eps,
        &mut scratch.normed_batch,
    )?;
    ops::linear_batched_i8_scaled(
        stream,
        module,
        &scratch.normed_batch,
        w1,
        batch,
        config.dim,
        config.hidden_dim,
        &mut scratch.gate_batch,
    )?;
    ops::linear_batched_i8_scaled(
        stream,
        module,
        &scratch.normed_batch,
        w3,
        batch,
        config.dim,
        config.hidden_dim,
        &mut scratch.up_batch,
    )?;
    ops::silu_mul_prefix(
        stream,
        module,
        &scratch.gate_batch,
        &scratch.up_batch,
        intermediate_batch_len,
        &mut scratch.activated_batch,
    )?;
    ops::linear_batched_i8_scaled(
        stream,
        module,
        &scratch.activated_batch,
        w2,
        batch,
        config.hidden_dim,
        config.dim,
        &mut scratch.down_batch,
    )?;
    ops::add_prefix(
        stream,
        module,
        hidden,
        &scratch.down_batch,
        hidden_batch_len,
        output,
    )?;
    Ok(())
}

fn run_ffn_bf16_from_hidden_into(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    hidden: &DeviceBuffer<f32>,
    ffn_norm: &DeviceBuffer<Bf16>,
    w1: &DeviceBuffer<Bf16>,
    w3: &DeviceBuffer<Bf16>,
    w2: &DeviceBuffer<Bf16>,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut FfnScratch,
) -> Result<()> {
    if output.len() != config.dim {
        return Err(invalid_data(format!(
            "FFN output has length {}, expected {}",
            output.len(),
            config.dim
        )));
    }

    ops::rmsnorm(
        stream,
        module,
        hidden,
        ffn_norm,
        config.norm_eps,
        &mut scratch.normed,
    )?;
    ops::silu_gate_up_bf16(
        stream,
        module,
        &scratch.normed,
        w1,
        w3,
        &mut scratch.activated,
    )?;
    ops::linear_residual_bf16(stream, module, &scratch.activated, w2, hidden, output)?;
    Ok(())
}

fn run_ffn_from_hidden_into<W1, W3, W2>(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    config: &TextConfig,
    hidden: &DeviceBuffer<f32>,
    ffn_norm: &DeviceBuffer<Bf16>,
    w1: &W1,
    w3: &W3,
    w2: &W2,
    output: &mut DeviceBuffer<f32>,
    scratch: &mut FfnScratch,
) -> Result<()>
where
    W1: ops::CudaLinearWeight,
    W3: ops::CudaLinearWeight,
    W2: ops::CudaLinearWeight,
{
    if output.len() != config.dim {
        return Err(invalid_data(format!(
            "FFN output has length {}, expected {}",
            output.len(),
            config.dim
        )));
    }

    ops::rmsnorm(
        stream,
        module,
        hidden,
        ffn_norm,
        config.norm_eps,
        &mut scratch.normed,
    )?;
    ops::linear(stream, module, &scratch.normed, w1, &mut scratch.gate)?;
    ops::linear(stream, module, &scratch.normed, w3, &mut scratch.up)?;
    ops::silu_mul(
        stream,
        module,
        &scratch.gate,
        &scratch.up,
        &mut scratch.activated,
    )?;
    ops::linear(stream, module, &scratch.activated, w2, &mut scratch.down)?;
    ops::add(stream, module, hidden, &scratch.down, output)?;
    Ok(())
}

fn storage_to_f32_vec(input: &[Bf16]) -> Vec<f32> {
    input.iter().map(|value| value.to_f32()).collect()
}

fn cpu_rmsnorm_bf16(input: &[f32], weight: &[Bf16], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = input.iter().map(|value| value * value).sum();
    let inv_scale = 1.0 / (sum_sq / input.len() as f32 + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(value, weight)| value * inv_scale * weight.to_f32())
        .collect()
}

fn cpu_matvec_bf16(input: &[f32], weight: &[Bf16], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; out_dim];
    for row in 0..out_dim {
        let row_offset = row * in_dim;
        let mut acc = 0.0;
        for col in 0..in_dim {
            acc += weight[row_offset + col].to_f32() * input[col];
        }
        out[row] = acc;
    }
    out
}

fn cpu_single_token_gqa(
    value: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let heads_per_kv = n_heads / n_kv_heads;
    let mut out = vec![0.0; n_heads * head_dim];
    for q_head in 0..n_heads {
        let kv_head = q_head / heads_per_kv;
        let src = kv_head * head_dim;
        let dst = q_head * head_dim;
        out[dst..dst + head_dim].copy_from_slice(&value[src..src + head_dim]);
    }
    out
}

fn cpu_apply_rope(input: &[f32], freqs: &[f32], position: usize, head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = vec![0.0; input.len()];
    for i in 0..input.len() {
        let head = i / head_dim;
        let dim = i - head * head_dim;
        let pair_dim = if dim < half { dim } else { dim - half };
        let base = head * head_dim;
        let first = input[base + pair_dim];
        let second = input[base + pair_dim + half];
        let angle = position as f32 * freqs[pair_dim];
        let sin = angle.sin();
        let cos = angle.cos();
        out[i] = if dim < half {
            first * cos - second * sin
        } else {
            second * cos + first * sin
        };
    }
    out
}

fn cpu_attention_scores(
    query: &[f32],
    key_cache: &[f32],
    seq_len: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let heads_per_kv = n_heads / n_kv_heads;
    let mut scores = vec![0.0; n_heads * max_seq_len];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for head in 0..n_heads {
        let kv_head = head / heads_per_kv;
        let q_offset = head * head_dim;
        for pos in 0..seq_len {
            let k_offset = pos * n_kv_heads * head_dim + kv_head * head_dim;
            let mut dot = 0.0;
            for dim in 0..head_dim {
                dot += query[q_offset + dim] * key_cache[k_offset + dim];
            }
            scores[head * max_seq_len + pos] = dot * scale;
        }
    }

    scores
}

fn cpu_softmax_value(
    scores: &[f32],
    value_cache: &[f32],
    seq_len: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let heads_per_kv = n_heads / n_kv_heads;
    let mut out = vec![0.0; n_heads * head_dim];

    for head in 0..n_heads {
        let score_base = head * max_seq_len;
        let mut max_score = scores[score_base];
        for pos in 1..seq_len {
            max_score = max_score.max(scores[score_base + pos]);
        }

        let kv_head = head / heads_per_kv;
        let mut denom = 0.0;
        let mut weights = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let weight = (scores[score_base + pos] - max_score).exp();
            denom += weight;
            weights.push(weight);
        }

        for dim in 0..head_dim {
            let mut acc = 0.0;
            for (pos, weight) in weights.iter().copied().enumerate() {
                let value = value_cache[pos * n_kv_heads * head_dim + kv_head * head_dim + dim];
                acc += weight * value;
            }
            out[head * head_dim + dim] = acc / denom;
        }
    }

    out
}

fn cpu_add(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect()
}

fn cpu_ffn_from_hidden(config: &TextConfig, hidden: &[f32], layer: &LayerHostWeights) -> Vec<f32> {
    let normed = cpu_rmsnorm_bf16(hidden, &layer.ffn_norm, config.norm_eps);
    let gate = cpu_matvec_bf16(&normed, &layer.w1, config.hidden_dim, config.dim);
    let up = cpu_matvec_bf16(&normed, &layer.w3, config.hidden_dim, config.dim);
    let activated: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| (gate / (1.0 + (-gate).exp())) * up)
        .collect();
    let down = cpu_matvec_bf16(&activated, &layer.w2, config.dim, config.hidden_dim);
    cpu_add(hidden, &down)
}

fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f32::max)
}

fn mean_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    let sum: f32 = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum();
    sum / lhs.len().max(1) as f32
}

fn kl_divergence_from_logits(reference_logits: &[f32], candidate_logits: &[f32]) -> Result<f64> {
    if reference_logits.len() != candidate_logits.len() {
        return Err(invalid_data(format!(
            "KL logits length mismatch: reference={}, candidate={}",
            reference_logits.len(),
            candidate_logits.len()
        )));
    }
    if reference_logits.is_empty() {
        return Err(invalid_data("KL logits must be nonempty"));
    }
    if reference_logits
        .iter()
        .chain(candidate_logits.iter())
        .any(|logit| !logit.is_finite())
    {
        return Err(invalid_data("KL logits must be finite"));
    }

    let reference_logsumexp = logsum_exp(reference_logits);
    let candidate_logsumexp = logsum_exp(candidate_logits);
    let mut kl = 0.0_f64;
    for (&reference, &candidate) in reference_logits.iter().zip(candidate_logits.iter()) {
        let reference_logprob = reference - reference_logsumexp;
        let candidate_logprob = candidate - candidate_logsumexp;
        let probability = (reference_logprob as f64).exp();
        kl += probability * (reference_logprob - candidate_logprob) as f64;
    }

    Ok(kl)
}

fn logsum_exp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = values.iter().map(|value| (*value - max).exp()).sum();
    max + sum.ln()
}

fn validate_layer_token(config: &TextConfig, layer: usize, token_id: u32) -> Result<()> {
    if layer >= config.n_layers {
        return Err(invalid_data(format!(
            "layer {layer} is out of range for {} layers",
            config.n_layers
        )));
    }
    validate_token(config, token_id)
}

fn validate_token(config: &TextConfig, token_id: u32) -> Result<()> {
    if token_id as usize >= config.vocab_size {
        return Err(invalid_data(format!(
            "token {token_id} is out of range for vocab size {}",
            config.vocab_size
        )));
    }

    Ok(())
}

fn validate_runtime_max_seq_len(config: &TextConfig, max_seq_len: usize) -> Result<()> {
    if max_seq_len > config.max_position_embeddings {
        return Err(invalid_data(format!(
            "runtime max_seq_len {max_seq_len} exceeds model max_position_embeddings {}",
            config.max_position_embeddings
        )));
    }

    Ok(())
}

fn validate_hidden_states(config: &TextConfig, hidden_states: &[Vec<f32>]) -> Result<()> {
    if hidden_states.is_empty() {
        return Err(invalid_data("hidden state list must be nonempty"));
    }

    for (i, hidden) in hidden_states.iter().enumerate() {
        if hidden.len() != config.dim {
            return Err(invalid_data(format!(
                "hidden state {i} has length {}, expected {}",
                hidden.len(),
                config.dim
            )));
        }
    }

    Ok(())
}

fn validate_device_hidden_states(
    config: &TextConfig,
    hidden_states: &[DeviceBuffer<f32>],
) -> Result<()> {
    if hidden_states.is_empty() {
        return Err(invalid_data("hidden state list must be nonempty"));
    }

    for (i, hidden) in hidden_states.iter().enumerate() {
        if hidden.len() != config.dim {
            return Err(invalid_data(format!(
                "hidden state {i} has length {}, expected {}",
                hidden.len(),
                config.dim
            )));
        }
    }

    Ok(())
}

fn top_k_logits(logits: &[f32], top_k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token, logit)| (token as u32, logit))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(top_k.min(indexed.len()));
    indexed
}

fn unpack_packed_top_logit(packed: u64) -> (u32, f32) {
    let token = packed as u32;
    let logit = f32::from_bits((packed >> 32) as u32);
    (token, logit)
}

fn top_k_logits_device(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    logits: &DeviceBuffer<f32>,
    top_k: usize,
    argmax_packed: &mut DeviceBuffer<u64>,
    topk_tokens: &mut DeviceBuffer<u32>,
    topk_logits: &mut DeviceBuffer<f32>,
) -> Result<Vec<(u32, f32)>> {
    if top_k <= 1 {
        ops::argmax_f32_packed(stream, module, logits, argmax_packed)?;
        let packed = argmax_packed.to_host_vec(stream)?[0];
        return Ok(vec![unpack_packed_top_logit(packed)]);
    }
    if top_k <= topk_tokens.len() && top_k <= topk_logits.len() {
        ops::top_k_f32(stream, module, logits, top_k, topk_tokens, topk_logits)?;
        let tokens = topk_tokens.to_host_vec(stream)?;
        let logit_values = topk_logits.to_host_vec(stream)?;
        return Ok(tokens.into_iter().zip(logit_values).take(top_k).collect());
    }

    let logits_host = logits.to_host_vec(stream)?;
    Ok(top_k_logits(&logits_host, top_k))
}

fn top_logit_tokens_match(lhs: &[(u32, f32)], rhs: &[(u32, f32)]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|((lhs_token, _), (rhs_token, _))| lhs_token == rhs_token)
}

fn paired_top_logits_max_abs_diff(lhs: &[(u32, f32)], rhs: &[(u32, f32)]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|((_, lhs_logit), (_, rhs_logit))| (lhs_logit - rhs_logit).abs())
        .fold(0.0, f32::max)
}

fn parse_json_object_field<'a>(json: &'a str, field: &str) -> Result<&'a str> {
    let field_pos = find_json_field(json, field)?;
    let value_start = json_field_value_start(json, field_pos, field)?;
    json_object_at(json, value_start)
}

fn parse_json_usize_field(json: &str, field: &str) -> Result<usize> {
    let field_pos = find_json_field(json, field)?;
    let start = json_field_value_start(json, field_pos, field)?;
    let end = json_number_end(json, start)?;
    Ok(json[start..end].parse()?)
}

fn parse_json_f32_field(json: &str, field: &str) -> Result<f32> {
    parse_optional_json_f32_field(json, field)
        .ok_or_else(|| invalid_data(format!("missing JSON field {field}")))
}

fn parse_optional_json_f32_field(json: &str, field: &str) -> Option<f32> {
    let field_pos = find_json_field(json, field).ok()?;
    let start = json_field_value_start(json, field_pos, field).ok()?;
    let end = json_number_end(json, start).ok()?;
    json[start..end].parse().ok()
}

fn parse_json_string_field(json: &str, field: &str) -> Result<String> {
    let field_pos = find_json_field(json, field)?;
    let start = json_field_value_start(json, field_pos, field)?;
    parse_json_string_at(json, start).map(|(value, _)| value)
}

fn find_json_field(json: &str, field: &str) -> Result<usize> {
    let needle = format!("\"{field}\"");
    json.find(&needle)
        .ok_or_else(|| invalid_data(format!("missing JSON field {field}")))
}

fn json_field_value_start(json: &str, field_pos: usize, field: &str) -> Result<usize> {
    let mut i = skip_json_ws(json, field_pos + field.len() + 2);
    expect_json_byte(json, i, b':')?;
    i = skip_json_ws(json, i + 1);
    Ok(i)
}

fn json_object_at(json: &str, start: usize) -> Result<&str> {
    expect_json_byte(json, start, b'{')?;
    let bytes = json.as_bytes();
    let mut depth = 0usize;
    let mut i = start;

    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                let (_, next) = parse_json_string_at(json, i)?;
                i = next;
                continue;
            }
            b'{' => depth += 1,
            b'}' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| invalid_data("JSON object depth underflow"))?;
                if depth == 0 {
                    return Ok(&json[start..i + 1]);
                }
            }
            _ => {}
        }
        i += 1;
    }

    Err(invalid_data("unterminated JSON object"))
}

fn json_number_end(json: &str, start: usize) -> Result<usize> {
    let mut i = start;
    let bytes = json.as_bytes();
    while i < bytes.len() && matches!(bytes[i], b'0'..=b'9' | b'+' | b'-' | b'.' | b'e' | b'E') {
        i += 1;
    }
    if i == start {
        return Err(invalid_data("expected JSON number"));
    }
    Ok(i)
}

fn parse_json_string_at(input: &str, start: usize) -> Result<(String, usize)> {
    expect_json_byte(input, start, b'"')?;

    let mut out = String::new();
    let bytes = input.as_bytes();
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Ok((out, i + 1)),
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    return Err(invalid_data("unterminated JSON string escape"));
                }
                match bytes[i] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let end = i + 5;
                        if end > bytes.len() {
                            return Err(invalid_data("unterminated JSON unicode escape"));
                        }
                        let value = u16::from_str_radix(&input[i + 1..end], 16)?;
                        let ch = char::from_u32(value as u32)
                            .ok_or_else(|| invalid_data("invalid JSON unicode escape"))?;
                        out.push(ch);
                        i += 4;
                    }
                    byte => {
                        return Err(invalid_data(format!(
                            "unsupported JSON string escape byte {byte}"
                        )));
                    }
                }
            }
            byte => out.push(byte as char),
        }
        i += 1;
    }

    Err(invalid_data("unterminated JSON string"))
}

fn skip_json_ws(input: &str, mut i: usize) -> usize {
    while i < input.len() && input.as_bytes()[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

fn expect_json_byte(input: &str, offset: usize, expected: u8) -> Result<()> {
    match input.as_bytes().get(offset) {
        Some(actual) if *actual == expected => Ok(()),
        Some(actual) => Err(invalid_data(format!(
            "expected byte {}, got {} at offset {}",
            expected, actual, offset
        ))),
        None => Err(invalid_data(format!(
            "expected byte {} at offset {}, got EOF",
            expected, offset
        ))),
    }
}

fn read_bf16_shape(
    weights: &ModelWeights,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<Bf16>> {
    let tensor = weights.tensor(name)?;
    expect_shape(&tensor, expected_shape)?;
    weights.read_bf16_tensor(&tensor)
}

fn expect_shape(tensor: &TensorInfo, expected: &[usize]) -> Result<()> {
    if tensor.shape == expected {
        return Ok(());
    }

    Err(invalid_data(format!(
        "tensor {} has shape {:?}, expected {:?}",
        tensor.name, tensor.shape, expected
    )))
}

fn invalid_data(message: impl Into<String>) -> Box<dyn std::error::Error> {
    io::Error::new(io::ErrorKind::InvalidData, message.into()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_max_seq_len_respects_model_position_limit() {
        let config = TextConfig {
            max_position_embeddings: 8,
            ..TextConfig::MINISTRAL_3_8B_REASONING_2512
        };

        assert!(validate_runtime_max_seq_len(&config, 8).is_ok());
        let error = validate_runtime_max_seq_len(&config, 9).unwrap_err();
        assert_eq!(
            error.to_string(),
            "runtime max_seq_len 9 exceeds model max_position_embeddings 8"
        );
    }
}
