use std::{
    io::{self, ErrorKind},
    path::Path,
    sync::Arc,
    time::Instant,
};

use cuda_core::{CudaModule, CudaStream};

use crate::{
    chat,
    model::{
        self, AllLinearQuantizedRuntimeComparisonStepProbe, GreedyGenerationStep,
        MinistralAllLinearQuantizedRuntime, MinistralTextRuntime, RuntimeMemoryStats,
    },
    safetensors::Result,
    tokenizer::TekkenTokenizer,
};

#[derive(Debug, Clone)]
pub enum SystemPrompt {
    DefaultFromModel,
    None,
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackend {
    Bf16,
    AllLinearInt8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationComparisonDriver {
    Candidate,
    Reference,
}

pub type ChatBackend = InferenceBackend;

struct EncodedChatPrompt {
    formatted_prompt: String,
    prompt_tokens: Vec<u32>,
}

fn encode_single_turn_chat_prompt_with_system(
    tokenizer: &TekkenTokenizer,
    system_prompt: Option<&str>,
    user_prompt: &str,
) -> Result<EncodedChatPrompt> {
    let formatted_prompt = chat::format_ministral_single_turn_chat(system_prompt, user_prompt);
    let prompt_tokens = tokenizer.encode_lossy(&formatted_prompt, true)?;

    Ok(EncodedChatPrompt {
        formatted_prompt,
        prompt_tokens,
    })
}

fn encode_single_turn_chat_prompt(
    tokenizer: &TekkenTokenizer,
    default_system_prompt: &str,
    user_prompt: &str,
    system_prompt: &SystemPrompt,
) -> Result<EncodedChatPrompt> {
    let system_prompt = match system_prompt {
        SystemPrompt::DefaultFromModel => Some(default_system_prompt),
        SystemPrompt::None => None,
        SystemPrompt::Custom(prompt) => Some(prompt.as_str()),
    };

    encode_single_turn_chat_prompt_with_system(tokenizer, system_prompt, user_prompt)
}

fn encode_single_turn_chat_prompt_from_model(
    tokenizer: &TekkenTokenizer,
    model_dir: &Path,
    user_prompt: &str,
    system_prompt: &SystemPrompt,
) -> Result<EncodedChatPrompt> {
    match system_prompt {
        SystemPrompt::DefaultFromModel => {
            let default_system_prompt = chat::default_ministral_system_prompt(model_dir)?;
            encode_single_turn_chat_prompt_with_system(
                tokenizer,
                Some(default_system_prompt.as_str()),
                user_prompt,
            )
        }
        SystemPrompt::None => {
            encode_single_turn_chat_prompt_with_system(tokenizer, None, user_prompt)
        }
        SystemPrompt::Custom(prompt) => encode_single_turn_chat_prompt_with_system(
            tokenizer,
            Some(prompt.as_str()),
            user_prompt,
        ),
    }
}

trait TokenGenerationRuntime {
    fn backend(&self) -> InferenceBackend;
    fn memory_stats(&self) -> RuntimeMemoryStats;
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<()>;
    fn generate_greedy_until(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>>;
    fn next_logits_to_host(&self) -> Result<Vec<f32>>;
    fn next_top_logits(&mut self, top_k: usize) -> Result<Vec<(u32, f32)>>;
    fn advance_with_token(&mut self, token_id: u32) -> Result<()>;
    fn synchronize(&self) -> Result<()>;
    fn reset_sequence(&mut self);
    fn tokens(&self) -> &[u32];
    fn max_seq_len(&self) -> usize;

    fn run_generation_request(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerationOptions,
        context: &str,
    ) -> Result<GenerationResult>
    where
        Self: Sized,
    {
        run_generation_with_runtime(self, prompt_tokens, options, context)
    }
}

impl TokenGenerationRuntime for MinistralTextRuntime {
    fn backend(&self) -> InferenceBackend {
        InferenceBackend::Bf16
    }

    fn memory_stats(&self) -> RuntimeMemoryStats {
        MinistralTextRuntime::memory_stats(self)
    }

    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        MinistralTextRuntime::prefill(self, prompt_tokens)
    }

    fn generate_greedy_until(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        MinistralTextRuntime::generate_greedy_until(self, max_new_tokens, top_k, stop_token_id)
    }

    fn next_logits_to_host(&self) -> Result<Vec<f32>> {
        MinistralTextRuntime::next_logits_to_host(self)
    }

    fn next_top_logits(&mut self, top_k: usize) -> Result<Vec<(u32, f32)>> {
        MinistralTextRuntime::next_top_logits(self, top_k)
    }

    fn advance_with_token(&mut self, token_id: u32) -> Result<()> {
        MinistralTextRuntime::advance_with_token(self, token_id)
    }

    fn synchronize(&self) -> Result<()> {
        MinistralTextRuntime::synchronize(self)
    }

    fn reset_sequence(&mut self) {
        MinistralTextRuntime::reset_sequence(self);
    }

    fn tokens(&self) -> &[u32] {
        MinistralTextRuntime::tokens(self)
    }

    fn max_seq_len(&self) -> usize {
        MinistralTextRuntime::max_seq_len(self)
    }

    fn run_generation_request(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerationOptions,
        context: &str,
    ) -> Result<GenerationResult> {
        if let Some(result) =
            run_bf16_top1_generation_for_request(self, prompt_tokens, options, context)?
        {
            return Ok(result);
        }

        run_generation_with_runtime(self, prompt_tokens, options, context)
    }
}

impl TokenGenerationRuntime for MinistralAllLinearQuantizedRuntime {
    fn backend(&self) -> InferenceBackend {
        InferenceBackend::AllLinearInt8
    }

    fn memory_stats(&self) -> RuntimeMemoryStats {
        MinistralAllLinearQuantizedRuntime::memory_stats(self)
    }

    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        MinistralAllLinearQuantizedRuntime::prefill(self, prompt_tokens)
    }

    fn generate_greedy_until(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        MinistralAllLinearQuantizedRuntime::generate_greedy_until(
            self,
            max_new_tokens,
            top_k,
            stop_token_id,
        )
    }

    fn next_logits_to_host(&self) -> Result<Vec<f32>> {
        MinistralAllLinearQuantizedRuntime::next_logits_to_host(self)
    }

    fn next_top_logits(&mut self, top_k: usize) -> Result<Vec<(u32, f32)>> {
        MinistralAllLinearQuantizedRuntime::next_top_logits(self, top_k)
    }

    fn advance_with_token(&mut self, token_id: u32) -> Result<()> {
        MinistralAllLinearQuantizedRuntime::advance_with_token(self, token_id)
    }

    fn synchronize(&self) -> Result<()> {
        MinistralAllLinearQuantizedRuntime::synchronize(self)
    }

    fn reset_sequence(&mut self) {
        MinistralAllLinearQuantizedRuntime::reset_sequence(self);
    }

    fn tokens(&self) -> &[u32] {
        MinistralAllLinearQuantizedRuntime::tokens(self)
    }

    fn max_seq_len(&self) -> usize {
        MinistralAllLinearQuantizedRuntime::max_seq_len(self)
    }
}

impl TokenGenerationRuntime for MinistralGenerationBackendSession {
    fn backend(&self) -> InferenceBackend {
        MinistralGenerationBackendSession::backend(self)
    }

    fn memory_stats(&self) -> RuntimeMemoryStats {
        MinistralGenerationBackendSession::memory_stats(self)
    }

    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        MinistralGenerationBackendSession::prefill(self, prompt_tokens)
    }

    fn generate_greedy_until(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        MinistralGenerationBackendSession::generate_greedy_until(
            self,
            max_new_tokens,
            top_k,
            stop_token_id,
        )
    }

    fn next_logits_to_host(&self) -> Result<Vec<f32>> {
        MinistralGenerationBackendSession::next_logits_to_host(self)
    }

    fn next_top_logits(&mut self, top_k: usize) -> Result<Vec<(u32, f32)>> {
        MinistralGenerationBackendSession::next_top_logits(self, top_k)
    }

    fn advance_with_token(&mut self, token_id: u32) -> Result<()> {
        MinistralGenerationBackendSession::advance_with_token(self, token_id)
    }

    fn synchronize(&self) -> Result<()> {
        MinistralGenerationBackendSession::synchronize(self)
    }

    fn reset_sequence(&mut self) {
        MinistralGenerationBackendSession::reset_sequence(self);
    }

    fn tokens(&self) -> &[u32] {
        MinistralGenerationBackendSession::tokens(self)
    }

    fn max_seq_len(&self) -> usize {
        MinistralGenerationBackendSession::max_seq_len(self)
    }

    fn run_generation_request(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerationOptions,
        context: &str,
    ) -> Result<GenerationResult> {
        if let Self::Bf16(runtime) = self {
            if let Some(result) =
                run_bf16_top1_generation_for_request(runtime, prompt_tokens, options, context)?
            {
                return Ok(result);
            }
        }

        run_generation_with_runtime(self, prompt_tokens, options, context)
    }
}

#[derive(Debug, Clone)]
pub struct ChatInferenceOptions {
    pub max_new_tokens: usize,
    pub top_k: usize,
    pub system_prompt: SystemPrompt,
    pub stop_token_id: Option<u32>,
}

impl Default for ChatInferenceOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 1,
            top_k: 1,
            system_prompt: SystemPrompt::DefaultFromModel,
            stop_token_id: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingOptions {
    pub temperature: f32,
    pub seed: u64,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinishReason {
    MaxNewTokens,
    StopToken(u32),
}

impl GenerationFinishReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::MaxNewTokens => "max-new-tokens",
            Self::StopToken(_) => "stop-token",
        }
    }

    pub fn stop_token_id(self) -> Option<u32> {
        match self {
            Self::MaxNewTokens => None,
            Self::StopToken(token_id) => Some(token_id),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GenerationTimings {
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

impl GenerationTimings {
    pub fn new(prefill_seconds: f64, decode_seconds: f64) -> Self {
        Self {
            prefill_seconds,
            decode_seconds,
        }
    }

    pub fn total_seconds(self) -> f64 {
        self.prefill_seconds + self.decode_seconds
    }

    pub fn decode_tokens_per_second(self, generated_token_count: usize) -> Option<f64> {
        tokens_per_second(generated_token_count, self.decode_seconds)
    }

    pub fn total_tokens_per_second(self, generated_token_count: usize) -> Option<f64> {
        tokens_per_second(generated_token_count, self.total_seconds())
    }
}

#[derive(Debug)]
pub struct ChatInferenceResult {
    pub backend: InferenceBackend,
    pub formatted_prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub memory_stats: RuntimeMemoryStats,
    pub timings: GenerationTimings,
    pub generated_text: String,
    pub all_text_skip_special: String,
    pub finish_reason: GenerationFinishReason,
    pub steps: Vec<GreedyGenerationStep>,
}

#[derive(Debug)]
pub struct ChatGenerationPromptResult {
    pub user_prompt: String,
    pub result: ChatInferenceResult,
}

#[derive(Debug)]
pub struct ChatGenerationSuiteResult {
    pub prompts: Vec<ChatGenerationPromptResult>,
}

#[derive(Debug, Clone)]
pub struct ChatLogitsOptions {
    pub top_k: usize,
    pub system_prompt: SystemPrompt,
}

impl Default for ChatLogitsOptions {
    fn default() -> Self {
        Self {
            top_k: 8,
            system_prompt: SystemPrompt::DefaultFromModel,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenLogit {
    pub token_id: u32,
    pub logit: f32,
    pub logprob: f32,
}

#[derive(Debug)]
pub struct ChatLogitsResult {
    pub formatted_prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub logits: Vec<f32>,
    pub logsumexp: f32,
    pub top_logits: Vec<TokenLogit>,
}

#[derive(Debug)]
pub struct ChatLogitsPromptResult {
    pub user_prompt: String,
    pub result: ChatLogitsResult,
}

#[derive(Debug)]
pub struct ChatLogitsSuiteResult {
    pub prompts: Vec<ChatLogitsPromptResult>,
}

#[derive(Debug, Clone)]
pub struct ChatLogitsTraceOptions {
    pub max_new_tokens: usize,
    pub top_k: usize,
    pub logits_top_k: usize,
    pub system_prompt: SystemPrompt,
    pub stop_token_id: Option<u32>,
}

impl Default for ChatLogitsTraceOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 1,
            top_k: 1,
            logits_top_k: 8,
            system_prompt: SystemPrompt::DefaultFromModel,
            stop_token_id: None,
        }
    }
}

#[derive(Debug)]
pub struct ChatLogitsTraceStep {
    pub step: usize,
    pub token_id: u32,
    pub logit: f32,
    pub logits: Vec<f32>,
    pub logsumexp: f32,
    pub top_logits: Vec<TokenLogit>,
}

#[derive(Debug)]
pub struct ChatLogitsTraceResult {
    pub formatted_prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub generated_text: String,
    pub all_text_skip_special: String,
    pub finish_reason: GenerationFinishReason,
    pub steps: Vec<ChatLogitsTraceStep>,
}

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub top_k: usize,
    pub stop_token_id: Option<u32>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 1,
            top_k: 1,
            stop_token_id: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationLogitsTraceOptions {
    pub max_new_tokens: usize,
    pub top_k: usize,
    pub logits_top_k: usize,
    pub stop_token_id: Option<u32>,
}

impl Default for GenerationLogitsTraceOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 1,
            top_k: 1,
            logits_top_k: 8,
            stop_token_id: None,
        }
    }
}

#[derive(Debug)]
pub struct GenerationResult {
    pub backend: InferenceBackend,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub memory_stats: RuntimeMemoryStats,
    pub timings: GenerationTimings,
    pub finish_reason: GenerationFinishReason,
    pub steps: Vec<GreedyGenerationStep>,
}

#[derive(Debug)]
pub struct GenerationSuiteResult {
    pub prompts: Vec<GenerationResult>,
}

#[derive(Debug)]
pub struct GenerationLogitsResult {
    pub backend: InferenceBackend,
    pub prompt_tokens: Vec<u32>,
    pub logits: Vec<f32>,
    pub logsumexp: f32,
    pub top_logits: Vec<TokenLogit>,
    pub memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct GenerationLogitsSuiteResult {
    pub prompts: Vec<GenerationLogitsResult>,
}

#[derive(Debug)]
pub struct GenerationLogitsTraceResult {
    pub backend: InferenceBackend,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub memory_stats: RuntimeMemoryStats,
    pub finish_reason: GenerationFinishReason,
    pub steps: Vec<ChatLogitsTraceStep>,
}

#[derive(Debug)]
pub struct GenerationLogitsTraceSuiteResult {
    pub prompts: Vec<GenerationLogitsTraceResult>,
}

#[derive(Debug)]
pub struct GenerationBackendComparisonStep {
    pub step: usize,
    pub prefix_len: usize,
    pub reference_token_id: u32,
    pub reference_logit: f32,
    pub candidate_token_id: u32,
    pub candidate_logit: f32,
    pub token_matches: bool,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub reference_top_logits: Vec<TokenLogit>,
    pub candidate_top_logits: Vec<TokenLogit>,
}

impl GenerationBackendComparisonStep {
    pub fn top_tokens_match(&self) -> bool {
        token_logit_ids_match(&self.reference_top_logits, &self.candidate_top_logits)
    }
}

#[derive(Debug)]
pub struct GenerationBackendLogitsComparisonResult {
    pub prompt_tokens: Vec<u32>,
    pub reference: GenerationLogitsResult,
    pub candidate: GenerationLogitsResult,
    pub reference_token_id: u32,
    pub reference_logit: f32,
    pub candidate_token_id: u32,
    pub candidate_logit: f32,
    pub token_matches: bool,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

impl GenerationBackendLogitsComparisonResult {
    pub fn top_tokens_match(&self) -> bool {
        token_logit_ids_match(&self.reference.top_logits, &self.candidate.top_logits)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationBackendLogitsComparisonSuiteSummary {
    pub prompt_count: usize,
    pub token_match_count: usize,
    pub kl_sum: f64,
    pub max_kl: f64,
    pub max_abs_diff: f32,
    pub top_tokens_match: bool,
}

impl Default for GenerationBackendLogitsComparisonSuiteSummary {
    fn default() -> Self {
        Self {
            prompt_count: 0,
            token_match_count: 0,
            kl_sum: 0.0,
            max_kl: 0.0,
            max_abs_diff: 0.0,
            top_tokens_match: true,
        }
    }
}

impl GenerationBackendLogitsComparisonSuiteSummary {
    pub fn mean_kl(&self) -> f64 {
        if self.prompt_count == 0 {
            0.0
        } else {
            self.kl_sum / self.prompt_count as f64
        }
    }

    pub fn observe_comparison(&mut self, comparison: &GenerationBackendLogitsComparisonResult) {
        self.prompt_count += 1;
        self.token_match_count += usize::from(comparison.token_matches);
        self.kl_sum += comparison.kl_divergence;
        self.max_kl = self.max_kl.max(comparison.kl_divergence);
        self.max_abs_diff = self.max_abs_diff.max(comparison.max_abs_diff);
        self.top_tokens_match &= comparison.top_tokens_match();
    }
}

#[derive(Debug)]
pub struct GenerationBackendLogitsComparisonSuiteResult {
    pub prompts: Vec<GenerationBackendLogitsComparisonResult>,
    pub summary: GenerationBackendLogitsComparisonSuiteSummary,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct GenerationBackendComparisonResult {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub driver: GenerationComparisonDriver,
    pub reference_backend: InferenceBackend,
    pub candidate_backend: InferenceBackend,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
    pub steps: Vec<GenerationBackendComparisonStep>,
}

impl GenerationBackendComparisonResult {
    pub fn summary(&self) -> GenerationBackendComparisonSummary {
        let mut summary = GenerationBackendComparisonSummary::default();
        for step in &self.steps {
            summary.observe_step(step);
        }
        summary
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationBackendComparisonSummary {
    pub step_count: usize,
    pub token_match_count: usize,
    pub kl_sum: f64,
    pub max_kl: f64,
    pub max_abs_diff: f32,
    pub top_tokens_match: bool,
}

impl Default for GenerationBackendComparisonSummary {
    fn default() -> Self {
        Self {
            step_count: 0,
            token_match_count: 0,
            kl_sum: 0.0,
            max_kl: 0.0,
            max_abs_diff: 0.0,
            top_tokens_match: true,
        }
    }
}

impl GenerationBackendComparisonSummary {
    pub fn mean_kl(&self) -> f64 {
        if self.step_count == 0 {
            0.0
        } else {
            self.kl_sum / self.step_count as f64
        }
    }

    pub fn observe_step(&mut self, step: &GenerationBackendComparisonStep) {
        self.step_count += 1;
        self.token_match_count += usize::from(step.token_matches);
        self.kl_sum += step.kl_divergence;
        self.max_kl = self.max_kl.max(step.kl_divergence);
        self.max_abs_diff = self.max_abs_diff.max(step.max_abs_diff);
        self.top_tokens_match &= step.top_tokens_match();
    }

    pub fn merge(&mut self, other: &Self) {
        self.step_count += other.step_count;
        self.token_match_count += other.token_match_count;
        self.kl_sum += other.kl_sum;
        self.max_kl = self.max_kl.max(other.max_kl);
        self.max_abs_diff = self.max_abs_diff.max(other.max_abs_diff);
        self.top_tokens_match &= other.top_tokens_match;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationBackendComparisonSuiteSummary {
    pub prompt_count: usize,
    pub summary: GenerationBackendComparisonSummary,
}

impl Default for GenerationBackendComparisonSuiteSummary {
    fn default() -> Self {
        Self {
            prompt_count: 0,
            summary: GenerationBackendComparisonSummary::default(),
        }
    }
}

impl GenerationBackendComparisonSuiteSummary {
    pub fn observe_summary(&mut self, summary: &GenerationBackendComparisonSummary) {
        self.prompt_count += 1;
        self.summary.merge(summary);
    }

    pub fn observe_comparison(&mut self, comparison: &GenerationBackendComparisonResult) {
        self.observe_summary(&comparison.summary());
    }
}

#[derive(Debug)]
pub struct GenerationBackendComparisonSuiteResult {
    pub prompts: Vec<GenerationBackendComparisonResult>,
    pub summary: GenerationBackendComparisonSuiteSummary,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct GenerationBackendForcedComparisonStep {
    pub step: usize,
    pub prefix_len: usize,
    pub forced_token_id: u32,
    pub reference_token_id: u32,
    pub reference_logit: f32,
    pub reference_forced_logit: f32,
    pub reference_forced_logprob: f32,
    pub candidate_token_id: u32,
    pub candidate_logit: f32,
    pub candidate_forced_logit: f32,
    pub candidate_forced_logprob: f32,
    pub token_matches: bool,
    pub forced_token_matches_reference: bool,
    pub forced_token_matches_candidate: bool,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub reference_top_logits: Vec<TokenLogit>,
    pub candidate_top_logits: Vec<TokenLogit>,
}

impl GenerationBackendForcedComparisonStep {
    pub fn top_tokens_match(&self) -> bool {
        token_logit_ids_match(&self.reference_top_logits, &self.candidate_top_logits)
    }

    pub fn forced_logprob_abs_diff(&self) -> f32 {
        (self.reference_forced_logprob - self.candidate_forced_logprob).abs()
    }
}

#[derive(Debug)]
pub struct GenerationBackendForcedComparisonResult {
    pub sequence_tokens: Vec<u32>,
    pub prompt_tokens: Vec<u32>,
    pub forced_tokens: Vec<u32>,
    pub reference_backend: InferenceBackend,
    pub candidate_backend: InferenceBackend,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
    pub steps: Vec<GenerationBackendForcedComparisonStep>,
}

impl GenerationBackendForcedComparisonResult {
    pub fn summary(&self) -> GenerationBackendForcedComparisonSummary {
        let mut summary = GenerationBackendForcedComparisonSummary::default();
        for step in &self.steps {
            summary.observe_step(step);
        }
        summary
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationBackendForcedComparisonSummary {
    pub step_count: usize,
    pub token_match_count: usize,
    pub forced_reference_match_count: usize,
    pub forced_candidate_match_count: usize,
    pub kl_sum: f64,
    pub max_kl: f64,
    pub max_abs_diff: f32,
    pub forced_logprob_abs_diff_sum: f64,
    pub max_forced_logprob_abs_diff: f32,
    pub top_tokens_match: bool,
}

impl Default for GenerationBackendForcedComparisonSummary {
    fn default() -> Self {
        Self {
            step_count: 0,
            token_match_count: 0,
            forced_reference_match_count: 0,
            forced_candidate_match_count: 0,
            kl_sum: 0.0,
            max_kl: 0.0,
            max_abs_diff: 0.0,
            forced_logprob_abs_diff_sum: 0.0,
            max_forced_logprob_abs_diff: 0.0,
            top_tokens_match: true,
        }
    }
}

impl GenerationBackendForcedComparisonSummary {
    pub fn mean_kl(&self) -> f64 {
        if self.step_count == 0 {
            0.0
        } else {
            self.kl_sum / self.step_count as f64
        }
    }

    pub fn mean_forced_logprob_abs_diff(&self) -> f64 {
        if self.step_count == 0 {
            0.0
        } else {
            self.forced_logprob_abs_diff_sum / self.step_count as f64
        }
    }

    pub fn observe_step(&mut self, step: &GenerationBackendForcedComparisonStep) {
        self.step_count += 1;
        self.token_match_count += usize::from(step.token_matches);
        self.forced_reference_match_count += usize::from(step.forced_token_matches_reference);
        self.forced_candidate_match_count += usize::from(step.forced_token_matches_candidate);
        self.kl_sum += step.kl_divergence;
        self.max_kl = self.max_kl.max(step.kl_divergence);
        self.max_abs_diff = self.max_abs_diff.max(step.max_abs_diff);
        let forced_logprob_abs_diff = step.forced_logprob_abs_diff();
        self.forced_logprob_abs_diff_sum += forced_logprob_abs_diff as f64;
        self.max_forced_logprob_abs_diff = self
            .max_forced_logprob_abs_diff
            .max(forced_logprob_abs_diff);
        self.top_tokens_match &= step.top_tokens_match();
    }

    pub fn merge(&mut self, other: &Self) {
        self.step_count += other.step_count;
        self.token_match_count += other.token_match_count;
        self.forced_reference_match_count += other.forced_reference_match_count;
        self.forced_candidate_match_count += other.forced_candidate_match_count;
        self.kl_sum += other.kl_sum;
        self.max_kl = self.max_kl.max(other.max_kl);
        self.max_abs_diff = self.max_abs_diff.max(other.max_abs_diff);
        self.forced_logprob_abs_diff_sum += other.forced_logprob_abs_diff_sum;
        self.max_forced_logprob_abs_diff = self
            .max_forced_logprob_abs_diff
            .max(other.max_forced_logprob_abs_diff);
        self.top_tokens_match &= other.top_tokens_match;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationBackendForcedComparisonSuiteSummary {
    pub prompt_count: usize,
    pub summary: GenerationBackendForcedComparisonSummary,
}

impl Default for GenerationBackendForcedComparisonSuiteSummary {
    fn default() -> Self {
        Self {
            prompt_count: 0,
            summary: GenerationBackendForcedComparisonSummary::default(),
        }
    }
}

impl GenerationBackendForcedComparisonSuiteSummary {
    pub fn observe_summary(&mut self, summary: &GenerationBackendForcedComparisonSummary) {
        self.prompt_count += 1;
        self.summary.merge(summary);
    }

    pub fn observe_comparison(&mut self, comparison: &GenerationBackendForcedComparisonResult) {
        self.observe_summary(&comparison.summary());
    }
}

#[derive(Debug)]
pub struct GenerationBackendForcedComparisonSuiteResult {
    pub prompts: Vec<GenerationBackendForcedComparisonResult>,
    pub summary: GenerationBackendForcedComparisonSuiteSummary,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct TextGenerationResult {
    pub backend: InferenceBackend,
    pub prompt_text: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub memory_stats: RuntimeMemoryStats,
    pub timings: GenerationTimings,
    pub generated_text: String,
    pub all_text_skip_special: String,
    pub finish_reason: GenerationFinishReason,
    pub steps: Vec<GreedyGenerationStep>,
}

#[derive(Debug)]
pub struct TextGenerationSuiteResult {
    pub prompts: Vec<TextGenerationResult>,
}

#[derive(Debug)]
pub struct TextGenerationLogitsResult {
    pub prompt_text: String,
    pub result: GenerationLogitsResult,
}

#[derive(Debug)]
pub struct TextGenerationLogitsSuiteResult {
    pub prompts: Vec<TextGenerationLogitsResult>,
}

#[derive(Debug)]
pub struct TextGenerationBackendComparisonResult {
    pub prompt_text: String,
    pub generated_text: String,
    pub all_text_skip_special: String,
    pub comparison: GenerationBackendComparisonResult,
}

#[derive(Debug)]
pub struct TextGenerationBackendLogitsComparisonResult {
    pub prompt_text: String,
    pub comparison: GenerationBackendLogitsComparisonResult,
}

#[derive(Debug)]
pub struct TextGenerationBackendLogitsComparisonSuiteResult {
    pub prompts: Vec<TextGenerationBackendLogitsComparisonResult>,
    pub summary: GenerationBackendLogitsComparisonSuiteSummary,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct TextGenerationBackendComparisonSuiteResult {
    pub prompts: Vec<TextGenerationBackendComparisonResult>,
    pub summary: GenerationBackendComparisonSuiteSummary,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct TextGenerationBackendForcedComparisonResult {
    pub sequence_text: String,
    pub comparison: GenerationBackendForcedComparisonResult,
}

#[derive(Debug)]
pub struct TextGenerationBackendForcedComparisonSuiteResult {
    pub prompts: Vec<TextGenerationBackendForcedComparisonResult>,
    pub summary: GenerationBackendForcedComparisonSuiteSummary,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct TextGenerationBackendForcedTargetComparisonResult {
    pub prompt_text: String,
    pub target_text: String,
    pub comparison: GenerationBackendForcedComparisonResult,
}

#[derive(Debug)]
pub struct TextGenerationBackendForcedTargetComparisonSuiteResult {
    pub prompts: Vec<TextGenerationBackendForcedTargetComparisonResult>,
    pub summary: GenerationBackendForcedComparisonSuiteSummary,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct TextGenerationBackendLogitsTraceResult {
    pub prompt_text: String,
    pub generated_text: String,
    pub all_text_skip_special: String,
    pub trace: GenerationLogitsTraceResult,
}

#[derive(Debug)]
pub struct TextGenerationBackendLogitsTraceSuiteResult {
    pub prompts: Vec<TextGenerationBackendLogitsTraceResult>,
}

#[derive(Debug)]
pub struct ChatBackendComparisonResult {
    pub formatted_prompt: String,
    pub generated_text: String,
    pub all_text_skip_special: String,
    pub comparison: GenerationBackendComparisonResult,
}

#[derive(Debug)]
pub struct ChatBackendLogitsComparisonResult {
    pub formatted_prompt: String,
    pub comparison: GenerationBackendLogitsComparisonResult,
}

#[derive(Debug)]
pub struct ChatBackendLogitsComparisonPromptResult {
    pub user_prompt: String,
    pub result: ChatBackendLogitsComparisonResult,
}

#[derive(Debug)]
pub struct ChatBackendLogitsComparisonSuiteResult {
    pub prompts: Vec<ChatBackendLogitsComparisonPromptResult>,
    pub summary: GenerationBackendLogitsComparisonSuiteSummary,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct ChatBackendComparisonPromptResult {
    pub user_prompt: String,
    pub result: ChatBackendComparisonResult,
}

#[derive(Debug)]
pub struct ChatBackendComparisonSuiteResult {
    pub prompts: Vec<ChatBackendComparisonPromptResult>,
    pub summary: GenerationBackendComparisonSuiteSummary,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct ChatBackendForcedTargetComparisonResult {
    pub formatted_prompt: String,
    pub target_text: String,
    pub comparison: GenerationBackendForcedComparisonResult,
}

#[derive(Debug)]
pub struct ChatBackendForcedTargetComparisonPromptResult {
    pub user_prompt: String,
    pub result: ChatBackendForcedTargetComparisonResult,
}

#[derive(Debug)]
pub struct ChatBackendForcedTargetComparisonSuiteResult {
    pub prompts: Vec<ChatBackendForcedTargetComparisonPromptResult>,
    pub summary: GenerationBackendForcedComparisonSuiteSummary,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub candidate_memory_stats: RuntimeMemoryStats,
}

#[derive(Debug)]
pub struct GenerationBackendEvalSuiteResult {
    pub logits: GenerationBackendLogitsComparisonSuiteResult,
    pub generation: GenerationBackendComparisonSuiteResult,
}

#[derive(Debug)]
pub struct TextGenerationBackendEvalSuiteResult {
    pub logits: TextGenerationBackendLogitsComparisonSuiteResult,
    pub generation: TextGenerationBackendComparisonSuiteResult,
}

#[derive(Debug)]
pub struct ChatBackendEvalSuiteResult {
    pub logits: ChatBackendLogitsComparisonSuiteResult,
    pub generation: ChatBackendComparisonSuiteResult,
}

#[derive(Debug)]
pub struct ChatBackendLogitsTraceResult {
    pub formatted_prompt: String,
    pub generated_text: String,
    pub all_text_skip_special: String,
    pub trace: GenerationLogitsTraceResult,
}

#[derive(Debug)]
pub struct ChatBackendLogitsTracePromptResult {
    pub user_prompt: String,
    pub result: ChatBackendLogitsTraceResult,
}

#[derive(Debug)]
pub struct ChatBackendLogitsTraceSuiteResult {
    pub prompts: Vec<ChatBackendLogitsTracePromptResult>,
}

#[derive(Debug, Clone)]
pub struct ChatForcedLogitsTraceOptions {
    pub logits_top_k: usize,
    pub system_prompt: SystemPrompt,
}

impl Default for ChatForcedLogitsTraceOptions {
    fn default() -> Self {
        Self {
            logits_top_k: 8,
            system_prompt: SystemPrompt::DefaultFromModel,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogitsTraceComparisonStep {
    pub step: usize,
    pub reference_token_id: u32,
    pub candidate_token_id: u32,
    pub kl_divergence: f64,
    pub max_abs_diff: f32,
}

#[derive(Debug, Clone)]
pub struct LogitsTraceComparison {
    pub steps: Vec<LogitsTraceComparisonStep>,
    pub mean_kl_divergence: f64,
    pub max_kl_divergence: f64,
    pub max_abs_diff: f32,
    pub token_ids_match: bool,
}

#[derive(Debug, Clone)]
pub struct ChatTraceComparisonSuiteOptions {
    pub max_new_tokens: usize,
    pub top_k: usize,
    pub logits_top_k: usize,
    pub system_prompt: SystemPrompt,
    pub stop_token_id: Option<u32>,
}

impl Default for ChatTraceComparisonSuiteOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 1,
            top_k: 1,
            logits_top_k: 8,
            system_prompt: SystemPrompt::DefaultFromModel,
            stop_token_id: None,
        }
    }
}

#[derive(Debug)]
pub struct ChatTraceComparisonPromptResult {
    pub prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub generated_text: String,
    pub comparison: LogitsTraceComparison,
}

#[derive(Debug)]
pub struct ChatTraceComparisonSuiteResult {
    pub prompts: Vec<ChatTraceComparisonPromptResult>,
    pub total_steps: usize,
    pub mean_kl_divergence: f64,
    pub max_kl_divergence: f64,
    pub max_abs_diff: f32,
    pub all_token_ids_match: bool,
}

#[derive(Debug)]
pub struct ChatAllLinearQuantizedComparisonResult {
    pub formatted_prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub generated_text: String,
    pub all_text_skip_special: String,
    pub reference_memory_stats: RuntimeMemoryStats,
    pub quantized_memory_stats: RuntimeMemoryStats,
    pub steps: Vec<AllLinearQuantizedRuntimeComparisonStepProbe>,
}

pub struct MinistralChatSession {
    tokenizer: TekkenTokenizer,
    default_system_prompt: String,
    runtime: MinistralTextRuntime,
}

pub struct MinistralAllLinearQuantizedChatSession {
    tokenizer: TekkenTokenizer,
    default_system_prompt: String,
    runtime: MinistralAllLinearQuantizedRuntime,
}

pub struct MinistralTextBackendSession {
    tokenizer: TekkenTokenizer,
    runtime: MinistralGenerationBackendSession,
}

pub struct MinistralGenerationBackendComparisonSession {
    reference: MinistralGenerationBackendSession,
    candidate: MinistralGenerationBackendSession,
}

pub struct MinistralTextBackendComparisonSession {
    tokenizer: TekkenTokenizer,
    reference: MinistralGenerationBackendSession,
    candidate: MinistralGenerationBackendSession,
}

pub struct MinistralChatBackendComparisonSession {
    tokenizer: TekkenTokenizer,
    default_system_prompt: String,
    reference: MinistralGenerationBackendSession,
    candidate: MinistralGenerationBackendSession,
}

pub enum MinistralGenerationBackendSession {
    Bf16(MinistralTextRuntime),
    AllLinearInt8(MinistralAllLinearQuantizedRuntime),
}

impl MinistralGenerationBackendSession {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
        backend: InferenceBackend,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        match backend {
            InferenceBackend::Bf16 => Ok(Self::Bf16(MinistralTextRuntime::new(
                stream,
                module,
                model_dir,
                max_seq_len,
            )?)),
            InferenceBackend::AllLinearInt8 => Ok(Self::AllLinearInt8(
                MinistralAllLinearQuantizedRuntime::new(stream, module, model_dir, max_seq_len)?,
            )),
        }
    }

    pub fn new_all_linear_int8_export(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        export_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        Ok(Self::AllLinearInt8(
            MinistralAllLinearQuantizedRuntime::new_from_export(
                stream,
                module,
                model_dir,
                export_dir,
                max_seq_len,
            )?,
        ))
    }

    pub fn backend(&self) -> InferenceBackend {
        match self {
            Self::Bf16(_) => InferenceBackend::Bf16,
            Self::AllLinearInt8(_) => InferenceBackend::AllLinearInt8,
        }
    }

    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<()> {
        match self {
            Self::Bf16(runtime) => runtime.prefill(prompt_tokens),
            Self::AllLinearInt8(runtime) => runtime.prefill(prompt_tokens),
        }
    }

    pub fn generate_greedy_until(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<Vec<GreedyGenerationStep>> {
        match self {
            Self::Bf16(runtime) => {
                runtime.generate_greedy_until(max_new_tokens, top_k, stop_token_id)
            }
            Self::AllLinearInt8(runtime) => {
                runtime.generate_greedy_until(max_new_tokens, top_k, stop_token_id)
            }
        }
    }

    pub fn generate_top_k_sample_until(
        &mut self,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
        sampling: SamplingOptions,
    ) -> Result<Vec<GreedyGenerationStep>> {
        generate_sampled_top_k_until(self, max_new_tokens, top_k, stop_token_id, sampling)
    }

    pub fn next_logits_to_host(&self) -> Result<Vec<f32>> {
        match self {
            Self::Bf16(runtime) => runtime.next_logits_to_host(),
            Self::AllLinearInt8(runtime) => runtime.next_logits_to_host(),
        }
    }

    pub fn next_top_logits(&mut self, top_k: usize) -> Result<Vec<(u32, f32)>> {
        match self {
            Self::Bf16(runtime) => runtime.next_top_logits(top_k),
            Self::AllLinearInt8(runtime) => runtime.next_top_logits(top_k),
        }
    }

    pub fn advance_with_token(&mut self, token_id: u32) -> Result<()> {
        match self {
            Self::Bf16(runtime) => runtime.advance_with_token(token_id),
            Self::AllLinearInt8(runtime) => runtime.advance_with_token(token_id),
        }
    }

    pub fn reset_sequence(&mut self) {
        match self {
            Self::Bf16(runtime) => runtime.reset_sequence(),
            Self::AllLinearInt8(runtime) => runtime.reset_sequence(),
        }
    }

    pub fn tokens(&self) -> &[u32] {
        match self {
            Self::Bf16(runtime) => runtime.tokens(),
            Self::AllLinearInt8(runtime) => runtime.tokens(),
        }
    }

    pub fn max_seq_len(&self) -> usize {
        match self {
            Self::Bf16(runtime) => runtime.max_seq_len(),
            Self::AllLinearInt8(runtime) => runtime.max_seq_len(),
        }
    }

    pub fn memory_stats(&self) -> RuntimeMemoryStats {
        match self {
            Self::Bf16(runtime) => runtime.memory_stats(),
            Self::AllLinearInt8(runtime) => runtime.memory_stats(),
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        match self {
            Self::Bf16(runtime) => runtime.synchronize(),
            Self::AllLinearInt8(runtime) => runtime.synchronize(),
        }
    }

    pub fn run_generation(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerationOptions,
    ) -> Result<GenerationResult> {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            options.max_new_tokens,
            "backend generation request",
        )?;

        if let Self::Bf16(runtime) = self {
            if let Some(result) = run_bf16_top1_generation_for_request(
                runtime,
                prompt_tokens,
                options,
                "backend generation request",
            )? {
                return Ok(result);
            }
        }

        let prefill_start = Instant::now();
        prefill_runtime_for_request(
            self,
            prompt_tokens,
            requested_seq_len,
            "backend generation request",
        )?;
        self.synchronize()?;
        let prefill_seconds = prefill_start.elapsed().as_secs_f64();
        let memory_stats = self.memory_stats();
        let backend = self.backend();
        let decode_start = Instant::now();
        let steps = self.generate_greedy_until(
            options.max_new_tokens,
            options.top_k,
            options.stop_token_id,
        )?;
        self.synchronize()?;
        let decode_seconds = decode_start.elapsed().as_secs_f64();
        let all_tokens = self.tokens().to_vec();

        Ok(GenerationResult {
            backend,
            prompt_tokens: prompt_tokens.to_vec(),
            generated_tokens: all_tokens[prompt_tokens.len()..].to_vec(),
            all_tokens,
            memory_stats,
            timings: GenerationTimings::new(prefill_seconds, decode_seconds),
            finish_reason: generation_finish_reason(&steps, options.stop_token_id),
            steps,
        })
    }

    pub fn run_generation_sampled(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerationOptions,
        sampling: SamplingOptions,
    ) -> Result<GenerationResult> {
        validate_sampling_options(sampling)?;
        if options.top_k.max(1) == 1 {
            return self.run_generation_request(
                prompt_tokens,
                options,
                "backend sampled generation request",
            );
        }

        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            options.max_new_tokens,
            "backend sampled generation request",
        )?;
        let prefill_start = Instant::now();
        prefill_runtime_for_request(
            self,
            prompt_tokens,
            requested_seq_len,
            "backend sampled generation request",
        )?;
        self.synchronize()?;
        let prefill_seconds = prefill_start.elapsed().as_secs_f64();
        let memory_stats = self.memory_stats();
        let backend = self.backend();
        let decode_start = Instant::now();
        let steps = self.generate_top_k_sample_until(
            options.max_new_tokens,
            options.top_k,
            options.stop_token_id,
            sampling,
        )?;
        self.synchronize()?;
        let decode_seconds = decode_start.elapsed().as_secs_f64();
        let all_tokens = self.tokens().to_vec();

        Ok(GenerationResult {
            backend,
            prompt_tokens: prompt_tokens.to_vec(),
            generated_tokens: all_tokens[prompt_tokens.len()..].to_vec(),
            all_tokens,
            memory_stats,
            timings: GenerationTimings::new(prefill_seconds, decode_seconds),
            finish_reason: generation_finish_reason(&steps, options.stop_token_id),
            steps,
        })
    }

    pub fn run_generation_suite(
        &mut self,
        prompts: &[Vec<u32>],
        options: &GenerationOptions,
    ) -> Result<GenerationSuiteResult> {
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "generation suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        for prompt_tokens in prompts {
            results.push(self.run_generation(prompt_tokens, options)?);
        }

        Ok(GenerationSuiteResult { prompts: results })
    }

    pub fn run_generation_sampled_suite(
        &mut self,
        prompts: &[Vec<u32>],
        options: &GenerationOptions,
        sampling: SamplingOptions,
    ) -> Result<GenerationSuiteResult> {
        validate_sampling_options(sampling)?;
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "sampled generation suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        for (round, prompt_tokens) in prompts.iter().enumerate() {
            let prompt_sampling = SamplingOptions {
                seed: sampling.seed.wrapping_add(round as u64),
                ..sampling
            };
            results.push(self.run_generation_sampled(prompt_tokens, options, prompt_sampling)?);
        }

        Ok(GenerationSuiteResult { prompts: results })
    }

    pub fn run_next_logits(
        &mut self,
        prompt_tokens: &[u32],
        top_k: usize,
    ) -> Result<GenerationLogitsResult> {
        let requested_seq_len =
            requested_seq_len_for_prompt(prompt_tokens, 1, "backend logits request")?;
        prefill_runtime_for_request(
            self,
            prompt_tokens,
            requested_seq_len,
            "backend logits request",
        )?;
        let memory_stats = self.memory_stats();
        let backend = self.backend();
        let snapshot = collect_prefilled_next_logits(self, top_k)?;

        Ok(GenerationLogitsResult {
            backend,
            prompt_tokens: prompt_tokens.to_vec(),
            logits: snapshot.logits,
            logsumexp: snapshot.logsumexp,
            top_logits: snapshot.top_logits,
            memory_stats,
        })
    }

    pub fn run_next_logits_suite(
        &mut self,
        prompts: &[Vec<u32>],
        top_k: usize,
    ) -> Result<GenerationLogitsSuiteResult> {
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "generation logits suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        for prompt_tokens in prompts {
            results.push(self.run_next_logits(prompt_tokens, top_k)?);
        }

        Ok(GenerationLogitsSuiteResult { prompts: results })
    }

    pub fn run_logits_trace(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerationLogitsTraceOptions,
    ) -> Result<GenerationLogitsTraceResult> {
        run_generation_logits_trace_with_session(self, prompt_tokens, options)
    }

    pub fn run_logits_trace_suite(
        &mut self,
        prompts: &[Vec<u32>],
        options: &GenerationLogitsTraceOptions,
    ) -> Result<GenerationLogitsTraceSuiteResult> {
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "generation logits trace suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        for prompt_tokens in prompts {
            results.push(self.run_logits_trace(prompt_tokens, options)?);
        }

        Ok(GenerationLogitsTraceSuiteResult { prompts: results })
    }
}

impl MinistralGenerationBackendComparisonSession {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
        candidate_backend: InferenceBackend,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let reference = MinistralGenerationBackendSession::new(
            stream.clone(),
            module.clone(),
            model_dir,
            max_seq_len,
            InferenceBackend::Bf16,
        )?;
        let candidate = MinistralGenerationBackendSession::new(
            stream,
            module,
            model_dir,
            max_seq_len,
            candidate_backend,
        )?;

        Ok(Self {
            reference,
            candidate,
        })
    }

    pub fn new_all_linear_int8_export(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        export_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let reference = MinistralGenerationBackendSession::new(
            stream.clone(),
            module.clone(),
            model_dir,
            max_seq_len,
            InferenceBackend::Bf16,
        )?;
        let candidate = MinistralGenerationBackendSession::new_all_linear_int8_export(
            stream,
            module,
            model_dir,
            export_dir,
            max_seq_len,
        )?;

        Ok(Self {
            reference,
            candidate,
        })
    }

    pub fn reference_backend(&self) -> InferenceBackend {
        self.reference.backend()
    }

    pub fn candidate_backend(&self) -> InferenceBackend {
        self.candidate.backend()
    }

    pub fn max_seq_len(&self) -> usize {
        self.reference
            .max_seq_len()
            .min(self.candidate.max_seq_len())
    }

    pub fn reference_memory_stats(&self) -> RuntimeMemoryStats {
        self.reference.memory_stats()
    }

    pub fn candidate_memory_stats(&self) -> RuntimeMemoryStats {
        self.candidate.memory_stats()
    }

    pub fn compare_next_logits(
        &mut self,
        prompt_tokens: &[u32],
        top_k: usize,
    ) -> Result<GenerationBackendLogitsComparisonResult> {
        run_generation_backend_next_logits_comparison_with_sessions(
            &mut self.reference,
            &mut self.candidate,
            prompt_tokens,
            top_k,
        )
    }

    pub fn compare_next_logits_suite(
        &mut self,
        prompts: &[Vec<u32>],
        top_k: usize,
    ) -> Result<GenerationBackendLogitsComparisonSuiteResult> {
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "generation logits comparison suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        let mut summary = GenerationBackendLogitsComparisonSuiteSummary::default();
        for prompt_tokens in prompts {
            let result = self.compare_next_logits(prompt_tokens, top_k)?;
            summary.observe_comparison(&result);
            results.push(result);
        }

        Ok(GenerationBackendLogitsComparisonSuiteResult {
            prompts: results,
            summary,
            reference_memory_stats: self.reference_memory_stats(),
            candidate_memory_stats: self.candidate_memory_stats(),
        })
    }

    pub fn compare_generation(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerationLogitsTraceOptions,
    ) -> Result<GenerationBackendComparisonResult> {
        self.compare_generation_with_driver(
            prompt_tokens,
            options,
            GenerationComparisonDriver::Candidate,
        )
    }

    pub fn compare_generation_with_driver(
        &mut self,
        prompt_tokens: &[u32],
        options: &GenerationLogitsTraceOptions,
        driver: GenerationComparisonDriver,
    ) -> Result<GenerationBackendComparisonResult> {
        run_generation_backend_comparison_with_sessions_and_driver(
            &mut self.reference,
            &mut self.candidate,
            prompt_tokens,
            options,
            driver,
        )
    }

    pub fn compare_generation_suite(
        &mut self,
        prompts: &[Vec<u32>],
        options: &GenerationLogitsTraceOptions,
    ) -> Result<GenerationBackendComparisonSuiteResult> {
        self.compare_generation_suite_with_driver(
            prompts,
            options,
            GenerationComparisonDriver::Candidate,
        )
    }

    pub fn compare_generation_suite_with_driver(
        &mut self,
        prompts: &[Vec<u32>],
        options: &GenerationLogitsTraceOptions,
        driver: GenerationComparisonDriver,
    ) -> Result<GenerationBackendComparisonSuiteResult> {
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "generation comparison suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        let mut summary = GenerationBackendComparisonSuiteSummary::default();
        for prompt_tokens in prompts {
            let result = self.compare_generation_with_driver(prompt_tokens, options, driver)?;
            summary.observe_comparison(&result);
            results.push(result);
        }

        Ok(GenerationBackendComparisonSuiteResult {
            prompts: results,
            summary,
            reference_memory_stats: self.reference_memory_stats(),
            candidate_memory_stats: self.candidate_memory_stats(),
        })
    }

    pub fn compare_forced_sequence(
        &mut self,
        sequence_tokens: &[u32],
        logits_top_k: usize,
    ) -> Result<GenerationBackendForcedComparisonResult> {
        run_generation_backend_forced_comparison_with_sessions(
            &mut self.reference,
            &mut self.candidate,
            sequence_tokens,
            logits_top_k,
        )
    }

    pub fn compare_forced_sequence_suite(
        &mut self,
        sequences: &[Vec<u32>],
        logits_top_k: usize,
    ) -> Result<GenerationBackendForcedComparisonSuiteResult> {
        if sequences.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "forced comparison suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(sequences.len());
        let mut summary = GenerationBackendForcedComparisonSuiteSummary::default();
        for sequence_tokens in sequences {
            let result = self.compare_forced_sequence(sequence_tokens, logits_top_k)?;
            summary.observe_comparison(&result);
            results.push(result);
        }

        Ok(GenerationBackendForcedComparisonSuiteResult {
            prompts: results,
            summary,
            reference_memory_stats: self.reference_memory_stats(),
            candidate_memory_stats: self.candidate_memory_stats(),
        })
    }
}

impl MinistralTextBackendSession {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
        backend: InferenceBackend,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let runtime = MinistralGenerationBackendSession::new(
            stream,
            module,
            model_dir,
            max_seq_len,
            backend,
        )?;

        Ok(Self { tokenizer, runtime })
    }

    pub fn new_all_linear_int8_export(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        export_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let runtime = MinistralGenerationBackendSession::new_all_linear_int8_export(
            stream,
            module,
            model_dir,
            export_dir,
            max_seq_len,
        )?;

        Ok(Self { tokenizer, runtime })
    }

    pub fn backend(&self) -> InferenceBackend {
        self.runtime.backend()
    }

    pub fn max_seq_len(&self) -> usize {
        self.runtime.max_seq_len()
    }

    pub fn memory_stats(&self) -> RuntimeMemoryStats {
        self.runtime.memory_stats()
    }

    pub fn run_text_generation(
        &mut self,
        prompt_text: &str,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<TextGenerationResult> {
        let prompt_tokens = self.tokenizer.encode_lossy(prompt_text, true)?;
        let generation = self.runtime.run_generation(
            &prompt_tokens,
            &GenerationOptions {
                max_new_tokens,
                top_k,
                stop_token_id,
            },
        )?;
        let generated_text = self.tokenizer.decode_lossy(&generation.generated_tokens)?;
        let all_text_skip_special = self.tokenizer.decode_lossy(&generation.all_tokens)?;
        let finish_reason = generation.finish_reason;

        Ok(TextGenerationResult {
            backend: generation.backend,
            prompt_text: prompt_text.to_string(),
            prompt_tokens: generation.prompt_tokens,
            generated_tokens: generation.generated_tokens,
            all_tokens: generation.all_tokens,
            memory_stats: generation.memory_stats,
            timings: generation.timings,
            generated_text,
            all_text_skip_special,
            finish_reason,
            steps: generation.steps,
        })
    }

    pub fn run_text_generation_sampled(
        &mut self,
        prompt_text: &str,
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
        sampling: SamplingOptions,
    ) -> Result<TextGenerationResult> {
        let prompt_tokens = self.tokenizer.encode_lossy(prompt_text, true)?;
        let generation = self.runtime.run_generation_sampled(
            &prompt_tokens,
            &GenerationOptions {
                max_new_tokens,
                top_k,
                stop_token_id,
            },
            sampling,
        )?;
        let generated_text = self.tokenizer.decode_lossy(&generation.generated_tokens)?;
        let all_text_skip_special = self.tokenizer.decode_lossy(&generation.all_tokens)?;
        let finish_reason = generation.finish_reason;

        Ok(TextGenerationResult {
            backend: generation.backend,
            prompt_text: prompt_text.to_string(),
            prompt_tokens: generation.prompt_tokens,
            generated_tokens: generation.generated_tokens,
            all_tokens: generation.all_tokens,
            memory_stats: generation.memory_stats,
            timings: generation.timings,
            generated_text,
            all_text_skip_special,
            finish_reason,
            steps: generation.steps,
        })
    }

    pub fn run_text_generation_suite(
        &mut self,
        prompts: &[String],
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
    ) -> Result<TextGenerationSuiteResult> {
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "text generation suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            results.push(self.run_text_generation(prompt, max_new_tokens, top_k, stop_token_id)?);
        }

        Ok(TextGenerationSuiteResult { prompts: results })
    }

    pub fn run_text_generation_sampled_suite(
        &mut self,
        prompts: &[String],
        max_new_tokens: usize,
        top_k: usize,
        stop_token_id: Option<u32>,
        sampling: SamplingOptions,
    ) -> Result<TextGenerationSuiteResult> {
        validate_sampling_options(sampling)?;
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "sampled text generation suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        for (round, prompt) in prompts.iter().enumerate() {
            let prompt_sampling = SamplingOptions {
                seed: sampling.seed.wrapping_add(round as u64),
                ..sampling
            };
            results.push(self.run_text_generation_sampled(
                prompt,
                max_new_tokens,
                top_k,
                stop_token_id,
                prompt_sampling,
            )?);
        }

        Ok(TextGenerationSuiteResult { prompts: results })
    }

    pub fn run_text_next_logits(
        &mut self,
        prompt_text: &str,
        top_k: usize,
    ) -> Result<TextGenerationLogitsResult> {
        let prompt_tokens = self.tokenizer.encode_lossy(prompt_text, true)?;
        let result = self.runtime.run_next_logits(&prompt_tokens, top_k)?;

        Ok(TextGenerationLogitsResult {
            prompt_text: prompt_text.to_string(),
            result,
        })
    }

    pub fn run_text_next_logits_suite(
        &mut self,
        prompts: &[String],
        top_k: usize,
    ) -> Result<TextGenerationLogitsSuiteResult> {
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "text logits suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            results.push(self.run_text_next_logits(prompt, top_k)?);
        }

        Ok(TextGenerationLogitsSuiteResult { prompts: results })
    }

    pub fn run_text_logits_trace(
        &mut self,
        prompt_text: &str,
        options: &GenerationLogitsTraceOptions,
    ) -> Result<TextGenerationBackendLogitsTraceResult> {
        let prompt_tokens = self.tokenizer.encode_lossy(prompt_text, true)?;
        let trace = self.runtime.run_logits_trace(&prompt_tokens, options)?;
        let generated_text = self.tokenizer.decode_lossy(&trace.generated_tokens)?;
        let all_text_skip_special = self.tokenizer.decode_lossy(&trace.all_tokens)?;

        Ok(TextGenerationBackendLogitsTraceResult {
            prompt_text: prompt_text.to_string(),
            generated_text,
            all_text_skip_special,
            trace,
        })
    }

    pub fn run_text_logits_trace_suite(
        &mut self,
        prompts: &[String],
        options: &GenerationLogitsTraceOptions,
    ) -> Result<TextGenerationBackendLogitsTraceSuiteResult> {
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "text logits trace suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            results.push(self.run_text_logits_trace(prompt, options)?);
        }

        Ok(TextGenerationBackendLogitsTraceSuiteResult { prompts: results })
    }
}

impl MinistralTextBackendComparisonSession {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
        candidate_backend: InferenceBackend,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let reference = MinistralGenerationBackendSession::new(
            stream.clone(),
            module.clone(),
            model_dir,
            max_seq_len,
            InferenceBackend::Bf16,
        )?;
        let candidate = MinistralGenerationBackendSession::new(
            stream,
            module,
            model_dir,
            max_seq_len,
            candidate_backend,
        )?;

        Ok(Self {
            tokenizer,
            reference,
            candidate,
        })
    }

    pub fn new_all_linear_int8_export(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        export_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let reference = MinistralGenerationBackendSession::new(
            stream.clone(),
            module.clone(),
            model_dir,
            max_seq_len,
            InferenceBackend::Bf16,
        )?;
        let candidate = MinistralGenerationBackendSession::new_all_linear_int8_export(
            stream,
            module,
            model_dir,
            export_dir,
            max_seq_len,
        )?;

        Ok(Self {
            tokenizer,
            reference,
            candidate,
        })
    }

    pub fn reference_backend(&self) -> InferenceBackend {
        self.reference.backend()
    }

    pub fn candidate_backend(&self) -> InferenceBackend {
        self.candidate.backend()
    }

    pub fn max_seq_len(&self) -> usize {
        self.reference
            .max_seq_len()
            .min(self.candidate.max_seq_len())
    }

    pub fn reference_memory_stats(&self) -> RuntimeMemoryStats {
        self.reference.memory_stats()
    }

    pub fn candidate_memory_stats(&self) -> RuntimeMemoryStats {
        self.candidate.memory_stats()
    }

    pub fn compare_text_next_logits(
        &mut self,
        prompt_text: &str,
        top_k: usize,
    ) -> Result<TextGenerationBackendLogitsComparisonResult> {
        let prompt_tokens = self.tokenizer.encode_lossy(prompt_text, true)?;
        let comparison = run_generation_backend_next_logits_comparison_with_sessions(
            &mut self.reference,
            &mut self.candidate,
            &prompt_tokens,
            top_k,
        )?;

        Ok(TextGenerationBackendLogitsComparisonResult {
            prompt_text: prompt_text.to_string(),
            comparison,
        })
    }

    pub fn compare_text_next_logits_suite(
        &mut self,
        prompts: &[String],
        top_k: usize,
    ) -> Result<TextGenerationBackendLogitsComparisonSuiteResult> {
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "text logits comparison suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        let mut summary = GenerationBackendLogitsComparisonSuiteSummary::default();
        for prompt in prompts {
            let result = self.compare_text_next_logits(prompt, top_k)?;
            summary.observe_comparison(&result.comparison);
            results.push(result);
        }

        Ok(TextGenerationBackendLogitsComparisonSuiteResult {
            prompts: results,
            summary,
            reference_memory_stats: self.reference_memory_stats(),
            candidate_memory_stats: self.candidate_memory_stats(),
        })
    }

    pub fn compare_text_generation(
        &mut self,
        prompt_text: &str,
        options: &GenerationLogitsTraceOptions,
    ) -> Result<TextGenerationBackendComparisonResult> {
        self.compare_text_generation_with_driver(
            prompt_text,
            options,
            GenerationComparisonDriver::Candidate,
        )
    }

    pub fn compare_text_generation_with_driver(
        &mut self,
        prompt_text: &str,
        options: &GenerationLogitsTraceOptions,
        driver: GenerationComparisonDriver,
    ) -> Result<TextGenerationBackendComparisonResult> {
        let prompt_tokens = self.tokenizer.encode_lossy(prompt_text, true)?;
        let comparison = run_generation_backend_comparison_with_sessions_and_driver(
            &mut self.reference,
            &mut self.candidate,
            &prompt_tokens,
            options,
            driver,
        )?;
        let generated_text = self.tokenizer.decode_lossy(&comparison.generated_tokens)?;
        let all_text_skip_special = self.tokenizer.decode_lossy(&comparison.all_tokens)?;

        Ok(TextGenerationBackendComparisonResult {
            prompt_text: prompt_text.to_string(),
            generated_text,
            all_text_skip_special,
            comparison,
        })
    }

    pub fn compare_text_generation_suite(
        &mut self,
        prompts: &[String],
        options: &GenerationLogitsTraceOptions,
    ) -> Result<TextGenerationBackendComparisonSuiteResult> {
        self.compare_text_generation_suite_with_driver(
            prompts,
            options,
            GenerationComparisonDriver::Candidate,
        )
    }

    pub fn compare_text_generation_suite_with_driver(
        &mut self,
        prompts: &[String],
        options: &GenerationLogitsTraceOptions,
        driver: GenerationComparisonDriver,
    ) -> Result<TextGenerationBackendComparisonSuiteResult> {
        if prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "text comparison suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(prompts.len());
        let mut summary = GenerationBackendComparisonSuiteSummary::default();
        for prompt in prompts {
            let result = self.compare_text_generation_with_driver(prompt, options, driver)?;
            summary.observe_comparison(&result.comparison);
            results.push(result);
        }

        Ok(TextGenerationBackendComparisonSuiteResult {
            prompts: results,
            summary,
            reference_memory_stats: self.reference_memory_stats(),
            candidate_memory_stats: self.candidate_memory_stats(),
        })
    }

    pub fn compare_text_forced_sequence(
        &mut self,
        sequence_text: &str,
        logits_top_k: usize,
    ) -> Result<TextGenerationBackendForcedComparisonResult> {
        let sequence_tokens = self.tokenizer.encode_lossy(sequence_text, true)?;
        let comparison = run_generation_backend_forced_comparison_with_sessions(
            &mut self.reference,
            &mut self.candidate,
            &sequence_tokens,
            logits_top_k,
        )?;

        Ok(TextGenerationBackendForcedComparisonResult {
            sequence_text: sequence_text.to_string(),
            comparison,
        })
    }

    pub fn compare_text_forced_sequence_suite(
        &mut self,
        sequences: &[String],
        logits_top_k: usize,
    ) -> Result<TextGenerationBackendForcedComparisonSuiteResult> {
        if sequences.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "text forced comparison suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(sequences.len());
        let mut summary = GenerationBackendForcedComparisonSuiteSummary::default();
        for sequence_text in sequences {
            let result = self.compare_text_forced_sequence(sequence_text, logits_top_k)?;
            summary.observe_comparison(&result.comparison);
            results.push(result);
        }

        Ok(TextGenerationBackendForcedComparisonSuiteResult {
            prompts: results,
            summary,
            reference_memory_stats: self.reference_memory_stats(),
            candidate_memory_stats: self.candidate_memory_stats(),
        })
    }

    pub fn compare_text_forced_target(
        &mut self,
        prompt_text: &str,
        target_text: &str,
        logits_top_k: usize,
    ) -> Result<TextGenerationBackendForcedTargetComparisonResult> {
        let prompt_tokens = self.tokenizer.encode_lossy(prompt_text, true)?;
        let forced_tokens = self.tokenizer.encode_lossy(target_text, false)?;
        let comparison = run_generation_backend_forced_target_comparison_with_sessions(
            &mut self.reference,
            &mut self.candidate,
            &prompt_tokens,
            &forced_tokens,
            "text forced target comparison request",
            logits_top_k,
        )?;

        Ok(TextGenerationBackendForcedTargetComparisonResult {
            prompt_text: prompt_text.to_string(),
            target_text: target_text.to_string(),
            comparison,
        })
    }

    pub fn compare_text_forced_target_suite(
        &mut self,
        pairs: &[(String, String)],
        logits_top_k: usize,
    ) -> Result<TextGenerationBackendForcedTargetComparisonSuiteResult> {
        if pairs.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "text forced target comparison suite must be nonempty",
            )
            .into());
        }

        let mut results = Vec::with_capacity(pairs.len());
        let mut summary = GenerationBackendForcedComparisonSuiteSummary::default();
        for (prompt_text, target_text) in pairs {
            let result = self.compare_text_forced_target(prompt_text, target_text, logits_top_k)?;
            summary.observe_comparison(&result.comparison);
            results.push(result);
        }

        Ok(TextGenerationBackendForcedTargetComparisonSuiteResult {
            prompts: results,
            summary,
            reference_memory_stats: self.reference_memory_stats(),
            candidate_memory_stats: self.candidate_memory_stats(),
        })
    }
}

impl MinistralChatBackendComparisonSession {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
        candidate_backend: InferenceBackend,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let default_system_prompt = chat::default_ministral_system_prompt(model_dir)?;
        let reference = MinistralGenerationBackendSession::new(
            stream.clone(),
            module.clone(),
            model_dir,
            max_seq_len,
            InferenceBackend::Bf16,
        )?;
        let candidate = MinistralGenerationBackendSession::new(
            stream,
            module,
            model_dir,
            max_seq_len,
            candidate_backend,
        )?;

        Ok(Self {
            tokenizer,
            default_system_prompt,
            reference,
            candidate,
        })
    }

    pub fn new_all_linear_int8_export(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        export_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let default_system_prompt = chat::default_ministral_system_prompt(model_dir)?;
        let reference = MinistralGenerationBackendSession::new(
            stream.clone(),
            module.clone(),
            model_dir,
            max_seq_len,
            InferenceBackend::Bf16,
        )?;
        let candidate = MinistralGenerationBackendSession::new_all_linear_int8_export(
            stream,
            module,
            model_dir,
            export_dir,
            max_seq_len,
        )?;

        Ok(Self {
            tokenizer,
            default_system_prompt,
            reference,
            candidate,
        })
    }

    pub fn reference_backend(&self) -> InferenceBackend {
        self.reference.backend()
    }

    pub fn candidate_backend(&self) -> InferenceBackend {
        self.candidate.backend()
    }

    pub fn max_seq_len(&self) -> usize {
        self.reference
            .max_seq_len()
            .min(self.candidate.max_seq_len())
    }

    pub fn reference_memory_stats(&self) -> RuntimeMemoryStats {
        self.reference.memory_stats()
    }

    pub fn candidate_memory_stats(&self) -> RuntimeMemoryStats {
        self.candidate.memory_stats()
    }

    pub fn compare_single_turn_chat_logits(
        &mut self,
        user_prompt: &str,
        options: &ChatLogitsOptions,
    ) -> Result<ChatBackendLogitsComparisonResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        let comparison = run_generation_backend_next_logits_comparison_with_sessions(
            &mut self.reference,
            &mut self.candidate,
            &encoded.prompt_tokens,
            options.top_k,
        )?;

        Ok(ChatBackendLogitsComparisonResult {
            formatted_prompt: encoded.formatted_prompt,
            comparison,
        })
    }

    pub fn compare_single_turn_chat_logits_suite(
        &mut self,
        user_prompts: &[String],
        options: &ChatLogitsOptions,
    ) -> Result<ChatBackendLogitsComparisonSuiteResult> {
        if user_prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "chat logits comparison suite must be nonempty",
            )
            .into());
        }

        let mut prompt_results = Vec::with_capacity(user_prompts.len());
        let mut summary = GenerationBackendLogitsComparisonSuiteSummary::default();
        for user_prompt in user_prompts {
            let result = self.compare_single_turn_chat_logits(user_prompt, options)?;
            summary.observe_comparison(&result.comparison);
            prompt_results.push(ChatBackendLogitsComparisonPromptResult {
                user_prompt: user_prompt.clone(),
                result,
            });
        }

        Ok(ChatBackendLogitsComparisonSuiteResult {
            prompts: prompt_results,
            summary,
            reference_memory_stats: self.reference_memory_stats(),
            candidate_memory_stats: self.candidate_memory_stats(),
        })
    }

    pub fn compare_single_turn_chat(
        &mut self,
        user_prompt: &str,
        options: &ChatTraceComparisonSuiteOptions,
    ) -> Result<ChatBackendComparisonResult> {
        self.compare_single_turn_chat_with_driver(
            user_prompt,
            options,
            GenerationComparisonDriver::Candidate,
        )
    }

    pub fn compare_single_turn_chat_with_driver(
        &mut self,
        user_prompt: &str,
        options: &ChatTraceComparisonSuiteOptions,
        driver: GenerationComparisonDriver,
    ) -> Result<ChatBackendComparisonResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        let generation_options = GenerationLogitsTraceOptions {
            max_new_tokens: options.max_new_tokens,
            top_k: options.top_k,
            logits_top_k: options.logits_top_k,
            stop_token_id: options.stop_token_id,
        };
        let comparison = run_generation_backend_comparison_with_sessions_and_driver(
            &mut self.reference,
            &mut self.candidate,
            &encoded.prompt_tokens,
            &generation_options,
            driver,
        )?;
        let generated_text = self.tokenizer.decode_lossy(&comparison.generated_tokens)?;
        let all_text_skip_special = self.tokenizer.decode_lossy(&comparison.all_tokens)?;

        Ok(ChatBackendComparisonResult {
            formatted_prompt: encoded.formatted_prompt,
            generated_text,
            all_text_skip_special,
            comparison,
        })
    }

    pub fn compare_single_turn_chat_suite(
        &mut self,
        user_prompts: &[String],
        options: &ChatTraceComparisonSuiteOptions,
    ) -> Result<ChatBackendComparisonSuiteResult> {
        self.compare_single_turn_chat_suite_with_driver(
            user_prompts,
            options,
            GenerationComparisonDriver::Candidate,
        )
    }

    pub fn compare_single_turn_chat_suite_with_driver(
        &mut self,
        user_prompts: &[String],
        options: &ChatTraceComparisonSuiteOptions,
        driver: GenerationComparisonDriver,
    ) -> Result<ChatBackendComparisonSuiteResult> {
        if user_prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "chat comparison suite must be nonempty",
            )
            .into());
        }

        let mut prompt_results = Vec::with_capacity(user_prompts.len());
        let mut summary = GenerationBackendComparisonSuiteSummary::default();
        for user_prompt in user_prompts {
            let result = self.compare_single_turn_chat_with_driver(user_prompt, options, driver)?;
            summary.observe_comparison(&result.comparison);
            prompt_results.push(ChatBackendComparisonPromptResult {
                user_prompt: user_prompt.clone(),
                result,
            });
        }

        Ok(ChatBackendComparisonSuiteResult {
            prompts: prompt_results,
            summary,
            reference_memory_stats: self.reference_memory_stats(),
            candidate_memory_stats: self.candidate_memory_stats(),
        })
    }

    pub fn compare_single_turn_chat_forced_target(
        &mut self,
        user_prompt: &str,
        target_text: &str,
        options: &ChatForcedLogitsTraceOptions,
    ) -> Result<ChatBackendForcedTargetComparisonResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        let forced_tokens = self.tokenizer.encode_lossy(target_text, false)?;
        let comparison = run_generation_backend_forced_target_comparison_with_sessions(
            &mut self.reference,
            &mut self.candidate,
            &encoded.prompt_tokens,
            &forced_tokens,
            "chat forced target comparison request",
            options.logits_top_k,
        )?;

        Ok(ChatBackendForcedTargetComparisonResult {
            formatted_prompt: encoded.formatted_prompt,
            target_text: target_text.to_string(),
            comparison,
        })
    }

    pub fn compare_single_turn_chat_forced_target_suite(
        &mut self,
        pairs: &[(String, String)],
        options: &ChatForcedLogitsTraceOptions,
    ) -> Result<ChatBackendForcedTargetComparisonSuiteResult> {
        if pairs.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "chat forced target comparison suite must be nonempty",
            )
            .into());
        }

        let mut prompt_results = Vec::with_capacity(pairs.len());
        let mut summary = GenerationBackendForcedComparisonSuiteSummary::default();
        for (user_prompt, target_text) in pairs {
            let result =
                self.compare_single_turn_chat_forced_target(user_prompt, target_text, options)?;
            summary.observe_comparison(&result.comparison);
            prompt_results.push(ChatBackendForcedTargetComparisonPromptResult {
                user_prompt: user_prompt.clone(),
                result,
            });
        }

        Ok(ChatBackendForcedTargetComparisonSuiteResult {
            prompts: prompt_results,
            summary,
            reference_memory_stats: self.reference_memory_stats(),
            candidate_memory_stats: self.candidate_memory_stats(),
        })
    }
}

pub struct MinistralChatBackendSession {
    tokenizer: TekkenTokenizer,
    default_system_prompt: String,
    runtime: MinistralGenerationBackendSession,
}

impl MinistralChatBackendSession {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
        backend: InferenceBackend,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let default_system_prompt = chat::default_ministral_system_prompt(model_dir)?;
        let runtime = MinistralGenerationBackendSession::new(
            stream,
            module,
            model_dir,
            max_seq_len,
            backend,
        )?;

        Ok(Self {
            tokenizer,
            default_system_prompt,
            runtime,
        })
    }

    pub fn new_all_linear_int8_export(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        export_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let default_system_prompt = chat::default_ministral_system_prompt(model_dir)?;
        let runtime = MinistralGenerationBackendSession::new_all_linear_int8_export(
            stream,
            module,
            model_dir,
            export_dir,
            max_seq_len,
        )?;

        Ok(Self {
            tokenizer,
            default_system_prompt,
            runtime,
        })
    }

    pub fn backend(&self) -> InferenceBackend {
        self.runtime.backend()
    }

    pub fn max_seq_len(&self) -> usize {
        self.runtime.max_seq_len()
    }

    pub fn memory_stats(&self) -> RuntimeMemoryStats {
        self.runtime.memory_stats()
    }

    pub fn run_single_turn_chat(
        &mut self,
        user_prompt: &str,
        options: &ChatInferenceOptions,
    ) -> Result<ChatInferenceResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        run_encoded_single_turn_chat(
            &self.tokenizer,
            &mut self.runtime,
            encoded.formatted_prompt,
            encoded.prompt_tokens,
            options,
        )
    }

    pub fn run_single_turn_chat_sampled(
        &mut self,
        user_prompt: &str,
        options: &ChatInferenceOptions,
        sampling: SamplingOptions,
    ) -> Result<ChatInferenceResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        run_encoded_single_turn_chat_sampled(
            &self.tokenizer,
            &mut self.runtime,
            encoded.formatted_prompt,
            encoded.prompt_tokens,
            options,
            sampling,
        )
    }

    pub fn run_single_turn_chat_suite(
        &mut self,
        user_prompts: &[String],
        options: &ChatInferenceOptions,
    ) -> Result<ChatGenerationSuiteResult> {
        if user_prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "chat generation suite must be nonempty",
            )
            .into());
        }

        let mut prompt_results = Vec::with_capacity(user_prompts.len());
        for user_prompt in user_prompts {
            let result = self.run_single_turn_chat(user_prompt, options)?;
            prompt_results.push(ChatGenerationPromptResult {
                user_prompt: user_prompt.clone(),
                result,
            });
        }

        Ok(ChatGenerationSuiteResult {
            prompts: prompt_results,
        })
    }

    pub fn run_single_turn_chat_sampled_suite(
        &mut self,
        user_prompts: &[String],
        options: &ChatInferenceOptions,
        sampling: SamplingOptions,
    ) -> Result<ChatGenerationSuiteResult> {
        if user_prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "chat generation suite must be nonempty",
            )
            .into());
        }

        let mut prompt_results = Vec::with_capacity(user_prompts.len());
        for (round, user_prompt) in user_prompts.iter().enumerate() {
            let prompt_sampling = SamplingOptions {
                seed: sampling.seed.wrapping_add(round as u64),
                ..sampling
            };
            let result =
                self.run_single_turn_chat_sampled(user_prompt, options, prompt_sampling)?;
            prompt_results.push(ChatGenerationPromptResult {
                user_prompt: user_prompt.clone(),
                result,
            });
        }

        Ok(ChatGenerationSuiteResult {
            prompts: prompt_results,
        })
    }

    pub fn run_single_turn_chat_logits(
        &mut self,
        user_prompt: &str,
        options: &ChatLogitsOptions,
    ) -> Result<ChatLogitsResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        run_encoded_single_turn_chat_logits(
            &mut self.runtime,
            encoded.formatted_prompt,
            encoded.prompt_tokens,
            options.top_k,
        )
    }

    pub fn run_single_turn_chat_logits_suite(
        &mut self,
        user_prompts: &[String],
        options: &ChatLogitsOptions,
    ) -> Result<ChatLogitsSuiteResult> {
        if user_prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "chat logits suite must be nonempty",
            )
            .into());
        }

        let mut prompt_results = Vec::with_capacity(user_prompts.len());
        for user_prompt in user_prompts {
            let result = self.run_single_turn_chat_logits(user_prompt, options)?;
            prompt_results.push(ChatLogitsPromptResult {
                user_prompt: user_prompt.clone(),
                result,
            });
        }

        Ok(ChatLogitsSuiteResult {
            prompts: prompt_results,
        })
    }

    pub fn run_single_turn_chat_logits_trace(
        &mut self,
        user_prompt: &str,
        options: &ChatLogitsTraceOptions,
    ) -> Result<ChatBackendLogitsTraceResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        let trace = self.runtime.run_logits_trace(
            &encoded.prompt_tokens,
            &GenerationLogitsTraceOptions {
                max_new_tokens: options.max_new_tokens,
                top_k: options.top_k,
                logits_top_k: options.logits_top_k,
                stop_token_id: options.stop_token_id,
            },
        )?;
        let generated_text = self.tokenizer.decode_lossy(&trace.generated_tokens)?;
        let all_text_skip_special = self.tokenizer.decode_lossy(&trace.all_tokens)?;

        Ok(ChatBackendLogitsTraceResult {
            formatted_prompt: encoded.formatted_prompt,
            generated_text,
            all_text_skip_special,
            trace,
        })
    }

    pub fn run_single_turn_chat_logits_trace_suite(
        &mut self,
        user_prompts: &[String],
        options: &ChatLogitsTraceOptions,
    ) -> Result<ChatBackendLogitsTraceSuiteResult> {
        if user_prompts.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "chat logits trace suite must be nonempty",
            )
            .into());
        }

        let mut prompt_results = Vec::with_capacity(user_prompts.len());
        for user_prompt in user_prompts {
            let result = self.run_single_turn_chat_logits_trace(user_prompt, options)?;
            prompt_results.push(ChatBackendLogitsTracePromptResult {
                user_prompt: user_prompt.clone(),
                result,
            });
        }

        Ok(ChatBackendLogitsTraceSuiteResult {
            prompts: prompt_results,
        })
    }
}

impl MinistralChatSession {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let default_system_prompt = chat::default_ministral_system_prompt(model_dir)?;
        let runtime = MinistralTextRuntime::new(stream, module, model_dir, max_seq_len)?;

        Ok(Self {
            tokenizer,
            default_system_prompt,
            runtime,
        })
    }

    pub fn run_single_turn_chat(
        &mut self,
        user_prompt: &str,
        options: &ChatInferenceOptions,
    ) -> Result<ChatInferenceResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        run_encoded_single_turn_chat(
            &self.tokenizer,
            &mut self.runtime,
            encoded.formatted_prompt,
            encoded.prompt_tokens,
            options,
        )
    }

    pub fn next_logits_for_single_turn_chat(
        &mut self,
        user_prompt: &str,
        options: &ChatLogitsOptions,
    ) -> Result<ChatLogitsResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        run_encoded_single_turn_chat_logits(
            &mut self.runtime,
            encoded.formatted_prompt,
            encoded.prompt_tokens,
            options.top_k,
        )
    }

    pub fn logits_trace_for_single_turn_chat(
        &mut self,
        user_prompt: &str,
        options: &ChatLogitsTraceOptions,
    ) -> Result<ChatLogitsTraceResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        run_encoded_single_turn_chat_logits_trace(
            &self.tokenizer,
            &mut self.runtime,
            encoded.formatted_prompt,
            encoded.prompt_tokens,
            options,
        )
    }

    pub fn forced_logits_trace_for_single_turn_chat(
        &mut self,
        user_prompt: &str,
        forced_tokens: &[u32],
        options: &ChatForcedLogitsTraceOptions,
    ) -> Result<ChatLogitsTraceResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        run_encoded_single_turn_chat_forced_logits_trace(
            &self.tokenizer,
            &mut self.runtime,
            encoded.formatted_prompt,
            encoded.prompt_tokens,
            forced_tokens,
            options.logits_top_k,
        )
    }
}

impl MinistralAllLinearQuantizedChatSession {
    pub fn new(
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let default_system_prompt = chat::default_ministral_system_prompt(model_dir)?;
        let runtime =
            MinistralAllLinearQuantizedRuntime::new(stream, module, model_dir, max_seq_len)?;

        Ok(Self {
            tokenizer,
            default_system_prompt,
            runtime,
        })
    }

    pub fn run_single_turn_chat(
        &mut self,
        user_prompt: &str,
        options: &ChatInferenceOptions,
    ) -> Result<ChatInferenceResult> {
        let encoded = encode_single_turn_chat_prompt(
            &self.tokenizer,
            &self.default_system_prompt,
            user_prompt,
            &options.system_prompt,
        )?;
        run_encoded_single_turn_chat(
            &self.tokenizer,
            &mut self.runtime,
            encoded.formatted_prompt,
            encoded.prompt_tokens,
            options,
        )
    }
}

pub fn run_ministral_generation_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
    backend: InferenceBackend,
    stop_token_id: Option<u32>,
) -> Result<GenerationResult> {
    let max_seq_len =
        requested_seq_len_for_prompt(prompt_tokens, max_new_tokens, "backend generation request")?;

    let mut session =
        MinistralGenerationBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_generation(
        prompt_tokens,
        &GenerationOptions {
            max_new_tokens,
            top_k,
            stop_token_id,
        },
    )
}

pub fn run_ministral_generation_sampled_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    top_k: usize,
    backend: InferenceBackend,
    stop_token_id: Option<u32>,
    sampling: SamplingOptions,
) -> Result<GenerationResult> {
    let max_seq_len = requested_seq_len_for_prompt(
        prompt_tokens,
        max_new_tokens,
        "sampled backend generation request",
    )?;

    let mut session =
        MinistralGenerationBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_generation_sampled(
        prompt_tokens,
        &GenerationOptions {
            max_new_tokens,
            top_k,
            stop_token_id,
        },
        sampling,
    )
}

pub fn run_ministral_generation_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    backend: InferenceBackend,
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<GenerationSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "backend generation suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            max_new_tokens,
            "backend generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session =
        MinistralGenerationBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_generation_suite(
        prompts,
        &GenerationOptions {
            max_new_tokens,
            top_k,
            stop_token_id,
        },
    )
}

pub fn run_ministral_generation_sampled_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    backend: InferenceBackend,
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
    sampling: SamplingOptions,
) -> Result<GenerationSuiteResult> {
    validate_sampling_options(sampling)?;
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "sampled backend generation suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            max_new_tokens,
            "sampled backend generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session =
        MinistralGenerationBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_generation_sampled_suite(
        prompts,
        &GenerationOptions {
            max_new_tokens,
            top_k,
            stop_token_id,
        },
        sampling,
    )
}

pub fn run_ministral_generation_exported_all_linear_int8_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<GenerationSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported all-linear int8 generation suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            max_new_tokens,
            "exported all-linear int8 generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session = MinistralGenerationBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_generation_suite(
        prompts,
        &GenerationOptions {
            max_new_tokens,
            top_k,
            stop_token_id,
        },
    )
}

pub fn run_ministral_generation_exported_all_linear_int8_sampled_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
    sampling: SamplingOptions,
) -> Result<GenerationSuiteResult> {
    validate_sampling_options(sampling)?;
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "sampled exported all-linear int8 generation suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            max_new_tokens,
            "sampled exported all-linear int8 generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session = MinistralGenerationBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_generation_sampled_suite(
        prompts,
        &GenerationOptions {
            max_new_tokens,
            top_k,
            stop_token_id,
        },
        sampling,
    )
}

pub fn run_ministral_generation_logits_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    backend: InferenceBackend,
    top_k: usize,
) -> Result<GenerationLogitsSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "backend generation logits suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len =
            requested_seq_len_for_prompt(prompt_tokens, 1, "backend generation logits request")?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session =
        MinistralGenerationBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_next_logits_suite(prompts, top_k)
}

pub fn run_ministral_generation_exported_all_linear_int8_logits_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    top_k: usize,
) -> Result<GenerationLogitsSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported all-linear int8 generation logits suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            1,
            "exported all-linear int8 generation logits request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session = MinistralGenerationBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_next_logits_suite(prompts, top_k)
}

pub fn run_ministral_generation_logits_trace_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
) -> Result<GenerationLogitsTraceResult> {
    let max_seq_len = requested_seq_len_for_prompt(
        prompt_tokens,
        options.max_new_tokens,
        "backend generation logits trace request",
    )?;

    let mut session =
        MinistralGenerationBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    run_generation_logits_trace_with_session(&mut session, prompt_tokens, options)
}

pub fn run_ministral_generation_logits_trace_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
) -> Result<GenerationLogitsTraceSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "backend generation logits trace suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            options.max_new_tokens,
            "backend generation logits trace suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session =
        MinistralGenerationBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_logits_trace_suite(prompts, options)
}

pub fn run_ministral_generation_exported_all_linear_int8_logits_trace_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    options: &GenerationLogitsTraceOptions,
) -> Result<GenerationLogitsTraceSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported all-linear int8 generation logits trace suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            options.max_new_tokens,
            "exported all-linear int8 generation logits trace suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session = MinistralGenerationBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_logits_trace_suite(prompts, options)
}

pub fn run_ministral_generation_backend_comparison(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    candidate_backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
) -> Result<GenerationBackendComparisonResult> {
    let model_dir = model_dir.as_ref();
    let max_seq_len = requested_seq_len_for_prompt(
        prompt_tokens,
        options.max_new_tokens,
        "backend generation comparison request",
    )?;
    let mut reference = MinistralGenerationBackendSession::new(
        stream.clone(),
        module.clone(),
        model_dir,
        max_seq_len,
        InferenceBackend::Bf16,
    )?;
    let mut candidate = MinistralGenerationBackendSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    run_generation_backend_comparison_with_sessions(
        &mut reference,
        &mut candidate,
        prompt_tokens,
        options,
    )
}

pub fn run_ministral_generation_backend_comparison_suite_with_driver(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    candidate_backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
    driver: GenerationComparisonDriver,
) -> Result<GenerationBackendComparisonSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "backend generation comparison suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            options.max_new_tokens,
            "backend generation comparison suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session = MinistralGenerationBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    session.compare_generation_suite_with_driver(prompts, options, driver)
}

pub fn run_ministral_generation_backend_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    candidate_backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
) -> Result<GenerationBackendComparisonSuiteResult> {
    run_ministral_generation_backend_comparison_suite_with_driver(
        stream,
        module,
        model_dir,
        prompts,
        candidate_backend,
        options,
        GenerationComparisonDriver::Candidate,
    )
}

pub fn run_ministral_generation_exported_all_linear_int8_comparison_suite_with_driver(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    options: &GenerationLogitsTraceOptions,
    driver: GenerationComparisonDriver,
) -> Result<GenerationBackendComparisonSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported all-linear int8 comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let export_dir = export_dir.as_ref();
    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            options.max_new_tokens,
            "exported all-linear int8 comparison suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut reference = MinistralGenerationBackendSession::new(
        stream.clone(),
        module.clone(),
        model_dir,
        max_seq_len,
        InferenceBackend::Bf16,
    )?;
    let mut candidate = MinistralGenerationBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    let mut results = Vec::with_capacity(prompts.len());
    let mut summary = GenerationBackendComparisonSuiteSummary::default();
    for prompt_tokens in prompts {
        let result = run_generation_backend_comparison_with_sessions_and_driver(
            &mut reference,
            &mut candidate,
            prompt_tokens,
            options,
            driver,
        )?;
        summary.observe_comparison(&result);
        results.push(result);
    }

    Ok(GenerationBackendComparisonSuiteResult {
        prompts: results,
        summary,
        reference_memory_stats: reference.memory_stats(),
        candidate_memory_stats: candidate.memory_stats(),
    })
}

pub fn run_ministral_generation_exported_all_linear_int8_next_logits_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    top_k: usize,
) -> Result<GenerationBackendLogitsComparisonSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported all-linear int8 logits comparison suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            1,
            "exported all-linear int8 logits comparison request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session = MinistralGenerationBackendComparisonSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.compare_next_logits_suite(prompts, top_k)
}

pub fn run_ministral_generation_exported_all_linear_int8_forced_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    sequences: &[Vec<u32>],
    logits_top_k: usize,
) -> Result<GenerationBackendForcedComparisonSuiteResult> {
    if sequences.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported all-linear int8 forced comparison suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for sequence_tokens in sequences {
        validate_forced_sequence(
            sequence_tokens,
            "exported all-linear int8 forced comparison request",
        )?;
        max_seq_len = max_seq_len.max(sequence_tokens.len());
    }

    let mut session = MinistralGenerationBackendComparisonSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.compare_forced_sequence_suite(sequences, logits_top_k)
}

pub fn run_ministral_generation_exported_all_linear_int8_eval_suite_with_driver(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    options: &GenerationLogitsTraceOptions,
    driver: GenerationComparisonDriver,
) -> Result<GenerationBackendEvalSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported all-linear int8 eval suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let logits_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            1,
            "exported all-linear int8 eval logits request",
        )?;
        let generation_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            options.max_new_tokens,
            "exported all-linear int8 eval generation request",
        )?;
        max_seq_len = max_seq_len.max(logits_seq_len).max(generation_seq_len);
    }

    let mut session = MinistralGenerationBackendComparisonSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    let logits = session.compare_next_logits_suite(prompts, options.logits_top_k)?;
    let generation = session.compare_generation_suite_with_driver(prompts, options, driver)?;

    Ok(GenerationBackendEvalSuiteResult { logits, generation })
}

pub fn run_ministral_generation_backend_next_logits_comparison(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_tokens: &[u32],
    candidate_backend: InferenceBackend,
    top_k: usize,
) -> Result<GenerationBackendLogitsComparisonResult> {
    let model_dir = model_dir.as_ref();
    let max_seq_len =
        requested_seq_len_for_prompt(prompt_tokens, 1, "backend logits comparison request")?;
    let mut reference = MinistralGenerationBackendSession::new(
        stream.clone(),
        module.clone(),
        model_dir,
        max_seq_len,
        InferenceBackend::Bf16,
    )?;
    let mut candidate = MinistralGenerationBackendSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    run_generation_backend_next_logits_comparison_with_sessions(
        &mut reference,
        &mut candidate,
        prompt_tokens,
        top_k,
    )
}

pub fn run_ministral_generation_backend_next_logits_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    candidate_backend: InferenceBackend,
    top_k: usize,
) -> Result<GenerationBackendLogitsComparisonSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "backend logits comparison suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let requested_seq_len =
            requested_seq_len_for_prompt(prompt_tokens, 1, "backend logits comparison request")?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }

    let mut session = MinistralGenerationBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    session.compare_next_logits_suite(prompts, top_k)
}

pub fn run_ministral_generation_backend_forced_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    sequences: &[Vec<u32>],
    candidate_backend: InferenceBackend,
    logits_top_k: usize,
) -> Result<GenerationBackendForcedComparisonSuiteResult> {
    if sequences.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "backend forced comparison suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for sequence_tokens in sequences {
        validate_forced_sequence(sequence_tokens, "backend forced comparison suite request")?;
        max_seq_len = max_seq_len.max(sequence_tokens.len());
    }

    let mut session = MinistralGenerationBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    session.compare_forced_sequence_suite(sequences, logits_top_k)
}

pub fn run_ministral_generation_backend_eval_suite_with_driver(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[Vec<u32>],
    candidate_backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
    driver: GenerationComparisonDriver,
) -> Result<GenerationBackendEvalSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "backend eval suite must be nonempty",
        )
        .into());
    }

    let mut max_seq_len = 0;
    for prompt_tokens in prompts {
        let logits_seq_len =
            requested_seq_len_for_prompt(prompt_tokens, 1, "backend eval logits request")?;
        let generation_seq_len = requested_seq_len_for_prompt(
            prompt_tokens,
            options.max_new_tokens,
            "backend eval generation request",
        )?;
        max_seq_len = max_seq_len.max(logits_seq_len).max(generation_seq_len);
    }

    let mut session = MinistralGenerationBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    let logits = session.compare_next_logits_suite(prompts, options.logits_top_k)?;
    let generation = session.compare_generation_suite_with_driver(prompts, options, driver)?;

    Ok(GenerationBackendEvalSuiteResult { logits, generation })
}

fn run_generation_backend_next_logits_comparison_with_sessions(
    reference: &mut MinistralGenerationBackendSession,
    candidate: &mut MinistralGenerationBackendSession,
    prompt_tokens: &[u32],
    top_k: usize,
) -> Result<GenerationBackendLogitsComparisonResult> {
    let requested_seq_len =
        requested_seq_len_for_prompt(prompt_tokens, 1, "backend logits comparison request")?;
    prefill_runtime_pair_for_request(
        reference,
        candidate,
        prompt_tokens,
        requested_seq_len,
        "backend logits comparison request",
        "reference logits comparison request",
        "candidate logits comparison request",
    )?;
    let reference_backend = reference.backend();
    let candidate_backend = candidate.backend();
    let reference_memory_stats = reference.memory_stats();
    let candidate_memory_stats = candidate.memory_stats();
    let reference_snapshot = collect_prefilled_next_logits(reference, top_k)?;
    let candidate_snapshot = collect_prefilled_next_logits(candidate, top_k)?;
    let kl_divergence =
        kl_divergence_from_logits(&reference_snapshot.logits, &candidate_snapshot.logits)?;
    let max_abs_diff =
        max_abs_diff_between(&reference_snapshot.logits, &candidate_snapshot.logits)?;
    let mean_abs_diff =
        mean_abs_diff_between(&reference_snapshot.logits, &candidate_snapshot.logits)?;
    let (reference_token_id, reference_logit) =
        top_token_logit(&reference_snapshot.top_logits, "reference")?;
    let (candidate_token_id, candidate_logit) =
        top_token_logit(&candidate_snapshot.top_logits, "candidate")?;
    let reference_result = GenerationLogitsResult {
        backend: reference_backend,
        prompt_tokens: prompt_tokens.to_vec(),
        logits: reference_snapshot.logits,
        logsumexp: reference_snapshot.logsumexp,
        top_logits: reference_snapshot.top_logits,
        memory_stats: reference_memory_stats,
    };
    let candidate_result = GenerationLogitsResult {
        backend: candidate_backend,
        prompt_tokens: prompt_tokens.to_vec(),
        logits: candidate_snapshot.logits,
        logsumexp: candidate_snapshot.logsumexp,
        top_logits: candidate_snapshot.top_logits,
        memory_stats: candidate_memory_stats,
    };

    Ok(GenerationBackendLogitsComparisonResult {
        prompt_tokens: prompt_tokens.to_vec(),
        reference: reference_result,
        candidate: candidate_result,
        reference_token_id,
        reference_logit,
        candidate_token_id,
        candidate_logit,
        token_matches: reference_token_id == candidate_token_id,
        kl_divergence,
        max_abs_diff,
        mean_abs_diff,
    })
}

fn run_generation_backend_forced_comparison_with_sessions(
    reference: &mut MinistralGenerationBackendSession,
    candidate: &mut MinistralGenerationBackendSession,
    sequence_tokens: &[u32],
    logits_top_k: usize,
) -> Result<GenerationBackendForcedComparisonResult> {
    validate_forced_sequence(sequence_tokens, "backend forced comparison request")?;
    let prompt_tokens = &sequence_tokens[..1];
    let forced_tokens = &sequence_tokens[1..];
    run_generation_backend_forced_target_comparison_with_sessions(
        reference,
        candidate,
        prompt_tokens,
        forced_tokens,
        "backend forced comparison request",
        logits_top_k,
    )
}

fn run_generation_backend_forced_target_comparison_with_sessions(
    reference: &mut MinistralGenerationBackendSession,
    candidate: &mut MinistralGenerationBackendSession,
    prompt_tokens: &[u32],
    forced_tokens: &[u32],
    context: &str,
    logits_top_k: usize,
) -> Result<GenerationBackendForcedComparisonResult> {
    validate_encoded_prompt(prompt_tokens, context)?;
    validate_forced_tokens(forced_tokens, context)?;
    let requested_seq_len =
        requested_seq_len_for_prompt(prompt_tokens, forced_tokens.len(), context)?;
    prefill_runtime_pair_for_request(
        reference,
        candidate,
        prompt_tokens,
        requested_seq_len,
        context,
        "reference forced comparison request",
        "candidate forced comparison request",
    )?;

    let reference_memory_stats = reference.memory_stats();
    let candidate_memory_stats = candidate.memory_stats();
    let reference_backend = reference.backend();
    let candidate_backend = candidate.backend();
    let mut steps = Vec::with_capacity(forced_tokens.len());
    let logits_top_k = logits_top_k.max(1);

    for (step, &forced_token_id) in forced_tokens.iter().enumerate() {
        let prefix_len = reference.tokens().len();
        let reference_snapshot = collect_prefilled_next_logits(reference, logits_top_k)?;
        let candidate_snapshot = collect_prefilled_next_logits(candidate, logits_top_k)?;
        let kl_divergence =
            kl_divergence_from_logits(&reference_snapshot.logits, &candidate_snapshot.logits)?;
        let max_abs_diff =
            max_abs_diff_between(&reference_snapshot.logits, &candidate_snapshot.logits)?;
        let mean_abs_diff =
            mean_abs_diff_between(&reference_snapshot.logits, &candidate_snapshot.logits)?;
        let (reference_token_id, reference_logit) =
            top_token_logit(&reference_snapshot.top_logits, "reference")?;
        let (candidate_token_id, candidate_logit) =
            top_token_logit(&candidate_snapshot.top_logits, "candidate")?;
        let reference_forced_logit = token_logit(
            &reference_snapshot.logits,
            forced_token_id,
            "reference forced",
        )?;
        let candidate_forced_logit = token_logit(
            &candidate_snapshot.logits,
            forced_token_id,
            "candidate forced",
        )?;
        let reference_forced_logprob = reference_forced_logit - reference_snapshot.logsumexp;
        let candidate_forced_logprob = candidate_forced_logit - candidate_snapshot.logsumexp;

        steps.push(GenerationBackendForcedComparisonStep {
            step,
            prefix_len,
            forced_token_id,
            reference_token_id,
            reference_logit,
            reference_forced_logit,
            reference_forced_logprob,
            candidate_token_id,
            candidate_logit,
            candidate_forced_logit,
            candidate_forced_logprob,
            token_matches: reference_token_id == candidate_token_id,
            forced_token_matches_reference: reference_token_id == forced_token_id,
            forced_token_matches_candidate: candidate_token_id == forced_token_id,
            kl_divergence,
            max_abs_diff,
            mean_abs_diff,
            reference_top_logits: reference_snapshot.top_logits,
            candidate_top_logits: candidate_snapshot.top_logits,
        });

        reference.advance_with_token(forced_token_id)?;
        candidate.advance_with_token(forced_token_id)?;
    }

    Ok(GenerationBackendForcedComparisonResult {
        sequence_tokens: prompt_tokens
            .iter()
            .chain(forced_tokens.iter())
            .copied()
            .collect(),
        prompt_tokens: prompt_tokens.to_vec(),
        forced_tokens: forced_tokens.to_vec(),
        reference_backend,
        candidate_backend,
        reference_memory_stats,
        candidate_memory_stats,
        steps,
    })
}

fn run_generation_backend_comparison_with_sessions(
    reference: &mut MinistralGenerationBackendSession,
    candidate: &mut MinistralGenerationBackendSession,
    prompt_tokens: &[u32],
    options: &GenerationLogitsTraceOptions,
) -> Result<GenerationBackendComparisonResult> {
    run_generation_backend_comparison_with_sessions_and_driver(
        reference,
        candidate,
        prompt_tokens,
        options,
        GenerationComparisonDriver::Candidate,
    )
}

fn run_generation_backend_comparison_with_sessions_and_driver(
    reference: &mut MinistralGenerationBackendSession,
    candidate: &mut MinistralGenerationBackendSession,
    prompt_tokens: &[u32],
    options: &GenerationLogitsTraceOptions,
    driver: GenerationComparisonDriver,
) -> Result<GenerationBackendComparisonResult> {
    let requested_seq_len = requested_seq_len_for_prompt(
        prompt_tokens,
        options.max_new_tokens,
        "backend generation comparison request",
    )?;
    prefill_runtime_pair_for_request(
        reference,
        candidate,
        prompt_tokens,
        requested_seq_len,
        "backend generation comparison request",
        "reference comparison request",
        "candidate comparison request",
    )?;
    let reference_memory_stats = reference.memory_stats();
    let candidate_memory_stats = candidate.memory_stats();
    let reference_backend = reference.backend();
    let candidate_backend = candidate.backend();
    let mut generated_tokens = Vec::with_capacity(options.max_new_tokens);
    let mut steps = Vec::with_capacity(options.max_new_tokens);
    let logits_top_k = options.logits_top_k.max(1);

    for step in 0..options.max_new_tokens {
        let prefix_len = candidate.tokens().len();
        let reference_snapshot = collect_prefilled_next_logits(reference, logits_top_k)?;
        let candidate_snapshot = collect_prefilled_next_logits(candidate, logits_top_k)?;
        let kl_divergence =
            kl_divergence_from_logits(&reference_snapshot.logits, &candidate_snapshot.logits)?;
        let max_abs_diff =
            max_abs_diff_between(&reference_snapshot.logits, &candidate_snapshot.logits)?;
        let mean_abs_diff =
            mean_abs_diff_between(&reference_snapshot.logits, &candidate_snapshot.logits)?;
        let (reference_token_id, reference_logit) =
            top_token_logit(&reference_snapshot.top_logits, "reference")?;
        let (candidate_token_id, candidate_logit) =
            top_token_logit(&candidate_snapshot.top_logits, "candidate")?;
        let driver_token_id = match driver {
            GenerationComparisonDriver::Candidate => candidate_token_id,
            GenerationComparisonDriver::Reference => reference_token_id,
        };

        steps.push(GenerationBackendComparisonStep {
            step,
            prefix_len,
            reference_token_id,
            reference_logit,
            candidate_token_id,
            candidate_logit,
            token_matches: reference_token_id == candidate_token_id,
            kl_divergence,
            max_abs_diff,
            mean_abs_diff,
            reference_top_logits: reference_snapshot.top_logits,
            candidate_top_logits: candidate_snapshot.top_logits,
        });

        reference.advance_with_token(driver_token_id)?;
        candidate.advance_with_token(driver_token_id)?;
        generated_tokens.push(driver_token_id);

        if options.stop_token_id == Some(driver_token_id) {
            break;
        }
    }

    Ok(GenerationBackendComparisonResult {
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens,
        all_tokens: candidate.tokens().to_vec(),
        driver,
        reference_backend,
        candidate_backend,
        reference_memory_stats,
        candidate_memory_stats,
        steps,
    })
}

pub fn run_ministral_text_generation_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_text: &str,
    backend: InferenceBackend,
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<TextGenerationResult> {
    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let prompt_tokens = tokenizer.encode_lossy(prompt_text, true)?;
    let generation = run_ministral_generation_with_backend(
        stream,
        module,
        model_dir,
        &prompt_tokens,
        max_new_tokens,
        top_k,
        backend,
        stop_token_id,
    )?;
    let generated_text = tokenizer.decode_lossy(&generation.generated_tokens)?;
    let all_text_skip_special = tokenizer.decode_lossy(&generation.all_tokens)?;
    let finish_reason = generation.finish_reason;

    Ok(TextGenerationResult {
        backend: generation.backend,
        prompt_text: prompt_text.to_string(),
        prompt_tokens: generation.prompt_tokens,
        generated_tokens: generation.generated_tokens,
        all_tokens: generation.all_tokens,
        memory_stats: generation.memory_stats,
        timings: generation.timings,
        generated_text,
        all_text_skip_special,
        finish_reason,
        steps: generation.steps,
    })
}

pub fn run_ministral_text_generation_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[String],
    backend: InferenceBackend,
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<TextGenerationSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "text generation suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            max_new_tokens,
            "text generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session =
        MinistralTextBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_text_generation_suite(prompts, max_new_tokens, top_k, stop_token_id)
}

pub fn run_ministral_text_generation_sampled_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[String],
    backend: InferenceBackend,
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
    sampling: SamplingOptions,
) -> Result<TextGenerationSuiteResult> {
    validate_sampling_options(sampling)?;
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "sampled text generation suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            max_new_tokens,
            "sampled text generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session =
        MinistralTextBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_text_generation_sampled_suite(
        prompts,
        max_new_tokens,
        top_k,
        stop_token_id,
        sampling,
    )
}

pub fn run_ministral_text_generation_exported_all_linear_int8_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[String],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<TextGenerationSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported text generation suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            max_new_tokens,
            "exported text generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_text_generation_suite(prompts, max_new_tokens, top_k, stop_token_id)
}

pub fn run_ministral_text_generation_exported_all_linear_int8_sampled_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[String],
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
    sampling: SamplingOptions,
) -> Result<TextGenerationSuiteResult> {
    validate_sampling_options(sampling)?;
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "sampled exported text generation suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            max_new_tokens,
            "sampled exported text generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_text_generation_sampled_suite(
        prompts,
        max_new_tokens,
        top_k,
        stop_token_id,
        sampling,
    )
}

pub fn run_ministral_text_generation_logits_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[String],
    backend: InferenceBackend,
    top_k: usize,
) -> Result<TextGenerationLogitsSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "text generation logits suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            1,
            "text generation logits suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session =
        MinistralTextBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_text_next_logits_suite(prompts, top_k)
}

pub fn run_ministral_text_generation_exported_all_linear_int8_logits_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[String],
    top_k: usize,
) -> Result<TextGenerationLogitsSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported text generation logits suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            1,
            "exported text generation logits suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_text_next_logits_suite(prompts, top_k)
}

pub fn run_ministral_text_generation_backend_comparison(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_text: &str,
    candidate_backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
) -> Result<TextGenerationBackendComparisonResult> {
    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let prompt_tokens = tokenizer.encode_lossy(prompt_text, true)?;
    let comparison = run_ministral_generation_backend_comparison(
        stream,
        module,
        model_dir,
        &prompt_tokens,
        candidate_backend,
        options,
    )?;
    let generated_text = tokenizer.decode_lossy(&comparison.generated_tokens)?;
    let all_text_skip_special = tokenizer.decode_lossy(&comparison.all_tokens)?;

    Ok(TextGenerationBackendComparisonResult {
        prompt_text: prompt_text.to_string(),
        generated_text,
        all_text_skip_special,
        comparison,
    })
}

pub fn run_ministral_text_generation_backend_comparison_suite_with_driver(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[String],
    candidate_backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
    driver: GenerationComparisonDriver,
) -> Result<TextGenerationBackendComparisonSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "text generation comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            options.max_new_tokens,
            "text generation comparison suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    session.compare_text_generation_suite_with_driver(prompts, options, driver)
}

pub fn run_ministral_text_generation_backend_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[String],
    candidate_backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
) -> Result<TextGenerationBackendComparisonSuiteResult> {
    run_ministral_text_generation_backend_comparison_suite_with_driver(
        stream,
        module,
        model_dir,
        prompts,
        candidate_backend,
        options,
        GenerationComparisonDriver::Candidate,
    )
}

pub fn run_ministral_text_generation_exported_all_linear_int8_comparison_suite_with_driver(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[String],
    options: &GenerationLogitsTraceOptions,
    driver: GenerationComparisonDriver,
) -> Result<TextGenerationBackendComparisonSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported text comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut encoded_prompts = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        encoded_prompts.push(tokenizer.encode_lossy(prompt, true)?);
    }

    let raw_suite = run_ministral_generation_exported_all_linear_int8_comparison_suite_with_driver(
        stream,
        module,
        model_dir,
        export_dir,
        &encoded_prompts,
        options,
        driver,
    )?;
    let prompt_results = prompts
        .iter()
        .zip(raw_suite.prompts)
        .map(|(prompt_text, comparison)| {
            let generated_text = tokenizer.decode_lossy(&comparison.generated_tokens)?;
            let all_text_skip_special = tokenizer.decode_lossy(&comparison.all_tokens)?;
            Ok(TextGenerationBackendComparisonResult {
                prompt_text: prompt_text.clone(),
                generated_text,
                all_text_skip_special,
                comparison,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(TextGenerationBackendComparisonSuiteResult {
        prompts: prompt_results,
        summary: raw_suite.summary,
        reference_memory_stats: raw_suite.reference_memory_stats,
        candidate_memory_stats: raw_suite.candidate_memory_stats,
    })
}

pub fn run_ministral_text_generation_exported_all_linear_int8_forced_target_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    pairs: &[(String, String)],
    logits_top_k: usize,
) -> Result<TextGenerationBackendForcedTargetComparisonSuiteResult> {
    if pairs.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported text forced target comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for (prompt_text, target_text) in pairs {
        let prompt_tokens = tokenizer.encode_lossy(prompt_text, true)?;
        let forced_tokens = tokenizer.encode_lossy(target_text, false)?;
        validate_encoded_prompt(
            &prompt_tokens,
            "exported text forced target comparison suite request",
        )?;
        validate_forced_tokens(
            &forced_tokens,
            "exported text forced target comparison suite request",
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            forced_tokens.len(),
            "exported text forced target comparison suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendComparisonSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.compare_text_forced_target_suite(pairs, logits_top_k)
}

pub fn run_ministral_text_generation_exported_all_linear_int8_eval_suite_with_driver(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[String],
    options: &GenerationLogitsTraceOptions,
    driver: GenerationComparisonDriver,
) -> Result<TextGenerationBackendEvalSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported text generation eval suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let logits_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            1,
            "exported text generation eval logits request",
        )?;
        let generation_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            options.max_new_tokens,
            "exported text generation eval request",
        )?;
        max_seq_len = max_seq_len.max(logits_seq_len).max(generation_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendComparisonSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    let logits = session.compare_text_next_logits_suite(prompts, options.logits_top_k)?;
    let generation = session.compare_text_generation_suite_with_driver(prompts, options, driver)?;

    Ok(TextGenerationBackendEvalSuiteResult { logits, generation })
}

pub fn run_ministral_text_generation_backend_forced_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    sequences: &[String],
    candidate_backend: InferenceBackend,
    logits_top_k: usize,
) -> Result<TextGenerationBackendForcedComparisonSuiteResult> {
    if sequences.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "text forced comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for sequence_text in sequences {
        let sequence_tokens = tokenizer.encode_lossy(sequence_text, true)?;
        validate_forced_sequence(&sequence_tokens, "text forced comparison suite request")?;
        max_seq_len = max_seq_len.max(sequence_tokens.len());
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    session.compare_text_forced_sequence_suite(sequences, logits_top_k)
}

pub fn run_ministral_text_generation_backend_forced_target_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    pairs: &[(String, String)],
    candidate_backend: InferenceBackend,
    logits_top_k: usize,
) -> Result<TextGenerationBackendForcedTargetComparisonSuiteResult> {
    if pairs.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "text forced target comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for (prompt_text, target_text) in pairs {
        let prompt_tokens = tokenizer.encode_lossy(prompt_text, true)?;
        let forced_tokens = tokenizer.encode_lossy(target_text, false)?;
        validate_encoded_prompt(
            &prompt_tokens,
            "text forced target comparison suite request",
        )?;
        validate_forced_tokens(
            &forced_tokens,
            "text forced target comparison suite request",
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            forced_tokens.len(),
            "text forced target comparison suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    session.compare_text_forced_target_suite(pairs, logits_top_k)
}

pub fn run_ministral_text_generation_backend_next_logits_comparison(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_text: &str,
    candidate_backend: InferenceBackend,
    top_k: usize,
) -> Result<TextGenerationBackendLogitsComparisonResult> {
    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let prompt_tokens = tokenizer.encode_lossy(prompt_text, true)?;
    let comparison = run_ministral_generation_backend_next_logits_comparison(
        stream,
        module,
        model_dir,
        &prompt_tokens,
        candidate_backend,
        top_k,
    )?;

    Ok(TextGenerationBackendLogitsComparisonResult {
        prompt_text: prompt_text.to_string(),
        comparison,
    })
}

pub fn run_ministral_text_generation_backend_next_logits_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[String],
    candidate_backend: InferenceBackend,
    top_k: usize,
) -> Result<TextGenerationBackendLogitsComparisonSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "text generation logits comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            1,
            "text generation logits comparison suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    session.compare_text_next_logits_suite(prompts, top_k)
}

pub fn run_ministral_text_generation_exported_all_linear_int8_next_logits_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[String],
    top_k: usize,
) -> Result<TextGenerationBackendLogitsComparisonSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported text generation logits comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let encoded_prompts = prompts
        .iter()
        .map(|prompt| tokenizer.encode_lossy(prompt, true))
        .collect::<Result<Vec<_>>>()?;
    let raw_suite = run_ministral_generation_exported_all_linear_int8_next_logits_comparison_suite(
        stream,
        module,
        model_dir,
        export_dir,
        &encoded_prompts,
        top_k,
    )?;
    let prompt_results = prompts
        .iter()
        .zip(raw_suite.prompts)
        .map(
            |(prompt_text, comparison)| TextGenerationBackendLogitsComparisonResult {
                prompt_text: prompt_text.clone(),
                comparison,
            },
        )
        .collect();

    Ok(TextGenerationBackendLogitsComparisonSuiteResult {
        prompts: prompt_results,
        summary: raw_suite.summary,
        reference_memory_stats: raw_suite.reference_memory_stats,
        candidate_memory_stats: raw_suite.candidate_memory_stats,
    })
}

pub fn run_ministral_text_generation_backend_eval_suite_with_driver(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[String],
    candidate_backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
    driver: GenerationComparisonDriver,
) -> Result<TextGenerationBackendEvalSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "text generation eval suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let logits_seq_len =
            requested_seq_len_for_prompt(&prompt_tokens, 1, "text generation eval logits request")?;
        let generation_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            options.max_new_tokens,
            "text generation eval request",
        )?;
        max_seq_len = max_seq_len.max(logits_seq_len).max(generation_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    let logits = session.compare_text_next_logits_suite(prompts, options.logits_top_k)?;
    let generation = session.compare_text_generation_suite_with_driver(prompts, options, driver)?;

    Ok(TextGenerationBackendEvalSuiteResult { logits, generation })
}

pub fn run_ministral_text_generation_logits_trace_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompt_text: &str,
    backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
) -> Result<TextGenerationBackendLogitsTraceResult> {
    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let prompt_tokens = tokenizer.encode_lossy(prompt_text, true)?;
    let max_seq_len = requested_seq_len_for_prompt(
        &prompt_tokens,
        options.max_new_tokens,
        "text generation logits trace request",
    )?;
    drop((tokenizer, prompt_tokens));
    let mut session =
        MinistralTextBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_text_logits_trace(prompt_text, options)
}

pub fn run_ministral_text_generation_logits_trace_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[String],
    backend: InferenceBackend,
    options: &GenerationLogitsTraceOptions,
) -> Result<TextGenerationBackendLogitsTraceSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "text generation logits trace suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            options.max_new_tokens,
            "text generation logits trace suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session =
        MinistralTextBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_text_logits_trace_suite(prompts, options)
}

pub fn run_ministral_text_generation_exported_all_linear_int8_logits_trace_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    prompts: &[String],
    options: &GenerationLogitsTraceOptions,
) -> Result<TextGenerationBackendLogitsTraceSuiteResult> {
    if prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported all-linear int8 text generation logits trace suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for prompt in prompts {
        let prompt_tokens = tokenizer.encode_lossy(prompt, true)?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &prompt_tokens,
            options.max_new_tokens,
            "exported all-linear int8 text generation logits trace suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralTextBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_text_logits_trace_suite(prompts, options)
}

pub fn run_ministral_single_turn_chat(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    options: &ChatInferenceOptions,
) -> Result<ChatInferenceResult> {
    run_ministral_single_turn_chat_with_backend(
        stream,
        module,
        model_dir,
        user_prompt,
        InferenceBackend::Bf16,
        options,
    )
}

pub fn run_ministral_all_linear_quantized_single_turn_chat(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    options: &ChatInferenceOptions,
) -> Result<ChatInferenceResult> {
    run_ministral_single_turn_chat_with_backend(
        stream,
        module,
        model_dir,
        user_prompt,
        InferenceBackend::AllLinearInt8,
        options,
    )
}

pub fn run_ministral_single_turn_chat_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    backend: InferenceBackend,
    options: &ChatInferenceOptions,
) -> Result<ChatInferenceResult> {
    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let encoded = encode_single_turn_chat_prompt_from_model(
        &tokenizer,
        model_dir,
        user_prompt,
        &options.system_prompt,
    )?;
    let max_seq_len = requested_seq_len_for_prompt(
        &encoded.prompt_tokens,
        options.max_new_tokens,
        "chat request",
    )?;
    let mut runtime =
        MinistralGenerationBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    run_encoded_single_turn_chat(
        &tokenizer,
        &mut runtime,
        encoded.formatted_prompt,
        encoded.prompt_tokens,
        options,
    )
}

pub fn run_ministral_single_turn_chat_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompts: &[String],
    backend: InferenceBackend,
    options: &ChatInferenceOptions,
) -> Result<ChatGenerationSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "chat generation suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            options.max_new_tokens,
            "chat generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session =
        MinistralChatBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_single_turn_chat_suite(user_prompts, options)
}

pub fn run_ministral_single_turn_chat_sampled_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompts: &[String],
    backend: InferenceBackend,
    options: &ChatInferenceOptions,
    sampling: SamplingOptions,
) -> Result<ChatGenerationSuiteResult> {
    validate_sampling_options(sampling)?;
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "sampled chat generation suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            options.max_new_tokens,
            "sampled chat generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session =
        MinistralChatBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_single_turn_chat_sampled_suite(user_prompts, options, sampling)
}

pub fn run_ministral_single_turn_chat_exported_all_linear_int8_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    user_prompts: &[String],
    options: &ChatInferenceOptions,
) -> Result<ChatGenerationSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported chat generation suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            options.max_new_tokens,
            "exported chat generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralChatBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_single_turn_chat_suite(user_prompts, options)
}

pub fn run_ministral_single_turn_chat_exported_all_linear_int8_sampled_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    user_prompts: &[String],
    options: &ChatInferenceOptions,
    sampling: SamplingOptions,
) -> Result<ChatGenerationSuiteResult> {
    validate_sampling_options(sampling)?;
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "sampled exported chat generation suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            options.max_new_tokens,
            "sampled exported chat generation suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralChatBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_single_turn_chat_sampled_suite(user_prompts, options, sampling)
}

pub fn run_ministral_single_turn_chat_backend_comparison(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    candidate_backend: InferenceBackend,
    options: &ChatTraceComparisonSuiteOptions,
) -> Result<ChatBackendComparisonResult> {
    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let encoded = encode_single_turn_chat_prompt_from_model(
        &tokenizer,
        model_dir,
        user_prompt,
        &options.system_prompt,
    )?;
    let generation_options = GenerationLogitsTraceOptions {
        max_new_tokens: options.max_new_tokens,
        top_k: options.top_k,
        logits_top_k: options.logits_top_k,
        stop_token_id: options.stop_token_id,
    };
    let comparison = run_ministral_generation_backend_comparison(
        stream,
        module,
        model_dir,
        &encoded.prompt_tokens,
        candidate_backend,
        &generation_options,
    )?;
    let generated_text = tokenizer.decode_lossy(&comparison.generated_tokens)?;
    let all_text_skip_special = tokenizer.decode_lossy(&comparison.all_tokens)?;

    Ok(ChatBackendComparisonResult {
        formatted_prompt: encoded.formatted_prompt,
        generated_text,
        all_text_skip_special,
        comparison,
    })
}

pub fn run_ministral_single_turn_chat_backend_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompts: &[String],
    candidate_backend: InferenceBackend,
    options: &ChatTraceComparisonSuiteOptions,
    driver: GenerationComparisonDriver,
) -> Result<ChatBackendComparisonSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "chat comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            options.max_new_tokens,
            "chat comparison suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralChatBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    session.compare_single_turn_chat_suite_with_driver(user_prompts, options, driver)
}

pub fn run_ministral_single_turn_chat_exported_all_linear_int8_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    user_prompts: &[String],
    options: &ChatTraceComparisonSuiteOptions,
    driver: GenerationComparisonDriver,
) -> Result<ChatBackendComparisonSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported chat comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut encoded_prompts = Vec::with_capacity(user_prompts.len());
    for user_prompt in user_prompts {
        encoded_prompts.push(encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?);
    }
    let prompt_tokens = encoded_prompts
        .iter()
        .map(|encoded| encoded.prompt_tokens.clone())
        .collect::<Vec<_>>();
    let generation_options = GenerationLogitsTraceOptions {
        max_new_tokens: options.max_new_tokens,
        top_k: options.top_k,
        logits_top_k: options.logits_top_k,
        stop_token_id: options.stop_token_id,
    };
    let raw_suite = run_ministral_generation_exported_all_linear_int8_comparison_suite_with_driver(
        stream,
        module,
        model_dir,
        export_dir,
        &prompt_tokens,
        &generation_options,
        driver,
    )?;
    let prompt_results = user_prompts
        .iter()
        .zip(encoded_prompts.into_iter())
        .zip(raw_suite.prompts)
        .map(|((user_prompt, encoded), comparison)| {
            let generated_text = tokenizer.decode_lossy(&comparison.generated_tokens)?;
            let all_text_skip_special = tokenizer.decode_lossy(&comparison.all_tokens)?;
            Ok(ChatBackendComparisonPromptResult {
                user_prompt: user_prompt.clone(),
                result: ChatBackendComparisonResult {
                    formatted_prompt: encoded.formatted_prompt,
                    generated_text,
                    all_text_skip_special,
                    comparison,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(ChatBackendComparisonSuiteResult {
        prompts: prompt_results,
        summary: raw_suite.summary,
        reference_memory_stats: raw_suite.reference_memory_stats,
        candidate_memory_stats: raw_suite.candidate_memory_stats,
    })
}

pub fn run_ministral_single_turn_chat_exported_all_linear_int8_forced_target_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    pairs: &[(String, String)],
    options: &ChatForcedLogitsTraceOptions,
) -> Result<ChatBackendForcedTargetComparisonSuiteResult> {
    if pairs.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported chat forced target comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for (user_prompt, target_text) in pairs {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let forced_tokens = tokenizer.encode_lossy(target_text, false)?;
        validate_forced_tokens(
            &forced_tokens,
            "exported chat forced target comparison suite request",
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            forced_tokens.len(),
            "exported chat forced target comparison suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralChatBackendComparisonSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.compare_single_turn_chat_forced_target_suite(pairs, options)
}

pub fn run_ministral_single_turn_chat_exported_all_linear_int8_eval_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    user_prompts: &[String],
    options: &ChatTraceComparisonSuiteOptions,
    driver: GenerationComparisonDriver,
) -> Result<ChatBackendEvalSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported chat eval suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let logits_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            1,
            "exported chat eval logits request",
        )?;
        let generation_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            options.max_new_tokens,
            "exported chat eval generation request",
        )?;
        max_seq_len = max_seq_len.max(logits_seq_len).max(generation_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralChatBackendComparisonSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    let logits_options = ChatLogitsOptions {
        top_k: options.logits_top_k,
        system_prompt: options.system_prompt.clone(),
    };
    let logits = session.compare_single_turn_chat_logits_suite(user_prompts, &logits_options)?;
    let generation =
        session.compare_single_turn_chat_suite_with_driver(user_prompts, options, driver)?;

    Ok(ChatBackendEvalSuiteResult { logits, generation })
}

pub fn run_ministral_single_turn_chat_backend_forced_target_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    pairs: &[(String, String)],
    candidate_backend: InferenceBackend,
    options: &ChatForcedLogitsTraceOptions,
) -> Result<ChatBackendForcedTargetComparisonSuiteResult> {
    if pairs.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "chat forced target comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for (user_prompt, target_text) in pairs {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let forced_tokens = tokenizer.encode_lossy(target_text, false)?;
        validate_forced_tokens(
            &forced_tokens,
            "chat forced target comparison suite request",
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            forced_tokens.len(),
            "chat forced target comparison suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralChatBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    session.compare_single_turn_chat_forced_target_suite(pairs, options)
}

pub fn run_ministral_single_turn_chat_logits_comparison(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    candidate_backend: InferenceBackend,
    options: &ChatLogitsOptions,
) -> Result<ChatBackendLogitsComparisonResult> {
    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let encoded = encode_single_turn_chat_prompt_from_model(
        &tokenizer,
        model_dir,
        user_prompt,
        &options.system_prompt,
    )?;
    let comparison = run_ministral_generation_backend_next_logits_comparison(
        stream,
        module,
        model_dir,
        &encoded.prompt_tokens,
        candidate_backend,
        options.top_k,
    )?;

    Ok(ChatBackendLogitsComparisonResult {
        formatted_prompt: encoded.formatted_prompt,
        comparison,
    })
}

pub fn run_ministral_single_turn_chat_logits_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompts: &[String],
    candidate_backend: InferenceBackend,
    options: &ChatLogitsOptions,
) -> Result<ChatBackendLogitsComparisonSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "chat logits comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            1,
            "chat logits comparison suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralChatBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    session.compare_single_turn_chat_logits_suite(user_prompts, options)
}

pub fn run_ministral_single_turn_chat_exported_all_linear_int8_logits_comparison_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    user_prompts: &[String],
    options: &ChatLogitsOptions,
) -> Result<ChatBackendLogitsComparisonSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported all-linear int8 chat logits comparison suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let encoded_prompts = user_prompts
        .iter()
        .map(|user_prompt| {
            encode_single_turn_chat_prompt_from_model(
                &tokenizer,
                model_dir,
                user_prompt,
                &options.system_prompt,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let prompt_tokens = encoded_prompts
        .iter()
        .map(|encoded| encoded.prompt_tokens.clone())
        .collect::<Vec<_>>();
    let raw_suite = run_ministral_generation_exported_all_linear_int8_next_logits_comparison_suite(
        stream,
        module,
        model_dir,
        export_dir,
        &prompt_tokens,
        options.top_k,
    )?;
    let GenerationBackendLogitsComparisonSuiteResult {
        prompts: comparisons,
        summary,
        reference_memory_stats,
        candidate_memory_stats,
    } = raw_suite;
    let prompt_results = user_prompts
        .iter()
        .zip(encoded_prompts)
        .zip(comparisons)
        .map(
            |((user_prompt, encoded), comparison)| ChatBackendLogitsComparisonPromptResult {
                user_prompt: user_prompt.clone(),
                result: ChatBackendLogitsComparisonResult {
                    formatted_prompt: encoded.formatted_prompt,
                    comparison,
                },
            },
        )
        .collect();

    Ok(ChatBackendLogitsComparisonSuiteResult {
        prompts: prompt_results,
        summary,
        reference_memory_stats,
        candidate_memory_stats,
    })
}

pub fn run_ministral_single_turn_chat_backend_eval_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompts: &[String],
    candidate_backend: InferenceBackend,
    options: &ChatTraceComparisonSuiteOptions,
    driver: GenerationComparisonDriver,
) -> Result<ChatBackendEvalSuiteResult> {
    if user_prompts.is_empty() {
        return Err(
            io::Error::new(ErrorKind::InvalidInput, "chat eval suite must be nonempty").into(),
        );
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let logits_seq_len =
            requested_seq_len_for_prompt(&encoded.prompt_tokens, 1, "chat eval logits request")?;
        let generation_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            options.max_new_tokens,
            "chat eval generation request",
        )?;
        max_seq_len = max_seq_len.max(logits_seq_len).max(generation_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralChatBackendComparisonSession::new(
        stream,
        module,
        model_dir,
        max_seq_len,
        candidate_backend,
    )?;
    let logits_options = ChatLogitsOptions {
        top_k: options.logits_top_k,
        system_prompt: options.system_prompt.clone(),
    };
    let logits = session.compare_single_turn_chat_logits_suite(user_prompts, &logits_options)?;
    let generation =
        session.compare_single_turn_chat_suite_with_driver(user_prompts, options, driver)?;

    Ok(ChatBackendEvalSuiteResult { logits, generation })
}

pub fn run_ministral_all_linear_quantized_chat_comparison(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    options: &ChatInferenceOptions,
) -> Result<ChatAllLinearQuantizedComparisonResult> {
    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let encoded = encode_single_turn_chat_prompt_from_model(
        &tokenizer,
        model_dir,
        user_prompt,
        &options.system_prompt,
    )?;
    let comparison = model::run_ministral_all_linear_quantized_runtime_comparison_probe(
        &stream,
        &module,
        model_dir,
        &encoded.prompt_tokens,
        options.max_new_tokens,
        options.top_k,
        options.stop_token_id,
    )?;
    let generated_text = tokenizer.decode_lossy(&comparison.generated_tokens)?;
    let all_text_skip_special = tokenizer.decode_lossy(&comparison.all_tokens)?;

    Ok(ChatAllLinearQuantizedComparisonResult {
        formatted_prompt: encoded.formatted_prompt,
        prompt_tokens: comparison.prompt_tokens,
        generated_tokens: comparison.generated_tokens,
        all_tokens: comparison.all_tokens,
        generated_text,
        all_text_skip_special,
        reference_memory_stats: comparison.reference_memory_stats,
        quantized_memory_stats: comparison.quantized_memory_stats,
        steps: comparison.steps,
    })
}

pub fn run_ministral_single_turn_chat_logits(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    options: &ChatLogitsOptions,
) -> Result<ChatLogitsResult> {
    run_ministral_single_turn_chat_logits_with_backend(
        stream,
        module,
        model_dir,
        user_prompt,
        InferenceBackend::Bf16,
        options,
    )
}

pub fn run_ministral_single_turn_chat_logits_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    backend: InferenceBackend,
    options: &ChatLogitsOptions,
) -> Result<ChatLogitsResult> {
    let model_dir = model_dir.as_ref();
    let max_seq_len = {
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        requested_seq_len_for_prompt(&encoded.prompt_tokens, 1, "chat logits request")?
    };
    let mut session =
        MinistralChatBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_single_turn_chat_logits(user_prompt, options)
}

pub fn run_ministral_single_turn_chat_logits_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompts: &[String],
    backend: InferenceBackend,
    options: &ChatLogitsOptions,
) -> Result<ChatLogitsSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "chat logits suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let requested_seq_len =
            requested_seq_len_for_prompt(&encoded.prompt_tokens, 1, "chat logits suite request")?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session =
        MinistralChatBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_single_turn_chat_logits_suite(user_prompts, options)
}

pub fn run_ministral_single_turn_chat_exported_all_linear_int8_logits_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    user_prompts: &[String],
    options: &ChatLogitsOptions,
) -> Result<ChatLogitsSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported chat logits suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            1,
            "exported chat logits suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralChatBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_single_turn_chat_logits_suite(user_prompts, options)
}

pub fn run_ministral_single_turn_chat_logits_trace(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    options: &ChatLogitsTraceOptions,
) -> Result<ChatLogitsTraceResult> {
    let result = run_ministral_single_turn_chat_logits_trace_with_backend(
        stream,
        module,
        model_dir,
        user_prompt,
        InferenceBackend::Bf16,
        options,
    )?;
    let trace = result.trace;

    Ok(ChatLogitsTraceResult {
        formatted_prompt: result.formatted_prompt,
        prompt_tokens: trace.prompt_tokens,
        generated_tokens: trace.generated_tokens,
        all_tokens: trace.all_tokens,
        generated_text: result.generated_text,
        all_text_skip_special: result.all_text_skip_special,
        finish_reason: trace.finish_reason,
        steps: trace.steps,
    })
}

pub fn run_ministral_single_turn_chat_logits_trace_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    backend: InferenceBackend,
    options: &ChatLogitsTraceOptions,
) -> Result<ChatBackendLogitsTraceResult> {
    let model_dir = model_dir.as_ref();
    let max_seq_len = {
        let tokenizer = TekkenTokenizer::open(model_dir)?;
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            options.max_new_tokens,
            "chat logits trace request",
        )?
    };
    let mut session =
        MinistralChatBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_single_turn_chat_logits_trace(user_prompt, options)
}

pub fn run_ministral_single_turn_chat_logits_trace_suite_with_backend(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompts: &[String],
    backend: InferenceBackend,
    options: &ChatLogitsTraceOptions,
) -> Result<ChatBackendLogitsTraceSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "chat logits trace suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            options.max_new_tokens,
            "chat logits trace suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session =
        MinistralChatBackendSession::new(stream, module, model_dir, max_seq_len, backend)?;
    session.run_single_turn_chat_logits_trace_suite(user_prompts, options)
}

pub fn run_ministral_single_turn_chat_exported_all_linear_int8_logits_trace_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    export_dir: impl AsRef<Path>,
    user_prompts: &[String],
    options: &ChatLogitsTraceOptions,
) -> Result<ChatBackendLogitsTraceSuiteResult> {
    if user_prompts.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "exported all-linear int8 chat logits trace suite must be nonempty",
        )
        .into());
    }

    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let mut max_seq_len = 0;
    for user_prompt in user_prompts {
        let encoded = encode_single_turn_chat_prompt_from_model(
            &tokenizer,
            model_dir,
            user_prompt,
            &options.system_prompt,
        )?;
        let requested_seq_len = requested_seq_len_for_prompt(
            &encoded.prompt_tokens,
            options.max_new_tokens,
            "exported all-linear int8 chat logits trace suite request",
        )?;
        max_seq_len = max_seq_len.max(requested_seq_len);
    }
    drop(tokenizer);

    let mut session = MinistralChatBackendSession::new_all_linear_int8_export(
        stream,
        module,
        model_dir,
        export_dir,
        max_seq_len,
    )?;
    session.run_single_turn_chat_logits_trace_suite(user_prompts, options)
}

pub fn run_ministral_single_turn_chat_forced_logits_trace(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    user_prompt: &str,
    forced_tokens: &[u32],
    options: &ChatForcedLogitsTraceOptions,
) -> Result<ChatLogitsTraceResult> {
    let model_dir = model_dir.as_ref();
    let tokenizer = TekkenTokenizer::open(model_dir)?;
    let encoded = encode_single_turn_chat_prompt_from_model(
        &tokenizer,
        model_dir,
        user_prompt,
        &options.system_prompt,
    )?;
    let max_seq_len = requested_seq_len_for_prompt(
        &encoded.prompt_tokens,
        forced_tokens.len(),
        "forced chat logits trace request",
    )?;
    let mut runtime = MinistralTextRuntime::new(stream, module, model_dir, max_seq_len)?;
    run_encoded_single_turn_chat_forced_logits_trace(
        &tokenizer,
        &mut runtime,
        encoded.formatted_prompt,
        encoded.prompt_tokens,
        forced_tokens,
        options.logits_top_k,
    )
}

pub fn run_ministral_chat_forced_trace_suite(
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    model_dir: impl AsRef<Path>,
    prompts: &[String],
    options: &ChatTraceComparisonSuiteOptions,
) -> Result<ChatTraceComparisonSuiteResult> {
    if prompts.is_empty() {
        return Err(
            io::Error::new(ErrorKind::InvalidInput, "prompt suite must be nonempty").into(),
        );
    }

    let model_dir = model_dir.as_ref();
    let trace_options = ChatLogitsTraceOptions {
        max_new_tokens: options.max_new_tokens,
        top_k: options.top_k,
        logits_top_k: options.logits_top_k,
        system_prompt: options.system_prompt.clone(),
        stop_token_id: options.stop_token_id,
    };
    let forced_options = ChatForcedLogitsTraceOptions {
        logits_top_k: options.logits_top_k,
        system_prompt: options.system_prompt.clone(),
    };

    let mut prompt_results = Vec::with_capacity(prompts.len());
    let mut kl_sum = 0.0_f64;
    let mut total_steps = 0;
    let mut max_kl = 0.0_f64;
    let mut max_abs_diff = 0.0_f32;
    let mut all_token_ids_match = true;

    for prompt in prompts {
        let generated_trace = run_ministral_single_turn_chat_logits_trace(
            stream.clone(),
            module.clone(),
            model_dir,
            prompt,
            &trace_options,
        )?;
        let forced_trace = run_ministral_single_turn_chat_forced_logits_trace(
            stream.clone(),
            module.clone(),
            model_dir,
            prompt,
            &generated_trace.generated_tokens,
            &forced_options,
        )?;
        let comparison = compare_logits_traces(&generated_trace.steps, &forced_trace.steps)?;

        for step in &comparison.steps {
            kl_sum += step.kl_divergence;
            total_steps += 1;
        }
        max_kl = max_kl.max(comparison.max_kl_divergence);
        max_abs_diff = max_abs_diff.max(comparison.max_abs_diff);
        all_token_ids_match &= comparison.token_ids_match;

        prompt_results.push(ChatTraceComparisonPromptResult {
            prompt: prompt.clone(),
            prompt_tokens: generated_trace.prompt_tokens,
            generated_tokens: generated_trace.generated_tokens,
            generated_text: generated_trace.generated_text,
            comparison,
        });
    }

    let mean_kl = if total_steps == 0 {
        0.0
    } else {
        kl_sum / total_steps as f64
    };

    Ok(ChatTraceComparisonSuiteResult {
        prompts: prompt_results,
        total_steps,
        mean_kl_divergence: mean_kl,
        max_kl_divergence: max_kl,
        max_abs_diff,
        all_token_ids_match,
    })
}

fn run_generation_logits_trace_with_session(
    session: &mut MinistralGenerationBackendSession,
    prompt_tokens: &[u32],
    options: &GenerationLogitsTraceOptions,
) -> Result<GenerationLogitsTraceResult> {
    let requested_seq_len = requested_seq_len_for_prompt(
        prompt_tokens,
        options.max_new_tokens,
        "backend generation logits trace request",
    )?;
    prefill_runtime_for_request(
        session,
        prompt_tokens,
        requested_seq_len,
        "backend generation logits trace request",
    )?;
    let memory_stats = session.memory_stats();
    let backend = session.backend();
    let steps = collect_greedy_logits_trace_steps(
        session,
        options.max_new_tokens,
        options.top_k,
        options.logits_top_k,
        options.stop_token_id,
    )?;

    let all_tokens = session.tokens().to_vec();
    Ok(GenerationLogitsTraceResult {
        backend,
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens: all_tokens[prompt_tokens.len()..].to_vec(),
        all_tokens,
        memory_stats,
        finish_reason: generation_finish_reason_from_last_token(
            steps.last().map(|step| step.token_id),
            options.stop_token_id,
        ),
        steps,
    })
}

fn validate_encoded_prompt(prompt_tokens: &[u32], context: &str) -> Result<()> {
    if prompt_tokens.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!("{context} requires at least one prompt token"),
        )
        .into());
    }

    Ok(())
}

fn validate_sequence_capacity(
    requested_seq_len: usize,
    runtime_capacity: usize,
    context: &str,
) -> Result<()> {
    if requested_seq_len > runtime_capacity {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!(
                "{context} needs sequence length {requested_seq_len}, runtime capacity is {runtime_capacity}"
            ),
        )
        .into());
    }

    Ok(())
}

fn requested_seq_len_for_prompt(
    prompt_tokens: &[u32],
    extra_tokens: usize,
    context: &str,
) -> Result<usize> {
    validate_encoded_prompt(prompt_tokens, context)?;
    prompt_tokens
        .len()
        .checked_add(extra_tokens)
        .ok_or_else(|| {
            io::Error::new(
            ErrorKind::InvalidInput,
            format!(
                "{context} sequence length overflow: prompt tokens={}, extra tokens={extra_tokens}",
                prompt_tokens.len()
            ),
        )
        .into()
        })
}

fn validate_forced_sequence(sequence_tokens: &[u32], context: &str) -> Result<()> {
    if sequence_tokens.len() < 2 {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!(
                "{context} requires at least two tokens: one initial prompt token and one forced token"
            ),
        )
        .into());
    }

    Ok(())
}

fn validate_forced_tokens(forced_tokens: &[u32], context: &str) -> Result<()> {
    if forced_tokens.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!("{context} requires at least one forced target token"),
        )
        .into());
    }

    Ok(())
}

fn prefill_runtime_for_request<R>(
    runtime: &mut R,
    prompt_tokens: &[u32],
    requested_seq_len: usize,
    context: &str,
) -> Result<()>
where
    R: TokenGenerationRuntime,
{
    validate_encoded_prompt(prompt_tokens, context)?;
    validate_sequence_capacity(requested_seq_len, runtime.max_seq_len(), context)?;
    runtime.reset_sequence();
    runtime.prefill(prompt_tokens)
}

fn run_generation_with_runtime<R>(
    runtime: &mut R,
    prompt_tokens: &[u32],
    options: &GenerationOptions,
    context: &str,
) -> Result<GenerationResult>
where
    R: TokenGenerationRuntime,
{
    let requested_seq_len =
        requested_seq_len_for_prompt(prompt_tokens, options.max_new_tokens, context)?;
    let prefill_start = Instant::now();
    prefill_runtime_for_request(runtime, prompt_tokens, requested_seq_len, context)?;
    runtime.synchronize()?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    let memory_stats = runtime.memory_stats();
    let backend = runtime.backend();

    let decode_start = Instant::now();
    let steps = runtime.generate_greedy_until(
        options.max_new_tokens,
        options.top_k,
        options.stop_token_id,
    )?;
    runtime.synchronize()?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let all_tokens = runtime.tokens().to_vec();

    Ok(GenerationResult {
        backend,
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens: all_tokens[prompt_tokens.len()..].to_vec(),
        all_tokens,
        memory_stats,
        timings: GenerationTimings::new(prefill_seconds, decode_seconds),
        finish_reason: generation_finish_reason(&steps, options.stop_token_id),
        steps,
    })
}

fn run_bf16_top1_generation_for_request(
    runtime: &mut MinistralTextRuntime,
    prompt_tokens: &[u32],
    options: &GenerationOptions,
    context: &str,
) -> Result<Option<GenerationResult>> {
    if options.top_k.max(1) != 1 || options.max_new_tokens == 0 {
        return Ok(None);
    }

    let requested_seq_len =
        requested_seq_len_for_prompt(prompt_tokens, options.max_new_tokens, context)?;
    validate_encoded_prompt(prompt_tokens, context)?;
    validate_sequence_capacity(requested_seq_len, runtime.max_seq_len(), context)?;
    runtime.reset_sequence();

    let prefill_start = Instant::now();
    let initial_top1 = runtime.prefill_top1(prompt_tokens)?;
    runtime.synchronize()?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    let memory_stats = runtime.memory_stats();

    let decode_start = Instant::now();
    let steps = runtime.generate_greedy_until_from_top1(
        options.max_new_tokens,
        options.stop_token_id,
        initial_top1,
    )?;
    runtime.synchronize()?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let all_tokens = runtime.tokens().to_vec();

    Ok(Some(GenerationResult {
        backend: InferenceBackend::Bf16,
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens: all_tokens[prompt_tokens.len()..].to_vec(),
        all_tokens,
        memory_stats,
        timings: GenerationTimings::new(prefill_seconds, decode_seconds),
        finish_reason: generation_finish_reason(&steps, options.stop_token_id),
        steps,
    }))
}

fn prefill_runtime_pair_for_request<L, R>(
    lhs: &mut L,
    rhs: &mut R,
    prompt_tokens: &[u32],
    requested_seq_len: usize,
    empty_context: &str,
    lhs_context: &str,
    rhs_context: &str,
) -> Result<()>
where
    L: TokenGenerationRuntime,
    R: TokenGenerationRuntime,
{
    validate_encoded_prompt(prompt_tokens, empty_context)?;
    validate_sequence_capacity(requested_seq_len, lhs.max_seq_len(), lhs_context)?;
    validate_sequence_capacity(requested_seq_len, rhs.max_seq_len(), rhs_context)?;

    lhs.reset_sequence();
    rhs.reset_sequence();
    lhs.prefill(prompt_tokens)?;
    rhs.prefill(prompt_tokens)
}

struct NextLogitsSnapshot {
    logits: Vec<f32>,
    logsumexp: f32,
    top_logits: Vec<TokenLogit>,
}

fn collect_prefilled_next_logits<R>(runtime: &R, top_k: usize) -> Result<NextLogitsSnapshot>
where
    R: TokenGenerationRuntime,
{
    let logits = runtime.next_logits_to_host()?;
    let logsumexp = logsum_exp(&logits);
    let top_logits = top_k_logprobs_with_logsumexp(&logits, top_k.max(1), logsumexp);

    Ok(NextLogitsSnapshot {
        logits,
        logsumexp,
        top_logits,
    })
}

fn top_token_logit(top_logits: &[TokenLogit], context: &str) -> Result<(u32, f32)> {
    let choice = top_logits.first().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!("{context} produced no logits"),
        )
    })?;

    Ok((choice.token_id, choice.logit))
}

fn token_logit(logits: &[f32], token_id: u32, context: &str) -> Result<f32> {
    let token_index = token_id as usize;
    logits.get(token_index).copied().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "{context} token {token_id} is out of range for logits length {}",
                logits.len()
            ),
        )
        .into()
    })
}

fn generate_sampled_top_k_until<R>(
    runtime: &mut R,
    max_new_tokens: usize,
    top_k: usize,
    stop_token_id: Option<u32>,
    sampling: SamplingOptions,
) -> Result<Vec<GreedyGenerationStep>>
where
    R: TokenGenerationRuntime,
{
    let mut rng = XorShift64::new(sampling.seed);
    let mut steps = Vec::with_capacity(max_new_tokens);
    let top_k = top_k.max(1);

    for step in 0..max_new_tokens {
        let top_logits = runtime.next_top_logits(top_k)?;
        let (token_id, logit, top_logits) = sample_from_top_logits(top_logits, sampling, &mut rng)?;
        runtime.advance_with_token(token_id)?;
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

fn collect_greedy_logits_trace_steps<R>(
    runtime: &mut R,
    max_new_tokens: usize,
    top_k: usize,
    logits_top_k: usize,
    stop_token_id: Option<u32>,
) -> Result<Vec<ChatLogitsTraceStep>>
where
    R: TokenGenerationRuntime,
{
    let mut steps = Vec::with_capacity(max_new_tokens);
    let logits_top_k = logits_top_k.max(1);

    for step in 0..max_new_tokens {
        let snapshot = collect_prefilled_next_logits(runtime, logits_top_k)?;
        let generated_step = runtime.generate_greedy_until(1, top_k, stop_token_id)?;
        let generated_step = generated_step
            .into_iter()
            .next()
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "runtime generated no token"))?;
        let logit = token_logit(&snapshot.logits, generated_step.token_id, "generated")?;

        steps.push(ChatLogitsTraceStep {
            step,
            token_id: generated_step.token_id,
            logit,
            logits: snapshot.logits,
            logsumexp: snapshot.logsumexp,
            top_logits: snapshot.top_logits,
        });

        if stop_token_id == Some(generated_step.token_id) {
            break;
        }
    }

    Ok(steps)
}

fn collect_forced_logits_trace_steps<R>(
    runtime: &mut R,
    forced_tokens: &[u32],
    logits_top_k: usize,
) -> Result<Vec<ChatLogitsTraceStep>>
where
    R: TokenGenerationRuntime,
{
    let mut steps = Vec::with_capacity(forced_tokens.len());

    for (step, &token_id) in forced_tokens.iter().enumerate() {
        let snapshot = collect_prefilled_next_logits(runtime, logits_top_k)?;
        let logit = token_logit(&snapshot.logits, token_id, "forced")?;

        runtime.advance_with_token(token_id)?;
        steps.push(ChatLogitsTraceStep {
            step,
            token_id,
            logit,
            logits: snapshot.logits,
            logsumexp: snapshot.logsumexp,
            top_logits: snapshot.top_logits,
        });
    }

    Ok(steps)
}

fn run_encoded_single_turn_chat_sampled<R>(
    tokenizer: &TekkenTokenizer,
    runtime: &mut R,
    formatted_prompt: String,
    prompt_tokens: Vec<u32>,
    options: &ChatInferenceOptions,
    sampling: SamplingOptions,
) -> Result<ChatInferenceResult>
where
    R: TokenGenerationRuntime,
{
    validate_sampling_options(sampling)?;
    if options.top_k.max(1) == 1 {
        let generation = runtime.run_generation_request(
            &prompt_tokens,
            &GenerationOptions {
                max_new_tokens: options.max_new_tokens,
                top_k: options.top_k,
                stop_token_id: options.stop_token_id,
            },
            "sampled chat request",
        )?;
        let generated_tokens = generation.generated_tokens;
        let generated_text = tokenizer.decode_lossy(&generated_tokens)?;
        let all_text_skip_special = tokenizer.decode_lossy(&generation.all_tokens)?;

        return Ok(ChatInferenceResult {
            backend: generation.backend,
            formatted_prompt,
            prompt_tokens: generation.prompt_tokens,
            generated_tokens,
            all_tokens: generation.all_tokens,
            memory_stats: generation.memory_stats,
            timings: generation.timings,
            generated_text,
            all_text_skip_special,
            finish_reason: generation.finish_reason,
            steps: generation.steps,
        });
    }

    let requested_seq_len = requested_seq_len_for_prompt(
        &prompt_tokens,
        options.max_new_tokens,
        "sampled chat request",
    )?;
    let prefill_start = Instant::now();
    prefill_runtime_for_request(
        runtime,
        &prompt_tokens,
        requested_seq_len,
        "sampled chat request",
    )?;
    runtime.synchronize()?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    let backend = runtime.backend();
    let memory_stats = runtime.memory_stats();
    let decode_start = Instant::now();
    let steps = generate_sampled_top_k_until(
        runtime,
        options.max_new_tokens,
        options.top_k,
        options.stop_token_id,
        sampling,
    )?;
    runtime.synchronize()?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let all_tokens = runtime.tokens().to_vec();
    let generated_tokens = all_tokens[prompt_tokens.len()..].to_vec();
    let generated_text = tokenizer.decode_lossy(&generated_tokens)?;
    let all_text_skip_special = tokenizer.decode_lossy(&all_tokens)?;
    let finish_reason = generation_finish_reason(&steps, options.stop_token_id);

    Ok(ChatInferenceResult {
        backend,
        formatted_prompt,
        prompt_tokens,
        generated_tokens,
        all_tokens,
        memory_stats,
        timings: GenerationTimings::new(prefill_seconds, decode_seconds),
        generated_text,
        all_text_skip_special,
        finish_reason,
        steps,
    })
}

fn run_encoded_single_turn_chat<R>(
    tokenizer: &TekkenTokenizer,
    runtime: &mut R,
    formatted_prompt: String,
    prompt_tokens: Vec<u32>,
    options: &ChatInferenceOptions,
) -> Result<ChatInferenceResult>
where
    R: TokenGenerationRuntime,
{
    let generation = runtime.run_generation_request(
        &prompt_tokens,
        &GenerationOptions {
            max_new_tokens: options.max_new_tokens,
            top_k: options.top_k,
            stop_token_id: options.stop_token_id,
        },
        "chat request",
    )?;
    let generated_tokens = generation.generated_tokens;
    let generated_text = tokenizer.decode_lossy(&generated_tokens)?;
    let all_text_skip_special = tokenizer.decode_lossy(&generation.all_tokens)?;

    Ok(ChatInferenceResult {
        backend: generation.backend,
        formatted_prompt,
        prompt_tokens: generation.prompt_tokens,
        generated_tokens,
        all_tokens: generation.all_tokens,
        memory_stats: generation.memory_stats,
        timings: generation.timings,
        generated_text,
        all_text_skip_special,
        finish_reason: generation.finish_reason,
        steps: generation.steps,
    })
}

fn run_encoded_single_turn_chat_forced_logits_trace<R>(
    tokenizer: &TekkenTokenizer,
    runtime: &mut R,
    formatted_prompt: String,
    prompt_tokens: Vec<u32>,
    forced_tokens: &[u32],
    logits_top_k: usize,
) -> Result<ChatLogitsTraceResult>
where
    R: TokenGenerationRuntime,
{
    let requested_seq_len = requested_seq_len_for_prompt(
        &prompt_tokens,
        forced_tokens.len(),
        "forced chat logits trace request",
    )?;
    prefill_runtime_for_request(
        runtime,
        &prompt_tokens,
        requested_seq_len,
        "forced chat logits trace request",
    )?;

    let steps = collect_forced_logits_trace_steps(runtime, forced_tokens, logits_top_k)?;

    let all_tokens = runtime.tokens().to_vec();
    let generated_tokens = all_tokens[prompt_tokens.len()..].to_vec();
    let generated_text = tokenizer.decode_lossy(&generated_tokens)?;
    let all_text_skip_special = tokenizer.decode_lossy(&all_tokens)?;

    Ok(ChatLogitsTraceResult {
        formatted_prompt,
        prompt_tokens,
        generated_tokens,
        all_tokens,
        generated_text,
        all_text_skip_special,
        finish_reason: GenerationFinishReason::MaxNewTokens,
        steps,
    })
}

fn run_encoded_single_turn_chat_logits_trace<R>(
    tokenizer: &TekkenTokenizer,
    runtime: &mut R,
    formatted_prompt: String,
    prompt_tokens: Vec<u32>,
    options: &ChatLogitsTraceOptions,
) -> Result<ChatLogitsTraceResult>
where
    R: TokenGenerationRuntime,
{
    let requested_seq_len = requested_seq_len_for_prompt(
        &prompt_tokens,
        options.max_new_tokens,
        "chat logits trace request",
    )?;
    prefill_runtime_for_request(
        runtime,
        &prompt_tokens,
        requested_seq_len,
        "chat logits trace request",
    )?;

    let steps = collect_greedy_logits_trace_steps(
        runtime,
        options.max_new_tokens,
        options.top_k,
        options.logits_top_k,
        options.stop_token_id,
    )?;

    let all_tokens = runtime.tokens().to_vec();
    let generated_tokens = all_tokens[prompt_tokens.len()..].to_vec();
    let generated_text = tokenizer.decode_lossy(&generated_tokens)?;
    let all_text_skip_special = tokenizer.decode_lossy(&all_tokens)?;

    Ok(ChatLogitsTraceResult {
        formatted_prompt,
        prompt_tokens,
        generated_tokens,
        all_tokens,
        generated_text,
        all_text_skip_special,
        finish_reason: generation_finish_reason_from_last_token(
            steps.last().map(|step| step.token_id),
            options.stop_token_id,
        ),
        steps,
    })
}

fn run_encoded_single_turn_chat_logits<R>(
    runtime: &mut R,
    formatted_prompt: String,
    prompt_tokens: Vec<u32>,
    top_k: usize,
) -> Result<ChatLogitsResult>
where
    R: TokenGenerationRuntime,
{
    let requested_seq_len = requested_seq_len_for_prompt(&prompt_tokens, 1, "chat logits request")?;
    prefill_runtime_for_request(
        runtime,
        &prompt_tokens,
        requested_seq_len,
        "chat logits request",
    )?;
    let snapshot = collect_prefilled_next_logits(runtime, top_k)?;

    Ok(ChatLogitsResult {
        formatted_prompt,
        prompt_tokens,
        logits: snapshot.logits,
        logsumexp: snapshot.logsumexp,
        top_logits: snapshot.top_logits,
    })
}

pub fn compare_logits_traces(
    reference: &[ChatLogitsTraceStep],
    candidate: &[ChatLogitsTraceStep],
) -> Result<LogitsTraceComparison> {
    if reference.len() != candidate.len() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!(
                "logits trace length mismatch: reference={}, candidate={}",
                reference.len(),
                candidate.len()
            ),
        )
        .into());
    }

    let mut steps = Vec::with_capacity(reference.len());
    let mut kl_sum = 0.0_f64;
    let mut max_kl = 0.0_f64;
    let mut max_abs_diff = 0.0_f32;
    let mut token_ids_match = true;

    for (reference_step, candidate_step) in reference.iter().zip(candidate.iter()) {
        let kl = kl_divergence_from_logits(&reference_step.logits, &candidate_step.logits)?;
        let step_max_abs_diff =
            max_abs_diff_between(&reference_step.logits, &candidate_step.logits)?;
        kl_sum += kl;
        max_kl = max_kl.max(kl);
        max_abs_diff = max_abs_diff.max(step_max_abs_diff);
        token_ids_match &= reference_step.token_id == candidate_step.token_id;
        steps.push(LogitsTraceComparisonStep {
            step: reference_step.step,
            reference_token_id: reference_step.token_id,
            candidate_token_id: candidate_step.token_id,
            kl_divergence: kl,
            max_abs_diff: step_max_abs_diff,
        });
    }

    let mean_kl = if steps.is_empty() {
        0.0
    } else {
        kl_sum / steps.len() as f64
    };

    Ok(LogitsTraceComparison {
        steps,
        mean_kl_divergence: mean_kl,
        max_kl_divergence: max_kl,
        max_abs_diff,
        token_ids_match,
    })
}

pub fn logsum_exp(logits: &[f32]) -> f32 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|logit| (*logit - max_logit).exp()).sum();
    max_logit + sum_exp.ln()
}

pub fn top_k_logprobs(logits: &[f32], top_k: usize) -> Vec<TokenLogit> {
    let logsumexp = logsum_exp(logits);
    top_k_logprobs_with_logsumexp(logits, top_k, logsumexp)
}

fn generation_finish_reason(
    steps: &[GreedyGenerationStep],
    stop_token_id: Option<u32>,
) -> GenerationFinishReason {
    generation_finish_reason_from_last_token(steps.last().map(|step| step.token_id), stop_token_id)
}

fn tokens_per_second(token_count: usize, seconds: f64) -> Option<f64> {
    (token_count > 0 && seconds > 0.0 && seconds.is_finite())
        .then_some(token_count as f64 / seconds)
}

fn generation_finish_reason_from_last_token(
    last_token_id: Option<u32>,
    stop_token_id: Option<u32>,
) -> GenerationFinishReason {
    if let (Some(stop_token_id), Some(last_token_id)) = (stop_token_id, last_token_id) {
        if last_token_id == stop_token_id {
            return GenerationFinishReason::StopToken(stop_token_id);
        }
    }

    GenerationFinishReason::MaxNewTokens
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                SamplingOptions::default().seed
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut state = self.state;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.state = state;
        state
    }

    fn next_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
}

fn validate_sampling_options(sampling: SamplingOptions) -> Result<()> {
    if !sampling.temperature.is_finite() || sampling.temperature < 0.0 {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!(
                "sampling temperature must be finite and nonnegative, got {}",
                sampling.temperature
            ),
        )
        .into());
    }

    Ok(())
}

fn sample_from_top_logits(
    top_logits: Vec<(u32, f32)>,
    sampling: SamplingOptions,
    rng: &mut XorShift64,
) -> Result<(u32, f32, Vec<(u32, f32)>)> {
    if top_logits.is_empty() {
        return Err(
            io::Error::new(ErrorKind::InvalidData, "runtime produced no top logits").into(),
        );
    }
    if top_logits.iter().any(|(_, logit)| !logit.is_finite()) {
        return Err(io::Error::new(ErrorKind::InvalidInput, "top logits must be finite").into());
    }
    validate_sampling_options(sampling)?;

    let first = top_logits
        .first()
        .copied()
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "runtime produced no top logits"))?;
    if top_logits.len() == 1 || sampling.temperature <= f32::EPSILON {
        return Ok((first.0, first.1, top_logits));
    }

    let max_scaled = top_logits
        .iter()
        .map(|(_, logit)| *logit / sampling.temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights = Vec::with_capacity(top_logits.len());
    let mut total = 0.0_f64;
    for (_, logit) in &top_logits {
        let weight = ((*logit / sampling.temperature) - max_scaled).exp() as f64;
        total += weight;
        weights.push(weight);
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("invalid sampling weight total {total}"),
        )
        .into());
    }

    let mut threshold = rng.next_unit_f64() * total;
    for ((token_id, logit), weight) in top_logits.iter().copied().zip(weights.iter()) {
        threshold -= weight;
        if threshold <= 0.0 {
            return Ok((token_id, logit, top_logits));
        }
    }

    let last = top_logits
        .last()
        .copied()
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "runtime produced no top logits"))?;
    Ok((last.0, last.1, top_logits))
}

fn token_logit_ids_match(lhs: &[TokenLogit], rhs: &[TokenLogit]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(lhs, rhs)| lhs.token_id == rhs.token_id)
}

pub fn kl_divergence_from_logits(
    reference_logits: &[f32],
    candidate_logits: &[f32],
) -> Result<f64> {
    if reference_logits.len() != candidate_logits.len() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!(
                "KL logits length mismatch: reference={}, candidate={}",
                reference_logits.len(),
                candidate_logits.len()
            ),
        )
        .into());
    }
    if reference_logits.is_empty() {
        return Err(io::Error::new(ErrorKind::InvalidInput, "KL logits must be nonempty").into());
    }
    if reference_logits
        .iter()
        .chain(candidate_logits.iter())
        .any(|logit| !logit.is_finite())
    {
        return Err(io::Error::new(ErrorKind::InvalidInput, "KL logits must be finite").into());
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

fn max_abs_diff_between(reference: &[f32], candidate: &[f32]) -> Result<f32> {
    if reference.len() != candidate.len() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!(
                "logits length mismatch: reference={}, candidate={}",
                reference.len(),
                candidate.len()
            ),
        )
        .into());
    }

    Ok(reference
        .iter()
        .zip(candidate.iter())
        .map(|(reference, candidate)| (reference - candidate).abs())
        .fold(0.0, f32::max))
}

fn mean_abs_diff_between(reference: &[f32], candidate: &[f32]) -> Result<f32> {
    if reference.len() != candidate.len() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!(
                "logits length mismatch: reference={}, candidate={}",
                reference.len(),
                candidate.len()
            ),
        )
        .into());
    }
    if reference.is_empty() {
        return Err(io::Error::new(ErrorKind::InvalidInput, "logits must be nonempty").into());
    }

    let sum: f32 = reference
        .iter()
        .zip(candidate.iter())
        .map(|(reference, candidate)| (reference - candidate).abs())
        .sum();
    Ok(sum / reference.len() as f32)
}

fn top_k_logprobs_with_logsumexp(logits: &[f32], top_k: usize, logsumexp: f32) -> Vec<TokenLogit> {
    let mut indexed: Vec<_> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, logit)| TokenLogit {
            token_id: token_id as u32,
            logit,
            logprob: logit - logsumexp,
        })
        .collect();
    indexed.sort_by(|left, right| right.logit.total_cmp(&left.logit));
    indexed.truncate(top_k.min(indexed.len()));
    indexed
}
