use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer};
use nn_rust_inference::{
    dtypes::Bf16,
    layout::{MatrixLayout, RowMajor},
    ops,
};
use nn_rust_profiling::{OptimizationTiming, ProfileDuration, ProfileTimeSource, SampleStats};

use super::{
    KernelAutotuneMeasureOptions, KernelAutotuneMeasureResult, invalid_data, invalid_input,
    validation::{fill_bf16_matrix, fill_stress_slice},
};
use crate::autotune::{KernelCandidateMetadata, SearchScore, Top1Bf16SearchProblem};

pub struct Top1Bf16MeasuredAutotuneScorer<'a> {
    stream: &'a Arc<CudaStream>,
    module: &'a Arc<CudaModule>,
    dev_input: DeviceBuffer<f32>,
    dev_weight: DeviceBuffer<Bf16>,
    dev_partial_tokens: DeviceBuffer<u32>,
    dev_partial_logits: DeviceBuffer<f32>,
    dev_packed: DeviceBuffer<u64>,
    expected_token: u32,
    expected_logit: f32,
    tolerance: f32,
    options: KernelAutotuneMeasureOptions,
}

impl<'a> Top1Bf16MeasuredAutotuneScorer<'a> {
    pub fn new(
        stream: &'a Arc<CudaStream>,
        module: &'a Arc<CudaModule>,
        rows: usize,
        cols: usize,
        options: KernelAutotuneMeasureOptions,
    ) -> KernelAutotuneMeasureResult<Self> {
        let weight_layout = MatrixLayout::<RowMajor>::packed(rows, cols);
        let mut seed = 0x544f_5031_4246_3136_u64 ^ ((rows as u64) << 32) ^ cols as u64;
        let mut input = vec![0.0_f32; cols];
        let mut weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
        fill_stress_slice(&mut input, &mut seed, 1.0);
        fill_bf16_matrix::<RowMajor>(&mut weight, &weight_layout, rows, cols, &mut seed);

        let forced_row = if rows == 1 { 0 } else { rows / 3 };
        let mut expected_logit = 0.0_f32;
        for col in 0..cols {
            let forced_weight = if input[col] >= 0.0 { 4.0 } else { -4.0 };
            weight[weight_layout.offset(forced_row, col)] = Bf16::from_f32(forced_weight);
            expected_logit += input[col] * forced_weight;
        }
        let tolerance =
            (1.0e-4_f32 * (cols.max(1) as f32).sqrt()).max(expected_logit.abs() * 2.0e-5);

        Ok(Self {
            stream,
            module,
            dev_input: DeviceBuffer::from_host(stream, &input)?,
            dev_weight: DeviceBuffer::from_host(stream, &weight)?,
            dev_partial_tokens: DeviceBuffer::<u32>::zeroed(stream, rows)?,
            dev_partial_logits: DeviceBuffer::<f32>::zeroed(stream, rows)?,
            dev_packed: DeviceBuffer::<u64>::zeroed(stream, 1)?,
            expected_token: forced_row as u32,
            expected_logit,
            tolerance,
            options,
        })
    }

    pub fn score_candidate(
        &mut self,
        candidate: &KernelCandidateMetadata,
    ) -> KernelAutotuneMeasureResult<Option<SearchScore>> {
        if candidate.family != Top1Bf16SearchProblem::FAMILY {
            return Ok(None);
        }
        let Some(rows_per_block) = Top1Bf16SearchProblem::candidate_rows_per_block(candidate)
        else {
            return Ok(None);
        };

        for _ in 0..self.options.warmup_count {
            self.launch_rows_per_block(rows_per_block)?;
        }
        self.stream.synchronize()?;

        let mut samples = Vec::with_capacity(self.options.repeat_count);
        for _ in 0..self.options.repeat_count {
            let start = self
                .stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            self.launch_rows_per_block(rows_per_block)?;
            let end = self
                .stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            let seconds = start.elapsed_ms(&end)? as f64 / 1_000.0;
            if seconds.is_finite() {
                samples.push(seconds);
            }
        }

        let sample_stats = SampleStats::from_finite_samples(&samples).ok_or_else(|| {
            invalid_data("kernel-autotune-top1-bf16 measurement produced no finite samples")
        })?;
        let selected = ProfileDuration::from_seconds_f64(sample_stats.median).ok_or_else(|| {
            invalid_data("kernel-autotune-top1-bf16 measurement median was not finite")
        })?;
        let timing = OptimizationTiming::new(
            ProfileTimeSource::CudaEvent,
            self.options.warmup_count,
            sample_stats,
            selected,
        );
        self.verify_output(candidate)?;
        Ok(SearchScore::measured_with_timing(
            sample_stats.median,
            timing,
        ))
    }

    fn launch_rows_per_block(&mut self, rows_per_block: u32) -> KernelAutotuneMeasureResult<()> {
        match rows_per_block {
            1 => ops::linear_top1_bf16_rows1(
                self.stream,
                self.module,
                &self.dev_input,
                &self.dev_weight,
                &mut self.dev_partial_tokens,
                &mut self.dev_partial_logits,
                &mut self.dev_packed,
            )?,
            2 => ops::linear_top1_bf16_rows2(
                self.stream,
                self.module,
                &self.dev_input,
                &self.dev_weight,
                &mut self.dev_partial_tokens,
                &mut self.dev_partial_logits,
                &mut self.dev_packed,
            )?,
            4 => ops::linear_top1_bf16(
                self.stream,
                self.module,
                &self.dev_input,
                &self.dev_weight,
                &mut self.dev_partial_tokens,
                &mut self.dev_partial_logits,
                &mut self.dev_packed,
            )?,
            8 => ops::linear_top1_bf16_rows8(
                self.stream,
                self.module,
                &self.dev_input,
                &self.dev_weight,
                &mut self.dev_partial_tokens,
                &mut self.dev_partial_logits,
                &mut self.dev_packed,
            )?,
            other => {
                return Err(invalid_input(format!(
                    "kernel-autotune-top1-bf16 cannot measure rows_per_block={other}"
                )));
            }
        }
        Ok(())
    }

    fn verify_output(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> KernelAutotuneMeasureResult<()> {
        let packed = self.dev_packed.to_host_vec(self.stream)?[0];
        let token = packed as u32;
        let logit = f32::from_bits((packed >> 32) as u32);
        let diff = (logit - self.expected_logit).abs();
        if token != self.expected_token || diff > self.tolerance {
            return Err(invalid_data(format!(
                "kernel-autotune-top1-bf16 {} verification failed: expected=({}, {:.8}) got=({}, {:.8}) tolerance={:.8}",
                candidate.launch.kernel,
                self.expected_token,
                self.expected_logit,
                token,
                logit,
                self.tolerance
            )));
        }
        Ok(())
    }
}
