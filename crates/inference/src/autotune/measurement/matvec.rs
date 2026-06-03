use std::{collections::HashMap, sync::Arc};

use cuda_core::{CudaFunction, CudaModule, CudaStream, DeviceBuffer};
use nn_rust_profiling::{
    OptimizationTiming, OptimizationTimingSegment, ProfileDuration, ProfileTimeSource,
    ProfileTimer, SampleStats,
};

use super::super::{
    KernelArtifactStore, KernelCandidateMetadata, KernelMaterialization, MatvecRustCudaGenerator,
    SearchScore,
};
use super::{
    KernelAutotuneMeasureOptions, KernelAutotuneMeasureResult,
    compile::emit_and_compile_generated_kernel_scratch,
    invalid_data, invalid_input,
    matvec_launch::{GeneratedMatvecModule, launch_matvec_symbol},
    validation::{
        compare_matvec_output, cpu_matvec_bf16_reference, fill_bf16_matrix, fill_stress_slice,
    },
};
use crate::{
    dtypes::Bf16,
    layout::{MatrixLayout, RowMajor},
};

pub struct MatvecBf16MeasuredAutotuneScorer<'a> {
    stream: &'a Arc<CudaStream>,
    generated_store: KernelArtifactStore,
    generated_modules: HashMap<String, GeneratedMatvecModule>,
    existing_function: CudaFunction,
    dev_input: DeviceBuffer<f32>,
    dev_weight: DeviceBuffer<Bf16>,
    dev_output: DeviceBuffer<f32>,
    expected: Vec<f32>,
    rows: usize,
    cols: usize,
    options: KernelAutotuneMeasureOptions,
}

impl<'a> MatvecBf16MeasuredAutotuneScorer<'a> {
    pub fn new(
        stream: &'a Arc<CudaStream>,
        module: &'a Arc<CudaModule>,
        rows: usize,
        cols: usize,
        options: KernelAutotuneMeasureOptions,
        generated_store: KernelArtifactStore,
    ) -> KernelAutotuneMeasureResult<Self> {
        let weight_layout = MatrixLayout::<RowMajor>::packed(rows, cols);
        let mut seed = 0x4d41_5456_4543_4155_u64 ^ ((rows as u64) << 32) ^ (cols as u64);
        let mut input = vec![0.0_f32; cols];
        let mut weight = vec![Bf16::from_bits(0); weight_layout.capacity()];
        let output = vec![0.0_f32; rows];
        fill_stress_slice(&mut input, &mut seed, 1.0);
        fill_bf16_matrix::<RowMajor>(&mut weight, &weight_layout, rows, cols, &mut seed);
        let expected = cpu_matvec_bf16_reference(&input, &weight, &weight_layout, rows, cols);
        let existing_function = module.load_function("matvec_bf16_kernel")?;

        Ok(Self {
            stream,
            generated_store,
            generated_modules: HashMap::new(),
            existing_function,
            dev_input: DeviceBuffer::from_host(stream, &input)?,
            dev_weight: DeviceBuffer::from_host(stream, &weight)?,
            dev_output: DeviceBuffer::from_host(stream, &output)?,
            expected,
            rows,
            cols,
            options,
        })
    }

    pub fn score_candidate(
        &mut self,
        candidate: &KernelCandidateMetadata,
    ) -> KernelAutotuneMeasureResult<Option<SearchScore>> {
        if candidate.family != "matvec-bf16-row-major" {
            return Ok(None);
        }
        let setup_segments = self.prepare_candidate(candidate)?;
        for _ in 0..self.options.warmup_count {
            self.launch_candidate(candidate)?;
        }
        self.stream.synchronize()?;

        let mut samples = Vec::with_capacity(self.options.repeat_count);
        for _ in 0..self.options.repeat_count {
            let start = self
                .stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            self.launch_candidate(candidate)?;
            let end = self
                .stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            let seconds = start.elapsed_ms(&end)? as f64 / 1_000.0;
            if seconds.is_finite() {
                samples.push(seconds);
            }
        }

        let sample_stats = SampleStats::from_finite_samples(&samples).ok_or_else(|| {
            invalid_data("kernel-autotune-matvec measurement produced no finite samples")
        })?;
        let selected = ProfileDuration::from_seconds_f64(sample_stats.median).ok_or_else(|| {
            invalid_data("kernel-autotune-matvec measurement median was not a finite duration")
        })?;
        let timing = OptimizationTiming::new(
            ProfileTimeSource::CudaEvent,
            self.options.warmup_count,
            sample_stats,
            selected,
        )
        .with_setup_segments(&setup_segments);
        let actual = self.dev_output.to_host_vec(self.stream)?;
        compare_matvec_output(
            &format!("kernel-autotune-matvec {}", candidate.launch.kernel),
            &actual,
            &self.expected,
            self.rows,
            self.cols,
        )?;
        Ok(SearchScore::measured_with_timing(
            sample_stats.median,
            timing,
        ))
    }

    fn prepare_candidate(
        &mut self,
        candidate: &KernelCandidateMetadata,
    ) -> KernelAutotuneMeasureResult<Vec<OptimizationTimingSegment>> {
        match &candidate.generated.materialization {
            KernelMaterialization::Existing { .. } => Ok(Vec::new()),
            KernelMaterialization::Generated { .. }
            | KernelMaterialization::DeferredGenerated { .. } => {
                self.ensure_generated_bf16_matvec(candidate)
            }
        }
    }

    fn launch_candidate(
        &mut self,
        candidate: &KernelCandidateMetadata,
    ) -> KernelAutotuneMeasureResult<()> {
        match &candidate.generated.materialization {
            KernelMaterialization::Existing { symbol } => {
                if *symbol != "matvec_bf16_kernel" {
                    return Err(invalid_input(format!(
                        "kernel-autotune-matvec cannot measure existing matvec symbol {symbol:?}"
                    )));
                }
                launch_matvec_symbol(
                    self.stream,
                    &self.existing_function,
                    candidate,
                    &self.dev_input,
                    &self.dev_weight,
                    &mut self.dev_output,
                    self.rows,
                    self.cols,
                )
            }
            KernelMaterialization::Generated { .. }
            | KernelMaterialization::DeferredGenerated { .. } => {
                self.launch_generated_bf16_matvec(candidate)
            }
        }
    }

    fn ensure_generated_bf16_matvec(
        &mut self,
        candidate: &KernelCandidateMetadata,
    ) -> KernelAutotuneMeasureResult<Vec<OptimizationTimingSegment>> {
        let artifact_key = candidate.artifact_key().hex();
        if self.generated_modules.contains_key(&artifact_key) {
            return Ok(Vec::new());
        }

        let (emitted, compiled, scratch, mut setup_segments) =
            emit_and_compile_generated_kernel_scratch(
                &self.generated_store,
                candidate,
                &MatvecRustCudaGenerator,
            )?;

        let ptx_path = compiled
            .ptx_path
            .to_str()
            .ok_or_else(|| invalid_input("generated PTX path is not valid UTF-8"))?;
        let timer = ProfileTimer::start();
        let module = self.stream.context().load_module_from_file(ptx_path)?;
        setup_segments.push(OptimizationTimingSegment::new(
            "load-generated-module",
            ProfileTimeSource::WallClock,
            timer.elapsed(),
        ));

        let timer = ProfileTimer::start();
        let function = module.load_function(&emitted.symbol)?;
        setup_segments.push(OptimizationTimingSegment::new(
            "load-generated-symbol",
            ProfileTimeSource::WallClock,
            timer.elapsed(),
        ));

        setup_segments.push(OptimizationTimingSegment::new(
            "cleanup-compile-scratch",
            ProfileTimeSource::WallClock,
            scratch.cleanup()?,
        ));

        self.generated_modules.insert(
            artifact_key.clone(),
            GeneratedMatvecModule {
                _module: module,
                function,
            },
        );

        Ok(setup_segments)
    }

    fn launch_generated_bf16_matvec(
        &mut self,
        candidate: &KernelCandidateMetadata,
    ) -> KernelAutotuneMeasureResult<()> {
        let artifact_key = candidate.artifact_key().hex();
        self.ensure_generated_bf16_matvec(candidate)?;
        let Some(generated) = self.generated_modules.get(&artifact_key) else {
            return Err(invalid_input(format!(
                "generated matvec function cache miss for artifact {artifact_key}"
            )));
        };
        launch_matvec_symbol(
            self.stream,
            &generated.function,
            candidate,
            &self.dev_input,
            &self.dev_weight,
            &mut self.dev_output,
            self.rows,
            self.cols,
        )
    }
}
