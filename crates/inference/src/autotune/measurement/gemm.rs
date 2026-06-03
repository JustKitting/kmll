use std::{collections::HashMap, sync::Arc};

use cuda_core::{CudaModule, CudaStream, DeviceBuffer};
use nn_rust_profiling::{
    OptimizationTiming, OptimizationTimingSegment, ProfileDuration, ProfileTimeSource,
    ProfileTimer, SampleStats,
};

use super::super::{
    GemmRustCudaGenerator, KernelArtifactStore, KernelCandidateMetadata, KernelMaterialization,
    SearchScore,
};
use super::{
    KernelAutotuneMeasureOptions, KernelAutotuneMeasureResult,
    compile::emit_and_compile_generated_kernel_scratch,
    gemm_launch::{GeneratedGemmModule, launch_generated_gemm_symbol},
    invalid_data, invalid_input,
    validation::{compare_gemm_output, cpu_gemm_bf16_reference, fill_bf16_matrix, fill_matrix},
};
use crate::{
    dtypes::Bf16,
    layout::{ColumnMajor, MatrixLayout, RowMajor},
    ops,
};

pub struct GemmF32Bf16MeasuredAutotuneScorer<'a> {
    stream: &'a Arc<CudaStream>,
    module: &'a Arc<CudaModule>,
    generated_store: KernelArtifactStore,
    generated_modules: HashMap<String, GeneratedGemmModule>,
    dev_a: DeviceBuffer<f32>,
    dev_b: DeviceBuffer<Bf16>,
    dev_c: DeviceBuffer<f32>,
    expected: Vec<f32>,
    c_layout: MatrixLayout<RowMajor>,
    m: usize,
    n: usize,
    k: usize,
    options: KernelAutotuneMeasureOptions,
}

impl<'a> GemmF32Bf16MeasuredAutotuneScorer<'a> {
    pub fn new(
        stream: &'a Arc<CudaStream>,
        module: &'a Arc<CudaModule>,
        m: usize,
        n: usize,
        k: usize,
        options: KernelAutotuneMeasureOptions,
        generated_store: KernelArtifactStore,
    ) -> KernelAutotuneMeasureResult<Self> {
        let a_layout = MatrixLayout::<RowMajor>::packed(m, k);
        let b_layout = MatrixLayout::<ColumnMajor>::packed(k, n);
        let c_layout = MatrixLayout::<RowMajor>::packed(m, n);
        let mut seed =
            0x5045_5246_4155_544f_u64 ^ ((m as u64) << 32) ^ ((n as u64) << 16) ^ k as u64;
        let mut a = vec![0.0_f32; a_layout.capacity()];
        let mut b = vec![Bf16::from_bits(0); b_layout.capacity()];
        let c = vec![0.0_f32; c_layout.capacity()];
        fill_matrix::<RowMajor>(&mut a, &a_layout, m, k, &mut seed);
        fill_bf16_matrix::<ColumnMajor>(&mut b, &b_layout, k, n, &mut seed);
        let expected = cpu_gemm_bf16_reference(
            &a, &a_layout, &b, &b_layout, &c, &c_layout, m, n, k, 1.0, 0.0,
        );

        Ok(Self {
            stream,
            module,
            generated_store,
            generated_modules: HashMap::new(),
            dev_a: DeviceBuffer::from_host(stream, &a)?,
            dev_b: DeviceBuffer::from_host(stream, &b)?,
            dev_c: DeviceBuffer::from_host(stream, &c)?,
            expected,
            c_layout,
            m,
            n,
            k,
            options,
        })
    }

    pub fn score_candidate(
        &mut self,
        candidate: &KernelCandidateMetadata,
    ) -> KernelAutotuneMeasureResult<Option<SearchScore>> {
        if candidate.family != "gemm-f32-bf16-row-col-row" {
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
            invalid_data("kernel-autotune-gemm measurement produced no finite samples")
        })?;
        let selected = ProfileDuration::from_seconds_f64(sample_stats.median).ok_or_else(|| {
            invalid_data("kernel-autotune-gemm measurement median was not a finite duration")
        })?;
        let timing = OptimizationTiming::new(
            ProfileTimeSource::CudaEvent,
            self.options.warmup_count,
            sample_stats,
            selected,
        )
        .with_setup_segments(&setup_segments);
        let actual = self.dev_c.to_host_vec(self.stream)?;
        compare_gemm_output(
            &format!("kernel-autotune-gemm {}", candidate.launch.kernel),
            &actual,
            &self.expected,
            &self.c_layout,
            self.m,
            self.n,
            self.k,
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
                self.ensure_generated_bf16_gemm(candidate)
            }
        }
    }

    fn launch_candidate(
        &mut self,
        candidate: &KernelCandidateMetadata,
    ) -> KernelAutotuneMeasureResult<()> {
        match &candidate.generated.materialization {
            KernelMaterialization::Existing { symbol } => {
                if *symbol != "gemm_f32_bf16_tiled_kernel" {
                    return Err(invalid_input(format!(
                        "kernel-autotune-gemm cannot measure existing GEMM symbol {symbol:?}"
                    )));
                }
                self.launch_existing_bf16_gemm()
            }
            KernelMaterialization::Generated { .. }
            | KernelMaterialization::DeferredGenerated { .. } => {
                self.launch_generated_bf16_gemm(candidate)
            }
        }
    }

    fn launch_existing_bf16_gemm(&mut self) -> KernelAutotuneMeasureResult<()> {
        ops::gemm_f32_bf16::<RowMajor, ColumnMajor, RowMajor>(
            self.stream,
            self.module,
            &self.dev_a,
            &self.dev_b,
            &mut self.dev_c,
            self.m,
            self.n,
            self.k,
            1.0,
            0.0,
        )?;
        Ok(())
    }

    fn ensure_generated_bf16_gemm(
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
                &GemmRustCudaGenerator,
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
            GeneratedGemmModule {
                _module: module,
                function,
            },
        );

        Ok(setup_segments)
    }

    fn launch_generated_bf16_gemm(
        &mut self,
        candidate: &KernelCandidateMetadata,
    ) -> KernelAutotuneMeasureResult<()> {
        let artifact_key = candidate.artifact_key().hex();
        self.ensure_generated_bf16_gemm(candidate)?;
        let Some(generated) = self.generated_modules.get(&artifact_key) else {
            return Err(invalid_input(format!(
                "generated GEMM module cache miss for artifact {artifact_key}"
            )));
        };
        launch_generated_gemm_symbol(
            self.stream,
            generated,
            candidate,
            &self.dev_a,
            &self.dev_b,
            &mut self.dev_c,
            self.m,
            self.n,
            self.k,
        )
    }
}
