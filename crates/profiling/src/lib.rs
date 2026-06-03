use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileDuration(Duration);

impl ProfileDuration {
    pub const ZERO: Self = Self(Duration::ZERO);

    pub const fn from_duration(duration: Duration) -> Self {
        Self(duration)
    }

    pub const fn from_nanos(nanos: u64) -> Self {
        Self(Duration::from_nanos(nanos))
    }

    pub fn from_seconds_f64(seconds: f64) -> Option<Self> {
        if seconds.is_finite() && seconds >= 0.0 {
            Some(Self(Duration::from_secs_f64(seconds)))
        } else {
            None
        }
    }

    pub const fn as_duration(self) -> Duration {
        self.0
    }

    pub fn as_seconds_f64(self) -> f64 {
        self.0.as_secs_f64()
    }

    pub fn as_nanos_u128(self) -> u128 {
        self.0.as_nanos()
    }

    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

#[derive(Debug, Clone)]
pub struct ProfileTimer {
    started_at: Instant,
}

impl ProfileTimer {
    pub fn start() -> Self {
        Self {
            started_at: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> ProfileDuration {
        ProfileDuration::from_duration(self.started_at.elapsed())
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed().as_seconds_f64()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileTimeSource {
    WallClock,
    CudaEvent,
    SelfTimeAccounting,
}

impl ProfileTimeSource {
    pub const fn label(self) -> &'static str {
        match self {
            Self::WallClock => "wall-clock",
            Self::CudaEvent => "cuda-event",
            Self::SelfTimeAccounting => "host-self-time",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchDimensions {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl LaunchDimensions {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub const fn from_tuple(dim: (u32, u32, u32)) -> Self {
        Self::new(dim.0, dim.1, dim.2)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaLaunchSpec {
    pub kernel: String,
    pub grid_dim: LaunchDimensions,
    pub block_dim: LaunchDimensions,
    pub shared_mem_bytes: u32,
}

impl CudaLaunchSpec {
    pub fn new(
        kernel: impl Into<String>,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
    ) -> Self {
        Self {
            kernel: kernel.into(),
            grid_dim: LaunchDimensions::from_tuple(grid_dim),
            block_dim: LaunchDimensions::from_tuple(block_dim),
            shared_mem_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericKind {
    Bool,
    Fp4,
    F6E2M3,
    F6E3M2,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    F8E8M0,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    C64,
    F64,
    I64,
    U64,
}

impl NumericKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Fp4 => "fp4",
            Self::F6E2M3 => "f6_e2m3",
            Self::F6E3M2 => "f6_e3m2",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::F8E5M2 => "f8_e5m2",
            Self::F8E4M3 => "f8_e4m3",
            Self::F8E8M0 => "f8_e8m0",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::F32 => "f32",
            Self::C64 => "c64",
            Self::F64 => "f64",
            Self::I64 => "i64",
            Self::U64 => "u64",
        }
    }

    pub const fn bits(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Fp4 => 4,
            Self::F6E2M3 | Self::F6E3M2 => 6,
            Self::U8 | Self::I8 | Self::F8E5M2 | Self::F8E4M3 | Self::F8E8M0 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::C64 | Self::F64 | Self::I64 | Self::U64 => 64,
        }
    }
}

pub trait ProfileNumericType: Copy + 'static {
    const KIND: NumericKind;
    const ACCUMULATOR: NumericKind;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32ProfileType;

impl ProfileNumericType for F32ProfileType {
    const KIND: NumericKind = NumericKind::F32;
    const ACCUMULATOR: NumericKind = NumericKind::F32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16ProfileType;

impl ProfileNumericType for Bf16ProfileType {
    const KIND: NumericKind = NumericKind::Bf16;
    const ACCUMULATOR: NumericKind = NumericKind::F32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I8ProfileType;

impl ProfileNumericType for I8ProfileType {
    const KIND: NumericKind = NumericKind::I8;
    const ACCUMULATOR: NumericKind = NumericKind::F32;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorTypeSpec {
    pub dtype: NumericKind,
    pub accumulator: NumericKind,
    pub shape: Vec<usize>,
    pub layout: Option<String>,
}

impl TensorTypeSpec {
    pub fn new(dtype: NumericKind, accumulator: NumericKind, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            dtype,
            accumulator,
            shape: shape.into(),
            layout: None,
        }
    }

    pub fn with_layout(mut self, layout: impl Into<String>) -> Self {
        self.layout = Some(layout.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType<T, const RANK: usize>
where
    T: ProfileNumericType,
{
    pub shape: [usize; RANK],
    pub layout: Option<&'static str>,
    _dtype: std::marker::PhantomData<T>,
}

impl<T, const RANK: usize> TensorType<T, RANK>
where
    T: ProfileNumericType,
{
    pub const fn new(shape: [usize; RANK]) -> Self {
        Self {
            shape,
            layout: None,
            _dtype: std::marker::PhantomData,
        }
    }

    pub const fn with_static_layout(mut self, layout: &'static str) -> Self {
        self.layout = Some(layout);
        self
    }

    pub fn erase(&self) -> TensorTypeSpec {
        let mut spec = TensorTypeSpec::new(T::KIND, T::ACCUMULATOR, self.shape.to_vec());
        if let Some(layout) = self.layout {
            spec = spec.with_layout(layout);
        }
        spec
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationKind {
    Tokenize,
    RuntimeInit,
    Upload,
    Download,
    Elementwise,
    Matvec,
    Gemm,
    RmsNorm,
    Activation,
    Attention,
    Logits,
    QueueSubmit,
    WorkerDispatch,
    CudaSync,
    ModelPhase,
}

impl OperationKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Tokenize => "tokenize",
            Self::RuntimeInit => "runtime-init",
            Self::Upload => "upload",
            Self::Download => "download",
            Self::Elementwise => "elementwise",
            Self::Matvec => "matvec",
            Self::Gemm => "gemm",
            Self::RmsNorm => "rmsnorm",
            Self::Activation => "activation",
            Self::Attention => "attention",
            Self::Logits => "logits",
            Self::QueueSubmit => "queue-submit",
            Self::WorkerDispatch => "worker-dispatch",
            Self::CudaSync => "cuda-sync",
            Self::ModelPhase => "model-phase",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationRoute {
    Host,
    TokioCudaWorker,
    CudaKernel,
    CudaMemcpy,
    CudaSynchronize,
}

impl OperationRoute {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::TokioCudaWorker => "tokio-cuda-worker",
            Self::CudaKernel => "cuda-kernel",
            Self::CudaMemcpy => "cuda-memcpy",
            Self::CudaSynchronize => "cuda-synchronize",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedOperationSpec {
    pub name: String,
    pub kind: OperationKind,
    pub route: OperationRoute,
    pub inputs: Vec<TensorTypeSpec>,
    pub outputs: Vec<TensorTypeSpec>,
    pub launch: Option<CudaLaunchSpec>,
}

impl TypedOperationSpec {
    pub fn new(name: impl Into<String>, kind: OperationKind, route: OperationRoute) -> Self {
        Self {
            name: name.into(),
            kind,
            route,
            inputs: Vec::new(),
            outputs: Vec::new(),
            launch: None,
        }
    }

    pub fn with_input(mut self, input: TensorTypeSpec) -> Self {
        self.inputs.push(input);
        self
    }

    pub fn with_output(mut self, output: TensorTypeSpec) -> Self {
        self.outputs.push(output);
        self
    }

    pub fn with_launch(mut self, launch: CudaLaunchSpec) -> Self {
        self.launch = Some(launch);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationActionOp {
    Split,
    Unroll,
    TileGemm,
    StrideOrder,
}

impl OptimizationActionOp {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Split => "split",
            Self::Unroll => "unroll",
            Self::TileGemm => "tile-gemm",
            Self::StrideOrder => "stride-order",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationActionMaterialization {
    Existing,
    DeferredGenerated,
}

impl OptimizationActionMaterialization {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Existing => "existing",
            Self::DeferredGenerated => "deferred-generated",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationActionArg {
    Factor(u32),
    Tile3d { m: u32, n: u32, k: u32 },
    AxisOrder(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptimizationActionSpec {
    pub op: OptimizationActionOp,
    pub axis: Option<u8>,
    pub arg: OptimizationActionArg,
    pub materialization: OptimizationActionMaterialization,
}

impl OptimizationActionSpec {
    pub const fn split(
        axis: u8,
        factor: u32,
        materialization: OptimizationActionMaterialization,
    ) -> Self {
        Self {
            op: OptimizationActionOp::Split,
            axis: Some(axis),
            arg: OptimizationActionArg::Factor(factor),
            materialization,
        }
    }

    pub const fn unroll(axis: u8, factor: u32) -> Self {
        Self {
            op: OptimizationActionOp::Unroll,
            axis: Some(axis),
            arg: OptimizationActionArg::Factor(factor),
            materialization: OptimizationActionMaterialization::DeferredGenerated,
        }
    }

    pub const fn tile_gemm(
        m: u32,
        n: u32,
        k: u32,
        materialization: OptimizationActionMaterialization,
    ) -> Self {
        Self {
            op: OptimizationActionOp::TileGemm,
            axis: None,
            arg: OptimizationActionArg::Tile3d { m, n, k },
            materialization,
        }
    }

    pub fn stride_order(axes: Vec<u8>) -> Self {
        Self {
            op: OptimizationActionOp::StrideOrder,
            axis: None,
            arg: OptimizationActionArg::AxisOrder(axes),
            materialization: OptimizationActionMaterialization::DeferredGenerated,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationScoreSource {
    Heuristic,
    Measured,
}

impl OptimizationScoreSource {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Heuristic => "heuristic",
            Self::Measured => "measured",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizationTiming {
    pub source: ProfileTimeSource,
    pub warmup_count: usize,
    pub samples: SampleStats,
    pub selected: ProfileDuration,
}

impl OptimizationTiming {
    pub const fn new(
        source: ProfileTimeSource,
        warmup_count: usize,
        samples: SampleStats,
        selected: ProfileDuration,
    ) -> Self {
        Self {
            source,
            warmup_count,
            samples,
            selected,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizationScore {
    pub value: f64,
    pub source: OptimizationScoreSource,
    pub timing: Option<OptimizationTiming>,
}

impl OptimizationScore {
    pub fn heuristic(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self {
            value,
            source: OptimizationScoreSource::Heuristic,
            timing: None,
        })
    }

    pub fn measured(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self {
            value,
            source: OptimizationScoreSource::Measured,
            timing: None,
        })
    }

    pub fn measured_with_timing(value: f64, timing: OptimizationTiming) -> Option<Self> {
        value.is_finite().then_some(Self {
            value,
            source: OptimizationScoreSource::Measured,
            timing: Some(timing),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationCandidateSpec {
    pub family: String,
    pub artifact_key: String,
    pub generator: String,
    pub launch: CudaLaunchSpec,
    pub operation: TypedOperationSpec,
    pub action_trace: Vec<OptimizationActionSpec>,
    pub score: Option<OptimizationScore>,
}

impl OptimizationCandidateSpec {
    pub fn new(
        family: impl Into<String>,
        artifact_key: impl Into<String>,
        generator: impl Into<String>,
        launch: CudaLaunchSpec,
        operation: TypedOperationSpec,
    ) -> Self {
        Self {
            family: family.into(),
            artifact_key: artifact_key.into(),
            generator: generator.into(),
            launch,
            operation,
            action_trace: Vec::new(),
            score: None,
        }
    }

    pub fn with_action_trace(mut self, action_trace: Vec<OptimizationActionSpec>) -> Self {
        self.action_trace = action_trace;
        self
    }

    pub fn with_score(mut self, score: Option<OptimizationScore>) -> Self {
        self.score = score;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProfileSegment {
    pub name: String,
    pub category: String,
    pub source: ProfileTimeSource,
    pub start_offset: Option<ProfileDuration>,
    pub duration: ProfileDuration,
    pub launch: Option<CudaLaunchSpec>,
    pub operation: Option<TypedOperationSpec>,
}

impl ProfileSegment {
    pub fn new(
        name: impl Into<String>,
        category: impl Into<String>,
        source: ProfileTimeSource,
        start_offset: Option<ProfileDuration>,
        duration: ProfileDuration,
    ) -> Self {
        Self {
            name: name.into(),
            category: category.into(),
            source,
            start_offset,
            duration,
            launch: None,
            operation: None,
        }
    }

    pub fn wall_clock(
        name: impl Into<String>,
        category: impl Into<String>,
        start_offset: ProfileDuration,
        duration: ProfileDuration,
    ) -> Self {
        Self::new(
            name,
            category,
            ProfileTimeSource::WallClock,
            Some(start_offset),
            duration,
        )
    }

    pub fn cuda_event(
        name: impl Into<String>,
        category: impl Into<String>,
        duration: ProfileDuration,
        launch: Option<CudaLaunchSpec>,
    ) -> Self {
        let mut segment = Self::new(name, category, ProfileTimeSource::CudaEvent, None, duration);
        segment.launch = launch;
        segment
    }

    pub fn self_time_accounting(
        name: impl Into<String>,
        category: impl Into<String>,
        start_offset: ProfileDuration,
        duration: ProfileDuration,
    ) -> Self {
        Self::new(
            name,
            category,
            ProfileTimeSource::SelfTimeAccounting,
            Some(start_offset),
            duration,
        )
    }

    pub fn with_launch(mut self, launch: CudaLaunchSpec) -> Self {
        self.launch = Some(launch);
        self
    }

    pub fn with_operation(mut self, operation: TypedOperationSpec) -> Self {
        if self.launch.is_none() {
            self.launch = operation.launch.clone();
        }
        self.operation = Some(operation);
        self
    }

    pub fn end_offset(&self) -> Option<ProfileDuration> {
        self.start_offset.map(|start| {
            ProfileDuration::from_duration(start.as_duration().saturating_add(self.duration.0))
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileCoverage {
    pub total_seconds: f64,
    pub measured_covered_seconds: f64,
    pub self_time_accounting_seconds: f64,
    pub credited_work_seconds: f64,
    pub coverage_ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccreditedProfile {
    pub name: String,
    pub total: ProfileDuration,
    pub coverage: ProfileCoverage,
    pub segments: Vec<ProfileSegment>,
}

impl AccreditedProfile {
    pub fn new(
        name: impl Into<String>,
        total: ProfileDuration,
        measured_segments: Vec<ProfileSegment>,
        accounting_name: impl AsRef<str>,
        accounting_category: impl AsRef<str>,
    ) -> Self {
        let total_nanos = total.as_nanos_u128();
        let mut segments = measured_segments;
        let measured_intervals = timeline_intervals(total, &segments);
        let measured_covered_nanos = covered_nanos(&measured_intervals);
        let accounting_intervals = complement_intervals(total_nanos, &measured_intervals);
        for (index, (start, end)) in accounting_intervals.into_iter().enumerate() {
            let duration = end.saturating_sub(start);
            if duration == 0 {
                continue;
            }
            let label = if index == 0 {
                accounting_name.as_ref().to_string()
            } else {
                format!("{}[{index}]", accounting_name.as_ref())
            };
            segments.push(ProfileSegment::self_time_accounting(
                label,
                accounting_category.as_ref(),
                ProfileDuration::from_nanos(saturating_u128_to_u64(start)),
                ProfileDuration::from_nanos(saturating_u128_to_u64(duration)),
            ));
        }

        let self_time_accounting_nanos = segments
            .iter()
            .filter(|segment| segment.source == ProfileTimeSource::SelfTimeAccounting)
            .map(|segment| segment.duration.as_nanos_u128())
            .sum::<u128>();
        let credited_work_nanos = segments
            .iter()
            .map(|segment| segment.duration.as_nanos_u128())
            .sum::<u128>();
        let coverage_ratio = if total_nanos == 0 {
            1.0
        } else {
            (measured_covered_nanos + self_time_accounting_nanos) as f64 / total_nanos as f64
        };
        let coverage = ProfileCoverage {
            total_seconds: total.as_seconds_f64(),
            measured_covered_seconds: nanos_to_seconds(measured_covered_nanos),
            self_time_accounting_seconds: nanos_to_seconds(self_time_accounting_nanos),
            credited_work_seconds: nanos_to_seconds(credited_work_nanos),
            coverage_ratio,
        };

        Self {
            name: name.into(),
            total,
            coverage,
            segments,
        }
    }

    pub fn push_json(&self, out: &mut String, indent: usize) {
        push_indent(out, indent);
        out.push_str("{\n");
        push_json_field_string(out, "name", &self.name, indent + 2, true);
        push_json_field_f64(
            out,
            "total_seconds",
            self.total.as_seconds_f64(),
            indent + 2,
            true,
        );
        push_profile_coverage_json(out, "coverage", self.coverage, indent + 2, true);
        push_indent(out, indent + 2);
        out.push_str("\"segments\": [\n");
        for (index, segment) in self.segments.iter().enumerate() {
            if index > 0 {
                out.push_str(",\n");
            }
            push_profile_segment_json(out, segment, indent + 4);
        }
        out.push('\n');
        push_indent(out, indent + 2);
        out.push_str("]\n");
        push_indent(out, indent);
        out.push('}');
    }

    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.push_json(&mut out, 0);
        out.push('\n');
        out
    }
}

#[derive(Debug, Clone)]
pub struct ProfileTimeline {
    name: String,
    started_at: Instant,
    segments: Vec<ProfileSegment>,
}

impl ProfileTimeline {
    pub fn start(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            started_at: Instant::now(),
            segments: Vec::new(),
        }
    }

    pub fn time<T>(
        &mut self,
        name: impl Into<String>,
        category: impl Into<String>,
        f: impl FnOnce() -> T,
    ) -> T {
        let name = name.into();
        let category = category.into();
        let started_at = Instant::now();
        let result = f();
        let ended_at = Instant::now();
        self.push_wall_clock_span(name, category, started_at, ended_at);
        result
    }

    pub fn record_cuda_event(
        &mut self,
        name: impl Into<String>,
        category: impl Into<String>,
        duration: ProfileDuration,
        launch: Option<CudaLaunchSpec>,
    ) {
        self.segments
            .push(ProfileSegment::cuda_event(name, category, duration, launch));
    }

    pub fn push_segment(&mut self, segment: ProfileSegment) {
        self.segments.push(segment);
    }

    pub fn finalize(
        self,
        accounting_name: impl AsRef<str>,
        accounting_category: impl AsRef<str>,
    ) -> AccreditedProfile {
        let total = ProfileDuration::from_duration(self.started_at.elapsed());
        AccreditedProfile::new(
            self.name,
            total,
            self.segments,
            accounting_name,
            accounting_category,
        )
    }

    fn push_wall_clock_span(
        &mut self,
        name: String,
        category: String,
        started_at: Instant,
        ended_at: Instant,
    ) {
        let start_offset = started_at
            .checked_duration_since(self.started_at)
            .unwrap_or(Duration::ZERO);
        let duration = ended_at.duration_since(started_at);
        self.segments.push(ProfileSegment::wall_clock(
            name,
            category,
            ProfileDuration::from_duration(start_offset),
            ProfileDuration::from_duration(duration),
        ));
    }
}

pub type ProfileSpanId = usize;

#[derive(Debug, Clone, PartialEq)]
pub struct ProfileSpan {
    pub id: ProfileSpanId,
    pub parent_id: Option<ProfileSpanId>,
    pub name: String,
    pub category: String,
    pub source: ProfileTimeSource,
    pub start_offset: ProfileDuration,
    pub duration: ProfileDuration,
    pub self_duration: ProfileDuration,
    pub operation: Option<TypedOperationSpec>,
}

impl ProfileSpan {
    pub fn end_offset(&self) -> ProfileDuration {
        ProfileDuration::from_duration(
            self.start_offset
                .as_duration()
                .saturating_add(self.duration.0),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProfileTrace {
    pub name: String,
    pub total: ProfileDuration,
    pub spans: Vec<ProfileSpan>,
}

impl ProfileTrace {
    pub fn root(&self) -> Option<&ProfileSpan> {
        self.spans.iter().find(|span| span.parent_id.is_none())
    }

    pub fn push_json(&self, out: &mut String, indent: usize) {
        push_indent(out, indent);
        out.push_str("{\n");
        push_json_field_string(out, "name", &self.name, indent + 2, true);
        push_json_field_f64(
            out,
            "total_seconds",
            self.total.as_seconds_f64(),
            indent + 2,
            true,
        );
        push_indent(out, indent + 2);
        out.push_str("\"spans\": [\n");
        for (index, span) in self.spans.iter().enumerate() {
            if index > 0 {
                out.push_str(",\n");
            }
            push_profile_span_json(out, span, indent + 4);
        }
        out.push('\n');
        push_indent(out, indent + 2);
        out.push_str("]\n");
        push_indent(out, indent);
        out.push('}');
    }

    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.push_json(&mut out, 0);
        out.push('\n');
        out
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueueOperationProfile {
    pub operation: TypedOperationSpec,
    pub worker_index: usize,
    pub send_wait: ProfileDuration,
    pub queue_to_worker: ProfileDuration,
    pub worker_execute: ProfileDuration,
    pub completion_wait: ProfileDuration,
    pub total: ProfileDuration,
}

impl QueueOperationProfile {
    pub fn push_json(&self, out: &mut String, indent: usize) {
        push_indent(out, indent);
        out.push_str("{\n");
        push_json_field_usize(out, "worker_index", self.worker_index, indent + 2, true);
        push_json_field_operation_spec(out, "operation", Some(&self.operation), indent + 2, true);
        push_json_field_f64(
            out,
            "send_wait_seconds",
            self.send_wait.as_seconds_f64(),
            indent + 2,
            true,
        );
        push_json_field_f64(
            out,
            "queue_to_worker_seconds",
            self.queue_to_worker.as_seconds_f64(),
            indent + 2,
            true,
        );
        push_json_field_f64(
            out,
            "worker_execute_seconds",
            self.worker_execute.as_seconds_f64(),
            indent + 2,
            true,
        );
        push_json_field_f64(
            out,
            "completion_wait_seconds",
            self.completion_wait.as_seconds_f64(),
            indent + 2,
            true,
        );
        push_json_field_f64(
            out,
            "total_seconds",
            self.total.as_seconds_f64(),
            indent + 2,
            false,
        );
        push_indent(out, indent);
        out.push('}');
    }
}

#[derive(Debug, Clone)]
pub struct ProfileTraceBuilder {
    name: String,
    started_at: Instant,
    next_id: ProfileSpanId,
    open: Vec<OpenProfileSpan>,
    spans: Vec<ProfileSpan>,
}

#[derive(Debug, Clone)]
struct OpenProfileSpan {
    id: ProfileSpanId,
}

impl ProfileTraceBuilder {
    pub fn start(name: impl Into<String>) -> Self {
        let started_at = Instant::now();
        Self {
            name: name.into(),
            started_at,
            next_id: 1,
            open: vec![OpenProfileSpan { id: 0 }],
            spans: Vec::new(),
        }
    }

    pub fn time<T>(
        &mut self,
        name: impl Into<String>,
        category: impl Into<String>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.time_inner(
            name.into(),
            category.into(),
            ProfileTimeSource::WallClock,
            None,
            f,
        )
    }

    pub fn time_operation<T>(
        &mut self,
        operation: TypedOperationSpec,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let name = operation.name.clone();
        let category = operation.kind.label().to_string();
        self.time_inner(
            name,
            category,
            ProfileTimeSource::WallClock,
            Some(operation),
            f,
        )
    }

    pub fn finalize(mut self) -> ProfileTrace {
        let total = ProfileDuration::from_duration(self.started_at.elapsed());
        let root = ProfileSpan {
            id: 0,
            parent_id: None,
            name: self.name.clone(),
            category: "root".to_string(),
            source: ProfileTimeSource::WallClock,
            start_offset: ProfileDuration::ZERO,
            duration: total,
            self_duration: ProfileDuration::ZERO,
            operation: None,
        };
        self.spans.push(root);
        let spans = with_self_durations(self.spans);
        ProfileTrace {
            name: self.name,
            total,
            spans,
        }
    }

    fn time_inner<T>(
        &mut self,
        name: String,
        category: String,
        source: ProfileTimeSource,
        operation: Option<TypedOperationSpec>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let id = self.next_id;
        self.next_id += 1;
        let parent_id = self.open.last().map(|span| span.id);
        let started_at = Instant::now();
        self.open.push(OpenProfileSpan { id });
        let result = f(self);
        let ended_at = Instant::now();
        let popped = self
            .open
            .pop()
            .expect("profile span stack must contain the active span");
        debug_assert_eq!(popped.id, id);
        let start_offset = started_at
            .checked_duration_since(self.started_at)
            .unwrap_or(Duration::ZERO);
        let duration = ended_at.duration_since(started_at);
        self.spans.push(ProfileSpan {
            id,
            parent_id,
            name,
            category,
            source,
            start_offset: ProfileDuration::from_duration(start_offset),
            duration: ProfileDuration::from_duration(duration),
            self_duration: ProfileDuration::ZERO,
            operation,
        });
        result
    }
}

fn with_self_durations(mut spans: Vec<ProfileSpan>) -> Vec<ProfileSpan> {
    let self_durations = spans
        .iter()
        .map(|span| {
            let child_intervals = spans
                .iter()
                .filter(|candidate| candidate.parent_id == Some(span.id))
                .filter_map(|child| clipped_child_interval(span, child))
                .collect::<Vec<_>>();
            let child_covered = covered_nanos(&merge_intervals(child_intervals));
            let self_nanos = span.duration.as_nanos_u128().saturating_sub(child_covered);
            ProfileDuration::from_nanos(saturating_u128_to_u64(self_nanos))
        })
        .collect::<Vec<_>>();
    for (span, self_duration) in spans.iter_mut().zip(self_durations) {
        span.self_duration = self_duration;
    }
    spans.sort_by_key(|span| span.id);
    spans
}

fn clipped_child_interval(parent: &ProfileSpan, child: &ProfileSpan) -> Option<(u128, u128)> {
    let parent_start = parent.start_offset.as_nanos_u128();
    let parent_end = parent.end_offset().as_nanos_u128();
    let child_start = child.start_offset.as_nanos_u128().max(parent_start);
    let child_end = child.end_offset().as_nanos_u128().min(parent_end);
    (child_end > child_start).then_some((child_start, child_end))
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationTimings {
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

impl GenerationTimings {
    pub const fn new(prefill_seconds: f64, decode_seconds: f64) -> Self {
        Self {
            prefill_seconds,
            decode_seconds,
        }
    }

    pub fn from_durations(prefill: ProfileDuration, decode: ProfileDuration) -> Self {
        Self::new(prefill.as_seconds_f64(), decode.as_seconds_f64())
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleStats {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl SampleStats {
    pub fn from_finite_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return None;
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let count = sorted.len();
        let sum = sorted.iter().sum::<f64>();
        let mean = sum / count as f64;
        let median = median_sorted(&sorted);
        let min = sorted[0];
        let max = sorted[count - 1];

        Some(Self {
            count,
            mean,
            median,
            min,
            max,
        })
    }
}

pub fn rate_per_second(event_count: usize, seconds: f64) -> Option<f64> {
    if seconds.is_finite() && seconds > 0.0 {
        Some(event_count as f64 / seconds)
    } else {
        None
    }
}

pub fn tokens_per_second(token_count: usize, seconds: f64) -> Option<f64> {
    rate_per_second(token_count, seconds)
}

pub fn tokens_per_second_or_zero(token_count: usize, seconds: f64) -> f64 {
    tokens_per_second(token_count, seconds).unwrap_or(0.0)
}

pub fn median_f64(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return None;
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    Some(median_sorted(&sorted))
}

fn median_sorted(sorted: &[f64]) -> f64 {
    debug_assert!(!sorted.is_empty());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn timeline_intervals(total: ProfileDuration, segments: &[ProfileSegment]) -> Vec<(u128, u128)> {
    let total_nanos = total.as_nanos_u128();
    let mut intervals = segments
        .iter()
        .filter(|segment| segment.source != ProfileTimeSource::SelfTimeAccounting)
        .filter_map(|segment| {
            let start = segment.start_offset?.as_nanos_u128().min(total_nanos);
            let end = start
                .saturating_add(segment.duration.as_nanos_u128())
                .min(total_nanos);
            (end > start).then_some((start, end))
        })
        .collect::<Vec<_>>();
    intervals.sort_by_key(|&(start, end)| (start, end));
    merge_intervals(intervals)
}

fn merge_intervals(intervals: Vec<(u128, u128)>) -> Vec<(u128, u128)> {
    let mut merged: Vec<(u128, u128)> = Vec::new();
    for (start, end) in intervals {
        if let Some((_, last_end)) = merged.last_mut()
            && start <= *last_end
        {
            *last_end = (*last_end).max(end);
            continue;
        }
        merged.push((start, end));
    }
    merged
}

fn complement_intervals(total_nanos: u128, covered: &[(u128, u128)]) -> Vec<(u128, u128)> {
    let mut result = Vec::new();
    let mut cursor = 0;
    for &(start, end) in covered {
        if start > cursor {
            result.push((cursor, start));
        }
        cursor = cursor.max(end);
    }
    if cursor < total_nanos {
        result.push((cursor, total_nanos));
    }
    result
}

fn covered_nanos(intervals: &[(u128, u128)]) -> u128 {
    intervals
        .iter()
        .map(|(start, end)| end.saturating_sub(*start))
        .sum()
}

fn saturating_u128_to_u64(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}

fn nanos_to_seconds(nanos: u128) -> f64 {
    nanos as f64 / 1_000_000_000.0
}

fn push_profile_coverage_json(
    out: &mut String,
    name: &str,
    coverage: ProfileCoverage,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": {\n");
    push_json_field_f64(
        out,
        "total_seconds",
        coverage.total_seconds,
        indent + 2,
        true,
    );
    push_json_field_f64(
        out,
        "measured_covered_seconds",
        coverage.measured_covered_seconds,
        indent + 2,
        true,
    );
    push_json_field_f64(
        out,
        "self_time_accounting_seconds",
        coverage.self_time_accounting_seconds,
        indent + 2,
        true,
    );
    push_json_field_f64(
        out,
        "credited_work_seconds",
        coverage.credited_work_seconds,
        indent + 2,
        true,
    );
    push_json_field_f64(
        out,
        "coverage_ratio",
        coverage.coverage_ratio,
        indent + 2,
        false,
    );
    push_indent(out, indent);
    out.push('}');
    push_optional_comma(out, comma);
}

fn push_profile_segment_json(out: &mut String, segment: &ProfileSegment, indent: usize) {
    push_indent(out, indent);
    out.push_str("{\n");
    push_json_field_string(out, "name", &segment.name, indent + 2, true);
    push_json_field_string(out, "category", &segment.category, indent + 2, true);
    push_json_field_string(out, "source", segment.source.label(), indent + 2, true);
    push_json_field_optional_duration(out, "start_seconds", segment.start_offset, indent + 2, true);
    push_json_field_f64(
        out,
        "duration_seconds",
        segment.duration.as_seconds_f64(),
        indent + 2,
        true,
    );
    push_json_field_optional_duration(out, "end_seconds", segment.end_offset(), indent + 2, true);
    push_json_field_launch_spec(out, "launch", segment.launch.as_ref(), indent + 2, true);
    push_json_field_operation_spec(
        out,
        "operation",
        segment.operation.as_ref(),
        indent + 2,
        false,
    );
    push_indent(out, indent);
    out.push('}');
}

fn push_profile_span_json(out: &mut String, span: &ProfileSpan, indent: usize) {
    push_indent(out, indent);
    out.push_str("{\n");
    push_json_field_usize(out, "id", span.id, indent + 2, true);
    push_json_field_optional_usize(out, "parent_id", span.parent_id, indent + 2, true);
    push_json_field_string(out, "name", &span.name, indent + 2, true);
    push_json_field_string(out, "category", &span.category, indent + 2, true);
    push_json_field_string(out, "source", span.source.label(), indent + 2, true);
    push_json_field_f64(
        out,
        "start_seconds",
        span.start_offset.as_seconds_f64(),
        indent + 2,
        true,
    );
    push_json_field_f64(
        out,
        "duration_seconds",
        span.duration.as_seconds_f64(),
        indent + 2,
        true,
    );
    push_json_field_f64(
        out,
        "self_seconds",
        span.self_duration.as_seconds_f64(),
        indent + 2,
        true,
    );
    push_json_field_operation_spec(out, "operation", span.operation.as_ref(), indent + 2, false);
    push_indent(out, indent);
    out.push('}');
}

fn push_json_field_operation_spec(
    out: &mut String,
    name: &str,
    operation: Option<&TypedOperationSpec>,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": ");
    if let Some(operation) = operation {
        out.push_str("{\n");
        push_json_field_string(out, "name", &operation.name, indent + 2, true);
        push_json_field_string(out, "kind", operation.kind.label(), indent + 2, true);
        push_json_field_string(out, "route", operation.route.label(), indent + 2, true);
        push_json_field_tensor_specs(out, "inputs", &operation.inputs, indent + 2, true);
        push_json_field_tensor_specs(out, "outputs", &operation.outputs, indent + 2, true);
        push_json_field_launch_spec(out, "launch", operation.launch.as_ref(), indent + 2, false);
        push_indent(out, indent);
        out.push('}');
    } else {
        out.push_str("null");
    }
    push_optional_comma(out, comma);
}

fn push_json_field_tensor_specs(
    out: &mut String,
    name: &str,
    specs: &[TensorTypeSpec],
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": [");
    for (index, spec) in specs.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str("{\"dtype\": ");
        push_json_string(out, spec.dtype.label());
        out.push_str(", \"accumulator\": ");
        push_json_string(out, spec.accumulator.label());
        out.push_str(", \"shape\": ");
        push_usize_array(out, &spec.shape);
        out.push_str(", \"layout\": ");
        if let Some(layout) = &spec.layout {
            push_json_string(out, layout);
        } else {
            out.push_str("null");
        }
        out.push('}');
    }
    out.push(']');
    push_optional_comma(out, comma);
}

fn push_json_field_launch_spec(
    out: &mut String,
    name: &str,
    launch: Option<&CudaLaunchSpec>,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": ");
    if let Some(launch) = launch {
        out.push_str("{\n");
        push_json_field_string(out, "kernel", &launch.kernel, indent + 2, true);
        push_json_field_launch_dim(out, "grid_dim", launch.grid_dim, indent + 2, true);
        push_json_field_launch_dim(out, "block_dim", launch.block_dim, indent + 2, true);
        push_json_field_u32(
            out,
            "shared_mem_bytes",
            launch.shared_mem_bytes,
            indent + 2,
            false,
        );
        push_indent(out, indent);
        out.push('}');
    } else {
        out.push_str("null");
    }
    push_optional_comma(out, comma);
}

fn push_json_field_launch_dim(
    out: &mut String,
    name: &str,
    dim: LaunchDimensions,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(&format!(
        ": {{\"x\": {}, \"y\": {}, \"z\": {}}}",
        dim.x, dim.y, dim.z
    ));
    push_optional_comma(out, comma);
}

fn push_json_field_optional_duration(
    out: &mut String,
    name: &str,
    value: Option<ProfileDuration>,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": ");
    if let Some(value) = value {
        out.push_str(&format!("{:.12}", value.as_seconds_f64()));
    } else {
        out.push_str("null");
    }
    push_optional_comma(out, comma);
}

fn push_json_field_string(out: &mut String, name: &str, value: &str, indent: usize, comma: bool) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": ");
    push_json_string(out, value);
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

fn push_json_field_optional_usize(
    out: &mut String,
    name: &str,
    value: Option<usize>,
    indent: usize,
    comma: bool,
) {
    push_indent(out, indent);
    push_json_string(out, name);
    out.push_str(": ");
    if let Some(value) = value {
        out.push_str(&value.to_string());
    } else {
        out.push_str("null");
    }
    push_optional_comma(out, comma);
}

fn push_usize_array(out: &mut String, values: &[usize]) {
    out.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_rejects_invalid_elapsed_time() {
        assert_eq!(tokens_per_second(10, 0.0), None);
        assert_eq!(tokens_per_second(10, -1.0), None);
        assert_eq!(tokens_per_second(10, f64::NAN), None);
        assert_eq!(tokens_per_second(10, f64::INFINITY), None);
    }

    #[test]
    fn rate_counts_zero_events_over_positive_time() {
        assert_eq!(tokens_per_second(0, 2.0), Some(0.0));
    }

    #[test]
    fn generation_timings_report_decode_and_total_rates() {
        let timings = GenerationTimings::new(1.0, 2.0);
        assert_eq!(timings.total_seconds(), 3.0);
        assert_eq!(timings.decode_tokens_per_second(8), Some(4.0));
        assert_eq!(timings.total_tokens_per_second(9), Some(3.0));
    }

    #[test]
    fn median_accepts_unsorted_finite_samples() {
        assert_eq!(median_f64(&[5.0, 1.0, 3.0]), Some(3.0));
        assert_eq!(median_f64(&[10.0, 2.0, 4.0, 8.0]), Some(6.0));
    }

    #[test]
    fn sample_stats_reject_empty_and_non_finite_samples() {
        assert_eq!(SampleStats::from_finite_samples(&[]), None);
        assert_eq!(SampleStats::from_finite_samples(&[1.0, f64::NAN]), None);
    }

    #[test]
    fn sample_stats_compute_basic_summary() {
        let stats = SampleStats::from_finite_samples(&[4.0, 1.0, 7.0]).unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean, 4.0);
        assert_eq!(stats.median, 4.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 7.0);
    }

    #[test]
    fn typed_tensor_erases_precision_and_accumulator() {
        let tensor = TensorType::<Bf16ProfileType, 2>::new([4096, 5120])
            .with_static_layout("row-major")
            .erase();

        assert_eq!(tensor.dtype, NumericKind::Bf16);
        assert_eq!(tensor.accumulator, NumericKind::F32);
        assert_eq!(tensor.shape, vec![4096, 5120]);
        assert_eq!(tensor.layout.as_deref(), Some("row-major"));
    }

    #[test]
    fn operation_spec_carries_launch_metadata() {
        let input = TensorType::<F32ProfileType, 1>::new([1024]).erase();
        let output = TensorType::<F32ProfileType, 1>::new([1024]).erase();
        let launch = CudaLaunchSpec::new("relu_kernel", (4, 1, 1), (256, 1, 1), 0);
        let operation = TypedOperationSpec::new(
            "relu",
            OperationKind::Elementwise,
            OperationRoute::CudaKernel,
        )
        .with_input(input)
        .with_output(output)
        .with_launch(launch.clone());

        assert_eq!(operation.kind, OperationKind::Elementwise);
        assert_eq!(operation.route, OperationRoute::CudaKernel);
        assert_eq!(operation.launch, Some(launch));
    }

    #[test]
    fn optimization_candidate_carries_action_trace_and_timing_score() {
        let launch = CudaLaunchSpec::new("matvec_bf16_rows8", (16, 1, 1), (256, 1, 1), 0);
        let operation = TypedOperationSpec::new(
            "row-major-warp-rows8::bf16",
            OperationKind::Matvec,
            OperationRoute::CudaKernel,
        )
        .with_launch(launch.clone());
        let action = OptimizationActionSpec::split(
            0,
            8,
            OptimizationActionMaterialization::DeferredGenerated,
        );
        let samples = SampleStats::from_finite_samples(&[0.000004, 0.000003, 0.000005]).unwrap();
        let timing = OptimizationTiming::new(
            ProfileTimeSource::CudaEvent,
            1,
            samples,
            ProfileDuration::from_seconds_f64(samples.median).unwrap(),
        );
        let score = OptimizationScore::measured_with_timing(samples.median, timing);
        let candidate = OptimizationCandidateSpec::new(
            "matvec-bf16-row-major",
            "abc123",
            "row-major-matvec-generator",
            launch,
            operation,
        )
        .with_action_trace(vec![action])
        .with_score(score);

        assert_eq!(candidate.action_trace[0].op, OptimizationActionOp::Split);
        assert_eq!(
            candidate.score.unwrap().source,
            OptimizationScoreSource::Measured
        );
        assert_eq!(candidate.score.unwrap().timing.unwrap().samples.count, 3);
    }

    #[test]
    fn trace_credits_parent_self_time() {
        let parent = ProfileSpan {
            id: 1,
            parent_id: Some(0),
            name: "parent".to_string(),
            category: "model-phase".to_string(),
            source: ProfileTimeSource::WallClock,
            start_offset: ProfileDuration::from_nanos(0),
            duration: ProfileDuration::from_nanos(100),
            self_duration: ProfileDuration::ZERO,
            operation: None,
        };
        let child_a = ProfileSpan {
            id: 2,
            parent_id: Some(1),
            name: "child-a".to_string(),
            category: "matvec".to_string(),
            source: ProfileTimeSource::WallClock,
            start_offset: ProfileDuration::from_nanos(10),
            duration: ProfileDuration::from_nanos(40),
            self_duration: ProfileDuration::ZERO,
            operation: None,
        };
        let child_b = ProfileSpan {
            id: 3,
            parent_id: Some(1),
            name: "child-b".to_string(),
            category: "activation".to_string(),
            source: ProfileTimeSource::WallClock,
            start_offset: ProfileDuration::from_nanos(40),
            duration: ProfileDuration::from_nanos(30),
            self_duration: ProfileDuration::ZERO,
            operation: None,
        };

        let spans = with_self_durations(vec![parent, child_a, child_b]);
        let parent = spans.iter().find(|span| span.id == 1).unwrap();
        assert_eq!(parent.self_duration.as_nanos_u128(), 40);
    }
}
