use super::{
    hashing::{implementation_key, metadata_key},
    *,
};

pub(super) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
pub(super) const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelScheduleActionTemplate {
    pub split_factors: &'static [u32],
    pub upcast_factors: &'static [u32],
    pub unroll_factors: &'static [u32],
    pub local_tile_factors: &'static [u32],
    pub group_top_factors: &'static [u32],
    pub group_factors: &'static [u32],
    pub thread_group_factors: &'static [u32],
}

impl KernelScheduleActionTemplate {
    pub const TINYGRAD_LIKE: Self = Self {
        split_factors: &[8, 13, 16, 24, 32],
        upcast_factors: &[2, 3, 4, 5, 7],
        unroll_factors: &[4, 7],
        local_tile_factors: &[2, 3, 4, 8, 13, 16, 29, 32],
        group_top_factors: &[13, 16, 28, 29, 32, 49, 64, 256],
        group_factors: &[4, 8, 16],
        thread_group_factors: &[2, 3, 4, 5, 8, 12, 16, 24, 32, 64],
    };

    pub const INFERENCE_DEFAULT: Self = Self {
        split_factors: &[8, 13, 16, 24, 32],
        upcast_factors: &[2, 3, 4, 5, 7],
        unroll_factors: &[4, 7],
        local_tile_factors: &[2, 3, 4, 8, 13, 16, 24, 29, 32],
        group_top_factors: &[13, 16, 28, 29, 32, 49, 64, 256],
        group_factors: &[4, 8, 16],
        thread_group_factors: &[2, 3, 4, 5, 8, 12, 16, 24, 32, 64, 128, 256],
    };

    pub fn bounded_split_factors(
        self,
        extent: usize,
        max_factor: u32,
        required_factor: Option<u32>,
    ) -> Vec<u32> {
        let upper = extent.min(max_factor as usize) as u32;
        let mut factors = self
            .split_factors
            .iter()
            .copied()
            .filter(|factor| *factor <= upper)
            .collect::<Vec<_>>();
        if upper > 0 {
            factors.push(upper);
        }
        if let Some(required) = required_factor {
            factors.push(required);
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub fn exhaustive_unroll_factors(
        self,
        extent: usize,
        max_factor: u32,
        excluded_factor: Option<u32>,
    ) -> Vec<u32> {
        let upper = extent.min(max_factor as usize) as u32;
        (1..=upper)
            .filter(|factor| Some(*factor) != excluded_factor)
            .collect()
    }

    pub fn legal_upcast_factors(self, mut accepts: impl FnMut(u32) -> bool) -> Vec<u32> {
        self.upcast_factors
            .iter()
            .copied()
            .filter(|factor| accepts(*factor))
            .collect()
    }

    pub fn legal_thread_group_factors(self, mut accepts: impl FnMut(u32) -> bool) -> Vec<u32> {
        self.thread_group_factors
            .iter()
            .copied()
            .filter(|factor| accepts(*factor))
            .collect()
    }
}

pub(super) fn bounded_unroll_factors(
    extent: usize,
    max_factor: u32,
    excluded_factor: Option<u32>,
) -> Vec<u32> {
    KernelScheduleActionTemplate::INFERENCE_DEFAULT.exhaustive_unroll_factors(
        extent,
        max_factor,
        excluded_factor,
    )
}

pub(super) fn bounded_tile_factors(
    extent: usize,
    max_factor: u32,
    required_factor: Option<u32>,
) -> Vec<u32> {
    KernelScheduleActionTemplate::INFERENCE_DEFAULT.bounded_split_factors(
        extent,
        max_factor,
        required_factor,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelAxisKind {
    Spatial,
    Reduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelAxis {
    pub id: u8,
    pub name: &'static str,
    pub extent: usize,
    pub kind: KernelAxisKind,
    pub stride: Option<usize>,
}

impl KernelAxis {
    pub const fn spatial(id: u8, name: &'static str, extent: usize, stride: Option<usize>) -> Self {
        Self {
            id,
            name,
            extent,
            kind: KernelAxisKind::Spatial,
            stride,
        }
    }

    pub const fn reduction(
        id: u8,
        name: &'static str,
        extent: usize,
        stride: Option<usize>,
    ) -> Self {
        Self {
            id,
            name,
            extent,
            kind: KernelAxisKind::Reduction,
            stride,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScheduleTransform {
    Split { axis: u8, factor: u32 },
    Upcast { axis: u8, factor: u32 },
    Unroll { axis: u8, factor: u32 },
    LocalTile { axis: u8, factor: u32 },
    GroupTop { axis: u8, factor: u32 },
    Group { axis: u8, factor: u32 },
    ThreadGroup { axis: u8, factor: u32 },
    TileGemm { m: u32, n: u32, k: u32 },
    StrideOrder { axes: Vec<u8> },
    Swap { axis_a: u8, axis_b: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct KernelSchedule {
    pub transforms: Vec<ScheduleTransform>,
}

impl KernelSchedule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_transform(mut self, transform: ScheduleTransform) -> Self {
        self.transforms.push(transform);
        self
    }

    pub fn depth(&self) -> usize {
        self.transforms.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelMetadataKey(pub(super) u64);

impl KernelMetadataKey {
    pub const fn raw(self) -> u64 {
        self.0
    }

    pub fn hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelImplementationKey(pub(super) u64);

impl KernelImplementationKey {
    pub const fn raw(self) -> u64 {
        self.0
    }

    pub fn hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelMaterialization {
    Existing { symbol: &'static str },
    Generated { symbol: String },
    DeferredGenerated { symbol_hint: String, reason: String },
}

impl KernelMaterialization {
    pub const fn is_launchable(&self) -> bool {
        matches!(self, Self::Existing { .. })
    }

    pub const fn is_materializable(&self) -> bool {
        matches!(self, Self::Existing { .. } | Self::Generated { .. })
    }

    pub const fn is_existing(&self) -> bool {
        matches!(self, Self::Existing { .. })
    }

    pub const fn requires_generated_module(&self) -> bool {
        matches!(
            self,
            Self::Generated { .. } | Self::DeferredGenerated { .. }
        )
    }

    pub const fn profiling_materialization(&self) -> ProfilingCandidateMaterialization {
        match self {
            Self::Existing { .. } => ProfilingCandidateMaterialization::Existing,
            Self::Generated { .. } => ProfilingCandidateMaterialization::Generated,
            Self::DeferredGenerated { .. } => ProfilingCandidateMaterialization::DeferredGenerated,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedKernelMetadata {
    pub generator: &'static str,
    pub artifact_key: KernelMetadataKey,
    pub materialization: KernelMaterialization,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelCandidateMetadata {
    pub family: String,
    pub axes: Vec<KernelAxis>,
    pub schedule: KernelSchedule,
    pub action_trace: Vec<KernelScheduleAction>,
    pub generated: GeneratedKernelMetadata,
    pub launch: CudaLaunchSpec,
    pub operation: TypedOperationSpec,
    pub resources: Option<KernelResourceUsage>,
    pub score: Option<SearchScore>,
}

impl KernelCandidateMetadata {
    pub fn is_launchable(&self) -> bool {
        self.generated.materialization.is_launchable()
    }

    pub fn artifact_key(&self) -> KernelMetadataKey {
        self.generated.artifact_key
    }

    pub fn implementation_key(&self) -> KernelImplementationKey {
        implementation_key(&self.family, &self.schedule, &self.launch, &self.generated)
    }

    pub fn optimization_spec(&self) -> OptimizationCandidateSpec {
        OptimizationCandidateSpec::new(
            self.family.clone(),
            self.artifact_key().hex(),
            self.generated.generator,
            self.launch.clone(),
            self.operation.clone(),
        )
        .with_launchable(self.is_launchable())
        .with_materialization(self.generated.materialization.profiling_materialization())
        .with_action_trace(self.action_trace.clone())
        .with_resource_usage(self.resources)
        .with_score(self.score)
    }

    pub fn new(
        family: impl Into<String>,
        axes: Vec<KernelAxis>,
        schedule: KernelSchedule,
        generator: &'static str,
        materialization: KernelMaterialization,
        launch: CudaLaunchSpec,
        operation: TypedOperationSpec,
    ) -> Self {
        let family = family.into();
        let artifact_key = metadata_key(&family, &axes, &schedule, &launch);
        Self {
            family,
            axes,
            schedule,
            action_trace: Vec::new(),
            generated: GeneratedKernelMetadata {
                generator,
                artifact_key,
                materialization,
            },
            launch,
            operation,
            resources: None,
            score: None,
        }
    }
}

pub(super) fn candidate_with_action_trace(
    parent: &KernelCandidateMetadata,
    action: &KernelScheduleAction,
    mut candidate: KernelCandidateMetadata,
) -> KernelCandidateMetadata {
    candidate.action_trace = parent.action_trace.clone();
    candidate.action_trace.push(action.clone());
    candidate
}

#[derive(Debug)]
pub enum KernelGenerationError {
    UnsupportedCandidate {
        family: String,
        generator: &'static str,
    },
    UnsupportedOperation {
        name: String,
        kind: OperationKind,
        reason: String,
    },
    NoOptimizationCandidate {
        name: String,
        kind: OperationKind,
    },
    MissingTransform {
        family: String,
        transform: &'static str,
    },
    InvalidSelection {
        reason: String,
    },
    Io(io::Error),
    Json(serde_json::Error),
}

impl fmt::Display for KernelGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCandidate { family, generator } => {
                write!(
                    f,
                    "candidate family {family:?} is not supported by generator {generator}"
                )
            }
            Self::UnsupportedOperation { name, kind, reason } => {
                write!(
                    f,
                    "operation {name:?} ({}) is not supported by inference autotune: {reason}",
                    kind.label()
                )
            }
            Self::NoOptimizationCandidate { name, kind } => {
                write!(
                    f,
                    "operation {name:?} ({}) did not produce an inference autotune candidate",
                    kind.label()
                )
            }
            Self::MissingTransform { family, transform } => {
                write!(
                    f,
                    "candidate family {family:?} is missing required {transform} transform"
                )
            }
            Self::InvalidSelection { reason } => {
                write!(
                    f,
                    "kernel optimization selection metadata is invalid: {reason}"
                )
            }
            Self::Io(error) => write!(f, "kernel artifact I/O failed: {error}"),
            Self::Json(error) => write!(f, "kernel artifact manifest JSON failed: {error}"),
        }
    }
}

impl std::error::Error for KernelGenerationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for KernelGenerationError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for KernelGenerationError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedKernelSource {
    pub symbol: String,
    pub source: String,
}

pub trait KernelSourceGenerator {
    fn name(&self) -> &'static str;
    fn source_for(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<GeneratedKernelSource, KernelGenerationError>;
}
