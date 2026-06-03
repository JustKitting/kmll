use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KernelActionSpaceSet {
    pub spaces: Vec<KernelActionSpace>,
}

impl KernelActionSpaceSet {
    pub fn new(spaces: impl Into<Vec<KernelActionSpace>>) -> Self {
        Self {
            spaces: spaces.into(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.spaces.is_empty()
    }

    pub fn actions(&self) -> Vec<KernelScheduleAction> {
        self.spaces
            .iter()
            .flat_map(KernelActionSpace::actions)
            .collect()
    }

    pub fn optimization_spec(&self) -> ProfilingActionSpaceSet {
        ProfilingActionSpaceSet::new(
            self.spaces
                .iter()
                .map(KernelActionSpace::optimization_spec)
                .collect::<Vec<_>>(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelActionSpace {
    Split {
        variants: Vec<KernelAxisFactorAction>,
    },
    Upcast {
        axis: u8,
        factors: Vec<u32>,
    },
    Unroll {
        axis: u8,
        factors: Vec<u32>,
    },
    LocalTile {
        axis: u8,
        factors: Vec<u32>,
    },
    ThreadGroup {
        axis: u8,
        factors: Vec<u32>,
    },
    TileGemm {
        variants: Vec<KernelTile3dAction>,
    },
    StrideOrder {
        orders: Vec<Vec<u8>>,
    },
    Swap {
        pairs: Vec<(u8, u8)>,
    },
}

impl KernelActionSpace {
    pub fn actions(&self) -> Vec<KernelScheduleAction> {
        match self {
            Self::Split { variants } => variants
                .iter()
                .map(|variant| {
                    KernelScheduleAction::split(
                        variant.axis,
                        variant.factor,
                        variant.materialization,
                    )
                })
                .collect(),
            Self::Upcast { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::upcast(*axis, factor))
                .collect(),
            Self::Unroll { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::unroll(*axis, factor))
                .collect(),
            Self::LocalTile { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::local_tile(*axis, factor))
                .collect(),
            Self::ThreadGroup { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::thread_group(*axis, factor))
                .collect(),
            Self::TileGemm { variants } => variants
                .iter()
                .map(|variant| {
                    KernelScheduleAction::tile_gemm(
                        variant.tile.m,
                        variant.tile.n,
                        variant.tile.k,
                        variant.materialization,
                    )
                })
                .collect(),
            Self::StrideOrder { orders } => orders
                .iter()
                .cloned()
                .map(KernelScheduleAction::stride_order)
                .collect(),
            Self::Swap { pairs } => pairs
                .iter()
                .copied()
                .map(|(axis_a, axis_b)| KernelScheduleAction::swap(axis_a, axis_b))
                .collect(),
        }
    }

    fn optimization_spec(&self) -> ProfilingActionSpace {
        match self {
            Self::Split { variants } => ProfilingActionSpace::Split {
                variants: variants
                    .iter()
                    .map(|variant| {
                        ProfilingAxisFactorChoice::new(
                            variant.axis,
                            variant.factor,
                            variant.materialization,
                        )
                    })
                    .collect(),
            },
            Self::Upcast { axis, factors } => ProfilingActionSpace::Upcast {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::Unroll { axis, factors } => ProfilingActionSpace::Unroll {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::LocalTile { axis, factors } => ProfilingActionSpace::LocalTile {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::ThreadGroup { axis, factors } => ProfilingActionSpace::ThreadGroup {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::TileGemm { variants } => ProfilingActionSpace::TileGemm {
                variants: variants
                    .iter()
                    .map(|variant| {
                        ProfilingTile3dChoice::new(
                            variant.tile.m,
                            variant.tile.n,
                            variant.tile.k,
                            variant.materialization,
                        )
                    })
                    .collect(),
            },
            Self::StrideOrder { orders } => ProfilingActionSpace::StrideOrder {
                orders: orders.clone(),
            },
            Self::Swap { pairs } => ProfilingActionSpace::Swap {
                pairs: pairs.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelAxisFactorAction {
    pub axis: u8,
    pub factor: u32,
    pub materialization: KernelActionMaterialization,
}

impl KernelAxisFactorAction {
    pub const fn new(axis: u8, factor: u32, materialization: KernelActionMaterialization) -> Self {
        Self {
            axis,
            factor,
            materialization,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelTile3d {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl KernelTile3d {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelTile3dAction {
    pub tile: KernelTile3d,
    pub materialization: KernelActionMaterialization,
}

impl KernelTile3dAction {
    pub const fn new(tile: KernelTile3d, materialization: KernelActionMaterialization) -> Self {
        Self {
            tile,
            materialization,
        }
    }
}

pub trait KernelMetadataSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata;
    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata>;
    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore>;
}

pub trait KernelActionSearchProblem: KernelMetadataSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet;

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet;

    fn schedule_actions(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelScheduleAction> {
        self.action_spaces(candidate).actions()
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelExpansionPolicy {
    pub require_launchable: bool,
    pub max_threads_per_block: Option<u32>,
    pub max_shared_memory_bytes: Option<u32>,
    pub max_accumulator_elements_per_thread: Option<u32>,
    pub max_output_elements_per_thread: Option<u32>,
    pub max_load_elements_per_block: Option<u32>,
}

impl KernelExpansionPolicy {
    pub const DEFAULT_MAX_THREADS_PER_BLOCK: u32 = 1024;
    pub const DEFAULT_MAX_SHARED_MEMORY_BYTES: u32 = 48 * 1024;

    pub const fn new() -> Self {
        Self {
            require_launchable: false,
            max_threads_per_block: Some(Self::DEFAULT_MAX_THREADS_PER_BLOCK),
            max_shared_memory_bytes: Some(Self::DEFAULT_MAX_SHARED_MEMORY_BYTES),
            max_accumulator_elements_per_thread: None,
            max_output_elements_per_thread: None,
            max_load_elements_per_block: None,
        }
    }

    pub const fn for_search_config(require_launchable: bool) -> Self {
        Self {
            require_launchable,
            ..Self::new()
        }
    }

    pub const fn with_require_launchable(mut self, require_launchable: bool) -> Self {
        self.require_launchable = require_launchable;
        self
    }

    pub const fn with_max_threads_per_block(mut self, max: Option<u32>) -> Self {
        self.max_threads_per_block = max;
        self
    }

    pub const fn with_max_shared_memory_bytes(mut self, max: Option<u32>) -> Self {
        self.max_shared_memory_bytes = max;
        self
    }

    pub const fn with_max_accumulator_elements_per_thread(mut self, max: Option<u32>) -> Self {
        self.max_accumulator_elements_per_thread = max;
        self
    }

    pub const fn with_max_output_elements_per_thread(mut self, max: Option<u32>) -> Self {
        self.max_output_elements_per_thread = max;
        self
    }

    pub const fn with_max_load_elements_per_block(mut self, max: Option<u32>) -> Self {
        self.max_load_elements_per_block = max;
        self
    }

    pub fn allows(
        self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<(), KernelCandidateRejectReason> {
        if self.require_launchable && !candidate.is_launchable() {
            return Err(KernelCandidateRejectReason::DeferredGenerated);
        }
        let threads_per_block = launch_threads_per_block(&candidate.launch);
        if let Some(max) = self.max_threads_per_block
            && threads_per_block > max
        {
            return Err(KernelCandidateRejectReason::ThreadsPerBlock {
                actual: threads_per_block,
                max,
            });
        }
        let shared_memory_bytes = candidate
            .resources
            .map(|resources| resources.shared_memory_bytes)
            .unwrap_or(candidate.launch.shared_mem_bytes)
            .max(candidate.launch.shared_mem_bytes);
        if let Some(max) = self.max_shared_memory_bytes
            && shared_memory_bytes > max
        {
            return Err(KernelCandidateRejectReason::SharedMemoryBytes {
                actual: shared_memory_bytes,
                max,
            });
        }
        let Some(resources) = candidate.resources else {
            return Ok(());
        };
        if let Some(max) = self.max_accumulator_elements_per_thread
            && resources.accumulator_elements_per_thread > max
        {
            return Err(KernelCandidateRejectReason::AccumulatorElementsPerThread {
                actual: resources.accumulator_elements_per_thread,
                max,
            });
        }
        if let Some(max) = self.max_output_elements_per_thread
            && resources.output_elements_per_thread > max
        {
            return Err(KernelCandidateRejectReason::OutputElementsPerThread {
                actual: resources.output_elements_per_thread,
                max,
            });
        }
        if let Some(max) = self.max_load_elements_per_block
            && resources.load_elements_per_block > max
        {
            return Err(KernelCandidateRejectReason::LoadElementsPerBlock {
                actual: resources.load_elements_per_block,
                max,
            });
        }
        Ok(())
    }
}

impl Default for KernelExpansionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelCandidateRejectReason {
    DeferredGenerated,
    ThreadsPerBlock { actual: u32, max: u32 },
    SharedMemoryBytes { actual: u32, max: u32 },
    AccumulatorElementsPerThread { actual: u32, max: u32 },
    OutputElementsPerThread { actual: u32, max: u32 },
    LoadElementsPerBlock { actual: u32, max: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelCandidateExpansion {
    pub candidates: Vec<KernelCandidateMetadata>,
    pub accepted: usize,
    pub rejected: usize,
    pub duplicates: usize,
    pub last_reject_reason: Option<KernelCandidateRejectReason>,
}

impl KernelCandidateExpansion {
    pub fn explored(&self) -> usize {
        self.accepted + self.rejected
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelActionReplayError {
    InvalidAction {
        index: usize,
        action: KernelScheduleAction,
    },
    FamilyMismatch {
        expected: String,
        actual: String,
    },
    GeneratorMismatch {
        expected: String,
        actual: String,
    },
    ArtifactKeyMismatch {
        expected: String,
        actual: String,
    },
    LaunchabilityMismatch {
        expected: bool,
        actual: bool,
    },
}

impl fmt::Display for KernelActionReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAction { index, action } => {
                write!(f, "invalid schedule action at index {index}: {action:?}")
            }
            Self::FamilyMismatch { expected, actual } => {
                write!(
                    f,
                    "replayed candidate family {actual:?} did not match expected {expected:?}"
                )
            }
            Self::GeneratorMismatch { expected, actual } => {
                write!(
                    f,
                    "replayed candidate generator {actual:?} did not match expected {expected:?}"
                )
            }
            Self::ArtifactKeyMismatch { expected, actual } => {
                write!(
                    f,
                    "replayed candidate artifact key {actual} did not match expected {expected}"
                )
            }
            Self::LaunchabilityMismatch { expected, actual } => {
                write!(
                    f,
                    "replayed candidate launchable={actual} did not match expected launchable={expected}"
                )
            }
        }
    }
}

impl std::error::Error for KernelActionReplayError {}

pub fn expand_with_schedule_actions<P>(
    problem: &P,
    candidate: &KernelCandidateMetadata,
) -> Vec<KernelCandidateMetadata>
where
    P: KernelActionSearchProblem,
{
    problem
        .schedule_actions(candidate)
        .into_iter()
        .filter_map(|action| problem.apply_schedule_action(candidate, &action))
        .collect()
}

pub fn expand_metadata_candidates<P>(
    problem: &P,
    candidate: &KernelCandidateMetadata,
    policy: KernelExpansionPolicy,
    seen: &mut HashSet<KernelMetadataKey>,
) -> KernelCandidateExpansion
where
    P: KernelMetadataSearchProblem,
{
    let mut candidates = Vec::new();
    let mut rejected = 0;
    let mut duplicates = 0;
    let mut last_reject_reason = None;

    for next in problem.expand(candidate) {
        if !seen.insert(next.artifact_key()) {
            duplicates += 1;
            continue;
        }
        match policy.allows(&next) {
            Ok(()) => candidates.push(next),
            Err(reason) => {
                rejected += 1;
                last_reject_reason = Some(reason);
            }
        }
    }

    KernelCandidateExpansion {
        accepted: candidates.len(),
        candidates,
        rejected,
        duplicates,
        last_reject_reason,
    }
}

fn launch_threads_per_block(launch: &CudaLaunchSpec) -> u32 {
    let threads = u64::from(launch.block_dim.x)
        .saturating_mul(u64::from(launch.block_dim.y))
        .saturating_mul(u64::from(launch.block_dim.z));
    threads.min(u64::from(u32::MAX)) as u32
}

pub fn replay_schedule_actions<P>(
    problem: &P,
    actions: &[KernelScheduleAction],
) -> Result<KernelCandidateMetadata, KernelActionReplayError>
where
    P: KernelActionSearchProblem,
{
    let mut candidate = problem.seed();
    for (index, action) in actions.iter().enumerate() {
        candidate = problem
            .apply_schedule_action(&candidate, action)
            .ok_or_else(|| KernelActionReplayError::InvalidAction {
                index,
                action: action.clone(),
            })?;
    }
    Ok(candidate)
}

pub fn replay_optimization_candidate_spec<P>(
    problem: &P,
    spec: &OptimizationCandidateSpec,
) -> Result<KernelCandidateMetadata, KernelActionReplayError>
where
    P: KernelActionSearchProblem,
{
    let candidate = replay_schedule_actions(problem, &spec.action_trace)?;
    if candidate.family != spec.family {
        return Err(KernelActionReplayError::FamilyMismatch {
            expected: spec.family.clone(),
            actual: candidate.family,
        });
    }
    let actual_key = candidate.artifact_key().hex();
    if actual_key != spec.artifact_key {
        return Err(KernelActionReplayError::ArtifactKeyMismatch {
            expected: spec.artifact_key.clone(),
            actual: actual_key,
        });
    }
    Ok(candidate)
}
