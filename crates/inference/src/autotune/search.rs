#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeamSearchConfig {
    pub beam_width: usize,
    pub max_depth: usize,
    pub require_launchable: bool,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_depth: 2,
            require_launchable: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoOptimizeConfig {
    pub beam_width: usize,
    pub max_steps: usize,
    pub require_launchable: bool,
    pub min_score_improvement: f64,
}

impl AutoOptimizeConfig {
    pub const fn from_beam_search_config(config: BeamSearchConfig) -> Self {
        Self {
            beam_width: config.beam_width,
            max_steps: config.max_depth,
            require_launchable: config.require_launchable,
            min_score_improvement: 0.0,
        }
    }

    pub const fn as_beam_search_config(self) -> BeamSearchConfig {
        BeamSearchConfig {
            beam_width: self.beam_width,
            max_depth: self.max_steps,
            require_launchable: self.require_launchable,
        }
    }
}

impl Default for AutoOptimizeConfig {
    fn default() -> Self {
        Self::from_beam_search_config(BeamSearchConfig::default())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeamSearchResult {
    pub best: Option<KernelCandidateMetadata>,
    pub beam: Vec<KernelCandidateMetadata>,
    pub explored: usize,
    pub rejected: usize,
}

impl BeamSearchResult {
    pub fn optimization_report(
        &self,
        family: impl Into<String>,
        config: BeamSearchConfig,
    ) -> OptimizationSearchReport {
        OptimizationSearchReport::new(
            family,
            OptimizationSearchConfig::new(
                config.beam_width,
                config.max_depth,
                config.require_launchable,
            ),
            self.explored,
            self.rejected,
            self.best
                .as_ref()
                .map(KernelCandidateMetadata::optimization_spec),
            self.beam
                .iter()
                .map(KernelCandidateMetadata::optimization_spec)
                .collect(),
        )
    }

    pub fn optimization_report_with_action_space(
        &self,
        family: impl Into<String>,
        config: BeamSearchConfig,
        action_space: &KernelActionSpaceSet,
    ) -> OptimizationSearchReport {
        self.optimization_report(family, config)
            .with_action_space(action_space.optimization_spec())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutoOptimizeExitReason {
    CacheHit,
    MaxSteps,
    NoCandidates,
    NoImprovement { best_delta: f64 },
}

impl AutoOptimizeExitReason {
    pub const fn label(self) -> &'static str {
        match self {
            Self::CacheHit => "cache-hit",
            Self::MaxSteps => "max-steps",
            Self::NoCandidates => "no-candidates",
            Self::NoImprovement { .. } => "no-improvement",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AutoOptimizeStep {
    pub depth: usize,
    pub input_beam_len: usize,
    pub generated: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub best_before: Option<SearchScore>,
    pub best_after: Option<SearchScore>,
    pub best_candidate: Option<KernelCandidateMetadata>,
    pub improvement: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AutoOptimizeResult {
    pub best: Option<KernelCandidateMetadata>,
    pub beam: Vec<KernelCandidateMetadata>,
    pub explored: usize,
    pub rejected: usize,
    pub steps: Vec<AutoOptimizeStep>,
    pub exit_reason: AutoOptimizeExitReason,
}

impl AutoOptimizeResult {
    pub fn as_beam_search_result(&self) -> BeamSearchResult {
        BeamSearchResult {
            best: self.best.clone(),
            beam: self.beam.clone(),
            explored: self.explored,
            rejected: self.rejected,
        }
    }

    pub fn optimization_report(
        &self,
        family: impl Into<String>,
        config: AutoOptimizeConfig,
    ) -> OptimizationSearchReport {
        self.as_beam_search_result()
            .optimization_report(family, config.as_beam_search_config())
    }

    pub fn optimization_report_with_action_space(
        &self,
        family: impl Into<String>,
        config: AutoOptimizeConfig,
        action_space: &KernelActionSpaceSet,
    ) -> OptimizationSearchReport {
        self.as_beam_search_result()
            .optimization_report_with_action_space(
                family,
                config.as_beam_search_config(),
                action_space,
            )
    }

    pub fn auto_optimization_report(
        &self,
        family: impl Into<String>,
        config: AutoOptimizeConfig,
    ) -> AutoOptimizationSearchReport {
        AutoOptimizationSearchReport::new(
            family,
            AutoOptimizationSearchConfig::new(
                config.beam_width,
                config.max_steps,
                config.require_launchable,
                config.min_score_improvement,
            ),
            self.explored,
            self.rejected,
            profiling_auto_exit_reason(self.exit_reason),
            self.steps.iter().map(profiling_auto_search_step).collect(),
            self.best
                .as_ref()
                .map(KernelCandidateMetadata::optimization_spec),
            self.beam
                .iter()
                .map(KernelCandidateMetadata::optimization_spec)
                .collect(),
        )
    }

    pub fn auto_optimization_report_with_action_space(
        &self,
        family: impl Into<String>,
        config: AutoOptimizeConfig,
        action_space: &KernelActionSpaceSet,
    ) -> AutoOptimizationSearchReport {
        self.auto_optimization_report(family, config)
            .with_action_space(action_space.optimization_spec())
    }
}

fn profiling_auto_exit_reason(
    reason: AutoOptimizeExitReason,
) -> ProfilingAutoOptimizationExitReason {
    match reason {
        AutoOptimizeExitReason::CacheHit => ProfilingAutoOptimizationExitReason::CacheHit,
        AutoOptimizeExitReason::MaxSteps => ProfilingAutoOptimizationExitReason::MaxSteps,
        AutoOptimizeExitReason::NoCandidates => ProfilingAutoOptimizationExitReason::NoCandidates,
        AutoOptimizeExitReason::NoImprovement { best_delta } => {
            ProfilingAutoOptimizationExitReason::NoImprovement { best_delta }
        }
    }
}

fn profiling_auto_search_step(step: &AutoOptimizeStep) -> AutoOptimizationSearchStep {
    AutoOptimizationSearchStep {
        depth: step.depth,
        input_beam_len: step.input_beam_len,
        generated: step.generated,
        accepted: step.accepted,
        rejected: step.rejected,
        best_before: step.best_before,
        best_after: step.best_after,
        best_candidate: step
            .best_candidate
            .as_ref()
            .map(KernelCandidateMetadata::optimization_spec),
        improvement: step.improvement,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionCacheStatus {
    Hit,
    Miss,
    Stale { reason: String },
}

impl SelectionCacheStatus {
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::Stale { .. } => "stale",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedBeamSearchResult {
    pub result: BeamSearchResult,
    pub cache_key: KernelOptimizationCacheKey,
    pub cache_status: SelectionCacheStatus,
    pub cache_write: Option<EmittedKernelOptimizationSelection>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedAutoOptimizeResult {
    pub result: AutoOptimizeResult,
    pub cache_key: KernelOptimizationCacheKey,
    pub cache_status: SelectionCacheStatus,
    pub cache_write: Option<EmittedKernelOptimizationSelection>,
}

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

pub fn beam_search_metadata<P>(problem: &P, config: BeamSearchConfig) -> BeamSearchResult
where
    P: KernelMetadataSearchProblem,
{
    beam_search_metadata_with_scorer(problem, config, |candidate| problem.score(candidate))
}

pub fn beam_search_metadata_with_scorer<P, F>(
    problem: &P,
    config: BeamSearchConfig,
    mut score_candidate: F,
) -> BeamSearchResult
where
    P: KernelMetadataSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    assert!(config.beam_width > 0, "beam width must be nonzero");

    let mut seed = problem.seed();
    seed.score = score_candidate(&seed);
    let mut seen = HashSet::new();
    seen.insert(seed.artifact_key());
    let mut beam = vec![seed];
    let mut explored = 0;
    let mut rejected = 0;
    let expansion_policy = KernelExpansionPolicy::for_search_config(config.require_launchable);

    for _ in 0..config.max_depth {
        let mut candidates = Vec::new();
        for candidate in &beam {
            let expansion =
                expand_metadata_candidates(problem, candidate, expansion_policy, &mut seen);
            explored += expansion.explored();
            rejected += expansion.rejected;
            for mut next in expansion.candidates {
                match score_candidate(&next) {
                    Some(score) => {
                        next.score = Some(score);
                        candidates.push(next);
                    }
                    None => rejected += 1,
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        candidates.sort_by(compare_candidates);
        beam = candidates.into_iter().take(config.beam_width).collect();
    }

    let best = beam.first().cloned();
    BeamSearchResult {
        best,
        beam,
        explored,
        rejected,
    }
}

pub fn auto_optimize_metadata<P>(problem: &P, config: AutoOptimizeConfig) -> AutoOptimizeResult
where
    P: KernelMetadataSearchProblem,
{
    auto_optimize_metadata_with_scorer(problem, config, |candidate| problem.score(candidate))
}

pub fn auto_optimize_metadata_with_scorer<P, F>(
    problem: &P,
    config: AutoOptimizeConfig,
    mut score_candidate: F,
) -> AutoOptimizeResult
where
    P: KernelMetadataSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    assert!(config.beam_width > 0, "beam width must be nonzero");
    assert!(
        config.min_score_improvement.is_finite() && config.min_score_improvement >= 0.0,
        "minimum score improvement must be finite and nonnegative"
    );

    let mut seed = problem.seed();
    seed.score = score_candidate(&seed);
    let mut seen = HashSet::new();
    seen.insert(seed.artifact_key());
    let mut beam = vec![seed];
    let mut explored = 0;
    let mut rejected = 0;
    let mut steps = Vec::new();
    let mut exit_reason = AutoOptimizeExitReason::MaxSteps;
    let expansion_policy = KernelExpansionPolicy::for_search_config(config.require_launchable);

    for depth in 0..config.max_steps {
        let input_beam_len = beam.len();
        let best_before = beam.first().and_then(|candidate| candidate.score);
        let mut candidates = Vec::new();
        let mut generated = 0;
        let mut step_rejected = 0;

        for candidate in &beam {
            let expansion =
                expand_metadata_candidates(problem, candidate, expansion_policy, &mut seen);
            explored += expansion.explored();
            generated += expansion.explored();
            rejected += expansion.rejected;
            step_rejected += expansion.rejected;
            for mut next in expansion.candidates {
                match score_candidate(&next) {
                    Some(score) => {
                        next.score = Some(score);
                        candidates.push(next);
                    }
                    None => {
                        rejected += 1;
                        step_rejected += 1;
                    }
                }
            }
        }

        if candidates.is_empty() {
            steps.push(AutoOptimizeStep {
                depth,
                input_beam_len,
                generated,
                accepted: 0,
                rejected: step_rejected,
                best_before,
                best_after: None,
                best_candidate: None,
                improvement: None,
            });
            exit_reason = AutoOptimizeExitReason::NoCandidates;
            break;
        }

        candidates.sort_by(compare_candidates);
        let accepted = candidates.len().min(config.beam_width);
        let next_beam = candidates
            .into_iter()
            .take(config.beam_width)
            .collect::<Vec<_>>();
        let best_after = next_beam.first().and_then(|candidate| candidate.score);
        let best_candidate = next_beam.first().cloned();
        let improvement =
            best_before.and_then(|before| best_after.map(|after| before.value - after.value));
        let stop_for_no_improvement = improvement
            .map(|delta| delta <= config.min_score_improvement)
            .unwrap_or(false);

        steps.push(AutoOptimizeStep {
            depth,
            input_beam_len,
            generated,
            accepted,
            rejected: step_rejected,
            best_before,
            best_after,
            best_candidate: best_candidate.clone(),
            improvement,
        });

        if stop_for_no_improvement {
            if improvement.is_some_and(|delta| delta > 0.0)
                && let Some(best_next) = best_candidate
            {
                beam = vec![best_next];
            }
            exit_reason = AutoOptimizeExitReason::NoImprovement {
                best_delta: improvement.unwrap_or(0.0),
            };
            break;
        }

        beam = next_beam;
    }

    let best = beam.first().cloned();
    AutoOptimizeResult {
        best,
        beam,
        explored,
        rejected,
        steps,
        exit_reason,
    }
}

pub fn beam_search_metadata_with_selection_cache<P, F>(
    store: &KernelArtifactStore,
    problem: &P,
    config: BeamSearchConfig,
    score_namespace: &str,
    mut score_candidate: F,
) -> Result<CachedBeamSearchResult, KernelGenerationError>
where
    P: KernelActionSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    let cache_key = optimization_selection_cache_key(problem, config, score_namespace);
    let cache_status = match store.read_selection_cache(&cache_key)? {
        Some(selection) => match selection.replay(problem) {
            Ok(mut candidate) => {
                candidate.score = selection.score;
                return Ok(CachedBeamSearchResult {
                    result: BeamSearchResult {
                        best: Some(candidate.clone()),
                        beam: vec![candidate],
                        explored: 0,
                        rejected: 0,
                    },
                    cache_key,
                    cache_status: SelectionCacheStatus::Hit,
                    cache_write: None,
                });
            }
            Err(error) => SelectionCacheStatus::Stale {
                reason: error.to_string(),
            },
        },
        None => SelectionCacheStatus::Miss,
    };
    let result =
        beam_search_metadata_with_scorer(problem, config, |candidate| score_candidate(candidate));
    let cache_write = result
        .best
        .as_ref()
        .map(|candidate| store.emit_selection_cache_for_candidate(&cache_key, candidate))
        .transpose()?;
    Ok(CachedBeamSearchResult {
        result,
        cache_key,
        cache_status,
        cache_write,
    })
}

pub fn auto_optimize_metadata_with_selection_cache<P, F>(
    store: &KernelArtifactStore,
    problem: &P,
    config: AutoOptimizeConfig,
    score_namespace: &str,
    mut score_candidate: F,
) -> Result<CachedAutoOptimizeResult, KernelGenerationError>
where
    P: KernelActionSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    let cache_key = auto_optimization_selection_cache_key(problem, config, score_namespace);
    let cache_status = match store.read_selection_cache(&cache_key)? {
        Some(selection) => match selection.replay(problem) {
            Ok(mut candidate) => {
                candidate.score = selection.score;
                return Ok(CachedAutoOptimizeResult {
                    result: AutoOptimizeResult {
                        best: Some(candidate.clone()),
                        beam: vec![candidate],
                        explored: 0,
                        rejected: 0,
                        steps: Vec::new(),
                        exit_reason: AutoOptimizeExitReason::CacheHit,
                    },
                    cache_key,
                    cache_status: SelectionCacheStatus::Hit,
                    cache_write: None,
                });
            }
            Err(error) => SelectionCacheStatus::Stale {
                reason: error.to_string(),
            },
        },
        None => SelectionCacheStatus::Miss,
    };
    let result =
        auto_optimize_metadata_with_scorer(problem, config, |candidate| score_candidate(candidate));
    let cache_write = result
        .best
        .as_ref()
        .map(|candidate| store.emit_selection_cache_for_candidate(&cache_key, candidate))
        .transpose()?;
    Ok(CachedAutoOptimizeResult {
        result,
        cache_key,
        cache_status,
        cache_write,
    })
}

pub fn optimization_selection_cache_key<P>(
    problem: &P,
    config: BeamSearchConfig,
    score_namespace: &str,
) -> KernelOptimizationCacheKey
where
    P: KernelActionSearchProblem,
{
    let seed = problem.seed();
    let mut state = FNV_OFFSET;
    state = hash_str(state, "optimization-selection-cache");
    state = hash_str(state, score_namespace);
    state = hash_u64(state, config.beam_width as u64);
    state = hash_u64(state, config.max_depth as u64);
    state = hash_u64(state, u64::from(config.require_launchable));
    state = hash_optimization_candidate(state, &seed.optimization_spec());
    state = hash_action_space_set(state, &problem.search_space());
    KernelOptimizationCacheKey {
        family: seed.family,
        key: KernelMetadataKey(state),
    }
}

pub fn auto_optimization_selection_cache_key<P>(
    problem: &P,
    config: AutoOptimizeConfig,
    score_namespace: &str,
) -> KernelOptimizationCacheKey
where
    P: KernelActionSearchProblem,
{
    let seed = problem.seed();
    let mut state = FNV_OFFSET;
    state = hash_str(state, "auto-optimization-selection-cache");
    state = hash_str(state, score_namespace);
    state = hash_u64(state, config.beam_width as u64);
    state = hash_u64(state, config.max_steps as u64);
    state = hash_u64(state, u64::from(config.require_launchable));
    state = hash_u64(state, config.min_score_improvement.to_bits());
    state = hash_optimization_candidate(state, &seed.optimization_spec());
    state = hash_action_space_set(state, &problem.search_space());
    KernelOptimizationCacheKey {
        family: seed.family,
        key: KernelMetadataKey(state),
    }
}
