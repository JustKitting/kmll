use super::*;

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
