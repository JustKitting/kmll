use super::{
    generator::InferenceKernelRustCudaGenerator, inference::InferenceKernelOptimizationProblem, *,
};

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceKernelAutoOptimize {
    pub operation: TypedOperationSpec,
    pub problem: InferenceKernelOptimizationProblem,
    pub result: AutoOptimizeResult,
}

impl InferenceKernelAutoOptimize {
    pub fn best_candidate(&self) -> Option<&KernelCandidateMetadata> {
        self.result.best.as_ref()
    }

    pub fn auto_optimization_report(
        &self,
        config: AutoOptimizeConfig,
    ) -> AutoOptimizationSearchReport {
        self.result.auto_optimization_report_with_action_space(
            self.problem.family(),
            config,
            &self.problem.search_space(),
        )
    }

    pub fn render_best_source(&self) -> Result<GeneratedKernelSource, KernelGenerationError> {
        let candidate = self.best_candidate().ok_or_else(|| {
            KernelGenerationError::NoOptimizationCandidate {
                name: self.operation.name.clone(),
                kind: self.operation.kind,
            }
        })?;
        InferenceKernelRustCudaGenerator.source_for(candidate)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedInferenceKernelAutoOptimize {
    pub optimization: InferenceKernelAutoOptimize,
    pub cache_key: KernelOptimizationCacheKey,
    pub cache_status: SelectionCacheStatus,
    pub cache_write: Option<EmittedKernelOptimizationSelection>,
}

impl CachedInferenceKernelAutoOptimize {
    pub fn result(&self) -> &AutoOptimizeResult {
        &self.optimization.result
    }

    pub fn best_candidate(&self) -> Option<&KernelCandidateMetadata> {
        self.optimization.best_candidate()
    }

    pub fn auto_optimization_report(
        &self,
        config: AutoOptimizeConfig,
    ) -> AutoOptimizationSearchReport {
        self.optimization.auto_optimization_report(config)
    }

    pub fn render_best_source(&self) -> Result<GeneratedKernelSource, KernelGenerationError> {
        self.optimization.render_best_source()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedInferenceKernelSource {
    pub optimization: InferenceKernelAutoOptimize,
    pub candidate: KernelCandidateMetadata,
    pub source: GeneratedKernelSource,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedGeneratedInferenceKernelSource {
    pub optimization: CachedInferenceKernelAutoOptimize,
    pub candidate: KernelCandidateMetadata,
    pub source: GeneratedKernelSource,
}

pub fn auto_optimize_inference_kernel(
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
) -> Result<InferenceKernelAutoOptimize, KernelGenerationError> {
    auto_optimize_inference_kernel_with_policy(
        operation,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
    )
}

pub fn auto_optimize_inference_kernel_with_policy(
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
) -> Result<InferenceKernelAutoOptimize, KernelGenerationError> {
    let problem = InferenceKernelOptimizationProblem::from_operation(operation)?;
    let result = auto_optimize_metadata_with_policy(&problem, config, policy);
    Ok(InferenceKernelAutoOptimize {
        operation: operation.clone(),
        problem,
        result,
    })
}

pub fn auto_optimize_inference_kernel_with_scorer<F>(
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    mut score_candidate: F,
) -> Result<InferenceKernelAutoOptimize, KernelGenerationError>
where
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    auto_optimize_inference_kernel_with_policy_scorer(
        operation,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
        |candidate, _problem| score_candidate(candidate),
    )
}

pub fn auto_optimize_inference_kernel_with_policy_scorer<F>(
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
    mut score_candidate: F,
) -> Result<InferenceKernelAutoOptimize, KernelGenerationError>
where
    F: FnMut(&KernelCandidateMetadata, &InferenceKernelOptimizationProblem) -> Option<SearchScore>,
{
    let problem = InferenceKernelOptimizationProblem::from_operation(operation)?;
    let result = auto_optimize_metadata_with_policy_scorer(&problem, config, policy, |candidate| {
        score_candidate(candidate, &problem)
    });
    Ok(InferenceKernelAutoOptimize {
        operation: operation.clone(),
        problem,
        result,
    })
}

pub fn auto_optimize_inference_kernel_with_selection_cache(
    store: &KernelArtifactStore,
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    score_namespace: &str,
) -> Result<CachedInferenceKernelAutoOptimize, KernelGenerationError> {
    auto_optimize_inference_kernel_with_selection_cache_and_policy(
        store,
        operation,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
        score_namespace,
    )
}

pub fn auto_optimize_inference_kernel_with_selection_cache_and_policy(
    store: &KernelArtifactStore,
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
    score_namespace: &str,
) -> Result<CachedInferenceKernelAutoOptimize, KernelGenerationError> {
    auto_optimize_inference_kernel_with_selection_cache_and_policy_scorer(
        store,
        operation,
        config,
        policy,
        score_namespace,
        |candidate, problem| problem.score(candidate),
    )
}

pub fn auto_optimize_inference_kernel_with_selection_cache_scorer<F>(
    store: &KernelArtifactStore,
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    score_namespace: &str,
    score_candidate: F,
) -> Result<CachedInferenceKernelAutoOptimize, KernelGenerationError>
where
    F: FnMut(&KernelCandidateMetadata, &InferenceKernelOptimizationProblem) -> Option<SearchScore>,
{
    auto_optimize_inference_kernel_with_selection_cache_and_policy_scorer(
        store,
        operation,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
        score_namespace,
        score_candidate,
    )
}

pub fn auto_optimize_inference_kernel_with_selection_cache_and_policy_scorer<F>(
    store: &KernelArtifactStore,
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
    score_namespace: &str,
    mut score_candidate: F,
) -> Result<CachedInferenceKernelAutoOptimize, KernelGenerationError>
where
    F: FnMut(&KernelCandidateMetadata, &InferenceKernelOptimizationProblem) -> Option<SearchScore>,
{
    let problem = InferenceKernelOptimizationProblem::from_operation(operation)?;
    let cached = auto_optimize_metadata_with_selection_cache_and_policy(
        store,
        &problem,
        config,
        policy,
        score_namespace,
        |candidate| score_candidate(candidate, &problem),
    )?;
    Ok(CachedInferenceKernelAutoOptimize {
        optimization: InferenceKernelAutoOptimize {
            operation: operation.clone(),
            problem,
            result: cached.result,
        },
        cache_key: cached.cache_key,
        cache_status: cached.cache_status,
        cache_write: cached.cache_write,
    })
}

pub fn generate_inference_kernel_source(
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
) -> Result<GeneratedInferenceKernelSource, KernelGenerationError> {
    generate_inference_kernel_source_with_policy(
        operation,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
    )
}

pub fn generate_inference_kernel_source_with_policy(
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
) -> Result<GeneratedInferenceKernelSource, KernelGenerationError> {
    let optimization = auto_optimize_inference_kernel_with_policy(operation, config, policy)?;
    let candidate = optimization.best_candidate().cloned().ok_or_else(|| {
        KernelGenerationError::NoOptimizationCandidate {
            name: operation.name.clone(),
            kind: operation.kind,
        }
    })?;
    let source = InferenceKernelRustCudaGenerator.source_for(&candidate)?;
    Ok(GeneratedInferenceKernelSource {
        optimization,
        candidate,
        source,
    })
}

pub fn generate_inference_kernel_source_with_selection_cache(
    store: &KernelArtifactStore,
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    score_namespace: &str,
) -> Result<CachedGeneratedInferenceKernelSource, KernelGenerationError> {
    generate_inference_kernel_source_with_selection_cache_and_policy(
        store,
        operation,
        config,
        KernelExpansionPolicy::for_search_config(config.require_launchable),
        score_namespace,
    )
}

pub fn generate_inference_kernel_source_with_selection_cache_and_policy(
    store: &KernelArtifactStore,
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    policy: KernelExpansionPolicy,
    score_namespace: &str,
) -> Result<CachedGeneratedInferenceKernelSource, KernelGenerationError> {
    let optimization = auto_optimize_inference_kernel_with_selection_cache_and_policy(
        store,
        operation,
        config,
        policy,
        score_namespace,
    )?;
    let candidate = optimization.best_candidate().cloned().ok_or_else(|| {
        KernelGenerationError::NoOptimizationCandidate {
            name: operation.name.clone(),
            kind: operation.kind,
        }
    })?;
    let source = InferenceKernelRustCudaGenerator.source_for(&candidate)?;
    Ok(CachedGeneratedInferenceKernelSource {
        optimization,
        candidate,
        source,
    })
}
