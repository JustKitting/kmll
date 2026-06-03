use super::{
    super::{
        hashing::{implementation_key, metadata_key},
        *,
    },
    *,
};

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

pub(in crate::autotune) fn candidate_with_action_trace(
    parent: &KernelCandidateMetadata,
    action: &KernelScheduleAction,
    mut candidate: KernelCandidateMetadata,
) -> KernelCandidateMetadata {
    candidate.action_trace = parent.action_trace.clone();
    candidate.action_trace.push(action.clone());
    candidate
}
