use super::super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArtifactPaths {
    pub directory: PathBuf,
    pub manifest_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandaloneKernelCratePaths {
    pub crate_dir: PathBuf,
    pub cargo_toml_path: PathBuf,
    pub source_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedKernelMetadata {
    pub artifact_key: KernelMetadataKey,
    pub paths: KernelArtifactPaths,
    pub manifest_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedSearchReport {
    pub report_key: KernelMetadataKey,
    pub report_path: PathBuf,
    pub report_bytes: usize,
    pub visual_svg_path: Option<PathBuf>,
    pub visual_svg_bytes: Option<usize>,
    pub visual_html_path: Option<PathBuf>,
    pub visual_html_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelOptimizationCacheKey {
    pub family: String,
    pub key: KernelMetadataKey,
}

impl KernelOptimizationCacheKey {
    pub fn hex(&self) -> String {
        self.key.hex()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelMaterializationDescriptor {
    Existing { symbol: String },
    Generated { symbol: String },
    DeferredGenerated { symbol_hint: String, reason: String },
}

impl KernelMaterializationDescriptor {
    pub fn from_materialization(materialization: &KernelMaterialization) -> Self {
        match materialization {
            KernelMaterialization::Existing { symbol } => Self::Existing {
                symbol: (*symbol).to_string(),
            },
            KernelMaterialization::Generated { symbol } => Self::Generated {
                symbol: symbol.clone(),
            },
            KernelMaterialization::DeferredGenerated {
                symbol_hint,
                reason,
            } => Self::DeferredGenerated {
                symbol_hint: symbol_hint.clone(),
                reason: reason.clone(),
            },
        }
    }

    pub fn matches_materialization(&self, materialization: &KernelMaterialization) -> bool {
        match (self, materialization) {
            (Self::Existing { symbol: expected }, KernelMaterialization::Existing { symbol }) => {
                expected == symbol
            }
            (Self::Generated { symbol: expected }, KernelMaterialization::Generated { symbol }) => {
                expected == symbol
            }
            (
                Self::DeferredGenerated {
                    symbol_hint: expected_hint,
                    reason: expected_reason,
                },
                KernelMaterialization::DeferredGenerated {
                    symbol_hint,
                    reason,
                },
            ) => expected_hint == symbol_hint && expected_reason == reason,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelOptimizationSelection {
    pub family: String,
    pub artifact_key: String,
    pub generator: String,
    pub launchable: bool,
    pub materialization: Option<KernelMaterializationDescriptor>,
    pub action_trace: Vec<KernelScheduleAction>,
    pub score: Option<SearchScore>,
}

impl KernelOptimizationSelection {
    pub fn from_candidate(candidate: &KernelCandidateMetadata) -> Self {
        Self {
            family: candidate.family.clone(),
            artifact_key: candidate.artifact_key().hex(),
            generator: candidate.generated.generator.to_string(),
            launchable: candidate.is_launchable(),
            materialization: Some(KernelMaterializationDescriptor::from_materialization(
                &candidate.generated.materialization,
            )),
            action_trace: candidate.action_trace.clone(),
            score: candidate.score,
        }
    }

    pub fn replay<P>(&self, problem: &P) -> Result<KernelCandidateMetadata, KernelActionReplayError>
    where
        P: KernelActionSearchProblem,
    {
        let candidate = replay_schedule_actions(problem, &self.action_trace)?;
        if candidate.family != self.family {
            return Err(KernelActionReplayError::FamilyMismatch {
                expected: self.family.clone(),
                actual: candidate.family,
            });
        }
        let actual_key = candidate.artifact_key().hex();
        if actual_key != self.artifact_key {
            return Err(KernelActionReplayError::ArtifactKeyMismatch {
                expected: self.artifact_key.clone(),
                actual: actual_key,
            });
        }
        if candidate.generated.generator != self.generator {
            return Err(KernelActionReplayError::GeneratorMismatch {
                expected: self.generator.clone(),
                actual: candidate.generated.generator.to_string(),
            });
        }
        let actual_launchable = candidate.is_launchable();
        if actual_launchable != self.launchable {
            return Err(KernelActionReplayError::LaunchabilityMismatch {
                expected: self.launchable,
                actual: actual_launchable,
            });
        }
        if let Some(materialization) = &self.materialization
            && !materialization.matches_materialization(&candidate.generated.materialization)
        {
            return Err(KernelActionReplayError::MaterializationMismatch {
                expected: materialization.clone(),
                actual: KernelMaterializationDescriptor::from_materialization(
                    &candidate.generated.materialization,
                ),
            });
        }
        Ok(candidate)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelOptimizationScoreRecord {
    pub score_namespace: String,
    pub family: String,
    pub artifact_key: String,
    pub generator: String,
    pub launchable: bool,
    pub materialization: Option<KernelMaterializationDescriptor>,
    pub action_trace: Vec<KernelScheduleAction>,
    pub score: SearchScore,
}

impl KernelOptimizationScoreRecord {
    pub fn from_candidate(
        score_namespace: &str,
        candidate: &KernelCandidateMetadata,
    ) -> Option<Self> {
        Some(Self {
            score_namespace: score_namespace.to_string(),
            family: candidate.family.clone(),
            artifact_key: candidate.artifact_key().hex(),
            generator: candidate.generated.generator.to_string(),
            launchable: candidate.is_launchable(),
            materialization: Some(KernelMaterializationDescriptor::from_materialization(
                &candidate.generated.materialization,
            )),
            action_trace: candidate.action_trace.clone(),
            score: candidate.score?,
        })
    }

    pub fn matches_candidate(
        &self,
        score_namespace: &str,
        candidate: &KernelCandidateMetadata,
    ) -> bool {
        self.score_namespace == score_namespace
            && self.family == candidate.family
            && self.artifact_key == candidate.artifact_key().hex()
            && self.generator == candidate.generated.generator
            && self.launchable == candidate.is_launchable()
            && self
                .materialization
                .as_ref()
                .map(|materialization| {
                    materialization.matches_materialization(&candidate.generated.materialization)
                })
                .unwrap_or(true)
            && self.action_trace == candidate.action_trace
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedKernelOptimizationSelection {
    pub artifact_key: String,
    pub selection_path: PathBuf,
    pub selection_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedKernelOptimizationScore {
    pub artifact_key: String,
    pub score_path: PathBuf,
    pub score_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedStandaloneKernelCrate {
    pub artifact_key: KernelMetadataKey,
    pub package_name: String,
    pub symbol: String,
    pub paths: StandaloneKernelCratePaths,
    pub cargo_toml_bytes: usize,
    pub source_bytes: usize,
}
