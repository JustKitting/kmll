use super::*;

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
    seen: &mut HashSet<KernelImplementationKey>,
) -> KernelCandidateExpansion
where
    P: KernelMetadataSearchProblem,
{
    let mut candidates = Vec::new();
    let mut rejected = 0;
    let mut duplicates = 0;
    let mut last_reject_reason = None;

    for next in problem.expand(candidate) {
        if !seen.insert(next.implementation_key()) {
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
