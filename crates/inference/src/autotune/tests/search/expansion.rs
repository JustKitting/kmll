use super::*;

#[test]
fn metadata_expansion_filters_launchability_and_tracks_duplicates() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let mut seen = HashSet::new();
    seen.insert(seed.implementation_key());

    let expansion = expand_metadata_candidates(
        &problem,
        &seed,
        KernelExpansionPolicy::for_search_config(true),
        &mut seen,
    );

    assert_eq!(expansion.accepted, 4);
    assert_eq!(expansion.rejected, 32);
    assert_eq!(expansion.explored(), 36);
    assert_eq!(expansion.duplicates, 0);
    assert_eq!(
        expansion.last_reject_reason,
        Some(KernelCandidateRejectReason::DeferredGenerated)
    );
    assert!(
        expansion
            .candidates
            .iter()
            .all(|candidate| candidate.is_launchable())
    );

    let duplicate_expansion = expand_metadata_candidates(
        &problem,
        &seed,
        KernelExpansionPolicy::for_search_config(true),
        &mut seen,
    );
    assert_eq!(duplicate_expansion.accepted, 0);
    assert_eq!(duplicate_expansion.rejected, 0);
    assert_eq!(duplicate_expansion.explored(), 0);
    assert_eq!(duplicate_expansion.duplicates, 36);
}

#[test]
fn implementation_key_ignores_problem_extent_and_launch_grid_for_generated_source() {
    let first = MatvecSearchProblem::bf16_row_major(128, 256)
        .generated_candidate_for_plan(MatvecSchedulePlan::new(RowMajorWarpRows::Rows8));
    let second = MatvecSearchProblem::bf16_row_major(256, 256)
        .generated_candidate_for_plan(MatvecSchedulePlan::new(RowMajorWarpRows::Rows8));

    assert_ne!(first.artifact_key(), second.artifact_key());
    assert_eq!(first.implementation_key(), second.implementation_key());

    let existing =
        MatvecSearchProblem::bf16_row_major(128, 256).candidate_for_rows(RowMajorWarpRows::Rows8);
    assert_ne!(first.implementation_key(), existing.implementation_key());
}

#[derive(Debug, Clone, Copy)]
struct ImplementationDuplicateSearchProblem;

impl ImplementationDuplicateSearchProblem {
    fn candidate(axis_extent: usize, grid_x: u32) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new("impl_duplicate_kernel", (grid_x, 1, 1), (32, 1, 1), 0);
        let operation = TypedOperationSpec::new(
            "implementation-duplicate",
            OperationKind::Matvec,
            OperationRoute::CudaKernel,
        )
        .with_launch(launch.clone());
        KernelCandidateMetadata::new(
            "implementation-duplicate",
            vec![KernelAxis::spatial(0, "row", axis_extent, Some(1))],
            KernelSchedule::new().with_transform(ScheduleTransform::Split { axis: 0, factor: 8 }),
            "implementation-duplicate-generator",
            KernelMaterialization::DeferredGenerated {
                symbol_hint: "impl_duplicate_kernel".to_string(),
                reason: "test candidate only carries generated implementation metadata".to_string(),
            },
            launch,
            operation,
        )
    }
}

impl KernelMetadataSearchProblem for ImplementationDuplicateSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new("impl_duplicate_seed", (1, 1, 1), (32, 1, 1), 0);
        let operation = TypedOperationSpec::new(
            "implementation-duplicate-seed",
            OperationKind::Matvec,
            OperationRoute::CudaKernel,
        )
        .with_launch(launch.clone());
        KernelCandidateMetadata::new(
            "implementation-duplicate",
            vec![KernelAxis::spatial(0, "row", 1, Some(1))],
            KernelSchedule::new(),
            "implementation-duplicate-generator",
            KernelMaterialization::DeferredGenerated {
                symbol_hint: "impl_duplicate_seed".to_string(),
                reason: "seed metadata".to_string(),
            },
            launch,
            operation,
        )
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        if candidate.schedule.transforms.is_empty() {
            vec![Self::candidate(128, 16), Self::candidate(256, 32)]
        } else {
            Vec::new()
        }
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        if candidate.schedule.transforms.is_empty() {
            SearchScore::heuristic(10.0)
        } else {
            SearchScore::heuristic(1.0)
        }
    }
}

#[test]
fn expansion_dedupes_candidates_by_implementation_key() {
    let problem = ImplementationDuplicateSearchProblem;
    let seed = problem.seed();
    let mut seen = HashSet::new();
    seen.insert(seed.implementation_key());

    let expansion = expand_metadata_candidates(
        &problem,
        &seed,
        KernelExpansionPolicy::for_search_config(false),
        &mut seen,
    );
    assert_eq!(expansion.accepted, 1);
    assert_eq!(expansion.duplicates, 1);
    assert_ne!(
        problem.expand(&seed)[0].artifact_key(),
        problem.expand(&seed)[1].artifact_key()
    );
    assert_eq!(
        problem.expand(&seed)[0].implementation_key(),
        problem.expand(&seed)[1].implementation_key()
    );

    let result = beam_search_metadata(
        &problem,
        BeamSearchConfig {
            beam_width: 2,
            max_depth: 1,
            require_launchable: false,
        },
    );
    assert_eq!(result.duplicates, 1);
    assert_eq!(result.beam.len(), 1);
}
