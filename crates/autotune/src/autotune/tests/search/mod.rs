use super::*;

#[derive(Debug, Clone, Copy)]
struct BudgetedSearchProblem;

impl BudgetedSearchProblem {
    fn candidate(factor: u32, accumulators: u32) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new(
            format!("budget_candidate_{factor}"),
            (1, 1, 1),
            (1, 1, 1),
            0,
        );
        let operation = TypedOperationSpec::new(
            format!("budget-candidate-{factor}"),
            OperationKind::Gemm,
            OperationRoute::CudaKernel,
        )
        .with_launch(launch.clone());
        let mut candidate = KernelCandidateMetadata::new(
            "budget-test",
            vec![KernelAxis::spatial(0, "x", 1, Some(1))],
            KernelSchedule::new().with_transform(ScheduleTransform::Split { axis: 0, factor }),
            "budget-generator",
            KernelMaterialization::DeferredGenerated {
                symbol_hint: format!("budget_candidate_{factor}"),
                reason: "test candidate only carries metadata".to_string(),
            },
            launch,
            operation,
        );
        candidate.resources = Some(KernelResourceUsage::new(
            1,
            0,
            accumulators,
            accumulators,
            1,
        ));
        candidate
    }
}

impl KernelMetadataSearchProblem for BudgetedSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        Self::candidate(0, 1)
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        if candidate.schedule.depth() == 1 {
            vec![Self::candidate(1, 4), Self::candidate(2, 16)]
        } else {
            Vec::new()
        }
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let accumulators = candidate.resources?.accumulator_elements_per_thread;
        if candidate
            .schedule
            .transforms
            .iter()
            .any(|transform| matches!(transform, ScheduleTransform::Split { axis: 0, factor: 0 }))
        {
            SearchScore::measured(20.0)
        } else if accumulators > 8 {
            SearchScore::measured(1.0)
        } else {
            SearchScore::measured(10.0)
        }
    }
}

mod auto;
mod expansion;
mod policy;
mod replay;
mod templates;
