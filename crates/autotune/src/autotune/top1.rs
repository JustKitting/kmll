use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Top1Bf16SearchProblem {
    pub rows: usize,
    pub cols: usize,
}

impl Top1Bf16SearchProblem {
    pub const FAMILY: &'static str = "top1-bf16-row-major";
    const GENERATOR: &'static str = "row-major-top1-existing";
    const KERNEL: &'static str = "matvec_top1_bf16_stage_kernel";

    pub const fn row_major(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub const fn family(&self) -> &'static str {
        Self::FAMILY
    }

    fn row_factors(&self) -> [u32; 4] {
        [1, 2, 4, 8]
    }

    fn candidate_for_rows_per_block(&self, rows_per_block: u32) -> KernelCandidateMetadata {
        let block_threads = rows_per_block * 32;
        let schedule = KernelSchedule::new()
            .with_transform(ScheduleTransform::Split {
                axis: 0,
                factor: rows_per_block,
            })
            .with_transform(ScheduleTransform::ThreadGroup {
                axis: 0,
                factor: block_threads,
            });
        let launch = CudaLaunchSpec::new(
            Self::KERNEL,
            (self.rows.div_ceil(rows_per_block as usize) as u32, 1, 1),
            (block_threads, 1, 1),
            0,
        );
        let operation = TypedOperationSpec::new(
            format!("top1-bf16-row-major-rows{rows_per_block}"),
            OperationKind::Logits,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [self.cols])
                .with_layout("contiguous"),
        )
        .with_input(
            TensorTypeSpec::new(NumericKind::Bf16, NumericKind::F32, [self.rows, self.cols])
                .with_layout("row-major"),
        )
        .with_output(
            TensorTypeSpec::new(NumericKind::U64, NumericKind::U64, [1]).with_layout("packed"),
        )
        .with_launch(launch.clone());

        KernelCandidateMetadata::new(
            Self::FAMILY,
            self.axes(),
            schedule,
            Self::GENERATOR,
            KernelMaterialization::Existing {
                symbol: Self::KERNEL,
            },
            launch,
            operation,
        )
    }

    fn axes(&self) -> Vec<KernelAxis> {
        vec![
            KernelAxis::spatial(0, "row", self.rows, Some(self.cols)),
            KernelAxis::reduction(1, "col", self.cols, Some(1)),
        ]
    }

    pub fn candidate_rows_per_block(candidate: &KernelCandidateMetadata) -> Option<u32> {
        candidate
            .schedule
            .transforms
            .iter()
            .find_map(|transform| match transform {
                ScheduleTransform::Split { axis: 0, factor } => Some(*factor),
                _ => None,
            })
    }
}

impl KernelMetadataSearchProblem for Top1Bf16SearchProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new(Self::KERNEL, (1, 1, 1), (32, 1, 1), 0);
        let operation = TypedOperationSpec::new(
            "top1-bf16-row-major-seed",
            OperationKind::Logits,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [self.cols])
                .with_layout("contiguous"),
        )
        .with_input(
            TensorTypeSpec::new(NumericKind::Bf16, NumericKind::F32, [self.rows, self.cols])
                .with_layout("row-major"),
        )
        .with_output(
            TensorTypeSpec::new(NumericKind::U64, NumericKind::U64, [1]).with_layout("packed"),
        )
        .with_launch(launch.clone());
        KernelCandidateMetadata::new(
            Self::FAMILY,
            self.axes(),
            KernelSchedule::new(),
            Self::GENERATOR,
            KernelMaterialization::DeferredGenerated {
                symbol_hint: Self::KERNEL.to_string(),
                reason: "seed descriptor has no concrete top1 row split".to_string(),
            },
            launch,
            operation,
        )
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        expand_with_schedule_actions(self, candidate)
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let rows_per_block = Self::candidate_rows_per_block(candidate)? as usize;
        let blocks = self.rows.div_ceil(rows_per_block);
        let useful_fma_ops = self.rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let stage_overhead = blocks as f64 * (2048.0 + rows_per_block as f64 * 128.0);
        let reduction_overhead = blocks.max(1).next_power_of_two() as f64 * 32.0;
        SearchScore::heuristic(useful_fma_ops + stage_overhead + reduction_overhead)
    }
}

impl KernelActionSearchProblem for Top1Bf16SearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet {
        KernelActionSpaceSet::new([KernelActionSpace::Split {
            variants: self
                .row_factors()
                .into_iter()
                .map(|factor| {
                    KernelAxisFactorAction::new(0, factor, KernelActionMaterialization::Existing)
                })
                .collect(),
        }])
    }

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
        if Self::candidate_rows_per_block(candidate).is_some() {
            KernelActionSpaceSet::default()
        } else {
            self.search_space()
        }
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata> {
        if action.op != KernelScheduleActionOp::Split || action.axis != Some(0) {
            return None;
        }
        let KernelScheduleActionArg::Factor(rows_per_block) = action.arg else {
            return None;
        };
        if !self.row_factors().contains(&rows_per_block) {
            return None;
        }
        let next = self.candidate_for_rows_per_block(rows_per_block);
        Some(candidate_with_action_trace(candidate, action, next))
    }
}
