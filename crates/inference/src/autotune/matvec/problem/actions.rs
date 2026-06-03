use super::{
    candidate::{MatvecReduceGroupingTransform, MatvecRowGroupingTransform},
    *,
};

impl KernelActionSearchProblem for MatvecSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet {
        let split_variants = self.split_variants();
        let group_top_factors = self.group_top_factors();
        let row_upcast_factors = self.row_upcast_factors();
        let unroll_factors = self.reduce_unroll_factors();
        let group_factors = self.group_factors();
        KernelActionSpaceSet::new(vec![
            KernelActionSpace::Split {
                variants: split_variants,
            },
            KernelActionSpace::GroupTop {
                axis: 0,
                factors: group_top_factors,
            },
            KernelActionSpace::Upcast {
                axis: 0,
                factors: row_upcast_factors,
            },
            KernelActionSpace::Unroll {
                axis: 1,
                factors: unroll_factors,
            },
            KernelActionSpace::Group {
                axis: 1,
                factors: group_factors,
            },
        ])
    }

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
        if candidate.family != "matvec-bf16-row-major" {
            return KernelActionSpaceSet::default();
        }
        if candidate.schedule.depth() == 0 {
            return KernelActionSpaceSet::new(vec![
                KernelActionSpace::Split {
                    variants: self.split_variants(),
                },
                KernelActionSpace::GroupTop {
                    axis: 0,
                    factors: self.group_top_factors(),
                },
            ]);
        }
        let Some(plan) = schedule_matvec_plan(&candidate.schedule) else {
            return KernelActionSpaceSet::default();
        };
        if candidate.generated.materialization.is_existing() {
            return KernelActionSpaceSet::default();
        }
        let mut spaces = Vec::new();
        if plan.row_upcast.is_default() {
            let factors = self.row_upcast_factors_for_rows(plan.rows);
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Upcast { axis: 0, factors });
            }
        }
        if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
            spaces.push(KernelActionSpace::Unroll {
                axis: 1,
                factors: self.reduce_unroll_factors(),
            });
        }
        if plan.thread_group.is_default() {
            spaces.push(KernelActionSpace::Group {
                axis: 1,
                factors: self.group_factors(),
            });
        }
        KernelActionSpaceSet::new(spaces)
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata> {
        if candidate.family != "matvec-bf16-row-major" {
            return None;
        }
        match action {
            KernelScheduleAction {
                op: KernelScheduleActionOp::Split,
                axis: Some(0),
                arg: KernelScheduleActionArg::Factor(rows_per_block),
                materialization,
            } => {
                if candidate.schedule.depth() > 0 {
                    return None;
                }
                let next = match materialization {
                    KernelActionMaterialization::Existing => {
                        let rows = RowMajorWarpRows::from_rows_per_block(*rows_per_block)?;
                        self.candidate_for_rows(rows)
                    }
                    KernelActionMaterialization::DeferredGenerated => {
                        if !self.deferred_row_split_factors().contains(rows_per_block) {
                            return None;
                        }
                        let rows = MatvecRowSplit::new(*rows_per_block)?;
                        self.generated_candidate_for_row_split(rows)
                    }
                };
                Some(candidate_with_action_trace(candidate, action, next))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::GroupTop,
                axis: Some(0),
                arg: KernelScheduleActionArg::Factor(rows_per_block),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.schedule.depth() > 0
                    || !self.group_top_factors().contains(rows_per_block)
                {
                    return None;
                }
                let rows = MatvecRowSplit::new(*rows_per_block)?;
                let next = self.generated_candidate_for_row_group_top(rows);
                Some(candidate_with_action_trace(candidate, action, next))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Upcast,
                axis: Some(0),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.generated.materialization.is_existing() {
                    return None;
                }
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if !plan.row_upcast.is_default()
                    || !self.row_upcast_factors_for_rows(plan.rows).contains(factor)
                {
                    return None;
                }
                let row_upcast = MatvecRowUpcast::new(*factor)?;
                let next = self.generated_candidate_for_plan_with_grouping(
                    plan.with_row_upcast(row_upcast),
                    row_grouping_transform(&candidate.schedule),
                    reduce_grouping_transform(&candidate.schedule),
                );
                Some(candidate_with_action_trace(candidate, action, next))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(1),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.generated.materialization.is_existing()
                    || !self.reduce_unroll_factors().contains(factor)
                {
                    return None;
                }
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if plan.reduce_unroll != MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
                    return None;
                }
                let next = self.generated_candidate_for_plan_with_grouping(
                    plan.with_reduce_unroll(*factor),
                    row_grouping_transform(&candidate.schedule),
                    reduce_grouping_transform(&candidate.schedule),
                );
                Some(candidate_with_action_trace(candidate, action, next))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Group,
                axis: Some(1),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.generated.materialization.is_existing()
                    || !self.group_factors().contains(factor)
                {
                    return None;
                }
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if !plan.thread_group.is_default() {
                    return None;
                }
                let thread_group = MatvecThreadGroup::new(*factor)?;
                let next = self.generated_candidate_for_plan_with_grouping(
                    plan.with_thread_group(thread_group),
                    row_grouping_transform(&candidate.schedule),
                    MatvecReduceGroupingTransform::Group,
                );
                Some(candidate_with_action_trace(candidate, action, next))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::ThreadGroup,
                axis: Some(1),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.generated.materialization.is_existing()
                    || !self.thread_group_factors().contains(factor)
                {
                    return None;
                }
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if !plan.thread_group.is_default() {
                    return None;
                }
                let thread_group = MatvecThreadGroup::new(*factor)?;
                let next = self.generated_candidate_for_plan_with_grouping(
                    plan.with_thread_group(thread_group),
                    row_grouping_transform(&candidate.schedule),
                    MatvecReduceGroupingTransform::ThreadGroup,
                );
                Some(candidate_with_action_trace(candidate, action, next))
            }
            _ => None,
        }
    }
}

fn row_grouping_transform(schedule: &KernelSchedule) -> MatvecRowGroupingTransform {
    if schedule
        .transforms
        .iter()
        .any(|transform| matches!(transform, ScheduleTransform::GroupTop { axis: 0, .. }))
    {
        MatvecRowGroupingTransform::GroupTop
    } else {
        MatvecRowGroupingTransform::Split
    }
}

fn reduce_grouping_transform(schedule: &KernelSchedule) -> MatvecReduceGroupingTransform {
    if schedule
        .transforms
        .iter()
        .any(|transform| matches!(transform, ScheduleTransform::Group { axis: 1, .. }))
    {
        MatvecReduceGroupingTransform::Group
    } else {
        MatvecReduceGroupingTransform::ThreadGroup
    }
}
