use super::super::{
    candidate::{
        GemmLoadGroupingTransform, gemm_a_load_grouping_transform, gemm_b_load_grouping_transform,
    },
    *,
};

pub(super) fn apply_schedule_action(
    problem: &GemmSearchProblem,
    candidate: &KernelCandidateMetadata,
    action: &KernelScheduleAction,
) -> Option<KernelCandidateMetadata> {
    match action {
        KernelScheduleAction {
            op: KernelScheduleActionOp::TileGemm,
            axis: None,
            arg: KernelScheduleActionArg::Tile3d { m, n, k },
            materialization,
        } => {
            if schedule_gemm_plan(&candidate.schedule).is_some() {
                return None;
            }
            let tile = GemmTileShape::new(*m, *n, *k);
            if !problem.tile_shapes().contains(&tile) {
                return None;
            }
            let plan = GemmSchedulePlan::new(tile);
            if *materialization != GemmSearchProblem::action_materialization_for_plan(plan) {
                return None;
            }
            problem.candidate_for_checked_plan(candidate, action, plan)
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::Split,
            axis: Some(axis @ 0..=2),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization,
        } => {
            let plan = match schedule_gemm_plan(&candidate.schedule) {
                Some(plan) => plan,
                None if candidate.schedule.transforms.is_empty() => {
                    GemmSchedulePlan::new(GemmSearchProblem::EXISTING_TILE)
                }
                None => return None,
            };
            let variants = problem.split_action_variants_for_plan(plan);
            if !variants.contains(&KernelAxisFactorAction::new(
                *axis,
                *factor,
                *materialization,
            )) {
                return None;
            }
            let next_tile = plan.tile.with_axis(*axis, *factor)?;
            problem.candidate_for_checked_plan(candidate, action, plan.with_tile(next_tile))
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::LocalTile,
            axis: Some(axis @ 0..=2),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = match schedule_gemm_plan(&candidate.schedule) {
                Some(plan) => plan,
                None if candidate.schedule.transforms.is_empty() => {
                    GemmSchedulePlan::new(GemmSearchProblem::EXISTING_TILE)
                }
                None => return None,
            };
            let factors = problem
                .local_tile_action_spaces_for_plan(plan, true)
                .into_iter()
                .find_map(|space| match space {
                    KernelActionSpace::LocalTile {
                        axis: space_axis,
                        factors,
                    } if space_axis == *axis => Some(factors),
                    _ => None,
                })?;
            if !factors.contains(factor) {
                return None;
            }
            let next_tile = plan.tile.with_axis(*axis, *factor)?;
            problem.candidate_for_checked_plan(candidate, action, plan.with_tile(next_tile))
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::Unroll,
            axis: Some(2),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if plan.reduce_unroll != 1
                || !GemmSearchProblem::reduce_unroll_factors_for_tile(plan.tile).contains(factor)
            {
                return None;
            }
            problem.candidate_for_checked_plan(candidate, action, plan.with_reduce_unroll(*factor))
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::Unroll,
            axis: Some(3),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if plan.a_load_unroll != 1
                || !GemmSearchProblem::a_load_unroll_factors_for_plan(plan).contains(factor)
            {
                return None;
            }
            problem.candidate_for_checked_plan(candidate, action, plan.with_a_load_unroll(*factor))
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::Unroll,
            axis: Some(4),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if plan.b_load_unroll != 1
                || !GemmSearchProblem::b_load_unroll_factors_for_plan(plan).contains(factor)
            {
                return None;
            }
            problem.candidate_for_checked_plan(candidate, action, plan.with_b_load_unroll(*factor))
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::Upcast,
            axis: Some(0),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if plan.m_per_thread != 1
                || !GemmSearchProblem::m_per_thread_factors_for_tile(plan.tile).contains(factor)
            {
                return None;
            }
            problem.candidate_for_checked_plan(candidate, action, plan.with_m_per_thread(*factor))
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::Upcast,
            axis: Some(1),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if plan.n_per_thread != 1
                || !GemmSearchProblem::n_per_thread_factors_for_tile(plan.tile).contains(factor)
            {
                return None;
            }
            problem.candidate_for_checked_plan(candidate, action, plan.with_n_per_thread(*factor))
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::Group,
            axis: Some(3),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if plan.has_custom_a_load_thread_group()
                || !GemmSearchProblem::load_thread_group_factors_for_plan(plan).contains(factor)
            {
                return None;
            }
            problem.candidate_for_checked_plan_with_load_grouping(
                candidate,
                action,
                plan.with_a_load_thread_group(*factor),
                GemmLoadGroupingTransform::Group,
                gemm_b_load_grouping_transform(&candidate.schedule),
            )
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::Group,
            axis: Some(4),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if plan.has_custom_b_load_thread_group()
                || !GemmSearchProblem::load_thread_group_factors_for_plan(plan).contains(factor)
            {
                return None;
            }
            problem.candidate_for_checked_plan_with_load_grouping(
                candidate,
                action,
                plan.with_b_load_thread_group(*factor),
                gemm_a_load_grouping_transform(&candidate.schedule),
                GemmLoadGroupingTransform::Group,
            )
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::ThreadGroup,
            axis: Some(3),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if plan.has_custom_a_load_thread_group()
                || !GemmSearchProblem::load_thread_group_factors_for_plan(plan).contains(factor)
            {
                return None;
            }
            problem.candidate_for_checked_plan_with_load_grouping(
                candidate,
                action,
                plan.with_a_load_thread_group(*factor),
                GemmLoadGroupingTransform::ThreadGroup,
                gemm_b_load_grouping_transform(&candidate.schedule),
            )
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::ThreadGroup,
            axis: Some(4),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if plan.has_custom_b_load_thread_group()
                || !GemmSearchProblem::load_thread_group_factors_for_plan(plan).contains(factor)
            {
                return None;
            }
            problem.candidate_for_checked_plan_with_load_grouping(
                candidate,
                action,
                plan.with_b_load_thread_group(*factor),
                gemm_a_load_grouping_transform(&candidate.schedule),
                GemmLoadGroupingTransform::ThreadGroup,
            )
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::StrideOrder,
            axis: None,
            arg: KernelScheduleActionArg::AxisOrder(axes),
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if let Some(order) = GemmATileLoadOrder::from_action_axes(axes) {
                if plan.a_load_order != GemmATileLoadOrder::KContiguous
                    || order == GemmATileLoadOrder::KContiguous
                {
                    return None;
                }
                return problem.candidate_for_checked_plan(
                    candidate,
                    action,
                    plan.with_a_load_order(order),
                );
            }
            let order = GemmBTileLoadOrder::from_action_axes(axes)?;
            if plan.b_load_order != GemmBTileLoadOrder::TileLinear
                || order == GemmBTileLoadOrder::TileLinear
            {
                return None;
            }
            problem.candidate_for_checked_plan(candidate, action, plan.with_b_load_order(order))
        }
        KernelScheduleAction {
            op: KernelScheduleActionOp::Swap,
            axis: None,
            arg: KernelScheduleActionArg::AxisPair { axis_a, axis_b },
            materialization: KernelActionMaterialization::DeferredGenerated,
        } => {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            if (*axis_a, *axis_b) != (0, 1) || plan.thread_order != GemmThreadOrder::NThenM {
                return None;
            }
            problem.candidate_for_checked_plan(
                candidate,
                action,
                plan.with_thread_order(GemmThreadOrder::MThenN),
            )
        }
        _ => None,
    }
}
