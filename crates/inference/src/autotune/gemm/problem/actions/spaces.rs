use super::super::*;

pub(super) fn search_space(problem: &GemmSearchProblem) -> KernelActionSpaceSet {
    let unroll_factors = problem.reduce_unroll_factors();
    let m_per_thread_factors = problem.m_per_thread_factors();
    let n_per_thread_factors = problem.n_per_thread_factors();
    let a_load_unroll_factors = problem.a_load_unroll_factors();
    let b_load_unroll_factors = problem.b_load_unroll_factors();
    let a_load_thread_group_factors = problem.a_load_thread_group_factors();
    let b_load_thread_group_factors = problem.b_load_thread_group_factors();
    let stride_orders = GemmSearchProblem::stride_orders_for_plan(GemmSchedulePlan::new(
        GemmSearchProblem::EXISTING_TILE,
    ));
    let mut spaces = problem.local_tile_action_spaces_for_plan(
        GemmSchedulePlan::new(GemmSearchProblem::EXISTING_TILE),
        false,
    );
    spaces.push(KernelActionSpace::TileGemm {
        variants: GemmSearchProblem::seed_tile_action_variants(),
    });
    spaces.push(KernelActionSpace::Unroll {
        axis: 2,
        factors: unroll_factors,
    });
    spaces.push(KernelActionSpace::Upcast {
        axis: 0,
        factors: m_per_thread_factors,
    });
    spaces.push(KernelActionSpace::Upcast {
        axis: 1,
        factors: n_per_thread_factors,
    });
    if !a_load_unroll_factors.is_empty() {
        spaces.push(KernelActionSpace::Unroll {
            axis: 3,
            factors: a_load_unroll_factors,
        });
    }
    if !b_load_unroll_factors.is_empty() {
        spaces.push(KernelActionSpace::Unroll {
            axis: 4,
            factors: b_load_unroll_factors,
        });
    }
    if !a_load_thread_group_factors.is_empty() {
        spaces.push(KernelActionSpace::Group {
            axis: 3,
            factors: a_load_thread_group_factors,
        });
    }
    if !b_load_thread_group_factors.is_empty() {
        spaces.push(KernelActionSpace::Group {
            axis: 4,
            factors: b_load_thread_group_factors,
        });
    }
    spaces.push(KernelActionSpace::Swap {
        pairs: vec![(0, 1)],
    });
    spaces.push(KernelActionSpace::StrideOrder {
        orders: stride_orders,
    });
    KernelActionSpaceSet::new(spaces)
}

pub(super) fn action_spaces(
    problem: &GemmSearchProblem,
    candidate: &KernelCandidateMetadata,
) -> KernelActionSpaceSet {
    let Some(plan) = schedule_gemm_plan(&candidate.schedule) else {
        if !candidate.schedule.transforms.is_empty() {
            return KernelActionSpaceSet::default();
        }
        let mut spaces = problem.local_tile_action_spaces_for_plan(
            GemmSchedulePlan::new(GemmSearchProblem::EXISTING_TILE),
            true,
        );
        spaces.push(KernelActionSpace::TileGemm {
            variants: GemmSearchProblem::seed_tile_action_variants(),
        });
        return KernelActionSpaceSet::new(spaces);
    };

    let mut spaces = Vec::new();
    spaces.extend(problem.local_tile_action_spaces_for_plan(plan, true));
    if plan.reduce_unroll == 1 {
        let factors = GemmSearchProblem::reduce_unroll_factors_for_tile(plan.tile)
            .into_iter()
            .filter(|factor| {
                GemmSearchProblem::plan_within_resource_limits(plan.with_reduce_unroll(*factor))
            })
            .collect::<Vec<_>>();
        if !factors.is_empty() {
            spaces.push(KernelActionSpace::Unroll { axis: 2, factors });
        }
    }
    if plan.m_per_thread == 1 {
        let factors = GemmSearchProblem::m_per_thread_factors_for_tile(plan.tile)
            .into_iter()
            .filter(|factor| {
                GemmSearchProblem::plan_within_resource_limits(plan.with_m_per_thread(*factor))
            })
            .collect::<Vec<_>>();
        if !factors.is_empty() {
            spaces.push(KernelActionSpace::Upcast { axis: 0, factors });
        }
    }
    if plan.n_per_thread == 1 {
        let factors = GemmSearchProblem::n_per_thread_factors_for_tile(plan.tile)
            .into_iter()
            .filter(|factor| {
                GemmSearchProblem::plan_within_resource_limits(plan.with_n_per_thread(*factor))
            })
            .collect::<Vec<_>>();
        if !factors.is_empty() {
            spaces.push(KernelActionSpace::Upcast { axis: 1, factors });
        }
    }
    if plan.a_load_unroll == 1 {
        let factors = GemmSearchProblem::a_load_unroll_factors_for_plan(plan)
            .into_iter()
            .filter(|factor| {
                GemmSearchProblem::plan_within_resource_limits(plan.with_a_load_unroll(*factor))
            })
            .collect::<Vec<_>>();
        if !factors.is_empty() {
            spaces.push(KernelActionSpace::Unroll { axis: 3, factors });
        }
    }
    if plan.b_load_unroll == 1 {
        let factors = GemmSearchProblem::b_load_unroll_factors_for_plan(plan)
            .into_iter()
            .filter(|factor| {
                GemmSearchProblem::plan_within_resource_limits(plan.with_b_load_unroll(*factor))
            })
            .collect::<Vec<_>>();
        if !factors.is_empty() {
            spaces.push(KernelActionSpace::Unroll { axis: 4, factors });
        }
    }
    if !plan.has_custom_a_load_thread_group() {
        let factors = GemmSearchProblem::load_thread_group_factors_for_plan(plan)
            .into_iter()
            .filter(|factor| {
                GemmSearchProblem::plan_within_resource_limits(
                    plan.with_a_load_thread_group(*factor),
                )
            })
            .collect::<Vec<_>>();
        if !factors.is_empty() {
            spaces.push(KernelActionSpace::Group { axis: 3, factors });
        }
    }
    if !plan.has_custom_b_load_thread_group() {
        let factors = GemmSearchProblem::load_thread_group_factors_for_plan(plan)
            .into_iter()
            .filter(|factor| {
                GemmSearchProblem::plan_within_resource_limits(
                    plan.with_b_load_thread_group(*factor),
                )
            })
            .collect::<Vec<_>>();
        if !factors.is_empty() {
            spaces.push(KernelActionSpace::Group { axis: 4, factors });
        }
    }
    if plan.thread_order == GemmThreadOrder::NThenM
        && GemmSearchProblem::plan_within_resource_limits(
            plan.with_thread_order(GemmThreadOrder::MThenN),
        )
    {
        spaces.push(KernelActionSpace::Swap {
            pairs: vec![(0, 1)],
        });
    }
    let orders = GemmSearchProblem::stride_orders_for_plan(plan)
        .into_iter()
        .filter(|axes| {
            GemmATileLoadOrder::from_action_axes(axes)
                .map(|order| {
                    GemmSearchProblem::plan_within_resource_limits(plan.with_a_load_order(order))
                })
                .or_else(|| {
                    GemmBTileLoadOrder::from_action_axes(axes).map(|order| {
                        GemmSearchProblem::plan_within_resource_limits(
                            plan.with_b_load_order(order),
                        )
                    })
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if !orders.is_empty() {
        spaces.push(KernelActionSpace::StrideOrder { orders });
    }
    KernelActionSpaceSet::new(spaces)
}
