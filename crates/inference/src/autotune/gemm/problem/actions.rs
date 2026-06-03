use super::*;

impl KernelActionSearchProblem for GemmSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet {
        let split_variants = self.split_action_variants();
        let unroll_factors = self.reduce_unroll_factors();
        let m_per_thread_factors = self.m_per_thread_factors();
        let n_per_thread_factors = self.n_per_thread_factors();
        let a_load_unroll_factors = self.a_load_unroll_factors();
        let b_load_unroll_factors = self.b_load_unroll_factors();
        let a_load_thread_group_factors = self.a_load_thread_group_factors();
        let b_load_thread_group_factors = self.b_load_thread_group_factors();
        let stride_orders =
            Self::stride_orders_for_plan(GemmSchedulePlan::new(Self::EXISTING_TILE));
        let mut spaces = vec![
            KernelActionSpace::Split {
                variants: split_variants,
            },
            KernelActionSpace::TileGemm {
                variants: Self::seed_tile_action_variants(),
            },
            KernelActionSpace::Unroll {
                axis: 2,
                factors: unroll_factors,
            },
            KernelActionSpace::Upcast {
                axis: 0,
                factors: m_per_thread_factors,
            },
            KernelActionSpace::Upcast {
                axis: 1,
                factors: n_per_thread_factors,
            },
        ];
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
            spaces.push(KernelActionSpace::ThreadGroup {
                axis: 3,
                factors: a_load_thread_group_factors,
            });
        }
        if !b_load_thread_group_factors.is_empty() {
            spaces.push(KernelActionSpace::ThreadGroup {
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

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
        let Some(plan) = schedule_gemm_plan(&candidate.schedule) else {
            if !candidate.schedule.transforms.is_empty() {
                return KernelActionSpaceSet::default();
            }
            let split_variants =
                self.split_action_variants_for_plan(GemmSchedulePlan::new(Self::EXISTING_TILE));
            return KernelActionSpaceSet::new(vec![
                KernelActionSpace::Split {
                    variants: split_variants,
                },
                KernelActionSpace::TileGemm {
                    variants: Self::seed_tile_action_variants(),
                },
            ]);
        };

        let mut spaces = Vec::new();
        let split_variants = self.split_action_variants_for_plan(plan);
        if !split_variants.is_empty() {
            spaces.push(KernelActionSpace::Split {
                variants: split_variants,
            });
        }
        if plan.reduce_unroll == 1 {
            let factors = Self::reduce_unroll_factors_for_tile(plan.tile)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_reduce_unroll(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Unroll { axis: 2, factors });
            }
        }
        if plan.m_per_thread == 1 {
            let factors = Self::m_per_thread_factors_for_tile(plan.tile)
                .into_iter()
                .filter(|factor| Self::plan_within_resource_limits(plan.with_m_per_thread(*factor)))
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Upcast { axis: 0, factors });
            }
        }
        if plan.n_per_thread == 1 {
            let factors = Self::n_per_thread_factors_for_tile(plan.tile)
                .into_iter()
                .filter(|factor| Self::plan_within_resource_limits(plan.with_n_per_thread(*factor)))
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Upcast { axis: 1, factors });
            }
        }
        if plan.a_load_unroll == 1 {
            let factors = Self::a_load_unroll_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_a_load_unroll(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Unroll { axis: 3, factors });
            }
        }
        if plan.b_load_unroll == 1 {
            let factors = Self::b_load_unroll_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_b_load_unroll(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Unroll { axis: 4, factors });
            }
        }
        if !plan.has_custom_a_load_thread_group() {
            let factors = Self::load_thread_group_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_a_load_thread_group(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::ThreadGroup { axis: 3, factors });
            }
        }
        if !plan.has_custom_b_load_thread_group() {
            let factors = Self::load_thread_group_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_b_load_thread_group(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::ThreadGroup { axis: 4, factors });
            }
        }
        if plan.thread_order == GemmThreadOrder::NThenM
            && Self::plan_within_resource_limits(plan.with_thread_order(GemmThreadOrder::MThenN))
        {
            spaces.push(KernelActionSpace::Swap {
                pairs: vec![(0, 1)],
            });
        }
        let orders = Self::stride_orders_for_plan(plan)
            .into_iter()
            .filter(|axes| {
                GemmATileLoadOrder::from_action_axes(axes)
                    .map(|order| Self::plan_within_resource_limits(plan.with_a_load_order(order)))
                    .or_else(|| {
                        GemmBTileLoadOrder::from_action_axes(axes).map(|order| {
                            Self::plan_within_resource_limits(plan.with_b_load_order(order))
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

    fn apply_schedule_action(
        &self,
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
                if !self.tile_shapes().contains(&tile) {
                    return None;
                }
                let plan = GemmSchedulePlan::new(tile);
                if *materialization != Self::action_materialization_for_plan(plan) {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan)
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
                        GemmSchedulePlan::new(Self::EXISTING_TILE)
                    }
                    None => return None,
                };
                let variants = self.split_action_variants_for_plan(plan);
                if !variants.contains(&KernelAxisFactorAction::new(
                    *axis,
                    *factor,
                    *materialization,
                )) {
                    return None;
                }
                let next_tile = plan.tile.with_axis(*axis, *factor)?;
                self.candidate_for_checked_plan(candidate, action, plan.with_tile(next_tile))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(2),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.reduce_unroll != 1
                    || !Self::reduce_unroll_factors_for_tile(plan.tile).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_reduce_unroll(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(3),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.a_load_unroll != 1
                    || !Self::a_load_unroll_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_a_load_unroll(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(4),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.b_load_unroll != 1
                    || !Self::b_load_unroll_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_b_load_unroll(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Upcast,
                axis: Some(0),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.m_per_thread != 1
                    || !Self::m_per_thread_factors_for_tile(plan.tile).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_m_per_thread(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Upcast,
                axis: Some(1),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.n_per_thread != 1
                    || !Self::n_per_thread_factors_for_tile(plan.tile).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_n_per_thread(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::ThreadGroup,
                axis: Some(3),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.has_custom_a_load_thread_group()
                    || !Self::load_thread_group_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(
                    candidate,
                    action,
                    plan.with_a_load_thread_group(*factor),
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
                    || !Self::load_thread_group_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(
                    candidate,
                    action,
                    plan.with_b_load_thread_group(*factor),
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
                    return self.candidate_for_checked_plan(
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
                self.candidate_for_checked_plan(candidate, action, plan.with_b_load_order(order))
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
                self.candidate_for_checked_plan(
                    candidate,
                    action,
                    plan.with_thread_order(GemmThreadOrder::MThenN),
                )
            }
            _ => None,
        }
    }
}
