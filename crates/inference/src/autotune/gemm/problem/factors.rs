use super::*;

impl GemmSearchProblem {
    pub(in crate::autotune) fn reduce_unroll_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            factors.extend(Self::reduce_unroll_factors_for_tile(tile));
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub(in crate::autotune) fn reduce_unroll_factors_for_tile(tile: GemmTileShape) -> Vec<u32> {
        bounded_unroll_factors(tile.k as usize, Self::MAX_REDUCE_UNROLL_FACTOR, Some(1))
    }

    pub(in crate::autotune) fn m_per_thread_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            factors.extend(Self::m_per_thread_factors_for_tile(tile));
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub(in crate::autotune) fn m_per_thread_factors_for_tile(tile: GemmTileShape) -> Vec<u32> {
        const MAX_M_PER_THREAD: u32 = 4;
        KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .legal_upcast_factors(|factor| factor <= MAX_M_PER_THREAD && tile.m % factor == 0)
    }

    pub(in crate::autotune) fn n_per_thread_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            factors.extend(Self::n_per_thread_factors_for_tile(tile));
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub(in crate::autotune) fn n_per_thread_factors_for_tile(tile: GemmTileShape) -> Vec<u32> {
        const MAX_N_PER_THREAD: u32 = 4;
        KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .legal_upcast_factors(|factor| factor <= MAX_N_PER_THREAD && tile.n % factor == 0)
    }

    pub(in crate::autotune) fn per_thread_plans_for_tile(
        tile: GemmTileShape,
    ) -> Vec<GemmSchedulePlan> {
        let mut m_factors = vec![1];
        m_factors.extend(Self::m_per_thread_factors_for_tile(tile));
        let mut n_factors = vec![1];
        n_factors.extend(Self::n_per_thread_factors_for_tile(tile));

        let mut plans = Vec::new();
        for m_per_thread in m_factors {
            for n_per_thread in n_factors.iter().copied() {
                plans.push(
                    GemmSchedulePlan::new(tile)
                        .with_m_per_thread(m_per_thread)
                        .with_n_per_thread(n_per_thread),
                );
            }
        }
        plans
    }

    pub(in crate::autotune) fn a_load_unroll_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            for plan in Self::per_thread_plans_for_tile(tile) {
                factors.extend(Self::a_load_unroll_factors_for_plan(plan));
            }
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub(in crate::autotune) fn b_load_unroll_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            for plan in Self::per_thread_plans_for_tile(tile) {
                factors.extend(Self::b_load_unroll_factors_for_plan(plan));
            }
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub(in crate::autotune) fn a_load_unroll_factors_for_plan(plan: GemmSchedulePlan) -> Vec<u32> {
        bounded_unroll_factors(
            plan.a_load_rounds() as usize,
            Self::MAX_LOAD_UNROLL_FACTOR,
            Some(1),
        )
    }

    pub(in crate::autotune) fn b_load_unroll_factors_for_plan(plan: GemmSchedulePlan) -> Vec<u32> {
        bounded_unroll_factors(
            plan.b_load_rounds() as usize,
            Self::MAX_LOAD_UNROLL_FACTOR,
            Some(1),
        )
    }

    pub(in crate::autotune) fn split_factors_for_axis(&self, axis: u8) -> Vec<u32> {
        match axis {
            0 => bounded_tile_factors(self.m, Self::MAX_TILE_DIM, None),
            1 => bounded_tile_factors(self.n, Self::MAX_TILE_DIM, None),
            2 => bounded_tile_factors(self.k, Self::MAX_TILE_DIM, None),
            _ => Vec::new(),
        }
    }

    pub(in crate::autotune) fn split_action_variants(&self) -> Vec<KernelAxisFactorAction> {
        let mut variants = Vec::new();
        for axis in 0..=2 {
            variants.extend(self.split_factors_for_axis(axis).into_iter().map(|factor| {
                KernelAxisFactorAction::new(
                    axis,
                    factor,
                    KernelActionMaterialization::DeferredGenerated,
                )
            }));
        }
        variants
    }

    pub(in crate::autotune) fn split_action_variants_for_plan(
        &self,
        plan: GemmSchedulePlan,
    ) -> Vec<KernelAxisFactorAction> {
        let mut variants = Vec::new();
        for axis in 0..=2 {
            let Some(current_factor) = plan.tile.axis_factor(axis) else {
                continue;
            };
            for factor in self.split_factors_for_axis(axis) {
                if factor == current_factor {
                    continue;
                }
                let Some(tile) = plan.tile.with_axis(axis, factor) else {
                    continue;
                };
                if !tile.is_launchable_shape() {
                    continue;
                }
                let next_plan = plan.with_tile(tile);
                if !Self::plan_within_resource_limits(next_plan) {
                    continue;
                }
                variants.push(KernelAxisFactorAction::new(
                    axis,
                    factor,
                    Self::action_materialization_for_plan(next_plan),
                ));
            }
        }
        variants
    }

    pub(in crate::autotune) fn a_load_thread_group_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            for plan in Self::per_thread_plans_for_tile(tile) {
                factors.extend(Self::load_thread_group_factors_for_plan(plan));
            }
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub(in crate::autotune) fn b_load_thread_group_factors(&self) -> Vec<u32> {
        self.a_load_thread_group_factors()
    }

    pub(in crate::autotune) fn load_thread_group_factors_for_plan(
        plan: GemmSchedulePlan,
    ) -> Vec<u32> {
        let thread_count = plan.thread_count();
        KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .legal_thread_group_factors(|factor| factor >= 32 && factor < thread_count)
    }

    pub(in crate::autotune) fn tile_shapes(&self) -> Vec<GemmTileShape> {
        let m_factors =
            bounded_tile_factors(self.m, Self::MAX_TILE_DIM, Some(Self::EXISTING_TILE.m));
        let n_factors =
            bounded_tile_factors(self.n, Self::MAX_TILE_DIM, Some(Self::EXISTING_TILE.n));
        let k_factors =
            bounded_tile_factors(self.k, Self::MAX_TILE_DIM, Some(Self::EXISTING_TILE.k));
        let mut tiles = Vec::new();
        for m in m_factors {
            for n in n_factors.iter().copied() {
                for k in k_factors.iter().copied() {
                    let tile = GemmTileShape::new(m, n, k);
                    if Self::plan_within_resource_limits(GemmSchedulePlan::new(tile)) {
                        tiles.push(tile);
                    }
                }
            }
        }
        tiles.sort_unstable_by_key(|tile| (tile.m, tile.n, tile.k));
        tiles.dedup();
        tiles
    }

    pub(in crate::autotune) fn seed_tile_action_variants() -> Vec<KernelTile3dAction> {
        vec![KernelTile3dAction::new(
            Self::EXISTING_TILE.into(),
            KernelActionMaterialization::Existing,
        )]
    }

    pub(in crate::autotune) fn stride_orders_for_plan(plan: GemmSchedulePlan) -> Vec<Vec<u8>> {
        let mut orders = Vec::new();
        if plan.a_load_order == GemmATileLoadOrder::KContiguous {
            orders.push(GemmATileLoadOrder::MContiguous.action_axes());
        }
        if plan.b_load_order == GemmBTileLoadOrder::TileLinear {
            orders.push(GemmBTileLoadOrder::KContiguous.action_axes());
        }
        orders
    }
}
