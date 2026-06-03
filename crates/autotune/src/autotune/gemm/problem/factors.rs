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

    pub(in crate::autotune) fn reduce_group_top_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            factors.extend(Self::reduce_group_top_factors_for_plan(
                GemmSchedulePlan::new(tile),
            ));
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub(in crate::autotune) fn reduce_group_top_factors_for_plan(
        plan: GemmSchedulePlan,
    ) -> Vec<u32> {
        let plan = plan.normalized();
        let min_factor = plan.reduce_unroll.max(1);
        KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .group_top_factors
            .iter()
            .copied()
            .filter(|factor| *factor >= min_factor && *factor < plan.tile.k)
            .collect()
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
            0 => self.local_tile_factors_for_axis(axis),
            1 => self.local_tile_factors_for_axis(axis),
            2 => self.local_tile_factors_for_axis(axis),
            _ => Vec::new(),
        }
    }

    pub(in crate::autotune) fn local_tile_factors_for_axis(&self, axis: u8) -> Vec<u32> {
        self.local_tile_factors_for_axis_with_required(axis, None)
    }

    fn local_tile_factors_for_axis_with_required(
        &self,
        axis: u8,
        required_factor: Option<u32>,
    ) -> Vec<u32> {
        let extent = match axis {
            0 => self.m,
            1 => self.n,
            2 => self.k,
            _ => return Vec::new(),
        };
        let upper = extent.min(Self::MAX_TILE_DIM as usize) as u32;
        let mut factors = KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .local_tile_factors
            .iter()
            .copied()
            .filter(|factor| *factor <= upper)
            .collect::<Vec<_>>();
        if let Some(required) = required_factor {
            factors.push(required);
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub(in crate::autotune) fn local_tile_action_spaces_for_plan(
        &self,
        plan: GemmSchedulePlan,
        exclude_current: bool,
    ) -> Vec<KernelActionSpace> {
        let mut spaces = Vec::new();
        for axis in 0..=2 {
            let Some(current_factor) = plan.tile.axis_factor(axis) else {
                continue;
            };
            let factors = self
                .local_tile_factors_for_axis(axis)
                .into_iter()
                .filter(|factor| !exclude_current || *factor != current_factor)
                .filter(|factor| {
                    plan.tile
                        .with_axis(axis, *factor)
                        .map(|tile| Self::plan_within_resource_limits(plan.with_tile(tile)))
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::LocalTile { axis, factors });
            }
        }
        spaces
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
            self.local_tile_factors_for_axis_with_required(0, Some(Self::EXISTING_TILE.m));
        let n_factors =
            self.local_tile_factors_for_axis_with_required(1, Some(Self::EXISTING_TILE.n));
        let k_factors =
            self.local_tile_factors_for_axis_with_required(2, Some(Self::EXISTING_TILE.k));
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
