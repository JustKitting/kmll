use super::*;

impl GemmSearchProblem {
    pub fn candidate_for_tile(&self, tile: GemmTileShape) -> KernelCandidateMetadata {
        self.candidate_for_plan(GemmSchedulePlan::new(tile))
    }

    pub fn candidate_for_plan(&self, plan: GemmSchedulePlan) -> KernelCandidateMetadata {
        let plan = plan.normalized();
        let tile = plan.tile;
        let existing_tile = Self::is_existing_plan(plan);
        let a_order_suffix = plan.a_load_order.symbol_suffix();
        let b_order_suffix = plan.b_load_order.symbol_suffix();
        let per_thread_symbol_suffix = plan.per_thread_symbol_suffix();
        let per_thread_operation_suffix = plan.per_thread_operation_suffix();
        let load_unroll_symbol_suffix = plan.load_unroll_symbol_suffix();
        let load_unroll_operation_suffix = plan.load_unroll_operation_suffix();
        let load_thread_group_symbol_suffix = plan.load_thread_group_symbol_suffix();
        let load_thread_group_operation_suffix = plan.load_thread_group_operation_suffix();
        let thread_order_symbol_suffix = plan.thread_order_symbol_suffix();
        let thread_order_operation_suffix = plan.thread_order_operation_suffix();
        let symbol_hint = if plan.reduce_unroll == 1 {
            format!(
                "gemm_f32_bf16_tile_{}x{}x{}{}{}{}{}{}{}",
                tile.m,
                tile.n,
                tile.k,
                per_thread_symbol_suffix,
                load_unroll_symbol_suffix,
                load_thread_group_symbol_suffix,
                a_order_suffix,
                b_order_suffix,
                thread_order_symbol_suffix
            )
        } else {
            format!(
                "gemm_f32_bf16_tile_{}x{}x{}_u{}{}{}{}{}{}{}",
                tile.m,
                tile.n,
                tile.k,
                plan.reduce_unroll,
                per_thread_symbol_suffix,
                load_unroll_symbol_suffix,
                load_thread_group_symbol_suffix,
                a_order_suffix,
                b_order_suffix,
                thread_order_symbol_suffix
            )
        };
        let launch_kernel = if existing_tile {
            "gemm_f32_bf16_tiled_kernel".to_string()
        } else {
            symbol_hint.clone()
        };
        let launch = CudaLaunchSpec::new(
            launch_kernel,
            tile.grid_dim(self.m, self.n),
            plan.block_dim(),
            0,
        );
        let operation = TypedOperationSpec::new(
            if plan.reduce_unroll == 1 {
                format!(
                    "gemm-f32-bf16-{}x{}x{}{}{}{}{}{}{}",
                    tile.m,
                    tile.n,
                    tile.k,
                    per_thread_operation_suffix,
                    load_unroll_operation_suffix,
                    load_thread_group_operation_suffix,
                    a_order_suffix,
                    b_order_suffix,
                    thread_order_operation_suffix
                )
            } else {
                format!(
                    "gemm-f32-bf16-{}x{}x{}-u{}{}{}{}{}{}{}",
                    tile.m,
                    tile.n,
                    tile.k,
                    plan.reduce_unroll,
                    per_thread_operation_suffix,
                    load_unroll_operation_suffix,
                    load_thread_group_operation_suffix,
                    a_order_suffix,
                    b_order_suffix,
                    thread_order_operation_suffix
                )
            },
            OperationKind::Gemm,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(self.a_dtype, self.accumulator, [self.m, self.k])
                .with_layout("row-major"),
        )
        .with_input(
            TensorTypeSpec::new(self.b_dtype, self.accumulator, [self.k, self.n])
                .with_layout("column-major"),
        )
        .with_output(
            TensorTypeSpec::new(self.c_dtype, self.accumulator, [self.m, self.n])
                .with_layout("row-major"),
        )
        .with_launch(launch.clone());
        let materialization = if existing_tile {
            KernelMaterialization::Existing {
                symbol: "gemm_f32_bf16_tiled_kernel",
            }
        } else {
            KernelMaterialization::Generated {
                symbol: symbol_hint,
            }
        };
        let mut schedule = KernelSchedule::new().with_transform(ScheduleTransform::TileGemm {
            m: tile.m,
            n: tile.n,
            k: tile.k,
        });
        if plan.reduce_unroll > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 2,
                factor: plan.reduce_unroll,
            });
        }
        if plan.m_per_thread > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Upcast {
                axis: 0,
                factor: plan.m_per_thread,
            });
        }
        if plan.n_per_thread > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Upcast {
                axis: 1,
                factor: plan.n_per_thread,
            });
        }
        if plan.a_load_unroll > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 3,
                factor: plan.a_load_unroll,
            });
        }
        if plan.b_load_unroll > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 4,
                factor: plan.b_load_unroll,
            });
        }
        if plan.has_custom_a_load_thread_group() {
            schedule = schedule.with_transform(ScheduleTransform::ThreadGroup {
                axis: 3,
                factor: plan.a_load_thread_count(),
            });
        }
        if plan.has_custom_b_load_thread_group() {
            schedule = schedule.with_transform(ScheduleTransform::ThreadGroup {
                axis: 4,
                factor: plan.b_load_thread_count(),
            });
        }
        if plan.a_load_order == GemmATileLoadOrder::MContiguous {
            schedule = schedule.with_transform(ScheduleTransform::StrideOrder { axes: vec![0, 2] });
        }
        if plan.b_load_order == GemmBTileLoadOrder::KContiguous {
            schedule = schedule.with_transform(ScheduleTransform::StrideOrder { axes: vec![2, 1] });
        }
        if plan.thread_order == GemmThreadOrder::MThenN {
            schedule = schedule.with_transform(ScheduleTransform::Swap {
                axis_a: 0,
                axis_b: 1,
            });
        }

        let mut candidate = KernelCandidateMetadata::new(
            "gemm-f32-bf16-row-col-row",
            self.axes(),
            schedule,
            "tiled-gemm-generator",
            materialization,
            launch,
            operation,
        );
        candidate.resources = Some(plan.resource_usage());
        candidate
    }

    pub(in crate::autotune) fn axes(&self) -> Vec<KernelAxis> {
        vec![
            KernelAxis::spatial(0, "m", self.m, Some(self.k)),
            KernelAxis::spatial(1, "n", self.n, Some(1)),
            KernelAxis::reduction(2, "k", self.k, Some(1)),
        ]
    }

    pub(in crate::autotune) fn is_existing_plan(plan: GemmSchedulePlan) -> bool {
        let plan = plan.normalized();
        plan.tile == Self::EXISTING_TILE
            && plan.reduce_unroll == 1
            && plan.m_per_thread == 1
            && plan.n_per_thread == 1
            && plan.a_load_unroll == 1
            && plan.b_load_unroll == 1
            && !plan.has_custom_a_load_thread_group()
            && !plan.has_custom_b_load_thread_group()
            && plan.a_load_order == GemmATileLoadOrder::KContiguous
            && plan.b_load_order == GemmBTileLoadOrder::TileLinear
            && plan.thread_order == GemmThreadOrder::NThenM
    }

    pub(in crate::autotune) fn action_materialization_for_plan(
        plan: GemmSchedulePlan,
    ) -> KernelActionMaterialization {
        if Self::is_existing_plan(plan) {
            KernelActionMaterialization::Existing
        } else {
            KernelActionMaterialization::DeferredGenerated
        }
    }

    pub(in crate::autotune) fn plan_within_resource_limits(plan: GemmSchedulePlan) -> bool {
        let plan = plan.normalized();
        let resources = plan.resource_usage();
        plan.tile.is_launchable_shape()
            && resources.threads_per_block <= Self::MAX_THREADS_PER_BLOCK
            && resources.shared_memory_bytes <= Self::MAX_SHARED_MEMORY_BYTES
            && resources.accumulator_elements_per_thread
                <= Self::MAX_ACCUMULATOR_ELEMENTS_PER_THREAD
    }

    pub(in crate::autotune) fn candidate_for_checked_plan(
        &self,
        parent: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
        plan: GemmSchedulePlan,
    ) -> Option<KernelCandidateMetadata> {
        let plan = plan.normalized();
        Self::plan_within_resource_limits(plan)
            .then(|| candidate_with_action_trace(parent, action, self.candidate_for_plan(plan)))
    }
}
