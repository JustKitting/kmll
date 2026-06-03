use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::autotune::matvec::problem) enum MatvecRowGroupingTransform {
    Split,
    GroupTop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::autotune::matvec::problem) enum MatvecReduceGroupingTransform {
    ThreadGroup,
    Group,
}

impl MatvecSearchProblem {
    pub fn candidate_for_rows(&self, plan: RowMajorWarpRows) -> KernelCandidateMetadata {
        self.candidate_for_plan_with_materialization(
            MatvecSchedulePlan::new(plan),
            "matvec_bf16_kernel".to_string(),
            KernelMaterialization::Existing {
                symbol: "matvec_bf16_kernel",
            },
        )
    }

    pub fn generated_candidate_for_rows(&self, plan: RowMajorWarpRows) -> KernelCandidateMetadata {
        self.generated_candidate_for_plan(MatvecSchedulePlan::new(plan))
    }

    pub fn generated_candidate_for_row_split(
        &self,
        rows: MatvecRowSplit,
    ) -> KernelCandidateMetadata {
        self.generated_candidate_for_plan(MatvecSchedulePlan::new(rows))
    }

    pub(in crate::autotune::matvec::problem) fn generated_candidate_for_row_group_top(
        &self,
        rows: MatvecRowSplit,
    ) -> KernelCandidateMetadata {
        self.generated_candidate_for_plan_with_grouping(
            MatvecSchedulePlan::new(rows),
            MatvecRowGroupingTransform::GroupTop,
            MatvecReduceGroupingTransform::ThreadGroup,
        )
    }

    pub fn generated_candidate_for_plan(
        &self,
        plan: MatvecSchedulePlan,
    ) -> KernelCandidateMetadata {
        self.generated_candidate_for_plan_with_grouping(
            plan,
            MatvecRowGroupingTransform::Split,
            MatvecReduceGroupingTransform::ThreadGroup,
        )
    }

    pub(in crate::autotune::matvec::problem) fn generated_candidate_for_plan_with_grouping(
        &self,
        plan: MatvecSchedulePlan,
        row_grouping: MatvecRowGroupingTransform,
        reduce_grouping: MatvecReduceGroupingTransform,
    ) -> KernelCandidateMetadata {
        let plan = plan.normalized();
        let symbol_hint = matvec_symbol_hint(plan);
        self.candidate_for_plan_with_materialization_and_grouping(
            plan,
            symbol_hint.clone(),
            KernelMaterialization::Generated {
                symbol: symbol_hint,
            },
            row_grouping,
            reduce_grouping,
        )
    }

    pub(in crate::autotune::matvec::problem) fn candidate_for_plan_with_materialization(
        &self,
        plan: MatvecSchedulePlan,
        launch_kernel: String,
        materialization: KernelMaterialization,
    ) -> KernelCandidateMetadata {
        self.candidate_for_plan_with_materialization_and_grouping(
            plan,
            launch_kernel,
            materialization,
            MatvecRowGroupingTransform::Split,
            MatvecReduceGroupingTransform::ThreadGroup,
        )
    }

    pub(in crate::autotune::matvec::problem) fn candidate_for_plan_with_materialization_and_grouping(
        &self,
        plan: MatvecSchedulePlan,
        launch_kernel: String,
        materialization: KernelMaterialization,
        row_grouping: MatvecRowGroupingTransform,
        reduce_grouping: MatvecReduceGroupingTransform,
    ) -> KernelCandidateMetadata {
        let plan = plan.normalized();
        let rows = plan.rows;
        let rows_per_block = rows.rows_per_block();
        let row_transform = match row_grouping {
            MatvecRowGroupingTransform::Split => ScheduleTransform::Split {
                axis: 0,
                factor: rows_per_block,
            },
            MatvecRowGroupingTransform::GroupTop => ScheduleTransform::GroupTop {
                axis: 0,
                factor: rows_per_block,
            },
        };
        let mut schedule = KernelSchedule::new()
            .with_transform(row_transform)
            .with_transform(ScheduleTransform::ThreadGroup {
                axis: 0,
                factor: plan.block_threads(),
            });
        if !plan.row_upcast.is_default() {
            schedule = schedule.with_transform(ScheduleTransform::Upcast {
                axis: 0,
                factor: plan.row_upcast.factor(),
            });
        }
        if !plan.thread_group.is_default() {
            let reduce_grouping_transform = match reduce_grouping {
                MatvecReduceGroupingTransform::ThreadGroup => ScheduleTransform::ThreadGroup {
                    axis: 1,
                    factor: plan.thread_group.lanes_per_row(),
                },
                MatvecReduceGroupingTransform::Group => ScheduleTransform::Group {
                    axis: 1,
                    factor: plan.thread_group.lanes_per_row(),
                },
            };
            schedule = schedule.with_transform(reduce_grouping_transform);
        }
        if plan.reduce_unroll != MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 1,
                factor: plan.reduce_unroll,
            });
        }
        if plan.has_custom_reduce_group() {
            schedule = schedule.with_transform(ScheduleTransform::GroupTop {
                axis: 1,
                factor: plan.reduce_group_size(),
            });
        }
        if !plan.loop_order.is_default() {
            schedule = schedule.with_transform(ScheduleTransform::StrideOrder {
                axes: plan.loop_order.action_axes().to_vec(),
            });
        }
        let launch = CudaLaunchSpec::new(
            launch_kernel,
            (rows.grid_rows(self.rows), 1, 1),
            (plan.block_threads(), 1, 1),
            0,
        );
        let operation = TypedOperationSpec::new(
            matvec_operation_name(plan),
            OperationKind::Matvec,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(self.input_dtype, self.accumulator, [self.cols])
                .with_layout("contiguous"),
        )
        .with_input(
            TensorTypeSpec::new(self.weight_dtype, self.accumulator, [self.rows, self.cols])
                .with_layout("row-major"),
        )
        .with_output(
            TensorTypeSpec::new(self.accumulator, self.accumulator, [self.rows])
                .with_layout("contiguous"),
        )
        .with_launch(launch.clone());

        KernelCandidateMetadata::new(
            "matvec-bf16-row-major",
            self.axes(),
            schedule,
            "row-major-matvec-generator",
            materialization,
            launch,
            operation,
        )
    }

    pub(in crate::autotune::matvec::problem) fn axes(&self) -> Vec<KernelAxis> {
        vec![
            KernelAxis::spatial(0, "row", self.rows, Some(self.cols)),
            KernelAxis::reduction(1, "col", self.cols, Some(1)),
        ]
    }
}
