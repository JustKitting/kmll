use super::*;

impl KernelMetadataSearchProblem for MatvecSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new("matvec_bf16_kernel", (1, 1, 1), (32, 1, 1), 0);
        let operation = TypedOperationSpec::new(
            "matvec-bf16-row-major-seed",
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
            KernelSchedule::new(),
            "row-major-matvec-generator",
            KernelMaterialization::DeferredGenerated {
                symbol_hint: "matvec_bf16_kernel".to_string(),
                reason: "seed descriptor has no concrete schedule yet".to_string(),
            },
            launch,
            operation,
        )
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        expand_with_schedule_actions(self, candidate)
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let plan = schedule_matvec_plan(&candidate.schedule)?;
        let is_naive = matches!(
            &candidate.generated.materialization,
            KernelMaterialization::Generated { symbol } if symbol == "matvec_bf16_naive"
        );
        let rows_per_block = plan.rows.rows_per_block() as usize;
        let lanes_per_row = plan.thread_group.lanes_per_row() as usize;
        let row_upcast = plan.row_upcast.factor() as usize;
        let blocks = self.rows.div_ceil(rows_per_block);
        let padded_rows = blocks * rows_per_block;
        let useful_fma_ops = self.rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let wasted_rows = padded_rows.saturating_sub(self.rows);
        let wasted_fma_ops = wasted_rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let block_overhead = blocks as f64 * 2048.0;
        let unroll = f64::from(plan.reduce_unroll.max(1));
        let reduce_group_loops = if plan.has_custom_reduce_group() {
            self.cols
                .div_ceil(plan.reduce_group_size().max(1) as usize)
                .max(1)
        } else {
            1
        };
        let loop_overhead = blocks as f64
            * row_upcast as f64
            * (self.cols as f64 / lanes_per_row as f64).ceil()
            * 64.0
            / unroll;
        let reduce_group_overhead =
            blocks as f64 * row_upcast as f64 * reduce_group_loops as f64 * 24.0;
        let thread_overhead = blocks as f64 * f64::from(plan.block_threads()) * 8.0;
        let subgroup_pressure =
            blocks as f64 * (32.0_f64 / lanes_per_row as f64 - 1.0_f64).max(0.0_f64) * 256.0;
        let row_upcast_pressure =
            blocks as f64 * (row_upcast as f64 - 1.0_f64).max(0.0_f64) * 384.0;
        let register_pressure = blocks as f64 * (unroll - 1.0_f64).max(0.0_f64) * 32.0;
        let generic_runtime_penalty = if candidate.generated.materialization.is_existing() {
            blocks as f64 * 64.0
        } else {
            0.0
        };
        let naive_serial_penalty = if is_naive {
            useful_fma_ops * 16.0 + self.rows as f64 * 4096.0
        } else {
            0.0
        };
        SearchScore::heuristic(
            useful_fma_ops
                + wasted_fma_ops * 8.0
                + block_overhead
                + loop_overhead
                + reduce_group_overhead
                + thread_overhead
                + subgroup_pressure
                + row_upcast_pressure
                + register_pressure
                + generic_runtime_penalty
                + naive_serial_penalty,
        )
    }
}
