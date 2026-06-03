use super::*;

impl KernelMetadataSearchProblem for GemmSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new(
            "gemm_f32_bf16_tiled_kernel",
            TiledGemm16Plan::<RowMajor, ColumnMajor, RowMajor>::grid_dim(self.m, self.n),
            TiledGemm16Plan::<RowMajor, ColumnMajor, RowMajor>::block_dim(),
            0,
        );
        let operation = TypedOperationSpec::new(
            "gemm-f32-bf16-seed",
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
        KernelCandidateMetadata::new(
            "gemm-f32-bf16-row-col-row",
            self.axes(),
            KernelSchedule::new(),
            "tiled-gemm-generator",
            KernelMaterialization::DeferredGenerated {
                symbol_hint: "gemm_f32_bf16_tiled_kernel".to_string(),
                reason: "seed descriptor has no concrete tile schedule yet".to_string(),
            },
            launch,
            operation,
        )
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        expand_with_schedule_actions(self, candidate)
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let plan = schedule_gemm_plan(&candidate.schedule)?;
        let tile = plan.tile;
        let m_per_thread = plan.m_per_thread.max(1);
        let n_per_thread = plan.n_per_thread.max(1);
        let tile_m = tile.m as usize;
        let tile_n = tile.n as usize;
        let tile_k = tile.k as usize;
        let padded_m = self.m.div_ceil(tile_m).checked_mul(tile_m)?;
        let padded_n = self.n.div_ceil(tile_n).checked_mul(tile_n)?;
        let padded_k = self.k.div_ceil(tile_k).checked_mul(tile_k)?;
        let padded_fma_ops = padded_m
            .checked_mul(padded_n)?
            .checked_mul(padded_k)?
            .checked_mul(2)? as f64;
        let block_count = self
            .m
            .div_ceil(tile_m)
            .checked_mul(self.n.div_ceil(tile_n))? as f64;
        let unroll = f64::from(plan.reduce_unroll.max(1));
        let per_thread_work = f64::from(m_per_thread * n_per_thread);
        let thread_count_u32 = plan.thread_count();
        let thread_count = f64::from(thread_count_u32);
        let a_load_unroll = plan.a_load_unroll.max(1);
        let b_load_unroll = plan.b_load_unroll.max(1);
        let a_load_threads = plan.a_load_thread_count();
        let b_load_threads = plan.b_load_thread_count();
        let a_load_loop_rounds = plan.a_load_rounds().div_ceil(a_load_unroll);
        let b_load_loop_rounds = plan.b_load_rounds().div_ceil(b_load_unroll);
        let loop_overhead = block_count * 4096.0 / unroll / per_thread_work.sqrt();
        let thread_overhead = block_count * thread_count * 16.0;
        let load_loop_overhead =
            block_count * f64::from(a_load_loop_rounds + b_load_loop_rounds) * 256.0;
        let average_load_threads = (a_load_threads + b_load_threads) / 2;
        let load_thread_penalty =
            block_count * f64::from(thread_count_u32.saturating_sub(average_load_threads)) * 8.0;
        let register_pressure =
            block_count * ((unroll - 1.0) * 256.0 + (per_thread_work - 1.0) * 1024.0);
        let load_unroll_pressure =
            block_count * f64::from(a_load_unroll + b_load_unroll - 2) * 128.0;
        let threads_m = tile.m.div_ceil(m_per_thread);
        let threads_n = tile.n.div_ceil(n_per_thread);
        let thread_order_penalty = match plan.thread_order {
            GemmThreadOrder::NThenM if threads_n < 16 && threads_m >= 16 => block_count * 128.0,
            GemmThreadOrder::MThenN if threads_n >= 16 => block_count * 96.0,
            _ => 0.0,
        };
        let a_load_penalty = match plan.a_load_order {
            GemmATileLoadOrder::KContiguous => block_count * 64.0,
            GemmATileLoadOrder::MContiguous => block_count * 512.0,
        };
        let b_load_penalty = match plan.b_load_order {
            GemmBTileLoadOrder::TileLinear => block_count * 512.0,
            GemmBTileLoadOrder::KContiguous => block_count * 64.0,
        };
        SearchScore::heuristic(
            padded_fma_ops
                + loop_overhead
                + thread_overhead
                + load_loop_overhead
                + load_thread_penalty
                + register_pressure
                + load_unroll_pressure
                + thread_order_penalty
                + a_load_penalty
                + b_load_penalty,
        )
    }
}
