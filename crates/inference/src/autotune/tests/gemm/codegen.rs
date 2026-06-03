use super::*;

#[test]
fn gemm_generator_renders_tile_specific_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_tile(GemmTileShape::new(13, 24, 13));
    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render deferred tile source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_13x24x13");
    assert!(generated.source.contains("#[kernel]"));
    assert!(
        generated
            .source
            .contains("pub fn gemm_f32_bf16_tile_13x24x13(")
    );
    assert!(generated.source.contains("pub struct Bf16(u16);"));
    assert!(generated.source.contains(".to_f32()"));
    assert!(generated.source.contains("const TILE_M: usize = 13;"));
    assert!(generated.source.contains("const TILE_N: usize = 24;"));
    assert!(generated.source.contains("const TILE_K: usize = 13;"));
    assert!(generated.source.contains("while load < TILE_A_ELEMS"));
    assert!(generated.source.contains("while load < TILE_B_ELEMS"));
}

#[test]
fn gemm_search_expands_tile_metadata_into_reduce_unroll_variants() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
    let candidates = problem.expand(&tile_candidate);
    assert_eq!(candidates.len(), 42);

    let m_split = candidates
        .iter()
        .find(|candidate| {
            schedule_gemm_tile(&candidate.schedule) == Some(GemmTileShape::new(24, 32, 16))
        })
        .expect("GEMM search should expose an M-axis split descriptor");
    assert_eq!(m_split.launch.kernel, "gemm_f32_bf16_tile_24x32x16");
    assert_eq!(
        m_split.action_trace,
        vec![KernelScheduleAction::split(
            0,
            24,
            KernelActionMaterialization::DeferredGenerated
        )]
    );

    let unroll7 = candidates
        .iter()
        .find(|candidate| schedule_gemm_reduce_unroll(&candidate.schedule) == Some(7))
        .expect("GEMM search should expose a reduce unroll factor 7 descriptor");
    assert_eq!(
        schedule_gemm_tile(&unroll7.schedule),
        Some(GemmTileShape::new(16, 32, 16))
    );
    assert_ne!(tile_candidate.artifact_key(), unroll7.artifact_key());
    assert_eq!(unroll7.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u7");
    assert!(!unroll7.is_launchable());
    assert_eq!(
        unroll7.generated.materialization,
        KernelMaterialization::Generated {
            symbol: "gemm_f32_bf16_tile_16x32x16_u7".to_string()
        }
    );
}

#[test]
fn gemm_search_expands_tile_metadata_into_2d_upcast_variants() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
    let candidates = problem.expand(&tile_candidate);

    let m_upcast = candidates
        .iter()
        .find(|candidate| schedule_gemm_m_per_thread(&candidate.schedule) == Some(2))
        .expect("GEMM search should expose an M upcast factor 2 descriptor");
    let m_plan = schedule_gemm_plan(&m_upcast.schedule).expect("candidate should have GEMM plan");
    assert_eq!(m_plan.tile, GemmTileShape::new(16, 32, 16));
    assert_eq!(m_plan.m_per_thread, 2);
    assert_eq!(m_plan.n_per_thread, 1);
    assert_ne!(tile_candidate.artifact_key(), m_upcast.artifact_key());
    assert_eq!(m_upcast.launch.kernel, "gemm_f32_bf16_tile_16x32x16_mt2");
    assert_eq!(m_upcast.launch.block_dim.x, 32);
    assert_eq!(m_upcast.launch.block_dim.y, 8);
    assert_eq!(m_upcast.launch.block_dim.z, 1);
    assert!(m_upcast.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::Upcast { axis: 0, factor: 2 })
    }));

    let n_upcast = candidates
        .iter()
        .find(|candidate| schedule_gemm_n_per_thread(&candidate.schedule) == Some(2))
        .expect("GEMM search should expose an N upcast factor 2 descriptor");
    let plan = schedule_gemm_plan(&n_upcast.schedule).expect("candidate should have GEMM plan");
    assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
    assert_eq!(plan.m_per_thread, 1);
    assert_eq!(plan.n_per_thread, 2);
    assert_eq!(plan.reduce_unroll, 1);
    assert_ne!(tile_candidate.artifact_key(), n_upcast.artifact_key());
    assert_eq!(n_upcast.launch.kernel, "gemm_f32_bf16_tile_16x32x16_nt2");
    assert_eq!(n_upcast.launch.block_dim.x, 16);
    assert_eq!(n_upcast.launch.block_dim.y, 16);
    assert_eq!(n_upcast.launch.block_dim.z, 1);
    assert!(n_upcast.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::Upcast { axis: 1, factor: 2 })
    }));
}

#[test]
fn gemm_search_expands_upcast_metadata_into_shared_load_unroll_variants() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let upcast_candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
            .with_m_per_thread(2)
            .with_n_per_thread(2),
    );
    let candidates = problem.expand(&upcast_candidate);

    let a_load_unrolled = candidates
        .iter()
        .find(|candidate| schedule_gemm_a_load_unroll(&candidate.schedule) == Some(2))
        .expect("GEMM search should expose A shared-load unroll factor 2 metadata");
    let a_plan = schedule_gemm_plan(&a_load_unrolled.schedule)
        .expect("A load-unrolled candidate should have GEMM plan");
    assert_eq!(a_plan.tile, GemmTileShape::new(16, 32, 16));
    assert_eq!(a_plan.m_per_thread, 2);
    assert_eq!(a_plan.n_per_thread, 2);
    assert_eq!(a_plan.a_load_unroll, 2);
    assert_eq!(a_plan.b_load_unroll, 1);
    assert_eq!(
        a_load_unrolled.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_au2"
    );
    assert!(a_load_unrolled.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::Unroll { axis: 3, factor: 2 })
    }));

    let b_load_unrolled = candidates
        .iter()
        .find(|candidate| schedule_gemm_b_load_unroll(&candidate.schedule) == Some(4))
        .expect("GEMM search should expose B shared-load unroll factor 4 metadata");
    let b_plan = schedule_gemm_plan(&b_load_unrolled.schedule)
        .expect("B load-unrolled candidate should have GEMM plan");
    assert_eq!(b_plan.tile, GemmTileShape::new(16, 32, 16));
    assert_eq!(b_plan.m_per_thread, 2);
    assert_eq!(b_plan.n_per_thread, 2);
    assert_eq!(b_plan.a_load_unroll, 1);
    assert_eq!(b_plan.b_load_unroll, 4);
    assert_eq!(
        b_load_unrolled.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_bu4"
    );
    assert!(b_load_unrolled.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::Unroll { axis: 4, factor: 4 })
    }));
}

#[test]
fn gemm_search_expands_tile_metadata_into_a_load_stride_order() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
    let candidates = problem.expand(&tile_candidate);

    let m_contiguous = candidates
        .iter()
        .find(|candidate| {
            schedule_gemm_plan(&candidate.schedule)
                .map(|plan| plan.a_load_order == GemmATileLoadOrder::MContiguous)
                .unwrap_or(false)
        })
        .expect("GEMM search should expose M-contiguous A load order metadata");
    let plan = schedule_gemm_plan(&m_contiguous.schedule).expect("candidate should have GEMM plan");
    assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
    assert_eq!(plan.reduce_unroll, 1);
    assert_eq!(plan.a_load_order, GemmATileLoadOrder::MContiguous);
    assert_eq!(plan.b_load_order, GemmBTileLoadOrder::TileLinear);
    assert_ne!(tile_candidate.artifact_key(), m_contiguous.artifact_key());
    assert_eq!(m_contiguous.launch.kernel, "gemm_f32_bf16_tile_16x32x16_am");
    assert!(m_contiguous.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::StrideOrder { axes } if axes == &[0, 2])
    }));
}

#[test]
fn gemm_search_expands_tile_metadata_into_b_load_stride_order() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
    let candidates = problem.expand(&tile_candidate);

    let k_contiguous = candidates
        .iter()
        .find(|candidate| {
            schedule_gemm_b_load_order(&candidate.schedule) == GemmBTileLoadOrder::KContiguous
        })
        .expect("GEMM search should expose K-contiguous B load order metadata");
    let plan = schedule_gemm_plan(&k_contiguous.schedule).expect("candidate should have GEMM plan");
    assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
    assert_eq!(plan.reduce_unroll, 1);
    assert_eq!(plan.b_load_order, GemmBTileLoadOrder::KContiguous);
    assert_ne!(tile_candidate.artifact_key(), k_contiguous.artifact_key());
    assert_eq!(k_contiguous.launch.kernel, "gemm_f32_bf16_tile_16x32x16_bk");
    assert!(k_contiguous.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::StrideOrder { axes } if axes == &[2, 1])
    }));
}

#[test]
fn gemm_generator_renders_reduce_unroll_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16)).with_reduce_unroll(7),
    );
    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render reduce-unrolled source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_u7");
    assert!(generated.source.contains("const REDUCE_UNROLL: usize = 7;"));
    assert!(
        generated
            .source
            .contains("while kk + REDUCE_UNROLL <= TILE_K")
    );
    assert!(
        generated
            .source
            .contains("TILE_A[tile_row0 * TILE_K + kk + 6]")
    );
    assert!(
        generated
            .source
            .contains("TILE_B[(kk + 6) * TILE_N + tile_col0]")
    );
    assert!(generated.source.contains("while kk < TILE_K"));
}

#[test]
fn gemm_generator_renders_shared_load_unroll_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::tile_gemm(
                16,
                32,
                16,
                KernelActionMaterialization::DeferredGenerated,
            ),
            KernelScheduleAction::upcast(0, 2),
            KernelScheduleAction::upcast(1, 2),
            KernelScheduleAction::unroll(3, 2),
            KernelScheduleAction::unroll(4, 2),
        ],
    )
    .expect("valid shared-load-unroll GEMM action trace should replay");
    let plan =
        schedule_gemm_plan(&candidate.schedule).expect("load-unrolled candidate should plan");
    assert_eq!(plan.m_per_thread, 2);
    assert_eq!(plan.n_per_thread, 2);
    assert_eq!(plan.a_load_unroll, 2);
    assert_eq!(plan.b_load_unroll, 2);

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render shared-load-unrolled source");

    assert_eq!(
        generated.symbol,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_au2_bu2"
    );
    assert!(generated.source.contains("const A_LOAD_UNROLL: usize = 2;"));
    assert!(generated.source.contains("const B_LOAD_UNROLL: usize = 2;"));
    assert!(
        generated
            .source
            .contains("let a_load1 = load + A_LOAD_THREADS;")
    );
    assert!(generated.source.contains("if a_load1 < TILE_A_ELEMS"));
    assert!(
        generated
            .source
            .contains("load += A_LOAD_THREADS * A_LOAD_UNROLL;")
    );
    assert!(
        generated
            .source
            .contains("let b_load1 = load + B_LOAD_THREADS;")
    );
    assert!(generated.source.contains("if b_load1 < TILE_B_ELEMS"));
    assert!(
        generated
            .source
            .contains("load += B_LOAD_THREADS * B_LOAD_UNROLL;")
    );
}

#[test]
fn gemm_generator_renders_shared_load_group_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::tile_gemm(
                16,
                32,
                16,
                KernelActionMaterialization::DeferredGenerated,
            ),
            KernelScheduleAction::upcast(0, 2),
            KernelScheduleAction::upcast(1, 2),
            KernelScheduleAction::group(3, 32),
            KernelScheduleAction::group(4, 64),
        ],
    )
    .expect("valid shared-load-group GEMM action trace should replay");
    let plan = schedule_gemm_plan(&candidate.schedule).expect("load-grouped candidate should plan");
    assert_eq!(plan.a_load_thread_count(), 32);
    assert_eq!(plan.b_load_thread_count(), 64);

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render shared-load-grouped source");

    assert_eq!(
        generated.symbol,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_atg32_btg64"
    );
    assert!(
        generated
            .source
            .contains("const A_LOAD_THREADS: usize = 32;")
    );
    assert!(
        generated
            .source
            .contains("const B_LOAD_THREADS: usize = 64;")
    );
    assert!(
        generated
            .source
            .contains("let mut load = if tid < A_LOAD_THREADS { tid } else { TILE_A_ELEMS };")
    );
    assert!(
        generated
            .source
            .contains("load = if tid < B_LOAD_THREADS { tid } else { TILE_B_ELEMS };")
    );
    assert!(generated.source.contains("load += A_LOAD_THREADS;"));
    assert!(generated.source.contains("load += B_LOAD_THREADS;"));
}

#[test]
fn gemm_generator_renders_a_load_stride_order_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
            .with_a_load_order(GemmATileLoadOrder::MContiguous),
    );
    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render A stride-order source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_am");
    assert!(generated.source.contains("let tile_row = load % TILE_M;"));
    assert!(generated.source.contains("let tile_col = load / TILE_M;"));
    assert!(
        generated
            .source
            .contains("let a_smem_index = tile_row * TILE_K + tile_col;")
    );
    assert!(generated.source.contains("TILE_A[a_smem_index] = if"));
}

#[test]
fn gemm_generator_renders_b_load_stride_order_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
            .with_b_load_order(GemmBTileLoadOrder::KContiguous),
    );
    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render stride-order source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_bk");
    assert!(generated.source.contains("let tile_row = load % TILE_K;"));
    assert!(generated.source.contains("let tile_col = load / TILE_K;"));
    assert!(
        generated
            .source
            .contains("let b_smem_index = tile_row * TILE_N + tile_col;")
    );
    assert!(generated.source.contains("TILE_B[b_smem_index] = if"));
}

#[test]
fn gemm_generator_renders_thread_axis_swap_source_on_demand() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = replay_schedule_actions(
        &problem,
        &[
            KernelScheduleAction::tile_gemm(
                16,
                32,
                16,
                KernelActionMaterialization::DeferredGenerated,
            ),
            KernelScheduleAction::swap(0, 1),
        ],
    )
    .expect("valid swapped GEMM action trace should replay");
    let plan = schedule_gemm_plan(&candidate.schedule).expect("swapped candidate should plan");
    assert_eq!(plan.thread_order, GemmThreadOrder::MThenN);
    assert_eq!(candidate.launch.block_dim.x, 16);
    assert_eq!(candidate.launch.block_dim.y, 32);

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render thread-axis-swapped source");

    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_sw01");
    assert!(
        generated
            .source
            .contains("let thread_m = thread::threadIdx_x() as usize;")
    );
    assert!(
        generated
            .source
            .contains("let thread_n = thread::threadIdx_y() as usize;")
    );
    assert!(
        generated
            .source
            .contains("if thread_m >= THREADS_M || thread_n >= THREADS_N")
    );
    assert!(
        generated
            .source
            .contains("let tid = thread_n * THREADS_M + thread_m;")
    );
    assert!(
        generated
            .source
            .contains("let tile_row0 = thread_m * THREAD_TILE_M;")
    );
    assert!(
        generated
            .source
            .contains("let tile_col0 = thread_n * THREAD_TILE_N;")
    );
}
