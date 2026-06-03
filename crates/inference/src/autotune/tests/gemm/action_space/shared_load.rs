use super::*;

#[test]
fn gemm_action_space_exposes_shared_load_unroll_after_local_tiling() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let upcast_candidate = problem.candidate_for_plan(
        GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
            .with_m_per_thread(2)
            .with_n_per_thread(2),
    );
    let spaces = problem.action_spaces(&upcast_candidate);
    let actions = spaces.actions();

    assert_eq!(spaces.spaces.len(), 10);
    let KernelActionSpace::LocalTile {
        axis: m_tile_axis,
        factors: m_tile_factors,
    } = &spaces.spaces[0]
    else {
        panic!("upcast GEMM should expose M-axis local-tile metadata");
    };
    assert_eq!(*m_tile_axis, 0);
    assert_eq!(m_tile_factors, &[2, 3, 4, 8, 13, 24, 29, 32]);
    let KernelActionSpace::LocalTile {
        axis: n_tile_axis,
        factors: n_tile_factors,
    } = &spaces.spaces[1]
    else {
        panic!("upcast GEMM should expose N-axis local-tile metadata");
    };
    assert_eq!(*n_tile_axis, 1);
    assert_eq!(n_tile_factors, &[2, 3, 4, 8, 13, 16, 24, 29]);
    let KernelActionSpace::LocalTile {
        axis: k_tile_axis,
        factors: k_tile_factors,
    } = &spaces.spaces[2]
    else {
        panic!("upcast GEMM should expose K-axis local-tile metadata");
    };
    assert_eq!(*k_tile_axis, 2);
    assert_eq!(k_tile_factors, &[2, 3, 4, 8, 13, 24, 29, 32]);
    let KernelActionSpace::Unroll {
        axis: reduce_axis, ..
    } = &spaces.spaces[3]
    else {
        panic!("upcast GEMM should still expose reduce unroll metadata");
    };
    assert_eq!(*reduce_axis, 2);
    let KernelActionSpace::Unroll {
        axis: a_load_axis,
        factors: a_load_factors,
    } = &spaces.spaces[4]
    else {
        panic!("upcast GEMM should expose A shared-load unroll metadata");
    };
    assert_eq!(*a_load_axis, 3);
    assert_eq!(a_load_factors, &[2]);
    let KernelActionSpace::Unroll {
        axis: b_load_axis,
        factors: b_load_factors,
    } = &spaces.spaces[5]
    else {
        panic!("upcast GEMM should expose B shared-load unroll metadata");
    };
    assert_eq!(*b_load_axis, 4);
    assert_eq!(b_load_factors, &[2, 3, 4]);
    let KernelActionSpace::Group {
        axis: a_load_thread_axis,
        factors: a_load_thread_factors,
    } = &spaces.spaces[6]
    else {
        panic!("upcast GEMM should expose A shared-load group metadata");
    };
    assert_eq!(*a_load_thread_axis, 3);
    assert_eq!(a_load_thread_factors, &[32, 64]);
    let KernelActionSpace::Group {
        axis: b_load_thread_axis,
        factors: b_load_thread_factors,
    } = &spaces.spaces[7]
    else {
        panic!("upcast GEMM should expose B shared-load group metadata");
    };
    assert_eq!(*b_load_thread_axis, 4);
    assert_eq!(b_load_thread_factors, &[32, 64]);
    assert!(matches!(spaces.spaces[8], KernelActionSpace::Swap { .. }));
    assert!(matches!(
        spaces.spaces[9],
        KernelActionSpace::StrideOrder { .. }
    ));

    assert_eq!(actions.len(), 50);
    assert!(actions.contains(&KernelScheduleAction::local_tile(0, 24)));
    assert!(actions.contains(&KernelScheduleAction::local_tile(1, 16)));
    assert!(actions.contains(&KernelScheduleAction::local_tile(2, 32)));
    assert!(actions.contains(&KernelScheduleAction::unroll(3, 2)));
    assert!(!actions.contains(&KernelScheduleAction::unroll(3, 3)));
    assert!(actions.contains(&KernelScheduleAction::unroll(4, 4)));
    assert!(actions.contains(&KernelScheduleAction::group(3, 32)));
    assert!(actions.contains(&KernelScheduleAction::group(4, 64)));
    assert!(!actions.contains(&KernelScheduleAction::group(3, 128)));
    assert!(actions.contains(&KernelScheduleAction::swap(0, 1)));

    let a_unrolled = problem
        .apply_schedule_action(&upcast_candidate, &KernelScheduleAction::unroll(3, 2))
        .expect("A shared-load unroll should produce candidate metadata");
    let a_unrolled_plan =
        schedule_gemm_plan(&a_unrolled.schedule).expect("A load-unrolled candidate should plan");
    assert_eq!(a_unrolled_plan.a_load_unroll, 2);
    assert_eq!(a_unrolled_plan.b_load_unroll, 1);
    assert_eq!(
        a_unrolled.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_au2"
    );

    let b_unrolled = problem
        .apply_schedule_action(&upcast_candidate, &KernelScheduleAction::unroll(4, 4))
        .expect("B shared-load unroll should produce candidate metadata");
    let b_unrolled_plan =
        schedule_gemm_plan(&b_unrolled.schedule).expect("B load-unrolled candidate should plan");
    assert_eq!(b_unrolled_plan.a_load_unroll, 1);
    assert_eq!(b_unrolled_plan.b_load_unroll, 4);
    assert_eq!(
        b_unrolled.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_bu4"
    );

    let a_thread_grouped = problem
        .apply_schedule_action(&upcast_candidate, &KernelScheduleAction::group(3, 32))
        .expect("A shared-load group should produce candidate metadata");
    let a_thread_grouped_plan = schedule_gemm_plan(&a_thread_grouped.schedule)
        .expect("A shared-load thread-grouped candidate should plan");
    assert_eq!(a_thread_grouped_plan.a_load_thread_count(), 32);
    assert_eq!(a_thread_grouped_plan.b_load_thread_count(), 128);
    assert_eq!(
        a_thread_grouped.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_atg32"
    );

    let b_thread_grouped = problem
        .apply_schedule_action(&upcast_candidate, &KernelScheduleAction::group(4, 64))
        .expect("B shared-load group should produce candidate metadata");
    let b_thread_grouped_plan = schedule_gemm_plan(&b_thread_grouped.schedule)
        .expect("B shared-load thread-grouped candidate should plan");
    assert_eq!(b_thread_grouped_plan.a_load_thread_count(), 128);
    assert_eq!(b_thread_grouped_plan.b_load_thread_count(), 64);
    assert_eq!(
        b_thread_grouped.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_btg64"
    );
}
