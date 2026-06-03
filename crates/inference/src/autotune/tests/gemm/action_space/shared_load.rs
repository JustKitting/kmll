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

    assert_eq!(spaces.spaces.len(), 8);
    let KernelActionSpace::Split { variants } = &spaces.spaces[0] else {
        panic!("upcast GEMM should still expose one-axis retile split metadata");
    };
    assert_eq!(variants.len(), 12);
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        24,
        KernelActionMaterialization::DeferredGenerated
    )));
    let KernelActionSpace::Unroll {
        axis: reduce_axis, ..
    } = &spaces.spaces[1]
    else {
        panic!("upcast GEMM should still expose reduce unroll metadata");
    };
    assert_eq!(*reduce_axis, 2);
    let KernelActionSpace::Unroll {
        axis: a_load_axis,
        factors: a_load_factors,
    } = &spaces.spaces[2]
    else {
        panic!("upcast GEMM should expose A shared-load unroll metadata");
    };
    assert_eq!(*a_load_axis, 3);
    assert_eq!(a_load_factors, &[2]);
    let KernelActionSpace::Unroll {
        axis: b_load_axis,
        factors: b_load_factors,
    } = &spaces.spaces[3]
    else {
        panic!("upcast GEMM should expose B shared-load unroll metadata");
    };
    assert_eq!(*b_load_axis, 4);
    assert_eq!(b_load_factors, &[2, 3, 4]);
    let KernelActionSpace::ThreadGroup {
        axis: a_load_thread_axis,
        factors: a_load_thread_factors,
    } = &spaces.spaces[4]
    else {
        panic!("upcast GEMM should expose A shared-load thread-group metadata");
    };
    assert_eq!(*a_load_thread_axis, 3);
    assert_eq!(a_load_thread_factors, &[32, 64]);
    let KernelActionSpace::ThreadGroup {
        axis: b_load_thread_axis,
        factors: b_load_thread_factors,
    } = &spaces.spaces[5]
    else {
        panic!("upcast GEMM should expose B shared-load thread-group metadata");
    };
    assert_eq!(*b_load_thread_axis, 4);
    assert_eq!(b_load_thread_factors, &[32, 64]);
    assert!(matches!(spaces.spaces[6], KernelActionSpace::Swap { .. }));
    assert!(matches!(
        spaces.spaces[7],
        KernelActionSpace::StrideOrder { .. }
    ));

    assert!(actions.contains(&KernelScheduleAction::split(
        0,
        24,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(actions.contains(&KernelScheduleAction::unroll(3, 2)));
    assert!(!actions.contains(&KernelScheduleAction::unroll(3, 3)));
    assert!(actions.contains(&KernelScheduleAction::unroll(4, 4)));
    assert!(actions.contains(&KernelScheduleAction::thread_group(3, 32)));
    assert!(actions.contains(&KernelScheduleAction::thread_group(4, 64)));
    assert!(!actions.contains(&KernelScheduleAction::thread_group(3, 128)));
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
        .apply_schedule_action(
            &upcast_candidate,
            &KernelScheduleAction::thread_group(3, 32),
        )
        .expect("A shared-load thread-group should produce candidate metadata");
    let a_thread_grouped_plan = schedule_gemm_plan(&a_thread_grouped.schedule)
        .expect("A shared-load thread-grouped candidate should plan");
    assert_eq!(a_thread_grouped_plan.a_load_thread_count(), 32);
    assert_eq!(a_thread_grouped_plan.b_load_thread_count(), 128);
    assert_eq!(
        a_thread_grouped.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_atg32"
    );

    let b_thread_grouped = problem
        .apply_schedule_action(
            &upcast_candidate,
            &KernelScheduleAction::thread_group(4, 64),
        )
        .expect("B shared-load thread-group should produce candidate metadata");
    let b_thread_grouped_plan = schedule_gemm_plan(&b_thread_grouped.schedule)
        .expect("B shared-load thread-grouped candidate should plan");
    assert_eq!(b_thread_grouped_plan.a_load_thread_count(), 128);
    assert_eq!(b_thread_grouped_plan.b_load_thread_count(), 64);
    assert_eq!(
        b_thread_grouped.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2_btg64"
    );
}
