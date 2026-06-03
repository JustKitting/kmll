use super::*;

#[test]
fn action_trace_replay_reconstructs_generated_gemm_candidate() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let actions = vec![
        KernelScheduleAction::tile_gemm(16, 32, 16, KernelActionMaterialization::DeferredGenerated),
        KernelScheduleAction::unroll(2, 4),
        KernelScheduleAction::stride_order(vec![2, 1]),
    ];

    let candidate = replay_schedule_actions(&problem, &actions)
        .expect("valid GEMM action trace should replay into candidate metadata");
    let plan = schedule_gemm_plan(&candidate.schedule).expect("replayed GEMM should have plan");

    assert_eq!(candidate.family, "gemm-f32-bf16-row-col-row");
    assert_eq!(candidate.action_trace, actions);
    assert_eq!(candidate.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u4_bk");
    assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
    assert_eq!(plan.reduce_unroll, 4);
    assert_eq!(plan.b_load_order, GemmBTileLoadOrder::KContiguous);
    assert!(!candidate.is_launchable());

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("replayed generated GEMM candidate should render source on demand");
    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_u4_bk");
    assert!(generated.source.contains("const TILE_M: usize = 16;"));
    assert!(generated.source.contains("const TILE_N: usize = 32;"));
    assert!(generated.source.contains("const REDUCE_UNROLL: usize = 4;"));
    assert!(generated.source.contains("let tile_row = load % TILE_K;"));
}

#[test]
fn action_trace_replay_reconstructs_2d_upcast_gemm_candidate() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let actions = vec![
        KernelScheduleAction::tile_gemm(16, 32, 16, KernelActionMaterialization::DeferredGenerated),
        KernelScheduleAction::upcast(0, 2),
        KernelScheduleAction::upcast(1, 2),
    ];

    let candidate = replay_schedule_actions(&problem, &actions)
        .expect("valid GEMM 2D upcast trace should replay into candidate metadata");
    let plan = schedule_gemm_plan(&candidate.schedule).expect("upcast GEMM should have plan");

    assert_eq!(candidate.family, "gemm-f32-bf16-row-col-row");
    assert_eq!(candidate.action_trace, actions);
    assert_eq!(
        candidate.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_mt2_nt2"
    );
    assert_eq!(candidate.launch.block_dim.x, 16);
    assert_eq!(candidate.launch.block_dim.y, 8);
    assert_eq!(candidate.launch.block_dim.z, 1);
    assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
    assert_eq!(plan.m_per_thread, 2);
    assert_eq!(plan.n_per_thread, 2);
    assert!(candidate.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::Upcast { axis: 0, factor: 2 })
    }));
    assert!(candidate.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::Upcast { axis: 1, factor: 2 })
    }));

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("2D upcast generated GEMM candidate should render source");
    assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_mt2_nt2");
    assert!(generated.source.contains("const THREAD_TILE_M: usize = 2;"));
    assert!(generated.source.contains("const THREAD_TILE_N: usize = 2;"));
    assert!(
        generated
            .source
            .contains("let tile_row1 = thread_m * THREAD_TILE_M + 1;")
    );
    assert!(
        generated
            .source
            .contains("let tile_col1 = thread_n * THREAD_TILE_N + 1;")
    );
    assert!(generated.source.contains("let mut acc3 = 0.0_f32;"));
    assert!(generated.source.contains("TILE_A[tile_row1 * TILE_K + kk]"));
    assert!(generated.source.contains("TILE_B[kk * TILE_N + tile_col1]"));
    assert!(
        generated
            .source
            .contains("*c_elem = alpha * acc3 + beta * current;")
    );
}

#[test]
fn gemm_describes_deferred_tiles_without_storing_kernel_payloads() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let actions = vec![
        KernelScheduleAction::split(0, 13, KernelActionMaterialization::DeferredGenerated),
        KernelScheduleAction::split(1, 24, KernelActionMaterialization::DeferredGenerated),
        KernelScheduleAction::split(2, 13, KernelActionMaterialization::DeferredGenerated),
    ];
    let deferred = replay_schedule_actions(&problem, &actions)
        .expect("GEMM split trace should expose arbitrary deferred generated tile metadata");
    assert_eq!(
        schedule_gemm_tile(&deferred.schedule),
        Some(GemmTileShape::new(13, 24, 13))
    );
    assert!(!deferred.is_launchable());
    assert_eq!(deferred.launch.kernel, "gemm_f32_bf16_tile_13x24x13");
    assert!(matches!(
        deferred.generated.materialization,
        KernelMaterialization::DeferredGenerated { .. }
    ));
    assert_eq!(deferred.action_trace, actions);
    assert_eq!(deferred.generated.generator, "tiled-gemm-generator");
}

#[test]
fn gemm_action_space_exposes_tile_unroll_upcast_and_stride_metadata() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let seed = problem.seed();
    let full_space = problem.search_space();
    assert_eq!(full_space.spaces.len(), 11);
    let KernelActionSpace::Split { variants } = &full_space.spaces[0] else {
        panic!("GEMM global action space should expose per-axis split metadata");
    };
    assert_eq!(variants.len(), 15);
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        13,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        1,
        24,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        2,
        32,
        KernelActionMaterialization::DeferredGenerated
    )));
    let KernelActionSpace::TileGemm { variants } = &full_space.spaces[1] else {
        panic!("GEMM global action space should expose existing tile materialization metadata");
    };
    assert_eq!(
        variants,
        &[KernelTile3dAction::new(
            KernelTile3d::new(16, 16, 16),
            KernelActionMaterialization::Existing
        )]
    );
    assert!(matches!(
        full_space.spaces[2],
        KernelActionSpace::Unroll { .. }
    ));
    assert!(matches!(
        full_space.spaces[3],
        KernelActionSpace::Upcast { .. }
    ));
    assert!(matches!(
        full_space.spaces[4],
        KernelActionSpace::Upcast { .. }
    ));
    let KernelActionSpace::Unroll {
        axis: a_load_axis,
        factors: a_load_factors,
    } = &full_space.spaces[5]
    else {
        panic!("GEMM global action space should expose A shared-load unroll metadata");
    };
    assert_eq!(*a_load_axis, 3);
    assert_eq!(a_load_factors, &[2, 3, 4]);
    let KernelActionSpace::Unroll {
        axis: b_load_axis,
        factors: b_load_factors,
    } = &full_space.spaces[6]
    else {
        panic!("GEMM global action space should expose B shared-load unroll metadata");
    };
    assert_eq!(*b_load_axis, 4);
    assert_eq!(b_load_factors, &[2, 3, 4]);
    let KernelActionSpace::ThreadGroup {
        axis: a_load_thread_axis,
        factors: a_load_thread_factors,
    } = &full_space.spaces[7]
    else {
        panic!("GEMM global action space should expose A shared-load thread-group metadata");
    };
    assert_eq!(*a_load_thread_axis, 3);
    assert_eq!(a_load_thread_factors, &[32, 64, 128, 256]);
    let KernelActionSpace::ThreadGroup {
        axis: b_load_thread_axis,
        factors: b_load_thread_factors,
    } = &full_space.spaces[8]
    else {
        panic!("GEMM global action space should expose B shared-load thread-group metadata");
    };
    assert_eq!(*b_load_thread_axis, 4);
    assert_eq!(b_load_thread_factors, &[32, 64, 128, 256]);
    assert!(matches!(
        full_space.spaces[9],
        KernelActionSpace::Swap { .. }
    ));
    assert!(matches!(
        full_space.spaces[10],
        KernelActionSpace::StrideOrder { .. }
    ));

    let tile_spaces = problem.action_spaces(&seed);
    let tile_actions = problem.schedule_actions(&seed);
    let deferred_tile_action =
        KernelScheduleAction::tile_gemm(13, 24, 13, KernelActionMaterialization::DeferredGenerated);

    assert_eq!(tile_spaces.spaces.len(), 2);
    assert_eq!(tile_spaces.actions(), tile_actions);
    let KernelActionSpace::Split { variants } = &tile_spaces.spaces[0] else {
        panic!("GEMM seed should expose one-axis split action-space metadata");
    };
    assert_eq!(variants.len(), 12);
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        13,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        1,
        24,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        2,
        32,
        KernelActionMaterialization::DeferredGenerated
    )));
    let KernelActionSpace::TileGemm { variants } = &tile_spaces.spaces[1] else {
        panic!("GEMM seed should expose existing tile materialization metadata");
    };
    assert_eq!(
        variants,
        &[KernelTile3dAction::new(
            KernelTile3d::new(16, 16, 16),
            KernelActionMaterialization::Existing
        )]
    );
    assert_eq!(tile_actions.len(), 13);
    assert!(tile_actions.contains(&KernelScheduleAction::split(
        0,
        13,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(tile_actions.contains(&KernelScheduleAction::tile_gemm(
        16,
        16,
        16,
        KernelActionMaterialization::Existing
    )));
    assert!(!tile_actions.contains(&deferred_tile_action));
    let direct_tile = problem
        .apply_schedule_action(&seed, &deferred_tile_action)
        .expect("direct tile-gemm action should remain replay-compatible");
    assert_eq!(
        schedule_gemm_tile(&direct_tile.schedule),
        Some(GemmTileShape::new(13, 24, 13))
    );

    let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
    let schedule_spaces = problem.action_spaces(&tile_candidate);
    let schedule_actions = problem.schedule_actions(&tile_candidate);
    assert_eq!(schedule_spaces.spaces.len(), 8);
    assert_eq!(schedule_spaces.actions(), schedule_actions);
    let KernelActionSpace::Split { variants } = &schedule_spaces.spaces[0] else {
        panic!("GEMM tile should expose one-axis retile split metadata");
    };
    assert_eq!(variants.len(), 12);
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        24,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        1,
        16,
        KernelActionMaterialization::Existing
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        2,
        32,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(!variants.contains(&KernelAxisFactorAction::new(
        0,
        16,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(matches!(
        schedule_spaces.spaces[1],
        KernelActionSpace::Unroll { .. }
    ));
    let KernelActionSpace::Upcast {
        axis: m_axis,
        factors: m_factors,
    } = &schedule_spaces.spaces[2]
    else {
        panic!("GEMM tile should expose M-axis upcast metadata");
    };
    assert_eq!(*m_axis, 0);
    assert_eq!(m_factors, &[2, 4]);
    let KernelActionSpace::Upcast {
        axis: n_axis,
        factors: n_factors,
    } = &schedule_spaces.spaces[3]
    else {
        panic!("GEMM tile should expose N-axis upcast metadata");
    };
    assert_eq!(*n_axis, 1);
    assert_eq!(n_factors, &[2, 4]);
    let KernelActionSpace::ThreadGroup {
        axis: a_load_thread_axis,
        factors: a_load_thread_factors,
    } = &schedule_spaces.spaces[4]
    else {
        panic!("GEMM tile should expose A shared-load thread-group metadata");
    };
    assert_eq!(*a_load_thread_axis, 3);
    assert_eq!(a_load_thread_factors, &[32, 64, 128, 256]);
    let KernelActionSpace::ThreadGroup {
        axis: b_load_thread_axis,
        factors: b_load_thread_factors,
    } = &schedule_spaces.spaces[5]
    else {
        panic!("GEMM tile should expose B shared-load thread-group metadata");
    };
    assert_eq!(*b_load_thread_axis, 4);
    assert_eq!(b_load_thread_factors, &[32, 64, 128, 256]);
    assert!(matches!(
        schedule_spaces.spaces[6],
        KernelActionSpace::Swap { .. }
    ));
    assert!(matches!(
        schedule_spaces.spaces[7],
        KernelActionSpace::StrideOrder { .. }
    ));
    assert_eq!(schedule_actions.len(), 42);
    assert!(schedule_actions.contains(&KernelScheduleAction::split(
        0,
        24,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(schedule_actions.contains(&KernelScheduleAction::split(
        1,
        16,
        KernelActionMaterialization::Existing
    )));
    assert!(schedule_actions.contains(&KernelScheduleAction::split(
        2,
        32,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(schedule_actions.contains(&KernelScheduleAction::unroll(2, 7)));
    assert!(schedule_actions.contains(&KernelScheduleAction::unroll(2, 16)));
    assert!(!schedule_actions.contains(&KernelScheduleAction::unroll(2, 1)));
    assert!(schedule_actions.contains(&KernelScheduleAction::upcast(0, 2)));
    assert!(schedule_actions.contains(&KernelScheduleAction::upcast(0, 4)));
    assert!(!schedule_actions.contains(&KernelScheduleAction::upcast(0, 3)));
    assert!(schedule_actions.contains(&KernelScheduleAction::upcast(1, 2)));
    assert!(schedule_actions.contains(&KernelScheduleAction::upcast(1, 4)));
    assert!(!schedule_actions.contains(&KernelScheduleAction::upcast(1, 3)));
    assert!(schedule_actions.contains(&KernelScheduleAction::stride_order(vec![0, 2])));
    assert!(schedule_actions.contains(&KernelScheduleAction::stride_order(vec![2, 1])));
    assert!(schedule_actions.contains(&KernelScheduleAction::swap(0, 1)));
    assert!(schedule_actions.contains(&KernelScheduleAction::thread_group(3, 64)));
    assert!(schedule_actions.contains(&KernelScheduleAction::thread_group(4, 64)));
    assert!(!schedule_actions.contains(&KernelScheduleAction::thread_group(3, 16)));

    let m_split = problem
        .apply_schedule_action(
            &tile_candidate,
            &KernelScheduleAction::split(0, 24, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("M-axis split action should retile candidate metadata");
    let m_split_plan =
        schedule_gemm_plan(&m_split.schedule).expect("M split candidate should have plan");
    assert_eq!(m_split_plan.tile, GemmTileShape::new(24, 32, 16));
    assert_eq!(m_split.launch.kernel, "gemm_f32_bf16_tile_24x32x16");
    assert_eq!(
        m_split.action_trace,
        vec![KernelScheduleAction::split(
            0,
            24,
            KernelActionMaterialization::DeferredGenerated
        )]
    );

    let existing_split = problem
        .apply_schedule_action(
            &tile_candidate,
            &KernelScheduleAction::split(1, 16, KernelActionMaterialization::Existing),
        )
        .expect("N-axis split to the existing tile should produce existing candidate metadata");
    let existing_split_plan = schedule_gemm_plan(&existing_split.schedule)
        .expect("existing split candidate should have plan");
    assert_eq!(existing_split_plan.tile, GemmTileShape::new(16, 16, 16));
    assert_eq!(existing_split.launch.kernel, "gemm_f32_bf16_tiled_kernel");
    assert!(existing_split.is_launchable());
    assert!(matches!(
        existing_split.generated.materialization,
        KernelMaterialization::Existing { .. }
    ));
    assert!(
            problem
                .apply_schedule_action(
                    &tile_candidate,
                    &KernelScheduleAction::split(
                        1,
                        16,
                        KernelActionMaterialization::DeferredGenerated,
                    ),
                )
                .is_none()
        );

    let unrolled = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::unroll(2, 7))
        .expect("unroll action should produce candidate metadata");
    let unrolled_plan =
        schedule_gemm_plan(&unrolled.schedule).expect("unrolled candidate should have plan");
    assert_eq!(unrolled_plan.reduce_unroll, 7);
    assert_eq!(unrolled.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u7");

    let m_upcast = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::upcast(0, 2))
        .expect("M upcast action should produce candidate metadata");
    let m_upcast_plan =
        schedule_gemm_plan(&m_upcast.schedule).expect("M upcast candidate should have plan");
    assert_eq!(m_upcast_plan.m_per_thread, 2);
    assert_eq!(m_upcast.launch.kernel, "gemm_f32_bf16_tile_16x32x16_mt2");
    assert_eq!(m_upcast.launch.block_dim.x, 32);
    assert_eq!(m_upcast.launch.block_dim.y, 8);
    assert_eq!(m_upcast.launch.block_dim.z, 1);

    let n_upcast = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::upcast(1, 2))
        .expect("N upcast action should produce candidate metadata");
    let n_upcast_plan =
        schedule_gemm_plan(&n_upcast.schedule).expect("N upcast candidate should have plan");
    assert_eq!(n_upcast_plan.n_per_thread, 2);
    assert_eq!(n_upcast.launch.kernel, "gemm_f32_bf16_tile_16x32x16_nt2");
    assert_eq!(n_upcast.launch.block_dim.x, 16);
    assert_eq!(n_upcast.launch.block_dim.y, 16);
    assert_eq!(n_upcast.launch.block_dim.z, 1);

    let a_load_thread_group = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::thread_group(3, 64))
        .expect("A shared-load thread-group action should produce candidate metadata");
    let a_load_thread_group_plan = schedule_gemm_plan(&a_load_thread_group.schedule)
        .expect("A shared-load thread-group candidate should have plan");
    assert_eq!(a_load_thread_group_plan.a_load_thread_count(), 64);
    assert_eq!(a_load_thread_group_plan.b_load_thread_count(), 512);
    assert_eq!(
        a_load_thread_group.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_atg64"
    );
    assert!(
        a_load_thread_group
            .schedule
            .transforms
            .iter()
            .any(|transform| {
                matches!(
                    transform,
                    ScheduleTransform::ThreadGroup {
                        axis: 3,
                        factor: 64
                    }
                )
            })
    );

    let b_load_thread_group = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::thread_group(4, 64))
        .expect("B shared-load thread-group action should produce candidate metadata");
    let b_load_thread_group_plan = schedule_gemm_plan(&b_load_thread_group.schedule)
        .expect("B shared-load thread-group candidate should have plan");
    assert_eq!(b_load_thread_group_plan.a_load_thread_count(), 512);
    assert_eq!(b_load_thread_group_plan.b_load_thread_count(), 64);
    assert_eq!(
        b_load_thread_group.launch.kernel,
        "gemm_f32_bf16_tile_16x32x16_btg64"
    );
    assert!(
        b_load_thread_group
            .schedule
            .transforms
            .iter()
            .any(|transform| {
                matches!(
                    transform,
                    ScheduleTransform::ThreadGroup {
                        axis: 4,
                        factor: 64
                    }
                )
            })
    );

    let traced_tile = problem
        .apply_schedule_action(&seed, &deferred_tile_action)
        .expect("tile action should produce candidate metadata");
    let traced_unrolled = problem
        .apply_schedule_action(&traced_tile, &KernelScheduleAction::unroll(2, 7))
        .expect("unroll action should extend candidate action trace");
    assert_eq!(
        traced_unrolled.action_trace,
        vec![deferred_tile_action, KernelScheduleAction::unroll(2, 7)]
    );
    let traced_unrolled_plan = schedule_gemm_plan(&traced_unrolled.schedule)
        .expect("traced unrolled candidate should have plan");
    assert_eq!(traced_unrolled_plan.tile, GemmTileShape::new(13, 24, 13));
    assert_eq!(traced_unrolled_plan.reduce_unroll, 7);
    assert_eq!(
        traced_unrolled.launch.kernel,
        "gemm_f32_bf16_tile_13x24x13_u7"
    );

    let a_reordered = problem
        .apply_schedule_action(
            &tile_candidate,
            &KernelScheduleAction::stride_order(vec![0, 2]),
        )
        .expect("A stride-order action should produce candidate metadata");
    let a_reordered_plan =
        schedule_gemm_plan(&a_reordered.schedule).expect("A reordered candidate should plan");
    assert_eq!(
        a_reordered_plan.a_load_order,
        GemmATileLoadOrder::MContiguous
    );
    assert_eq!(
        a_reordered_plan.b_load_order,
        GemmBTileLoadOrder::TileLinear
    );
    assert_eq!(a_reordered.launch.kernel, "gemm_f32_bf16_tile_16x32x16_am");

    let reordered = problem
        .apply_schedule_action(
            &tile_candidate,
            &KernelScheduleAction::stride_order(vec![2, 1]),
        )
        .expect("stride-order action should produce candidate metadata");
    let reordered_plan =
        schedule_gemm_plan(&reordered.schedule).expect("reordered candidate should have plan");
    assert_eq!(reordered_plan.b_load_order, GemmBTileLoadOrder::KContiguous);
    assert_eq!(reordered.launch.kernel, "gemm_f32_bf16_tile_16x32x16_bk");

    let swapped = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::swap(0, 1))
        .expect("swap action should produce candidate metadata");
    let swapped_plan =
        schedule_gemm_plan(&swapped.schedule).expect("swapped candidate should have plan");
    assert_eq!(swapped_plan.thread_order, GemmThreadOrder::MThenN);
    assert_eq!(swapped.launch.kernel, "gemm_f32_bf16_tile_16x32x16_sw01");
    assert_eq!(swapped.launch.block_dim.x, 16);
    assert_eq!(swapped.launch.block_dim.y, 32);
    assert_eq!(swapped.launch.block_dim.z, 1);
    assert!(swapped.schedule.transforms.iter().any(|transform| {
        matches!(
            transform,
            ScheduleTransform::Swap {
                axis_a: 0,
                axis_b: 1
            }
        )
    }));
}

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
    assert!(matches!(
        unroll7.generated.materialization,
        KernelMaterialization::DeferredGenerated { .. }
    ));
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
fn gemm_generator_renders_shared_load_thread_group_source_on_demand() {
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
            KernelScheduleAction::thread_group(3, 32),
            KernelScheduleAction::thread_group(4, 64),
        ],
    )
    .expect("valid shared-load-thread-group GEMM action trace should replay");
    let plan =
        schedule_gemm_plan(&candidate.schedule).expect("load-thread-grouped candidate should plan");
    assert_eq!(plan.a_load_thread_count(), 32);
    assert_eq!(plan.b_load_thread_count(), 64);

    let generated = GemmRustCudaGenerator
        .source_for(&candidate)
        .expect("GEMM generator should render shared-load-thread-grouped source");

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

#[test]
fn gemm_resource_limits_reject_overbudget_plan_metadata() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let seed = problem.seed();
    let action = KernelScheduleAction::tile_gemm(
        128,
        128,
        64,
        KernelActionMaterialization::DeferredGenerated,
    );
    let overbudget_plan = GemmSchedulePlan::new(GemmTileShape::new(128, 128, 64));

    assert_eq!(overbudget_plan.resource_usage().threads_per_block, 16_384);
    assert_eq!(overbudget_plan.resource_usage().shared_memory_bytes, 65_536);
    assert!(!GemmSearchProblem::plan_within_resource_limits(
        overbudget_plan
    ));
    assert!(
        problem
            .candidate_for_checked_plan(&seed, &action, overbudget_plan)
            .is_none()
    );
}

#[test]
fn gemm_search_can_rank_deferred_generated_descriptors_when_allowed() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let result = beam_search_metadata(
        &problem,
        BeamSearchConfig {
            beam_width: 1,
            max_depth: 1,
            require_launchable: false,
        },
    );
    let best = result
        .best
        .expect("GEMM search should keep a generated descriptor when allowed");

    assert_ne!(
        schedule_gemm_tile(&best.schedule),
        Some(GemmSearchProblem::EXISTING_TILE)
    );
    assert!(!best.is_launchable());
    assert!(matches!(
        best.action_trace.as_slice(),
        [KernelScheduleAction {
            op: KernelScheduleActionOp::Split,
            ..
        }]
    ));
    assert!(matches!(
        best.generated.materialization,
        KernelMaterialization::DeferredGenerated { .. }
    ));
}

#[test]
fn gemm_search_accepts_external_measured_scores_across_unroll_depth() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let result = beam_search_metadata_with_scorer(
        &problem,
        BeamSearchConfig {
            beam_width: 64,
            max_depth: 4,
            require_launchable: false,
        },
        |candidate| {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            let target = GemmTileShape::new(13, 24, 13);
            let tile_distance = plan.tile.m.abs_diff(target.m)
                + plan.tile.n.abs_diff(target.n)
                + plan.tile.k.abs_diff(target.k);
            let score = f64::from(tile_distance) * 1000.0
                + if plan.reduce_unroll == 7 {
                    0.0
                } else {
                    100.0 + f64::from(plan.reduce_unroll)
                };
            SearchScore::measured(score)
        },
    );
    let best = result
        .best
        .expect("GEMM search should keep the externally best unroll descriptor");
    let plan = schedule_gemm_plan(&best.schedule).expect("best candidate should have a plan");

    assert_eq!(plan.tile, GemmTileShape::new(13, 24, 13));
    assert_eq!(plan.reduce_unroll, 7);
    assert_eq!(
        best.score.map(|score| score.source),
        Some(SearchScoreSource::Measured)
    );
}

#[test]
fn gemm_search_accepts_external_measured_scores_across_stride_order_depth() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let result = beam_search_metadata_with_scorer(
        &problem,
        BeamSearchConfig {
            beam_width: 96,
            max_depth: 5,
            require_launchable: false,
        },
        |candidate| {
            let plan = schedule_gemm_plan(&candidate.schedule)?;
            let target = GemmTileShape::new(13, 24, 13);
            let tile_distance = plan.tile.m.abs_diff(target.m)
                + plan.tile.n.abs_diff(target.n)
                + plan.tile.k.abs_diff(target.k);
            let score = f64::from(tile_distance) * 10_000.0
                + if plan.reduce_unroll == 7 {
                    0.0
                } else {
                    1000.0 + f64::from(plan.reduce_unroll)
                }
                + if plan.b_load_order == GemmBTileLoadOrder::KContiguous {
                    0.0
                } else {
                    100.0
                };
            SearchScore::measured(score)
        },
    );
    let best = result
        .best
        .expect("GEMM search should keep the externally best stride-order descriptor");
    let plan = schedule_gemm_plan(&best.schedule).expect("best candidate should have a plan");

    assert_eq!(plan.tile, GemmTileShape::new(13, 24, 13));
    assert_eq!(plan.reduce_unroll, 7);
    assert_eq!(plan.b_load_order, GemmBTileLoadOrder::KContiguous);
    assert_eq!(best.launch.kernel, "gemm_f32_bf16_tile_13x24x13_u7_bk");
}

#[test]
fn gemm_search_accepts_external_measured_scores() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let result = beam_search_metadata_with_scorer(
        &problem,
        BeamSearchConfig {
            beam_width: 8,
            max_depth: 3,
            require_launchable: false,
        },
        |candidate| {
            let tile = schedule_gemm_tile(&candidate.schedule)?;
            let target = GemmTileShape::new(13, 24, 13);
            let distance =
                tile.m.abs_diff(target.m) + tile.n.abs_diff(target.n) + tile.k.abs_diff(target.k);
            SearchScore::measured(f64::from(distance))
        },
    );
    let best = result
        .best
        .expect("GEMM search should accept externally scored candidates");

    assert_eq!(
        schedule_gemm_tile(&best.schedule),
        Some(GemmTileShape::new(13, 24, 13))
    );
    assert_eq!(
        best.score.map(|score| score.source),
        Some(SearchScoreSource::Measured)
    );
}

#[test]
fn gemm_default_search_keeps_only_currently_launchable_tile() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let result = beam_search_metadata(&problem, BeamSearchConfig::default());
    let best = result
        .best
        .expect("GEMM search should keep the existing tile");
    assert_eq!(result.explored, 41);
    assert_eq!(result.rejected, 40);
    assert_eq!(
        schedule_gemm_tile(&best.schedule),
        Some(GemmTileShape::new(16, 16, 16))
    );
    assert!(best.is_launchable());
}
