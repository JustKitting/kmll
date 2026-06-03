use super::*;

#[test]
fn gemm_action_space_exposes_tile_unroll_upcast_and_stride_metadata() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let seed = problem.seed();
    let full_space = problem.search_space();
    assert_eq!(full_space.spaces.len(), 13);
    let KernelActionSpace::LocalTile {
        axis: m_tile_axis,
        factors: m_tile_factors,
    } = &full_space.spaces[0]
    else {
        panic!("GEMM global action space should expose M-axis local-tile metadata");
    };
    assert_eq!(*m_tile_axis, 0);
    assert_eq!(m_tile_factors, &[2, 3, 4, 8, 13, 16, 24, 29, 32]);
    let KernelActionSpace::LocalTile {
        axis: n_tile_axis,
        factors: n_tile_factors,
    } = &full_space.spaces[1]
    else {
        panic!("GEMM global action space should expose N-axis local-tile metadata");
    };
    assert_eq!(*n_tile_axis, 1);
    assert_eq!(n_tile_factors, &[2, 3, 4, 8, 13, 16, 24, 29, 32]);
    let KernelActionSpace::LocalTile {
        axis: k_tile_axis,
        factors: k_tile_factors,
    } = &full_space.spaces[2]
    else {
        panic!("GEMM global action space should expose K-axis local-tile metadata");
    };
    assert_eq!(*k_tile_axis, 2);
    assert_eq!(k_tile_factors, &[2, 3, 4, 8, 13, 16, 24, 29, 32]);
    let KernelActionSpace::TileGemm { variants } = &full_space.spaces[3] else {
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
        full_space.spaces[4],
        KernelActionSpace::Unroll { .. }
    ));
    assert!(matches!(
        full_space.spaces[5],
        KernelActionSpace::Upcast { .. }
    ));
    assert!(matches!(
        full_space.spaces[6],
        KernelActionSpace::Upcast { .. }
    ));
    let KernelActionSpace::Unroll {
        axis: a_load_axis,
        factors: a_load_factors,
    } = &full_space.spaces[7]
    else {
        panic!("GEMM global action space should expose A shared-load unroll metadata");
    };
    assert_eq!(*a_load_axis, 3);
    assert_eq!(a_load_factors, &[2, 3, 4]);
    let KernelActionSpace::Unroll {
        axis: b_load_axis,
        factors: b_load_factors,
    } = &full_space.spaces[8]
    else {
        panic!("GEMM global action space should expose B shared-load unroll metadata");
    };
    assert_eq!(*b_load_axis, 4);
    assert_eq!(b_load_factors, &[2, 3, 4]);
    let KernelActionSpace::Group {
        axis: a_load_thread_axis,
        factors: a_load_thread_factors,
    } = &full_space.spaces[9]
    else {
        panic!("GEMM global action space should expose A shared-load group metadata");
    };
    assert_eq!(*a_load_thread_axis, 3);
    assert_eq!(a_load_thread_factors, &[32, 64, 128, 256]);
    let KernelActionSpace::Group {
        axis: b_load_thread_axis,
        factors: b_load_thread_factors,
    } = &full_space.spaces[10]
    else {
        panic!("GEMM global action space should expose B shared-load group metadata");
    };
    assert_eq!(*b_load_thread_axis, 4);
    assert_eq!(b_load_thread_factors, &[32, 64, 128, 256]);
    assert!(matches!(
        full_space.spaces[11],
        KernelActionSpace::Swap { .. }
    ));
    assert!(matches!(
        full_space.spaces[12],
        KernelActionSpace::StrideOrder { .. }
    ));

    let tile_spaces = problem.action_spaces(&seed);
    let tile_actions = problem.schedule_actions(&seed);
    let deferred_tile_action =
        KernelScheduleAction::tile_gemm(13, 24, 13, KernelActionMaterialization::DeferredGenerated);

    assert_eq!(tile_spaces.spaces.len(), 4);
    assert_eq!(tile_spaces.actions(), tile_actions);
    let KernelActionSpace::LocalTile {
        axis: m_tile_axis,
        factors: m_tile_factors,
    } = &tile_spaces.spaces[0]
    else {
        panic!("GEMM seed should expose M-axis local-tile metadata");
    };
    assert_eq!(*m_tile_axis, 0);
    assert_eq!(m_tile_factors, &[2, 3, 4, 8, 13, 24, 29, 32]);
    let KernelActionSpace::LocalTile {
        axis: n_tile_axis,
        factors: n_tile_factors,
    } = &tile_spaces.spaces[1]
    else {
        panic!("GEMM seed should expose N-axis local-tile metadata");
    };
    assert_eq!(*n_tile_axis, 1);
    assert_eq!(n_tile_factors, &[2, 3, 4, 8, 13, 24, 29, 32]);
    let KernelActionSpace::LocalTile {
        axis: k_tile_axis,
        factors: k_tile_factors,
    } = &tile_spaces.spaces[2]
    else {
        panic!("GEMM seed should expose K-axis local-tile metadata");
    };
    assert_eq!(*k_tile_axis, 2);
    assert_eq!(k_tile_factors, &[2, 3, 4, 8, 13, 24, 29, 32]);
    let KernelActionSpace::TileGemm { variants } = &tile_spaces.spaces[3] else {
        panic!("GEMM seed should expose existing tile materialization metadata");
    };
    assert_eq!(
        variants,
        &[KernelTile3dAction::new(
            KernelTile3d::new(16, 16, 16),
            KernelActionMaterialization::Existing
        )]
    );
    assert_eq!(tile_actions.len(), 25);
    assert!(tile_actions.contains(&KernelScheduleAction::local_tile(0, 13)));
    assert!(tile_actions.contains(&KernelScheduleAction::local_tile(1, 24)));
    assert!(tile_actions.contains(&KernelScheduleAction::local_tile(2, 32)));
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
    assert_eq!(schedule_spaces.spaces.len(), 10);
    assert_eq!(schedule_spaces.actions(), schedule_actions);
    let KernelActionSpace::LocalTile {
        axis: m_tile_axis,
        factors: m_tile_factors,
    } = &schedule_spaces.spaces[0]
    else {
        panic!("GEMM tile should expose M-axis local-tile metadata");
    };
    assert_eq!(*m_tile_axis, 0);
    assert_eq!(m_tile_factors, &[2, 3, 4, 8, 13, 24, 29, 32]);
    let KernelActionSpace::LocalTile {
        axis: n_tile_axis,
        factors: n_tile_factors,
    } = &schedule_spaces.spaces[1]
    else {
        panic!("GEMM tile should expose N-axis local-tile metadata");
    };
    assert_eq!(*n_tile_axis, 1);
    assert_eq!(n_tile_factors, &[2, 3, 4, 8, 13, 16, 24, 29]);
    let KernelActionSpace::LocalTile {
        axis: k_tile_axis,
        factors: k_tile_factors,
    } = &schedule_spaces.spaces[2]
    else {
        panic!("GEMM tile should expose K-axis local-tile metadata");
    };
    assert_eq!(*k_tile_axis, 2);
    assert_eq!(k_tile_factors, &[2, 3, 4, 8, 13, 24, 29, 32]);
    assert!(matches!(
        schedule_spaces.spaces[3],
        KernelActionSpace::Unroll { .. }
    ));
    let KernelActionSpace::Upcast {
        axis: m_axis,
        factors: m_factors,
    } = &schedule_spaces.spaces[4]
    else {
        panic!("GEMM tile should expose M-axis upcast metadata");
    };
    assert_eq!(*m_axis, 0);
    assert_eq!(m_factors, &[2, 4]);
    let KernelActionSpace::Upcast {
        axis: n_axis,
        factors: n_factors,
    } = &schedule_spaces.spaces[5]
    else {
        panic!("GEMM tile should expose N-axis upcast metadata");
    };
    assert_eq!(*n_axis, 1);
    assert_eq!(n_factors, &[2, 4]);
    let KernelActionSpace::Group {
        axis: a_load_thread_axis,
        factors: a_load_thread_factors,
    } = &schedule_spaces.spaces[6]
    else {
        panic!("GEMM tile should expose A shared-load group metadata");
    };
    assert_eq!(*a_load_thread_axis, 3);
    assert_eq!(a_load_thread_factors, &[32, 64, 128, 256]);
    let KernelActionSpace::Group {
        axis: b_load_thread_axis,
        factors: b_load_thread_factors,
    } = &schedule_spaces.spaces[7]
    else {
        panic!("GEMM tile should expose B shared-load group metadata");
    };
    assert_eq!(*b_load_thread_axis, 4);
    assert_eq!(b_load_thread_factors, &[32, 64, 128, 256]);
    assert!(matches!(
        schedule_spaces.spaces[8],
        KernelActionSpace::Swap { .. }
    ));
    assert!(matches!(
        schedule_spaces.spaces[9],
        KernelActionSpace::StrideOrder { .. }
    ));
    assert_eq!(schedule_actions.len(), 54);
    assert!(schedule_actions.contains(&KernelScheduleAction::local_tile(0, 24)));
    assert!(schedule_actions.contains(&KernelScheduleAction::local_tile(1, 16)));
    assert!(schedule_actions.contains(&KernelScheduleAction::local_tile(2, 32)));
    assert!(!schedule_actions.contains(&KernelScheduleAction::local_tile(0, 16)));
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
    assert!(schedule_actions.contains(&KernelScheduleAction::group(3, 64)));
    assert!(schedule_actions.contains(&KernelScheduleAction::group(4, 64)));
    assert!(!schedule_actions.contains(&KernelScheduleAction::group(3, 16)));

    let m_local_tile = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::local_tile(0, 24))
        .expect("M-axis local-tile action should retile candidate metadata");
    let m_local_tile_plan = schedule_gemm_plan(&m_local_tile.schedule)
        .expect("M local-tile candidate should have plan");
    assert_eq!(m_local_tile_plan.tile, GemmTileShape::new(24, 32, 16));
    assert_eq!(m_local_tile.launch.kernel, "gemm_f32_bf16_tile_24x32x16");
    assert_eq!(
        m_local_tile.action_trace,
        vec![KernelScheduleAction::local_tile(0, 24)]
    );

    let existing_local_tile = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::local_tile(1, 16))
        .expect(
            "N-axis local-tile to the existing tile should produce existing candidate metadata",
        );
    let existing_local_tile_plan = schedule_gemm_plan(&existing_local_tile.schedule)
        .expect("existing local-tile candidate should have plan");
    assert_eq!(
        existing_local_tile_plan.tile,
        GemmTileShape::new(16, 16, 16)
    );
    assert_eq!(
        existing_local_tile.launch.kernel,
        "gemm_f32_bf16_tiled_kernel"
    );
    assert!(existing_local_tile.is_launchable());
    assert!(matches!(
        existing_local_tile.generated.materialization,
        KernelMaterialization::Existing { .. }
    ));

    let legacy_existing_split = problem
        .apply_schedule_action(
            &tile_candidate,
            &KernelScheduleAction::split(1, 16, KernelActionMaterialization::Existing),
        )
        .expect("legacy N-axis split to the existing tile should replay");
    let legacy_existing_split_plan = schedule_gemm_plan(&legacy_existing_split.schedule)
        .expect("legacy existing split candidate should have plan");
    assert_eq!(
        legacy_existing_split_plan.tile,
        GemmTileShape::new(16, 16, 16)
    );
    assert!(matches!(
        legacy_existing_split.generated.materialization,
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
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::group(3, 64))
        .expect("A shared-load group action should produce candidate metadata");
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
                    ScheduleTransform::Group {
                        axis: 3,
                        factor: 64
                    }
                )
            })
    );

    let b_load_thread_group = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::group(4, 64))
        .expect("B shared-load group action should produce candidate metadata");
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
                    ScheduleTransform::Group {
                        axis: 4,
                        factor: 64
                    }
                )
            })
    );

    let legacy_a_load_thread_group = problem
        .apply_schedule_action(&tile_candidate, &KernelScheduleAction::thread_group(3, 64))
        .expect("legacy A shared-load thread-group action should replay");
    let legacy_a_plan = schedule_gemm_plan(&legacy_a_load_thread_group.schedule)
        .expect("legacy A shared-load thread-group candidate should have plan");
    assert_eq!(legacy_a_plan.a_load_thread_count(), 64);
    assert!(
        legacy_a_load_thread_group
            .schedule
            .transforms
            .iter()
            .any(|transform| matches!(
                transform,
                ScheduleTransform::ThreadGroup {
                    axis: 3,
                    factor: 64
                }
            ))
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
