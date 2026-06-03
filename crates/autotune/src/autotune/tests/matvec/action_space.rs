use super::*;

#[test]
fn matvec_action_space_exposes_existing_and_generated_row_splits() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let spaces = problem.action_spaces(&seed);
    let actions = problem.schedule_actions(&seed);

    assert_eq!(spaces.spaces.len(), 2);
    assert_eq!(spaces.actions(), actions);
    let KernelActionSpace::Split { variants } = &spaces.spaces[0] else {
        panic!("matvec seed should expose existing split action-space metadata");
    };
    assert_eq!(variants.len(), 5);
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        1,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        8,
        KernelActionMaterialization::Existing
    )));
    assert!(!variants.contains(&KernelAxisFactorAction::new(
        0,
        32,
        KernelActionMaterialization::Existing
    )));
    let KernelActionSpace::GroupTop { axis, factors } = &spaces.spaces[1] else {
        panic!("matvec seed should expose generated group-top action-space metadata");
    };
    assert_eq!(*axis, 0);
    assert_eq!(factors.len(), 32);
    assert_eq!(factors.first().copied(), Some(1));
    assert_eq!(factors.last().copied(), Some(32));
    let complete_space = problem.search_space();
    assert_eq!(complete_space.spaces.len(), 9);
    assert!(matches!(
        complete_space.spaces[0],
        KernelActionSpace::Split { .. }
    ));
    assert!(matches!(
        complete_space.spaces[1],
        KernelActionSpace::GroupTop { .. }
    ));
    let KernelActionSpace::LocalTile { axis, factors } = &complete_space.spaces[2] else {
        panic!("matvec global action space should expose local-tile metadata");
    };
    assert_eq!(*axis, 0);
    assert_eq!(factors.as_slice(), &[2, 3, 4, 8, 13, 16, 24, 29, 32]);
    assert!(matches!(
        complete_space.spaces[3],
        KernelActionSpace::Upcast { .. }
    ));
    assert!(matches!(
        complete_space.spaces[4],
        KernelActionSpace::Unroll { .. }
    ));
    assert!(matches!(
        complete_space.spaces[5],
        KernelActionSpace::GroupTop { .. }
    ));
    assert!(matches!(
        complete_space.spaces[6],
        KernelActionSpace::Group { .. }
    ));
    let KernelActionSpace::ThreadGroup {
        axis: thread_axis,
        factors: thread_factors,
    } = &complete_space.spaces[7]
    else {
        panic!("matvec global action space should expose non-duplicate thread-group metadata");
    };
    assert_eq!(*thread_axis, 1);
    assert_eq!(thread_factors.as_slice(), &[2]);
    assert!(matches!(
        complete_space.spaces[8],
        KernelActionSpace::StrideOrder { .. }
    ));
    assert_eq!(actions.len(), 37);
    assert!(actions.contains(&KernelScheduleAction::split(
        0,
        1,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(actions.contains(&KernelScheduleAction::split(
        0,
        8,
        KernelActionMaterialization::Existing
    )));
    assert!(actions.contains(&KernelScheduleAction::group_top(0, 13)));
    assert!(actions.contains(&KernelScheduleAction::group_top(0, 32)));

    let generated = problem
        .apply_schedule_action(&seed, &KernelScheduleAction::group_top(0, 13))
        .expect("row group-top action should produce candidate metadata");
    assert_eq!(generated.launch.kernel, "matvec_bf16_rows13");
    assert_eq!(schedule_rows_per_block(&generated.schedule), Some(13));
    assert!(generated.schedule.transforms.iter().any(|transform| {
        matches!(
            transform,
            ScheduleTransform::GroupTop {
                axis: 0,
                factor: 13
            }
        )
    }));
    assert_eq!(
        generated.action_trace,
        vec![KernelScheduleAction::group_top(0, 13)]
    );
    assert_eq!(
        generated.generated.materialization,
        KernelMaterialization::Generated {
            symbol: "matvec_bf16_rows13".to_string()
        }
    );

    let naive = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 1, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("deferred row split of one should produce naive baseline metadata");
    assert_eq!(naive.launch.kernel, "matvec_bf16_naive");
    assert_eq!(naive.launch.grid_dim.x, 4096);
    assert_eq!(naive.launch.block_dim.x, 1);
    assert_eq!(
        naive.generated.materialization,
        KernelMaterialization::Generated {
            symbol: "matvec_bf16_naive".to_string()
        }
    );

    let existing_rows8 = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::Existing),
        )
        .expect("existing row split should produce candidate metadata");
    let existing_spaces = problem.action_spaces(&existing_rows8);
    assert_eq!(existing_spaces.spaces.len(), 1);
    let KernelActionSpace::LocalTile { axis, factors } = &existing_spaces.spaces[0] else {
        panic!("existing matvec row split should expose local-tile metadata");
    };
    assert_eq!(*axis, 0);
    assert_eq!(factors.as_slice(), &[2, 3, 4, 13, 16, 24, 29, 32]);
    let retiled_existing = problem
        .apply_schedule_action(&existing_rows8, &KernelScheduleAction::local_tile(0, 13))
        .expect("local-tile action should turn existing matvec row split into generated metadata");
    assert_eq!(retiled_existing.launch.kernel, "matvec_bf16_rows13");
    assert_eq!(
        schedule_rows_per_block(&retiled_existing.schedule),
        Some(13)
    );
    assert_eq!(
        retiled_existing.action_trace,
        vec![
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::Existing),
            KernelScheduleAction::local_tile(0, 13),
        ]
    );
    assert_eq!(
        retiled_existing.generated.materialization,
        KernelMaterialization::Generated {
            symbol: "matvec_bf16_rows13".to_string()
        }
    );
}

#[test]
fn matvec_generated_row_split_exposes_reduce_unroll_actions() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let rows8 = problem
        .apply_schedule_action(&seed, &KernelScheduleAction::group_top(0, 8))
        .expect("row group-top action should produce generated candidate metadata");
    let spaces = problem.action_spaces(&rows8);
    let actions = problem.schedule_actions(&rows8);

    assert_eq!(spaces.spaces.len(), 6);
    assert_eq!(spaces.actions(), actions);
    let KernelActionSpace::LocalTile { axis, factors } = &spaces.spaces[0] else {
        panic!("generated matvec split should expose local-tile action-space metadata");
    };
    assert_eq!(*axis, 0);
    assert_eq!(factors.as_slice(), &[2, 3, 4, 13, 16, 24, 29, 32]);
    assert!(actions.contains(&KernelScheduleAction::local_tile(0, 13)));
    assert!(!actions.contains(&KernelScheduleAction::local_tile(0, 8)));

    let KernelActionSpace::Upcast { axis, factors } = &spaces.spaces[1] else {
        panic!("generated matvec split should expose upcast action-space metadata");
    };
    assert_eq!(*axis, 0);
    assert_eq!(
        factors.as_slice(),
        MatvecRowUpcast::SEARCH_FACTORS.as_slice()
    );
    assert!(actions.contains(&KernelScheduleAction::upcast(0, 2)));
    assert!(actions.contains(&KernelScheduleAction::upcast(0, 4)));

    let KernelActionSpace::Unroll { axis, factors } = &spaces.spaces[2] else {
        panic!("generated matvec split should expose unroll action-space metadata");
    };
    assert_eq!(*axis, 1);
    assert_eq!(factors.len(), 31);
    assert_eq!(factors.first().copied(), Some(1));
    assert_eq!(factors.last().copied(), Some(32));
    assert!(!factors.contains(&MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL));
    assert!(factors.contains(&7));
    assert!(actions.contains(&KernelScheduleAction::unroll(1, 1)));
    assert!(actions.contains(&KernelScheduleAction::unroll(1, 2)));
    assert!(actions.contains(&KernelScheduleAction::unroll(1, 7)));
    assert!(actions.contains(&KernelScheduleAction::unroll(1, 32)));
    let KernelActionSpace::GroupTop { axis, factors } = &spaces.spaces[3] else {
        panic!("generated matvec split should expose reduce group-top action-space metadata");
    };
    assert_eq!(*axis, 1);
    assert_eq!(factors.as_slice(), &[13, 16, 28, 29, 32, 49, 64, 256]);
    assert!(actions.contains(&KernelScheduleAction::group_top(1, 64)));

    let KernelActionSpace::Group { axis, factors } = &spaces.spaces[4] else {
        panic!("generated matvec split should expose group action-space metadata");
    };
    assert_eq!(*axis, 1);
    assert_eq!(factors.as_slice(), &[4, 8, 16]);
    assert!(actions.contains(&KernelScheduleAction::group(1, 16)));
    assert!(!actions.contains(&KernelScheduleAction::group(1, 2)));

    let KernelActionSpace::ThreadGroup { axis, factors } = &spaces.spaces[5] else {
        panic!("generated matvec split should expose non-duplicate thread-group metadata");
    };
    assert_eq!(*axis, 1);
    assert_eq!(factors.as_slice(), &[2]);
    assert!(actions.contains(&KernelScheduleAction::thread_group(1, 2)));

    let unrolled = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::unroll(1, 7))
        .expect("reduce unroll action should produce generated candidate metadata");
    assert_eq!(unrolled.launch.kernel, "matvec_bf16_rows8_u7");
    assert_eq!(schedule_matvec_reduce_unroll(&unrolled.schedule), Some(7));
    assert_eq!(
        unrolled.action_trace,
        vec![
            KernelScheduleAction::group_top(0, 8),
            KernelScheduleAction::unroll(1, 7),
        ]
    );
    assert_ne!(rows8.artifact_key(), unrolled.artifact_key());

    let reduce_grouped = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::group_top(1, 64))
        .expect("reduce group-top action should produce generated candidate metadata");
    assert_eq!(reduce_grouped.launch.kernel, "matvec_bf16_rows8_cg64");
    let reduce_grouped_plan =
        schedule_matvec_plan(&reduce_grouped.schedule).expect("grouped reduce should plan");
    assert_eq!(reduce_grouped_plan.reduce_group_size(), 64);
    assert!(reduce_grouped.schedule.transforms.iter().any(|transform| {
        matches!(
            transform,
            ScheduleTransform::GroupTop {
                axis: 1,
                factor: 64
            }
        )
    }));
    assert_eq!(
        reduce_grouped.action_trace,
        vec![
            KernelScheduleAction::group_top(0, 8),
            KernelScheduleAction::group_top(1, 64),
        ]
    );
    assert_ne!(rows8.artifact_key(), reduce_grouped.artifact_key());

    let retiled = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::local_tile(0, 13))
        .expect("local-tile action should retile generated matvec candidate metadata");
    assert_eq!(retiled.launch.kernel, "matvec_bf16_rows13");
    assert_eq!(schedule_rows_per_block(&retiled.schedule), Some(13));
    assert!(retiled.schedule.transforms.iter().any(|transform| {
        matches!(
            transform,
            ScheduleTransform::GroupTop {
                axis: 0,
                factor: 13
            }
        )
    }));
    assert_eq!(
        retiled.action_trace,
        vec![
            KernelScheduleAction::group_top(0, 8),
            KernelScheduleAction::local_tile(0, 13),
        ]
    );
    assert_ne!(rows8.artifact_key(), retiled.artifact_key());

    let upcast = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::upcast(0, 2))
        .expect("row upcast action should produce generated candidate metadata");
    assert_eq!(upcast.launch.kernel, "matvec_bf16_rows8_up2");
    assert_eq!(upcast.launch.block_dim.x, 128);
    assert_eq!(
        schedule_matvec_row_upcast(&upcast.schedule).map(MatvecRowUpcast::factor),
        Some(2)
    );
    assert_eq!(
        upcast.action_trace,
        vec![
            KernelScheduleAction::group_top(0, 8),
            KernelScheduleAction::upcast(0, 2),
        ]
    );
    assert_ne!(rows8.artifact_key(), upcast.artifact_key());

    let upcast_spaces = problem.action_spaces(&upcast);
    let KernelActionSpace::StrideOrder { orders } = upcast_spaces
        .spaces
        .iter()
        .find(|space| matches!(space, KernelActionSpace::StrideOrder { .. }))
        .expect("row-upcast matvec should expose stride-order metadata")
    else {
        panic!("row-upcast matvec should expose stride-order metadata");
    };
    assert_eq!(
        orders.as_slice(),
        &[MatvecLoopOrder::ReductionThenRow.action_axes().to_vec()]
    );
    let reduction_first = problem
        .apply_schedule_action(&upcast, &KernelScheduleAction::stride_order(vec![1, 0]))
        .expect("stride-order action should produce generated candidate metadata");
    assert_eq!(reduction_first.launch.kernel, "matvec_bf16_rows8_up2_rf");
    let reduction_first_plan =
        schedule_matvec_plan(&reduction_first.schedule).expect("reduction-first plan should parse");
    assert_eq!(
        reduction_first_plan.loop_order,
        MatvecLoopOrder::ReductionThenRow
    );
    assert!(reduction_first.schedule.transforms.iter().any(|transform| {
        matches!(transform, ScheduleTransform::StrideOrder { axes } if axes == &[1, 0])
    }));
    assert_eq!(
        reduction_first.action_trace,
        vec![
            KernelScheduleAction::group_top(0, 8),
            KernelScheduleAction::upcast(0, 2),
            KernelScheduleAction::stride_order(vec![1, 0]),
        ]
    );
    assert_ne!(upcast.artifact_key(), reduction_first.artifact_key());

    let grouped = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::group(1, 16))
        .expect("group action should produce generated candidate metadata");
    assert_eq!(grouped.launch.kernel, "matvec_bf16_rows8_tg16");
    assert_eq!(grouped.launch.block_dim.x, 128);
    assert_eq!(
        schedule_matvec_thread_group(&grouped.schedule).map(MatvecThreadGroup::lanes_per_row),
        Some(16)
    );
    assert!(grouped.schedule.transforms.iter().any(|transform| {
        matches!(
            transform,
            ScheduleTransform::Group {
                axis: 1,
                factor: 16
            }
        )
    }));
    assert_eq!(
        grouped.action_trace,
        vec![
            KernelScheduleAction::group_top(0, 8),
            KernelScheduleAction::group(1, 16),
        ]
    );
    assert_ne!(rows8.artifact_key(), grouped.artifact_key());

    let two_lane_thread_grouped = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::thread_group(1, 2))
        .expect("thread-group action should produce two-lane matvec metadata");
    assert_eq!(
        two_lane_thread_grouped.launch.kernel,
        "matvec_bf16_rows8_tg2"
    );
    assert_eq!(two_lane_thread_grouped.launch.block_dim.x, 16);
    assert_eq!(
        schedule_matvec_thread_group(&two_lane_thread_grouped.schedule)
            .map(MatvecThreadGroup::lanes_per_row),
        Some(2)
    );
    assert!(
        two_lane_thread_grouped
            .schedule
            .transforms
            .iter()
            .any(|transform| {
                matches!(
                    transform,
                    ScheduleTransform::ThreadGroup { axis: 1, factor: 2 }
                )
            })
    );
    assert_eq!(
        two_lane_thread_grouped.action_trace,
        vec![
            KernelScheduleAction::group_top(0, 8),
            KernelScheduleAction::thread_group(1, 2),
        ]
    );
    assert_ne!(rows8.artifact_key(), two_lane_thread_grouped.artifact_key());

    let legacy_thread_grouped = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::thread_group(1, 16))
        .expect("legacy duplicate thread-group factor should remain replay-compatible");
    assert_eq!(
        schedule_matvec_thread_group(&legacy_thread_grouped.schedule)
            .map(MatvecThreadGroup::lanes_per_row),
        Some(16)
    );
}

#[test]
fn matvec_unroll_action_space_is_bounded_by_problem_shape() {
    let problem = MatvecSearchProblem::bf16_row_major(16, 3);
    let seed = problem.seed();
    let rows8 = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("row split action should produce generated candidate metadata");
    let spaces = problem.action_spaces(&rows8);
    let KernelActionSpace::Unroll { factors, .. } = spaces
        .spaces
        .iter()
        .find(|space| matches!(space, KernelActionSpace::Unroll { .. }))
        .expect("generated matvec split should expose unroll metadata")
    else {
        panic!("generated matvec split should expose unroll metadata");
    };

    assert_eq!(factors.as_slice(), &[1, 2, 3]);
    assert!(
        problem
            .apply_schedule_action(&rows8, &KernelScheduleAction::unroll(1, 3))
            .is_some()
    );
    assert!(
        problem
            .apply_schedule_action(&rows8, &KernelScheduleAction::unroll(1, 5))
            .is_none()
    );
}
