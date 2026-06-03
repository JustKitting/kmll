use super::*;

#[test]
fn matvec_action_space_exposes_existing_and_generated_row_splits() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let spaces = problem.action_spaces(&seed);
    let actions = problem.schedule_actions(&seed);

    assert_eq!(spaces.spaces.len(), 1);
    assert_eq!(spaces.actions(), actions);
    let KernelActionSpace::Split { variants } = &spaces.spaces[0] else {
        panic!("matvec seed should expose split action-space metadata");
    };
    assert_eq!(variants.len(), 36);
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        8,
        KernelActionMaterialization::Existing
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        13,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        32,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(!variants.contains(&KernelAxisFactorAction::new(
        0,
        32,
        KernelActionMaterialization::Existing
    )));
    let complete_space = problem.search_space();
    assert_eq!(complete_space.spaces.len(), 4);
    assert!(matches!(
        complete_space.spaces[0],
        KernelActionSpace::Split { .. }
    ));
    assert!(matches!(
        complete_space.spaces[1],
        KernelActionSpace::Upcast { .. }
    ));
    assert!(matches!(
        complete_space.spaces[2],
        KernelActionSpace::Unroll { .. }
    ));
    assert!(matches!(
        complete_space.spaces[3],
        KernelActionSpace::ThreadGroup { .. }
    ));
    assert_eq!(actions.len(), 36);
    assert!(actions.contains(&KernelScheduleAction::split(
        0,
        8,
        KernelActionMaterialization::Existing
    )));
    assert!(actions.contains(&KernelScheduleAction::split(
        0,
        13,
        KernelActionMaterialization::DeferredGenerated
    )));

    let generated = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 13, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("row split action should produce candidate metadata");
    assert_eq!(generated.launch.kernel, "matvec_bf16_rows13");
    assert_eq!(schedule_rows_per_block(&generated.schedule), Some(13));
    assert_eq!(
        generated.action_trace,
        vec![KernelScheduleAction::split(
            0,
            13,
            KernelActionMaterialization::DeferredGenerated
        )]
    );
    assert_eq!(
        generated.generated.materialization,
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
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("row split action should produce generated candidate metadata");
    let spaces = problem.action_spaces(&rows8);
    let actions = problem.schedule_actions(&rows8);

    assert_eq!(spaces.spaces.len(), 3);
    assert_eq!(spaces.actions(), actions);
    let KernelActionSpace::Upcast { axis, factors } = &spaces.spaces[0] else {
        panic!("generated matvec split should expose upcast action-space metadata");
    };
    assert_eq!(*axis, 0);
    assert_eq!(
        factors.as_slice(),
        MatvecRowUpcast::SEARCH_FACTORS.as_slice()
    );
    assert!(actions.contains(&KernelScheduleAction::upcast(0, 2)));
    assert!(actions.contains(&KernelScheduleAction::upcast(0, 4)));

    let KernelActionSpace::Unroll { axis, factors } = &spaces.spaces[1] else {
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
    let KernelActionSpace::ThreadGroup { axis, factors } = &spaces.spaces[2] else {
        panic!("generated matvec split should expose thread-group action-space metadata");
    };
    assert_eq!(*axis, 1);
    assert_eq!(
        factors.as_slice(),
        MatvecThreadGroup::SEARCH_LANES_PER_ROW.as_slice()
    );
    assert!(actions.contains(&KernelScheduleAction::thread_group(1, 16)));

    let unrolled = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::unroll(1, 7))
        .expect("reduce unroll action should produce generated candidate metadata");
    assert_eq!(unrolled.launch.kernel, "matvec_bf16_rows8_u7");
    assert_eq!(schedule_matvec_reduce_unroll(&unrolled.schedule), Some(7));
    assert_eq!(
        unrolled.action_trace,
        vec![
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::unroll(1, 7),
        ]
    );
    assert_ne!(rows8.artifact_key(), unrolled.artifact_key());

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
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::upcast(0, 2),
        ]
    );
    assert_ne!(rows8.artifact_key(), upcast.artifact_key());

    let grouped = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::thread_group(1, 16))
        .expect("thread-group action should produce generated candidate metadata");
    assert_eq!(grouped.launch.kernel, "matvec_bf16_rows8_tg16");
    assert_eq!(grouped.launch.block_dim.x, 128);
    assert_eq!(
        schedule_matvec_thread_group(&grouped.schedule).map(MatvecThreadGroup::lanes_per_row),
        Some(16)
    );
    assert_eq!(
        grouped.action_trace,
        vec![
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::thread_group(1, 16),
        ]
    );
    assert_ne!(rows8.artifact_key(), grouped.artifact_key());
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
