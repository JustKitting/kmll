use super::*;

#[test]
fn gemm_tiled_candidate_action_space_exposes_schedule_metadata() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
    let schedule_spaces = problem.action_spaces(&tile_candidate);
    let schedule_actions = problem.schedule_actions(&tile_candidate);

    assert_eq!(schedule_spaces.spaces.len(), 11);
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
    let KernelActionSpace::GroupTop {
        axis: reduce_group_axis,
        factors: reduce_group_factors,
    } = &schedule_spaces.spaces[4]
    else {
        panic!("GEMM tile should expose reduce group-top metadata");
    };
    assert_eq!(*reduce_group_axis, 2);
    assert_eq!(reduce_group_factors, &[13]);
    let KernelActionSpace::Upcast {
        axis: m_axis,
        factors: m_factors,
    } = &schedule_spaces.spaces[5]
    else {
        panic!("GEMM tile should expose M-axis upcast metadata");
    };
    assert_eq!(*m_axis, 0);
    assert_eq!(m_factors, &[2, 4]);
    let KernelActionSpace::Upcast {
        axis: n_axis,
        factors: n_factors,
    } = &schedule_spaces.spaces[6]
    else {
        panic!("GEMM tile should expose N-axis upcast metadata");
    };
    assert_eq!(*n_axis, 1);
    assert_eq!(n_factors, &[2, 4]);
    let KernelActionSpace::Group {
        axis: a_load_thread_axis,
        factors: a_load_thread_factors,
    } = &schedule_spaces.spaces[7]
    else {
        panic!("GEMM tile should expose A shared-load group metadata");
    };
    assert_eq!(*a_load_thread_axis, 3);
    assert_eq!(a_load_thread_factors, &[32, 64, 128, 256]);
    let KernelActionSpace::Group {
        axis: b_load_thread_axis,
        factors: b_load_thread_factors,
    } = &schedule_spaces.spaces[8]
    else {
        panic!("GEMM tile should expose B shared-load group metadata");
    };
    assert_eq!(*b_load_thread_axis, 4);
    assert_eq!(b_load_thread_factors, &[32, 64, 128, 256]);
    assert!(matches!(
        schedule_spaces.spaces[9],
        KernelActionSpace::Swap { .. }
    ));
    assert!(matches!(
        schedule_spaces.spaces[10],
        KernelActionSpace::StrideOrder { .. }
    ));
    assert_eq!(schedule_actions.len(), 55);
    assert!(schedule_actions.contains(&KernelScheduleAction::local_tile(0, 24)));
    assert!(schedule_actions.contains(&KernelScheduleAction::local_tile(1, 16)));
    assert!(schedule_actions.contains(&KernelScheduleAction::local_tile(2, 32)));
    assert!(!schedule_actions.contains(&KernelScheduleAction::local_tile(0, 16)));
    assert!(schedule_actions.contains(&KernelScheduleAction::unroll(2, 7)));
    assert!(schedule_actions.contains(&KernelScheduleAction::unroll(2, 16)));
    assert!(!schedule_actions.contains(&KernelScheduleAction::unroll(2, 1)));
    assert!(schedule_actions.contains(&KernelScheduleAction::group_top(2, 13)));
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
}
