use super::*;

#[test]
fn gemm_global_action_space_exposes_tile_unroll_upcast_and_stride_metadata() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let full_space = problem.search_space();

    assert_eq!(full_space.spaces.len(), 14);
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
    let KernelActionSpace::GroupTop {
        axis: reduce_group_axis,
        factors: reduce_group_factors,
    } = &full_space.spaces[5]
    else {
        panic!("GEMM global action space should expose reduce group-top metadata");
    };
    assert_eq!(*reduce_group_axis, 2);
    assert_eq!(reduce_group_factors, &[13, 16, 28, 29]);
    assert!(matches!(
        full_space.spaces[6],
        KernelActionSpace::Upcast { .. }
    ));
    assert!(matches!(
        full_space.spaces[7],
        KernelActionSpace::Upcast { .. }
    ));
    let KernelActionSpace::Unroll {
        axis: a_load_axis,
        factors: a_load_factors,
    } = &full_space.spaces[8]
    else {
        panic!("GEMM global action space should expose A shared-load unroll metadata");
    };
    assert_eq!(*a_load_axis, 3);
    assert_eq!(a_load_factors, &[2, 3, 4]);
    let KernelActionSpace::Unroll {
        axis: b_load_axis,
        factors: b_load_factors,
    } = &full_space.spaces[9]
    else {
        panic!("GEMM global action space should expose B shared-load unroll metadata");
    };
    assert_eq!(*b_load_axis, 4);
    assert_eq!(b_load_factors, &[2, 3, 4]);
    let KernelActionSpace::Group {
        axis: a_load_thread_axis,
        factors: a_load_thread_factors,
    } = &full_space.spaces[10]
    else {
        panic!("GEMM global action space should expose A shared-load group metadata");
    };
    assert_eq!(*a_load_thread_axis, 3);
    assert_eq!(a_load_thread_factors, &[32, 64, 128, 256]);
    let KernelActionSpace::Group {
        axis: b_load_thread_axis,
        factors: b_load_thread_factors,
    } = &full_space.spaces[11]
    else {
        panic!("GEMM global action space should expose B shared-load group metadata");
    };
    assert_eq!(*b_load_thread_axis, 4);
    assert_eq!(b_load_thread_factors, &[32, 64, 128, 256]);
    assert!(matches!(
        full_space.spaces[12],
        KernelActionSpace::Swap { .. }
    ));
    assert!(matches!(
        full_space.spaces[13],
        KernelActionSpace::StrideOrder { .. }
    ));
}
