use super::*;

#[test]
fn gemm_seed_action_space_exposes_local_tiles_and_existing_tile() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let seed = problem.seed();
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
}
