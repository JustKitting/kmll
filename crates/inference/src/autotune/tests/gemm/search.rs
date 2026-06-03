use super::*;

#[test]
fn gemm_search_can_rank_generated_descriptors_when_allowed() {
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
            op: KernelScheduleActionOp::LocalTile,
            ..
        }]
    ));
    assert!(matches!(
        best.generated.materialization,
        KernelMaterialization::Generated { .. }
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
    assert_eq!(result.explored, 53);
    assert_eq!(result.rejected, 52);
    assert_eq!(
        schedule_gemm_tile(&best.schedule),
        Some(GemmTileShape::new(16, 16, 16))
    );
    assert!(best.is_launchable());
}
