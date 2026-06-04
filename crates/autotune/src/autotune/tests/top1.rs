use super::*;

#[test]
fn top1_search_exposes_existing_production_row_splits() {
    let problem = Top1Bf16SearchProblem::row_major(131_072, 4096);
    let seed = problem.seed();
    let candidates = problem.expand(&seed);

    assert_eq!(candidates.len(), 4);
    for candidate in &candidates {
        assert_eq!(candidate.family, Top1Bf16SearchProblem::FAMILY);
        assert!(candidate.is_launchable());
        assert_eq!(candidate.launch.kernel, "matvec_top1_bf16_stage_kernel");
        assert_eq!(
            candidate.generated.materialization,
            KernelMaterialization::Existing {
                symbol: "matvec_top1_bf16_stage_kernel"
            }
        );
    }

    let rows8 = candidates
        .iter()
        .find(|candidate| schedule_rows_per_block(&candidate.schedule) == Some(8))
        .expect("top1 search should expose rows8 candidate");
    assert_eq!(rows8.launch.grid_dim.x, 16_384);
    assert_eq!(rows8.launch.block_dim.x, 256);
}

#[test]
fn top1_search_accepts_measured_rows8_winner() {
    let problem = Top1Bf16SearchProblem::row_major(131_072, 4096);
    let result = beam_search_metadata_with_scorer(
        &problem,
        BeamSearchConfig {
            beam_width: 3,
            max_depth: 2,
            require_launchable: true,
        },
        |candidate| {
            let rows_per_block = schedule_rows_per_block(&candidate.schedule)?;
            SearchScore::measured(match rows_per_block {
                8 => 1.0,
                4 => 2.0,
                2 => 3.0,
                1 => 4.0,
                _ => return None,
            })
        },
    );
    let best = result
        .best
        .expect("top1 search should produce a measured candidate");

    assert_eq!(schedule_rows_per_block(&best.schedule), Some(8));
    assert_eq!(best.launch.kernel, "matvec_top1_bf16_stage_kernel");
    assert_eq!(
        best.score.map(|score| score.source),
        Some(SearchScoreSource::Measured)
    );
}
