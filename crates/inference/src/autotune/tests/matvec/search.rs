use super::*;

#[test]
fn matvec_search_keeps_only_metadata_for_rows_per_block_variants() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let result = beam_search_metadata(
        &problem,
        BeamSearchConfig {
            beam_width: 2,
            max_depth: 1,
            require_launchable: true,
        },
    );
    let best = result
        .best
        .expect("matvec search should produce a candidate");
    assert_eq!(result.explored, 36);
    assert_eq!(result.rejected, 32);
    assert!(best.is_launchable());
    assert_eq!(best.launch.kernel, "matvec_bf16_kernel");
    assert_eq!(best.launch.grid_dim.x, 512);
    assert_eq!(best.launch.block_dim.x, 256);
    assert_eq!(schedule_rows_per_block(&best.schedule), Some(8));
}

#[test]
fn matvec_search_can_prefer_smaller_row_groups_for_tiny_outputs() {
    let problem = MatvecSearchProblem::bf16_row_major(10, 4096);
    let result = beam_search_metadata(
        &problem,
        BeamSearchConfig {
            beam_width: 1,
            max_depth: 1,
            require_launchable: true,
        },
    );
    let best = result
        .best
        .expect("matvec search should produce a candidate");
    assert_eq!(schedule_rows_per_block(&best.schedule), Some(2));
}

#[test]
fn matvec_search_exposes_generated_row_split_metadata() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let candidates = problem.expand(&seed);
    assert_eq!(candidates.len(), 36);

    let generated = candidates
        .iter()
        .find(|candidate| {
            !candidate.is_launchable() && schedule_rows_per_block(&candidate.schedule) == Some(13)
        })
        .expect("matvec search should expose arbitrary generated rows-per-block metadata");
    assert_eq!(generated.launch.kernel, "matvec_bf16_rows13");
    assert_eq!(generated.launch.grid_dim.x, 316);
    assert_eq!(generated.launch.block_dim.x, 416);
    assert_eq!(
        generated.generated.materialization,
        KernelMaterialization::Generated {
            symbol: "matvec_bf16_rows13".to_string()
        }
    );
    assert_eq!(generated.generated.generator, "row-major-matvec-generator");
}

#[test]
fn matvec_search_can_rank_reduce_unroll_variants_when_allowed() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let result = beam_search_metadata_with_scorer(
        &problem,
        BeamSearchConfig {
            beam_width: 40,
            max_depth: 2,
            require_launchable: false,
        },
        |candidate| {
            let plan = schedule_matvec_plan(&candidate.schedule)?;
            if plan.rows.rows_per_block() == 13 && plan.reduce_unroll == 7 {
                SearchScore::measured(0.0)
            } else {
                SearchScore::measured(
                    100.0 + f64::from(plan.rows.rows_per_block()) + f64::from(plan.reduce_unroll),
                )
            }
        },
    );
    let best = result
        .best
        .expect("matvec search should produce an unrolled generated candidate");
    let plan = schedule_matvec_plan(&best.schedule).expect("best candidate should have plan");

    assert_eq!(plan.rows.rows_per_block(), 13);
    assert_eq!(plan.reduce_unroll, 7);
    assert_eq!(best.launch.kernel, "matvec_bf16_rows13_u7");
    assert_eq!(
        best.score.map(|score| score.source),
        Some(SearchScoreSource::Measured)
    );
}

#[test]
fn matvec_search_accepts_external_measured_scores_for_generated_candidate() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let result = beam_search_metadata_with_scorer(
        &problem,
        BeamSearchConfig {
            beam_width: 1,
            max_depth: 1,
            require_launchable: false,
        },
        |candidate| {
            let rows_per_block = schedule_rows_per_block(&candidate.schedule)?;
            let generated_bonus = if candidate.is_launchable() {
                100.0
            } else {
                0.0
            };
            SearchScore::measured(generated_bonus + (32.0 - f64::from(rows_per_block)))
        },
    );
    let best = result
        .best
        .expect("matvec search should keep externally best generated candidate");

    assert!(!best.is_launchable());
    assert_eq!(best.launch.kernel, "matvec_bf16_rows32");
    assert_eq!(schedule_rows_per_block(&best.schedule), Some(32));
    assert_eq!(
        best.score.map(|score| score.source),
        Some(SearchScoreSource::Measured)
    );
}

#[test]
fn metadata_key_changes_when_schedule_changes() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let rows4 = problem.candidate_for_rows(RowMajorWarpRows::Rows4);
    let rows8 = problem.candidate_for_rows(RowMajorWarpRows::Rows8);
    assert_ne!(rows4.artifact_key(), rows8.artifact_key());
    assert_ne!(
        rows4.generated.artifact_key.hex(),
        rows8.generated.artifact_key.hex()
    );
}
