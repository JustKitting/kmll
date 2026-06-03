use super::*;

#[test]
fn auto_optimize_preserves_parent_when_children_do_not_improve() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let result = auto_optimize_metadata_with_scorer(
        &problem,
        AutoOptimizeConfig {
            beam_width: 8,
            max_steps: 2,
            require_launchable: false,
            min_score_improvement: 0.0,
        },
        |candidate| {
            let plan = schedule_matvec_plan(&candidate.schedule)?;
            if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL
                && plan.row_upcast.is_default()
                && plan.thread_group.is_default()
            {
                SearchScore::measured(1.0)
            } else {
                SearchScore::measured(10.0)
            }
        },
    );
    let best = result
        .best
        .expect("auto optimize should keep the best parent candidate");
    let best_plan = schedule_matvec_plan(&best.schedule).expect("best candidate should plan");

    assert_eq!(
        result.exit_reason,
        AutoOptimizeExitReason::NoImprovement { best_delta: -9.0 }
    );
    assert_eq!(result.steps.len(), 2);
    assert_eq!(
        best_plan.reduce_unroll,
        MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL
    );
    assert_eq!(best.score.and_then(|score| Some(score.value)), Some(1.0));
}

#[test]
fn auto_optimize_accepts_improving_generated_action() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let result = auto_optimize_metadata_with_scorer(
        &problem,
        AutoOptimizeConfig {
            beam_width: 8,
            max_steps: 2,
            require_launchable: false,
            min_score_improvement: 0.0,
        },
        |candidate| {
            let plan = schedule_matvec_plan(&candidate.schedule)?;
            match plan.reduce_unroll {
                7 => SearchScore::measured(1.0),
                factor if factor == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL => {
                    SearchScore::measured(10.0)
                }
                _ => SearchScore::measured(5.0),
            }
        },
    );
    let best = result
        .best
        .expect("auto optimize should accept the improving generated candidate");
    let best_plan = schedule_matvec_plan(&best.schedule).expect("best candidate should plan");

    assert_eq!(result.exit_reason, AutoOptimizeExitReason::MaxSteps);
    assert_eq!(result.steps.len(), 2);
    assert_eq!(result.steps[1].improvement, Some(9.0));
    assert_eq!(best_plan.reduce_unroll, 7);
    assert_eq!(best.score.and_then(|score| Some(score.value)), Some(1.0));
}
