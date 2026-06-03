use super::*;

#[test]
fn gemm_resource_limits_reject_overbudget_plan_metadata() {
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let seed = problem.seed();
    let action = KernelScheduleAction::tile_gemm(
        128,
        128,
        64,
        KernelActionMaterialization::DeferredGenerated,
    );
    let overbudget_plan = GemmSchedulePlan::new(GemmTileShape::new(128, 128, 64));

    assert_eq!(overbudget_plan.resource_usage().threads_per_block, 16_384);
    assert_eq!(overbudget_plan.resource_usage().shared_memory_bytes, 65_536);
    assert!(!GemmSearchProblem::plan_within_resource_limits(
        overbudget_plan
    ));
    assert!(
        problem
            .candidate_for_checked_plan(&seed, &action, overbudget_plan)
            .is_none()
    );
}
