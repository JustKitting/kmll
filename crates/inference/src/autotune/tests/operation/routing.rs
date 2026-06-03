use super::*;

#[test]
fn operation_problem_routes_matvec_search_and_source_generation() {
    let problem = InferenceKernelOptimizationProblem::from_operation(&matvec_operation(128, 256))
        .expect("supported matvec operation should route to an autotune problem");
    let InferenceKernelOptimizationProblem::MatvecBf16RowMajor(matvec) = problem else {
        panic!("matvec operation should route to matvec search");
    };
    assert_eq!(matvec.rows, 128);
    assert_eq!(matvec.cols, 256);

    let result = beam_search_metadata(
        &problem,
        BeamSearchConfig {
            beam_width: 4,
            max_depth: 1,
            require_launchable: false,
        },
    );
    assert_eq!(result.explored, 37);
    let seed = problem.seed();
    let generated = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("generic matvec problem should replay split action");
    let source = InferenceKernelRustCudaGenerator
        .source_for(&generated)
        .expect("generic generator should render matvec source");

    assert_eq!(problem.family(), "matvec-bf16-row-major");
    assert_eq!(generated.family, "matvec-bf16-row-major");
    assert_eq!(source.symbol, "matvec_bf16_rows8");
    assert!(source.source.contains("pub fn matvec_bf16_rows8("));
}

#[test]
fn operation_problem_routes_gemm_search_and_source_generation() {
    let problem =
        InferenceKernelOptimizationProblem::from_operation(&gemm_operation(128, 128, 256))
            .expect("supported GEMM operation should route to an autotune problem");
    let InferenceKernelOptimizationProblem::GemmF32Bf16RowColRow(gemm) = problem else {
        panic!("GEMM operation should route to GEMM search");
    };
    assert_eq!(gemm.m, 128);
    assert_eq!(gemm.n, 128);
    assert_eq!(gemm.k, 256);

    let result = beam_search_metadata(
        &problem,
        BeamSearchConfig {
            beam_width: 4,
            max_depth: 1,
            require_launchable: false,
        },
    );
    assert_eq!(result.explored, 26);
    let seed = problem.seed();
    let generated = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::tile_gemm(
                16,
                32,
                16,
                KernelActionMaterialization::DeferredGenerated,
            ),
        )
        .expect("generic GEMM problem should replay tile action");
    let source = InferenceKernelRustCudaGenerator
        .source_for(&generated)
        .expect("generic generator should render GEMM source");

    assert_eq!(problem.family(), "gemm-f32-bf16-row-col-row");
    assert_eq!(generated.family, "gemm-f32-bf16-row-col-row");
    assert_eq!(source.symbol, "gemm_f32_bf16_tile_16x32x16");
    assert!(
        source
            .source
            .contains("pub fn gemm_f32_bf16_tile_16x32x16(")
    );
}

#[test]
fn operation_problem_rejects_unsupported_layouts_explicitly() {
    let operation = TypedOperationSpec::new(
        "bad-matvec-layout",
        OperationKind::Matvec,
        OperationRoute::CudaKernel,
    )
    .with_input(
        TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [256]).with_layout("contiguous"),
    )
    .with_input(
        TensorTypeSpec::new(NumericKind::Bf16, NumericKind::F32, [128, 256])
            .with_layout("column-major"),
    )
    .with_output(
        TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [128]).with_layout("contiguous"),
    );

    let error = InferenceKernelOptimizationProblem::from_operation(&operation)
        .expect_err("unsupported matvec layout should be rejected");

    let KernelGenerationError::UnsupportedOperation { name, kind, reason } = error else {
        panic!("unsupported operation should report the operation-level reason");
    };
    assert_eq!(name, "bad-matvec-layout");
    assert_eq!(kind, OperationKind::Matvec);
    assert!(reason.contains("row-major weights"));
}
