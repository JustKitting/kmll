use super::*;

fn matvec_operation(rows: usize, cols: usize) -> TypedOperationSpec {
    TypedOperationSpec::new(
        "test-matvec",
        OperationKind::Matvec,
        OperationRoute::CudaKernel,
    )
    .with_input(
        TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [cols]).with_layout("contiguous"),
    )
    .with_input(
        TensorTypeSpec::new(NumericKind::Bf16, NumericKind::F32, [rows, cols])
            .with_layout("row-major"),
    )
    .with_output(
        TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [rows]).with_layout("contiguous"),
    )
}

fn gemm_operation(m: usize, n: usize, k: usize) -> TypedOperationSpec {
    TypedOperationSpec::new("test-gemm", OperationKind::Gemm, OperationRoute::CudaKernel)
        .with_input(
            TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [m, k])
                .with_layout("row-major"),
        )
        .with_input(
            TensorTypeSpec::new(NumericKind::Bf16, NumericKind::F32, [k, n])
                .with_layout("column-major"),
        )
        .with_output(
            TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [m, n])
                .with_layout("row-major"),
        )
}

fn score_gemm_threads(candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
    let threads = candidate.resources?.threads_per_block;
    if threads > 128 {
        SearchScore::measured(1.0)
    } else {
        SearchScore::measured(10.0 + f64::from(threads))
    }
}

mod generation;
mod routing;
mod selection_cache;
