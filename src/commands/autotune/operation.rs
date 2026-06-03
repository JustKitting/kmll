use nn_rust_profiling::{
    NumericKind, OperationKind, OperationRoute, TensorTypeSpec, TypedOperationSpec,
};

pub(super) fn matvec_autotune_operation(rows: usize, cols: usize) -> TypedOperationSpec {
    TypedOperationSpec::new(
        "kernel-autotune-matvec",
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

pub(super) fn gemm_autotune_operation(m: usize, n: usize, k: usize) -> TypedOperationSpec {
    TypedOperationSpec::new(
        "kernel-autotune-gemm",
        OperationKind::Gemm,
        OperationRoute::CudaKernel,
    )
    .with_input(
        TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [m, k]).with_layout("row-major"),
    )
    .with_input(
        TensorTypeSpec::new(NumericKind::Bf16, NumericKind::F32, [k, n])
            .with_layout("column-major"),
    )
    .with_output(
        TensorTypeSpec::new(NumericKind::F32, NumericKind::F32, [m, n]).with_layout("row-major"),
    )
}
