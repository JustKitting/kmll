use super::*;

pub(in crate::autotune::problem) fn unsupported_operation(
    operation: &TypedOperationSpec,
    reason: impl Into<String>,
) -> KernelGenerationError {
    KernelGenerationError::UnsupportedOperation {
        name: operation.name.clone(),
        kind: operation.kind,
        reason: reason.into(),
    }
}

pub(in crate::autotune::problem) fn tensor_pair(
    tensors: &[TensorTypeSpec],
) -> Option<[&TensorTypeSpec; 2]> {
    match tensors {
        [lhs, rhs] => Some([lhs, rhs]),
        _ => None,
    }
}

pub(in crate::autotune::problem) fn single_tensor(
    tensors: &[TensorTypeSpec],
) -> Option<&TensorTypeSpec> {
    match tensors {
        [tensor] => Some(tensor),
        _ => None,
    }
}

pub(in crate::autotune::problem) fn tensor_shape_1(tensor: &TensorTypeSpec) -> Option<[usize; 1]> {
    match tensor.shape.as_slice() {
        [x] => Some([*x]),
        _ => None,
    }
}

pub(in crate::autotune::problem) fn tensor_shape_2(tensor: &TensorTypeSpec) -> Option<[usize; 2]> {
    match tensor.shape.as_slice() {
        [x, y] => Some([*x, *y]),
        _ => None,
    }
}

pub(in crate::autotune::problem) fn layout_is(tensor: &TensorTypeSpec, expected: &str) -> bool {
    tensor.layout.as_deref() == Some(expected)
}
