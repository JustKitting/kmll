use std::collections::BTreeSet;

use nn_rust_profiling::{
    NumericKind, OperationKind, OperationRoute, TensorTypeSpec, TypedOperationSpec,
};

use super::{
    ControlTarget, ControlTargetKind, KernelIrFunction, KernelIrModule, KernelIrOpKind,
    MemorySpace, SassSemanticPatternCategory, SassSymbol, SassSyncKind, recover_sass_patterns,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompiledAutotuneShape {
    MatvecBf16RowMajor { rows: usize, cols: usize },
    GemmF32Bf16RowColRow { m: usize, n: usize, k: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompiledAutotuneOperation {
    pub function: SassSymbol,
    pub operation: TypedOperationSpec,
    pub evidence: DecompiledAutotuneEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompiledAutotuneEvidence {
    pub pattern_categories: Vec<SassSemanticPatternCategory>,
    pub has_bf16_descriptor_load: bool,
    pub has_f32_descriptor_load: bool,
    pub has_bf16_widen: bool,
    pub has_f32_mul_add: bool,
    pub has_f32_fused_multiply_add: bool,
    pub has_shared_load: bool,
    pub has_shared_store: bool,
    pub has_barrier: bool,
    pub has_descriptor_store: bool,
    pub has_warp_reduce_sum: bool,
}

impl DecompiledAutotuneEvidence {
    pub fn supports_bf16_row_major_matvec(&self) -> bool {
        self.has_bf16_descriptor_load
            && self.has_bf16_widen
            && (self.has_f32_mul_add || self.has_f32_fused_multiply_add)
    }

    pub fn supports_f32_bf16_row_col_row_gemm(&self) -> bool {
        self.has_f32_descriptor_load
            && self.has_bf16_descriptor_load
            && self.has_bf16_widen
            && self.has_shared_store
            && self.has_shared_load
            && self.has_barrier
            && (self.has_f32_mul_add || self.has_f32_fused_multiply_add)
            && self.has_descriptor_store
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompiledAutotuneError {
    UnsupportedEvidence {
        function: SassSymbol,
        shape: DecompiledAutotuneShape,
        evidence: DecompiledAutotuneEvidence,
    },
}

pub fn decompiled_autotune_operation(
    function: &KernelIrFunction,
    shape: DecompiledAutotuneShape,
) -> Result<DecompiledAutotuneOperation, DecompiledAutotuneError> {
    decompiled_autotune_operation_with_context(None, function, shape)
}

pub fn decompiled_autotune_operation_with_module(
    module: &KernelIrModule,
    function: &KernelIrFunction,
    shape: DecompiledAutotuneShape,
) -> Result<DecompiledAutotuneOperation, DecompiledAutotuneError> {
    decompiled_autotune_operation_with_context(Some(module), function, shape)
}

fn decompiled_autotune_operation_with_context(
    module: Option<&KernelIrModule>,
    function: &KernelIrFunction,
    shape: DecompiledAutotuneShape,
) -> Result<DecompiledAutotuneOperation, DecompiledAutotuneError> {
    match shape {
        DecompiledAutotuneShape::MatvecBf16RowMajor { rows, cols } => {
            let evidence = matvec_bf16_row_major_evidence(module, function);
            if !evidence.supports_bf16_row_major_matvec() {
                return Err(DecompiledAutotuneError::UnsupportedEvidence {
                    function: function.name.clone(),
                    shape,
                    evidence,
                });
            }
            Ok(DecompiledAutotuneOperation {
                function: function.name.clone(),
                operation: matvec_bf16_row_major_operation(&function.name, rows, cols),
                evidence,
            })
        }
        DecompiledAutotuneShape::GemmF32Bf16RowColRow { m, n, k } => {
            let evidence = gemm_f32_bf16_row_col_row_evidence(module, function);
            if !evidence.supports_f32_bf16_row_col_row_gemm() {
                return Err(DecompiledAutotuneError::UnsupportedEvidence {
                    function: function.name.clone(),
                    shape,
                    evidence,
                });
            }
            Ok(DecompiledAutotuneOperation {
                function: function.name.clone(),
                operation: gemm_f32_bf16_row_col_row_operation(&function.name, m, n, k),
                evidence,
            })
        }
    }
}

fn matvec_bf16_row_major_operation(
    function: &SassSymbol,
    rows: usize,
    cols: usize,
) -> TypedOperationSpec {
    TypedOperationSpec::new(
        format!(
            "decompiled-sass:{}:matvec-bf16-row-major",
            function.as_str()
        ),
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

fn gemm_f32_bf16_row_col_row_operation(
    function: &SassSymbol,
    m: usize,
    n: usize,
    k: usize,
) -> TypedOperationSpec {
    TypedOperationSpec::new(
        format!(
            "decompiled-sass:{}:gemm-f32-bf16-row-col-row",
            function.as_str()
        ),
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

fn matvec_bf16_row_major_evidence(
    module: Option<&KernelIrModule>,
    function: &KernelIrFunction,
) -> DecompiledAutotuneEvidence {
    decompiled_evidence(module, function)
}

fn gemm_f32_bf16_row_col_row_evidence(
    module: Option<&KernelIrModule>,
    function: &KernelIrFunction,
) -> DecompiledAutotuneEvidence {
    decompiled_evidence(module, function)
}

fn decompiled_evidence(
    module: Option<&KernelIrModule>,
    function: &KernelIrFunction,
) -> DecompiledAutotuneEvidence {
    let pattern_categories = typed_pattern_categories(module, function);
    let functions = scoped_functions(module, function);
    DecompiledAutotuneEvidence {
        has_bf16_descriptor_load: functions.iter().any(|function| {
            function.ops.iter().any(|op| {
                matches!(
                    &op.kind,
                    KernelIrOpKind::Load { space, access, .. }
                        if *space == MemorySpace::Descriptor && access.width_bits == Some(16)
                )
            })
        }),
        has_f32_descriptor_load: functions.iter().any(|function| {
            function
                .ops
                .iter()
                .any(|op| matches_descriptor_f32_load(&op.kind))
        }),
        has_bf16_widen: pattern_categories.contains(&SassSemanticPatternCategory::Bf16WidenBits),
        has_f32_mul_add: pattern_categories.contains(&SassSemanticPatternCategory::F32MulAddPair),
        has_f32_fused_multiply_add: functions.iter().any(|function| {
            function.ops.iter().any(|op| {
                matches!(
                    &op.kind,
                    KernelIrOpKind::FusedMultiplyAdd {
                        lane_bits: None,
                        ..
                    }
                )
            })
        }),
        has_shared_load: functions
            .iter()
            .any(|function| function.ops.iter().any(|op| matches_shared_load(&op.kind))),
        has_shared_store: functions
            .iter()
            .any(|function| function.ops.iter().any(|op| matches_shared_store(&op.kind))),
        has_barrier: functions
            .iter()
            .any(|function| function.ops.iter().any(|op| matches_barrier(&op.kind))),
        has_descriptor_store: functions.iter().any(|function| {
            function
                .ops
                .iter()
                .any(|op| matches_descriptor_store(&op.kind))
        }),
        has_warp_reduce_sum: pattern_categories
            .contains(&SassSemanticPatternCategory::WarpReduceSum),
        pattern_categories,
    }
}

fn matches_descriptor_f32_load(kind: &KernelIrOpKind) -> bool {
    matches!(
        kind,
        KernelIrOpKind::Load { space, access, .. }
            if *space == MemorySpace::Descriptor && access.width_bits != Some(16)
    )
}

fn matches_shared_load(kind: &KernelIrOpKind) -> bool {
    matches!(
        kind,
        KernelIrOpKind::Load { space, .. } if *space == MemorySpace::Shared
    )
}

fn matches_shared_store(kind: &KernelIrOpKind) -> bool {
    matches!(
        kind,
        KernelIrOpKind::Store { space, .. } if *space == MemorySpace::Shared
    )
}

fn matches_barrier(kind: &KernelIrOpKind) -> bool {
    matches!(
        kind,
        KernelIrOpKind::Sync {
            kind: SassSyncKind::Barrier,
            ..
        }
    )
}

fn matches_descriptor_store(kind: &KernelIrOpKind) -> bool {
    matches!(
        kind,
        KernelIrOpKind::Store { space, .. } if *space == MemorySpace::Descriptor
    )
}

fn scoped_functions<'a>(
    module: Option<&'a KernelIrModule>,
    function: &'a KernelIrFunction,
) -> Vec<&'a KernelIrFunction> {
    let Some(module) = module else {
        return vec![function];
    };
    let reachable = reachable_function_names(module, function);
    module
        .functions
        .iter()
        .filter(|function| reachable.contains(&function.name))
        .collect()
}

fn typed_pattern_categories(
    module: Option<&KernelIrModule>,
    function: &KernelIrFunction,
) -> Vec<SassSemanticPatternCategory> {
    let owned_module;
    let module = match module {
        Some(module) => module,
        None => {
            owned_module = KernelIrModule {
                target: None,
                functions: vec![function.clone()],
            };
            &owned_module
        }
    };
    let reachable = reachable_function_names(module, function);
    let mut categories = recover_sass_patterns(module)
        .functions
        .into_iter()
        .filter(|function| reachable.contains(&function.name))
        .flat_map(|function| function.patterns)
        .map(|pattern| pattern.kind.category())
        .collect::<Vec<_>>();
    categories.sort();
    categories.dedup();
    categories
}

fn reachable_function_names(
    module: &KernelIrModule,
    root: &KernelIrFunction,
) -> BTreeSet<SassSymbol> {
    let mut reachable = BTreeSet::from([root.name.clone()]);
    let mut queue = vec![root.name.clone()];
    while let Some(name) = queue.pop() {
        let Some(function) = module
            .functions
            .iter()
            .find(|function| function.name == name)
        else {
            continue;
        };
        for op in &function.ops {
            if let KernelIrOpKind::Call {
                target:
                    Some(ControlTarget {
                        kind: ControlTargetKind::Label(label),
                        ..
                    }),
                ..
            } = &op.kind
                && reachable.insert(label.clone())
            {
                queue.push(label.clone());
            }
        }
    }
    reachable
}
