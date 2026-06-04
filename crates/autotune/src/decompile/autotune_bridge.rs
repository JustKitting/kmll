use std::collections::BTreeSet;

use nn_rust_profiling::{
    NumericKind, OperationKind, OperationRoute, TensorTypeSpec, TypedOperationSpec,
};

use super::{
    ControlTarget, ControlTargetKind, KernelIrFunction, KernelIrModule, KernelIrOpKind,
    MemorySpace, SassSemanticPatternCategory, SassSymbol, recover_sass_patterns,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompiledAutotuneShape {
    MatvecBf16RowMajor { rows: usize, cols: usize },
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
    pub has_bf16_widen: bool,
    pub has_f32_mul_add: bool,
    pub has_f32_fused_multiply_add: bool,
    pub has_warp_reduce_sum: bool,
}

impl DecompiledAutotuneEvidence {
    pub fn supports_bf16_row_major_matvec(&self) -> bool {
        self.has_bf16_descriptor_load
            && self.has_bf16_widen
            && (self.has_f32_mul_add || self.has_f32_fused_multiply_add)
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

fn matvec_bf16_row_major_evidence(
    module: Option<&KernelIrModule>,
    function: &KernelIrFunction,
) -> DecompiledAutotuneEvidence {
    let pattern_categories = typed_pattern_categories(module, function);
    DecompiledAutotuneEvidence {
        has_bf16_descriptor_load: function.ops.iter().any(|op| {
            matches!(
                &op.kind,
                KernelIrOpKind::Load { space, access, .. }
                    if *space == MemorySpace::Descriptor && access.width_bits == Some(16)
            )
        }),
        has_bf16_widen: pattern_categories.contains(&SassSemanticPatternCategory::Bf16WidenBits),
        has_f32_mul_add: pattern_categories.contains(&SassSemanticPatternCategory::F32MulAddPair),
        has_f32_fused_multiply_add: function.ops.iter().any(|op| {
            matches!(
                &op.kind,
                KernelIrOpKind::FusedMultiplyAdd {
                    lane_bits: None,
                    ..
                }
            )
        }),
        has_warp_reduce_sum: pattern_categories
            .contains(&SassSemanticPatternCategory::WarpReduceSum),
        pattern_categories,
    }
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
