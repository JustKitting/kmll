use crate::decompile::{KernelIrModule, KernelIrOpKind, SassSymbol};

use super::TensorCoreOpSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCoreIrOp {
    pub function: SassSymbol,
    pub address: u64,
    pub spec: TensorCoreOpSpec,
}

pub fn tensor_core_ops_from_ir_module(module: &KernelIrModule) -> Vec<TensorCoreIrOp> {
    module
        .functions
        .iter()
        .flat_map(|function| {
            function.ops.iter().filter_map(|op| {
                let KernelIrOpKind::TensorCoreMma {
                    opcode,
                    signature: Some(signature),
                    scope,
                    ..
                } = &op.kind
                else {
                    return None;
                };
                let spec = TensorCoreOpSpec::from_sass(
                    opcode.kind(),
                    signature,
                    scope.as_ref(),
                    &op.source_modifiers,
                )?;
                Some(TensorCoreIrOp {
                    function: function.name.clone(),
                    address: op.address,
                    spec,
                })
            })
        })
        .collect()
}
