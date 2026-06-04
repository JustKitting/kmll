use std::collections::BTreeMap;

use super::super::{
    KernelIrFunction, KernelIrModule, KernelIrOp, SassAnalysisFunction, SassAnalysisModule,
    SassOpcode,
};
use super::{
    classify::classify_op,
    semantics::lift_semantics,
    types::{
        SassLiftedFunction, SassLiftedModule, SassLiftedOp, SassLiftedOpDetail, SassLiftedValueRef,
    },
};

pub fn lift_sass_value_ir(
    module: &KernelIrModule,
    analysis: &SassAnalysisModule,
) -> SassLiftedModule {
    let analysis_by_name = analysis
        .functions
        .iter()
        .map(|function| (function.name.clone(), function))
        .collect::<BTreeMap<_, _>>();
    SassLiftedModule {
        target: module.target.clone(),
        functions: module
            .functions
            .iter()
            .filter_map(|function| {
                analysis_by_name
                    .get(&function.name)
                    .map(|analysis| lift_function(function, analysis))
            })
            .collect(),
    }
}

fn lift_function(
    function: &KernelIrFunction,
    analysis: &SassAnalysisFunction,
) -> SassLiftedFunction {
    let mut input_refs_by_address = BTreeMap::<u64, Vec<SassLiftedValueRef>>::new();
    for edge in &analysis.def_use_edges {
        input_refs_by_address
            .entry(edge.use_address)
            .or_default()
            .push(SassLiftedValueRef {
                value_id: edge.value_id,
                register: edge.register.clone(),
            });
    }

    let value_registers = analysis
        .ssa_values
        .iter()
        .map(|value| (value.value_id, value.register.clone()))
        .collect::<BTreeMap<_, _>>();

    let value_ops_by_address = analysis
        .value_ops
        .iter()
        .map(|op| (op.address, op))
        .collect::<BTreeMap<_, _>>();

    let ops = function
        .ops
        .iter()
        .map(|op| {
            let value_op = value_ops_by_address.get(&op.address);
            let outputs = value_op
                .into_iter()
                .flat_map(|op| op.output_value_ids.iter().copied())
                .filter_map(|value_id| {
                    value_registers
                        .get(&value_id)
                        .map(|register| SassLiftedValueRef {
                            value_id,
                            register: register.clone(),
                        })
                })
                .collect::<Vec<_>>();
            lift_op(
                op,
                value_op.and_then(|op| op.block_id),
                input_refs_by_address
                    .get(&op.address)
                    .cloned()
                    .unwrap_or_default(),
                outputs,
            )
        })
        .collect();

    SassLiftedFunction {
        name: function.name.clone(),
        ops,
    }
}

fn lift_op(
    op: &KernelIrOp,
    block_id: Option<usize>,
    inputs: Vec<SassLiftedValueRef>,
    outputs: Vec<SassLiftedValueRef>,
) -> SassLiftedOp {
    let (class, kind) = classify_op(&op.kind);
    SassLiftedOp {
        address: op.address,
        block_id,
        predicate: op.predicate.clone(),
        opcode: SassOpcode::from_ir_op(op),
        class,
        kind,
        semantics: lift_semantics(&op.kind),
        inputs,
        outputs,
        source_operands: op.source_operands.clone(),
        detail: SassLiftedOpDetail::new(class, kind),
        source: op.source.clone(),
    }
}
