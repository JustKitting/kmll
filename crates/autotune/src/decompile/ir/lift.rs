use super::super::sass::{SassInstruction, SassModule};
use super::{
    opcodes::{lift_kind, predicate_text},
    types::{KernelIrFunction, KernelIrModule, KernelIrOp},
};

pub fn lift_sass_module(module: &SassModule) -> KernelIrModule {
    KernelIrModule {
        target: module.target.clone(),
        functions: module
            .functions
            .iter()
            .map(|function| KernelIrFunction {
                name: function.name.clone(),
                ops: function.instructions.iter().map(lift_instruction).collect(),
            })
            .collect(),
    }
}

fn lift_instruction(instruction: &SassInstruction) -> KernelIrOp {
    let source_operands = instruction
        .operands
        .iter()
        .map(|operand| operand.raw.clone())
        .collect::<Vec<_>>();
    let (kind, confidence) = lift_kind(instruction);
    KernelIrOp {
        address: instruction.address,
        label: instruction.label.clone(),
        predicate: instruction.predicate.as_ref().map(predicate_text),
        kind,
        confidence,
        source_opcode: instruction.opcode.clone(),
        source_modifiers: instruction.modifiers.clone(),
        source_operands,
        source: instruction.raw.clone(),
    }
}
