use super::super::SassTarget;
use super::super::sass::{SassInstruction, SassModule};
use super::{
    opcodes::{SassLiftInput, aggregate_operands, lift_kind, predicate_condition},
    types::{KernelIrFunction, KernelIrModule, KernelIrOp, SassModifier, SassOpcode, SassSymbol},
};

pub fn lift_sass_module(module: &SassModule) -> KernelIrModule {
    KernelIrModule {
        target: module.target.clone().map(SassTarget::parse),
        functions: module
            .functions
            .iter()
            .map(|function| KernelIrFunction {
                name: SassSymbol::new(function.name.clone()),
                ops: function.instructions.iter().map(lift_instruction).collect(),
            })
            .collect(),
    }
}

fn lift_instruction(instruction: &SassInstruction) -> KernelIrOp {
    let source_opcode = SassOpcode::new(instruction.opcode.clone());
    let source_operands = aggregate_operands(instruction);
    let source_modifiers = instruction
        .modifiers
        .iter()
        .map(|modifier| SassModifier::parse(modifier.as_str()))
        .collect::<Vec<_>>();
    let lift_input = SassLiftInput {
        instruction,
        opcode: &source_opcode,
        modifiers: &source_modifiers,
        aggregate_operands: &source_operands,
    };
    let (kind, confidence) = lift_kind(&lift_input);
    KernelIrOp {
        address: instruction.address,
        source_position: instruction.source_position,
        label: instruction.label.clone().map(SassSymbol::new),
        predicate: instruction.predicate.as_ref().map(predicate_condition),
        kind,
        confidence,
        source_opcode,
        source_modifiers,
        source_operands,
        source_text: instruction.raw.clone(),
    }
}
