mod address;
mod bitwise;
mod control;
mod math;
mod memory;
mod movement;
mod operands;
mod predicate;
mod sync;
mod tensor;
mod warp;

use super::super::sass::{SassInstruction, SassPredicate};
use super::types::{KernelIrOpKind, PredicateCondition, SassMappingConfidence};

pub(super) type LiftResult = (KernelIrOpKind, SassMappingConfidence);

pub(super) fn predicate_condition(predicate: &SassPredicate) -> PredicateCondition {
    operands::predicate_condition(predicate)
}

pub(super) fn lift_kind(instruction: &SassInstruction) -> LiftResult {
    let operands = operands::raw_operands(instruction);
    let opcode = instruction.opcode.as_str();
    control::lift(opcode, instruction, &operands)
        .or_else(|| warp::lift(opcode, instruction))
        .or_else(|| tensor::lift(opcode, &operands))
        .or_else(|| movement::lift(opcode, instruction, &operands))
        .or_else(|| memory::lift(opcode, instruction))
        .or_else(|| math::lift(opcode, instruction, &operands))
        .or_else(|| predicate::lift(opcode, instruction, &operands))
        .or_else(|| bitwise::lift(opcode, &operands))
        .or_else(|| address::lift(opcode, &operands))
        .or_else(|| sync::lift(opcode, &operands))
        .unwrap_or_else(|| unsupported_opcode(instruction))
}

fn unsupported_opcode(instruction: &SassInstruction) -> LiftResult {
    (
        KernelIrOpKind::Unsupported {
            opcode: instruction.opcode.clone(),
            reason: "no local mapping for opcode yet".to_string(),
        },
        SassMappingConfidence::Unsupported,
    )
}
