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
use super::types::{
    AggregateOperand, KernelIrOpKind, PredicateCondition, SassMappingConfidence, SassModifier,
    SassOpcode, SassUnsupportedReason,
};

pub(super) type LiftResult = (KernelIrOpKind, SassMappingConfidence);

pub(super) fn predicate_condition(predicate: &SassPredicate) -> PredicateCondition {
    operands::predicate_condition(predicate)
}

pub(super) fn aggregate_operands(instruction: &SassInstruction) -> Vec<AggregateOperand> {
    operands::aggregate_operands(instruction)
}

pub(super) struct SassLiftInput<'a> {
    pub instruction: &'a SassInstruction,
    pub opcode: &'a SassOpcode,
    pub modifiers: &'a [SassModifier],
    pub aggregate_operands: &'a [AggregateOperand],
}

pub(super) fn lift_kind(input: &SassLiftInput<'_>) -> LiftResult {
    control::lift(input.opcode, input.instruction, input.aggregate_operands)
        .or_else(|| warp::lift(input.opcode, input.modifiers, input.aggregate_operands))
        .or_else(|| tensor::lift(input.opcode, input.modifiers, input.aggregate_operands))
        .or_else(|| movement::lift(input.opcode, input.aggregate_operands))
        .or_else(|| memory::lift(input.opcode, input.instruction, input.modifiers))
        .or_else(|| math::lift(input.opcode, input.modifiers, input.aggregate_operands))
        .or_else(|| predicate::lift(input.opcode, input.modifiers, input.aggregate_operands))
        .or_else(|| bitwise::lift(input.opcode, input.aggregate_operands))
        .or_else(|| address::lift(input.opcode, input.aggregate_operands))
        .or_else(|| sync::lift(input.opcode, input.aggregate_operands))
        .unwrap_or_else(|| unsupported_opcode(input.opcode.clone()))
}

fn unsupported_opcode(opcode: SassOpcode) -> LiftResult {
    (
        KernelIrOpKind::Unsupported {
            opcode,
            reason: SassUnsupportedReason::no_local_mapping(),
        },
        SassMappingConfidence::Unsupported,
    )
}
