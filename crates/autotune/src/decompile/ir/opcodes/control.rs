use super::super::super::sass::SassInstruction;
use super::super::types::{AggregateOperand, KernelIrOpKind, SassMappingConfidence};
use super::{
    LiftResult,
    operands::{branch_condition_operand, predicate_condition, target_operand},
};

pub(super) fn lift(
    opcode: &str,
    instruction: &SassInstruction,
    operands: &[AggregateOperand],
) -> Option<LiftResult> {
    Some(match opcode {
        "NOP" => (KernelIrOpKind::NoOp, SassMappingConfidence::LocallyParsed),
        "EXIT" => (
            KernelIrOpKind::Exit {
                condition: instruction.predicate.as_ref().map(predicate_condition),
            },
            SassMappingConfidence::LocallyParsed,
        ),
        "BRA" => (
            KernelIrOpKind::Branch {
                target: target_operand(instruction),
                condition: instruction
                    .predicate
                    .as_ref()
                    .map(predicate_condition)
                    .or_else(|| branch_condition_operand(instruction)),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "CALL" => (
            KernelIrOpKind::Call {
                target: target_operand(instruction),
                operands: operands.to_vec(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "RET" => (
            KernelIrOpKind::Return {
                target: target_operand(instruction),
                operands: operands.to_vec(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
