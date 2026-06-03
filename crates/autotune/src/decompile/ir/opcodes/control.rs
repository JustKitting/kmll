use super::super::super::sass::SassInstruction;
use super::super::types::{KernelIrOpKind, SassMappingConfidence};
use super::{
    LiftResult,
    operands::{branch_condition_operand, label_operand, predicate_text},
};

pub(super) fn lift(
    opcode: &str,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    Some(match opcode {
        "NOP" => (KernelIrOpKind::NoOp, SassMappingConfidence::LocallyParsed),
        "EXIT" => (
            KernelIrOpKind::Exit {
                condition: instruction.predicate.as_ref().map(predicate_text),
            },
            SassMappingConfidence::LocallyParsed,
        ),
        "BRA" => (
            KernelIrOpKind::Branch {
                target: label_operand(instruction),
                condition: instruction
                    .predicate
                    .as_ref()
                    .map(predicate_text)
                    .or_else(|| branch_condition_operand(instruction)),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "CALL" => (
            KernelIrOpKind::Call {
                target: label_operand(instruction),
                operands: operands.to_vec(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "RET" => (
            KernelIrOpKind::Return {
                target: label_operand(instruction),
                operands: operands.to_vec(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
