use super::super::super::sass::SassInstruction;
use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassOpcode, SassOpcodeKind,
};
use super::{
    LiftResult,
    operands::{branch_condition_operand, predicate_condition, target_operand},
};

pub(super) fn lift(
    opcode: &SassOpcode,
    instruction: &SassInstruction,
    operands: &[AggregateOperand],
) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Nop => (KernelIrOpKind::NoOp, SassMappingConfidence::LocallyParsed),
        SassOpcodeKind::Exit => (
            KernelIrOpKind::Exit {
                condition: instruction.predicate.as_ref().map(predicate_condition),
            },
            SassMappingConfidence::LocallyParsed,
        ),
        SassOpcodeKind::Bra => (
            KernelIrOpKind::Branch {
                target: target_operand(operands),
                condition: instruction
                    .predicate
                    .as_ref()
                    .map(predicate_condition)
                    .or_else(|| branch_condition_operand(operands)),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Call => (
            KernelIrOpKind::Call {
                target: target_operand(operands),
                operands: operands.to_vec(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Ret => (
            KernelIrOpKind::Return {
                target: target_operand(operands),
                operands: operands.to_vec(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
