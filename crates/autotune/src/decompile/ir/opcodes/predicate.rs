use super::super::super::sass::SassInstruction;
use super::super::types::{KernelIrOpKind, SassMappingConfidence};
use super::{
    LiftResult,
    operands::{register_operand, scalar_operand},
};

pub(super) fn lift(
    opcode: &str,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    Some(match opcode {
        "ISETP" | "UISETP" | "FSETP" => (
            KernelIrOpKind::CompareSet {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                comparison: instruction.modifiers.first().cloned(),
                dtype: instruction
                    .modifiers
                    .iter()
                    .find(|modifier| modifier.starts_with('U') || modifier.starts_with('S'))
                    .cloned(),
                lhs: scalar_operand(operands.get(2).map(String::as_str).unwrap_or_default()),
                rhs: scalar_operand(operands.get(3).map(String::as_str).unwrap_or_default()),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
