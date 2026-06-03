use super::super::super::sass::SassInstruction;
use super::super::types::{KernelIrOpKind, SassMappingConfidence};
use super::LiftResult;

pub(super) fn lift(
    opcode: &str,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    Some(match opcode {
        "ISETP" | "UISETP" | "FSETP" => (
            KernelIrOpKind::CompareSet {
                dst: operands.first().cloned().unwrap_or_default(),
                comparison: instruction.modifiers.first().cloned(),
                dtype: instruction
                    .modifiers
                    .iter()
                    .find(|modifier| modifier.starts_with('U') || modifier.starts_with('S'))
                    .cloned(),
                lhs: operands.get(2).cloned().unwrap_or_default(),
                rhs: operands.get(3).cloned().unwrap_or_default(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
