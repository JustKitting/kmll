use super::super::types::{KernelIrOpKind, SassMappingConfidence};
use super::LiftResult;

pub(super) fn lift(opcode: &str, operands: &[String]) -> Option<LiftResult> {
    Some(match opcode {
        "SHF" | "USHF" => (
            KernelIrOpKind::Shift {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "LOP3" | "ULOP3" | "PLOP3" => (
            KernelIrOpKind::LogicLut {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
