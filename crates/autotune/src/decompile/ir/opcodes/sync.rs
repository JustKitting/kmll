use super::super::types::{KernelIrOpKind, SassMappingConfidence};
use super::LiftResult;

pub(super) fn lift(opcode: &str, operands: &[String]) -> Option<LiftResult> {
    Some(match opcode {
        "BSSY" | "BSYNC" | "BAR" => (
            KernelIrOpKind::Sync {
                kind: opcode.to_string(),
                operands: operands.to_vec(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
