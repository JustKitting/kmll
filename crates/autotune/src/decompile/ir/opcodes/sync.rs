use super::super::types::{AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassSyncKind};
use super::LiftResult;

pub(super) fn lift(opcode: &str, operands: &[AggregateOperand]) -> Option<LiftResult> {
    Some(match opcode {
        "BSSY" | "BSYNC" | "BAR" => (
            KernelIrOpKind::Sync {
                kind: SassSyncKind::parse(opcode),
                operands: operands.to_vec(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
