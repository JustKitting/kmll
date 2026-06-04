use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassOpcode, SassOpcodeKind,
    SassSyncKind,
};
use super::LiftResult;

pub(super) fn lift(opcode: &SassOpcode, operands: &[AggregateOperand]) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Bssy
        | SassOpcodeKind::Bsync
        | SassOpcodeKind::Bar
        | SassOpcodeKind::Utmacmdflush => (
            KernelIrOpKind::Sync {
                kind: sync_kind(opcode.kind()),
                operands: operands.to_vec(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}

fn sync_kind(opcode: &SassOpcodeKind) -> SassSyncKind {
    match opcode {
        SassOpcodeKind::Bssy => SassSyncKind::BarrierSet,
        SassOpcodeKind::Bsync => SassSyncKind::BarrierSync,
        SassOpcodeKind::Bar => SassSyncKind::Barrier,
        SassOpcodeKind::Utmacmdflush => SassSyncKind::TensorMemoryCommandFlush,
        _ => unreachable!("sync lifter only calls sync_kind for sync opcodes"),
    }
}
