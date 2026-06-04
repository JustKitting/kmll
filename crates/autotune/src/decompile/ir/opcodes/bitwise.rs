use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassOpcode, SassOpcodeKind,
};
use super::{
    LiftResult,
    operands::{operand_tail, register_operand, scalar_inputs},
};

pub(super) fn lift(opcode: &SassOpcode, operands: &[AggregateOperand]) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Shf | SassOpcodeKind::Ushf => (
            KernelIrOpKind::Shift {
                dst: register_operand(operands.first()),
                inputs: scalar_inputs(operand_tail(operands)),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Lop3 | SassOpcodeKind::Ulop3 | SassOpcodeKind::Plop3 => (
            KernelIrOpKind::LogicLut {
                dst: register_operand(operands.first()),
                inputs: scalar_inputs(operand_tail(operands)),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
