use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassOpcode, SassOpcodeKind,
};
use super::{
    LiftResult,
    operands::{operand_tail, register_operand, scalar_inputs},
};

pub(super) fn lift(opcode: &SassOpcode, operands: &[AggregateOperand]) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Lea | SassOpcodeKind::Ulea => (
            KernelIrOpKind::AddressCalc {
                dst: register_operand(operands.first()),
                inputs: scalar_inputs(operand_tail(operands)),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
