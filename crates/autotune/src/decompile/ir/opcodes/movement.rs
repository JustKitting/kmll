use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassOpcode, SassOpcodeKind,
};
use super::{
    LiftResult,
    operands::{
        map_register_register_operands, map_register_scalar_operands, operand_tail,
        register_operand, scalar_inputs,
    },
};

pub(super) fn lift(opcode: &SassOpcode, operands: &[AggregateOperand]) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Cs2r | SassOpcodeKind::S2r | SassOpcodeKind::S2ur => {
            map_register_register_operands(opcode, operands, |dst, special| {
                KernelIrOpKind::ReadSpecialRegister { dst, special }
            })
        }
        SassOpcodeKind::Mov | SassOpcodeKind::Umov => {
            map_register_scalar_operands(opcode, operands, |dst, src| KernelIrOpKind::Move {
                dst,
                src,
            })
        }
        SassOpcodeKind::Prmt => (
            KernelIrOpKind::Permute {
                dst: register_operand(operands.first()),
                inputs: scalar_inputs(operand_tail(operands)),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
