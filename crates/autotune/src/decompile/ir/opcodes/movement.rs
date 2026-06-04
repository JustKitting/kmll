use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassOpcode, SassOpcodeKind,
};
use super::{
    LiftResult,
    operands::{
        map_register_register_operands, map_register_scalar_operands, operand_tail,
        register_operand, scalar_inputs, scalar_operand, unsupported_arity,
    },
};

pub(super) fn lift(opcode: &SassOpcode, operands: &[AggregateOperand]) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Cs2r | SassOpcodeKind::S2r | SassOpcodeKind::S2ur => {
            map_register_register_operands(opcode, operands, |dst, special| {
                KernelIrOpKind::ReadSpecialRegister { dst, special }
            })
        }
        SassOpcodeKind::Mov | SassOpcodeKind::Movm | SassOpcodeKind::Umov => {
            map_register_scalar_operands(opcode, operands, |dst, src| KernelIrOpKind::Move {
                dst,
                src,
            })
        }
        SassOpcodeKind::Sel => {
            if operands.len() == 4 {
                (
                    KernelIrOpKind::Select {
                        dst: register_operand(operands.first()),
                        true_value: scalar_operand(operands.get(1)),
                        false_value: scalar_operand(operands.get(2)),
                        predicate: register_operand(operands.get(3)),
                    },
                    SassMappingConfidence::OpcodeHeuristic,
                )
            } else {
                unsupported_arity(opcode, operands.len(), 4)
            }
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
