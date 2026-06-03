use super::super::super::sass::SassInstruction;
use super::super::types::{KernelIrOpKind, SassMappingConfidence, SassOpcode, SassOpcodeKind};
use super::{
    LiftResult,
    operands::{
        map_register_register_operands, map_register_scalar_operands, register_operand,
        scalar_inputs,
    },
};

pub(super) fn lift(
    opcode: &SassOpcode,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Cs2r | SassOpcodeKind::S2r | SassOpcodeKind::S2ur => {
            map_register_register_operands(instruction, |dst, special| {
                KernelIrOpKind::ReadSpecialRegister { dst, special }
            })
        }
        SassOpcodeKind::Mov | SassOpcodeKind::Umov => {
            map_register_scalar_operands(instruction, |dst, src| KernelIrOpKind::Move { dst, src })
        }
        SassOpcodeKind::Prmt => (
            KernelIrOpKind::Permute {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                inputs: scalar_inputs(&operands.iter().skip(1).cloned().collect::<Vec<_>>()),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
