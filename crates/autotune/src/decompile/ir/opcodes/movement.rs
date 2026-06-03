use super::super::super::sass::SassInstruction;
use super::super::types::{KernelIrOpKind, SassMappingConfidence};
use super::{
    LiftResult,
    operands::{
        map_register_register_operands, map_register_scalar_operands, register_operand,
        scalar_inputs,
    },
};

pub(super) fn lift(
    opcode: &str,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    Some(match opcode {
        "CS2R" | "S2R" | "S2UR" => map_register_register_operands(instruction, |dst, special| {
            KernelIrOpKind::ReadSpecialRegister { dst, special }
        }),
        "MOV" | "UMOV" => {
            map_register_scalar_operands(instruction, |dst, src| KernelIrOpKind::Move { dst, src })
        }
        "PRMT" => (
            KernelIrOpKind::Permute {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                inputs: scalar_inputs(&operands.iter().skip(1).cloned().collect::<Vec<_>>()),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
