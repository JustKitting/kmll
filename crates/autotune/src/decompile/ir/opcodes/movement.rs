use super::super::super::sass::SassInstruction;
use super::super::types::{KernelIrOpKind, SassMappingConfidence};
use super::{LiftResult, operands::map_two_operands};

pub(super) fn lift(
    opcode: &str,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    Some(match opcode {
        "S2R" | "S2UR" => map_two_operands(instruction, |dst, special| {
            KernelIrOpKind::ReadSpecialRegister { dst, special }
        }),
        "MOV" | "UMOV" => {
            map_two_operands(instruction, |dst, src| KernelIrOpKind::Move { dst, src })
        }
        "PRMT" => (
            KernelIrOpKind::Permute {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
