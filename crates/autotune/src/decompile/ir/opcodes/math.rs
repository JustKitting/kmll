use super::super::super::sass::SassInstruction;
use super::super::types::{KernelIrOpKind, SassMappingConfidence};
use super::{
    LiftResult,
    operands::{has_modifier, map_four_operands, map_three_operands, width_modifier},
};

pub(super) fn lift(
    opcode: &str,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    Some(match opcode {
        "IADD" | "IADD3" | "UIADD3" => (
            KernelIrOpKind::IntegerAdd {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
                width_bits: width_modifier(&instruction.modifiers),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "FADD" => map_three_operands(instruction, |dst, lhs, rhs| KernelIrOpKind::FloatAdd {
            dst,
            lhs,
            rhs,
        }),
        "FMUL" => map_three_operands(instruction, |dst, lhs, rhs| KernelIrOpKind::FloatMul {
            dst,
            lhs,
            rhs,
        }),
        "HADD2" => (
            KernelIrOpKind::PackedHalfAdd {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
                lanes: 2,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "HMUL2" => (
            KernelIrOpKind::PackedHalfMul {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
                lanes: 2,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "FFMA" | "HFMA2" => map_four_operands(instruction, |dst, a, b, c| {
            KernelIrOpKind::FusedMultiplyAdd {
                dst,
                a,
                b,
                c,
                lane_bits: (opcode == "HFMA2").then_some(16),
            }
        }),
        "IMAD" | "UIMAD" => {
            map_four_operands(instruction, |dst, a, b, c| KernelIrOpKind::IntegerMad {
                dst,
                a,
                b,
                c,
                wide: has_modifier(&instruction.modifiers, "WIDE"),
            })
        }
        _ => return None,
    })
}
