use super::super::super::sass::SassInstruction;
use super::super::types::{KernelIrOpKind, SassMappingConfidence};
use super::{
    LiftResult,
    operands::{
        has_modifier, map_register_three_scalar_operands, map_register_two_scalar_operands,
        register_operand, scalar_inputs, width_modifier,
    },
};

pub(super) fn lift(
    opcode: &str,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    Some(match opcode {
        "IADD" | "IADD3" | "UIADD3" => (
            KernelIrOpKind::IntegerAdd {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                inputs: scalar_inputs(&operands.iter().skip(1).cloned().collect::<Vec<_>>()),
                width_bits: width_modifier(&instruction.modifiers),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "FADD" => map_register_two_scalar_operands(instruction, |dst, lhs, rhs| {
            KernelIrOpKind::FloatAdd { dst, lhs, rhs }
        }),
        "FMUL" => map_register_two_scalar_operands(instruction, |dst, lhs, rhs| {
            KernelIrOpKind::FloatMul { dst, lhs, rhs }
        }),
        "HADD2" => (
            KernelIrOpKind::PackedHalfAdd {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                inputs: scalar_inputs(&operands.iter().skip(1).cloned().collect::<Vec<_>>()),
                lanes: 2,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "HMUL2" => (
            KernelIrOpKind::PackedHalfMul {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                inputs: scalar_inputs(&operands.iter().skip(1).cloned().collect::<Vec<_>>()),
                lanes: 2,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "FFMA" | "HFMA2" => map_register_three_scalar_operands(instruction, |dst, a, b, c| {
            KernelIrOpKind::FusedMultiplyAdd {
                dst,
                a,
                b,
                c,
                lane_bits: (opcode == "HFMA2").then_some(16),
            }
        }),
        "IMAD" | "UIMAD" => map_register_three_scalar_operands(instruction, |dst, a, b, c| {
            KernelIrOpKind::IntegerMad {
                dst,
                a,
                b,
                c,
                wide: has_modifier(&instruction.modifiers, "WIDE"),
            }
        }),
        _ => return None,
    })
}
