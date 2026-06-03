use super::super::super::sass::SassInstruction;
use super::super::types::{
    KernelIrOpKind, SassMappingConfidence, SassModifier, SassModifierKind, SassOpcode,
    SassOpcodeKind,
};
use super::{
    LiftResult,
    operands::{
        map_register_three_scalar_operands, map_register_two_scalar_operands, register_operand,
        scalar_inputs,
    },
};

pub(super) fn lift(
    opcode: &SassOpcode,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    let modifiers = source_modifiers(instruction);
    Some(match opcode.kind() {
        SassOpcodeKind::Iadd | SassOpcodeKind::Iadd3 | SassOpcodeKind::Uiadd3 => (
            KernelIrOpKind::IntegerAdd {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                inputs: scalar_inputs(&operands.iter().skip(1).cloned().collect::<Vec<_>>()),
                width_bits: width_modifier(&modifiers),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Fadd => map_register_two_scalar_operands(instruction, |dst, lhs, rhs| {
            KernelIrOpKind::FloatAdd { dst, lhs, rhs }
        }),
        SassOpcodeKind::Fmul => map_register_two_scalar_operands(instruction, |dst, lhs, rhs| {
            KernelIrOpKind::FloatMul { dst, lhs, rhs }
        }),
        SassOpcodeKind::Hadd2 => (
            KernelIrOpKind::PackedHalfAdd {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                inputs: scalar_inputs(&operands.iter().skip(1).cloned().collect::<Vec<_>>()),
                lanes: 2,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Hmul2 => (
            KernelIrOpKind::PackedHalfMul {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                inputs: scalar_inputs(&operands.iter().skip(1).cloned().collect::<Vec<_>>()),
                lanes: 2,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Ffma | SassOpcodeKind::Hfma2 => {
            let lane_bits = matches!(opcode.kind(), SassOpcodeKind::Hfma2).then_some(16);
            map_register_three_scalar_operands(instruction, |dst, a, b, c| {
                KernelIrOpKind::FusedMultiplyAdd {
                    dst,
                    a,
                    b,
                    c,
                    lane_bits,
                }
            })
        }
        SassOpcodeKind::Imad | SassOpcodeKind::Uimad => {
            map_register_three_scalar_operands(instruction, |dst, a, b, c| {
                KernelIrOpKind::IntegerMad {
                    dst,
                    a,
                    b,
                    c,
                    wide: has_modifier(&modifiers, &SassModifierKind::Wide),
                }
            })
        }
        _ => return None,
    })
}

fn source_modifiers(instruction: &SassInstruction) -> Vec<SassModifier> {
    instruction
        .modifiers
        .iter()
        .map(|modifier| SassModifier::parse(modifier.as_str()))
        .collect()
}

fn has_modifier(modifiers: &[SassModifier], expected: &SassModifierKind) -> bool {
    modifiers.iter().any(|modifier| modifier.kind() == expected)
}

fn width_modifier(modifiers: &[SassModifier]) -> Option<u32> {
    modifiers
        .iter()
        .find_map(|modifier| modifier.kind().width_bits())
}
