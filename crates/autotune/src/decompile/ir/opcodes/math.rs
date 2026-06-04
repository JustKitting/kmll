use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassModifier, SassModifierKind,
    SassOpcode, SassOpcodeKind,
};
use super::{
    LiftResult,
    operands::{
        map_register_three_scalar_operands, map_register_two_scalar_operands, operand_tail,
        register_operand, scalar_inputs,
    },
};

pub(super) fn lift(
    opcode: &SassOpcode,
    modifiers: &[SassModifier],
    operands: &[AggregateOperand],
) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Iadd
        | SassOpcodeKind::Iadd3
        | SassOpcodeKind::Uiadd3
        | SassOpcodeKind::Viadd => (
            KernelIrOpKind::IntegerAdd {
                dst: register_operand(operands.first()),
                inputs: scalar_inputs(operand_tail(operands)),
                width_bits: width_modifier(modifiers),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Fadd => {
            map_register_two_scalar_operands(opcode, operands, |dst, lhs, rhs| {
                KernelIrOpKind::FloatAdd { dst, lhs, rhs }
            })
        }
        SassOpcodeKind::Fmul => {
            map_register_two_scalar_operands(opcode, operands, |dst, lhs, rhs| {
                KernelIrOpKind::FloatMul { dst, lhs, rhs }
            })
        }
        SassOpcodeKind::Hadd2 => (
            KernelIrOpKind::PackedHalfAdd {
                dst: register_operand(operands.first()),
                inputs: scalar_inputs(operand_tail(operands)),
                lanes: 2,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Hmul2 => (
            KernelIrOpKind::PackedHalfMul {
                dst: register_operand(operands.first()),
                inputs: scalar_inputs(operand_tail(operands)),
                lanes: 2,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Ffma | SassOpcodeKind::Hfma2 => {
            let lane_bits = matches!(opcode.kind(), SassOpcodeKind::Hfma2).then_some(16);
            map_register_three_scalar_operands(opcode, operands, |dst, a, b, c| {
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
            map_register_three_scalar_operands(opcode, operands, |dst, a, b, c| {
                KernelIrOpKind::IntegerMad {
                    dst,
                    a,
                    b,
                    c,
                    wide: has_modifier(modifiers, &SassModifierKind::Wide),
                }
            })
        }
        _ => return None,
    })
}

fn has_modifier(modifiers: &[SassModifier], expected: &SassModifierKind) -> bool {
    modifiers.iter().any(|modifier| modifier.kind() == expected)
}

fn width_modifier(modifiers: &[SassModifier]) -> Option<u32> {
    modifiers
        .iter()
        .find_map(|modifier| modifier.kind().width_bits())
}
