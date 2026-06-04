use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassModifier, SassModifierKind,
    SassOpcode, SassOpcodeKind, SassWarpShuffleMode,
};
use super::{
    LiftResult,
    operands::{map_warp_shuffle_operands, register_operand},
};

pub(super) fn lift(
    opcode: &SassOpcode,
    modifiers: &[SassModifier],
    operands: &[AggregateOperand],
) -> Option<LiftResult> {
    match opcode.kind() {
        SassOpcodeKind::Elect => Some((
            KernelIrOpKind::WarpElect {
                dst: register_operand(operands.first()),
                operands: operands.to_vec(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        )),
        SassOpcodeKind::Shfl => Some(map_warp_shuffle_operands(
            opcode,
            operands,
            |predicate, dst, src, offset, mask| KernelIrOpKind::WarpShuffle {
                mode: modifiers
                    .first()
                    .and_then(|modifier| warp_shuffle_mode(modifier.kind())),
                predicate,
                dst,
                src,
                offset,
                mask,
            },
        )),
        _ => None,
    }
}

fn warp_shuffle_mode(modifier: &SassModifierKind) -> Option<SassWarpShuffleMode> {
    match modifier {
        SassModifierKind::Up => Some(SassWarpShuffleMode::Up),
        SassModifierKind::Down => Some(SassWarpShuffleMode::Down),
        SassModifierKind::Bfly => Some(SassWarpShuffleMode::Bfly),
        SassModifierKind::Index => Some(SassWarpShuffleMode::Index),
        _ => None,
    }
}
