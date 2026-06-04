use super::super::super::sass::SassInstruction;
use super::super::types::{
    KernelIrOpKind, SassModifier, SassModifierKind, SassOpcode, SassOpcodeKind, SassWarpShuffleMode,
};
use super::{LiftResult, operands::map_warp_shuffle_operands};

pub(super) fn lift(
    opcode: &SassOpcode,
    instruction: &SassInstruction,
    modifiers: &[SassModifier],
) -> Option<LiftResult> {
    match opcode.kind() {
        SassOpcodeKind::Shfl => Some(map_warp_shuffle_operands(
            instruction,
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
