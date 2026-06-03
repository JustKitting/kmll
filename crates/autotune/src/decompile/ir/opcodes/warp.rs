use super::super::super::sass::SassInstruction;
use super::super::types::{KernelIrOpKind, SassWarpShuffleMode};
use super::{LiftResult, operands::map_warp_shuffle_operands};

pub(super) fn lift(opcode: &str, instruction: &SassInstruction) -> Option<LiftResult> {
    match opcode {
        "SHFL" => Some(map_warp_shuffle_operands(
            instruction,
            |predicate, dst, src, offset, mask| KernelIrOpKind::WarpShuffle {
                mode: instruction
                    .modifiers
                    .first()
                    .map(|modifier| SassWarpShuffleMode::parse(modifier.as_str())),
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
