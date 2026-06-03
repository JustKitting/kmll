use super::super::super::sass::SassInstruction;
use super::super::types::KernelIrOpKind;
use super::{LiftResult, operands::map_five_operands};

pub(super) fn lift(opcode: &str, instruction: &SassInstruction) -> Option<LiftResult> {
    match opcode {
        "SHFL" => Some(map_five_operands(
            instruction,
            |predicate, dst, src, offset, mask| KernelIrOpKind::WarpShuffle {
                mode: instruction.modifiers.first().cloned(),
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
