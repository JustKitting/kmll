use super::super::super::sass::SassInstruction;
use super::super::types::{
    KernelIrOpKind, SassCompareDType, SassComparisonKind, SassMappingConfidence, SassOpcode,
    SassOpcodeKind,
};
use super::{
    LiftResult,
    operands::{register_operand, scalar_operand},
};

pub(super) fn lift(
    opcode: &SassOpcode,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Isetp | SassOpcodeKind::Uisetp | SassOpcodeKind::Fsetp => (
            KernelIrOpKind::CompareSet {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                comparison: instruction
                    .modifiers
                    .first()
                    .map(|modifier| SassComparisonKind::parse(modifier.as_str())),
                dtype: instruction
                    .modifiers
                    .iter()
                    .find_map(|modifier| SassCompareDType::parse_known(modifier.as_str())),
                lhs: scalar_operand(operands.get(2).map(String::as_str).unwrap_or_default()),
                rhs: scalar_operand(operands.get(3).map(String::as_str).unwrap_or_default()),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
