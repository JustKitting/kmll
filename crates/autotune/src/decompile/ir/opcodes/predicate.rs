use super::super::super::sass::SassInstruction;
use super::super::types::{
    KernelIrOpKind, SassCompareDType, SassComparisonKind, SassMappingConfidence,
};
use super::{
    LiftResult,
    operands::{register_operand, scalar_operand},
};

pub(super) fn lift(
    opcode: &str,
    instruction: &SassInstruction,
    operands: &[String],
) -> Option<LiftResult> {
    Some(match opcode {
        "ISETP" | "UISETP" | "FSETP" => (
            KernelIrOpKind::CompareSet {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                comparison: instruction
                    .modifiers
                    .first()
                    .map(|modifier| SassComparisonKind::parse(modifier.as_str())),
                dtype: instruction
                    .modifiers
                    .iter()
                    .find(|modifier| is_compare_dtype_modifier(modifier))
                    .map(|modifier| SassCompareDType::parse(modifier.as_str())),
                lhs: scalar_operand(operands.get(2).map(String::as_str).unwrap_or_default()),
                rhs: scalar_operand(operands.get(3).map(String::as_str).unwrap_or_default()),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}

fn is_compare_dtype_modifier(modifier: &str) -> bool {
    let mut chars = modifier.chars();
    let Some(prefix) = chars.next() else {
        return false;
    };
    let mut saw_digit = false;
    for ch in chars {
        if !ch.is_ascii_digit() {
            return false;
        }
        saw_digit = true;
    }
    matches!(prefix, 'U' | 'S' | 'F') && saw_digit
}
