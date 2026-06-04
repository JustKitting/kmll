use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassCompareDType, SassComparisonKind, SassMappingConfidence,
    SassModifier, SassModifierKind, SassOpcode, SassOpcodeKind,
};
use super::{
    LiftResult,
    operands::{register_operand, scalar_operand},
};

pub(super) fn lift(
    opcode: &SassOpcode,
    modifiers: &[SassModifier],
    operands: &[AggregateOperand],
) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Isetp | SassOpcodeKind::Uisetp | SassOpcodeKind::Fsetp => (
            KernelIrOpKind::CompareSet {
                dst: register_operand(operands.first()),
                comparison: modifiers.first().map(comparison_kind),
                dtype: modifiers
                    .iter()
                    .find_map(|modifier| compare_dtype(modifier.kind())),
                lhs: scalar_operand(operands.get(2)),
                rhs: scalar_operand(operands.get(3)),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}

fn comparison_kind(modifier: &SassModifier) -> SassComparisonKind {
    match modifier.kind() {
        SassModifierKind::Equal => SassComparisonKind::Equal,
        SassModifierKind::NotEqual => SassComparisonKind::NotEqual,
        SassModifierKind::LessThan => SassComparisonKind::LessThan,
        SassModifierKind::LessEqual => SassComparisonKind::LessEqual,
        SassModifierKind::GreaterThan => SassComparisonKind::GreaterThan,
        SassModifierKind::GreaterEqual => SassComparisonKind::GreaterEqual,
        SassModifierKind::Low => SassComparisonKind::Lower,
        SassModifierKind::LowerSame => SassComparisonKind::LowerSame,
        SassModifierKind::High => SassComparisonKind::Higher,
        SassModifierKind::HigherSame => SassComparisonKind::HigherSame,
        SassModifierKind::Nan => SassComparisonKind::Nan,
        SassModifierKind::Num => SassComparisonKind::Num,
        SassModifierKind::Raw(raw) => SassComparisonKind::Raw(raw.clone()),
        _ => SassComparisonKind::Raw(modifier.as_str().to_string()),
    }
}

fn compare_dtype(modifier: &SassModifierKind) -> Option<SassCompareDType> {
    match modifier {
        SassModifierKind::UnsignedWidth(8) => Some(SassCompareDType::U8),
        SassModifierKind::SignedWidth(8) => Some(SassCompareDType::S8),
        SassModifierKind::UnsignedWidth(16) => Some(SassCompareDType::U16),
        SassModifierKind::SignedWidth(16) => Some(SassCompareDType::S16),
        SassModifierKind::UnsignedWidth(32) => Some(SassCompareDType::U32),
        SassModifierKind::SignedWidth(32) => Some(SassCompareDType::S32),
        SassModifierKind::UnsignedWidth(64) => Some(SassCompareDType::U64),
        SassModifierKind::SignedWidth(64) => Some(SassCompareDType::S64),
        SassModifierKind::F32 => Some(SassCompareDType::F32),
        SassModifierKind::F64 => Some(SassCompareDType::F64),
        _ => None,
    }
}
