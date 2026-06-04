use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassModifier, SassNumericDType,
    SassOpcode, SassOpcodeKind,
};
use super::{
    LiftResult,
    operands::{register_operand, scalar_operand, unsupported_arity},
};

pub(super) fn lift(
    opcode: &SassOpcode,
    modifiers: &[SassModifier],
    operands: &[AggregateOperand],
) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::I2f | SassOpcodeKind::I2fp => {
            if operands.len() == 2 {
                let (dst_dtype, src_dtype) = conversion_dtypes(modifiers);
                (
                    KernelIrOpKind::NumericConvert {
                        dst: register_operand(operands.first()),
                        src: scalar_operand(operands.get(1)),
                        dst_dtype,
                        src_dtype,
                    },
                    SassMappingConfidence::OpcodeHeuristic,
                )
            } else {
                unsupported_arity(opcode, operands.len(), 2)
            }
        }
        _ => return None,
    })
}

fn conversion_dtypes(
    modifiers: &[SassModifier],
) -> (Option<SassNumericDType>, Option<SassNumericDType>) {
    let mut dtypes = modifiers
        .iter()
        .filter_map(|modifier| SassNumericDType::from_modifier(modifier.kind()));
    (dtypes.next(), dtypes.next())
}
