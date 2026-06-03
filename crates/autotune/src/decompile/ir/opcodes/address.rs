use super::super::types::{KernelIrOpKind, SassMappingConfidence, SassOpcode, SassOpcodeKind};
use super::{
    LiftResult,
    operands::{register_operand, scalar_inputs},
};

pub(super) fn lift(opcode: &SassOpcode, operands: &[String]) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Lea | SassOpcodeKind::Ulea => (
            KernelIrOpKind::AddressCalc {
                dst: register_operand(operands.first().map(String::as_str).unwrap_or_default()),
                inputs: scalar_inputs(&operands.iter().skip(1).cloned().collect::<Vec<_>>()),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}
