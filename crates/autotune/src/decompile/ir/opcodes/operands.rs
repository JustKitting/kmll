use super::super::super::sass::{
    RegisterClass, SassInstruction, SassOperandKind, SassPredicate, SassRegister, label_in_text,
};
use super::super::types::{ControlTarget, KernelIrOpKind, SassMappingConfidence};
use super::LiftResult;

pub(super) fn raw_operands(instruction: &SassInstruction) -> Vec<String> {
    instruction
        .operands
        .iter()
        .map(|operand| operand.raw.clone())
        .collect()
}

pub(super) fn map_two_operands(
    instruction: &SassInstruction,
    f: impl FnOnce(String, String) -> KernelIrOpKind,
) -> LiftResult {
    if instruction.operands.len() == 2 {
        (
            f(
                instruction.operands[0].raw.clone(),
                instruction.operands[1].raw.clone(),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(instruction, 2)
    }
}

pub(super) fn map_three_operands(
    instruction: &SassInstruction,
    f: impl FnOnce(String, String, String) -> KernelIrOpKind,
) -> LiftResult {
    if instruction.operands.len() == 3 {
        (
            f(
                instruction.operands[0].raw.clone(),
                instruction.operands[1].raw.clone(),
                instruction.operands[2].raw.clone(),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(instruction, 3)
    }
}

pub(super) fn map_four_operands(
    instruction: &SassInstruction,
    f: impl FnOnce(String, String, String, String) -> KernelIrOpKind,
) -> LiftResult {
    if instruction.operands.len() >= 4 {
        (
            f(
                instruction.operands[0].raw.clone(),
                instruction.operands[1].raw.clone(),
                instruction.operands[2].raw.clone(),
                instruction.operands[3].raw.clone(),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(instruction, 4)
    }
}

pub(super) fn map_five_operands(
    instruction: &SassInstruction,
    f: impl FnOnce(String, String, String, String, String) -> KernelIrOpKind,
) -> LiftResult {
    if instruction.operands.len() >= 5 {
        (
            f(
                instruction.operands[0].raw.clone(),
                instruction.operands[1].raw.clone(),
                instruction.operands[2].raw.clone(),
                instruction.operands[3].raw.clone(),
                instruction.operands[4].raw.clone(),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(instruction, 5)
    }
}

fn unsupported_arity(instruction: &SassInstruction, expected: usize) -> LiftResult {
    (
        KernelIrOpKind::Unsupported {
            opcode: instruction.opcode.clone(),
            reason: format!(
                "expected at least {expected} operands, saw {}",
                instruction.operands.len()
            ),
        },
        SassMappingConfidence::Unsupported,
    )
}

pub(super) fn predicate_text(predicate: &SassPredicate) -> String {
    if predicate.negated {
        format!("!{}", predicate.register)
    } else {
        predicate.register.clone()
    }
}

pub(super) fn target_operand(instruction: &SassInstruction) -> Option<ControlTarget> {
    instruction
        .operands
        .iter()
        .find_map(|operand| match &operand.kind {
            SassOperandKind::Label(label) => {
                Some(ControlTarget::label(operand.raw.clone(), label.clone()))
            }
            SassOperandKind::Immediate(target) => {
                let raw = operand.raw.clone();
                parse_address_target(target)
                    .map(|address| ControlTarget::address(raw.clone(), address))
                    .or_else(|| Some(ControlTarget::raw(raw)))
            }
            _ => label_in_text(&operand.raw)
                .map(|label| ControlTarget::label(operand.raw.clone(), label)),
        })
}

fn parse_address_target(target: &str) -> Option<u64> {
    target
        .strip_prefix("0x")
        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
        .or_else(|| target.parse::<u64>().ok())
}

pub(super) fn branch_condition_operand(instruction: &SassInstruction) -> Option<String> {
    instruction
        .operands
        .iter()
        .find(|operand| {
            matches!(
                operand.kind,
                SassOperandKind::Register(SassRegister {
                    class: RegisterClass::Predicate
                        | RegisterClass::UniformPredicate
                        | RegisterClass::PredicateTrue
                        | RegisterClass::UniformPredicateTrue,
                    ..
                })
            )
        })
        .map(|operand| operand.raw.clone())
}

pub(super) fn has_modifier(modifiers: &[String], expected: &str) -> bool {
    modifiers.iter().any(|modifier| modifier == expected)
}

pub(super) fn width_modifier(modifiers: &[String]) -> Option<u32> {
    modifiers.iter().find_map(|modifier| {
        modifier
            .strip_prefix('U')
            .or_else(|| modifier.strip_prefix('S'))
            .and_then(|bits| bits.parse::<u32>().ok())
            .or_else(|| modifier.parse::<u32>().ok())
    })
}
