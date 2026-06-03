use super::super::super::sass::{
    RegisterClass, SassInstruction, SassOperandKind, SassPredicate, SassRegister, label_in_text,
};
use super::super::types::{
    ControlTarget, KernelIrOpKind, PredicateCondition, RegisterRef, SassMappingConfidence,
};
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

pub(super) fn predicate_condition(predicate: &SassPredicate) -> PredicateCondition {
    if predicate.negated {
        PredicateCondition::register(
            format!("!{}", predicate.register),
            RegisterRef::parse(predicate.register.clone()),
            true,
        )
    } else {
        PredicateCondition::register(
            predicate.register.clone(),
            RegisterRef::parse(predicate.register.clone()),
            false,
        )
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

pub(super) fn branch_condition_operand(
    instruction: &SassInstruction,
) -> Option<PredicateCondition> {
    instruction.operands.iter().find_map(|operand| {
        let SassOperandKind::Register(
            register @ SassRegister {
                class:
                    RegisterClass::Predicate
                    | RegisterClass::UniformPredicate
                    | RegisterClass::PredicateTrue
                    | RegisterClass::UniformPredicateTrue,
                ..
            },
        ) = &operand.kind
        else {
            return None;
        };
        let register_text =
            predicate_register_text(register).unwrap_or_else(|| register_text(&operand.raw));
        Some(PredicateCondition::register(
            operand.raw.clone(),
            RegisterRef::from_sass_register(register_text, register),
            register.negated,
        ))
    })
}

fn predicate_register_text(register: &SassRegister) -> Option<String> {
    match register.class {
        RegisterClass::Predicate => register.index.map(|index| format!("P{index}")),
        RegisterClass::UniformPredicate => register.index.map(|index| format!("UP{index}")),
        RegisterClass::PredicateTrue => Some("PT".to_string()),
        RegisterClass::UniformPredicateTrue => Some("UPT".to_string()),
        _ => None,
    }
}

fn register_text(raw: &str) -> String {
    raw.trim()
        .trim_start_matches('!')
        .trim_start_matches('-')
        .trim_matches('|')
        .to_string()
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
