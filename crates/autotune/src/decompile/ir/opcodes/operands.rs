use super::super::super::sass::{
    RegisterClass, SassInstruction, SassOperandKind, SassPredicate, SassRegister, label_in_text,
};
use super::super::types::{
    AggregateOperand, AggregateOperandKind, ControlTarget, KernelIrOpKind, PredicateCondition,
    RegisterRef, SassMappingConfidence, SassOpcode, SassUnsupportedReason, ScalarOperand,
    ScalarOperandKind,
};
use super::LiftResult;

pub(super) fn aggregate_operands(instruction: &SassInstruction) -> Vec<AggregateOperand> {
    instruction
        .operands
        .iter()
        .map(AggregateOperand::from_sass_operand)
        .collect()
}

pub(super) fn register_operand(operand: Option<&AggregateOperand>) -> RegisterRef {
    match operand {
        Some(AggregateOperand {
            kind: AggregateOperandKind::Register(register),
            ..
        }) => register.clone(),
        Some(operand) => RegisterRef::parse(operand.raw.clone()),
        None => RegisterRef::parse(String::new()),
    }
}

pub(super) fn scalar_operand(operand: Option<&AggregateOperand>) -> ScalarOperand {
    match operand {
        Some(AggregateOperand {
            kind: AggregateOperandKind::Register(register),
            raw,
        }) => ScalarOperand {
            kind: ScalarOperandKind::Register(register.clone()),
            raw: raw.clone(),
        },
        Some(AggregateOperand {
            kind: AggregateOperandKind::Immediate(immediate),
            raw,
        }) => ScalarOperand {
            kind: ScalarOperandKind::Immediate(immediate.clone()),
            raw: raw.clone(),
        },
        Some(operand) => ScalarOperand::parse(operand.raw.clone()),
        None => ScalarOperand::parse(String::new()),
    }
}

pub(super) fn scalar_inputs(operands: &[AggregateOperand]) -> Vec<ScalarOperand> {
    operands
        .iter()
        .map(|operand| scalar_operand(Some(operand)))
        .collect()
}

pub(super) fn operand_tail(operands: &[AggregateOperand]) -> &[AggregateOperand] {
    operands.get(1..).unwrap_or(&[])
}

pub(super) fn map_register_register_operands(
    opcode: &SassOpcode,
    operands: &[AggregateOperand],
    f: impl FnOnce(RegisterRef, RegisterRef) -> KernelIrOpKind,
) -> LiftResult {
    if operands.len() == 2 {
        (
            f(
                register_operand(operands.first()),
                register_operand(operands.get(1)),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(opcode, operands.len(), 2)
    }
}

pub(super) fn map_register_scalar_operands(
    opcode: &SassOpcode,
    operands: &[AggregateOperand],
    f: impl FnOnce(RegisterRef, ScalarOperand) -> KernelIrOpKind,
) -> LiftResult {
    if operands.len() == 2 {
        (
            f(
                register_operand(operands.first()),
                scalar_operand(operands.get(1)),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(opcode, operands.len(), 2)
    }
}

pub(super) fn map_register_two_scalar_operands(
    opcode: &SassOpcode,
    operands: &[AggregateOperand],
    f: impl FnOnce(RegisterRef, ScalarOperand, ScalarOperand) -> KernelIrOpKind,
) -> LiftResult {
    if operands.len() == 3 {
        (
            f(
                register_operand(operands.first()),
                scalar_operand(operands.get(1)),
                scalar_operand(operands.get(2)),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(opcode, operands.len(), 3)
    }
}

pub(super) fn map_register_three_scalar_operands(
    opcode: &SassOpcode,
    operands: &[AggregateOperand],
    f: impl FnOnce(RegisterRef, ScalarOperand, ScalarOperand, ScalarOperand) -> KernelIrOpKind,
) -> LiftResult {
    if operands.len() >= 4 {
        (
            f(
                register_operand(operands.first()),
                scalar_operand(operands.get(1)),
                scalar_operand(operands.get(2)),
                scalar_operand(operands.get(3)),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(opcode, operands.len(), 4)
    }
}

pub(super) fn map_warp_shuffle_operands(
    opcode: &SassOpcode,
    operands: &[AggregateOperand],
    f: impl FnOnce(
        RegisterRef,
        RegisterRef,
        ScalarOperand,
        ScalarOperand,
        ScalarOperand,
    ) -> KernelIrOpKind,
) -> LiftResult {
    if operands.len() >= 5 {
        (
            f(
                register_operand(operands.first()),
                register_operand(operands.get(1)),
                scalar_operand(operands.get(2)),
                scalar_operand(operands.get(3)),
                scalar_operand(operands.get(4)),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(opcode, operands.len(), 5)
    }
}

pub(super) fn unsupported_arity(opcode: &SassOpcode, actual: usize, expected: usize) -> LiftResult {
    (
        KernelIrOpKind::Unsupported {
            opcode: opcode.clone(),
            reason: SassUnsupportedReason::at_least_operand_arity(expected, actual),
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
