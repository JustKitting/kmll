use super::super::super::sass::{SassInstruction, SassPredicate, label_in_text};
use super::super::types::{
    AggregateOperand, AggregateOperandKind, ControlTarget, ImmediateValue, KernelIrOpKind,
    PredicateCondition, RegisterRef, RegisterRefKind, SassMappingConfidence, SassOpcode,
    SassUnsupportedReason, ScalarOperand, ScalarOperandKind,
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

pub(super) fn target_operand(operands: &[AggregateOperand]) -> Option<ControlTarget> {
    operands.iter().find_map(|operand| match &operand.kind {
        AggregateOperandKind::Label(label) => Some(ControlTarget::label(
            operand.raw.clone(),
            label.as_str().to_string(),
        )),
        AggregateOperandKind::Immediate(immediate) => {
            let raw = operand.raw.clone();
            parse_address_target(immediate)
                .map(|address| ControlTarget::address(raw.clone(), address))
                .or_else(|| Some(ControlTarget::raw(raw)))
        }
        _ => label_in_text(&operand.raw)
            .map(|label| ControlTarget::label(operand.raw.clone(), label)),
    })
}

fn parse_address_target(target: &ImmediateValue) -> Option<u64> {
    match target {
        ImmediateValue::Integer(value) => (*value).try_into().ok(),
        ImmediateValue::FloatBits(_) => None,
    }
}

pub(super) fn branch_condition_operand(
    operands: &[AggregateOperand],
) -> Option<PredicateCondition> {
    operands.iter().find_map(|operand| {
        let AggregateOperandKind::Register(register) = &operand.kind else {
            return None;
        };
        if !matches!(
            register.kind,
            RegisterRefKind::Predicate(_)
                | RegisterRefKind::UniformPredicate(_)
                | RegisterRefKind::PredicateTrue
                | RegisterRefKind::UniformPredicateTrue
        ) {
            return None;
        }
        Some(PredicateCondition::register(
            operand.raw.clone(),
            register.clone(),
            operand.raw.trim_start().starts_with('!'),
        ))
    })
}
