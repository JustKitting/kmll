use std::fmt;

use super::super::super::sass::{SassOperand, SassOperandKind, label_in_text};
use super::{MemoryAddress, RegisterRef, RegisterRefKind, SassSymbol};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScalarOperand {
    pub kind: ScalarOperandKind,
    pub raw: String,
}

impl ScalarOperand {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        let register = RegisterRef::parse(raw.clone());
        let kind = if matches!(register.kind, RegisterRefKind::Raw) {
            parse_immediate_operand(&raw)
                .map(ScalarOperandKind::Immediate)
                .unwrap_or(ScalarOperandKind::Raw)
        } else {
            ScalarOperandKind::Register(register)
        };
        Self { kind, raw }
    }

    pub fn register(raw: String, register: RegisterRef) -> Self {
        Self {
            kind: ScalarOperandKind::Register(register),
            raw,
        }
    }

    pub fn immediate(raw: String, immediate: ImmediateValue) -> Self {
        Self {
            kind: ScalarOperandKind::Immediate(immediate),
            raw,
        }
    }

    pub fn raw(raw: String) -> Self {
        Self {
            kind: ScalarOperandKind::Raw,
            raw,
        }
    }

    pub fn registers(&self) -> Vec<RegisterRef> {
        match &self.kind {
            ScalarOperandKind::Register(register) => vec![register.clone()],
            ScalarOperandKind::Immediate(_) | ScalarOperandKind::Raw => Vec::new(),
        }
    }

    pub fn as_register(&self) -> Option<&RegisterRef> {
        match &self.kind {
            ScalarOperandKind::Register(register) => Some(register),
            ScalarOperandKind::Immediate(_) | ScalarOperandKind::Raw => None,
        }
    }
}

impl fmt::Display for ScalarOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarOperandKind {
    Register(RegisterRef),
    Immediate(ImmediateValue),
    Raw,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImmediateValue {
    Integer(i128),
    FloatBits(u64),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AggregateOperand {
    pub kind: AggregateOperandKind,
    pub raw: String,
}

impl AggregateOperand {
    pub fn from_sass_operand(operand: &SassOperand) -> Self {
        let raw = operand.raw.clone();
        let kind = match &operand.kind {
            SassOperandKind::Register(register) => AggregateOperandKind::Register(
                RegisterRef::from_sass_register(raw.clone(), register),
            ),
            SassOperandKind::Immediate(immediate) => parse_immediate_operand(immediate)
                .map(AggregateOperandKind::Immediate)
                .unwrap_or_else(|| raw_aggregate_operand(&raw)),
            SassOperandKind::ConstantMemory { bank, offset } => AggregateOperandKind::Memory(
                MemoryAddress::constant(raw.clone(), bank.clone(), offset.clone()),
            ),
            SassOperandKind::DescriptorMemory {
                descriptor,
                address,
                address_width,
                offset,
            } => AggregateOperandKind::Memory(MemoryAddress::descriptor(
                raw.clone(),
                descriptor.clone(),
                address.clone(),
                *address_width,
                offset.clone(),
            )),
            SassOperandKind::IndexedMemory { base, offset } => AggregateOperandKind::Memory(
                MemoryAddress::indexed(raw.clone(), base.clone(), offset.clone()),
            ),
            SassOperandKind::Label(label) => {
                AggregateOperandKind::Label(SassSymbol::new(label.clone()))
            }
            SassOperandKind::Raw => raw_aggregate_operand(&raw),
        };
        Self { kind, raw }
    }

    pub fn registers(&self) -> Vec<RegisterRef> {
        match &self.kind {
            AggregateOperandKind::Register(register) => vec![register.clone()],
            AggregateOperandKind::Memory(address) => address.registers(),
            AggregateOperandKind::Raw { registers, .. } => registers.clone(),
            AggregateOperandKind::Immediate(_) | AggregateOperandKind::Label(_) => Vec::new(),
        }
    }

    pub fn single_register(&self) -> Option<&RegisterRef> {
        match &self.kind {
            AggregateOperandKind::Register(register) => Some(register),
            AggregateOperandKind::Raw { registers, .. } => match registers.as_slice() {
                [register] => Some(register),
                _ => None,
            },
            AggregateOperandKind::Immediate(_)
            | AggregateOperandKind::Memory(_)
            | AggregateOperandKind::Label(_) => None,
        }
    }

    pub fn as_scalar_operand(&self) -> ScalarOperand {
        match &self.kind {
            AggregateOperandKind::Register(register) => {
                ScalarOperand::register(self.raw.clone(), register.clone())
            }
            AggregateOperandKind::Immediate(immediate) => {
                ScalarOperand::immediate(self.raw.clone(), immediate.clone())
            }
            AggregateOperandKind::Raw {
                registers,
                label: None,
            } => match registers.as_slice() {
                [register] => ScalarOperand::register(self.raw.clone(), register.clone()),
                _ => ScalarOperand::raw(self.raw.clone()),
            },
            AggregateOperandKind::Raw { label: Some(_), .. }
            | AggregateOperandKind::Memory(_)
            | AggregateOperandKind::Label(_) => ScalarOperand::raw(self.raw.clone()),
        }
    }
}

impl fmt::Display for AggregateOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AggregateOperandKind {
    Register(RegisterRef),
    Immediate(ImmediateValue),
    Memory(MemoryAddress),
    Label(SassSymbol),
    Raw {
        registers: Vec<RegisterRef>,
        label: Option<SassSymbol>,
    },
}

fn raw_aggregate_operand(raw: &str) -> AggregateOperandKind {
    AggregateOperandKind::Raw {
        registers: RegisterRef::extract_all(raw),
        label: label_in_text(raw).map(SassSymbol::new),
    }
}

pub(super) fn parse_immediate_operand(raw: &str) -> Option<ImmediateValue> {
    let text = raw.trim();
    if text.is_empty() {
        return None;
    }
    let (negative, body) = text
        .strip_prefix('-')
        .map(|body| (true, body))
        .unwrap_or((false, text));
    let body = body.strip_prefix('+').unwrap_or(body);
    if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        let value = i128::from_str_radix(hex, 16).ok()?;
        return Some(ImmediateValue::Integer(if negative {
            value.saturating_neg()
        } else {
            value
        }));
    }
    if text.contains('.') || body.contains('e') || body.contains('E') {
        return text.parse::<f64>().ok().map(|value| {
            if value.fract() == 0.0 {
                ImmediateValue::Integer(value as i128)
            } else {
                ImmediateValue::FloatBits(value.to_bits())
            }
        });
    }
    text.parse::<i128>().ok().map(ImmediateValue::Integer)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PredicateCondition {
    pub kind: PredicateConditionKind,
    pub raw: String,
}

impl PredicateCondition {
    pub fn register(raw: String, register: RegisterRef, negated: bool) -> Self {
        Self {
            kind: PredicateConditionKind::Register { register, negated },
            raw,
        }
    }

    pub fn raw(raw: String) -> Self {
        Self {
            kind: PredicateConditionKind::Raw,
            raw,
        }
    }

    pub fn registers(&self) -> Vec<RegisterRef> {
        match &self.kind {
            PredicateConditionKind::Register { register, .. } => vec![register.clone()],
            PredicateConditionKind::Raw => Vec::new(),
        }
    }
}

impl fmt::Display for PredicateCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PredicateConditionKind {
    Register {
        register: RegisterRef,
        negated: bool,
    },
    Raw,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ControlTarget {
    pub kind: ControlTargetKind,
    pub raw: String,
}

impl ControlTarget {
    pub fn label(raw: String, label: String) -> Self {
        Self {
            kind: ControlTargetKind::Label(SassSymbol::new(label)),
            raw,
        }
    }

    pub fn address(raw: String, address: u64) -> Self {
        Self {
            kind: ControlTargetKind::Address(address),
            raw,
        }
    }

    pub fn raw(raw: String) -> Self {
        Self {
            kind: ControlTargetKind::Raw,
            raw,
        }
    }

    pub fn label_symbol(&self) -> Option<&SassSymbol> {
        match &self.kind {
            ControlTargetKind::Label(label) => Some(label),
            ControlTargetKind::Address(_) | ControlTargetKind::Raw => None,
        }
    }

    pub fn label_name(&self) -> Option<&str> {
        self.label_symbol().map(SassSymbol::as_str)
    }

    pub fn address_value(&self) -> Option<u64> {
        match self.kind {
            ControlTargetKind::Address(address) => Some(address),
            ControlTargetKind::Label(_) | ControlTargetKind::Raw => None,
        }
    }
}

impl fmt::Display for ControlTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ControlTargetKind::Label(label) => f.write_str(label.as_str()),
            ControlTargetKind::Address(_) | ControlTargetKind::Raw => f.write_str(&self.raw),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ControlTargetKind {
    Label(SassSymbol),
    Address(u64),
    Raw,
}
