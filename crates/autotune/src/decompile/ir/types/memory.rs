use std::fmt;

use super::{ImmediateValue, RegisterRef, SassModifier, SassModifierKind, parse_immediate_operand};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    Global,
    Shared,
    Local,
    Constant,
    Descriptor,
    Unknown,
}

impl fmt::Display for MemorySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => f.write_str("global"),
            Self::Shared => f.write_str("shared"),
            Self::Local => f.write_str("local"),
            Self::Constant => f.write_str("constant"),
            Self::Descriptor => f.write_str("descriptor"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryAccessInfo {
    pub width_bits: Option<u32>,
    pub modifiers: Vec<SassMemoryModifier>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassMemoryAtomicOp {
    Add,
    Min,
    Max,
    Inc,
    Dec,
    And,
    Or,
    Xor,
    Exchange,
    CompareAndSwap,
    Raw(String),
}

impl SassMemoryAtomicOp {
    pub fn from_modifier(modifier: &SassModifier) -> Option<Self> {
        Some(match modifier.kind() {
            SassModifierKind::Add => Self::Add,
            SassModifierKind::Min => Self::Min,
            SassModifierKind::Max => Self::Max,
            SassModifierKind::Inc => Self::Inc,
            SassModifierKind::Dec => Self::Dec,
            SassModifierKind::And => Self::And,
            SassModifierKind::Or => Self::Or,
            SassModifierKind::Xor => Self::Xor,
            SassModifierKind::Exch => Self::Exchange,
            SassModifierKind::Cas => Self::CompareAndSwap,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Add => "ADD",
            Self::Min => "MIN",
            Self::Max => "MAX",
            Self::Inc => "INC",
            Self::Dec => "DEC",
            Self::And => "AND",
            Self::Or => "OR",
            Self::Xor => "XOR",
            Self::Exchange => "EXCH",
            Self::CompareAndSwap => "CAS",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassMemoryAtomicOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassMemoryModifier {
    E,
    Unsigned(u32),
    Signed(u32),
    Width(u32),
    Raw(String),
}

impl SassMemoryModifier {
    pub fn from_modifier(modifier: &SassModifier) -> Self {
        match modifier.kind() {
            SassModifierKind::E => Self::E,
            SassModifierKind::UnsignedWidth(bits) => Self::Unsigned(*bits),
            SassModifierKind::SignedWidth(bits) => Self::Signed(*bits),
            SassModifierKind::Width(bits) => Self::Width(*bits),
            _ => Self::Raw(modifier.as_str().to_string()),
        }
    }

    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        if raw == "E" {
            return Self::E;
        }
        if let Some(bits) = raw.strip_prefix('U').and_then(|bits| bits.parse().ok()) {
            return Self::Unsigned(bits);
        }
        if let Some(bits) = raw.strip_prefix('S').and_then(|bits| bits.parse().ok()) {
            return Self::Signed(bits);
        }
        raw.parse::<u32>()
            .map(Self::Width)
            .unwrap_or(Self::Raw(raw))
    }

    pub fn width_bits(&self) -> Option<u32> {
        match self {
            Self::Unsigned(bits) | Self::Signed(bits) | Self::Width(bits) => Some(*bits),
            Self::E | Self::Raw(_) => None,
        }
    }
}

impl fmt::Display for SassMemoryModifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::E => f.write_str("E"),
            Self::Unsigned(bits) => write!(f, "U{bits}"),
            Self::Signed(bits) => write!(f, "S{bits}"),
            Self::Width(bits) => write!(f, "{bits}"),
            Self::Raw(raw) => f.write_str(raw),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemoryAddress {
    pub kind: MemoryAddressKind,
    pub raw: String,
}

impl MemoryAddress {
    pub fn constant(raw: String, bank: String, offset: String) -> Self {
        Self {
            kind: MemoryAddressKind::Constant {
                bank: MemoryAddressImmediate::parse(bank),
                offset: MemoryAddressImmediate::parse(offset),
            },
            raw,
        }
    }

    pub fn descriptor(
        raw: String,
        descriptor: String,
        address: String,
        address_width: Option<u32>,
        offset: Option<String>,
    ) -> Self {
        Self {
            kind: MemoryAddressKind::Descriptor {
                descriptor: RegisterRef::parse(descriptor),
                address: RegisterRef::parse(address),
                address_width,
                offset: offset.map(MemoryAddressImmediate::parse),
            },
            raw,
        }
    }

    pub fn indexed(raw: String, base: String, offset: Option<String>) -> Self {
        Self {
            kind: MemoryAddressKind::Indexed {
                base: RegisterRef::parse(base),
                offset: offset.map(MemoryAddressImmediate::parse),
            },
            raw,
        }
    }

    pub fn raw(raw: String) -> Self {
        Self {
            kind: MemoryAddressKind::Raw,
            raw,
        }
    }

    pub fn base(&self) -> Option<MemoryAddressBase> {
        match &self.kind {
            MemoryAddressKind::Constant { bank, .. } => {
                Some(MemoryAddressBase::ConstantBank(bank.clone()))
            }
            MemoryAddressKind::Descriptor { descriptor, .. } => {
                Some(MemoryAddressBase::Descriptor(descriptor.clone()))
            }
            MemoryAddressKind::Indexed { base, .. } => {
                Some(MemoryAddressBase::Indexed(base.clone()))
            }
            MemoryAddressKind::Raw => None,
        }
    }

    pub fn offset(&self) -> Option<&MemoryAddressImmediate> {
        match &self.kind {
            MemoryAddressKind::Constant { offset, .. } => Some(offset),
            MemoryAddressKind::Descriptor { offset, .. }
            | MemoryAddressKind::Indexed { offset, .. } => offset.as_ref(),
            MemoryAddressKind::Raw => None,
        }
    }

    pub fn registers(&self) -> Vec<RegisterRef> {
        match &self.kind {
            MemoryAddressKind::Descriptor {
                descriptor,
                address,
                ..
            } => vec![descriptor.clone(), address.clone()],
            MemoryAddressKind::Indexed { base, .. } => vec![base.clone()],
            MemoryAddressKind::Constant { .. } | MemoryAddressKind::Raw => Vec::new(),
        }
    }
}

impl fmt::Display for MemoryAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryAddressKind {
    Constant {
        bank: MemoryAddressImmediate,
        offset: MemoryAddressImmediate,
    },
    Descriptor {
        descriptor: RegisterRef,
        address: RegisterRef,
        address_width: Option<u32>,
        offset: Option<MemoryAddressImmediate>,
    },
    Indexed {
        base: RegisterRef,
        offset: Option<MemoryAddressImmediate>,
    },
    Raw,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryAddressImmediate {
    pub kind: MemoryAddressImmediateKind,
    pub raw: String,
}

impl MemoryAddressImmediate {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        let kind = parse_immediate_operand(&raw)
            .map(MemoryAddressImmediateKind::Immediate)
            .unwrap_or(MemoryAddressImmediateKind::Raw);
        Self { kind, raw }
    }

    pub fn as_integer(&self) -> Option<i128> {
        match self.kind {
            MemoryAddressImmediateKind::Immediate(ImmediateValue::Integer(value)) => Some(value),
            MemoryAddressImmediateKind::Immediate(ImmediateValue::FloatBits(_))
            | MemoryAddressImmediateKind::Raw => None,
        }
    }
}

impl fmt::Display for MemoryAddressImmediate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryAddressImmediateKind {
    Immediate(ImmediateValue),
    Raw,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryAddressBase {
    ConstantBank(MemoryAddressImmediate),
    Descriptor(RegisterRef),
    Indexed(RegisterRef),
}

impl fmt::Display for MemoryAddressBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConstantBank(bank) => write!(f, "{bank}"),
            Self::Descriptor(register) | Self::Indexed(register) => write!(f, "{register}"),
        }
    }
}

impl MemoryAccessInfo {
    pub fn new(width_bits: Option<u32>, modifiers: Vec<SassMemoryModifier>) -> Self {
        Self {
            width_bits,
            modifiers,
        }
    }
}
