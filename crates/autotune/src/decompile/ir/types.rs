use std::{
    cmp::Ordering,
    fmt::{self, Write as _},
    hash::{Hash, Hasher},
};

use super::super::sass::{
    RegisterClass, SassOperand, SassOperandKind, SassRegister, SassSourcePosition,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIrModule {
    pub target: Option<String>,
    pub functions: Vec<KernelIrFunction>,
}

impl KernelIrModule {
    pub fn unsupported_instruction_count(&self) -> usize {
        self.functions
            .iter()
            .flat_map(|function| function.ops.iter())
            .filter(|op| matches!(op.kind, KernelIrOpKind::Unsupported { .. }))
            .count()
    }

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        if let Some(target) = &self.target {
            writeln!(out, "target {target}").expect("write to string");
        }
        for function in &self.functions {
            writeln!(out, "fn {} {{", function.name).expect("write to string");
            for op in &function.ops {
                writeln!(
                    out,
                    "  {:#06x}: {:?} [{}] <- {}",
                    op.address, op.kind, op.confidence, op.source
                )
                .expect("write to string");
            }
            writeln!(out, "}}").expect("write to string");
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIrFunction {
    pub name: String,
    pub ops: Vec<KernelIrOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIrOp {
    pub address: u64,
    pub source_position: SassSourcePosition,
    pub label: Option<String>,
    pub predicate: Option<PredicateCondition>,
    pub kind: KernelIrOpKind,
    pub confidence: SassMappingConfidence,
    pub source_opcode: String,
    pub source_modifiers: Vec<String>,
    pub source_operands: Vec<String>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassOpcode {
    raw: String,
}

impl SassOpcode {
    pub fn new(raw: impl Into<String>) -> Self {
        Self { raw: raw.into() }
    }

    pub fn from_ir_op(op: &KernelIrOp) -> Self {
        match &op.kind {
            KernelIrOpKind::Unsupported { opcode, .. } => opcode.clone(),
            _ => Self::new(op.source_opcode.clone()),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }
}

impl fmt::Display for SassOpcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelIrOpKind {
    ReadSpecialRegister {
        dst: RegisterRef,
        special: RegisterRef,
    },
    Move {
        dst: RegisterRef,
        src: ScalarOperand,
    },
    LoadConst {
        dst: RegisterRef,
        source: MemoryAddress,
    },
    Load {
        dst: RegisterRef,
        address: MemoryAddress,
        space: MemorySpace,
        access: MemoryAccessInfo,
    },
    Store {
        address: MemoryAddress,
        value: RegisterRef,
        space: MemorySpace,
        access: MemoryAccessInfo,
    },
    IntegerAdd {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
        width_bits: Option<u32>,
    },
    FloatAdd {
        dst: RegisterRef,
        lhs: ScalarOperand,
        rhs: ScalarOperand,
    },
    FloatMul {
        dst: RegisterRef,
        lhs: ScalarOperand,
        rhs: ScalarOperand,
    },
    PackedHalfAdd {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
        lanes: u32,
    },
    PackedHalfMul {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
        lanes: u32,
    },
    FusedMultiplyAdd {
        dst: RegisterRef,
        a: ScalarOperand,
        b: ScalarOperand,
        c: ScalarOperand,
        lane_bits: Option<u32>,
    },
    IntegerMad {
        dst: RegisterRef,
        a: ScalarOperand,
        b: ScalarOperand,
        c: ScalarOperand,
        wide: bool,
    },
    TensorCoreMma {
        opcode: SassOpcode,
        operands: Vec<AggregateOperand>,
        element_type: Option<String>,
        scope: Option<String>,
    },
    TensorCoreMemory {
        opcode: SassOpcode,
        operands: Vec<AggregateOperand>,
    },
    TensorMemoryAccess {
        opcode: SassOpcode,
        operands: Vec<AggregateOperand>,
    },
    WarpGroup {
        opcode: SassOpcode,
        operands: Vec<AggregateOperand>,
    },
    CompareSet {
        dst: RegisterRef,
        comparison: Option<String>,
        dtype: Option<String>,
        lhs: ScalarOperand,
        rhs: ScalarOperand,
    },
    Branch {
        target: Option<ControlTarget>,
        condition: Option<PredicateCondition>,
    },
    Call {
        target: Option<ControlTarget>,
        operands: Vec<AggregateOperand>,
    },
    Return {
        target: Option<ControlTarget>,
        operands: Vec<AggregateOperand>,
    },
    Exit {
        condition: Option<PredicateCondition>,
    },
    WarpShuffle {
        mode: Option<SassWarpShuffleMode>,
        predicate: RegisterRef,
        dst: RegisterRef,
        src: ScalarOperand,
        offset: ScalarOperand,
        mask: ScalarOperand,
    },
    Shift {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
    },
    LogicLut {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
    },
    Permute {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
    },
    AddressCalc {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
    },
    Sync {
        kind: SassSyncKind,
        operands: Vec<AggregateOperand>,
    },
    NoOp,
    Unsupported {
        opcode: SassOpcode,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassSyncKind {
    BarrierSet,
    BarrierSync,
    Barrier,
    Raw(String),
}

impl SassSyncKind {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "BSSY" => Self::BarrierSet,
            "BSYNC" => Self::BarrierSync,
            "BAR" => Self::Barrier,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::BarrierSet => "BSSY",
            Self::BarrierSync => "BSYNC",
            Self::Barrier => "BAR",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassSyncKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassWarpShuffleMode {
    Up,
    Down,
    Bfly,
    Index,
    Raw(String),
}

impl SassWarpShuffleMode {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "UP" => Self::Up,
            "DOWN" => Self::Down,
            "BFLY" => Self::Bfly,
            "IDX" => Self::Index,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Up => "UP",
            Self::Down => "DOWN",
            Self::Bfly => "BFLY",
            Self::Index => "IDX",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassWarpShuffleMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    Global,
    Shared,
    Local,
    Constant,
    Descriptor,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct RegisterRef {
    pub kind: RegisterRefKind,
    pub raw: String,
}

impl RegisterRef {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        let kind = parse_register_ref_kind(&raw).unwrap_or(RegisterRefKind::Raw);
        Self {
            raw: canonical_register_ref_text(&kind, &raw),
            kind,
        }
    }

    pub fn from_sass_register(raw: impl Into<String>, register: &SassRegister) -> Self {
        let raw = raw.into();
        let kind = match register.class {
            RegisterClass::General => register
                .index
                .map(RegisterRefKind::General)
                .unwrap_or(RegisterRefKind::Raw),
            RegisterClass::Uniform => register
                .index
                .map(RegisterRefKind::Uniform)
                .unwrap_or(RegisterRefKind::Raw),
            RegisterClass::Predicate => register
                .index
                .map(RegisterRefKind::Predicate)
                .unwrap_or(RegisterRefKind::Raw),
            RegisterClass::UniformPredicate => register
                .index
                .map(RegisterRefKind::UniformPredicate)
                .unwrap_or(RegisterRefKind::Raw),
            RegisterClass::Special => {
                RegisterRefKind::Special(register_base_without_modifiers(&raw).to_string())
            }
            RegisterClass::Barrier => register
                .index
                .map(RegisterRefKind::Barrier)
                .unwrap_or(RegisterRefKind::Raw),
            RegisterClass::Zero => match register_base_without_modifiers(&raw) {
                "URZ" => RegisterRefKind::UniformZero,
                _ => RegisterRefKind::GeneralZero,
            },
            RegisterClass::PredicateTrue => RegisterRefKind::PredicateTrue,
            RegisterClass::UniformPredicateTrue => RegisterRefKind::UniformPredicateTrue,
        };
        Self {
            raw: canonical_register_ref_text(&kind, &raw),
            kind,
        }
    }

    pub fn is_pseudo(&self) -> bool {
        matches!(
            self.kind,
            RegisterRefKind::GeneralZero
                | RegisterRefKind::UniformZero
                | RegisterRefKind::PredicateTrue
                | RegisterRefKind::UniformPredicateTrue
        )
    }

    pub fn extract_all(text: &str) -> Vec<Self> {
        let bytes = text.as_bytes();
        let mut index = 0usize;
        let mut registers = Vec::new();
        while index < bytes.len() {
            let Some((raw_register, consumed)) = parse_register_at(text, index) else {
                index += 1;
                continue;
            };
            let register = Self::parse(raw_register);
            if !registers.contains(&register) {
                registers.push(register);
            }
            index += consumed;
        }
        registers
    }
}

impl fmt::Display for RegisterRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

impl PartialEq for RegisterRef {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && (!matches!(self.kind, RegisterRefKind::Raw) || self.raw == other.raw)
    }
}

impl Eq for RegisterRef {}

impl PartialOrd for RegisterRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RegisterRef {
    fn cmp(&self, other: &Self) -> Ordering {
        self.kind.cmp(&other.kind).then_with(|| {
            if matches!(self.kind, RegisterRefKind::Raw) {
                self.raw.cmp(&other.raw)
            } else {
                Ordering::Equal
            }
        })
    }
}

impl Hash for RegisterRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        if matches!(self.kind, RegisterRefKind::Raw) {
            self.raw.hash(state);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegisterRefKind {
    General(u16),
    Uniform(u16),
    Predicate(u16),
    UniformPredicate(u16),
    Special(String),
    Barrier(u16),
    GeneralZero,
    UniformZero,
    PredicateTrue,
    UniformPredicateTrue,
    Raw,
}

fn parse_register_ref_kind(raw: &str) -> Option<RegisterRefKind> {
    let base = register_base_without_modifiers(raw);
    match base {
        "RZ" => Some(RegisterRefKind::GeneralZero),
        "URZ" => Some(RegisterRefKind::UniformZero),
        "PT" => Some(RegisterRefKind::PredicateTrue),
        "UPT" => Some(RegisterRefKind::UniformPredicateTrue),
        _ => base
            .strip_prefix("SR_")
            .map(|_| RegisterRefKind::Special(base.to_string()))
            .or_else(|| {
                base.strip_prefix("UR")
                    .and_then(|index| index.parse::<u16>().ok())
                    .map(RegisterRefKind::Uniform)
            })
            .or_else(|| {
                base.strip_prefix("UP")
                    .and_then(|index| index.parse::<u16>().ok())
                    .map(RegisterRefKind::UniformPredicate)
            })
            .or_else(|| {
                base.strip_prefix('R')
                    .and_then(|index| index.parse::<u16>().ok())
                    .map(RegisterRefKind::General)
            })
            .or_else(|| {
                base.strip_prefix('P')
                    .and_then(|index| index.parse::<u16>().ok())
                    .map(RegisterRefKind::Predicate)
            })
            .or_else(|| {
                base.strip_prefix('B')
                    .and_then(|index| index.parse::<u16>().ok())
                    .map(RegisterRefKind::Barrier)
            }),
    }
}

fn canonical_register_ref_text(kind: &RegisterRefKind, raw: &str) -> String {
    match kind {
        RegisterRefKind::General(index) => format!("R{index}"),
        RegisterRefKind::Uniform(index) => format!("UR{index}"),
        RegisterRefKind::Predicate(index) => format!("P{index}"),
        RegisterRefKind::UniformPredicate(index) => format!("UP{index}"),
        RegisterRefKind::Special(special) => special.clone(),
        RegisterRefKind::Barrier(index) => format!("B{index}"),
        RegisterRefKind::GeneralZero => "RZ".to_string(),
        RegisterRefKind::UniformZero => "URZ".to_string(),
        RegisterRefKind::PredicateTrue => "PT".to_string(),
        RegisterRefKind::UniformPredicateTrue => "UPT".to_string(),
        RegisterRefKind::Raw => raw.to_string(),
    }
}

fn register_base_without_modifiers(raw: &str) -> &str {
    let text = raw
        .trim()
        .trim_start_matches('!')
        .trim_start_matches('-')
        .trim_matches('|');
    text.split('.').next().unwrap_or(text)
}

fn parse_register_at(text: &str, index: usize) -> Option<(String, usize)> {
    if !is_token_boundary(text, index) {
        return None;
    }
    let tail = &text[index..];
    for literal in ["SR_", "URZ", "UPT", "UR", "UP", "RZ", "PT", "R", "P", "B"] {
        if let Some(register) = parse_register_prefix(tail, literal) {
            return Some(register);
        }
    }
    None
}

fn parse_register_prefix(tail: &str, prefix: &str) -> Option<(String, usize)> {
    let rest = tail.strip_prefix(prefix)?;
    match prefix {
        "SR_" => {
            let len = rest
                .char_indices()
                .take_while(|(_, ch)| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '.')
                .map(|(index, ch)| index + ch.len_utf8())
                .last()
                .unwrap_or(0);
            (len > 0).then(|| (tail[..prefix.len() + len].to_string(), prefix.len() + len))
        }
        "URZ" | "UPT" | "RZ" | "PT" => Some((prefix.to_string(), prefix.len())),
        "UR" | "UP" | "R" | "P" | "B" => {
            let len = rest
                .char_indices()
                .take_while(|(_, ch)| ch.is_ascii_digit())
                .map(|(index, ch)| index + ch.len_utf8())
                .last()
                .unwrap_or(0);
            (len > 0).then(|| (tail[..prefix.len() + len].to_string(), prefix.len() + len))
        }
        _ => None,
    }
}

fn is_token_boundary(text: &str, index: usize) -> bool {
    if index == 0 {
        return true;
    }
    let before = text[..index]
        .chars()
        .next_back()
        .expect("index > 0 should have previous char");
    !before.is_ascii_alphanumeric() && before != '_'
}

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
                .unwrap_or_else(|| AggregateOperandKind::Raw {
                    registers: RegisterRef::extract_all(&raw),
                }),
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
            SassOperandKind::Label(label) => AggregateOperandKind::Label(label.clone()),
            SassOperandKind::Raw => AggregateOperandKind::Raw {
                registers: RegisterRef::extract_all(&raw),
            },
        };
        Self { kind, raw }
    }

    pub fn registers(&self) -> Vec<RegisterRef> {
        match &self.kind {
            AggregateOperandKind::Register(register) => vec![register.clone()],
            AggregateOperandKind::Memory(address) => address.registers(),
            AggregateOperandKind::Raw { registers } => registers.clone(),
            AggregateOperandKind::Immediate(_) | AggregateOperandKind::Label(_) => Vec::new(),
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
    Label(String),
    Raw { registers: Vec<RegisterRef> },
}

fn parse_immediate_operand(raw: &str) -> Option<ImmediateValue> {
    let text = raw.trim();
    if text.is_empty() {
        return None;
    }
    if text.contains('.') || text.contains('e') || text.contains('E') {
        return text.parse::<f64>().ok().map(|value| {
            if value.fract() == 0.0 {
                ImmediateValue::Integer(value as i128)
            } else {
                ImmediateValue::FloatBits(value.to_bits())
            }
        });
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
            kind: ControlTargetKind::Label(label),
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

    pub fn label_name(&self) -> Option<&str> {
        match &self.kind {
            ControlTargetKind::Label(label) => Some(label),
            ControlTargetKind::Address(_) | ControlTargetKind::Raw => None,
        }
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
            ControlTargetKind::Label(label) => f.write_str(label),
            ControlTargetKind::Address(_) | ControlTargetKind::Raw => f.write_str(&self.raw),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ControlTargetKind {
    Label(String),
    Address(u64),
    Raw,
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
    pub modifiers: Vec<String>,
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
    pub fn new(width_bits: Option<u32>, modifiers: Vec<String>) -> Self {
        Self {
            width_bits,
            modifiers,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassMappingConfidence {
    LocallyParsed,
    OpcodeHeuristic,
    Unsupported,
}

impl fmt::Display for SassMappingConfidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocallyParsed => f.write_str("locally-parsed"),
            Self::OpcodeHeuristic => f.write_str("opcode-heuristic"),
            Self::Unsupported => f.write_str("unsupported"),
        }
    }
}
