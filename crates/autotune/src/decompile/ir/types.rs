use std::fmt::{self, Write as _};

use super::super::sass::{RegisterClass, SassRegister, SassSourcePosition};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelIrOpKind {
    ReadSpecialRegister {
        dst: String,
        special: String,
    },
    Move {
        dst: String,
        src: String,
    },
    LoadConst {
        dst: String,
        source: MemoryAddress,
    },
    Load {
        dst: String,
        address: MemoryAddress,
        space: MemorySpace,
        access: MemoryAccessInfo,
    },
    Store {
        address: MemoryAddress,
        value: String,
        space: MemorySpace,
        access: MemoryAccessInfo,
    },
    IntegerAdd {
        dst: String,
        inputs: Vec<String>,
        width_bits: Option<u32>,
    },
    FloatAdd {
        dst: String,
        lhs: String,
        rhs: String,
    },
    FloatMul {
        dst: String,
        lhs: String,
        rhs: String,
    },
    PackedHalfAdd {
        dst: String,
        inputs: Vec<String>,
        lanes: u32,
    },
    PackedHalfMul {
        dst: String,
        inputs: Vec<String>,
        lanes: u32,
    },
    FusedMultiplyAdd {
        dst: String,
        a: String,
        b: String,
        c: String,
        lane_bits: Option<u32>,
    },
    IntegerMad {
        dst: String,
        a: String,
        b: String,
        c: String,
        wide: bool,
    },
    TensorCoreMma {
        opcode: String,
        operands: Vec<String>,
        element_type: Option<String>,
        scope: Option<String>,
    },
    TensorCoreMemory {
        opcode: String,
        operands: Vec<String>,
    },
    TensorMemoryAccess {
        opcode: String,
        operands: Vec<String>,
    },
    WarpGroup {
        opcode: String,
        operands: Vec<String>,
    },
    CompareSet {
        dst: String,
        comparison: Option<String>,
        dtype: Option<String>,
        lhs: String,
        rhs: String,
    },
    Branch {
        target: Option<ControlTarget>,
        condition: Option<PredicateCondition>,
    },
    Call {
        target: Option<ControlTarget>,
        operands: Vec<String>,
    },
    Return {
        target: Option<ControlTarget>,
        operands: Vec<String>,
    },
    Exit {
        condition: Option<PredicateCondition>,
    },
    WarpShuffle {
        mode: Option<String>,
        predicate: String,
        dst: String,
        src: String,
        offset: String,
        mask: String,
    },
    Shift {
        dst: String,
        inputs: Vec<String>,
    },
    LogicLut {
        dst: String,
        inputs: Vec<String>,
    },
    Permute {
        dst: String,
        inputs: Vec<String>,
    },
    AddressCalc {
        dst: String,
        inputs: Vec<String>,
    },
    Sync {
        kind: String,
        operands: Vec<String>,
    },
    NoOp,
    Unsupported {
        opcode: String,
        reason: String,
    },
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegisterRef {
    pub kind: RegisterRefKind,
    pub raw: String,
}

impl RegisterRef {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        Self {
            kind: parse_register_ref_kind(&raw).unwrap_or(RegisterRefKind::Raw),
            raw,
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
            RegisterClass::Zero => RegisterRefKind::Zero,
            RegisterClass::PredicateTrue => RegisterRefKind::PredicateTrue,
            RegisterClass::UniformPredicateTrue => RegisterRefKind::UniformPredicateTrue,
        };
        Self { kind, raw }
    }

    pub fn is_pseudo(&self) -> bool {
        matches!(
            self.kind,
            RegisterRefKind::Zero
                | RegisterRefKind::PredicateTrue
                | RegisterRefKind::UniformPredicateTrue
        )
    }
}

impl fmt::Display for RegisterRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
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
    Zero,
    PredicateTrue,
    UniformPredicateTrue,
    Raw,
}

fn parse_register_ref_kind(raw: &str) -> Option<RegisterRefKind> {
    let base = register_base_without_modifiers(raw);
    match base {
        "RZ" | "URZ" => Some(RegisterRefKind::Zero),
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

fn register_base_without_modifiers(raw: &str) -> &str {
    let text = raw
        .trim()
        .trim_start_matches('!')
        .trim_start_matches('-')
        .trim_matches('|');
    text.split('.').next().unwrap_or(text)
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
            kind: MemoryAddressKind::Constant { bank, offset },
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
                offset,
            },
            raw,
        }
    }

    pub fn indexed(raw: String, base: String, offset: Option<String>) -> Self {
        Self {
            kind: MemoryAddressKind::Indexed {
                base: RegisterRef::parse(base),
                offset,
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

    pub fn base(&self) -> Option<&str> {
        match &self.kind {
            MemoryAddressKind::Constant { bank, .. } => Some(bank),
            MemoryAddressKind::Descriptor { descriptor, .. } => Some(descriptor.raw.as_str()),
            MemoryAddressKind::Indexed { base, .. } => Some(base.raw.as_str()),
            MemoryAddressKind::Raw => None,
        }
    }

    pub fn offset(&self) -> Option<&str> {
        match &self.kind {
            MemoryAddressKind::Constant { offset, .. } => Some(offset),
            MemoryAddressKind::Descriptor { offset, .. }
            | MemoryAddressKind::Indexed { offset, .. } => offset.as_deref(),
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
        bank: String,
        offset: String,
    },
    Descriptor {
        descriptor: RegisterRef,
        address: RegisterRef,
        address_width: Option<u32>,
        offset: Option<String>,
    },
    Indexed {
        base: RegisterRef,
        offset: Option<String>,
    },
    Raw,
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
