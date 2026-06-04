use std::{
    cmp::Ordering,
    fmt::{self, Write as _},
    hash::{Hash, Hasher},
};

use super::super::SassTarget;
use super::super::sass::{
    RegisterClass, SassOperand, SassOperandKind, SassRegister, SassSourcePosition,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIrModule {
    pub target: Option<SassTarget>,
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
    pub name: SassSymbol,
    pub ops: Vec<KernelIrOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIrOp {
    pub address: u64,
    pub source_position: SassSourcePosition,
    pub label: Option<SassSymbol>,
    pub predicate: Option<PredicateCondition>,
    pub kind: KernelIrOpKind,
    pub confidence: SassMappingConfidence,
    pub source_opcode: SassOpcode,
    pub source_modifiers: Vec<SassModifier>,
    pub source_operands: Vec<AggregateOperand>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassSymbol {
    raw: String,
}

impl SassSymbol {
    pub fn new(raw: impl Into<String>) -> Self {
        Self { raw: raw.into() }
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }
}

impl fmt::Display for SassSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassOpcode {
    kind: SassOpcodeKind,
    raw: String,
}

impl SassOpcode {
    pub fn new(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        Self {
            kind: SassOpcodeKind::parse(raw.as_str()),
            raw,
        }
    }

    pub fn from_kind(kind: SassOpcodeKind) -> Self {
        let raw = kind.as_str().to_string();
        Self { kind, raw }
    }

    pub fn from_ir_op(op: &KernelIrOp) -> Self {
        match &op.kind {
            KernelIrOpKind::Unsupported { opcode, .. } => opcode.clone(),
            _ => op.source_opcode.clone(),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }

    pub fn kind(&self) -> &SassOpcodeKind {
        &self.kind
    }
}

impl fmt::Display for SassOpcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassModifier {
    kind: SassModifierKind,
    raw: String,
}

impl SassModifier {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        Self {
            kind: SassModifierKind::parse(raw.as_str()),
            raw,
        }
    }

    pub fn kind(&self) -> &SassModifierKind {
        &self.kind
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }
}

impl fmt::Display for SassModifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassTensorMmaShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl SassTensorMmaShape {
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    pub fn parse_compact(raw: &str) -> Option<Self> {
        if !raw.chars().all(|ch| ch.is_ascii_digit()) {
            return None;
        }
        for m in [64, 32, 16, 8] {
            for n in [32, 16, 8] {
                for k in [256, 128, 64, 32, 16, 8, 4] {
                    if raw == format!("{m}{n}{k}") {
                        return Some(Self::new(m, n, k));
                    }
                }
            }
        }
        None
    }
}

impl fmt::Display for SassTensorMmaShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m{}n{}k{}", self.m, self.n, self.k)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassModifierKind {
    E,
    UnsignedWidth(u32),
    SignedWidth(u32),
    Width(u32),
    TensorShape(SassTensorMmaShape),
    F16,
    Bf16,
    F32,
    F64,
    Tf32,
    Fp4,
    Fp8,
    E2M1,
    E4M3,
    E5M2,
    High,
    Low,
    Carry,
    And,
    Wide,
    Up,
    Down,
    Bfly,
    Index,
    Raw(String),
}

impl SassModifierKind {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "E" => return Self::E,
            "F16" | "FP16" => return Self::F16,
            "BF16" => return Self::Bf16,
            "F32" | "FP32" => return Self::F32,
            "F64" | "FP64" => return Self::F64,
            "TF32" => return Self::Tf32,
            "F4" | "FP4" => return Self::Fp4,
            "F8" | "FP8" => return Self::Fp8,
            "E2M1" => return Self::E2M1,
            "E4M3" => return Self::E4M3,
            "E5M2" => return Self::E5M2,
            "HI" => return Self::High,
            "LO" | "LOW" => return Self::Low,
            "X" => return Self::Carry,
            "AND" => return Self::And,
            "WIDE" => return Self::Wide,
            "UP" => return Self::Up,
            "DOWN" => return Self::Down,
            "BFLY" => return Self::Bfly,
            "IDX" => return Self::Index,
            _ => {}
        }
        if let Some(shape) = SassTensorMmaShape::parse_compact(raw.as_str()) {
            return Self::TensorShape(shape);
        }
        if let Some(bits) = raw.strip_prefix('U').and_then(|bits| bits.parse().ok()) {
            return Self::UnsignedWidth(bits);
        }
        if let Some(bits) = raw.strip_prefix('S').and_then(|bits| bits.parse().ok()) {
            return Self::SignedWidth(bits);
        }
        raw.parse::<u32>()
            .map(Self::Width)
            .unwrap_or(Self::Raw(raw))
    }

    pub fn width_bits(&self) -> Option<u32> {
        match self {
            Self::UnsignedWidth(bits) | Self::SignedWidth(bits) | Self::Width(bits) => Some(*bits),
            Self::E
            | Self::TensorShape(_)
            | Self::F16
            | Self::Bf16
            | Self::F32
            | Self::F64
            | Self::Tf32
            | Self::Fp4
            | Self::Fp8
            | Self::E2M1
            | Self::E4M3
            | Self::E5M2
            | Self::High
            | Self::Low
            | Self::Carry
            | Self::And
            | Self::Wide
            | Self::Up
            | Self::Down
            | Self::Bfly
            | Self::Index
            | Self::Raw(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassOpcodeKind {
    Bar,
    Bra,
    Bssy,
    Bsync,
    Call,
    Cs2r,
    Exit,
    Fadd,
    Ffma,
    Fmul,
    Fsetp,
    Hadd2,
    Hfma2,
    Hmul2,
    Iadd,
    Iadd3,
    Imad,
    Isetp,
    Ld,
    Ldc,
    Ldcu,
    Ldg,
    Ldl,
    Lds,
    Lea,
    Lop3,
    Mov,
    Nop,
    Plop3,
    Prmt,
    Ret,
    S2r,
    S2ur,
    Shf,
    Shfl,
    St,
    Stg,
    Stl,
    Sts,
    Uiadd3,
    Uimad,
    Uisetp,
    Uldc,
    Ulea,
    Ulop3,
    Umov,
    Ushf,
    Bgmma,
    Bmma,
    Dmma,
    Hgmma,
    Hmma,
    Igmma,
    Imma,
    Omma,
    Qgmma,
    Qmma,
    Ldt,
    Ldtm,
    Stt,
    Sttm,
    Ublkcp,
    Ublkpf,
    Ublkred,
    Utchmma,
    Utcimma,
    Utcomma,
    Utcqmma,
    Utmaldg,
    Utmapf,
    Utmaredg,
    Utmastg,
    Warpgroup,
    Warpgroupset,
    Raw(String),
}

impl SassOpcodeKind {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "BAR" => Self::Bar,
            "BRA" => Self::Bra,
            "BSSY" => Self::Bssy,
            "BSYNC" => Self::Bsync,
            "CALL" => Self::Call,
            "CS2R" => Self::Cs2r,
            "EXIT" => Self::Exit,
            "FADD" => Self::Fadd,
            "FFMA" => Self::Ffma,
            "FMUL" => Self::Fmul,
            "FSETP" => Self::Fsetp,
            "HADD2" => Self::Hadd2,
            "HFMA2" => Self::Hfma2,
            "HMUL2" => Self::Hmul2,
            "IADD" => Self::Iadd,
            "IADD3" => Self::Iadd3,
            "IMAD" => Self::Imad,
            "ISETP" => Self::Isetp,
            "LD" => Self::Ld,
            "LDC" => Self::Ldc,
            "LDCU" => Self::Ldcu,
            "LDG" => Self::Ldg,
            "LDL" => Self::Ldl,
            "LDS" => Self::Lds,
            "LEA" => Self::Lea,
            "LOP3" => Self::Lop3,
            "MOV" => Self::Mov,
            "NOP" => Self::Nop,
            "PLOP3" => Self::Plop3,
            "PRMT" => Self::Prmt,
            "RET" => Self::Ret,
            "S2R" => Self::S2r,
            "S2UR" => Self::S2ur,
            "SHF" => Self::Shf,
            "SHFL" => Self::Shfl,
            "ST" => Self::St,
            "STG" => Self::Stg,
            "STL" => Self::Stl,
            "STS" => Self::Sts,
            "UIADD3" => Self::Uiadd3,
            "UIMAD" => Self::Uimad,
            "UISETP" => Self::Uisetp,
            "ULDC" => Self::Uldc,
            "ULEA" => Self::Ulea,
            "ULOP3" => Self::Ulop3,
            "UMOV" => Self::Umov,
            "USHF" => Self::Ushf,
            "BGMMA" => Self::Bgmma,
            "BMMA" => Self::Bmma,
            "DMMA" => Self::Dmma,
            "HGMMA" => Self::Hgmma,
            "HMMA" => Self::Hmma,
            "IGMMA" => Self::Igmma,
            "IMMA" => Self::Imma,
            "OMMA" => Self::Omma,
            "QGMMA" => Self::Qgmma,
            "QMMA" => Self::Qmma,
            "LDT" => Self::Ldt,
            "LDTM" => Self::Ldtm,
            "STT" => Self::Stt,
            "STTM" => Self::Sttm,
            "UBLKCP" => Self::Ublkcp,
            "UBLKPF" => Self::Ublkpf,
            "UBLKRED" => Self::Ublkred,
            "UTCHMMA" => Self::Utchmma,
            "UTCIMMA" => Self::Utcimma,
            "UTCOMMA" => Self::Utcomma,
            "UTCQMMA" => Self::Utcqmma,
            "UTMALDG" => Self::Utmaldg,
            "UTMAPF" => Self::Utmapf,
            "UTMAREDG" => Self::Utmaredg,
            "UTMASTG" => Self::Utmastg,
            "WARPGROUP" => Self::Warpgroup,
            "WARPGROUPSET" => Self::Warpgroupset,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Bar => "BAR",
            Self::Bra => "BRA",
            Self::Bssy => "BSSY",
            Self::Bsync => "BSYNC",
            Self::Call => "CALL",
            Self::Cs2r => "CS2R",
            Self::Exit => "EXIT",
            Self::Fadd => "FADD",
            Self::Ffma => "FFMA",
            Self::Fmul => "FMUL",
            Self::Fsetp => "FSETP",
            Self::Hadd2 => "HADD2",
            Self::Hfma2 => "HFMA2",
            Self::Hmul2 => "HMUL2",
            Self::Iadd => "IADD",
            Self::Iadd3 => "IADD3",
            Self::Imad => "IMAD",
            Self::Isetp => "ISETP",
            Self::Ld => "LD",
            Self::Ldc => "LDC",
            Self::Ldcu => "LDCU",
            Self::Ldg => "LDG",
            Self::Ldl => "LDL",
            Self::Lds => "LDS",
            Self::Lea => "LEA",
            Self::Lop3 => "LOP3",
            Self::Mov => "MOV",
            Self::Nop => "NOP",
            Self::Plop3 => "PLOP3",
            Self::Prmt => "PRMT",
            Self::Ret => "RET",
            Self::S2r => "S2R",
            Self::S2ur => "S2UR",
            Self::Shf => "SHF",
            Self::Shfl => "SHFL",
            Self::St => "ST",
            Self::Stg => "STG",
            Self::Stl => "STL",
            Self::Sts => "STS",
            Self::Uiadd3 => "UIADD3",
            Self::Uimad => "UIMAD",
            Self::Uisetp => "UISETP",
            Self::Uldc => "ULDC",
            Self::Ulea => "ULEA",
            Self::Ulop3 => "ULOP3",
            Self::Umov => "UMOV",
            Self::Ushf => "USHF",
            Self::Bgmma => "BGMMA",
            Self::Bmma => "BMMA",
            Self::Dmma => "DMMA",
            Self::Hgmma => "HGMMA",
            Self::Hmma => "HMMA",
            Self::Igmma => "IGMMA",
            Self::Imma => "IMMA",
            Self::Omma => "OMMA",
            Self::Qgmma => "QGMMA",
            Self::Qmma => "QMMA",
            Self::Ldt => "LDT",
            Self::Ldtm => "LDTM",
            Self::Stt => "STT",
            Self::Sttm => "STTM",
            Self::Ublkcp => "UBLKCP",
            Self::Ublkpf => "UBLKPF",
            Self::Ublkred => "UBLKRED",
            Self::Utchmma => "UTCHMMA",
            Self::Utcimma => "UTCIMMA",
            Self::Utcomma => "UTCOMMA",
            Self::Utcqmma => "UTCQMMA",
            Self::Utmaldg => "UTMALDG",
            Self::Utmapf => "UTMAPF",
            Self::Utmaredg => "UTMAREDG",
            Self::Utmastg => "UTMASTG",
            Self::Warpgroup => "WARPGROUP",
            Self::Warpgroupset => "WARPGROUPSET",
            Self::Raw(raw) => raw,
        }
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
        element_type: Option<SassTensorElementType>,
        signature: Option<SassTensorMmaSignature>,
        scope: Option<SassTensorScope>,
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
        comparison: Option<SassComparisonKind>,
        dtype: Option<SassCompareDType>,
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
        reason: SassUnsupportedReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassUnsupportedReason {
    NoLocalMapping,
    OperandArity {
        expectation: SassOperandArityExpectation,
        expected: usize,
        actual: usize,
    },
}

impl SassUnsupportedReason {
    pub const fn no_local_mapping() -> Self {
        Self::NoLocalMapping
    }

    pub const fn exact_operand_arity(expected: usize, actual: usize) -> Self {
        Self::OperandArity {
            expectation: SassOperandArityExpectation::Exact,
            expected,
            actual,
        }
    }

    pub const fn at_least_operand_arity(expected: usize, actual: usize) -> Self {
        Self::OperandArity {
            expectation: SassOperandArityExpectation::AtLeast,
            expected,
            actual,
        }
    }
}

impl fmt::Display for SassUnsupportedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLocalMapping => f.write_str("no local mapping for opcode yet"),
            Self::OperandArity {
                expectation,
                expected,
                actual,
            } => match expectation {
                SassOperandArityExpectation::Exact => {
                    write!(f, "expected {expected} operands, saw {actual}")
                }
                SassOperandArityExpectation::AtLeast => {
                    write!(f, "expected at least {expected} operands, saw {actual}")
                }
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassOperandArityExpectation {
    Exact,
    AtLeast,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassTensorElementType {
    Bit,
    Fp64,
    Fp32,
    Tf32,
    F16,
    Bf16,
    Half,
    Integer,
    Fp4,
    E2M1,
    Fp8,
    E4M3,
    E5M2,
    Raw(String),
}

impl SassTensorElementType {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "bit" => Self::Bit,
            "fp64" => Self::Fp64,
            "fp32" => Self::Fp32,
            "tf32" => Self::Tf32,
            "f16" => Self::F16,
            "bf16" => Self::Bf16,
            "half" => Self::Half,
            "integer" => Self::Integer,
            "fp4" => Self::Fp4,
            "e2m1" => Self::E2M1,
            "fp8" => Self::Fp8,
            "e4m3" => Self::E4M3,
            "e5m2" => Self::E5M2,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Bit => "bit",
            Self::Fp64 => "fp64",
            Self::Fp32 => "fp32",
            Self::Tf32 => "tf32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Half => "half",
            Self::Integer => "integer",
            Self::Fp4 => "fp4",
            Self::E2M1 => "e2m1",
            Self::Fp8 => "fp8",
            Self::E4M3 => "e4m3",
            Self::E5M2 => "e5m2",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassTensorElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SassTensorMmaSignature {
    pub shape: Option<SassTensorMmaShape>,
    pub output_type: Option<SassTensorElementType>,
    pub lhs_type: Option<SassTensorElementType>,
    pub rhs_type: Option<SassTensorElementType>,
    pub accumulator_type: Option<SassTensorElementType>,
}

impl SassTensorMmaSignature {
    pub fn primary_element_type(&self) -> Option<SassTensorElementType> {
        self.lhs_type
            .clone()
            .or_else(|| self.rhs_type.clone())
            .or_else(|| self.accumulator_type.clone())
            .or_else(|| self.output_type.clone())
    }
}

impl fmt::Display for SassTensorMmaSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "shape={},output={},lhs={},rhs={},accumulator={}",
            option_display(self.shape.as_ref()),
            option_display(self.output_type.as_ref()),
            option_display(self.lhs_type.as_ref()),
            option_display(self.rhs_type.as_ref()),
            option_display(self.accumulator_type.as_ref())
        )
    }
}

fn option_display(value: Option<&impl fmt::Display>) -> String {
    value
        .map(ToString::to_string)
        .unwrap_or_else(|| "unknown".to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassTensorScope {
    Warp,
    WarpGroup,
    Uniform,
    Raw(String),
}

impl SassTensorScope {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "warp" => Self::Warp,
            "warpgroup" => Self::WarpGroup,
            "uniform" => Self::Uniform,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Warp => "warp",
            Self::WarpGroup => "warpgroup",
            Self::Uniform => "uniform",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassTensorScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassComparisonKind {
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Lower,
    LowerSame,
    Higher,
    HigherSame,
    Nan,
    Num,
    Raw(String),
}

impl SassComparisonKind {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "EQ" => Self::Equal,
            "NE" => Self::NotEqual,
            "LT" => Self::LessThan,
            "LE" => Self::LessEqual,
            "GT" => Self::GreaterThan,
            "GE" => Self::GreaterEqual,
            "LO" => Self::Lower,
            "LS" => Self::LowerSame,
            "HI" => Self::Higher,
            "HS" => Self::HigherSame,
            "NAN" => Self::Nan,
            "NUM" => Self::Num,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Equal => "EQ",
            Self::NotEqual => "NE",
            Self::LessThan => "LT",
            Self::LessEqual => "LE",
            Self::GreaterThan => "GT",
            Self::GreaterEqual => "GE",
            Self::Lower => "LO",
            Self::LowerSame => "LS",
            Self::Higher => "HI",
            Self::HigherSame => "HS",
            Self::Nan => "NAN",
            Self::Num => "NUM",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassComparisonKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassCompareDType {
    U8,
    S8,
    U16,
    S16,
    U32,
    S32,
    U64,
    S64,
    F32,
    F64,
    Raw(String),
}

impl SassCompareDType {
    pub fn parse_known(raw: impl Into<String>) -> Option<Self> {
        let parsed = Self::parse(raw);
        (!matches!(parsed, Self::Raw(_))).then_some(parsed)
    }

    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "U8" => Self::U8,
            "S8" => Self::S8,
            "U16" => Self::U16,
            "S16" => Self::S16,
            "U32" => Self::U32,
            "S32" => Self::S32,
            "U64" => Self::U64,
            "S64" => Self::S64,
            "F32" => Self::F32,
            "F64" => Self::F64,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::U8 => "U8",
            Self::S8 => "S8",
            Self::U16 => "U16",
            Self::S16 => "S16",
            Self::U32 => "U32",
            Self::S32 => "S32",
            Self::U64 => "U64",
            Self::S64 => "S64",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassCompareDType {
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
            SassOperandKind::Label(label) => {
                AggregateOperandKind::Label(SassSymbol::new(label.clone()))
            }
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
    Label(SassSymbol),
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
pub enum SassMemoryModifier {
    E,
    Unsigned(u32),
    Signed(u32),
    Width(u32),
    Raw(String),
}

impl SassMemoryModifier {
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
