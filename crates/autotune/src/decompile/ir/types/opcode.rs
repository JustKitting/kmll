use std::fmt;

use super::{KernelIrOp, KernelIrOpKind, SassTensorMmaShape};

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassModifierKind {
    E,
    Add,
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
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    LowerSame,
    HigherSame,
    Nan,
    Num,
    High,
    Low,
    Carry,
    And,
    Or,
    Xor,
    Min,
    Max,
    Inc,
    Dec,
    Exch,
    Cas,
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
            "ADD" => return Self::Add,
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
            "EQ" => return Self::Equal,
            "NE" => return Self::NotEqual,
            "LT" => return Self::LessThan,
            "LE" => return Self::LessEqual,
            "GT" => return Self::GreaterThan,
            "GE" => return Self::GreaterEqual,
            "LS" => return Self::LowerSame,
            "HS" => return Self::HigherSame,
            "NAN" => return Self::Nan,
            "NUM" => return Self::Num,
            "HI" => return Self::High,
            "LO" | "LOW" => return Self::Low,
            "X" => return Self::Carry,
            "AND" => return Self::And,
            "OR" => return Self::Or,
            "XOR" => return Self::Xor,
            "MIN" => return Self::Min,
            "MAX" => return Self::Max,
            "INC" => return Self::Inc,
            "DEC" => return Self::Dec,
            "EXCH" => return Self::Exch,
            "CAS" => return Self::Cas,
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
            | Self::Add
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
            | Self::Equal
            | Self::NotEqual
            | Self::LessThan
            | Self::LessEqual
            | Self::GreaterThan
            | Self::GreaterEqual
            | Self::LowerSame
            | Self::HigherSame
            | Self::Nan
            | Self::Num
            | Self::High
            | Self::Low
            | Self::Carry
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Min
            | Self::Max
            | Self::Inc
            | Self::Dec
            | Self::Exch
            | Self::Cas
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
    Atom,
    Atomg,
    Bar,
    Bra,
    Bssy,
    Bsync,
    Call,
    Cs2r,
    Elect,
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
    I2f,
    I2fp,
    Imad,
    Isetp,
    Ld,
    Ldc,
    Ldcu,
    Ldg,
    Ldl,
    Lds,
    Lea,
    Lepc,
    Lop3,
    Mov,
    Movm,
    Nop,
    Plop3,
    Prmt,
    Ret,
    Red,
    Redg,
    S2r,
    S2ur,
    Sel,
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
    Vote,
    Voteu,
    Viadd,
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
    Utccp,
    Utchmma,
    Utcimma,
    Utcomma,
    Utcqmma,
    Utmaldg,
    Utmapf,
    Utmaredg,
    Utmastg,
    Utmacmdflush,
    Usetmaxreg,
    Warpgroup,
    Warpgroupset,
    Raw(String),
}

impl SassOpcodeKind {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "ATOM" => Self::Atom,
            "ATOMG" => Self::Atomg,
            "BAR" => Self::Bar,
            "BRA" => Self::Bra,
            "BSSY" => Self::Bssy,
            "BSYNC" => Self::Bsync,
            "CALL" => Self::Call,
            "CS2R" => Self::Cs2r,
            "ELECT" => Self::Elect,
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
            "I2F" => Self::I2f,
            "I2FP" => Self::I2fp,
            "IMAD" => Self::Imad,
            "ISETP" => Self::Isetp,
            "LD" => Self::Ld,
            "LDC" => Self::Ldc,
            "LDCU" => Self::Ldcu,
            "LDG" => Self::Ldg,
            "LDL" => Self::Ldl,
            "LDS" => Self::Lds,
            "LEA" => Self::Lea,
            "LEPC" => Self::Lepc,
            "LOP3" => Self::Lop3,
            "MOV" => Self::Mov,
            "MOVM" => Self::Movm,
            "NOP" => Self::Nop,
            "PLOP3" => Self::Plop3,
            "PRMT" => Self::Prmt,
            "RET" => Self::Ret,
            "RED" => Self::Red,
            "REDG" => Self::Redg,
            "S2R" => Self::S2r,
            "S2UR" => Self::S2ur,
            "SEL" => Self::Sel,
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
            "VOTE" => Self::Vote,
            "VOTEU" => Self::Voteu,
            "VIADD" => Self::Viadd,
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
            "UTCCP" => Self::Utccp,
            "UTCHMMA" => Self::Utchmma,
            "UTCIMMA" => Self::Utcimma,
            "UTCOMMA" => Self::Utcomma,
            "UTCQMMA" => Self::Utcqmma,
            "UTMALDG" => Self::Utmaldg,
            "UTMAPF" => Self::Utmapf,
            "UTMAREDG" => Self::Utmaredg,
            "UTMASTG" => Self::Utmastg,
            "UTMACMDFLUSH" => Self::Utmacmdflush,
            "USETMAXREG" => Self::Usetmaxreg,
            "WARPGROUP" => Self::Warpgroup,
            "WARPGROUPSET" => Self::Warpgroupset,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Atom => "ATOM",
            Self::Atomg => "ATOMG",
            Self::Bar => "BAR",
            Self::Bra => "BRA",
            Self::Bssy => "BSSY",
            Self::Bsync => "BSYNC",
            Self::Call => "CALL",
            Self::Cs2r => "CS2R",
            Self::Elect => "ELECT",
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
            Self::I2f => "I2F",
            Self::I2fp => "I2FP",
            Self::Imad => "IMAD",
            Self::Isetp => "ISETP",
            Self::Ld => "LD",
            Self::Ldc => "LDC",
            Self::Ldcu => "LDCU",
            Self::Ldg => "LDG",
            Self::Ldl => "LDL",
            Self::Lds => "LDS",
            Self::Lea => "LEA",
            Self::Lepc => "LEPC",
            Self::Lop3 => "LOP3",
            Self::Mov => "MOV",
            Self::Movm => "MOVM",
            Self::Nop => "NOP",
            Self::Plop3 => "PLOP3",
            Self::Prmt => "PRMT",
            Self::Ret => "RET",
            Self::Red => "RED",
            Self::Redg => "REDG",
            Self::S2r => "S2R",
            Self::S2ur => "S2UR",
            Self::Sel => "SEL",
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
            Self::Vote => "VOTE",
            Self::Voteu => "VOTEU",
            Self::Viadd => "VIADD",
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
            Self::Utccp => "UTCCP",
            Self::Utchmma => "UTCHMMA",
            Self::Utcimma => "UTCIMMA",
            Self::Utcomma => "UTCOMMA",
            Self::Utcqmma => "UTCQMMA",
            Self::Utmaldg => "UTMALDG",
            Self::Utmapf => "UTMAPF",
            Self::Utmaredg => "UTMAREDG",
            Self::Utmastg => "UTMASTG",
            Self::Utmacmdflush => "UTMACMDFLUSH",
            Self::Usetmaxreg => "USETMAXREG",
            Self::Warpgroup => "WARPGROUP",
            Self::Warpgroupset => "WARPGROUPSET",
            Self::Raw(raw) => raw,
        }
    }
}
