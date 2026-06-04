use std::fmt;

use super::SassModifierKind;

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
pub enum SassNumericDType {
    U8,
    S8,
    U16,
    S16,
    U32,
    S32,
    U64,
    S64,
    F16,
    Bf16,
    F32,
    F64,
    Tf32,
    Fp4,
    Fp8,
    Raw(String),
}

impl SassNumericDType {
    pub fn from_modifier(modifier: &SassModifierKind) -> Option<Self> {
        match modifier {
            SassModifierKind::UnsignedWidth(8) => Some(Self::U8),
            SassModifierKind::SignedWidth(8) => Some(Self::S8),
            SassModifierKind::UnsignedWidth(16) => Some(Self::U16),
            SassModifierKind::SignedWidth(16) => Some(Self::S16),
            SassModifierKind::UnsignedWidth(32) => Some(Self::U32),
            SassModifierKind::SignedWidth(32) => Some(Self::S32),
            SassModifierKind::UnsignedWidth(64) => Some(Self::U64),
            SassModifierKind::SignedWidth(64) => Some(Self::S64),
            SassModifierKind::F16 => Some(Self::F16),
            SassModifierKind::Bf16 => Some(Self::Bf16),
            SassModifierKind::F32 => Some(Self::F32),
            SassModifierKind::F64 => Some(Self::F64),
            SassModifierKind::Tf32 => Some(Self::Tf32),
            SassModifierKind::Fp4 => Some(Self::Fp4),
            SassModifierKind::Fp8 => Some(Self::Fp8),
            SassModifierKind::Raw(raw) => Some(Self::Raw(raw.clone())),
            _ => None,
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
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::Tf32 => "TF32",
            Self::Fp4 => "FP4",
            Self::Fp8 => "FP8",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassNumericDType {
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
