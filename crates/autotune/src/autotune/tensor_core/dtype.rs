use std::fmt;

use crate::decompile::SassTensorElementType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorCoreDType {
    Bit,
    U4,
    S4,
    U8,
    S8,
    I8,
    S32,
    F16,
    Bf16,
    Tf32,
    F32,
    F64,
    Fp4,
    E2M1,
    Fp6,
    E2M3,
    E3M2,
    Fp8,
    E4M3,
    E5M2,
}

impl TensorCoreDType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Bit => "bit",
            Self::U4 => "u4",
            Self::S4 => "s4",
            Self::U8 => "u8",
            Self::S8 => "s8",
            Self::I8 => "i8",
            Self::S32 => "s32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Tf32 => "tf32",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Fp4 => "fp4",
            Self::E2M1 => "e2m1",
            Self::Fp6 => "fp6",
            Self::E2M3 => "e2m3",
            Self::E3M2 => "e3m2",
            Self::Fp8 => "fp8",
            Self::E4M3 => "e4m3",
            Self::E5M2 => "e5m2",
        }
    }

    pub fn from_sass(dtype: &SassTensorElementType) -> Option<Self> {
        match dtype {
            SassTensorElementType::Bit => Some(Self::Bit),
            SassTensorElementType::Fp64 => Some(Self::F64),
            SassTensorElementType::Fp32 => Some(Self::F32),
            SassTensorElementType::Tf32 => Some(Self::Tf32),
            SassTensorElementType::F16 | SassTensorElementType::Half => Some(Self::F16),
            SassTensorElementType::Bf16 => Some(Self::Bf16),
            SassTensorElementType::Integer => Some(Self::I8),
            SassTensorElementType::Fp4 => Some(Self::Fp4),
            SassTensorElementType::E2M1 => Some(Self::E2M1),
            SassTensorElementType::Fp6 => Some(Self::Fp6),
            SassTensorElementType::E2M3 => Some(Self::E2M3),
            SassTensorElementType::E3M2 => Some(Self::E3M2),
            SassTensorElementType::Fp8 => Some(Self::Fp8),
            SassTensorElementType::E4M3 => Some(Self::E4M3),
            SassTensorElementType::E5M2 => Some(Self::E5M2),
            SassTensorElementType::Raw(_) => None,
        }
    }
}

impl fmt::Display for TensorCoreDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorCoreAccumulator {
    F16,
    F32,
    F64,
    S32,
}

impl TensorCoreAccumulator {
    pub fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::S32 => "s32",
        }
    }

    pub fn from_dtype(dtype: TensorCoreDType) -> Option<Self> {
        match dtype {
            TensorCoreDType::F16 => Some(Self::F16),
            TensorCoreDType::F32 => Some(Self::F32),
            TensorCoreDType::F64 => Some(Self::F64),
            TensorCoreDType::S32
            | TensorCoreDType::I8
            | TensorCoreDType::S8
            | TensorCoreDType::U8
            | TensorCoreDType::U4
            | TensorCoreDType::S4
            | TensorCoreDType::Bit => Some(Self::S32),
            _ => None,
        }
    }
}

impl fmt::Display for TensorCoreAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}
