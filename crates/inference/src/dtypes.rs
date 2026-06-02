use crate::backends::Cuda;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bool,
    Fp4,
    F6E2M3,
    F6E3M2,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    F8E8M0,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    C64,
    F64,
    I64,
    U64,
}

impl DType {
    pub const ALL: [Self; 20] = [
        Self::Bool,
        Self::Fp4,
        Self::F6E2M3,
        Self::F6E3M2,
        Self::U8,
        Self::I8,
        Self::F8E5M2,
        Self::F8E4M3,
        Self::F8E8M0,
        Self::I16,
        Self::U16,
        Self::F16,
        Self::Bf16,
        Self::I32,
        Self::U32,
        Self::F32,
        Self::C64,
        Self::F64,
        Self::I64,
        Self::U64,
    ];

    pub fn from_safetensors_name(name: &str) -> Option<Self> {
        match name {
            "BOOL" => Some(Self::Bool),
            "F4" => Some(Self::Fp4),
            "F6_E2M3" => Some(Self::F6E2M3),
            "F6_E3M2" => Some(Self::F6E3M2),
            "U8" => Some(Self::U8),
            "I8" => Some(Self::I8),
            "F8_E5M2" => Some(Self::F8E5M2),
            "F8_E4M3" => Some(Self::F8E4M3),
            "F8_E8M0" => Some(Self::F8E8M0),
            "I16" => Some(Self::I16),
            "U16" => Some(Self::U16),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            "I32" => Some(Self::I32),
            "U32" => Some(Self::U32),
            "F32" => Some(Self::F32),
            "C64" => Some(Self::C64),
            "F64" => Some(Self::F64),
            "I64" => Some(Self::I64),
            "U64" => Some(Self::U64),
            _ => None,
        }
    }

    pub fn safetensors_name(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::Fp4 => "F4",
            Self::F6E2M3 => "F6_E2M3",
            Self::F6E3M2 => "F6_E3M2",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E8M0 => "F8_E8M0",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::F32 => "F32",
            Self::C64 => "C64",
            Self::F64 => "F64",
            Self::I64 => "I64",
            Self::U64 => "U64",
        }
    }

    pub fn bitsize(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Fp4 => 4,
            Self::F6E2M3 | Self::F6E3M2 => 6,
            Self::U8 | Self::I8 | Self::F8E5M2 | Self::F8E4M3 | Self::F8E8M0 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::C64 | Self::F64 | Self::I64 | Self::U64 => 64,
        }
    }

    pub fn byte_size(self) -> Option<usize> {
        let bits = self.bitsize();
        (bits % 8 == 0).then_some(bits / 8)
    }

    pub fn size_in_bytes(self) -> Option<usize> {
        self.byte_size()
    }

    pub fn packed_size_in_bytes(self, element_count: usize) -> usize {
        (element_count * self.bitsize()).div_ceil(8)
    }

    pub fn is_byte_aligned(self) -> bool {
        self.byte_size().is_some()
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bf16(u16);

impl Bf16 {
    #[inline(always)]
    pub fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline(always)]
    pub fn to_bits(self) -> u16 {
        self.0
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    #[inline]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let rounding_bias = ((bits >> 16) & 1) + 0x7fff;
        Self(((bits + rounding_bias) >> 16) as u16)
    }
}

pub trait DeviceFloat: Copy {
    const ACCUM_DTYPE: DType;
}

pub trait DeviceStorageElement: Copy {
    const DTYPE: DType;
}

impl DeviceFloat for Bf16 {
    const ACCUM_DTYPE: DType = DType::F32;
}

impl DeviceFloat for f32 {
    const ACCUM_DTYPE: DType = DType::F32;
}

impl DeviceStorageElement for i8 {
    const DTYPE: DType = DType::I8;
}

pub trait AccumulatorWith<Rhs, Backend>: DeviceFloat
where
    Rhs: DeviceFloat,
{
    type Accumulator: DeviceFloat;
}

impl AccumulatorWith<Bf16, Cuda> for Bf16 {
    type Accumulator = f32;
}

impl AccumulatorWith<Bf16, Cuda> for f32 {
    type Accumulator = f32;
}

impl AccumulatorWith<f32, Cuda> for f32 {
    type Accumulator = f32;
}

// Short-term CUDA promotion rule for inference kernels: all currently
// supported storage formats accumulate into f32. A real promotion lattice can
// replace this with per-op result rules once we add more storage types.
pub trait AccumulateToF32<Backend>: DeviceFloat {
    fn to_f32_accumulator(self) -> f32;
}

impl AccumulateToF32<Cuda> for Bf16 {
    #[inline(always)]
    fn to_f32_accumulator(self) -> f32 {
        self.to_f32()
    }
}

impl AccumulateToF32<Cuda> for f32 {
    #[inline(always)]
    fn to_f32_accumulator(self) -> f32 {
        self
    }
}

pub trait TensorElement: DeviceFloat {
    const DTYPE: DType;

    fn from_le_bytes(bytes: &[u8]) -> Self;
}

impl TensorElement for Bf16 {
    const DTYPE: DType = DType::Bf16;

    fn from_le_bytes(bytes: &[u8]) -> Self {
        Self(u16::from_le_bytes([bytes[0], bytes[1]]))
    }
}

impl TensorElement for f32 {
    const DTYPE: DType = DType::F32;

    fn from_le_bytes(bytes: &[u8]) -> Self {
        Self::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn safetensors_dtype_names_roundtrip() {
        for dtype in DType::ALL {
            assert_eq!(
                DType::from_safetensors_name(dtype.safetensors_name()),
                Some(dtype)
            );
        }
    }

    #[test]
    fn dtype_bitsizes_cover_sub_byte_and_standard_widths() {
        assert_eq!(DType::Bool.bitsize(), 1);
        assert_eq!(DType::Fp4.bitsize(), 4);
        assert_eq!(DType::F6E2M3.bitsize(), 6);
        assert_eq!(DType::F8E4M3.bitsize(), 8);
        assert_eq!(DType::Bf16.bitsize(), 16);
        assert_eq!(DType::F32.bitsize(), 32);
        assert_eq!(DType::C64.bitsize(), 64);
    }

    #[test]
    fn sub_byte_dtypes_are_not_byte_aligned() {
        assert_eq!(DType::Fp4.byte_size(), None);
        assert_eq!(DType::Fp4.packed_size_in_bytes(3), 2);
        assert_eq!(DType::Bf16.byte_size(), Some(2));
    }
}
