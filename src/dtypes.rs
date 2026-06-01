use crate::backends::Cuda;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    I8,
}

impl DType {
    pub fn from_safetensors_name(name: &str) -> Option<Self> {
        match name {
            "F32" => Some(Self::F32),
            "BF16" => Some(Self::Bf16),
            "I8" => Some(Self::I8),
            _ => None,
        }
    }

    pub fn safetensors_name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
            Self::I8 => "I8",
        }
    }

    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
            Self::I8 => 1,
        }
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

pub trait DeviceQuantized: Copy {
    const DTYPE: DType;
}

impl DeviceFloat for Bf16 {
    const ACCUM_DTYPE: DType = DType::F32;
}

impl DeviceFloat for f32 {
    const ACCUM_DTYPE: DType = DType::F32;
}

impl DeviceQuantized for i8 {
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
