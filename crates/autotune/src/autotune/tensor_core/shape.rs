use std::fmt;

use crate::decompile::SassTensorMmaShape;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorCoreMmaShape {
    M8N8K4,
    M8N8K16,
    M8N8K32,
    M8N8K128,
    M16N8K4,
    M16N8K8,
    M16N8K16,
    M16N8K32,
    M16N8K64,
    M16N8K128,
    M16N8K256,
    M64N8K8,
    M64N8K16,
    M64N8K32,
    M64N16K8,
    M64N16K16,
    M64N16K32,
    M64N32K8,
    M64N32K16,
    M64N32K32,
    M64N64K8,
    M64N64K16,
    M64N64K32,
    M64N128K8,
    M64N128K16,
    M64N128K32,
    M64N256K8,
    M64N256K16,
    M64N256K32,
    Custom { m: u32, n: u32, k: u32 },
}

impl TensorCoreMmaShape {
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        match (m, n, k) {
            (8, 8, 4) => Self::M8N8K4,
            (8, 8, 16) => Self::M8N8K16,
            (8, 8, 32) => Self::M8N8K32,
            (8, 8, 128) => Self::M8N8K128,
            (16, 8, 4) => Self::M16N8K4,
            (16, 8, 8) => Self::M16N8K8,
            (16, 8, 16) => Self::M16N8K16,
            (16, 8, 32) => Self::M16N8K32,
            (16, 8, 64) => Self::M16N8K64,
            (16, 8, 128) => Self::M16N8K128,
            (16, 8, 256) => Self::M16N8K256,
            (64, 8, 8) => Self::M64N8K8,
            (64, 8, 16) => Self::M64N8K16,
            (64, 8, 32) => Self::M64N8K32,
            (64, 16, 8) => Self::M64N16K8,
            (64, 16, 16) => Self::M64N16K16,
            (64, 16, 32) => Self::M64N16K32,
            (64, 32, 8) => Self::M64N32K8,
            (64, 32, 16) => Self::M64N32K16,
            (64, 32, 32) => Self::M64N32K32,
            (64, 64, 8) => Self::M64N64K8,
            (64, 64, 16) => Self::M64N64K16,
            (64, 64, 32) => Self::M64N64K32,
            (64, 128, 8) => Self::M64N128K8,
            (64, 128, 16) => Self::M64N128K16,
            (64, 128, 32) => Self::M64N128K32,
            (64, 256, 8) => Self::M64N256K8,
            (64, 256, 16) => Self::M64N256K16,
            (64, 256, 32) => Self::M64N256K32,
            _ => Self::Custom { m, n, k },
        }
    }

    pub fn dimensions(self) -> (u32, u32, u32) {
        match self {
            Self::M8N8K4 => (8, 8, 4),
            Self::M8N8K16 => (8, 8, 16),
            Self::M8N8K32 => (8, 8, 32),
            Self::M8N8K128 => (8, 8, 128),
            Self::M16N8K4 => (16, 8, 4),
            Self::M16N8K8 => (16, 8, 8),
            Self::M16N8K16 => (16, 8, 16),
            Self::M16N8K32 => (16, 8, 32),
            Self::M16N8K64 => (16, 8, 64),
            Self::M16N8K128 => (16, 8, 128),
            Self::M16N8K256 => (16, 8, 256),
            Self::M64N8K8 => (64, 8, 8),
            Self::M64N8K16 => (64, 8, 16),
            Self::M64N8K32 => (64, 8, 32),
            Self::M64N16K8 => (64, 16, 8),
            Self::M64N16K16 => (64, 16, 16),
            Self::M64N16K32 => (64, 16, 32),
            Self::M64N32K8 => (64, 32, 8),
            Self::M64N32K16 => (64, 32, 16),
            Self::M64N32K32 => (64, 32, 32),
            Self::M64N64K8 => (64, 64, 8),
            Self::M64N64K16 => (64, 64, 16),
            Self::M64N64K32 => (64, 64, 32),
            Self::M64N128K8 => (64, 128, 8),
            Self::M64N128K16 => (64, 128, 16),
            Self::M64N128K32 => (64, 128, 32),
            Self::M64N256K8 => (64, 256, 8),
            Self::M64N256K16 => (64, 256, 16),
            Self::M64N256K32 => (64, 256, 32),
            Self::Custom { m, n, k } => (m, n, k),
        }
    }

    pub fn from_sass(shape: SassTensorMmaShape) -> Self {
        Self::new(shape.m, shape.n, shape.k)
    }
}

impl fmt::Display for TensorCoreMmaShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (m, n, k) = self.dimensions();
        write!(f, "m{m}n{n}k{k}")
    }
}
