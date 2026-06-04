use std::fmt;

use crate::decompile::{SassOpcodeKind, SassTensorScope};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorCoreOpFamily {
    MmaSync,
    MmaSparse,
    WgmmaAsync,
    Tcgen05,
}

impl TensorCoreOpFamily {
    pub fn label(self) -> &'static str {
        match self {
            Self::MmaSync => "mma.sync",
            Self::MmaSparse => "mma.sp",
            Self::WgmmaAsync => "wgmma.mma_async",
            Self::Tcgen05 => "tcgen05.mma",
        }
    }

    pub fn from_sass_opcode(opcode: &SassOpcodeKind) -> Option<Self> {
        match opcode {
            SassOpcodeKind::Bmma
            | SassOpcodeKind::Dmma
            | SassOpcodeKind::Hmma
            | SassOpcodeKind::Imma
            | SassOpcodeKind::Omma
            | SassOpcodeKind::Qmma => Some(Self::MmaSync),
            SassOpcodeKind::Bgmma
            | SassOpcodeKind::Hgmma
            | SassOpcodeKind::Igmma
            | SassOpcodeKind::Qgmma => Some(Self::WgmmaAsync),
            SassOpcodeKind::Utchmma
            | SassOpcodeKind::Utcimma
            | SassOpcodeKind::Utcomma
            | SassOpcodeKind::Utcqmma => Some(Self::Tcgen05),
            _ => None,
        }
    }

    pub fn from_sass_scope(scope: &SassTensorScope) -> Option<Self> {
        match scope {
            SassTensorScope::Warp => Some(Self::MmaSync),
            SassTensorScope::WarpGroup => Some(Self::WgmmaAsync),
            SassTensorScope::Uniform => Some(Self::Tcgen05),
            SassTensorScope::Raw(_) => None,
        }
    }
}

impl fmt::Display for TensorCoreOpFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}
