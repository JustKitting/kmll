#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxDecompileProbeKind {
    TensorCoreHmma,
    TensorCoreImma,
    TensorCoreDmma,
    TensorCoreBmma,
    ScalarMemoryLogic,
    ArchitectureSm90Scalar,
    ScalarMemoryAtomic,
}

impl PtxDecompileProbeKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::TensorCoreHmma => "tensor-core-hmma",
            Self::TensorCoreImma => "tensor-core-imma",
            Self::TensorCoreDmma => "tensor-core-dmma",
            Self::TensorCoreBmma => "tensor-core-bmma",
            Self::ScalarMemoryLogic => "scalar-memory-logic",
            Self::ArchitectureSm90Scalar => "architecture-sm90-scalar",
            Self::ScalarMemoryAtomic => "scalar-memory-atomic",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "tensor-core-hmma" | "tensor_core_hmma" | "hmma" => Some(Self::TensorCoreHmma),
            "tensor-core-imma" | "tensor_core_imma" | "imma" => Some(Self::TensorCoreImma),
            "tensor-core-dmma" | "tensor_core_dmma" | "dmma" => Some(Self::TensorCoreDmma),
            "tensor-core-bmma" | "tensor_core_bmma" | "bmma" => Some(Self::TensorCoreBmma),
            "scalar-memory-logic" | "scalar_memory_logic" | "scalar" => {
                Some(Self::ScalarMemoryLogic)
            }
            "architecture-sm90-scalar" | "architecture_sm90_scalar" | "sm90-scalar" | "sm90" => {
                Some(Self::ArchitectureSm90Scalar)
            }
            "scalar-memory-atomic" | "scalar_memory_atomic" | "atomic" => {
                Some(Self::ScalarMemoryAtomic)
            }
            _ => None,
        }
    }
}
