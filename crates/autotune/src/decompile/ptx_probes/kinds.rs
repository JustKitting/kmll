#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxDecompileProbeKind {
    TensorCoreHmma,
    TensorCoreImma,
    TensorCoreDmma,
    TensorCoreBmma,
    TensorCoreWgmmaHgmma,
    TensorCoreWgmmaBgmma,
    TensorCoreWgmmaIgmma,
    TensorCoreWgmmaQgmma,
    TensorCoreTcgen05Utcqmma,
    TensorMemoryLdtm,
    TensorMemorySttm,
    TensorMemoryUtccp,
    WarpGroupRegisterSet,
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
            Self::TensorCoreWgmmaHgmma => "tensor-core-wgmma-hgmma",
            Self::TensorCoreWgmmaBgmma => "tensor-core-wgmma-bgmma",
            Self::TensorCoreWgmmaIgmma => "tensor-core-wgmma-igmma",
            Self::TensorCoreWgmmaQgmma => "tensor-core-wgmma-qgmma",
            Self::TensorCoreTcgen05Utcqmma => "tensor-core-tcgen05-utcqmma",
            Self::TensorMemoryLdtm => "tensor-memory-ldtm",
            Self::TensorMemorySttm => "tensor-memory-sttm",
            Self::TensorMemoryUtccp => "tensor-memory-utccp",
            Self::WarpGroupRegisterSet => "warpgroup-register-set",
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
            "tensor-core-wgmma-hgmma"
            | "tensor_core_wgmma_hgmma"
            | "wgmma-hgmma"
            | "wgmma_hgmma"
            | "hgmma" => Some(Self::TensorCoreWgmmaHgmma),
            "tensor-core-wgmma-bgmma"
            | "tensor_core_wgmma_bgmma"
            | "wgmma-bgmma"
            | "wgmma_bgmma"
            | "bgmma" => Some(Self::TensorCoreWgmmaBgmma),
            "tensor-core-wgmma-igmma"
            | "tensor_core_wgmma_igmma"
            | "wgmma-igmma"
            | "wgmma_igmma"
            | "igmma" => Some(Self::TensorCoreWgmmaIgmma),
            "tensor-core-wgmma-qgmma"
            | "tensor_core_wgmma_qgmma"
            | "wgmma-qgmma"
            | "wgmma_qgmma"
            | "qgmma" => Some(Self::TensorCoreWgmmaQgmma),
            "tensor-core-tcgen05-utcqmma"
            | "tensor_core_tcgen05_utcqmma"
            | "tcgen05-utcqmma"
            | "tcgen05_utcqmma"
            | "utcqmma" => Some(Self::TensorCoreTcgen05Utcqmma),
            "tensor-memory-ldtm" | "tensor_memory_ldtm" | "tmem-ldtm" | "tmem_ldtm" | "ldtm" => {
                Some(Self::TensorMemoryLdtm)
            }
            "tensor-memory-sttm" | "tensor_memory_sttm" | "tmem-sttm" | "tmem_sttm" | "sttm" => {
                Some(Self::TensorMemorySttm)
            }
            "tensor-memory-utccp"
            | "tensor_memory_utccp"
            | "tmem-utccp"
            | "tmem_utccp"
            | "utccp" => Some(Self::TensorMemoryUtccp),
            "warpgroup-register-set"
            | "warpgroup_register_set"
            | "warpgroup-set"
            | "warpgroup_set"
            | "setmaxnreg"
            | "usetmaxreg" => Some(Self::WarpGroupRegisterSet),
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
