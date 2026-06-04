mod sources;

use self::sources::{
    SCALAR_MEMORY_ATOMIC_PTX, SCALAR_MEMORY_LOGIC_PTX, TENSOR_CORE_BMMA_PTX, TENSOR_CORE_DMMA_PTX,
    TENSOR_CORE_HMMA_PTX, TENSOR_CORE_IMMA_PTX,
};

pub const AUTO_COMPILE_ARCH: &str = "auto";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxDecompileProbeKind {
    TensorCoreHmma,
    TensorCoreImma,
    TensorCoreDmma,
    TensorCoreBmma,
    ScalarMemoryLogic,
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
            "scalar-memory-atomic" | "scalar_memory_atomic" | "atomic" => {
                Some(Self::ScalarMemoryAtomic)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtxDecompileProbe {
    pub kind: PtxDecompileProbeKind,
    pub symbol: &'static str,
    pub behavior: &'static str,
    pub default_compile_arch: &'static str,
    pub source: &'static str,
}

impl PtxDecompileProbe {
    pub fn compile_arch_for<'a>(&self, requested_compile_arch: &'a str) -> &'a str {
        if requested_compile_arch == AUTO_COMPILE_ARCH {
            self.default_compile_arch
        } else {
            requested_compile_arch
        }
    }
}

pub fn ptx_decompile_probes() -> Vec<PtxDecompileProbe> {
    vec![
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreHmma,
            symbol: "tensor_core_hmma_probe",
            behavior: "one PTX half-precision tensor-core MMA kept alive by a global f32 store",
            default_compile_arch: "sm_120",
            source: TENSOR_CORE_HMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreImma,
            symbol: "tensor_core_imma_probe",
            behavior: "one PTX integer tensor-core MMA kept alive by a global s32 store",
            default_compile_arch: "sm_120",
            source: TENSOR_CORE_IMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreDmma,
            symbol: "tensor_core_dmma_probe",
            behavior: "one PTX fp64 tensor-core MMA kept alive by a global f64 store",
            default_compile_arch: "sm_120",
            source: TENSOR_CORE_DMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreBmma,
            symbol: "tensor_core_bmma_probe",
            behavior: "one PTX single-bit tensor-core MMA kept alive by a global s32 store",
            default_compile_arch: "sm_80",
            source: TENSOR_CORE_BMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::ScalarMemoryLogic,
            symbol: "scalar_memory_logic_probe",
            behavior: "scalar PTX memory, predicate, logic, integer, and f32 fused math instructions",
            default_compile_arch: "sm_75",
            source: SCALAR_MEMORY_LOGIC_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::ScalarMemoryAtomic,
            symbol: "scalar_memory_atomic_probe",
            behavior: "scalar PTX atomics, reductions, volatile local memory, and predicate composition",
            default_compile_arch: "sm_75",
            source: SCALAR_MEMORY_ATOMIC_PTX,
        },
    ]
}

pub fn all_ptx_decompile_probe_kinds() -> Vec<PtxDecompileProbeKind> {
    vec![
        PtxDecompileProbeKind::TensorCoreHmma,
        PtxDecompileProbeKind::TensorCoreImma,
        PtxDecompileProbeKind::TensorCoreDmma,
        PtxDecompileProbeKind::TensorCoreBmma,
        PtxDecompileProbeKind::ScalarMemoryLogic,
        PtxDecompileProbeKind::ScalarMemoryAtomic,
    ]
}
