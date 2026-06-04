use super::kinds::PtxDecompileProbeKind;
use super::sources::{
    SCALAR_MEMORY_ATOMIC_PTX, SCALAR_MEMORY_LOGIC_PTX, TENSOR_CORE_BMMA_PTX, TENSOR_CORE_DMMA_PTX,
    TENSOR_CORE_HMMA_PTX, TENSOR_CORE_IMMA_PTX, TENSOR_CORE_WGMMA_HGMMA_PTX,
    TENSOR_CORE_WGMMA_IGMMA_PTX, TENSOR_CORE_WGMMA_QGMMA_PTX,
};

pub const AUTO_COMPILE_ARCH: &str = "auto";

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
            kind: PtxDecompileProbeKind::TensorCoreWgmmaHgmma,
            symbol: "tensor_core_wgmma_hgmma_probe",
            behavior: "one PTX sm90a warpgroup half-precision MMA with warpgroup sync and wait",
            default_compile_arch: "sm_90a",
            source: TENSOR_CORE_WGMMA_HGMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreWgmmaIgmma,
            symbol: "tensor_core_wgmma_igmma_probe",
            behavior: "one PTX sm90a warpgroup integer MMA with warpgroup sync and wait",
            default_compile_arch: "sm_90a",
            source: TENSOR_CORE_WGMMA_IGMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreWgmmaQgmma,
            symbol: "tensor_core_wgmma_qgmma_probe",
            behavior: "one PTX sm90a warpgroup FP8 MMA with warpgroup sync and wait",
            default_compile_arch: "sm_90a",
            source: TENSOR_CORE_WGMMA_QGMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::ScalarMemoryLogic,
            symbol: "scalar_memory_logic_probe",
            behavior: "scalar PTX memory, predicate, logic, integer, and f32 fused math instructions",
            default_compile_arch: "sm_75",
            source: SCALAR_MEMORY_LOGIC_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::ArchitectureSm90Scalar,
            symbol: "scalar_memory_logic_probe",
            behavior: "sm90 scalar SASS architecture scan probe using memory, integer, predicate, and f32 math",
            default_compile_arch: "sm_90",
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
        PtxDecompileProbeKind::TensorCoreWgmmaHgmma,
        PtxDecompileProbeKind::TensorCoreWgmmaIgmma,
        PtxDecompileProbeKind::TensorCoreWgmmaQgmma,
        PtxDecompileProbeKind::ScalarMemoryLogic,
        PtxDecompileProbeKind::ArchitectureSm90Scalar,
        PtxDecompileProbeKind::ScalarMemoryAtomic,
    ]
}
