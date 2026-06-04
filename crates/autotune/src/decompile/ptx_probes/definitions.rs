use super::kinds::PtxDecompileProbeKind;
use super::sources::{
    SCALAR_MEMORY_ATOMIC_PTX, SCALAR_MEMORY_LOGIC_PTX, TENSOR_CORE_BMMA_PTX, TENSOR_CORE_DMMA_PTX,
    TENSOR_CORE_HMMA_PTX, TENSOR_CORE_IMMA_PTX, TENSOR_CORE_SM120A_QMMA_PTX,
    TENSOR_CORE_TCGEN05_UTCHMMA_UTCIMMA_PTX, TENSOR_CORE_TCGEN05_UTCOMMA_PTX,
    TENSOR_CORE_TCGEN05_UTCQMMA_PTX, TENSOR_CORE_WGMMA_BGMMA_PTX, TENSOR_CORE_WGMMA_HGMMA_PTX,
    TENSOR_CORE_WGMMA_IGMMA_PTX, TENSOR_CORE_WGMMA_QGMMA_PTX, TENSOR_MEMORY_BULK_ASYNC_PTX,
    TENSOR_MEMORY_BULK_REDUCE_PTX, TENSOR_MEMORY_LDTM_PTX, TENSOR_MEMORY_STTM_PTX,
    TENSOR_MEMORY_TMA_ASYNC_PTX, TENSOR_MEMORY_UTCCP_PTX, WARP_GROUP_REGISTER_SET_PTX,
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
            kind: PtxDecompileProbeKind::TensorCoreWgmmaBgmma,
            symbol: "tensor_core_wgmma_bgmma_probe",
            behavior: "one PTX sm90a warpgroup boolean MMA with warpgroup sync and wait",
            default_compile_arch: "sm_90a",
            source: TENSOR_CORE_WGMMA_BGMMA_PTX,
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
            kind: PtxDecompileProbeKind::TensorCoreSm120aQmma,
            symbol: "tensor_core_sm120a_qmma_probe",
            behavior: "one PTX sm120a warp-scope FP8 MMA that disassembles to QMMA",
            default_compile_arch: "sm_120a",
            source: TENSOR_CORE_SM120A_QMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreTcgen05UtchmmaUtcimma,
            symbol: "tensor_core_tcgen05_utchmma_utcimma_probe",
            behavior: "PTX sm100a tcgen05 F16 and I8 MMAs that disassemble to UTCHMMA/UTCIMMA",
            default_compile_arch: "sm_100a",
            source: TENSOR_CORE_TCGEN05_UTCHMMA_UTCIMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreTcgen05Utcomma,
            symbol: "tensor_core_tcgen05_utcomma_probe",
            behavior: "one PTX sm100a tcgen05 FP4 block-scaled MMA that disassembles to UTCOMMA",
            default_compile_arch: "sm_100a",
            source: TENSOR_CORE_TCGEN05_UTCOMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreTcgen05Utcqmma,
            symbol: "tensor_core_tcgen05_utcqmma_probe",
            behavior: "one PTX sm100a tcgen05 narrow-precision MMA that disassembles to UTCQMMA",
            default_compile_arch: "sm_100a",
            source: TENSOR_CORE_TCGEN05_UTCQMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorMemoryLdtm,
            symbol: "tensor_memory_ldtm_probe",
            behavior: "one PTX sm100a tcgen05 tensor-memory load kept alive by global stores",
            default_compile_arch: "sm_100a",
            source: TENSOR_MEMORY_LDTM_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorMemorySttm,
            symbol: "tensor_memory_sttm_probe",
            behavior: "one PTX sm100a tcgen05 tensor-memory store fed by global loads",
            default_compile_arch: "sm_100a",
            source: TENSOR_MEMORY_STTM_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorMemoryUtccp,
            symbol: "tensor_memory_utccp_probe",
            behavior: "one PTX sm100a tcgen05 tensor-memory bulk copy from global descriptor",
            default_compile_arch: "sm_100a",
            source: TENSOR_MEMORY_UTCCP_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorMemoryBulkAsync,
            symbol: "tensor_memory_bulk_async_probe",
            behavior: "PTX sm120 cp.async.bulk copy and prefetch forms that disassemble to UBLKCP/UBLKPF",
            default_compile_arch: "sm_120",
            source: TENSOR_MEMORY_BULK_ASYNC_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorMemoryBulkReduce,
            symbol: "tensor_memory_bulk_reduce_probe",
            behavior: "PTX sm120 non-tensor cp.reduce.async.bulk forms that disassemble to UBLKRED",
            default_compile_arch: "sm_120",
            source: TENSOR_MEMORY_BULK_REDUCE_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorMemoryTmaAsync,
            symbol: "tensor_memory_tma_async_probe",
            behavior: "PTX sm120 tensor TMA load, prefetch, store, and reduce forms that disassemble to UTMA*",
            default_compile_arch: "sm_120",
            source: TENSOR_MEMORY_TMA_ASYNC_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::WarpGroupRegisterSet,
            symbol: "warpgroup_register_set_probe",
            behavior: "one PTX sm90a warpgroup register reconfiguration using setmaxnreg",
            default_compile_arch: "sm_90a",
            source: WARP_GROUP_REGISTER_SET_PTX,
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
        PtxDecompileProbeKind::TensorCoreWgmmaBgmma,
        PtxDecompileProbeKind::TensorCoreWgmmaIgmma,
        PtxDecompileProbeKind::TensorCoreWgmmaQgmma,
        PtxDecompileProbeKind::TensorCoreSm120aQmma,
        PtxDecompileProbeKind::TensorCoreTcgen05UtchmmaUtcimma,
        PtxDecompileProbeKind::TensorCoreTcgen05Utcomma,
        PtxDecompileProbeKind::TensorCoreTcgen05Utcqmma,
        PtxDecompileProbeKind::TensorMemoryLdtm,
        PtxDecompileProbeKind::TensorMemorySttm,
        PtxDecompileProbeKind::TensorMemoryUtccp,
        PtxDecompileProbeKind::TensorMemoryBulkAsync,
        PtxDecompileProbeKind::TensorMemoryBulkReduce,
        PtxDecompileProbeKind::TensorMemoryTmaAsync,
        PtxDecompileProbeKind::WarpGroupRegisterSet,
        PtxDecompileProbeKind::ScalarMemoryLogic,
        PtxDecompileProbeKind::ArchitectureSm90Scalar,
        PtxDecompileProbeKind::ScalarMemoryAtomic,
    ]
}
