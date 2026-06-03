#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxDecompileProbeKind {
    TensorCoreHmma,
    TensorCoreImma,
    TensorCoreDmma,
}

impl PtxDecompileProbeKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::TensorCoreHmma => "tensor-core-hmma",
            Self::TensorCoreImma => "tensor-core-imma",
            Self::TensorCoreDmma => "tensor-core-dmma",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "tensor-core-hmma" | "tensor_core_hmma" | "hmma" => Some(Self::TensorCoreHmma),
            "tensor-core-imma" | "tensor_core_imma" | "imma" => Some(Self::TensorCoreImma),
            "tensor-core-dmma" | "tensor_core_dmma" | "dmma" => Some(Self::TensorCoreDmma),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtxDecompileProbe {
    pub kind: PtxDecompileProbeKind,
    pub symbol: &'static str,
    pub behavior: &'static str,
    pub source: &'static str,
}

pub fn ptx_decompile_probes() -> Vec<PtxDecompileProbe> {
    vec![
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreHmma,
            symbol: "tensor_core_hmma_probe",
            behavior: "one PTX half-precision tensor-core MMA kept alive by a global f32 store",
            source: TENSOR_CORE_HMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreImma,
            symbol: "tensor_core_imma_probe",
            behavior: "one PTX integer tensor-core MMA kept alive by a global s32 store",
            source: TENSOR_CORE_IMMA_PTX,
        },
        PtxDecompileProbe {
            kind: PtxDecompileProbeKind::TensorCoreDmma,
            symbol: "tensor_core_dmma_probe",
            behavior: "one PTX fp64 tensor-core MMA kept alive by a global f64 store",
            source: TENSOR_CORE_DMMA_PTX,
        },
    ]
}

pub fn all_ptx_decompile_probe_kinds() -> Vec<PtxDecompileProbeKind> {
    vec![
        PtxDecompileProbeKind::TensorCoreHmma,
        PtxDecompileProbeKind::TensorCoreImma,
        PtxDecompileProbeKind::TensorCoreDmma,
    ]
}

const TENSOR_CORE_HMMA_PTX: &str = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry tensor_core_hmma_probe(
    .param .u64 tensor_core_hmma_probe_out
)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<2>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd0, [tensor_core_hmma_probe_out];

    mov.b32 %r0, 0x3c003c00;
    mov.b32 %r1, 0x3c003c00;
    mov.b32 %r2, 0x3c003c00;
    mov.b32 %r3, 0x3c003c00;
    mov.b32 %r4, 0x3c003c00;
    mov.b32 %r5, 0x3c003c00;

    mov.f32 %f0, 0f00000000;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {%f0, %f1, %f2, %f3},
        {%r0, %r1, %r2, %r3},
        {%r4, %r5},
        {%f0, %f1, %f2, %f3};

    st.global.f32 [%rd0], %f0;
    ret;
}
"#;

const TENSOR_CORE_IMMA_PTX: &str = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry tensor_core_imma_probe(
    .param .u64 tensor_core_imma_probe_out
)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd0, [tensor_core_imma_probe_out];

    mov.b32 %r0, 0x01010101;
    mov.b32 %r1, 0x01010101;
    mov.b32 %r2, 0x01010101;
    mov.b32 %r3, 0x01010101;
    mov.b32 %r4, 0x01010101;
    mov.b32 %r5, 0x01010101;
    mov.b32 %r6, 0;
    mov.b32 %r7, 0;
    mov.b32 %r8, 0;
    mov.b32 %r9, 0;

    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r6, %r7, %r8, %r9},
        {%r0, %r1, %r2, %r3},
        {%r4, %r5},
        {%r6, %r7, %r8, %r9};

    st.global.s32 [%rd0], %r6;
    ret;
}
"#;

const TENSOR_CORE_DMMA_PTX: &str = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry tensor_core_dmma_probe(
    .param .u64 tensor_core_dmma_probe_out
)
{
    .reg .b64 %rd<2>;
    .reg .f64 %d<8>;

    ld.param.u64 %rd0, [tensor_core_dmma_probe_out];

    mov.f64 %d0, 0d3ff0000000000000;
    mov.f64 %d1, 0d0000000000000000;
    mov.f64 %d2, 0d3ff0000000000000;
    mov.f64 %d3, 0d3ff0000000000000;

    mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
        {%d0, %d1},
        {%d2},
        {%d3},
        {%d0, %d1};

    st.global.f64 [%rd0], %d0;
    ret;
}
"#;
