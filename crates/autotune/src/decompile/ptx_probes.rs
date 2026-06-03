#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxDecompileProbeKind {
    TensorCoreHmma,
}

impl PtxDecompileProbeKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::TensorCoreHmma => "tensor-core-hmma",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "tensor-core-hmma" | "tensor_core_hmma" | "hmma" => Some(Self::TensorCoreHmma),
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
    vec![PtxDecompileProbe {
        kind: PtxDecompileProbeKind::TensorCoreHmma,
        symbol: "tensor_core_hmma_probe",
        behavior: "one PTX half-precision tensor-core MMA kept alive by a global f32 store",
        source: TENSOR_CORE_HMMA_PTX,
    }]
}

pub fn all_ptx_decompile_probe_kinds() -> Vec<PtxDecompileProbeKind> {
    vec![PtxDecompileProbeKind::TensorCoreHmma]
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
