pub(super) const TENSOR_CORE_HMMA_PTX: &str = r#".version 8.0
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

pub(super) const TENSOR_CORE_IMMA_PTX: &str = r#".version 8.0
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

pub(super) const TENSOR_CORE_DMMA_PTX: &str = r#".version 8.0
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

pub(super) const TENSOR_CORE_BMMA_PTX: &str = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry tensor_core_bmma_probe(
    .param .u64 tensor_core_bmma_probe_out
)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd0, [tensor_core_bmma_probe_out];

    mov.b32 %r0, 0xffffffff;
    mov.b32 %r1, 0xaaaaaaaa;
    mov.s32 %r2, 0;
    mov.s32 %r3, 0;

    mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc
        {%r2, %r3},
        {%r0},
        {%r1},
        {%r2, %r3};

    st.global.s32 [%rd0], %r2;
    ret;
}
"#;

pub(super) const TENSOR_CORE_WGMMA_HGMMA_PTX: &str = r#".version 8.7
.target sm_90a
.address_size 64

.visible .entry tensor_core_wgmma_hgmma_probe(
    .param .u64 tensor_core_wgmma_hgmma_probe_out
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd0, [tensor_core_wgmma_hgmma_probe_out];

    mov.b32 %r0, 0x3c003c00;
    mov.b32 %r1, 0x3c003c00;
    mov.b32 %r2, 0x3c003c00;
    mov.b32 %r3, 0x3c003c00;
    mov.b32 %r4, 0;
    mov.b32 %r5, 0;
    mov.u64 %rd1, 0;
    setp.ne.b32 %p0, 1, 0;

    wgmma.fence.sync.aligned;
    wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16
        {%r4, %r5},
        {%r0, %r1, %r2, %r3},
        %rd1,
        %p0, 1, 1, 0;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;

    st.global.u32 [%rd0], %r4;
    ret;
}
"#;

pub(super) const TENSOR_CORE_WGMMA_IGMMA_PTX: &str = r#".version 8.7
.target sm_90a
.address_size 64

.visible .entry tensor_core_wgmma_igmma_probe(
    .param .u64 tensor_core_wgmma_igmma_probe_out
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd0, [tensor_core_wgmma_igmma_probe_out];

    mov.b32 %r0, 0x01010101;
    mov.b32 %r1, 0x01010101;
    mov.b32 %r2, 0x01010101;
    mov.b32 %r3, 0x01010101;
    mov.b32 %r4, 0;
    mov.b32 %r5, 0;
    mov.b32 %r6, 0;
    mov.b32 %r7, 0;
    mov.u64 %rd1, 0;
    setp.ne.b32 %p0, 1, 0;

    wgmma.fence.sync.aligned;
    wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8
        {%r4, %r5, %r6, %r7},
        {%r0, %r1, %r2, %r3},
        %rd1,
        %p0;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;

    st.global.u32 [%rd0], %r4;
    ret;
}
"#;

pub(super) const TENSOR_CORE_WGMMA_QGMMA_PTX: &str = r#".version 8.7
.target sm_90a
.address_size 64

.visible .entry tensor_core_wgmma_qgmma_probe(
    .param .u64 tensor_core_wgmma_qgmma_probe_out
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<4>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd0, [tensor_core_wgmma_qgmma_probe_out];

    mov.b32 %r0, 0x3f3f3f3f;
    mov.b32 %r1, 0x3f3f3f3f;
    mov.b32 %r2, 0x3f3f3f3f;
    mov.b32 %r3, 0x3f3f3f3f;
    mov.f32 %f0, 0f00000000;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.u64 %rd1, 0;
    setp.ne.b32 %p0, 1, 0;

    wgmma.fence.sync.aligned;
    wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3
        {%f0, %f1, %f2, %f3},
        {%r0, %r1, %r2, %r3},
        %rd1,
        %p0, 1, 1;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;

    st.global.f32 [%rd0], %f0;
    ret;
}
"#;

pub(super) const SCALAR_MEMORY_LOGIC_PTX: &str = r#".version 8.0
.target sm_75
.address_size 64

.visible .entry scalar_memory_logic_probe(
    .param .u64 scalar_memory_logic_probe_out,
    .param .u64 scalar_memory_logic_probe_in
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<8>;
    .local .align 4 .b8 scalar_memory_logic_probe_local[512];

    ld.param.u64 %rd0, [scalar_memory_logic_probe_out];
    ld.param.u64 %rd1, [scalar_memory_logic_probe_in];

    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, 31;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    mov.u64 %rd3, scalar_memory_logic_probe_local;
    add.u64 %rd4, %rd3, %rd2;

    ld.global.nc.u32 %r3, [%rd1];
    add.s32 %r4, %r3, %r1;
    add.s32 %r5, %r4, 17;
    lop3.b32 %r6, %r3, %r4, %r5, 0x96;

    st.local.u32 [%rd4], %r6;
    ld.local.u32 %r7, [%rd4];

    setp.gt.u32 %p0, %r7, %r5;
    setp.ne.u32 %p1, %r7, %r3;
    and.pred %p2, %p0, %p1;
    selp.u32 %r8, %r7, %r5, %p2;

    cvt.rn.f32.u32 %f0, %r8;
    mov.f32 %f1, 0f3f800000;
    fma.rn.f32 %f2, %f0, %f1, %f0;
    setp.gt.f32 %p3, %f2, %f0;
    selp.u32 %r9, %r8, %r3, %p3;

    st.global.u32 [%rd0], %r9;
    ret;
}
"#;

pub(super) const SCALAR_MEMORY_ATOMIC_PTX: &str = r#".version 9.1
.target sm_75
.address_size 64

.visible .entry scalar_memory_atomic_probe(
    .param .u64 scalar_memory_atomic_probe_out,
    .param .u64 scalar_memory_atomic_probe_data
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<10>;
    .local .align 4 .b8 scalar_memory_atomic_probe_local[128];

    ld.param.u64 %rd0, [scalar_memory_atomic_probe_out];
    ld.param.u64 %rd1, [scalar_memory_atomic_probe_data];

    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, 31;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    cvta.to.global.u64 %rd7, %rd4;

    mov.u32 %r3, 1;
    atom.add.u32 %r4, [%rd7], %r3;
    atom.global.add.u32 %r5, [%rd4], %r3;
    red.global.add.u32 [%rd4], %r3;

    mov.u64 %rd5, scalar_memory_atomic_probe_local;
    add.u64 %rd6, %rd5, %rd2;
    st.volatile.local.u32 [%rd6], %r4;
    ld.volatile.local.u32 %r6, [%rd6];

    setp.eq.u32 %p0, %r6, %r4;
    setp.ne.u32 %p1, %r6, %r3;
    setp.gt.u32 %p2, %r6, 0;
    and.pred %p3, %p0, %p1;
    or.pred %p4, %p2, %p3;
    xor.pred %p5, %p4, %p1;
    selp.u32 %r7, %r6, %r3, %p5;

    add.u32 %r8, %r4, %r5;
    add.u32 %r9, %r8, %r7;
    st.global.u32 [%rd3], %r9;
    ret;
}
"#;
