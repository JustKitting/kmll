#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimpleKernelFixtureKind {
    I32Add,
    F32Add,
    F32Mul,
    F32Fma,
    LoadStore,
    PredicateBranch,
    ThreadIndexRead,
    Bf16ToF32,
    F16Ops,
}

impl SimpleKernelFixtureKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::I32Add => "i32-add",
            Self::F32Add => "f32-add",
            Self::F32Mul => "f32-mul",
            Self::F32Fma => "f32-fma",
            Self::LoadStore => "load-store",
            Self::PredicateBranch => "predicate-branch",
            Self::ThreadIndexRead => "thread-index-read",
            Self::Bf16ToF32 => "bf16-to-f32",
            Self::F16Ops => "f16-ops",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "i32-add" | "i32_add" => Some(Self::I32Add),
            "f32-add" | "f32_add" => Some(Self::F32Add),
            "f32-mul" | "f32_mul" => Some(Self::F32Mul),
            "f32-fma" | "f32_fma" => Some(Self::F32Fma),
            "load-store" | "load_store" => Some(Self::LoadStore),
            "predicate-branch" | "predicate_branch" => Some(Self::PredicateBranch),
            "thread-index-read" | "thread_index_read" => Some(Self::ThreadIndexRead),
            "bf16-to-f32" | "bf16_to_f32" => Some(Self::Bf16ToF32),
            "f16-ops" | "f16_ops" => Some(Self::F16Ops),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimpleKernelFixture {
    pub kind: SimpleKernelFixtureKind,
    pub symbol: &'static str,
    pub behavior: &'static str,
    pub source: &'static str,
}

pub fn simple_kernel_fixtures() -> Vec<SimpleKernelFixture> {
    vec![
        SimpleKernelFixture {
            kind: SimpleKernelFixtureKind::I32Add,
            symbol: "sass_fixture_i32_add",
            behavior: "one i32 add from two global input slices into one output slice",
            source: I32_ADD_SOURCE,
        },
        SimpleKernelFixture {
            kind: SimpleKernelFixtureKind::F32Add,
            symbol: "sass_fixture_f32_add",
            behavior: "one f32 add from two global input slices into one output slice",
            source: F32_ADD_SOURCE,
        },
        SimpleKernelFixture {
            kind: SimpleKernelFixtureKind::F32Mul,
            symbol: "sass_fixture_f32_mul",
            behavior: "one f32 multiply from two global input slices into one output slice",
            source: F32_MUL_SOURCE,
        },
        SimpleKernelFixture {
            kind: SimpleKernelFixtureKind::F32Fma,
            symbol: "sass_fixture_f32_fma",
            behavior: "one f32 multiply-add expression from three global input slices into one output slice",
            source: F32_FMA_SOURCE,
        },
        SimpleKernelFixture {
            kind: SimpleKernelFixtureKind::LoadStore,
            symbol: "sass_fixture_load_store",
            behavior: "one global load copied to one global store",
            source: LOAD_STORE_SOURCE,
        },
        SimpleKernelFixture {
            kind: SimpleKernelFixtureKind::PredicateBranch,
            symbol: "sass_fixture_predicate_branch",
            behavior: "branch on loaded i32 sign before writing output",
            source: PREDICATE_BRANCH_SOURCE,
        },
        SimpleKernelFixture {
            kind: SimpleKernelFixtureKind::ThreadIndexRead,
            symbol: "sass_fixture_thread_index_read",
            behavior: "read thread index and write it to output",
            source: THREAD_INDEX_READ_SOURCE,
        },
        SimpleKernelFixture {
            kind: SimpleKernelFixtureKind::Bf16ToF32,
            symbol: "sass_fixture_bf16_to_f32",
            behavior: "load one transparent bf16 value, widen it to f32 bits, and store f32",
            source: BF16_TO_F32_SOURCE,
        },
        SimpleKernelFixture {
            kind: SimpleKernelFixtureKind::F16Ops,
            symbol: "sass_fixture_f16_ops",
            behavior: "native f16 loads, arithmetic, and f16 output store",
            source: F16_OPS_SOURCE,
        },
    ]
}

pub fn all_simple_kernel_fixture_kinds() -> Vec<SimpleKernelFixtureKind> {
    vec![
        SimpleKernelFixtureKind::I32Add,
        SimpleKernelFixtureKind::F32Add,
        SimpleKernelFixtureKind::F32Mul,
        SimpleKernelFixtureKind::F32Fma,
        SimpleKernelFixtureKind::LoadStore,
        SimpleKernelFixtureKind::PredicateBranch,
        SimpleKernelFixtureKind::ThreadIndexRead,
        SimpleKernelFixtureKind::Bf16ToF32,
        SimpleKernelFixtureKind::F16Ops,
    ]
}

const I32_ADD_SOURCE: &str = r#"use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn sass_fixture_i32_add(a: &[i32], b: &[i32], mut out: DisjointSlice<i32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = a[i] + b[i];
    }
}
"#;

const F32_ADD_SOURCE: &str = r#"use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn sass_fixture_f32_add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = a[i] + b[i];
    }
}
"#;

const F32_MUL_SOURCE: &str = r#"use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn sass_fixture_f32_mul(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = a[i] * b[i];
    }
}
"#;

const F32_FMA_SOURCE: &str = r#"use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn sass_fixture_f32_fma(a: &[f32], b: &[f32], c: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = a[i] * b[i] + c[i];
    }
}
"#;

const LOAD_STORE_SOURCE: &str = r#"use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn sass_fixture_load_store(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[i];
    }
}
"#;

const PREDICATE_BRANCH_SOURCE: &str = r#"use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn sass_fixture_predicate_branch(input: &[i32], mut out: DisjointSlice<i32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        let value = input[i];
        *out_elem = if value > 0 { value } else { 0 };
    }
}
"#;

const THREAD_INDEX_READ_SOURCE: &str = r#"use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn sass_fixture_thread_index_read(mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = thread::threadIdx_x();
    }
}
"#;

const BF16_TO_F32_SOURCE: &str = r#"use cuda_device::{DisjointSlice, kernel, thread};

#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct Bf16(u16);

impl Bf16 {
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

#[kernel]
pub fn sass_fixture_bf16_to_f32(input: &[Bf16], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[i].to_f32();
    }
}
"#;

const F16_OPS_SOURCE: &str = r#"#![feature(f16)]

use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn sass_fixture_f16_ops(a: &[f16], b: &[f16], mut out: DisjointSlice<f16>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        let sum = a[i] + b[i];
        *out_elem = sum * f16::from_bits(0x3800);
    }
}
"#;
