use super::SassOpcodeKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownSassOpcode {
    pub opcode: SassOpcodeKind,
    pub architectures: &'static [&'static str],
    pub class: &'static str,
    pub kind: &'static str,
    pub source: &'static str,
    pub locally_mapped: bool,
}

pub fn known_sass_opcodes() -> &'static [KnownSassOpcode] {
    KNOWN_SASS_OPCODES
}

const LOCAL_LIFTER: &str = "local-sass-lifter";
const NVIDIA_BINARY_UTILITIES: &str = "nvidia-cuda-binary-utilities-instruction-reference";

macro_rules! local {
    ($opcode:ident, $class:literal, $kind:literal) => {
        KnownSassOpcode {
            opcode: SassOpcodeKind::$opcode,
            architectures: &[],
            class: $class,
            kind: $kind,
            source: LOCAL_LIFTER,
            locally_mapped: true,
        }
    };
}

macro_rules! nvidia_mapped {
    ($opcode:ident, [$($arch:literal),* $(,)?], $class:literal, $kind:literal) => {
        KnownSassOpcode {
            opcode: SassOpcodeKind::$opcode,
            architectures: &[$($arch),*],
            class: $class,
            kind: $kind,
            source: NVIDIA_BINARY_UTILITIES,
            locally_mapped: true,
        }
    };
}

const KNOWN_SASS_OPCODES: &[KnownSassOpcode] = &[
    local!(Bar, "synchronization", "sync"),
    local!(Bra, "control-flow", "branch"),
    local!(Bssy, "synchronization", "sync"),
    local!(Bsync, "synchronization", "sync"),
    local!(Call, "control-flow", "call"),
    local!(Cs2r, "data-movement", "special-read"),
    local!(Exit, "control-flow", "exit"),
    local!(Fadd, "float-math", "float-add"),
    local!(Ffma, "float-math", "fused-multiply-add"),
    local!(Fmul, "float-math", "float-mul"),
    local!(Fsetp, "predicate", "compare-set"),
    local!(Hadd2, "float-math", "packed-half-add"),
    local!(Hfma2, "float-math", "fused-multiply-add"),
    local!(Hmul2, "float-math", "packed-half-mul"),
    local!(Iadd, "integer-math", "integer-add"),
    local!(Iadd3, "integer-math", "integer-add"),
    local!(Imad, "integer-math", "integer-mad"),
    local!(Isetp, "predicate", "compare-set"),
    local!(Ld, "memory", "load"),
    local!(Ldc, "memory", "load-const"),
    local!(Ldcu, "memory", "load-const"),
    local!(Ldg, "memory", "load"),
    local!(Ldl, "memory", "load"),
    local!(Lds, "memory", "load"),
    local!(Lea, "address", "address-calc"),
    local!(Lop3, "integer-math", "logic-lut"),
    local!(Mov, "data-movement", "move"),
    local!(Nop, "no-op", "no-op"),
    local!(Plop3, "integer-math", "logic-lut"),
    local!(Prmt, "data-movement", "permute"),
    local!(Ret, "control-flow", "return"),
    local!(S2r, "data-movement", "special-read"),
    local!(S2ur, "data-movement", "special-read"),
    local!(Shf, "integer-math", "shift"),
    local!(Shfl, "warp", "warp-shuffle"),
    local!(St, "memory", "store"),
    local!(Stg, "memory", "store"),
    local!(Stl, "memory", "store"),
    local!(Sts, "memory", "store"),
    local!(Uiadd3, "integer-math", "integer-add"),
    local!(Uimad, "integer-math", "integer-mad"),
    local!(Uisetp, "predicate", "compare-set"),
    local!(Uldc, "memory", "load-const"),
    local!(Ulea, "address", "address-calc"),
    local!(Ulop3, "integer-math", "logic-lut"),
    local!(Umov, "data-movement", "move"),
    local!(Ushf, "integer-math", "shift"),
    nvidia_mapped!(Bgmma, ["sm90"], "tensor-core", "warpgroup-mma"),
    nvidia_mapped!(
        Bmma,
        ["sm80", "sm86", "sm89", "sm90"],
        "tensor-core",
        "bit-mma"
    ),
    nvidia_mapped!(Dmma, ["sm100", "sm120"], "tensor-core", "fp64-mma"),
    nvidia_mapped!(Hgmma, ["sm90"], "tensor-core", "warpgroup-mma"),
    nvidia_mapped!(
        Hmma,
        ["sm80", "sm86", "sm89", "sm90", "sm100", "sm120"],
        "tensor-core",
        "half-mma"
    ),
    nvidia_mapped!(Igmma, ["sm90"], "tensor-core", "warpgroup-mma"),
    nvidia_mapped!(
        Imma,
        ["sm80", "sm86", "sm89", "sm90", "sm100", "sm120"],
        "tensor-core",
        "integer-mma"
    ),
    nvidia_mapped!(Omma, ["sm100", "sm120"], "tensor-core", "fp4-mma"),
    nvidia_mapped!(Qgmma, ["sm90"], "tensor-core", "warpgroup-mma"),
    nvidia_mapped!(Qmma, ["sm100", "sm120"], "tensor-core", "fp8-mma"),
    nvidia_mapped!(Ldt, ["sm100", "sm120"], "tensor-memory", "tensor-load"),
    nvidia_mapped!(
        Ldtm,
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-load-matrix"
    ),
    nvidia_mapped!(Stt, ["sm100", "sm120"], "tensor-memory", "tensor-store"),
    nvidia_mapped!(
        Sttm,
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-store-matrix"
    ),
    nvidia_mapped!(Ublkcp, ["sm100", "sm120"], "tensor-memory", "bulk-copy"),
    nvidia_mapped!(Ublkpf, ["sm100", "sm120"], "tensor-memory", "bulk-prefetch"),
    nvidia_mapped!(Ublkred, ["sm100", "sm120"], "tensor-memory", "bulk-reduce"),
    nvidia_mapped!(
        Utchmma,
        ["sm100", "sm120"],
        "tensor-core",
        "uniform-half-mma"
    ),
    nvidia_mapped!(
        Utcimma,
        ["sm100", "sm120"],
        "tensor-core",
        "uniform-integer-mma"
    ),
    nvidia_mapped!(
        Utcomma,
        ["sm100", "sm120"],
        "tensor-core",
        "uniform-fp4-mma"
    ),
    nvidia_mapped!(
        Utcqmma,
        ["sm100", "sm120"],
        "tensor-core",
        "uniform-fp8-mma"
    ),
    nvidia_mapped!(
        Utmaldg,
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-memory-load-global"
    ),
    nvidia_mapped!(
        Utmapf,
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-memory-prefetch"
    ),
    nvidia_mapped!(
        Utmaredg,
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-memory-reduce-global"
    ),
    nvidia_mapped!(
        Utmastg,
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-memory-store-global"
    ),
    nvidia_mapped!(Warpgroup, ["sm90"], "warpgroup", "warpgroup-control"),
    nvidia_mapped!(Warpgroupset, ["sm90"], "warpgroup", "warpgroup-control"),
];
