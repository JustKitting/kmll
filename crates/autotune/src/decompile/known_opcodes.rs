#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownSassOpcode {
    pub opcode: &'static str,
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
    ($opcode:literal, $class:literal, $kind:literal) => {
        KnownSassOpcode {
            opcode: $opcode,
            architectures: &[],
            class: $class,
            kind: $kind,
            source: LOCAL_LIFTER,
            locally_mapped: true,
        }
    };
}

macro_rules! nvidia_mapped {
    ($opcode:literal, [$($arch:literal),* $(,)?], $class:literal, $kind:literal) => {
        KnownSassOpcode {
            opcode: $opcode,
            architectures: &[$($arch),*],
            class: $class,
            kind: $kind,
            source: NVIDIA_BINARY_UTILITIES,
            locally_mapped: true,
        }
    };
}

const KNOWN_SASS_OPCODES: &[KnownSassOpcode] = &[
    local!("BAR", "synchronization", "sync"),
    local!("BRA", "control-flow", "branch"),
    local!("BSSY", "synchronization", "sync"),
    local!("BSYNC", "synchronization", "sync"),
    local!("CALL", "control-flow", "call"),
    local!("EXIT", "control-flow", "exit"),
    local!("FADD", "float-math", "float-add"),
    local!("FFMA", "float-math", "fused-multiply-add"),
    local!("FMUL", "float-math", "float-mul"),
    local!("FSETP", "predicate", "compare-set"),
    local!("HADD2", "float-math", "packed-half-add"),
    local!("HFMA2", "float-math", "fused-multiply-add"),
    local!("HMUL2", "float-math", "packed-half-mul"),
    local!("IADD", "integer-math", "integer-add"),
    local!("IADD3", "integer-math", "integer-add"),
    local!("IMAD", "integer-math", "integer-mad"),
    local!("ISETP", "predicate", "compare-set"),
    local!("LD", "memory", "load"),
    local!("LDC", "memory", "load-const"),
    local!("LDCU", "memory", "load-const"),
    local!("LDG", "memory", "load"),
    local!("LDL", "memory", "load"),
    local!("LDS", "memory", "load"),
    local!("LEA", "address", "address-calc"),
    local!("LOP3", "integer-math", "logic-lut"),
    local!("MOV", "data-movement", "move"),
    local!("NOP", "no-op", "no-op"),
    local!("PLOP3", "integer-math", "logic-lut"),
    local!("PRMT", "data-movement", "permute"),
    local!("RET", "control-flow", "return"),
    local!("S2R", "data-movement", "special-read"),
    local!("S2UR", "data-movement", "special-read"),
    local!("SHF", "integer-math", "shift"),
    local!("SHFL", "warp", "warp-shuffle"),
    local!("ST", "memory", "store"),
    local!("STG", "memory", "store"),
    local!("STL", "memory", "store"),
    local!("STS", "memory", "store"),
    local!("UIADD3", "integer-math", "integer-add"),
    local!("UIMAD", "integer-math", "integer-mad"),
    local!("UISETP", "predicate", "compare-set"),
    local!("ULDC", "memory", "load-const"),
    local!("ULEA", "address", "address-calc"),
    local!("ULOP3", "integer-math", "logic-lut"),
    local!("UMOV", "data-movement", "move"),
    local!("USHF", "integer-math", "shift"),
    nvidia_mapped!("BGMMA", ["sm90"], "tensor-core", "warpgroup-mma"),
    nvidia_mapped!(
        "BMMA",
        ["sm80", "sm86", "sm89", "sm90"],
        "tensor-core",
        "bit-mma"
    ),
    nvidia_mapped!("DMMA", ["sm100", "sm120"], "tensor-core", "fp64-mma"),
    nvidia_mapped!("HGMMA", ["sm90"], "tensor-core", "warpgroup-mma"),
    nvidia_mapped!(
        "HMMA",
        ["sm80", "sm86", "sm89", "sm90", "sm100", "sm120"],
        "tensor-core",
        "half-mma"
    ),
    nvidia_mapped!("IGMMA", ["sm90"], "tensor-core", "warpgroup-mma"),
    nvidia_mapped!(
        "IMMA",
        ["sm80", "sm86", "sm89", "sm90", "sm100", "sm120"],
        "tensor-core",
        "integer-mma"
    ),
    nvidia_mapped!("OMMA", ["sm100", "sm120"], "tensor-core", "fp4-mma"),
    nvidia_mapped!("QGMMA", ["sm90"], "tensor-core", "warpgroup-mma"),
    nvidia_mapped!("QMMA", ["sm100", "sm120"], "tensor-core", "fp8-mma"),
    nvidia_mapped!("LDT", ["sm100", "sm120"], "tensor-memory", "tensor-load"),
    nvidia_mapped!(
        "LDTM",
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-load-matrix"
    ),
    nvidia_mapped!("STT", ["sm100", "sm120"], "tensor-memory", "tensor-store"),
    nvidia_mapped!(
        "STTM",
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-store-matrix"
    ),
    nvidia_mapped!("UBLKCP", ["sm100", "sm120"], "tensor-memory", "bulk-copy"),
    nvidia_mapped!(
        "UBLKPF",
        ["sm100", "sm120"],
        "tensor-memory",
        "bulk-prefetch"
    ),
    nvidia_mapped!(
        "UBLKRED",
        ["sm100", "sm120"],
        "tensor-memory",
        "bulk-reduce"
    ),
    nvidia_mapped!(
        "UTCHMMA",
        ["sm100", "sm120"],
        "tensor-core",
        "uniform-half-mma"
    ),
    nvidia_mapped!(
        "UTCIMMA",
        ["sm100", "sm120"],
        "tensor-core",
        "uniform-integer-mma"
    ),
    nvidia_mapped!(
        "UTCOMMA",
        ["sm100", "sm120"],
        "tensor-core",
        "uniform-fp4-mma"
    ),
    nvidia_mapped!(
        "UTCQMMA",
        ["sm100", "sm120"],
        "tensor-core",
        "uniform-fp8-mma"
    ),
    nvidia_mapped!(
        "UTMALDG",
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-memory-load-global"
    ),
    nvidia_mapped!(
        "UTMAPF",
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-memory-prefetch"
    ),
    nvidia_mapped!(
        "UTMAREDG",
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-memory-reduce-global"
    ),
    nvidia_mapped!(
        "UTMASTG",
        ["sm100", "sm120"],
        "tensor-memory",
        "tensor-memory-store-global"
    ),
    nvidia_mapped!("WARPGROUP", ["sm90"], "warpgroup", "warpgroup-control"),
    nvidia_mapped!("WARPGROUPSET", ["sm90"], "warpgroup", "warpgroup-control"),
];
