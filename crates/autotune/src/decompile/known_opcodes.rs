use std::fmt;

use super::{SassArchitecture, SassLiftedOpClass, SassLiftedOpKind, SassOpcodeKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownSassOpcode {
    pub opcode: SassOpcodeKind,
    pub architectures: &'static [SassArchitecture],
    pub class: SassOpcodeCatalogClass,
    pub kind: SassOpcodeCatalogKind,
    pub source: SassOpcodeCatalogSource,
    pub locally_mapped: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassOpcodeCatalogSource {
    LocalSassLifter,
    NvidiaCudaBinaryUtilitiesInstructionReference,
}

impl fmt::Display for SassOpcodeCatalogSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalSassLifter => f.write_str("local-sass-lifter"),
            Self::NvidiaCudaBinaryUtilitiesInstructionReference => {
                f.write_str("nvidia-cuda-binary-utilities-instruction-reference")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassOpcodeCatalogClass {
    Address,
    ControlFlow,
    DataMovement,
    FloatMath,
    IntegerMath,
    Memory,
    NoOp,
    Predicate,
    Synchronization,
    TensorCore,
    TensorMemory,
    Unsupported,
    Warp,
    WarpGroup,
}

impl From<SassLiftedOpClass> for SassOpcodeCatalogClass {
    fn from(class: SassLiftedOpClass) -> Self {
        match class {
            SassLiftedOpClass::Address => Self::Address,
            SassLiftedOpClass::ControlFlow => Self::ControlFlow,
            SassLiftedOpClass::DataMovement => Self::DataMovement,
            SassLiftedOpClass::FloatMath => Self::FloatMath,
            SassLiftedOpClass::IntegerMath => Self::IntegerMath,
            SassLiftedOpClass::Memory => Self::Memory,
            SassLiftedOpClass::NoOp => Self::NoOp,
            SassLiftedOpClass::Predicate => Self::Predicate,
            SassLiftedOpClass::Synchronization => Self::Synchronization,
            SassLiftedOpClass::TensorCore => Self::TensorCore,
            SassLiftedOpClass::TensorMemory => Self::TensorMemory,
            SassLiftedOpClass::Unsupported => Self::Unsupported,
            SassLiftedOpClass::Warp => Self::Warp,
            SassLiftedOpClass::WarpGroup => Self::WarpGroup,
        }
    }
}

impl fmt::Display for SassOpcodeCatalogClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Address => f.write_str("address"),
            Self::ControlFlow => f.write_str("control-flow"),
            Self::DataMovement => f.write_str("data-movement"),
            Self::FloatMath => f.write_str("float-math"),
            Self::IntegerMath => f.write_str("integer-math"),
            Self::Memory => f.write_str("memory"),
            Self::NoOp => f.write_str("no-op"),
            Self::Predicate => f.write_str("predicate"),
            Self::Synchronization => f.write_str("synchronization"),
            Self::TensorCore => f.write_str("tensor-core"),
            Self::TensorMemory => f.write_str("tensor-memory"),
            Self::Unsupported => f.write_str("unsupported"),
            Self::Warp => f.write_str("warp"),
            Self::WarpGroup => f.write_str("warpgroup"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassOpcodeCatalogKind {
    AddressCalc,
    BitMma,
    Branch,
    BulkCopy,
    BulkPrefetch,
    BulkReduce,
    Call,
    CompareSet,
    Exit,
    FloatAdd,
    FloatMul,
    Fp4Mma,
    Fp8Mma,
    Fp64Mma,
    FusedMultiplyAdd,
    HalfMma,
    IntegerAdd,
    IntegerMad,
    IntegerMma,
    Load,
    LoadConst,
    LogicLut,
    Move,
    NoOp,
    PackedHalfAdd,
    PackedHalfMul,
    Permute,
    Return,
    Shift,
    SpecialRead,
    Store,
    Sync,
    TensorCoreMemory,
    TensorCoreMma,
    TensorLoad,
    TensorLoadMatrix,
    TensorMemoryAccess,
    TensorMemoryLoadGlobal,
    TensorMemoryPrefetch,
    TensorMemoryReduceGlobal,
    TensorMemoryStoreGlobal,
    TensorStore,
    TensorStoreMatrix,
    UniformFp4Mma,
    UniformFp8Mma,
    UniformHalfMma,
    UniformIntegerMma,
    Unsupported,
    WarpGroup,
    WarpGroupControl,
    WarpGroupMma,
    WarpShuffle,
}

impl From<SassLiftedOpKind> for SassOpcodeCatalogKind {
    fn from(kind: SassLiftedOpKind) -> Self {
        match kind {
            SassLiftedOpKind::AddressCalc => Self::AddressCalc,
            SassLiftedOpKind::Branch => Self::Branch,
            SassLiftedOpKind::Call => Self::Call,
            SassLiftedOpKind::CompareSet => Self::CompareSet,
            SassLiftedOpKind::Exit => Self::Exit,
            SassLiftedOpKind::FloatAdd => Self::FloatAdd,
            SassLiftedOpKind::FloatMul => Self::FloatMul,
            SassLiftedOpKind::FusedMultiplyAdd => Self::FusedMultiplyAdd,
            SassLiftedOpKind::IntegerAdd => Self::IntegerAdd,
            SassLiftedOpKind::IntegerMad => Self::IntegerMad,
            SassLiftedOpKind::Load => Self::Load,
            SassLiftedOpKind::LoadConst => Self::LoadConst,
            SassLiftedOpKind::LogicLut => Self::LogicLut,
            SassLiftedOpKind::Move => Self::Move,
            SassLiftedOpKind::NoOp => Self::NoOp,
            SassLiftedOpKind::PackedHalfAdd => Self::PackedHalfAdd,
            SassLiftedOpKind::PackedHalfMul => Self::PackedHalfMul,
            SassLiftedOpKind::Permute => Self::Permute,
            SassLiftedOpKind::Return => Self::Return,
            SassLiftedOpKind::Shift => Self::Shift,
            SassLiftedOpKind::SpecialRead => Self::SpecialRead,
            SassLiftedOpKind::Store => Self::Store,
            SassLiftedOpKind::Sync => Self::Sync,
            SassLiftedOpKind::TensorCoreMemory => Self::TensorCoreMemory,
            SassLiftedOpKind::TensorCoreMma => Self::TensorCoreMma,
            SassLiftedOpKind::TensorMemoryAccess => Self::TensorMemoryAccess,
            SassLiftedOpKind::Unsupported => Self::Unsupported,
            SassLiftedOpKind::WarpGroup => Self::WarpGroup,
            SassLiftedOpKind::WarpShuffle => Self::WarpShuffle,
        }
    }
}

impl fmt::Display for SassOpcodeCatalogKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AddressCalc => f.write_str("address-calc"),
            Self::BitMma => f.write_str("bit-mma"),
            Self::Branch => f.write_str("branch"),
            Self::BulkCopy => f.write_str("bulk-copy"),
            Self::BulkPrefetch => f.write_str("bulk-prefetch"),
            Self::BulkReduce => f.write_str("bulk-reduce"),
            Self::Call => f.write_str("call"),
            Self::CompareSet => f.write_str("compare-set"),
            Self::Exit => f.write_str("exit"),
            Self::FloatAdd => f.write_str("float-add"),
            Self::FloatMul => f.write_str("float-mul"),
            Self::Fp4Mma => f.write_str("fp4-mma"),
            Self::Fp8Mma => f.write_str("fp8-mma"),
            Self::Fp64Mma => f.write_str("fp64-mma"),
            Self::FusedMultiplyAdd => f.write_str("fused-multiply-add"),
            Self::HalfMma => f.write_str("half-mma"),
            Self::IntegerAdd => f.write_str("integer-add"),
            Self::IntegerMad => f.write_str("integer-mad"),
            Self::IntegerMma => f.write_str("integer-mma"),
            Self::Load => f.write_str("load"),
            Self::LoadConst => f.write_str("load-const"),
            Self::LogicLut => f.write_str("logic-lut"),
            Self::Move => f.write_str("move"),
            Self::NoOp => f.write_str("no-op"),
            Self::PackedHalfAdd => f.write_str("packed-half-add"),
            Self::PackedHalfMul => f.write_str("packed-half-mul"),
            Self::Permute => f.write_str("permute"),
            Self::Return => f.write_str("return"),
            Self::Shift => f.write_str("shift"),
            Self::SpecialRead => f.write_str("special-read"),
            Self::Store => f.write_str("store"),
            Self::Sync => f.write_str("sync"),
            Self::TensorCoreMemory => f.write_str("tensor-core-memory"),
            Self::TensorCoreMma => f.write_str("tensor-core-mma"),
            Self::TensorLoad => f.write_str("tensor-load"),
            Self::TensorLoadMatrix => f.write_str("tensor-load-matrix"),
            Self::TensorMemoryAccess => f.write_str("tensor-memory-access"),
            Self::TensorMemoryLoadGlobal => f.write_str("tensor-memory-load-global"),
            Self::TensorMemoryPrefetch => f.write_str("tensor-memory-prefetch"),
            Self::TensorMemoryReduceGlobal => f.write_str("tensor-memory-reduce-global"),
            Self::TensorMemoryStoreGlobal => f.write_str("tensor-memory-store-global"),
            Self::TensorStore => f.write_str("tensor-store"),
            Self::TensorStoreMatrix => f.write_str("tensor-store-matrix"),
            Self::UniformFp4Mma => f.write_str("uniform-fp4-mma"),
            Self::UniformFp8Mma => f.write_str("uniform-fp8-mma"),
            Self::UniformHalfMma => f.write_str("uniform-half-mma"),
            Self::UniformIntegerMma => f.write_str("uniform-integer-mma"),
            Self::Unsupported => f.write_str("unsupported"),
            Self::WarpGroup => f.write_str("warpgroup"),
            Self::WarpGroupControl => f.write_str("warpgroup-control"),
            Self::WarpGroupMma => f.write_str("warpgroup-mma"),
            Self::WarpShuffle => f.write_str("warp-shuffle"),
        }
    }
}

pub fn known_sass_opcodes() -> &'static [KnownSassOpcode] {
    KNOWN_SASS_OPCODES
}

macro_rules! local {
    ($opcode:ident, $class:ident, $kind:ident) => {
        KnownSassOpcode {
            opcode: SassOpcodeKind::$opcode,
            architectures: &[],
            class: SassOpcodeCatalogClass::$class,
            kind: SassOpcodeCatalogKind::$kind,
            source: SassOpcodeCatalogSource::LocalSassLifter,
            locally_mapped: true,
        }
    };
}

macro_rules! nvidia_mapped {
    ($opcode:ident, [$($arch:literal),* $(,)?], $class:ident, $kind:ident) => {
        KnownSassOpcode {
            opcode: SassOpcodeKind::$opcode,
            architectures: &[$(SassArchitecture::sm($arch)),*],
            class: SassOpcodeCatalogClass::$class,
            kind: SassOpcodeCatalogKind::$kind,
            source: SassOpcodeCatalogSource::NvidiaCudaBinaryUtilitiesInstructionReference,
            locally_mapped: true,
        }
    };
}

const KNOWN_SASS_OPCODES: &[KnownSassOpcode] = &[
    local!(Bar, Synchronization, Sync),
    local!(Bra, ControlFlow, Branch),
    local!(Bssy, Synchronization, Sync),
    local!(Bsync, Synchronization, Sync),
    local!(Call, ControlFlow, Call),
    local!(Cs2r, DataMovement, SpecialRead),
    local!(Exit, ControlFlow, Exit),
    local!(Fadd, FloatMath, FloatAdd),
    local!(Ffma, FloatMath, FusedMultiplyAdd),
    local!(Fmul, FloatMath, FloatMul),
    local!(Fsetp, Predicate, CompareSet),
    local!(Hadd2, FloatMath, PackedHalfAdd),
    local!(Hfma2, FloatMath, FusedMultiplyAdd),
    local!(Hmul2, FloatMath, PackedHalfMul),
    local!(Iadd, IntegerMath, IntegerAdd),
    local!(Iadd3, IntegerMath, IntegerAdd),
    local!(Imad, IntegerMath, IntegerMad),
    local!(Isetp, Predicate, CompareSet),
    local!(Ld, Memory, Load),
    local!(Ldc, Memory, LoadConst),
    local!(Ldcu, Memory, LoadConst),
    local!(Ldg, Memory, Load),
    local!(Ldl, Memory, Load),
    local!(Lds, Memory, Load),
    local!(Lea, Address, AddressCalc),
    local!(Lop3, IntegerMath, LogicLut),
    local!(Mov, DataMovement, Move),
    local!(Nop, NoOp, NoOp),
    local!(Plop3, IntegerMath, LogicLut),
    local!(Prmt, DataMovement, Permute),
    local!(Ret, ControlFlow, Return),
    local!(S2r, DataMovement, SpecialRead),
    local!(S2ur, DataMovement, SpecialRead),
    local!(Shf, IntegerMath, Shift),
    local!(Shfl, Warp, WarpShuffle),
    local!(St, Memory, Store),
    local!(Stg, Memory, Store),
    local!(Stl, Memory, Store),
    local!(Sts, Memory, Store),
    local!(Uiadd3, IntegerMath, IntegerAdd),
    local!(Uimad, IntegerMath, IntegerMad),
    local!(Uisetp, Predicate, CompareSet),
    local!(Uldc, Memory, LoadConst),
    local!(Ulea, Address, AddressCalc),
    local!(Ulop3, IntegerMath, LogicLut),
    local!(Umov, DataMovement, Move),
    local!(Ushf, IntegerMath, Shift),
    nvidia_mapped!(Bgmma, [90], TensorCore, WarpGroupMma),
    nvidia_mapped!(Bmma, [80, 86, 89, 90], TensorCore, BitMma),
    nvidia_mapped!(Dmma, [100, 120], TensorCore, Fp64Mma),
    nvidia_mapped!(Hgmma, [90], TensorCore, WarpGroupMma),
    nvidia_mapped!(Hmma, [80, 86, 89, 90, 100, 120], TensorCore, HalfMma),
    nvidia_mapped!(Igmma, [90], TensorCore, WarpGroupMma),
    nvidia_mapped!(Imma, [80, 86, 89, 90, 100, 120], TensorCore, IntegerMma),
    nvidia_mapped!(Omma, [100, 120], TensorCore, Fp4Mma),
    nvidia_mapped!(Qgmma, [90], TensorCore, WarpGroupMma),
    nvidia_mapped!(Qmma, [100, 120], TensorCore, Fp8Mma),
    nvidia_mapped!(Ldt, [100, 120], TensorMemory, TensorLoad),
    nvidia_mapped!(Ldtm, [100, 120], TensorMemory, TensorLoadMatrix),
    nvidia_mapped!(Stt, [100, 120], TensorMemory, TensorStore),
    nvidia_mapped!(Sttm, [100, 120], TensorMemory, TensorStoreMatrix),
    nvidia_mapped!(Ublkcp, [100, 120], TensorMemory, BulkCopy),
    nvidia_mapped!(Ublkpf, [100, 120], TensorMemory, BulkPrefetch),
    nvidia_mapped!(Ublkred, [100, 120], TensorMemory, BulkReduce),
    nvidia_mapped!(Utchmma, [100, 120], TensorCore, UniformHalfMma),
    nvidia_mapped!(Utcimma, [100, 120], TensorCore, UniformIntegerMma),
    nvidia_mapped!(Utcomma, [100, 120], TensorCore, UniformFp4Mma),
    nvidia_mapped!(Utcqmma, [100, 120], TensorCore, UniformFp8Mma),
    nvidia_mapped!(Utmaldg, [100, 120], TensorMemory, TensorMemoryLoadGlobal),
    nvidia_mapped!(Utmapf, [100, 120], TensorMemory, TensorMemoryPrefetch),
    nvidia_mapped!(Utmaredg, [100, 120], TensorMemory, TensorMemoryReduceGlobal),
    nvidia_mapped!(Utmastg, [100, 120], TensorMemory, TensorMemoryStoreGlobal),
    nvidia_mapped!(Warpgroup, [90], WarpGroup, WarpGroupControl),
    nvidia_mapped!(Warpgroupset, [90], WarpGroup, WarpGroupControl),
];
