use std::fmt;

use super::super::AggregateOperand;
use super::super::{PredicateCondition, RegisterRef, SassOpcode};
use super::semantics::SassLiftedSemantics;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassLiftedModule {
    pub target: Option<String>,
    pub functions: Vec<SassLiftedFunction>,
}

impl SassLiftedModule {
    pub fn op_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.ops.len())
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassLiftedFunction {
    pub name: String,
    pub ops: Vec<SassLiftedOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassLiftedOp {
    pub address: u64,
    pub block_id: Option<usize>,
    pub predicate: Option<PredicateCondition>,
    pub opcode: SassOpcode,
    pub class: SassLiftedOpClass,
    pub kind: SassLiftedOpKind,
    pub semantics: SassLiftedSemantics,
    pub inputs: Vec<SassLiftedValueRef>,
    pub outputs: Vec<SassLiftedValueRef>,
    pub source_operands: Vec<AggregateOperand>,
    pub detail: SassLiftedOpDetail,
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SassLiftedOpDetail {
    pub class: SassLiftedOpClass,
    pub kind: SassLiftedOpKind,
}

impl SassLiftedOpDetail {
    pub fn new(class: SassLiftedOpClass, kind: SassLiftedOpKind) -> Self {
        Self { class, kind }
    }
}

impl fmt::Display for SassLiftedOpDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "class={},kind={}", self.class, self.kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassLiftedValueRef {
    pub value_id: usize,
    pub register: RegisterRef,
}

impl SassLiftedValueRef {
    pub fn name(&self) -> String {
        format!("v{}:{}", self.value_id, self.register)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassLiftedOpClass {
    DataMovement,
    Memory,
    IntegerMath,
    FloatMath,
    TensorCore,
    TensorMemory,
    Predicate,
    ControlFlow,
    Warp,
    WarpGroup,
    Address,
    Synchronization,
    NoOp,
    Unsupported,
}

impl fmt::Display for SassLiftedOpClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataMovement => f.write_str("data-movement"),
            Self::Memory => f.write_str("memory"),
            Self::IntegerMath => f.write_str("integer-math"),
            Self::FloatMath => f.write_str("float-math"),
            Self::TensorCore => f.write_str("tensor-core"),
            Self::TensorMemory => f.write_str("tensor-memory"),
            Self::Predicate => f.write_str("predicate"),
            Self::ControlFlow => f.write_str("control-flow"),
            Self::Warp => f.write_str("warp"),
            Self::WarpGroup => f.write_str("warpgroup"),
            Self::Address => f.write_str("address"),
            Self::Synchronization => f.write_str("synchronization"),
            Self::NoOp => f.write_str("no-op"),
            Self::Unsupported => f.write_str("unsupported"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassLiftedOpKind {
    SpecialRead,
    Move,
    LoadConst,
    Load,
    Store,
    IntegerAdd,
    FloatAdd,
    FloatMul,
    PackedHalfAdd,
    PackedHalfMul,
    FusedMultiplyAdd,
    IntegerMad,
    TensorCoreMma,
    TensorCoreMemory,
    TensorMemoryAccess,
    WarpGroup,
    CompareSet,
    Branch,
    Call,
    Return,
    Exit,
    WarpShuffle,
    Shift,
    LogicLut,
    Permute,
    AddressCalc,
    Sync,
    NoOp,
    Unsupported,
}

impl fmt::Display for SassLiftedOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpecialRead => f.write_str("special-read"),
            Self::Move => f.write_str("move"),
            Self::LoadConst => f.write_str("load-const"),
            Self::Load => f.write_str("load"),
            Self::Store => f.write_str("store"),
            Self::IntegerAdd => f.write_str("integer-add"),
            Self::FloatAdd => f.write_str("float-add"),
            Self::FloatMul => f.write_str("float-mul"),
            Self::PackedHalfAdd => f.write_str("packed-half-add"),
            Self::PackedHalfMul => f.write_str("packed-half-mul"),
            Self::FusedMultiplyAdd => f.write_str("fused-multiply-add"),
            Self::IntegerMad => f.write_str("integer-mad"),
            Self::TensorCoreMma => f.write_str("tensor-core-mma"),
            Self::TensorCoreMemory => f.write_str("tensor-core-memory"),
            Self::TensorMemoryAccess => f.write_str("tensor-memory-access"),
            Self::WarpGroup => f.write_str("warpgroup"),
            Self::CompareSet => f.write_str("compare-set"),
            Self::Branch => f.write_str("branch"),
            Self::Call => f.write_str("call"),
            Self::Return => f.write_str("return"),
            Self::Exit => f.write_str("exit"),
            Self::WarpShuffle => f.write_str("warp-shuffle"),
            Self::Shift => f.write_str("shift"),
            Self::LogicLut => f.write_str("logic-lut"),
            Self::Permute => f.write_str("permute"),
            Self::AddressCalc => f.write_str("address-calc"),
            Self::Sync => f.write_str("sync"),
            Self::NoOp => f.write_str("no-op"),
            Self::Unsupported => f.write_str("unsupported"),
        }
    }
}
