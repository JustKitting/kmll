use std::fmt::{self, Write as _};

use super::super::super::SassTarget;
use super::super::super::sass::SassSourcePosition;
use super::{
    AggregateOperand, ControlTarget, MemoryAccessInfo, MemoryAddress, MemorySpace,
    PredicateCondition, RegisterRef, SassCompareDType, SassComparisonKind, SassMemoryAtomicOp,
    SassModifier, SassNumericDType, SassOpcode, SassTensorElementType, SassTensorMmaSignature,
    SassTensorScope, ScalarOperand,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIrModule {
    pub target: Option<SassTarget>,
    pub functions: Vec<KernelIrFunction>,
}

impl KernelIrModule {
    pub fn unsupported_instruction_count(&self) -> usize {
        self.functions
            .iter()
            .flat_map(|function| function.ops.iter())
            .filter(|op| matches!(op.kind, KernelIrOpKind::Unsupported { .. }))
            .count()
    }

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        if let Some(target) = &self.target {
            writeln!(out, "target {target}").expect("write to string");
        }
        for function in &self.functions {
            writeln!(out, "fn {} {{", function.name).expect("write to string");
            for op in &function.ops {
                writeln!(
                    out,
                    "  {:#06x}: {:?} [{}] <- {}",
                    op.address, op.kind, op.confidence, op.source_text
                )
                .expect("write to string");
            }
            writeln!(out, "}}").expect("write to string");
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIrFunction {
    pub name: SassSymbol,
    pub ops: Vec<KernelIrOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIrOp {
    pub address: u64,
    pub source_position: SassSourcePosition,
    pub label: Option<SassSymbol>,
    pub predicate: Option<PredicateCondition>,
    pub kind: KernelIrOpKind,
    pub confidence: SassMappingConfidence,
    pub source_opcode: SassOpcode,
    pub source_modifiers: Vec<SassModifier>,
    pub source_operands: Vec<AggregateOperand>,
    pub source_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassSymbol {
    raw: String,
}

impl SassSymbol {
    pub fn new(raw: impl Into<String>) -> Self {
        Self { raw: raw.into() }
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }
}

impl fmt::Display for SassSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelIrOpKind {
    ReadSpecialRegister {
        dst: RegisterRef,
        special: RegisterRef,
    },
    Move {
        dst: RegisterRef,
        src: ScalarOperand,
    },
    Select {
        dst: RegisterRef,
        true_value: ScalarOperand,
        false_value: ScalarOperand,
        predicate: RegisterRef,
    },
    NumericConvert {
        dst: RegisterRef,
        src: ScalarOperand,
        dst_dtype: Option<SassNumericDType>,
        src_dtype: Option<SassNumericDType>,
    },
    LoadConst {
        dst: RegisterRef,
        source: MemoryAddress,
    },
    Load {
        dst: RegisterRef,
        address: MemoryAddress,
        space: MemorySpace,
        access: MemoryAccessInfo,
    },
    Store {
        address: MemoryAddress,
        value: RegisterRef,
        space: MemorySpace,
        access: MemoryAccessInfo,
    },
    MemoryAtomic {
        predicate_dst: Option<RegisterRef>,
        dst: RegisterRef,
        address: MemoryAddress,
        values: Vec<ScalarOperand>,
        operation: Option<SassMemoryAtomicOp>,
        space: MemorySpace,
        access: MemoryAccessInfo,
    },
    MemoryReduction {
        address: MemoryAddress,
        values: Vec<ScalarOperand>,
        operation: Option<SassMemoryAtomicOp>,
        space: MemorySpace,
        access: MemoryAccessInfo,
    },
    IntegerAdd {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
        width_bits: Option<u32>,
    },
    FloatAdd {
        dst: RegisterRef,
        lhs: ScalarOperand,
        rhs: ScalarOperand,
    },
    FloatMul {
        dst: RegisterRef,
        lhs: ScalarOperand,
        rhs: ScalarOperand,
    },
    PackedHalfAdd {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
        lanes: u32,
    },
    PackedHalfMul {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
        lanes: u32,
    },
    FusedMultiplyAdd {
        dst: RegisterRef,
        a: ScalarOperand,
        b: ScalarOperand,
        c: ScalarOperand,
        lane_bits: Option<u32>,
    },
    IntegerMad {
        dst: RegisterRef,
        a: ScalarOperand,
        b: ScalarOperand,
        c: ScalarOperand,
        wide: bool,
    },
    TensorCoreMma {
        opcode: SassOpcode,
        operands: Vec<AggregateOperand>,
        element_type: Option<SassTensorElementType>,
        signature: Option<SassTensorMmaSignature>,
        scope: Option<SassTensorScope>,
    },
    TensorCoreMemory {
        opcode: SassOpcode,
        operands: Vec<AggregateOperand>,
    },
    TensorMemoryAccess {
        opcode: SassOpcode,
        operands: Vec<AggregateOperand>,
    },
    WarpGroup {
        opcode: SassOpcode,
        operands: Vec<AggregateOperand>,
    },
    CompareSet {
        dst: RegisterRef,
        comparison: Option<SassComparisonKind>,
        dtype: Option<SassCompareDType>,
        lhs: ScalarOperand,
        rhs: ScalarOperand,
    },
    Branch {
        target: Option<ControlTarget>,
        condition: Option<PredicateCondition>,
    },
    Call {
        target: Option<ControlTarget>,
        operands: Vec<AggregateOperand>,
    },
    Return {
        target: Option<ControlTarget>,
        operands: Vec<AggregateOperand>,
    },
    Exit {
        condition: Option<PredicateCondition>,
    },
    WarpShuffle {
        mode: Option<SassWarpShuffleMode>,
        predicate: RegisterRef,
        dst: RegisterRef,
        src: ScalarOperand,
        offset: ScalarOperand,
        mask: ScalarOperand,
    },
    Shift {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
    },
    LogicLut {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
    },
    Permute {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
    },
    AddressCalc {
        dst: RegisterRef,
        inputs: Vec<ScalarOperand>,
    },
    Sync {
        kind: SassSyncKind,
        operands: Vec<AggregateOperand>,
    },
    NoOp,
    Unsupported {
        opcode: SassOpcode,
        reason: SassUnsupportedReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassUnsupportedReason {
    NoLocalMapping,
    OperandArity {
        expectation: SassOperandArityExpectation,
        expected: usize,
        actual: usize,
    },
}

impl SassUnsupportedReason {
    pub const fn no_local_mapping() -> Self {
        Self::NoLocalMapping
    }

    pub const fn exact_operand_arity(expected: usize, actual: usize) -> Self {
        Self::OperandArity {
            expectation: SassOperandArityExpectation::Exact,
            expected,
            actual,
        }
    }

    pub const fn at_least_operand_arity(expected: usize, actual: usize) -> Self {
        Self::OperandArity {
            expectation: SassOperandArityExpectation::AtLeast,
            expected,
            actual,
        }
    }
}

impl fmt::Display for SassUnsupportedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLocalMapping => f.write_str("no local mapping for opcode yet"),
            Self::OperandArity {
                expectation,
                expected,
                actual,
            } => match expectation {
                SassOperandArityExpectation::Exact => {
                    write!(f, "expected {expected} operands, saw {actual}")
                }
                SassOperandArityExpectation::AtLeast => {
                    write!(f, "expected at least {expected} operands, saw {actual}")
                }
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassOperandArityExpectation {
    Exact,
    AtLeast,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassSyncKind {
    BarrierSet,
    BarrierSync,
    Barrier,
    Raw(String),
}

impl SassSyncKind {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "BSSY" => Self::BarrierSet,
            "BSYNC" => Self::BarrierSync,
            "BAR" => Self::Barrier,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::BarrierSet => "BSSY",
            Self::BarrierSync => "BSYNC",
            Self::Barrier => "BAR",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassSyncKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassWarpShuffleMode {
    Up,
    Down,
    Bfly,
    Index,
    Raw(String),
}

impl SassWarpShuffleMode {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "UP" => Self::Up,
            "DOWN" => Self::Down,
            "BFLY" => Self::Bfly,
            "IDX" => Self::Index,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Up => "UP",
            Self::Down => "DOWN",
            Self::Bfly => "BFLY",
            Self::Index => "IDX",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassWarpShuffleMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]

pub enum SassMappingConfidence {
    LocallyParsed,
    OpcodeHeuristic,
    Unsupported,
}

impl fmt::Display for SassMappingConfidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocallyParsed => f.write_str("locally-parsed"),
            Self::OpcodeHeuristic => f.write_str("opcode-heuristic"),
            Self::Unsupported => f.write_str("unsupported"),
        }
    }
}
