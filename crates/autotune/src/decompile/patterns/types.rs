use std::fmt;

use super::super::{KernelIrOp, RegisterRef, SassOpcode, SassSymbol, SassTarget, ScalarOperand};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassPatternModule {
    pub target: Option<SassTarget>,
    pub functions: Vec<SassPatternFunction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassPatternFunction {
    pub name: SassSymbol,
    pub patterns: Vec<SassSemanticPattern>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassSemanticPattern {
    pub start_address: u64,
    pub end_address: u64,
    pub source_addresses: Vec<u64>,
    pub kind: SassSemanticPatternKind,
    pub confidence: SassPatternConfidence,
}

impl SassSemanticPattern {
    pub fn kind_name(&self) -> &'static str {
        self.kind.name()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassSemanticPatternCategory {
    Bf16WidenBits,
    F32MulAddPair,
    AddressPair,
    WarpReduceSum,
}

impl SassSemanticPatternCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16WidenBits => "bf16-widen-bits",
            Self::F32MulAddPair => "f32-mul-add-pair",
            Self::AddressPair => "address-pair",
            Self::WarpReduceSum => "warp-reduce-sum",
        }
    }
}

impl fmt::Display for SassSemanticPatternCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum SassSemanticPatternKind {
    Bf16WidenBits {
        src: RegisterRef,
        dst: RegisterRef,
        producer: Option<SassPatternLinkedOp>,
        consumer: Option<SassPatternLinkedOp>,
    },
    F32MulAddPair {
        mul_dst: RegisterRef,
        mul_lhs: ScalarOperand,
        mul_rhs: ScalarOperand,
        add_dst: RegisterRef,
        add_other: ScalarOperand,
    },
    AddressPair {
        low_dst: RegisterRef,
        high_dst: RegisterRef,
        low_inputs: Vec<ScalarOperand>,
        high_inputs: Vec<ScalarOperand>,
    },
    WarpReduceSum {
        input: ScalarOperand,
        output: RegisterRef,
        offsets: Vec<ScalarOperand>,
        mask: Option<ScalarOperand>,
    },
}

impl SassSemanticPatternKind {
    pub fn category(&self) -> SassSemanticPatternCategory {
        match self {
            Self::Bf16WidenBits { .. } => SassSemanticPatternCategory::Bf16WidenBits,
            Self::F32MulAddPair { .. } => SassSemanticPatternCategory::F32MulAddPair,
            Self::AddressPair { .. } => SassSemanticPatternCategory::AddressPair,
            Self::WarpReduceSum { .. } => SassSemanticPatternCategory::WarpReduceSum,
        }
    }

    pub fn name(&self) -> &'static str {
        self.category().as_str()
    }
}

impl fmt::Debug for SassSemanticPatternKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bf16WidenBits {
                src,
                dst,
                producer,
                consumer,
            } => f
                .debug_struct("Bf16WidenBits")
                .field("src", &src.to_string())
                .field("dst", &dst.to_string())
                .field("producer", producer)
                .field("consumer", consumer)
                .finish(),
            Self::F32MulAddPair {
                mul_dst,
                mul_lhs,
                mul_rhs,
                add_dst,
                add_other,
            } => f
                .debug_struct("F32MulAddPair")
                .field("mul_dst", &mul_dst.to_string())
                .field("mul_lhs", &mul_lhs.to_string())
                .field("mul_rhs", &mul_rhs.to_string())
                .field("add_dst", &add_dst.to_string())
                .field("add_other", &add_other.to_string())
                .finish(),
            Self::AddressPair {
                low_dst,
                high_dst,
                low_inputs,
                high_inputs,
            } => f
                .debug_struct("AddressPair")
                .field("low_dst", &low_dst.to_string())
                .field("high_dst", &high_dst.to_string())
                .field("low_inputs", &display_vec(low_inputs))
                .field("high_inputs", &display_vec(high_inputs))
                .finish(),
            Self::WarpReduceSum {
                input,
                output,
                offsets,
                mask,
            } => f
                .debug_struct("WarpReduceSum")
                .field("input", &input.to_string())
                .field("output", &output.to_string())
                .field("offsets", &display_vec(offsets))
                .field("mask", &mask.as_ref().map(ToString::to_string))
                .finish(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassPatternLinkedOp {
    pub address: u64,
    pub opcode: SassOpcode,
}

impl SassPatternLinkedOp {
    pub(super) fn new(op: &KernelIrOp) -> Self {
        Self {
            address: op.address,
            opcode: op.source_opcode.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassPatternConfidence {
    ExactOpcodeSequence,
    HeuristicDataflow,
}

impl fmt::Display for SassPatternConfidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExactOpcodeSequence => f.write_str("exact-opcode-sequence"),
            Self::HeuristicDataflow => f.write_str("heuristic-dataflow"),
        }
    }
}

fn display_vec<T: fmt::Display>(values: &[T]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}
