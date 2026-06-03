use std::fmt::{self, Write as _};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIrModule {
    pub target: Option<String>,
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
                    op.address, op.kind, op.confidence, op.source
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
    pub name: String,
    pub ops: Vec<KernelIrOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIrOp {
    pub address: u64,
    pub label: Option<String>,
    pub predicate: Option<String>,
    pub kind: KernelIrOpKind,
    pub confidence: SassMappingConfidence,
    pub source_opcode: String,
    pub source_modifiers: Vec<String>,
    pub source_operands: Vec<String>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelIrOpKind {
    ReadSpecialRegister {
        dst: String,
        special: String,
    },
    Move {
        dst: String,
        src: String,
    },
    LoadConst {
        dst: String,
        source: String,
    },
    Load {
        dst: String,
        address: String,
        space: MemorySpace,
        access: MemoryAccessInfo,
    },
    Store {
        address: String,
        value: String,
        space: MemorySpace,
        access: MemoryAccessInfo,
    },
    IntegerAdd {
        dst: String,
        inputs: Vec<String>,
        width_bits: Option<u32>,
    },
    FloatAdd {
        dst: String,
        lhs: String,
        rhs: String,
    },
    FloatMul {
        dst: String,
        lhs: String,
        rhs: String,
    },
    PackedHalfAdd {
        dst: String,
        inputs: Vec<String>,
        lanes: u32,
    },
    PackedHalfMul {
        dst: String,
        inputs: Vec<String>,
        lanes: u32,
    },
    FusedMultiplyAdd {
        dst: String,
        a: String,
        b: String,
        c: String,
        lane_bits: Option<u32>,
    },
    IntegerMad {
        dst: String,
        a: String,
        b: String,
        c: String,
        wide: bool,
    },
    TensorCoreMma {
        opcode: String,
        operands: Vec<String>,
        element_type: Option<String>,
        scope: Option<String>,
    },
    TensorCoreMemory {
        opcode: String,
        operands: Vec<String>,
    },
    TensorMemoryAccess {
        opcode: String,
        operands: Vec<String>,
    },
    WarpGroup {
        opcode: String,
        operands: Vec<String>,
    },
    CompareSet {
        dst: String,
        comparison: Option<String>,
        dtype: Option<String>,
        lhs: String,
        rhs: String,
    },
    Branch {
        target: Option<String>,
        condition: Option<String>,
    },
    Call {
        target: Option<String>,
        operands: Vec<String>,
    },
    Return {
        target: Option<String>,
        operands: Vec<String>,
    },
    Exit {
        condition: Option<String>,
    },
    WarpShuffle {
        mode: Option<String>,
        predicate: String,
        dst: String,
        src: String,
        offset: String,
        mask: String,
    },
    Shift {
        dst: String,
        inputs: Vec<String>,
    },
    LogicLut {
        dst: String,
        inputs: Vec<String>,
    },
    Permute {
        dst: String,
        inputs: Vec<String>,
    },
    AddressCalc {
        dst: String,
        inputs: Vec<String>,
    },
    Sync {
        kind: String,
        operands: Vec<String>,
    },
    NoOp,
    Unsupported {
        opcode: String,
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    Global,
    Shared,
    Local,
    Constant,
    Descriptor,
    Unknown,
}

impl fmt::Display for MemorySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => f.write_str("global"),
            Self::Shared => f.write_str("shared"),
            Self::Local => f.write_str("local"),
            Self::Constant => f.write_str("constant"),
            Self::Descriptor => f.write_str("descriptor"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryAccessInfo {
    pub width_bits: Option<u32>,
    pub modifiers: Vec<String>,
}

impl MemoryAccessInfo {
    pub fn new(width_bits: Option<u32>, modifiers: Vec<String>) -> Self {
        Self {
            width_bits,
            modifiers,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
