use std::{error::Error, fmt};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassModule {
    pub target: Option<String>,
    pub functions: Vec<SassFunction>,
}

impl SassModule {
    pub fn instruction_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.instructions.len())
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassFunction {
    pub name: String,
    pub section: Option<String>,
    pub instructions: Vec<SassInstruction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassInstruction {
    pub address: u64,
    pub source_position: SassSourcePosition,
    pub label: Option<String>,
    pub predicate: Option<SassPredicate>,
    pub opcode: String,
    pub modifiers: Vec<String>,
    pub operands: Vec<SassOperand>,
    pub raw: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassSourcePosition {
    pub line: usize,
    pub instruction_ordinal: usize,
}

impl SassSourcePosition {
    pub fn new(line: usize, instruction_ordinal: usize) -> Self {
        Self {
            line,
            instruction_ordinal,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassPredicate {
    pub negated: bool,
    pub register: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassOperand {
    pub raw: String,
    pub kind: SassOperandKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SassOperandKind {
    Register(SassRegister),
    Immediate(String),
    ConstantMemory {
        bank: String,
        offset: String,
    },
    DescriptorMemory {
        descriptor: String,
        address: String,
        address_width: Option<u32>,
        offset: Option<String>,
    },
    IndexedMemory {
        base: String,
        offset: Option<String>,
    },
    Label(String),
    Raw,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassRegister {
    pub class: RegisterClass,
    pub index: Option<u16>,
    pub modifiers: Vec<String>,
    pub negated: bool,
    pub absolute: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterClass {
    General,
    Uniform,
    Predicate,
    UniformPredicate,
    Special,
    Barrier,
    Zero,
    PredicateTrue,
    UniformPredicateTrue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassParseError {
    message: String,
}

impl SassParseError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SassParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for SassParseError {}
