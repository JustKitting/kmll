use std::fmt::Write as _;

use super::sass::{label_in_text, *};

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
    },
    Store {
        address: String,
        value: String,
        space: MemorySpace,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassMappingConfidence {
    LocallyParsed,
    OpcodeHeuristic,
    Unsupported,
}

impl std::fmt::Display for SassMappingConfidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocallyParsed => f.write_str("locally-parsed"),
            Self::OpcodeHeuristic => f.write_str("opcode-heuristic"),
            Self::Unsupported => f.write_str("unsupported"),
        }
    }
}

pub fn lower_sass_module(module: &SassModule) -> KernelIrModule {
    KernelIrModule {
        target: module.target.clone(),
        functions: module
            .functions
            .iter()
            .map(|function| KernelIrFunction {
                name: function.name.clone(),
                ops: function
                    .instructions
                    .iter()
                    .map(lower_instruction)
                    .collect(),
            })
            .collect(),
    }
}

fn lower_instruction(instruction: &SassInstruction) -> KernelIrOp {
    let source_operands = instruction
        .operands
        .iter()
        .map(|operand| operand.raw.clone())
        .collect::<Vec<_>>();
    let (kind, confidence) = lower_kind(instruction);
    KernelIrOp {
        address: instruction.address,
        label: instruction.label.clone(),
        predicate: instruction.predicate.as_ref().map(predicate_text),
        kind,
        confidence,
        source_opcode: instruction.opcode.clone(),
        source_modifiers: instruction.modifiers.clone(),
        source_operands,
        source: instruction.raw.clone(),
    }
}

fn lower_kind(instruction: &SassInstruction) -> (KernelIrOpKind, SassMappingConfidence) {
    let operands = instruction
        .operands
        .iter()
        .map(|operand| operand.raw.clone())
        .collect::<Vec<_>>();
    let op = instruction.opcode.as_str();
    match op {
        "NOP" => (KernelIrOpKind::NoOp, SassMappingConfidence::LocallyParsed),
        "EXIT" => (
            KernelIrOpKind::Exit {
                condition: instruction.predicate.as_ref().map(predicate_text),
            },
            SassMappingConfidence::LocallyParsed,
        ),
        "BRA" => (
            KernelIrOpKind::Branch {
                target: label_operand(instruction),
                condition: instruction
                    .predicate
                    .as_ref()
                    .map(predicate_text)
                    .or_else(|| branch_condition_operand(instruction)),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "CALL" => (
            KernelIrOpKind::Call {
                target: label_operand(instruction),
                operands,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "RET" => (
            KernelIrOpKind::Return {
                target: label_operand(instruction),
                operands,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "SHFL" => map_five_operands(instruction, |predicate, dst, src, offset, mask| {
            KernelIrOpKind::WarpShuffle {
                mode: instruction.modifiers.first().cloned(),
                predicate,
                dst,
                src,
                offset,
                mask,
            }
        }),
        "S2R" | "S2UR" => map_two_operands(instruction, |dst, special| {
            KernelIrOpKind::ReadSpecialRegister { dst, special }
        }),
        "MOV" | "UMOV" => {
            map_two_operands(instruction, |dst, src| KernelIrOpKind::Move { dst, src })
        }
        "LDC" | "LDCU" | "ULDC" => map_two_operands(instruction, |dst, source| {
            KernelIrOpKind::LoadConst { dst, source }
        }),
        "LD" | "LDG" | "LDS" | "LDL" => {
            map_two_operands(instruction, |dst, address| KernelIrOpKind::Load {
                dst,
                space: memory_space(op, &address),
                address,
            })
        }
        "ST" | "STG" | "STS" | "STL" => {
            map_two_operands(instruction, |address, value| KernelIrOpKind::Store {
                space: memory_space(op, &address),
                address,
                value,
            })
        }
        "IADD" | "IADD3" | "UIADD3" => (
            KernelIrOpKind::IntegerAdd {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
                width_bits: width_modifier(&instruction.modifiers),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "FADD" => map_three_operands(instruction, |dst, lhs, rhs| KernelIrOpKind::FloatAdd {
            dst,
            lhs,
            rhs,
        }),
        "FMUL" => map_three_operands(instruction, |dst, lhs, rhs| KernelIrOpKind::FloatMul {
            dst,
            lhs,
            rhs,
        }),
        "HADD2" => (
            KernelIrOpKind::PackedHalfAdd {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
                lanes: 2,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "HMUL2" => (
            KernelIrOpKind::PackedHalfMul {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
                lanes: 2,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "FFMA" | "HFMA2" => map_four_operands(instruction, |dst, a, b, c| {
            KernelIrOpKind::FusedMultiplyAdd {
                dst,
                a,
                b,
                c,
                lane_bits: (op == "HFMA2").then_some(16),
            }
        }),
        "IMAD" | "UIMAD" => {
            map_four_operands(instruction, |dst, a, b, c| KernelIrOpKind::IntegerMad {
                dst,
                a,
                b,
                c,
                wide: has_modifier(&instruction.modifiers, "WIDE"),
            })
        }
        "ISETP" | "UISETP" | "FSETP" => (
            KernelIrOpKind::CompareSet {
                dst: operands.first().cloned().unwrap_or_default(),
                comparison: instruction.modifiers.first().cloned(),
                dtype: instruction
                    .modifiers
                    .iter()
                    .find(|modifier| modifier.starts_with('U') || modifier.starts_with('S'))
                    .cloned(),
                lhs: operands.get(2).cloned().unwrap_or_default(),
                rhs: operands.get(3).cloned().unwrap_or_default(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "SHF" | "USHF" => (
            KernelIrOpKind::Shift {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "LOP3" | "ULOP3" | "PLOP3" => (
            KernelIrOpKind::LogicLut {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "PRMT" => (
            KernelIrOpKind::Permute {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "LEA" | "ULEA" => (
            KernelIrOpKind::AddressCalc {
                dst: operands.first().cloned().unwrap_or_default(),
                inputs: operands.iter().skip(1).cloned().collect(),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "BSSY" | "BSYNC" | "BAR" => (
            KernelIrOpKind::Sync {
                kind: op.to_string(),
                operands,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => (
            KernelIrOpKind::Unsupported {
                opcode: instruction.opcode.clone(),
                reason: "no local mapping for opcode yet".to_string(),
            },
            SassMappingConfidence::Unsupported,
        ),
    }
}

fn map_two_operands(
    instruction: &SassInstruction,
    f: impl FnOnce(String, String) -> KernelIrOpKind,
) -> (KernelIrOpKind, SassMappingConfidence) {
    if instruction.operands.len() == 2 {
        (
            f(
                instruction.operands[0].raw.clone(),
                instruction.operands[1].raw.clone(),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(instruction, 2)
    }
}

fn map_three_operands(
    instruction: &SassInstruction,
    f: impl FnOnce(String, String, String) -> KernelIrOpKind,
) -> (KernelIrOpKind, SassMappingConfidence) {
    if instruction.operands.len() == 3 {
        (
            f(
                instruction.operands[0].raw.clone(),
                instruction.operands[1].raw.clone(),
                instruction.operands[2].raw.clone(),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(instruction, 3)
    }
}

fn map_four_operands(
    instruction: &SassInstruction,
    f: impl FnOnce(String, String, String, String) -> KernelIrOpKind,
) -> (KernelIrOpKind, SassMappingConfidence) {
    if instruction.operands.len() >= 4 {
        (
            f(
                instruction.operands[0].raw.clone(),
                instruction.operands[1].raw.clone(),
                instruction.operands[2].raw.clone(),
                instruction.operands[3].raw.clone(),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(instruction, 4)
    }
}

fn map_five_operands(
    instruction: &SassInstruction,
    f: impl FnOnce(String, String, String, String, String) -> KernelIrOpKind,
) -> (KernelIrOpKind, SassMappingConfidence) {
    if instruction.operands.len() >= 5 {
        (
            f(
                instruction.operands[0].raw.clone(),
                instruction.operands[1].raw.clone(),
                instruction.operands[2].raw.clone(),
                instruction.operands[3].raw.clone(),
                instruction.operands[4].raw.clone(),
            ),
            SassMappingConfidence::OpcodeHeuristic,
        )
    } else {
        unsupported_arity(instruction, 5)
    }
}

fn unsupported_arity(
    instruction: &SassInstruction,
    expected: usize,
) -> (KernelIrOpKind, SassMappingConfidence) {
    (
        KernelIrOpKind::Unsupported {
            opcode: instruction.opcode.clone(),
            reason: format!(
                "expected at least {expected} operands, saw {}",
                instruction.operands.len()
            ),
        },
        SassMappingConfidence::Unsupported,
    )
}

fn predicate_text(predicate: &SassPredicate) -> String {
    if predicate.negated {
        format!("!{}", predicate.register)
    } else {
        predicate.register.clone()
    }
}

fn label_operand(instruction: &SassInstruction) -> Option<String> {
    instruction
        .operands
        .iter()
        .find_map(|operand| match &operand.kind {
            SassOperandKind::Label(label) => Some(label.clone()),
            _ => label_in_text(&operand.raw),
        })
}

fn branch_condition_operand(instruction: &SassInstruction) -> Option<String> {
    instruction
        .operands
        .iter()
        .find(|operand| {
            matches!(
                operand.kind,
                SassOperandKind::Register(SassRegister {
                    class: RegisterClass::Predicate
                        | RegisterClass::UniformPredicate
                        | RegisterClass::PredicateTrue
                        | RegisterClass::UniformPredicateTrue,
                    ..
                })
            )
        })
        .map(|operand| operand.raw.clone())
}

fn memory_space(opcode: &str, address: &str) -> MemorySpace {
    if address.starts_with("desc[") {
        return MemorySpace::Descriptor;
    }
    match opcode {
        "LDG" | "STG" | "LD" | "ST" => MemorySpace::Global,
        "LDS" | "STS" => MemorySpace::Shared,
        "LDL" | "STL" => MemorySpace::Local,
        "LDC" | "LDCU" | "ULDC" => MemorySpace::Constant,
        _ => MemorySpace::Unknown,
    }
}

fn has_modifier(modifiers: &[String], expected: &str) -> bool {
    modifiers.iter().any(|modifier| modifier == expected)
}

fn width_modifier(modifiers: &[String]) -> Option<u32> {
    modifiers.iter().find_map(|modifier| {
        modifier
            .strip_prefix('U')
            .or_else(|| modifier.strip_prefix('S'))
            .and_then(|bits| bits.parse::<u32>().ok())
            .or_else(|| modifier.parse::<u32>().ok())
    })
}
