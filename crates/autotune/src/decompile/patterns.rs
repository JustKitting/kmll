use std::fmt::Write as _;

use super::{KernelIrFunction, KernelIrModule, KernelIrOp, KernelIrOpKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassPatternModule {
    pub target: Option<String>,
    pub functions: Vec<SassPatternFunction>,
}

impl SassPatternModule {
    pub fn pattern_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.patterns.len())
            .sum()
    }

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        if let Some(target) = &self.target {
            writeln!(out, "target {target}").expect("write to string");
        }
        for function in &self.functions {
            writeln!(out, "fn {} {{", function.name).expect("write to string");
            for pattern in &function.patterns {
                writeln!(
                    out,
                    "  {:#06x}-{:#06x}: {} {:?} [{}] <- {}",
                    pattern.start_address,
                    pattern.end_address,
                    pattern.kind_name(),
                    pattern.kind,
                    pattern.confidence,
                    format_addresses(&pattern.source_addresses)
                )
                .expect("write to string");
            }
            writeln!(out, "}}").expect("write to string");
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassPatternFunction {
    pub name: String,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SassSemanticPatternKind {
    Bf16WidenBits {
        src: String,
        dst: String,
        producer: Option<String>,
        consumer: Option<String>,
    },
    F32MulAddPair {
        mul_dst: String,
        mul_lhs: String,
        mul_rhs: String,
        add_dst: String,
        add_other: String,
    },
    AddressPair {
        low_dst: String,
        high_dst: String,
        low_inputs: Vec<String>,
        high_inputs: Vec<String>,
    },
    WarpReduceSum {
        input: String,
        output: String,
        offsets: Vec<String>,
        mask: Option<String>,
    },
}

impl SassSemanticPatternKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bf16WidenBits { .. } => "bf16-widen-bits",
            Self::F32MulAddPair { .. } => "f32-mul-add-pair",
            Self::AddressPair { .. } => "address-pair",
            Self::WarpReduceSum { .. } => "warp-reduce-sum",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassPatternConfidence {
    ExactOpcodeSequence,
    HeuristicDataflow,
}

impl std::fmt::Display for SassPatternConfidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExactOpcodeSequence => f.write_str("exact-opcode-sequence"),
            Self::HeuristicDataflow => f.write_str("heuristic-dataflow"),
        }
    }
}

pub fn recover_sass_patterns(module: &KernelIrModule) -> SassPatternModule {
    SassPatternModule {
        target: module.target.clone(),
        functions: module
            .functions
            .iter()
            .map(recover_function_patterns)
            .collect(),
    }
}

fn recover_function_patterns(function: &KernelIrFunction) -> SassPatternFunction {
    let mut patterns = Vec::new();
    recover_bf16_widen_bits(function, &mut patterns);
    recover_f32_mul_add_pairs(function, &mut patterns);
    recover_address_pairs(function, &mut patterns);
    recover_warp_reduce_sum(function, &mut patterns);
    patterns.sort_by_key(|pattern| (pattern.start_address, pattern.end_address));
    SassPatternFunction {
        name: function.name.clone(),
        patterns,
    }
}

fn recover_bf16_widen_bits(function: &KernelIrFunction, patterns: &mut Vec<SassSemanticPattern>) {
    for (index, op) in function.ops.iter().enumerate() {
        let KernelIrOpKind::IntegerMad { dst, a, b, c, wide } = &op.kind else {
            continue;
        };
        if *wide
            || op.source_opcode != "IMAD"
            || !op.source_modifiers.iter().any(|modifier| modifier == "U32")
            || b != "0x10000"
            || c != "RZ"
        {
            continue;
        }

        let producer = function.ops[..index]
            .iter()
            .rev()
            .find(|candidate| candidate.defines(a))
            .filter(|candidate| {
                candidate.source_opcode == "LD"
                    && candidate
                        .source_modifiers
                        .iter()
                        .any(|modifier| modifier == "U16")
            })
            .map(|candidate| candidate.source.clone());
        let consumer = function.ops[index + 1..]
            .iter()
            .take(6)
            .find(|candidate| {
                matches!(
                    &candidate.kind,
                    KernelIrOpKind::FloatMul { lhs, rhs, .. } if lhs == dst || rhs == dst
                )
            })
            .map(|candidate| candidate.source.clone());

        patterns.push(SassSemanticPattern {
            start_address: op.address,
            end_address: op.address,
            source_addresses: vec![op.address],
            kind: SassSemanticPatternKind::Bf16WidenBits {
                src: a.clone(),
                dst: dst.clone(),
                producer,
                consumer,
            },
            confidence: SassPatternConfidence::ExactOpcodeSequence,
        });
    }
}

fn recover_f32_mul_add_pairs(function: &KernelIrFunction, patterns: &mut Vec<SassSemanticPattern>) {
    for (index, op) in function.ops.iter().enumerate() {
        let KernelIrOpKind::FloatMul { dst, lhs, rhs } = &op.kind else {
            continue;
        };
        let Some((add, add_other)) = function.ops[index + 1..]
            .iter()
            .take(4)
            .find_map(|candidate| fadd_consumes(candidate, dst))
        else {
            continue;
        };
        let KernelIrOpKind::FloatAdd { dst: add_dst, .. } = &add.kind else {
            continue;
        };
        patterns.push(SassSemanticPattern {
            start_address: op.address,
            end_address: add.address,
            source_addresses: vec![op.address, add.address],
            kind: SassSemanticPatternKind::F32MulAddPair {
                mul_dst: dst.clone(),
                mul_lhs: lhs.clone(),
                mul_rhs: rhs.clone(),
                add_dst: add_dst.clone(),
                add_other,
            },
            confidence: SassPatternConfidence::HeuristicDataflow,
        });
    }
}

fn recover_address_pairs(function: &KernelIrFunction, patterns: &mut Vec<SassSemanticPattern>) {
    for ops in function.ops.windows(2) {
        let low = &ops[0];
        let high = &ops[1];
        let KernelIrOpKind::AddressCalc {
            dst: low_dst,
            inputs: low_inputs,
        } = &low.kind
        else {
            continue;
        };
        let KernelIrOpKind::AddressCalc {
            dst: high_dst,
            inputs: high_inputs,
        } = &high.kind
        else {
            continue;
        };
        if low.source_opcode != "LEA"
            || high.source_opcode != "LEA"
            || !high
                .source_modifiers
                .iter()
                .any(|modifier| modifier == "HI")
            || !high.source_modifiers.iter().any(|modifier| modifier == "X")
        {
            continue;
        }
        patterns.push(SassSemanticPattern {
            start_address: low.address,
            end_address: high.address,
            source_addresses: vec![low.address, high.address],
            kind: SassSemanticPatternKind::AddressPair {
                low_dst: low_dst.clone(),
                high_dst: high_dst.clone(),
                low_inputs: low_inputs.clone(),
                high_inputs: high_inputs.clone(),
            },
            confidence: SassPatternConfidence::ExactOpcodeSequence,
        });
    }
}

fn recover_warp_reduce_sum(function: &KernelIrFunction, patterns: &mut Vec<SassSemanticPattern>) {
    let shuffles = function
        .ops
        .iter()
        .enumerate()
        .filter_map(|(index, op)| match &op.kind {
            KernelIrOpKind::WarpShuffle {
                mode: Some(mode),
                offset,
                ..
            } if mode == "DOWN" => Some((index, op, offset.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    if shuffles.len() < 2 {
        return;
    }

    let pairs = shuffles
        .iter()
        .filter_map(|(index, shuffle, offset)| {
            let defined = shuffle.defined_register()?;
            let add = function.ops[*index + 1..]
                .iter()
                .take(3)
                .find(|candidate| fadd_consumes(candidate, defined).is_some())?;
            Some((*shuffle, add, offset.clone()))
        })
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return;
    };
    let first_shuffle = pairs[0].0;
    let last_add = pairs[pairs.len() - 1].1;

    let KernelIrOpKind::WarpShuffle {
        src,
        mask: first_mask,
        ..
    } = &first_shuffle.kind
    else {
        return;
    };
    let KernelIrOpKind::FloatAdd { dst: output, .. } = &last_add.kind else {
        return;
    };
    let masks = pairs
        .iter()
        .filter_map(|(op, _, _)| match &op.kind {
            KernelIrOpKind::WarpShuffle { mask, .. } => Some(mask.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mask = masks
        .iter()
        .all(|mask| mask == first_mask)
        .then(|| first_mask.clone());
    let offsets = pairs
        .iter()
        .map(|(_, _, offset)| offset.clone())
        .collect::<Vec<_>>();
    let source_addresses = pairs
        .iter()
        .flat_map(|(shuffle, add, _)| [shuffle.address, add.address])
        .collect::<Vec<_>>();

    patterns.push(SassSemanticPattern {
        start_address: first_shuffle.address,
        end_address: last_add.address,
        source_addresses,
        kind: SassSemanticPatternKind::WarpReduceSum {
            input: src.clone(),
            output: output.clone(),
            offsets,
            mask,
        },
        confidence: SassPatternConfidence::HeuristicDataflow,
    });
}

fn format_addresses(addresses: &[u64]) -> String {
    let body = addresses
        .iter()
        .map(|address| format!("{address:#06x}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

fn fadd_consumes<'a>(op: &'a KernelIrOp, value: &str) -> Option<(&'a KernelIrOp, String)> {
    match &op.kind {
        KernelIrOpKind::FloatAdd { lhs, rhs, .. } if lhs == value => Some((op, rhs.clone())),
        KernelIrOpKind::FloatAdd { lhs, rhs, .. } if rhs == value => Some((op, lhs.clone())),
        _ => None,
    }
}

trait KernelIrOpDef {
    fn defines(&self, register: &str) -> bool;
    fn defined_register(&self) -> Option<&str>;
}

impl KernelIrOpDef for KernelIrOp {
    fn defines(&self, register: &str) -> bool {
        self.defined_register()
            .is_some_and(|defined| defined == register)
    }

    fn defined_register(&self) -> Option<&str> {
        match &self.kind {
            KernelIrOpKind::ReadSpecialRegister { dst, .. }
            | KernelIrOpKind::Move { dst, .. }
            | KernelIrOpKind::LoadConst { dst, .. }
            | KernelIrOpKind::Load { dst, .. }
            | KernelIrOpKind::IntegerAdd { dst, .. }
            | KernelIrOpKind::FloatAdd { dst, .. }
            | KernelIrOpKind::FloatMul { dst, .. }
            | KernelIrOpKind::PackedHalfAdd { dst, .. }
            | KernelIrOpKind::PackedHalfMul { dst, .. }
            | KernelIrOpKind::FusedMultiplyAdd { dst, .. }
            | KernelIrOpKind::IntegerMad { dst, .. }
            | KernelIrOpKind::CompareSet { dst, .. }
            | KernelIrOpKind::WarpShuffle { dst, .. }
            | KernelIrOpKind::Shift { dst, .. }
            | KernelIrOpKind::LogicLut { dst, .. }
            | KernelIrOpKind::Permute { dst, .. }
            | KernelIrOpKind::AddressCalc { dst, .. } => Some(dst),
            KernelIrOpKind::Store { .. }
            | KernelIrOpKind::TensorCoreMma { .. }
            | KernelIrOpKind::TensorCoreMemory { .. }
            | KernelIrOpKind::TensorMemoryAccess { .. }
            | KernelIrOpKind::WarpGroup { .. }
            | KernelIrOpKind::Branch { .. }
            | KernelIrOpKind::Call { .. }
            | KernelIrOpKind::Return { .. }
            | KernelIrOpKind::Exit { .. }
            | KernelIrOpKind::Sync { .. }
            | KernelIrOpKind::NoOp
            | KernelIrOpKind::Unsupported { .. } => None,
        }
    }
}
