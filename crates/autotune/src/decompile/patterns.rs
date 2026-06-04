use std::{
    collections::BTreeSet,
    fmt::{self, Write as _},
};

use super::{
    ImmediateValue, KernelIrFunction, KernelIrModule, KernelIrOp, KernelIrOpKind, RegisterRef,
    RegisterRefKind, SassModifierKind, SassOpcode, SassOpcodeKind, SassSymbol, SassTarget,
    SassWarpShuffleMode, ScalarOperand, ScalarOperandKind,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassPatternModule {
    pub target: Option<SassTarget>,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassPatternLinkedOp {
    pub address: u64,
    pub opcode: SassOpcode,
}

impl SassPatternLinkedOp {
    fn new(op: &KernelIrOp) -> Self {
        Self {
            address: op.address,
            opcode: op.source_opcode.clone(),
        }
    }
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
            || op.source_opcode.kind() != &SassOpcodeKind::Imad
            || !has_source_modifier(op, &SassModifierKind::UnsignedWidth(32))
            || !scalar_integer_eq(b, 0x10000)
            || !scalar_is_zero_register(c)
        {
            continue;
        }
        let Some(src_register) = a.as_register() else {
            continue;
        };

        let producer = function.ops[..index]
            .iter()
            .rev()
            .find(|candidate| candidate.defines(src_register))
            .filter(|candidate| {
                candidate.source_opcode.kind() == &SassOpcodeKind::Ld
                    && has_source_modifier(candidate, &SassModifierKind::UnsignedWidth(16))
            })
            .map(SassPatternLinkedOp::new);
        let consumer = function.ops[index + 1..]
            .iter()
            .take(6)
            .find(|candidate| {
                matches!(
                    &candidate.kind,
                    KernelIrOpKind::FloatMul { lhs, rhs, .. }
                        if scalar_register_eq(lhs, dst) || scalar_register_eq(rhs, dst)
                )
            })
            .map(SassPatternLinkedOp::new);
        let source_addresses =
            linked_source_addresses(op.address, producer.as_ref(), consumer.as_ref());
        let start_address = source_addresses.first().copied().unwrap_or(op.address);
        let end_address = source_addresses.last().copied().unwrap_or(op.address);

        patterns.push(SassSemanticPattern {
            start_address,
            end_address,
            source_addresses,
            kind: SassSemanticPatternKind::Bf16WidenBits {
                src: src_register.clone(),
                dst: dst.clone(),
                producer,
                consumer,
            },
            confidence: SassPatternConfidence::ExactOpcodeSequence,
        });
    }
}

fn linked_source_addresses(
    current: u64,
    producer: Option<&SassPatternLinkedOp>,
    consumer: Option<&SassPatternLinkedOp>,
) -> Vec<u64> {
    let mut addresses = BTreeSet::from([current]);
    if let Some(producer) = producer {
        addresses.insert(producer.address);
    }
    if let Some(consumer) = consumer {
        addresses.insert(consumer.address);
    }
    addresses.into_iter().collect()
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
    let mut used_high_indices = BTreeSet::new();
    for (low_index, low) in function.ops.iter().enumerate() {
        let KernelIrOpKind::AddressCalc {
            dst: low_dst,
            inputs: low_inputs,
        } = &low.kind
        else {
            continue;
        };
        if low.source_opcode.kind() != &SassOpcodeKind::Lea || is_lea_high_x(low) {
            continue;
        }
        let Some((high_index, high, high_dst, high_inputs)) = function.ops[low_index + 1..]
            .iter()
            .enumerate()
            .take_while(|(_, candidate)| !stops_address_pair_scan(candidate, low_dst))
            .filter_map(|(relative_index, high)| {
                let high_index = low_index + 1 + relative_index;
                if used_high_indices.contains(&high_index) || !is_lea_high_x(high) {
                    return None;
                }
                let KernelIrOpKind::AddressCalc {
                    dst: high_dst,
                    inputs: high_inputs,
                } = &high.kind
                else {
                    return None;
                };
                (is_next_register(low_dst, high_dst) && lea_carry_matches(low_inputs, high_inputs))
                    .then_some((high_index, high, high_dst, high_inputs))
            })
            .next()
        else {
            continue;
        };
        used_high_indices.insert(high_index);
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

fn is_lea_high_x(op: &KernelIrOp) -> bool {
    op.source_opcode.kind() == &SassOpcodeKind::Lea
        && has_source_modifier(op, &SassModifierKind::High)
        && has_source_modifier(op, &SassModifierKind::Carry)
}

fn has_source_modifier(op: &KernelIrOp, kind: &SassModifierKind) -> bool {
    op.source_modifiers
        .iter()
        .any(|modifier| modifier.kind() == kind)
}

fn scalar_integer_eq(operand: &ScalarOperand, expected: i128) -> bool {
    matches!(
        operand.kind,
        ScalarOperandKind::Immediate(ImmediateValue::Integer(value)) if value == expected
    )
}

fn scalar_is_zero_register(operand: &ScalarOperand) -> bool {
    matches!(
        operand.as_register().map(|register| &register.kind),
        Some(RegisterRefKind::GeneralZero | RegisterRefKind::UniformZero)
    )
}

fn scalar_register_eq(operand: &ScalarOperand, register: &RegisterRef) -> bool {
    operand
        .as_register()
        .is_some_and(|operand| operand == register)
}

fn stops_address_pair_scan(op: &KernelIrOp, low_dst: &RegisterRef) -> bool {
    matches!(
        &op.kind,
        KernelIrOpKind::Branch { .. }
            | KernelIrOpKind::Call { .. }
            | KernelIrOpKind::Return { .. }
            | KernelIrOpKind::Exit { .. }
    ) || op.defines(low_dst)
}

fn lea_carry_matches(low_inputs: &[ScalarOperand], high_inputs: &[ScalarOperand]) -> bool {
    let Some(low_carry) = low_inputs
        .first()
        .filter(|input| is_predicate_operand(input))
    else {
        return true;
    };
    high_inputs.last() == Some(low_carry)
}

fn is_predicate_operand(input: &ScalarOperand) -> bool {
    matches!(
        input.as_register().map(|register| &register.kind),
        Some(
            RegisterRefKind::Predicate(_)
                | RegisterRefKind::UniformPredicate(_)
                | RegisterRefKind::PredicateTrue
                | RegisterRefKind::UniformPredicateTrue
        )
    )
}

fn is_next_register(low: &RegisterRef, high: &RegisterRef) -> bool {
    let Some((low_class, low_index)) = numbered_register(low) else {
        return false;
    };
    let Some((high_class, high_index)) = numbered_register(high) else {
        return false;
    };
    low_class == high_class && high_index == low_index + 1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberedRegisterClass {
    General,
    Uniform,
}

fn numbered_register(register: &RegisterRef) -> Option<(NumberedRegisterClass, u16)> {
    match register.kind {
        RegisterRefKind::General(index) => Some((NumberedRegisterClass::General, index)),
        RegisterRefKind::Uniform(index) => Some((NumberedRegisterClass::Uniform, index)),
        _ => None,
    }
}

fn recover_warp_reduce_sum(function: &KernelIrFunction, patterns: &mut Vec<SassSemanticPattern>) {
    let shuffles = function
        .ops
        .iter()
        .enumerate()
        .filter_map(|(index, op)| match &op.kind {
            KernelIrOpKind::WarpShuffle {
                mode: Some(SassWarpShuffleMode::Down),
                offset,
                ..
            } => Some((index, op, offset.clone())),
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

fn display_vec<T: fmt::Display>(values: &[T]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

fn fadd_consumes<'a>(
    op: &'a KernelIrOp,
    value: &RegisterRef,
) -> Option<(&'a KernelIrOp, ScalarOperand)> {
    match &op.kind {
        KernelIrOpKind::FloatAdd { lhs, rhs, .. } if scalar_register_eq(lhs, value) => {
            Some((op, rhs.clone()))
        }
        KernelIrOpKind::FloatAdd { lhs, rhs, .. } if scalar_register_eq(rhs, value) => {
            Some((op, lhs.clone()))
        }
        _ => None,
    }
}

trait KernelIrOpDef {
    fn defines(&self, register: &RegisterRef) -> bool;
    fn defined_register(&self) -> Option<&RegisterRef>;
}

impl KernelIrOpDef for KernelIrOp {
    fn defines(&self, register: &RegisterRef) -> bool {
        self.defined_register()
            .is_some_and(|defined| defined == register)
    }

    fn defined_register(&self) -> Option<&RegisterRef> {
        match &self.kind {
            KernelIrOpKind::ReadSpecialRegister { dst, .. }
            | KernelIrOpKind::Move { dst, .. }
            | KernelIrOpKind::LoadConst { dst, .. }
            | KernelIrOpKind::Load { dst, .. }
            | KernelIrOpKind::MemoryAtomic { dst, .. }
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
            | KernelIrOpKind::MemoryReduction { .. }
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
