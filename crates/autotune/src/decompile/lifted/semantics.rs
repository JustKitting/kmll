use std::fmt;

use super::super::{
    AggregateOperand, ControlTarget, KernelIrOpKind, MemoryAddress, MemorySpace,
    PredicateCondition, RegisterRef, SassOpcode, ScalarOperand,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SassLiftedSemantics {
    SpecialRead {
        dst: RegisterRef,
        special: RegisterRef,
    },
    Move {
        dst: RegisterRef,
        src: ScalarOperand,
    },
    LoadConst {
        dst: RegisterRef,
        source: MemoryAddress,
    },
    Load {
        space: MemorySpace,
        dst: RegisterRef,
        address: MemoryAddress,
        width_bits: Option<u32>,
        modifiers: Vec<String>,
    },
    Store {
        space: MemorySpace,
        address: MemoryAddress,
        value: RegisterRef,
        width_bits: Option<u32>,
        modifiers: Vec<String>,
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
        element_type: Option<String>,
        scope: Option<String>,
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
        comparison: Option<String>,
        dtype: Option<String>,
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
        mode: Option<String>,
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
        kind: String,
        operands: Vec<AggregateOperand>,
    },
    NoOp,
    Unsupported {
        opcode: SassOpcode,
        reason: String,
    },
}

impl fmt::Display for SassLiftedSemantics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpecialRead { dst, special } => {
                write!(f, "special-read(dst={dst},special={special})")
            }
            Self::Move { dst, src } => write!(f, "move(dst={dst},src={src})"),
            Self::LoadConst { dst, source } => {
                write!(f, "load-const(dst={dst},source={source})")
            }
            Self::Load {
                space,
                dst,
                address,
                width_bits,
                modifiers,
            } => write!(
                f,
                "load(space={space},dst={dst},address={address},width={},modifiers=[{}])",
                option_u32(*width_bits),
                modifiers.join(",")
            ),
            Self::Store {
                space,
                address,
                value,
                width_bits,
                modifiers,
            } => write!(
                f,
                "store(space={space},address={address},value={value},width={},modifiers=[{}])",
                option_u32(*width_bits),
                modifiers.join(",")
            ),
            Self::IntegerAdd {
                dst,
                inputs,
                width_bits,
            } => write!(
                f,
                "integer-add(dst={dst},inputs=[{}],width={})",
                format_display_list(inputs),
                option_u32(*width_bits)
            ),
            Self::FloatAdd { dst, lhs, rhs } => {
                write!(f, "float-add(dst={dst},lhs={lhs},rhs={rhs})")
            }
            Self::FloatMul { dst, lhs, rhs } => {
                write!(f, "float-mul(dst={dst},lhs={lhs},rhs={rhs})")
            }
            Self::PackedHalfAdd { dst, inputs, lanes } => write!(
                f,
                "packed-half-add(dst={dst},inputs=[{}],lanes={lanes})",
                format_display_list(inputs)
            ),
            Self::PackedHalfMul { dst, inputs, lanes } => write!(
                f,
                "packed-half-mul(dst={dst},inputs=[{}],lanes={lanes})",
                format_display_list(inputs)
            ),
            Self::FusedMultiplyAdd {
                dst,
                a,
                b,
                c,
                lane_bits,
            } => write!(
                f,
                "fused-multiply-add(dst={dst},a={a},b={b},c={c},lane-bits={})",
                option_u32(*lane_bits)
            ),
            Self::IntegerMad { dst, a, b, c, wide } => {
                write!(f, "integer-mad(dst={dst},a={a},b={b},c={c},wide={wide})")
            }
            Self::TensorCoreMma {
                opcode,
                operands,
                element_type,
                scope,
            } => write!(
                f,
                "tensor-core-mma(opcode={opcode},element-type={},scope={},operands=[{}])",
                option_str(element_type.as_deref()),
                option_str(scope.as_deref()),
                format_display_list(operands)
            ),
            Self::TensorCoreMemory { opcode, operands } => write!(
                f,
                "tensor-core-memory(opcode={opcode},operands=[{}])",
                format_display_list(operands)
            ),
            Self::TensorMemoryAccess { opcode, operands } => write!(
                f,
                "tensor-memory-access(opcode={opcode},operands=[{}])",
                format_display_list(operands)
            ),
            Self::WarpGroup { opcode, operands } => {
                write!(
                    f,
                    "warpgroup(opcode={opcode},operands=[{}])",
                    format_display_list(operands)
                )
            }
            Self::CompareSet {
                dst,
                comparison,
                dtype,
                lhs,
                rhs,
            } => write!(
                f,
                "compare-set(dst={dst},comparison={},dtype={},lhs={lhs},rhs={rhs})",
                option_str(comparison.as_deref()),
                option_str(dtype.as_deref())
            ),
            Self::Branch { target, condition } => write!(
                f,
                "branch(target={},condition={})",
                option_display(target.as_ref()),
                option_display(condition.as_ref())
            ),
            Self::Call { target, operands } => write!(
                f,
                "call(target={},operands=[{}])",
                option_display(target.as_ref()),
                format_display_list(operands)
            ),
            Self::Return { target, operands } => write!(
                f,
                "return(target={},operands=[{}])",
                option_display(target.as_ref()),
                format_display_list(operands)
            ),
            Self::Exit { condition } => {
                write!(f, "exit(condition={})", option_display(condition.as_ref()))
            }
            Self::WarpShuffle {
                mode,
                predicate,
                dst,
                src,
                offset,
                mask,
            } => write!(
                f,
                "warp-shuffle(mode={},predicate={predicate},dst={dst},src={src},offset={offset},mask={mask})",
                option_str(mode.as_deref())
            ),
            Self::Shift { dst, inputs } => {
                write!(
                    f,
                    "shift(dst={dst},inputs=[{}])",
                    format_display_list(inputs)
                )
            }
            Self::LogicLut { dst, inputs } => {
                write!(
                    f,
                    "logic-lut(dst={dst},inputs=[{}])",
                    format_display_list(inputs)
                )
            }
            Self::Permute { dst, inputs } => {
                write!(
                    f,
                    "permute(dst={dst},inputs=[{}])",
                    format_display_list(inputs)
                )
            }
            Self::AddressCalc { dst, inputs } => {
                write!(
                    f,
                    "address-calc(dst={dst},inputs=[{}])",
                    format_display_list(inputs)
                )
            }
            Self::Sync { kind, operands } => {
                write!(
                    f,
                    "sync(kind={kind},operands=[{}])",
                    format_display_list(operands)
                )
            }
            Self::NoOp => f.write_str("no-op"),
            Self::Unsupported { opcode, reason } => {
                write!(f, "unsupported(opcode={opcode},reason={reason})")
            }
        }
    }
}

pub(super) fn lift_semantics(kind: &KernelIrOpKind) -> SassLiftedSemantics {
    match kind {
        KernelIrOpKind::ReadSpecialRegister { dst, special } => SassLiftedSemantics::SpecialRead {
            dst: dst.clone(),
            special: special.clone(),
        },
        KernelIrOpKind::Move { dst, src } => SassLiftedSemantics::Move {
            dst: dst.clone(),
            src: src.clone(),
        },
        KernelIrOpKind::LoadConst { dst, source } => SassLiftedSemantics::LoadConst {
            dst: dst.clone(),
            source: source.clone(),
        },
        KernelIrOpKind::Load {
            dst,
            address,
            space,
            access,
        } => SassLiftedSemantics::Load {
            space: *space,
            dst: dst.clone(),
            address: address.clone(),
            width_bits: access.width_bits,
            modifiers: access.modifiers.clone(),
        },
        KernelIrOpKind::Store {
            address,
            value,
            space,
            access,
        } => SassLiftedSemantics::Store {
            space: *space,
            address: address.clone(),
            value: value.clone(),
            width_bits: access.width_bits,
            modifiers: access.modifiers.clone(),
        },
        KernelIrOpKind::IntegerAdd {
            dst,
            inputs,
            width_bits,
        } => SassLiftedSemantics::IntegerAdd {
            dst: dst.clone(),
            inputs: inputs.clone(),
            width_bits: *width_bits,
        },
        KernelIrOpKind::FloatAdd { dst, lhs, rhs } => SassLiftedSemantics::FloatAdd {
            dst: dst.clone(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        },
        KernelIrOpKind::FloatMul { dst, lhs, rhs } => SassLiftedSemantics::FloatMul {
            dst: dst.clone(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        },
        KernelIrOpKind::PackedHalfAdd { dst, inputs, lanes } => {
            SassLiftedSemantics::PackedHalfAdd {
                dst: dst.clone(),
                inputs: inputs.clone(),
                lanes: *lanes,
            }
        }
        KernelIrOpKind::PackedHalfMul { dst, inputs, lanes } => {
            SassLiftedSemantics::PackedHalfMul {
                dst: dst.clone(),
                inputs: inputs.clone(),
                lanes: *lanes,
            }
        }
        KernelIrOpKind::FusedMultiplyAdd {
            dst,
            a,
            b,
            c,
            lane_bits,
        } => SassLiftedSemantics::FusedMultiplyAdd {
            dst: dst.clone(),
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
            lane_bits: *lane_bits,
        },
        KernelIrOpKind::IntegerMad { dst, a, b, c, wide } => SassLiftedSemantics::IntegerMad {
            dst: dst.clone(),
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
            wide: *wide,
        },
        KernelIrOpKind::TensorCoreMma {
            opcode,
            operands,
            element_type,
            scope,
        } => SassLiftedSemantics::TensorCoreMma {
            opcode: SassOpcode::new(opcode.clone()),
            operands: operands.clone(),
            element_type: element_type.clone(),
            scope: scope.clone(),
        },
        KernelIrOpKind::TensorCoreMemory { opcode, operands } => {
            SassLiftedSemantics::TensorCoreMemory {
                opcode: SassOpcode::new(opcode.clone()),
                operands: operands.clone(),
            }
        }
        KernelIrOpKind::TensorMemoryAccess { opcode, operands } => {
            SassLiftedSemantics::TensorMemoryAccess {
                opcode: SassOpcode::new(opcode.clone()),
                operands: operands.clone(),
            }
        }
        KernelIrOpKind::WarpGroup { opcode, operands } => SassLiftedSemantics::WarpGroup {
            opcode: SassOpcode::new(opcode.clone()),
            operands: operands.clone(),
        },
        KernelIrOpKind::CompareSet {
            dst,
            comparison,
            dtype,
            lhs,
            rhs,
        } => SassLiftedSemantics::CompareSet {
            dst: dst.clone(),
            comparison: comparison.clone(),
            dtype: dtype.clone(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        },
        KernelIrOpKind::Branch { target, condition } => SassLiftedSemantics::Branch {
            target: target.clone(),
            condition: condition.clone(),
        },
        KernelIrOpKind::Call { target, operands } => SassLiftedSemantics::Call {
            target: target.clone(),
            operands: operands.clone(),
        },
        KernelIrOpKind::Return { target, operands } => SassLiftedSemantics::Return {
            target: target.clone(),
            operands: operands.clone(),
        },
        KernelIrOpKind::Exit { condition } => SassLiftedSemantics::Exit {
            condition: condition.clone(),
        },
        KernelIrOpKind::WarpShuffle {
            mode,
            predicate,
            dst,
            src,
            offset,
            mask,
        } => SassLiftedSemantics::WarpShuffle {
            mode: mode.clone(),
            predicate: predicate.clone(),
            dst: dst.clone(),
            src: src.clone(),
            offset: offset.clone(),
            mask: mask.clone(),
        },
        KernelIrOpKind::Shift { dst, inputs } => SassLiftedSemantics::Shift {
            dst: dst.clone(),
            inputs: inputs.clone(),
        },
        KernelIrOpKind::LogicLut { dst, inputs } => SassLiftedSemantics::LogicLut {
            dst: dst.clone(),
            inputs: inputs.clone(),
        },
        KernelIrOpKind::Permute { dst, inputs } => SassLiftedSemantics::Permute {
            dst: dst.clone(),
            inputs: inputs.clone(),
        },
        KernelIrOpKind::AddressCalc { dst, inputs } => SassLiftedSemantics::AddressCalc {
            dst: dst.clone(),
            inputs: inputs.clone(),
        },
        KernelIrOpKind::Sync { kind, operands } => SassLiftedSemantics::Sync {
            kind: kind.clone(),
            operands: operands.clone(),
        },
        KernelIrOpKind::NoOp => SassLiftedSemantics::NoOp,
        KernelIrOpKind::Unsupported { opcode, reason } => SassLiftedSemantics::Unsupported {
            opcode: SassOpcode::new(opcode.clone()),
            reason: reason.clone(),
        },
    }
}

fn option_str(value: Option<&str>) -> &str {
    value.unwrap_or("-")
}

fn option_display(value: Option<&impl fmt::Display>) -> String {
    value
        .map(ToString::to_string)
        .unwrap_or_else(|| "-".to_string())
}

fn option_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn format_display_list<T: fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}
