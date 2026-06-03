use super::{
    super::KernelIrOpKind,
    types::{SassLiftedOpClass, SassLiftedOpKind},
};

pub(super) fn classify_op(kind: &KernelIrOpKind) -> (SassLiftedOpClass, SassLiftedOpKind) {
    match kind {
        KernelIrOpKind::ReadSpecialRegister { .. } => (
            SassLiftedOpClass::DataMovement,
            SassLiftedOpKind::SpecialRead,
        ),
        KernelIrOpKind::Move { .. } => (SassLiftedOpClass::DataMovement, SassLiftedOpKind::Move),
        KernelIrOpKind::LoadConst { .. } => {
            (SassLiftedOpClass::Memory, SassLiftedOpKind::LoadConst)
        }
        KernelIrOpKind::Load { .. } => (SassLiftedOpClass::Memory, SassLiftedOpKind::Load),
        KernelIrOpKind::Store { .. } => (SassLiftedOpClass::Memory, SassLiftedOpKind::Store),
        KernelIrOpKind::IntegerAdd { .. } => {
            (SassLiftedOpClass::IntegerMath, SassLiftedOpKind::IntegerAdd)
        }
        KernelIrOpKind::FloatAdd { .. } => {
            (SassLiftedOpClass::FloatMath, SassLiftedOpKind::FloatAdd)
        }
        KernelIrOpKind::FloatMul { .. } => {
            (SassLiftedOpClass::FloatMath, SassLiftedOpKind::FloatMul)
        }
        KernelIrOpKind::PackedHalfAdd { .. } => (
            SassLiftedOpClass::FloatMath,
            SassLiftedOpKind::PackedHalfAdd,
        ),
        KernelIrOpKind::PackedHalfMul { .. } => (
            SassLiftedOpClass::FloatMath,
            SassLiftedOpKind::PackedHalfMul,
        ),
        KernelIrOpKind::FusedMultiplyAdd { .. } => (
            SassLiftedOpClass::FloatMath,
            SassLiftedOpKind::FusedMultiplyAdd,
        ),
        KernelIrOpKind::IntegerMad { .. } => {
            (SassLiftedOpClass::IntegerMath, SassLiftedOpKind::IntegerMad)
        }
        KernelIrOpKind::TensorCoreMma { .. } => (
            SassLiftedOpClass::TensorCore,
            SassLiftedOpKind::TensorCoreMma,
        ),
        KernelIrOpKind::TensorCoreMemory { .. } => (
            SassLiftedOpClass::TensorMemory,
            SassLiftedOpKind::TensorCoreMemory,
        ),
        KernelIrOpKind::TensorMemoryAccess { .. } => (
            SassLiftedOpClass::TensorMemory,
            SassLiftedOpKind::TensorMemoryAccess,
        ),
        KernelIrOpKind::WarpGroup { .. } => {
            (SassLiftedOpClass::WarpGroup, SassLiftedOpKind::WarpGroup)
        }
        KernelIrOpKind::CompareSet { .. } => {
            (SassLiftedOpClass::Predicate, SassLiftedOpKind::CompareSet)
        }
        KernelIrOpKind::Branch { .. } => (SassLiftedOpClass::ControlFlow, SassLiftedOpKind::Branch),
        KernelIrOpKind::Call { .. } => (SassLiftedOpClass::ControlFlow, SassLiftedOpKind::Call),
        KernelIrOpKind::Return { .. } => (SassLiftedOpClass::ControlFlow, SassLiftedOpKind::Return),
        KernelIrOpKind::Exit { .. } => (SassLiftedOpClass::ControlFlow, SassLiftedOpKind::Exit),
        KernelIrOpKind::WarpShuffle { .. } => {
            (SassLiftedOpClass::Warp, SassLiftedOpKind::WarpShuffle)
        }
        KernelIrOpKind::Shift { .. } => (SassLiftedOpClass::IntegerMath, SassLiftedOpKind::Shift),
        KernelIrOpKind::LogicLut { .. } => {
            (SassLiftedOpClass::IntegerMath, SassLiftedOpKind::LogicLut)
        }
        KernelIrOpKind::Permute { .. } => {
            (SassLiftedOpClass::DataMovement, SassLiftedOpKind::Permute)
        }
        KernelIrOpKind::AddressCalc { .. } => {
            (SassLiftedOpClass::Address, SassLiftedOpKind::AddressCalc)
        }
        KernelIrOpKind::Sync { .. } => (SassLiftedOpClass::Synchronization, SassLiftedOpKind::Sync),
        KernelIrOpKind::NoOp => (SassLiftedOpClass::NoOp, SassLiftedOpKind::NoOp),
        KernelIrOpKind::Unsupported { .. } => (
            SassLiftedOpClass::Unsupported,
            SassLiftedOpKind::Unsupported,
        ),
    }
}
