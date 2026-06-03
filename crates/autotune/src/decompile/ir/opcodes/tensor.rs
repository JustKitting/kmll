use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassOpcode, SassOpcodeKind,
    SassTensorElementType, SassTensorScope,
};
use super::LiftResult;

pub(super) fn lift(opcode: &SassOpcode, operands: &[AggregateOperand]) -> Option<LiftResult> {
    let operands = operands.to_vec();
    Some(match opcode.kind() {
        SassOpcodeKind::Bgmma
        | SassOpcodeKind::Bmma
        | SassOpcodeKind::Dmma
        | SassOpcodeKind::Hgmma
        | SassOpcodeKind::Hmma
        | SassOpcodeKind::Igmma
        | SassOpcodeKind::Imma
        | SassOpcodeKind::Omma
        | SassOpcodeKind::Qgmma
        | SassOpcodeKind::Qmma
        | SassOpcodeKind::Utchmma
        | SassOpcodeKind::Utcimma
        | SassOpcodeKind::Utcomma
        | SassOpcodeKind::Utcqmma => (
            KernelIrOpKind::TensorCoreMma {
                opcode: opcode.clone(),
                operands,
                element_type: tensor_core_element_type(opcode.kind()),
                scope: tensor_core_scope(opcode.kind()),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Ldt | SassOpcodeKind::Ldtm | SassOpcodeKind::Stt | SassOpcodeKind::Sttm => {
            (
                KernelIrOpKind::TensorCoreMemory {
                    opcode: opcode.clone(),
                    operands,
                },
                SassMappingConfidence::OpcodeHeuristic,
            )
        }
        SassOpcodeKind::Ublkcp
        | SassOpcodeKind::Ublkpf
        | SassOpcodeKind::Ublkred
        | SassOpcodeKind::Utmaldg
        | SassOpcodeKind::Utmapf
        | SassOpcodeKind::Utmaredg
        | SassOpcodeKind::Utmastg => (
            KernelIrOpKind::TensorMemoryAccess {
                opcode: opcode.clone(),
                operands,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        SassOpcodeKind::Warpgroup | SassOpcodeKind::Warpgroupset => (
            KernelIrOpKind::WarpGroup {
                opcode: opcode.clone(),
                operands,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}

fn tensor_core_element_type(opcode: &SassOpcodeKind) -> Option<SassTensorElementType> {
    match opcode {
        SassOpcodeKind::Bmma | SassOpcodeKind::Bgmma => Some(SassTensorElementType::Bit),
        SassOpcodeKind::Dmma => Some(SassTensorElementType::Fp64),
        SassOpcodeKind::Hgmma | SassOpcodeKind::Hmma | SassOpcodeKind::Utchmma => {
            Some(SassTensorElementType::Half)
        }
        SassOpcodeKind::Igmma | SassOpcodeKind::Imma | SassOpcodeKind::Utcimma => {
            Some(SassTensorElementType::Integer)
        }
        SassOpcodeKind::Omma | SassOpcodeKind::Utcomma => Some(SassTensorElementType::Fp4),
        SassOpcodeKind::Qgmma | SassOpcodeKind::Qmma | SassOpcodeKind::Utcqmma => {
            Some(SassTensorElementType::Fp8)
        }
        _ => None,
    }
}

fn tensor_core_scope(opcode: &SassOpcodeKind) -> Option<SassTensorScope> {
    match opcode {
        SassOpcodeKind::Bgmma
        | SassOpcodeKind::Hgmma
        | SassOpcodeKind::Igmma
        | SassOpcodeKind::Qgmma => Some(SassTensorScope::WarpGroup),
        SassOpcodeKind::Utchmma
        | SassOpcodeKind::Utcimma
        | SassOpcodeKind::Utcomma
        | SassOpcodeKind::Utcqmma => Some(SassTensorScope::Uniform),
        SassOpcodeKind::Bmma
        | SassOpcodeKind::Dmma
        | SassOpcodeKind::Hmma
        | SassOpcodeKind::Imma
        | SassOpcodeKind::Omma
        | SassOpcodeKind::Qmma => Some(SassTensorScope::Warp),
        _ => None,
    }
}
