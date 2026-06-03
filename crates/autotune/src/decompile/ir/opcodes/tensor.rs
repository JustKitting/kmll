use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassOpcode, SassTensorElementType,
    SassTensorScope,
};
use super::LiftResult;

pub(super) fn lift(opcode: &str, operands: &[AggregateOperand]) -> Option<LiftResult> {
    let operands = operands.to_vec();
    Some(match opcode {
        "BGMMA" | "BMMA" | "DMMA" | "HGMMA" | "HMMA" | "IGMMA" | "IMMA" | "OMMA" | "QGMMA"
        | "QMMA" | "UTCHMMA" | "UTCIMMA" | "UTCOMMA" | "UTCQMMA" => (
            KernelIrOpKind::TensorCoreMma {
                opcode: SassOpcode::new(opcode),
                operands,
                element_type: tensor_core_element_type(opcode),
                scope: tensor_core_scope(opcode),
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "LDT" | "LDTM" | "STT" | "STTM" => (
            KernelIrOpKind::TensorCoreMemory {
                opcode: SassOpcode::new(opcode),
                operands,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "UBLKCP" | "UBLKPF" | "UBLKRED" | "UTMALDG" | "UTMAPF" | "UTMAREDG" | "UTMASTG" => (
            KernelIrOpKind::TensorMemoryAccess {
                opcode: SassOpcode::new(opcode),
                operands,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        "WARPGROUP" | "WARPGROUPSET" => (
            KernelIrOpKind::WarpGroup {
                opcode: SassOpcode::new(opcode),
                operands,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}

fn tensor_core_element_type(opcode: &str) -> Option<SassTensorElementType> {
    match opcode {
        "BMMA" | "BGMMA" => Some(SassTensorElementType::Bit),
        "DMMA" => Some(SassTensorElementType::Fp64),
        "HGMMA" | "HMMA" | "UTCHMMA" => Some(SassTensorElementType::Half),
        "IGMMA" | "IMMA" | "UTCIMMA" => Some(SassTensorElementType::Integer),
        "OMMA" | "UTCOMMA" => Some(SassTensorElementType::Fp4),
        "QGMMA" | "QMMA" | "UTCQMMA" => Some(SassTensorElementType::Fp8),
        _ => None,
    }
}

fn tensor_core_scope(opcode: &str) -> Option<SassTensorScope> {
    match opcode {
        "BGMMA" | "HGMMA" | "IGMMA" | "QGMMA" => Some(SassTensorScope::WarpGroup),
        "UTCHMMA" | "UTCIMMA" | "UTCOMMA" | "UTCQMMA" => Some(SassTensorScope::Uniform),
        "BMMA" | "DMMA" | "HMMA" | "IMMA" | "OMMA" | "QMMA" => Some(SassTensorScope::Warp),
        _ => None,
    }
}
