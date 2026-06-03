use super::super::types::{AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassOpcode};
use super::LiftResult;

pub(super) fn lift(opcode: &str, operands: &[AggregateOperand]) -> Option<LiftResult> {
    let operands = operands.to_vec();
    Some(match opcode {
        "BGMMA" | "BMMA" | "DMMA" | "HGMMA" | "HMMA" | "IGMMA" | "IMMA" | "OMMA" | "QGMMA"
        | "QMMA" | "UTCHMMA" | "UTCIMMA" | "UTCOMMA" | "UTCQMMA" => (
            KernelIrOpKind::TensorCoreMma {
                opcode: SassOpcode::new(opcode),
                operands,
                element_type: tensor_core_element_type(opcode).map(str::to_string),
                scope: tensor_core_scope(opcode).map(str::to_string),
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

fn tensor_core_element_type(opcode: &str) -> Option<&'static str> {
    match opcode {
        "BMMA" | "BGMMA" => Some("bit"),
        "DMMA" => Some("fp64"),
        "HGMMA" | "HMMA" | "UTCHMMA" => Some("half"),
        "IGMMA" | "IMMA" | "UTCIMMA" => Some("integer"),
        "OMMA" | "UTCOMMA" => Some("fp4"),
        "QGMMA" | "QMMA" | "UTCQMMA" => Some("fp8"),
        _ => None,
    }
}

fn tensor_core_scope(opcode: &str) -> Option<&'static str> {
    match opcode {
        "BGMMA" | "HGMMA" | "IGMMA" | "QGMMA" => Some("warpgroup"),
        "UTCHMMA" | "UTCIMMA" | "UTCOMMA" | "UTCQMMA" => Some("uniform"),
        "BMMA" | "DMMA" | "HMMA" | "IMMA" | "OMMA" | "QMMA" => Some("warp"),
        _ => None,
    }
}
