use super::super::super::sass::SassInstruction;
use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassModifier, SassModifierKind,
    SassOpcode, SassOpcodeKind, SassTensorElementType, SassTensorScope,
};
use super::LiftResult;

pub(super) fn lift(
    opcode: &SassOpcode,
    instruction: &SassInstruction,
    operands: &[AggregateOperand],
) -> Option<LiftResult> {
    let operands = operands.to_vec();
    let modifiers = source_modifiers(instruction);
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
                element_type: tensor_core_element_type(opcode.kind(), &modifiers),
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

fn source_modifiers(instruction: &SassInstruction) -> Vec<SassModifier> {
    instruction
        .modifiers
        .iter()
        .map(|modifier| SassModifier::parse(modifier.as_str()))
        .collect()
}

fn tensor_core_element_type(
    opcode: &SassOpcodeKind,
    modifiers: &[SassModifier],
) -> Option<SassTensorElementType> {
    if let Some(dtype) = tensor_core_modifier_element_type(modifiers) {
        return Some(dtype);
    }

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

fn tensor_core_modifier_element_type(modifiers: &[SassModifier]) -> Option<SassTensorElementType> {
    [
        SassTensorElementType::Bf16,
        SassTensorElementType::F16,
        SassTensorElementType::Tf32,
        SassTensorElementType::Fp4,
        SassTensorElementType::Fp8,
        SassTensorElementType::Fp64,
        SassTensorElementType::Fp32,
    ]
    .into_iter()
    .find(|candidate| {
        modifiers.iter().any(|modifier| {
            modifier_tensor_element_type(modifier.kind()).as_ref() == Some(candidate)
        })
    })
}

fn modifier_tensor_element_type(modifier: &SassModifierKind) -> Option<SassTensorElementType> {
    match modifier {
        SassModifierKind::Bf16 => Some(SassTensorElementType::Bf16),
        SassModifierKind::F16 => Some(SassTensorElementType::F16),
        SassModifierKind::F32 => Some(SassTensorElementType::Fp32),
        SassModifierKind::F64 => Some(SassTensorElementType::Fp64),
        SassModifierKind::Tf32 => Some(SassTensorElementType::Tf32),
        SassModifierKind::Fp4 | SassModifierKind::E2M1 => Some(SassTensorElementType::Fp4),
        SassModifierKind::Fp8 | SassModifierKind::E4M3 | SassModifierKind::E5M2 => {
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
