use super::super::types::{
    AggregateOperand, KernelIrOpKind, SassMappingConfidence, SassModifier, SassModifierKind,
    SassOpcode, SassOpcodeKind, SassTensorElementType, SassTensorMmaSignature, SassTensorScope,
};
use super::LiftResult;

pub(super) fn lift(
    opcode: &SassOpcode,
    modifiers: &[SassModifier],
    operands: &[AggregateOperand],
) -> Option<LiftResult> {
    let operands = operands.to_vec();
    let signature = tensor_core_signature(opcode.kind(), modifiers);
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
                element_type: tensor_core_element_type(opcode.kind(), signature.as_ref()),
                signature,
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
        SassOpcodeKind::Usetmaxreg | SassOpcodeKind::Warpgroup | SassOpcodeKind::Warpgroupset => (
            KernelIrOpKind::WarpGroup {
                opcode: opcode.clone(),
                operands,
            },
            SassMappingConfidence::OpcodeHeuristic,
        ),
        _ => return None,
    })
}

fn tensor_core_element_type(
    opcode: &SassOpcodeKind,
    signature: Option<&SassTensorMmaSignature>,
) -> Option<SassTensorElementType> {
    if let Some(dtype) = signature.and_then(SassTensorMmaSignature::primary_element_type) {
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

fn tensor_core_signature(
    _opcode: &SassOpcodeKind,
    modifiers: &[SassModifier],
) -> Option<SassTensorMmaSignature> {
    let shape = modifiers.iter().find_map(|modifier| match modifier.kind() {
        SassModifierKind::TensorShape(shape) => Some(*shape),
        _ => None,
    });
    let dtypes = modifiers
        .iter()
        .filter_map(|modifier| modifier_tensor_element_type(modifier.kind()))
        .collect::<Vec<_>>();

    if shape.is_none() && dtypes.is_empty() {
        return None;
    }

    let mut signature = SassTensorMmaSignature {
        shape,
        ..Default::default()
    };
    match dtypes.as_slice() {
        [] => {}
        [dtype] => {
            signature.lhs_type = Some(dtype.clone());
            signature.rhs_type = Some(dtype.clone());
        }
        [compute, element] => {
            signature.output_type = Some(compute.clone());
            signature.lhs_type = Some(element.clone());
            signature.rhs_type = Some(element.clone());
            signature.accumulator_type = Some(compute.clone());
        }
        [output, lhs, rhs] => {
            signature.output_type = Some(output.clone());
            signature.lhs_type = Some(lhs.clone());
            signature.rhs_type = Some(rhs.clone());
            signature.accumulator_type = Some(output.clone());
        }
        [output, lhs, rhs, accumulator, ..] => {
            signature.output_type = Some(output.clone());
            signature.lhs_type = Some(lhs.clone());
            signature.rhs_type = Some(rhs.clone());
            signature.accumulator_type = Some(accumulator.clone());
        }
    }
    Some(signature)
}

fn modifier_tensor_element_type(modifier: &SassModifierKind) -> Option<SassTensorElementType> {
    match modifier {
        SassModifierKind::Bf16 => Some(SassTensorElementType::Bf16),
        SassModifierKind::F16 => Some(SassTensorElementType::F16),
        SassModifierKind::F32 => Some(SassTensorElementType::Fp32),
        SassModifierKind::F64 => Some(SassTensorElementType::Fp64),
        SassModifierKind::Tf32 => Some(SassTensorElementType::Tf32),
        SassModifierKind::Fp4 => Some(SassTensorElementType::Fp4),
        SassModifierKind::E2M1 => Some(SassTensorElementType::E2M1),
        SassModifierKind::Fp8 => Some(SassTensorElementType::Fp8),
        SassModifierKind::E4M3 => Some(SassTensorElementType::E4M3),
        SassModifierKind::E5M2 => Some(SassTensorElementType::E5M2),
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
