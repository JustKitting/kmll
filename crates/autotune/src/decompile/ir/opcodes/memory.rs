use super::super::super::sass::{SassInstruction, SassOperand, SassOperandKind};
use super::super::types::{
    KernelIrOpKind, MemoryAccessInfo, MemoryAddress, MemoryAddressKind, MemorySpace, RegisterRef,
    SassMappingConfidence, SassMemoryModifier, SassModifier, SassOpcode, SassOpcodeKind,
    SassUnsupportedReason,
};
use super::LiftResult;

pub(super) fn lift(
    opcode: &SassOpcode,
    instruction: &SassInstruction,
    modifiers: &[SassModifier],
) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Ldc | SassOpcodeKind::Ldcu | SassOpcodeKind::Uldc => {
            lift_load_const(opcode, instruction)
        }
        SassOpcodeKind::Ld | SassOpcodeKind::Ldg | SassOpcodeKind::Lds | SassOpcodeKind::Ldl => {
            if instruction.operands.len() != 2 {
                unsupported_arity(opcode, instruction, 2)
            } else {
                let dst = RegisterRef::parse(instruction.operands[0].raw.clone());
                let address = memory_address(&instruction.operands[1]);
                (
                    KernelIrOpKind::Load {
                        space: memory_space(opcode.kind(), &address),
                        address,
                        dst,
                        access: memory_access_info(modifiers),
                    },
                    SassMappingConfidence::OpcodeHeuristic,
                )
            }
        }
        SassOpcodeKind::St | SassOpcodeKind::Stg | SassOpcodeKind::Sts | SassOpcodeKind::Stl => {
            if instruction.operands.len() != 2 {
                unsupported_arity(opcode, instruction, 2)
            } else {
                let address = memory_address(&instruction.operands[0]);
                let value = RegisterRef::parse(instruction.operands[1].raw.clone());
                (
                    KernelIrOpKind::Store {
                        space: memory_space(opcode.kind(), &address),
                        address,
                        value,
                        access: memory_access_info(modifiers),
                    },
                    SassMappingConfidence::OpcodeHeuristic,
                )
            }
        }
        _ => return None,
    })
}

fn lift_load_const(opcode: &SassOpcode, instruction: &SassInstruction) -> LiftResult {
    if instruction.operands.len() != 2 {
        return unsupported_arity(opcode, instruction, 2);
    }
    (
        KernelIrOpKind::LoadConst {
            dst: RegisterRef::parse(instruction.operands[0].raw.clone()),
            source: memory_address(&instruction.operands[1]),
        },
        SassMappingConfidence::OpcodeHeuristic,
    )
}

fn unsupported_arity(
    opcode: &SassOpcode,
    instruction: &SassInstruction,
    expected: usize,
) -> LiftResult {
    (
        KernelIrOpKind::Unsupported {
            opcode: opcode.clone(),
            reason: SassUnsupportedReason::exact_operand_arity(
                expected,
                instruction.operands.len(),
            ),
        },
        SassMappingConfidence::Unsupported,
    )
}

fn memory_address(operand: &SassOperand) -> MemoryAddress {
    match &operand.kind {
        SassOperandKind::ConstantMemory { bank, offset } => {
            MemoryAddress::constant(operand.raw.clone(), bank.clone(), offset.clone())
        }
        SassOperandKind::DescriptorMemory {
            descriptor,
            address,
            address_width,
            offset,
        } => MemoryAddress::descriptor(
            operand.raw.clone(),
            descriptor.clone(),
            address.clone(),
            *address_width,
            offset.clone(),
        ),
        SassOperandKind::IndexedMemory { base, offset } => {
            MemoryAddress::indexed(operand.raw.clone(), base.clone(), offset.clone())
        }
        _ => MemoryAddress::raw(operand.raw.clone()),
    }
}

fn memory_space(opcode: &SassOpcodeKind, address: &MemoryAddress) -> MemorySpace {
    if matches!(address.kind, MemoryAddressKind::Descriptor { .. }) {
        return MemorySpace::Descriptor;
    }
    match opcode {
        SassOpcodeKind::Ldg | SassOpcodeKind::Stg | SassOpcodeKind::Ld | SassOpcodeKind::St => {
            MemorySpace::Global
        }
        SassOpcodeKind::Lds | SassOpcodeKind::Sts => MemorySpace::Shared,
        SassOpcodeKind::Ldl | SassOpcodeKind::Stl => MemorySpace::Local,
        SassOpcodeKind::Ldc | SassOpcodeKind::Ldcu | SassOpcodeKind::Uldc => MemorySpace::Constant,
        _ => MemorySpace::Unknown,
    }
}

fn memory_access_info(source_modifiers: &[SassModifier]) -> MemoryAccessInfo {
    let modifiers = source_modifiers
        .iter()
        .map(SassMemoryModifier::from_modifier)
        .collect::<Vec<_>>();
    let width_bits = modifiers.iter().find_map(SassMemoryModifier::width_bits);
    MemoryAccessInfo::new(width_bits, modifiers)
}
