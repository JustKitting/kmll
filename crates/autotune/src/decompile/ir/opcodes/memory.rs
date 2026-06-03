use super::super::super::sass::{SassInstruction, SassOperand, SassOperandKind};
use super::super::types::{
    KernelIrOpKind, MemoryAccessInfo, MemoryAddress, MemoryAddressKind, MemorySpace, RegisterRef,
    SassMappingConfidence,
};
use super::LiftResult;

pub(super) fn lift(opcode: &str, instruction: &SassInstruction) -> Option<LiftResult> {
    Some(match opcode {
        "LDC" | "LDCU" | "ULDC" => lift_load_const(instruction),
        "LD" | "LDG" | "LDS" | "LDL" => {
            if instruction.operands.len() != 2 {
                unsupported_arity(instruction, 2)
            } else {
                let dst = RegisterRef::parse(instruction.operands[0].raw.clone());
                let address = memory_address(&instruction.operands[1]);
                (
                    KernelIrOpKind::Load {
                        space: memory_space(opcode, &address),
                        address,
                        dst,
                        access: memory_access_info(instruction),
                    },
                    SassMappingConfidence::OpcodeHeuristic,
                )
            }
        }
        "ST" | "STG" | "STS" | "STL" => {
            if instruction.operands.len() != 2 {
                unsupported_arity(instruction, 2)
            } else {
                let address = memory_address(&instruction.operands[0]);
                let value = RegisterRef::parse(instruction.operands[1].raw.clone());
                (
                    KernelIrOpKind::Store {
                        space: memory_space(opcode, &address),
                        address,
                        value,
                        access: memory_access_info(instruction),
                    },
                    SassMappingConfidence::OpcodeHeuristic,
                )
            }
        }
        _ => return None,
    })
}

fn lift_load_const(instruction: &SassInstruction) -> LiftResult {
    if instruction.operands.len() != 2 {
        return unsupported_arity(instruction, 2);
    }
    (
        KernelIrOpKind::LoadConst {
            dst: RegisterRef::parse(instruction.operands[0].raw.clone()),
            source: memory_address(&instruction.operands[1]),
        },
        SassMappingConfidence::OpcodeHeuristic,
    )
}

fn unsupported_arity(instruction: &SassInstruction, expected: usize) -> LiftResult {
    (
        KernelIrOpKind::Unsupported {
            opcode: instruction.opcode.clone(),
            reason: format!(
                "expected {expected} operands, saw {}",
                instruction.operands.len()
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

fn memory_space(opcode: &str, address: &MemoryAddress) -> MemorySpace {
    if matches!(address.kind, MemoryAddressKind::Descriptor { .. }) {
        return MemorySpace::Descriptor;
    }
    match opcode {
        "LDG" | "STG" | "LD" | "ST" => MemorySpace::Global,
        "LDS" | "STS" => MemorySpace::Shared,
        "LDL" | "STL" => MemorySpace::Local,
        "LDC" | "LDCU" | "ULDC" => MemorySpace::Constant,
        _ => MemorySpace::Unknown,
    }
}

fn memory_access_info(instruction: &SassInstruction) -> MemoryAccessInfo {
    MemoryAccessInfo::new(
        memory_width_bits(&instruction.modifiers),
        instruction.modifiers.clone(),
    )
}

fn memory_width_bits(modifiers: &[String]) -> Option<u32> {
    modifiers.iter().find_map(|modifier| {
        modifier
            .strip_prefix('U')
            .or_else(|| modifier.strip_prefix('S'))
            .and_then(|bits| bits.parse::<u32>().ok())
            .or_else(|| modifier.parse::<u32>().ok())
    })
}
