use super::super::types::{
    AggregateOperand, AggregateOperandKind, KernelIrOpKind, MemoryAccessInfo, MemoryAddress,
    MemoryAddressKind, MemorySpace, SassMappingConfidence, SassMemoryModifier, SassModifier,
    SassOpcode, SassOpcodeKind, SassUnsupportedReason,
};
use super::{LiftResult, operands::register_operand};

pub(super) fn lift(
    opcode: &SassOpcode,
    operands: &[AggregateOperand],
    modifiers: &[SassModifier],
) -> Option<LiftResult> {
    Some(match opcode.kind() {
        SassOpcodeKind::Ldc | SassOpcodeKind::Ldcu | SassOpcodeKind::Uldc => {
            lift_load_const(opcode, operands)
        }
        SassOpcodeKind::Ld | SassOpcodeKind::Ldg | SassOpcodeKind::Lds | SassOpcodeKind::Ldl => {
            if operands.len() != 2 {
                unsupported_arity(opcode, operands.len(), 2)
            } else {
                let dst = register_operand(operands.first());
                let address = memory_address(operands.get(1));
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
            if operands.len() != 2 {
                unsupported_arity(opcode, operands.len(), 2)
            } else {
                let address = memory_address(operands.first());
                let value = register_operand(operands.get(1));
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

fn lift_load_const(opcode: &SassOpcode, operands: &[AggregateOperand]) -> LiftResult {
    if operands.len() != 2 {
        return unsupported_arity(opcode, operands.len(), 2);
    }
    (
        KernelIrOpKind::LoadConst {
            dst: register_operand(operands.first()),
            source: memory_address(operands.get(1)),
        },
        SassMappingConfidence::OpcodeHeuristic,
    )
}

fn unsupported_arity(opcode: &SassOpcode, actual: usize, expected: usize) -> LiftResult {
    (
        KernelIrOpKind::Unsupported {
            opcode: opcode.clone(),
            reason: SassUnsupportedReason::exact_operand_arity(expected, actual),
        },
        SassMappingConfidence::Unsupported,
    )
}

fn memory_address(operand: Option<&AggregateOperand>) -> MemoryAddress {
    match operand {
        Some(AggregateOperand {
            kind: AggregateOperandKind::Memory(address),
            ..
        }) => address.clone(),
        Some(operand) => MemoryAddress::raw(operand.raw.clone()),
        None => MemoryAddress::raw(String::new()),
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
