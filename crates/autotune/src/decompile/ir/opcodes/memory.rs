use super::super::super::sass::SassInstruction;
use super::super::types::{KernelIrOpKind, MemorySpace};
use super::{LiftResult, operands::map_two_operands};

pub(super) fn lift(opcode: &str, instruction: &SassInstruction) -> Option<LiftResult> {
    Some(match opcode {
        "LDC" | "LDCU" | "ULDC" => map_two_operands(instruction, |dst, source| {
            KernelIrOpKind::LoadConst { dst, source }
        }),
        "LD" | "LDG" | "LDS" | "LDL" => {
            map_two_operands(instruction, |dst, address| KernelIrOpKind::Load {
                dst,
                space: memory_space(opcode, &address),
                address,
            })
        }
        "ST" | "STG" | "STS" | "STL" => {
            map_two_operands(instruction, |address, value| KernelIrOpKind::Store {
                space: memory_space(opcode, &address),
                address,
                value,
            })
        }
        _ => return None,
    })
}

fn memory_space(opcode: &str, address: &str) -> MemorySpace {
    if address.starts_with("desc[") {
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
