use super::{
    super::{
        KernelIrFunction, KernelIrOp, KernelIrOpKind, MemoryAddress, MemorySpace, RegisterRef,
    },
    types::{SassMemoryAccess, SassMemoryAccessKind},
};

pub(super) fn analyze_memory_accesses(function: &KernelIrFunction) -> Vec<SassMemoryAccess> {
    let mut accesses = function
        .ops
        .iter()
        .filter_map(|op| match &op.kind {
            KernelIrOpKind::Load {
                dst,
                address,
                space,
                access,
            } => Some(memory_access(
                op,
                SassMemoryAccessKind::Load,
                *space,
                access.width_bits,
                dst,
                address,
            )),
            KernelIrOpKind::LoadConst { dst, source } => Some(memory_access(
                op,
                SassMemoryAccessKind::LoadConst,
                MemorySpace::Constant,
                memory_width_bits(&op.source_modifiers),
                dst,
                source,
            )),
            KernelIrOpKind::Store {
                address,
                value,
                space,
                access,
            } => Some(memory_access(
                op,
                SassMemoryAccessKind::Store,
                *space,
                access.width_bits,
                value,
                address,
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    accesses.sort_by_key(|access| access.address);
    accesses
}

fn memory_access(
    op: &KernelIrOp,
    kind: SassMemoryAccessKind,
    space: MemorySpace,
    width_bits: Option<u32>,
    value_register: &RegisterRef,
    address: &MemoryAddress,
) -> SassMemoryAccess {
    SassMemoryAccess {
        address: op.address,
        predicate: op.predicate.as_ref().map(ToString::to_string),
        kind,
        space,
        width_bits,
        value_register: value_register.clone(),
        memory_address: address.clone(),
        address_registers: address
            .registers()
            .into_iter()
            .filter(|register| !register.is_pseudo())
            .collect(),
        address_base: address.base(),
        offset: address.offset().cloned(),
        source: op.source.clone(),
    }
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
