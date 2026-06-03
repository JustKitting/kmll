use super::{
    super::{KernelIrFunction, KernelIrOp, KernelIrOpKind, MemorySpace},
    registers::extract_registers,
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
    value_register: &str,
    address_expr: &str,
) -> SassMemoryAccess {
    let (address_base, offset) = memory_base_and_offset(space, address_expr);
    SassMemoryAccess {
        address: op.address,
        predicate: op.predicate.clone(),
        kind,
        space,
        width_bits,
        value_register: value_register.to_string(),
        address_expr: address_expr.to_string(),
        address_registers: extract_registers(address_expr),
        address_base,
        offset,
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

fn memory_base_and_offset(
    space: MemorySpace,
    address_expr: &str,
) -> (Option<String>, Option<String>) {
    match space {
        MemorySpace::Descriptor => descriptor_base_and_offset(address_expr),
        MemorySpace::Constant => constant_base_and_offset(address_expr),
        _ => (None, offset_after_plus(address_expr)),
    }
}

fn descriptor_base_and_offset(address_expr: &str) -> (Option<String>, Option<String>) {
    let Some(rest) = address_expr.strip_prefix("desc") else {
        return (None, offset_after_plus(address_expr));
    };
    let Some((descriptor, rest)) = bracketed(rest) else {
        return (None, offset_after_plus(address_expr));
    };
    let offset = bracketed(rest)
        .and_then(|(address, tail)| tail.trim().is_empty().then_some(address))
        .and_then(offset_after_plus);
    (Some(descriptor.to_string()), offset)
}

fn constant_base_and_offset(address_expr: &str) -> (Option<String>, Option<String>) {
    let Some(rest) = address_expr.strip_prefix('c') else {
        return (None, offset_after_plus(address_expr));
    };
    let Some((bank, rest)) = bracketed(rest) else {
        return (None, offset_after_plus(address_expr));
    };
    let offset = bracketed(rest)
        .and_then(|(offset, tail)| tail.trim().is_empty().then_some(offset.to_string()));
    (Some(bank.to_string()), offset)
}

fn bracketed(text: &str) -> Option<(&str, &str)> {
    let text = text.strip_prefix('[')?;
    let end = text.find(']')?;
    Some((&text[..end], &text[end + 1..]))
}

fn offset_after_plus(text: &str) -> Option<String> {
    text.split_once('+')
        .map(|(_, offset)| offset.trim().to_string())
        .filter(|offset| !offset.is_empty())
}
