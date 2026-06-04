use super::{
    labels::label_in_text,
    registers::parse_register,
    types::{SassOperand, SassOperandKind},
};

pub(super) fn parse_operand(raw: &str) -> SassOperand {
    let raw = raw.trim();
    if let Some((bank, offset)) = parse_constant_memory(raw) {
        return SassOperand {
            raw: raw.to_string(),
            kind: SassOperandKind::ConstantMemory { bank, offset },
        };
    }
    if let Some((descriptor, address, address_width, offset)) = parse_descriptor_memory(raw) {
        return SassOperand {
            raw: raw.to_string(),
            kind: SassOperandKind::DescriptorMemory {
                descriptor,
                address,
                address_width,
                offset,
            },
        };
    }
    if let Some((base, offset)) = parse_indexed_memory(raw) {
        return SassOperand {
            raw: raw.to_string(),
            kind: SassOperandKind::IndexedMemory { base, offset },
        };
    }
    if let Some(label) = parse_label_operand(raw) {
        return SassOperand {
            raw: raw.to_string(),
            kind: SassOperandKind::Label(label),
        };
    }
    if let Some(register) = parse_register(raw) {
        return SassOperand {
            raw: raw.to_string(),
            kind: SassOperandKind::Register(register),
        };
    }
    if is_immediate(raw) {
        return SassOperand {
            raw: raw.to_string(),
            kind: SassOperandKind::Immediate(raw.to_string()),
        };
    }
    SassOperand {
        raw: raw.to_string(),
        kind: SassOperandKind::Raw,
    }
}

fn parse_constant_memory(raw: &str) -> Option<(String, String)> {
    let rest = raw.strip_prefix('c')?;
    let (bank, rest) = parse_bracketed(rest)?;
    let (offset, tail) = parse_bracketed(rest)?;
    if tail.trim().is_empty() {
        Some((bank.to_string(), offset.to_string()))
    } else {
        None
    }
}

fn parse_descriptor_memory(raw: &str) -> Option<(String, String, Option<u32>, Option<String>)> {
    let rest = raw.strip_prefix("desc")?;
    let (descriptor, rest) = parse_bracketed(rest)?;
    let (address_raw, tail) = parse_bracketed(rest)?;
    if !tail.trim().is_empty() {
        return None;
    }
    let (address_part, offset) = match address_raw.split_once('+') {
        Some((address, offset)) => (address.trim(), Some(offset.trim().to_string())),
        None => (address_raw.trim(), None),
    };
    let (address, address_width) = match address_part.split_once('.') {
        Some((address, width)) => (address.to_string(), width.parse::<u32>().ok()),
        None => (address_part.to_string(), None),
    };
    Some((descriptor.to_string(), address, address_width, offset))
}

fn parse_indexed_memory(raw: &str) -> Option<(String, Option<String>)> {
    let (inside, tail) = parse_bracketed(raw)?;
    if !tail.trim().is_empty() {
        return None;
    }
    let (base, offset) = match inside.split_once('+') {
        Some((base, offset)) => (base.trim(), Some(offset.trim().to_string())),
        None => (inside.trim(), None),
    };
    (!base.is_empty()).then(|| (base.to_string(), offset))
}

fn parse_bracketed(text: &str) -> Option<(&str, &str)> {
    let text = text.strip_prefix('[')?;
    let end = text.find(']')?;
    Some((&text[..end], &text[end + 1..]))
}

fn parse_label_operand(raw: &str) -> Option<String> {
    label_in_text(raw)
}

fn is_immediate(raw: &str) -> bool {
    raw.strip_prefix("0x")
        .map(|value| !value.is_empty() && value.chars().all(|ch| ch.is_ascii_hexdigit()))
        .unwrap_or_else(|| raw.parse::<i64>().is_ok())
}
