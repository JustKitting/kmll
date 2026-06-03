use super::super::RegisterRef;

pub(super) fn push_registers(text: &str, out: &mut Vec<RegisterRef>) {
    for register in extract_registers(text) {
        push_register_ref(register, out);
    }
}

pub(super) fn push_register_refs(
    registers: impl IntoIterator<Item = RegisterRef>,
    out: &mut Vec<RegisterRef>,
) {
    for register in registers {
        push_register_ref(register, out);
    }
}

fn push_register_ref(register: RegisterRef, out: &mut Vec<RegisterRef>) {
    if !register.is_pseudo() && !out.contains(&register) {
        out.push(register);
    }
}

pub(super) fn extract_registers(text: &str) -> Vec<RegisterRef> {
    let bytes = text.as_bytes();
    let mut index = 0usize;
    let mut registers = Vec::new();
    while index < bytes.len() {
        let Some((raw_register, consumed)) = parse_register_at(text, index) else {
            index += 1;
            continue;
        };
        push_register_ref(RegisterRef::parse(raw_register), &mut registers);
        index += consumed;
    }
    registers
}

fn parse_register_at(text: &str, index: usize) -> Option<(String, usize)> {
    if !is_token_boundary(text, index) {
        return None;
    }
    let tail = &text[index..];
    for literal in ["SR_", "URZ", "UPT", "UR", "UP", "RZ", "PT", "R", "P", "B"] {
        if let Some(register) = parse_register_prefix(tail, literal) {
            return Some(register);
        }
    }
    None
}

fn parse_register_prefix(tail: &str, prefix: &str) -> Option<(String, usize)> {
    let rest = tail.strip_prefix(prefix)?;
    match prefix {
        "SR_" => {
            let len = rest
                .char_indices()
                .take_while(|(_, ch)| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '.')
                .map(|(index, ch)| index + ch.len_utf8())
                .last()
                .unwrap_or(0);
            (len > 0).then(|| (tail[..prefix.len() + len].to_string(), prefix.len() + len))
        }
        "URZ" | "UPT" | "RZ" | "PT" => Some((prefix.to_string(), prefix.len())),
        "UR" | "UP" | "R" | "P" | "B" => {
            let len = rest
                .char_indices()
                .take_while(|(_, ch)| ch.is_ascii_digit())
                .map(|(index, ch)| index + ch.len_utf8())
                .last()
                .unwrap_or(0);
            (len > 0).then(|| (tail[..prefix.len() + len].to_string(), prefix.len() + len))
        }
        _ => None,
    }
}

fn is_token_boundary(text: &str, index: usize) -> bool {
    if index == 0 {
        return true;
    }
    let before = text[..index]
        .chars()
        .next_back()
        .expect("index > 0 should have previous char");
    !before.is_ascii_alphanumeric() && before != '_'
}
