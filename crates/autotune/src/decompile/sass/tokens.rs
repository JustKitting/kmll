pub(super) fn split_first_token(text: &str) -> Option<(&str, &str)> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            return Some((&text[..index], text[index..].trim()));
        }
    }
    Some((text, ""))
}

pub(super) fn split_operands(text: &str) -> Vec<&str> {
    let mut operands = Vec::new();
    let mut start = 0;
    let mut bracket_depth = 0usize;
    let mut paren_depth = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            ',' if bracket_depth == 0 && paren_depth == 0 => {
                operands.push(text[start..index].trim());
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    if start <= text.len() {
        let tail = text[start..].trim();
        if !tail.is_empty() {
            operands.push(tail);
        }
    }
    operands
}
