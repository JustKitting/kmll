use super::types::{RegisterClass, SassRegister};

pub(super) fn parse_register(raw: &str) -> Option<SassRegister> {
    let mut text = raw.trim();
    let negated = text
        .strip_prefix('!')
        .map(|rest| {
            text = rest;
            true
        })
        .unwrap_or(false)
        || text
            .strip_prefix('-')
            .map(|rest| {
                text = rest;
                true
            })
            .unwrap_or(false);
    let absolute = text.starts_with('|') && text.ends_with('|');
    if absolute {
        text = &text[1..text.len() - 1];
    }
    let mut parts = text.split('.');
    let base = parts.next()?;
    let modifiers = parts.map(str::to_string).collect::<Vec<_>>();
    let (class, index) = parse_register_base(base)?;
    Some(SassRegister {
        class,
        index,
        modifiers,
        negated,
        absolute,
    })
}

fn parse_register_base(base: &str) -> Option<(RegisterClass, Option<u16>)> {
    match base {
        "RZ" => return Some((RegisterClass::Zero, None)),
        "URZ" => return Some((RegisterClass::Zero, None)),
        "PT" => return Some((RegisterClass::PredicateTrue, None)),
        "UPT" => return Some((RegisterClass::UniformPredicateTrue, None)),
        _ => {}
    }
    if let Some(index) = base.strip_prefix("SR_") {
        return Some((RegisterClass::Special, parse_trailing_index(index)));
    }
    if let Some(index) = base.strip_prefix('R') {
        return index
            .parse::<u16>()
            .ok()
            .map(|index| (RegisterClass::General, Some(index)));
    }
    if let Some(index) = base.strip_prefix("UR") {
        return index
            .parse::<u16>()
            .ok()
            .map(|index| (RegisterClass::Uniform, Some(index)));
    }
    if let Some(index) = base.strip_prefix('P') {
        return index
            .parse::<u16>()
            .ok()
            .map(|index| (RegisterClass::Predicate, Some(index)));
    }
    if let Some(index) = base.strip_prefix("UP") {
        return index
            .parse::<u16>()
            .ok()
            .map(|index| (RegisterClass::UniformPredicate, Some(index)));
    }
    if let Some(index) = base.strip_prefix('B') {
        return index
            .parse::<u16>()
            .ok()
            .map(|index| (RegisterClass::Barrier, Some(index)));
    }
    None
}

fn parse_trailing_index(text: &str) -> Option<u16> {
    let digits = text
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.chars().rev().collect::<String>().parse().ok()
    }
}
