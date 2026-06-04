use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
};

use super::super::super::sass::{RegisterClass, SassRegister};

#[derive(Debug, Clone)]
pub struct RegisterRef {
    pub kind: RegisterRefKind,
    pub raw: String,
    pub negated: bool,
    pub absolute: bool,
}

impl RegisterRef {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        let kind = parse_register_ref_kind(&raw).unwrap_or(RegisterRefKind::Raw);
        let decorations = register_ref_decorations(&raw);
        Self {
            raw: canonical_register_ref_text(&kind, &raw),
            kind,
            negated: decorations.negated,
            absolute: decorations.absolute,
        }
    }

    pub fn from_sass_register(raw: impl Into<String>, register: &SassRegister) -> Self {
        let raw = raw.into();
        let kind = match register.class {
            RegisterClass::General => register
                .index
                .map(RegisterRefKind::General)
                .unwrap_or(RegisterRefKind::Raw),
            RegisterClass::Uniform => register
                .index
                .map(RegisterRefKind::Uniform)
                .unwrap_or(RegisterRefKind::Raw),
            RegisterClass::Predicate => register
                .index
                .map(RegisterRefKind::Predicate)
                .unwrap_or(RegisterRefKind::Raw),
            RegisterClass::UniformPredicate => register
                .index
                .map(RegisterRefKind::UniformPredicate)
                .unwrap_or(RegisterRefKind::Raw),
            RegisterClass::Special => {
                RegisterRefKind::Special(register_base_without_modifiers(&raw).to_string())
            }
            RegisterClass::Barrier => register
                .index
                .map(RegisterRefKind::Barrier)
                .unwrap_or(RegisterRefKind::Raw),
            RegisterClass::Zero => match register_base_without_modifiers(&raw) {
                "URZ" => RegisterRefKind::UniformZero,
                _ => RegisterRefKind::GeneralZero,
            },
            RegisterClass::PredicateTrue => RegisterRefKind::PredicateTrue,
            RegisterClass::UniformPredicateTrue => RegisterRefKind::UniformPredicateTrue,
        };
        Self {
            raw: canonical_register_ref_text(&kind, &raw),
            kind,
            negated: register.negated,
            absolute: register.absolute,
        }
    }

    pub fn is_pseudo(&self) -> bool {
        matches!(
            self.kind,
            RegisterRefKind::GeneralZero
                | RegisterRefKind::UniformZero
                | RegisterRefKind::PredicateTrue
                | RegisterRefKind::UniformPredicateTrue
        )
    }

    pub fn extract_all(text: &str) -> Vec<Self> {
        let bytes = text.as_bytes();
        let mut index = 0usize;
        let mut registers = Vec::new();
        while index < bytes.len() {
            let Some((raw_register, consumed)) = parse_register_at(text, index) else {
                index += 1;
                continue;
            };
            let register = Self::parse(raw_register);
            if !registers.contains(&register) {
                registers.push(register);
            }
            index += consumed;
        }
        registers
    }
}

impl fmt::Display for RegisterRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

impl PartialEq for RegisterRef {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && (!matches!(self.kind, RegisterRefKind::Raw) || self.raw == other.raw)
    }
}

impl Eq for RegisterRef {}

impl PartialOrd for RegisterRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RegisterRef {
    fn cmp(&self, other: &Self) -> Ordering {
        self.kind.cmp(&other.kind).then_with(|| {
            if matches!(self.kind, RegisterRefKind::Raw) {
                self.raw.cmp(&other.raw)
            } else {
                Ordering::Equal
            }
        })
    }
}

impl Hash for RegisterRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        if matches!(self.kind, RegisterRefKind::Raw) {
            self.raw.hash(state);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegisterRefKind {
    General(u16),
    Uniform(u16),
    Predicate(u16),
    UniformPredicate(u16),
    Special(String),
    Barrier(u16),
    GeneralZero,
    UniformZero,
    PredicateTrue,
    UniformPredicateTrue,
    Raw,
}

fn parse_register_ref_kind(raw: &str) -> Option<RegisterRefKind> {
    let base = register_base_without_modifiers(raw);
    match base {
        "RZ" => Some(RegisterRefKind::GeneralZero),
        "URZ" => Some(RegisterRefKind::UniformZero),
        "PT" => Some(RegisterRefKind::PredicateTrue),
        "UPT" => Some(RegisterRefKind::UniformPredicateTrue),
        _ => base
            .strip_prefix("SR_")
            .map(|_| RegisterRefKind::Special(base.to_string()))
            .or_else(|| {
                base.strip_prefix("UR")
                    .and_then(|index| index.parse::<u16>().ok())
                    .map(RegisterRefKind::Uniform)
            })
            .or_else(|| {
                base.strip_prefix("UP")
                    .and_then(|index| index.parse::<u16>().ok())
                    .map(RegisterRefKind::UniformPredicate)
            })
            .or_else(|| {
                base.strip_prefix('R')
                    .and_then(|index| index.parse::<u16>().ok())
                    .map(RegisterRefKind::General)
            })
            .or_else(|| {
                base.strip_prefix('P')
                    .and_then(|index| index.parse::<u16>().ok())
                    .map(RegisterRefKind::Predicate)
            })
            .or_else(|| {
                base.strip_prefix('B')
                    .and_then(|index| index.parse::<u16>().ok())
                    .map(RegisterRefKind::Barrier)
            }),
    }
}

fn canonical_register_ref_text(kind: &RegisterRefKind, raw: &str) -> String {
    match kind {
        RegisterRefKind::General(index) => format!("R{index}"),
        RegisterRefKind::Uniform(index) => format!("UR{index}"),
        RegisterRefKind::Predicate(index) => format!("P{index}"),
        RegisterRefKind::UniformPredicate(index) => format!("UP{index}"),
        RegisterRefKind::Special(special) => special.clone(),
        RegisterRefKind::Barrier(index) => format!("B{index}"),
        RegisterRefKind::GeneralZero => "RZ".to_string(),
        RegisterRefKind::UniformZero => "URZ".to_string(),
        RegisterRefKind::PredicateTrue => "PT".to_string(),
        RegisterRefKind::UniformPredicateTrue => "UPT".to_string(),
        RegisterRefKind::Raw => raw.to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegisterRefDecorations {
    negated: bool,
    absolute: bool,
}

fn register_ref_decorations(raw: &str) -> RegisterRefDecorations {
    let text = raw.trim();
    RegisterRefDecorations {
        negated: text.starts_with('!') || text.starts_with('-'),
        absolute: text
            .trim_start_matches('!')
            .trim_start_matches('-')
            .starts_with('|')
            && text.ends_with('|'),
    }
}

fn register_base_without_modifiers(raw: &str) -> &str {
    let text = raw
        .trim()
        .trim_start_matches('!')
        .trim_start_matches('-')
        .trim_matches('|');
    text.split('.').next().unwrap_or(text)
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
