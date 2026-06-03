use std::{error::Error, fmt};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassModule {
    pub target: Option<String>,
    pub functions: Vec<SassFunction>,
}

impl SassModule {
    pub fn instruction_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.instructions.len())
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassFunction {
    pub name: String,
    pub section: Option<String>,
    pub instructions: Vec<SassInstruction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassInstruction {
    pub address: u64,
    pub label: Option<String>,
    pub predicate: Option<SassPredicate>,
    pub opcode: String,
    pub modifiers: Vec<String>,
    pub operands: Vec<SassOperand>,
    pub raw: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassPredicate {
    pub negated: bool,
    pub register: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassOperand {
    pub raw: String,
    pub kind: SassOperandKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SassOperandKind {
    Register(SassRegister),
    Immediate(String),
    ConstantMemory {
        bank: String,
        offset: String,
    },
    DescriptorMemory {
        descriptor: String,
        address: String,
        address_width: Option<u32>,
        offset: Option<String>,
    },
    Label(String),
    Raw,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassRegister {
    pub class: RegisterClass,
    pub index: Option<u16>,
    pub modifiers: Vec<String>,
    pub negated: bool,
    pub absolute: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterClass {
    General,
    Uniform,
    Predicate,
    UniformPredicate,
    Special,
    Barrier,
    Zero,
    PredicateTrue,
    UniformPredicateTrue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassParseError {
    message: String,
}

impl SassParseError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SassParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for SassParseError {}

pub fn parse_nvidia_sass(input: &str) -> Result<SassModule, SassParseError> {
    let mut target = None;
    let mut functions = Vec::new();
    let mut current = None::<SassFunction>;
    let mut pending_global = None::<String>;
    let mut pending_section = None::<String>;
    let mut pending_label = None::<String>;

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix(".target") {
            target = Some(rest.trim().to_string());
            continue;
        }
        if let Some(section) = parse_text_section(trimmed) {
            pending_section = Some(section);
            continue;
        }
        if let Some(global) = parse_global(trimmed) {
            pending_global = Some(global);
            continue;
        }
        if let Some(function_name) = parse_cuobjdump_function(trimmed) {
            if let Some(function) = current.take() {
                functions.push(function);
            }
            current = Some(SassFunction {
                name: function_name.clone(),
                section: pending_section.clone().or(Some(function_name)),
                instructions: Vec::new(),
            });
            pending_global = None;
            pending_label = None;
            continue;
        }
        if is_label_line(trimmed) {
            let label = trimmed.trim_end_matches(':').to_string();
            if pending_global.as_deref() == Some(label.as_str()) {
                if let Some(function) = current.take() {
                    functions.push(function);
                }
                current = Some(SassFunction {
                    name: label,
                    section: pending_section.clone(),
                    instructions: Vec::new(),
                });
                continue;
            }
            if trimmed.starts_with(".text.") {
                continue;
            }
            pending_label = Some(label);
            continue;
        }

        let Some(mut instruction) = parse_instruction_line(trimmed)? else {
            continue;
        };
        instruction.label = pending_label.take();
        if current.is_none() {
            if let Some(name) = pending_global.clone() {
                current = Some(SassFunction {
                    name,
                    section: pending_section.clone(),
                    instructions: Vec::new(),
                });
            }
        }
        let function = current.as_mut().ok_or_else(|| {
            SassParseError::new(format!(
                "instruction appears before a function/global header: {trimmed}"
            ))
        })?;
        function.instructions.push(instruction);
    }

    if let Some(function) = current.take() {
        functions.push(function);
    }

    Ok(SassModule { target, functions })
}

fn parse_text_section(line: &str) -> Option<String> {
    let rest = line.strip_prefix(".section")?.trim();
    let section = rest.split(',').next()?.trim();
    section
        .strip_prefix(".text.")
        .map(|name| name.trim().to_string())
}

fn parse_global(line: &str) -> Option<String> {
    line.strip_prefix(".global")
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
}

fn parse_cuobjdump_function(line: &str) -> Option<String> {
    let rest = line.strip_prefix("Function")?.trim_start();
    let name = rest.strip_prefix(':')?.trim();
    (!name.is_empty()).then(|| name.to_string())
}

fn is_label_line(line: &str) -> bool {
    line.ends_with(':')
        && !line.contains(' ')
        && !line.contains('\t')
        && !line.starts_with(".type")
        && !line.starts_with(".size")
}

fn parse_instruction_line(line: &str) -> Result<Option<SassInstruction>, SassParseError> {
    let Some(rest) = line.strip_prefix("/*") else {
        return Ok(None);
    };
    let Some((address, rest)) = rest.split_once("*/") else {
        return Err(SassParseError::new(format!(
            "instruction address comment is not closed: {line}"
        )));
    };
    let address_text = address.trim();
    if address_text.starts_with("0x") {
        return Ok(None);
    }
    let address = u64::from_str_radix(address_text, 16).map_err(|error| {
        SassParseError::new(format!(
            "invalid instruction address {address_text:?}: {error}"
        ))
    })?;
    let instruction_text = rest.split("/*").next().unwrap_or(rest).trim();
    if instruction_text.is_empty() {
        return Ok(None);
    }
    let instruction_text = instruction_text.trim_end_matches(';').trim();
    if instruction_text.is_empty() {
        return Ok(None);
    }
    if instruction_text.starts_with('.') {
        return Ok(None);
    }

    let (predicate, body) = parse_predicate_prefix(instruction_text);
    let (opcode_with_modifiers, operands_text) =
        split_first_token(body).ok_or_else(|| SassParseError::new("missing opcode"))?;
    let mut opcode_parts = opcode_with_modifiers.split('.');
    let opcode = opcode_parts
        .next()
        .expect("split should yield at least one opcode part")
        .to_string();
    let modifiers = opcode_parts.map(str::to_string).collect::<Vec<_>>();
    let operands = split_operands(operands_text)
        .into_iter()
        .filter(|operand| !operand.is_empty())
        .map(parse_operand)
        .collect::<Vec<_>>();

    Ok(Some(SassInstruction {
        address,
        label: None,
        predicate,
        opcode,
        modifiers,
        operands,
        raw: instruction_text.to_string(),
    }))
}

fn parse_predicate_prefix(text: &str) -> (Option<SassPredicate>, &str) {
    let Some(rest) = text.strip_prefix('@') else {
        return (None, text);
    };
    let Some((predicate, body)) = split_first_token(rest) else {
        return (None, text);
    };
    let (negated, register) = predicate
        .strip_prefix('!')
        .map(|register| (true, register))
        .unwrap_or((false, predicate));
    (
        Some(SassPredicate {
            negated,
            register: register.to_string(),
        }),
        body,
    )
}

fn split_first_token(text: &str) -> Option<(&str, &str)> {
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

fn split_operands(text: &str) -> Vec<&str> {
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

fn parse_operand(raw: &str) -> SassOperand {
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

fn parse_bracketed(text: &str) -> Option<(&str, &str)> {
    let text = text.strip_prefix('[')?;
    let end = text.find(']')?;
    Some((&text[..end], &text[end + 1..]))
}

fn parse_label_operand(raw: &str) -> Option<String> {
    label_in_text(raw)
}

pub(super) fn label_in_text(raw: &str) -> Option<String> {
    let start = raw.find("`(")?;
    let rest = &raw[start + 2..];
    let end = rest.find(')')?;
    Some(rest[..end].to_string())
}

fn parse_register(raw: &str) -> Option<SassRegister> {
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

fn is_immediate(raw: &str) -> bool {
    raw.strip_prefix("0x")
        .map(|value| !value.is_empty() && value.chars().all(|ch| ch.is_ascii_hexdigit()))
        .unwrap_or_else(|| raw.parse::<i64>().is_ok())
}
