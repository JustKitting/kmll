use super::{
    operands::parse_operand,
    tokens::{split_first_token, split_operands},
    types::{SassInstruction, SassParseError, SassPredicate, SassSourcePosition},
};

pub(super) fn parse_instruction_line(
    line: &str,
    source_position: SassSourcePosition,
) -> Result<Option<SassInstruction>, SassParseError> {
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
        source_position,
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
