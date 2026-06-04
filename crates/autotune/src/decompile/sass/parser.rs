use super::{
    headers::{is_label_line, parse_cuobjdump_function, parse_global, parse_text_section},
    instruction::parse_instruction_line,
    types::{SassFunction, SassModule, SassParseError, SassSourcePosition},
};

pub fn parse_nvidia_sass(input: &str) -> Result<SassModule, SassParseError> {
    let mut target = None;
    let mut functions = Vec::new();
    let mut current = None::<SassFunction>;
    let mut pending_global = None::<String>;
    let mut pending_section = None::<String>;
    let mut pending_label = None::<String>;
    let mut instruction_ordinal = 0usize;

    for (line_index, line) in input.lines().enumerate() {
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

        let Some(mut instruction) = parse_instruction_line(
            trimmed,
            SassSourcePosition::new(line_index + 1, instruction_ordinal),
        )?
        else {
            continue;
        };
        instruction_ordinal += 1;
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
