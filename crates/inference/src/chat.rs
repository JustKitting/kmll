use std::{fs, path::Path};

use crate::safetensors::Result;

pub fn format_ministral_single_turn_chat(system_prompt: Option<&str>, user_prompt: &str) -> String {
    let mut prompt = String::new();

    if let Some(system_prompt) = system_prompt.filter(|prompt| !prompt.is_empty()) {
        prompt.push_str("[SYSTEM_PROMPT]");
        prompt.push_str(system_prompt);
        prompt.push_str("[/SYSTEM_PROMPT]");
    }

    prompt.push_str("[INST]");
    prompt.push_str(user_prompt);
    prompt.push_str("[/INST]");

    prompt
}

pub fn default_ministral_system_prompt(model_dir: impl AsRef<Path>) -> Result<String> {
    let prompt = fs::read_to_string(model_dir.as_ref().join("SYSTEM_PROMPT.txt"))?;
    Ok(trim_trailing_newlines(prompt))
}

fn trim_trailing_newlines(mut value: String) -> String {
    while value.ends_with('\n') || value.ends_with('\r') {
        value.pop();
    }
    value
}
