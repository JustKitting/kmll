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

pub fn format_qwen_single_turn_chat(
    system_prompt: Option<&str>,
    user_prompt: &str,
    enable_thinking: bool,
) -> String {
    let mut prompt = String::new();

    if let Some(system_prompt) = system_prompt.filter(|prompt| !prompt.trim().is_empty()) {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system_prompt.trim());
        prompt.push_str("<|im_end|>\n");
    }

    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(user_prompt.trim());
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>assistant\n");
    if enable_thinking {
        prompt.push_str("<think>\n");
    } else {
        prompt.push_str("<think>\n\n</think>\n\n");
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_single_turn_chat_formats_generation_prompt() {
        let prompt = format_qwen_single_turn_chat(None, "Hello", true);

        assert_eq!(
            prompt,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn qwen_single_turn_chat_formats_system_prompt_and_non_thinking_mode() {
        let prompt = format_qwen_single_turn_chat(Some(" Be concise.\n"), "Hello\n", false);

        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
