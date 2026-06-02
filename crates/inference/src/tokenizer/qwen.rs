use std::{
    collections::{HashMap, HashSet},
    fs, io,
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::safetensors::Result;

use super::tekken::tekken_pre_tokenize;

pub struct QwenByteLevelBpeTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<Option<String>>,
    merge_ranks: HashMap<(String, String), usize>,
    added_tokens_by_content: Vec<(String, u32)>,
    added_token_ids: HashSet<u32>,
    special_token_ids: HashSet<u32>,
    byte_to_char: Vec<char>,
    char_to_byte: HashMap<char, u8>,
}

impl QwenByteLevelBpeTokenizer {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        Self::from_paths(
            model_dir.join("vocab.json"),
            model_dir.join("merges.txt"),
            model_dir.join("tokenizer.json"),
        )
    }

    pub fn from_paths(
        vocab_path: impl AsRef<Path>,
        merges_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let vocab_json = fs::read_to_string(vocab_path)?;
        let merges = fs::read_to_string(merges_path)?;
        let tokenizer_json = fs::read_to_string(tokenizer_path)?;
        Self::from_model_files(&vocab_json, &merges, &tokenizer_json)
    }

    pub fn from_model_files(vocab_json: &str, merges: &str, tokenizer_json: &str) -> Result<Self> {
        let vocab: HashMap<String, u32> = serde_json::from_str(vocab_json)?;
        if vocab.is_empty() {
            return Err(invalid_data("Qwen vocab.json did not contain any entries"));
        }
        let max_id = vocab
            .values()
            .copied()
            .max()
            .ok_or_else(|| invalid_data("Qwen vocab.json did not contain any ids"))?
            as usize;
        let mut id_to_token = vec![None; max_id + 1];
        for (token, &id) in &vocab {
            let id = id as usize;
            if id >= id_to_token.len() {
                id_to_token.resize_with(id + 1, || None);
            }
            id_to_token[id] = Some(token.clone());
        }

        let merge_ranks = parse_merge_ranks(merges)?;
        let (added_tokens_by_content, added_token_ids, special_token_ids) =
            parse_added_tokens(tokenizer_json, &mut id_to_token)?;
        let (byte_to_char, char_to_byte) = byte_unicode_maps();

        Ok(Self {
            vocab,
            id_to_token,
            merge_ranks,
            added_tokens_by_content,
            added_token_ids,
            special_token_ids,
            byte_to_char,
            char_to_byte,
        })
    }

    pub fn encode_lossy(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        if add_bos {
            let bos = self
                .bos_token_id()
                .ok_or_else(|| invalid_data("Qwen tokenizer is missing a usable BOS token id"))?;
            tokens.push(bos);
        }

        let mut i = 0;
        while i < text.len() {
            if let Some((token, id)) = self.match_added_token_at(text, i) {
                tokens.push(id);
                i += token.len();
                continue;
            }

            let end = self.next_added_token_start(text, i).unwrap_or(text.len());
            self.encode_text_span(&text[i..end], &mut tokens)?;
            i = end;
        }

        Ok(tokens)
    }

    pub fn decode_lossy(&self, tokens: &[u32]) -> Result<String> {
        self.decode_lossy_with_options(tokens, true)
    }

    pub fn decode_lossy_with_options(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in tokens {
            if self.added_token_ids.contains(&id) {
                if skip_special && self.special_token_ids.contains(&id) {
                    continue;
                }
                let content = self
                    .id_to_token
                    .get(id as usize)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| invalid_data(format!("missing added token content for {id}")))?;
                bytes.extend_from_slice(content.as_bytes());
                continue;
            }

            let token = self
                .id_to_token
                .get(id as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| invalid_data(format!("missing Qwen token id {id}")))?;
            for ch in token.chars() {
                let byte = self.char_to_byte.get(&ch).ok_or_else(|| {
                    invalid_data(format!(
                        "Qwen token {id} contains non-bytelevel char {ch:?}"
                    ))
                })?;
                bytes.push(*byte);
            }
        }

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.added_tokens_by_content
            .iter()
            .find_map(|(content, id)| (content == token).then_some(*id))
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.special_token_id("<|endoftext|>")
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.special_token_id("<|endoftext|>")
    }

    pub fn im_end_token_id(&self) -> Option<u32> {
        self.special_token_id("<|im_end|>")
    }

    pub fn default_qwen_path(model_dir: impl AsRef<Path>) -> PathBuf {
        model_dir.as_ref().join("tokenizer.json")
    }

    fn encode_text_span(&self, text: &str, tokens: &mut Vec<u32>) -> Result<()> {
        for piece in tekken_pre_tokenize(text) {
            self.encode_bpe_bytes(piece.as_bytes(), tokens)?;
        }
        Ok(())
    }

    fn encode_bpe_bytes(&self, bytes: &[u8], tokens: &mut Vec<u32>) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }

        let bytelevel = bytes
            .iter()
            .map(|byte| self.byte_to_char[*byte as usize])
            .collect::<String>();
        let mut pieces: Vec<String> = bytelevel.chars().map(|ch| ch.to_string()).collect();
        while pieces.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..pieces.len() - 1 {
                let pair = (pieces[i].clone(), pieces[i + 1].clone());
                let Some(&rank) = self.merge_ranks.get(&pair) else {
                    continue;
                };
                if best.is_none_or(|(_, best_rank)| rank < best_rank) {
                    best = Some((i, rank));
                }
            }

            let Some((merge_at, _)) = best else {
                break;
            };
            let right = pieces.remove(merge_at + 1);
            pieces[merge_at].push_str(&right);
        }

        for piece in pieces {
            let token_id = self.vocab.get(&piece).ok_or_else(|| {
                invalid_data(format!(
                    "missing Qwen vocab entry for byte-level piece {piece:?}"
                ))
            })?;
            tokens.push(*token_id);
        }
        Ok(())
    }

    fn match_added_token_at(&self, text: &str, start: usize) -> Option<(String, u32)> {
        self.added_tokens_by_content
            .iter()
            .find(|(token, _)| text[start..].starts_with(token))
            .cloned()
    }

    fn next_added_token_start(&self, text: &str, start: usize) -> Option<usize> {
        self.added_tokens_by_content
            .iter()
            .filter_map(|(token, _)| {
                (!token.is_empty())
                    .then(|| text[start..].find(token))
                    .flatten()
                    .map(|offset| start + offset)
                    .filter(|offset| *offset > start)
            })
            .min()
    }
}

fn parse_merge_ranks(merges: &str) -> Result<HashMap<(String, String), usize>> {
    let mut ranks = HashMap::new();
    for (rank, raw_line) in merges.lines().enumerate() {
        let line = raw_line.trim_end();
        if line.is_empty() || line.starts_with("#version") {
            continue;
        }
        let Some((left, right)) = line.split_once(' ') else {
            return Err(invalid_data(format!(
                "Qwen merge line {} is missing a separator",
                rank + 1
            )));
        };
        ranks.insert((left.to_string(), right.to_string()), rank);
    }
    if ranks.is_empty() {
        return Err(invalid_data("Qwen merges.txt did not contain any merges"));
    }
    Ok(ranks)
}

fn parse_added_tokens(
    tokenizer_json: &str,
    id_to_token: &mut Vec<Option<String>>,
) -> Result<(Vec<(String, u32)>, HashSet<u32>, HashSet<u32>)> {
    let root: Value = serde_json::from_str(tokenizer_json)?;
    let Some(added_tokens) = root.get("added_tokens").and_then(Value::as_array) else {
        return Ok((Vec::new(), HashSet::new(), HashSet::new()));
    };

    let mut by_content = Vec::with_capacity(added_tokens.len());
    let mut added_ids = HashSet::with_capacity(added_tokens.len());
    let mut special_ids = HashSet::new();
    for token in added_tokens {
        let id = token
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid_data("Qwen added token missing numeric id"))?
            as u32;
        let content = token
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_data("Qwen added token missing content"))?
            .to_string();
        let special = token
            .get("special")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let id_index = id as usize;
        if id_index >= id_to_token.len() {
            id_to_token.resize_with(id_index + 1, || None);
        }
        id_to_token[id_index] = Some(content.clone());
        by_content.push((content, id));
        added_ids.insert(id);
        if special {
            special_ids.insert(id);
        }
    }

    by_content.sort_by(|(left, _), (right, _)| {
        right.len().cmp(&left.len()).then_with(|| left.cmp(right))
    });
    Ok((by_content, added_ids, special_ids))
}

fn byte_unicode_maps() -> (Vec<char>, HashMap<char, u8>) {
    let mut bytes = Vec::new();
    bytes.extend(0x21_u8..=0x7e);
    bytes.extend(0xa1_u8..=0xac);
    bytes.extend(0xae_u8..=0xff);

    let mut chars = bytes.iter().map(|byte| *byte as u32).collect::<Vec<_>>();
    let mut n = 0_u32;
    for byte in 0_u16..=255 {
        let byte = byte as u8;
        if !bytes.contains(&byte) {
            bytes.push(byte);
            chars.push(256 + n);
            n += 1;
        }
    }

    let mut byte_to_char = vec!['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    for (byte, char_code) in bytes.into_iter().zip(chars) {
        let ch = char::from_u32(char_code).expect("GPT byte-level codepoint is valid");
        byte_to_char[byte as usize] = ch;
        char_to_byte.insert(ch, byte);
    }
    (byte_to_char, char_to_byte)
}

fn invalid_data(message: impl Into<String>) -> Box<dyn std::error::Error> {
    io::Error::new(io::ErrorKind::InvalidData, message.into()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> QwenByteLevelBpeTokenizer {
        QwenByteLevelBpeTokenizer::from_model_files(
            r#"{
                "H": 0,
                "e": 1,
                "l": 2,
                "o": 3,
                "He": 4,
                "Hel": 5,
                "Hell": 6,
                "Hello": 7,
                "Ġ": 8,
                "ĠHello": 9
            }"#,
            "H e\nHe l\nHel l\nHell o\nĠ Hello\n",
            r#"{
                "added_tokens": [
                    {
                        "id": 10,
                        "content": "<|endoftext|>",
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true
                    }
                ]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn byte_level_mapping_round_trips_all_bytes() {
        let (byte_to_char, char_to_byte) = byte_unicode_maps();
        for byte in 0_u8..=255 {
            let ch = byte_to_char[byte as usize];
            assert_eq!(char_to_byte.get(&ch), Some(&byte));
        }
    }

    #[test]
    fn byte_bpe_encodes_and_decodes_text() {
        let tokenizer = tiny_tokenizer();

        let tokens = tokenizer.encode_lossy("Hello", false).unwrap();
        assert_eq!(tokens, vec![7]);
        assert_eq!(tokenizer.decode_lossy(&tokens).unwrap(), "Hello");

        let tokens = tokenizer.encode_lossy(" Hello", false).unwrap();
        assert_eq!(tokens, vec![9]);
        assert_eq!(tokenizer.decode_lossy(&tokens).unwrap(), " Hello");
    }

    #[test]
    fn added_special_tokens_encode_and_skip_on_decode() {
        let tokenizer = tiny_tokenizer();

        let tokens = tokenizer.encode_lossy("<|endoftext|>Hello", false).unwrap();
        assert_eq!(tokens, vec![10, 7]);
        assert_eq!(tokenizer.decode_lossy(&tokens).unwrap(), "Hello");
        assert_eq!(
            tokenizer.decode_lossy_with_options(&tokens, false).unwrap(),
            "<|endoftext|>Hello"
        );
        assert_eq!(tokenizer.eos_token_id(), Some(10));
    }
}
