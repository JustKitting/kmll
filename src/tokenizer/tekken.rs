use std::{
    collections::HashMap,
    fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
};

use crate::safetensors::Result;

pub struct TekkenJsonTokenizer {
    token_bytes: Vec<Option<Vec<u8>>>,
    ordinary_token_by_bytes: HashMap<Vec<u8>, u32>,
    ordinary_token_start: usize,
    special_tokens: Vec<Option<String>>,
    special_token_ids: Vec<(String, u32)>,
}

impl TekkenJsonTokenizer {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_tekken_path(default_tekken_path(model_dir))
    }

    pub fn from_tekken_path(path: impl AsRef<Path>) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let ordinary_token_start =
            parse_usize_field(&json, "default_num_special_tokens").unwrap_or(0);
        let token_bytes = parse_vocab_token_bytes(&json)?;
        let special_tokens = parse_special_tokens(&json, ordinary_token_start)?;
        let ordinary_token_by_bytes =
            build_ordinary_token_lookup(&token_bytes, ordinary_token_start);
        let mut special_token_ids: Vec<_> = special_tokens
            .iter()
            .enumerate()
            .filter_map(|(id, token)| token.as_ref().map(|token| (token.clone(), id as u32)))
            .collect();
        special_token_ids.sort_by(|(left, _), (right, _)| {
            right.len().cmp(&left.len()).then_with(|| left.cmp(right))
        });

        Ok(Self {
            token_bytes,
            ordinary_token_by_bytes,
            ordinary_token_start,
            special_tokens,
            special_token_ids,
        })
    }

    pub fn decode_lossy(&self, tokens: &[u32]) -> Result<String> {
        self.decode_lossy_with_options(tokens, true)
    }

    pub fn decode_lossy_with_options(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        let mut bytes = Vec::new();

        for &token in tokens {
            let token = token as usize;
            if token < self.ordinary_token_start {
                if skip_special {
                    continue;
                }
                let piece = self
                    .special_tokens
                    .get(token)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        invalid_data(format!("missing special tokenizer entry for token {token}"))
                    })?;
                bytes.extend_from_slice(piece.as_bytes());
            } else {
                let rank = token - self.ordinary_token_start;
                let piece = self
                    .token_bytes
                    .get(rank)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        invalid_data(format!(
                            "missing ordinary tokenizer entry for token {token} rank {rank}"
                        ))
                    })?;
                bytes.extend_from_slice(piece);
            }
        }

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn encode_lossy(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        if add_bos {
            let bos = self
                .bos_token_id()
                .ok_or_else(|| invalid_data("Tekken tokenizer is missing <s> BOS token"))?;
            tokens.push(bos);
        }

        let mut i = 0;
        while i < text.len() {
            if let Some((token, id)) = self.match_special_at(text, i) {
                tokens.push(id);
                i += token.len();
                continue;
            }

            let end = self.next_special_start(text, i).unwrap_or(text.len());
            self.encode_text_span(&text[i..end], &mut tokens)?;
            i = end;
        }

        Ok(tokens)
    }

    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens
            .iter()
            .enumerate()
            .find_map(|(id, candidate)| (candidate.as_deref() == Some(token)).then_some(id as u32))
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.special_token_id("<s>")
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.special_token_id("</s>")
    }

    pub fn pad_token_id(&self) -> Option<u32> {
        self.special_token_id("<pad>")
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

        let mut pieces: Vec<Vec<u8>> = bytes.iter().map(|byte| vec![*byte]).collect();
        while pieces.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..pieces.len() - 1 {
                let mut merged = Vec::with_capacity(pieces[i].len() + pieces[i + 1].len());
                merged.extend_from_slice(&pieces[i]);
                merged.extend_from_slice(&pieces[i + 1]);
                let Some(&token_id) = self.ordinary_token_by_bytes.get(&merged) else {
                    continue;
                };
                let rank = token_id as usize - self.ordinary_token_start;
                if best.is_none_or(|(_, best_rank)| rank < best_rank) {
                    best = Some((i, rank));
                }
            }

            let Some((merge_at, _)) = best else {
                break;
            };
            let right = pieces.remove(merge_at + 1);
            pieces[merge_at].extend_from_slice(&right);
        }

        for piece in pieces {
            let token_id = self.ordinary_token_by_bytes.get(&piece).ok_or_else(|| {
                invalid_data(format!(
                    "missing ordinary tokenizer entry for bytes {:?}",
                    String::from_utf8_lossy(&piece)
                ))
            })?;
            tokens.push(*token_id);
        }

        Ok(())
    }

    fn match_special_at(&self, text: &str, start: usize) -> Option<(String, u32)> {
        self.special_token_ids
            .iter()
            .find(|(token, _)| text[start..].starts_with(token))
            .cloned()
    }

    fn next_special_start(&self, text: &str, start: usize) -> Option<usize> {
        self.special_token_ids
            .iter()
            .filter_map(|(token, _)| {
                text[start..]
                    .find(token)
                    .map(|offset| start + offset)
                    .filter(|offset| *offset > start)
            })
            .min()
    }
}

pub fn default_tekken_path(model_dir: impl AsRef<Path>) -> PathBuf {
    model_dir.as_ref().join("tekken.json")
}

fn parse_vocab_token_bytes(json: &str) -> Result<Vec<Option<Vec<u8>>>> {
    let mut out = Vec::new();
    let mut i = 0;

    while let Some(rank_rel) = json[i..].find("\"rank\"") {
        let rank_pos = i + rank_rel;
        let rank = match parse_usize_at_field(json, rank_pos, "rank") {
            Ok(rank) => rank,
            Err(_) => {
                i = rank_pos + 1;
                continue;
            }
        };
        let token_bytes_pos = match find_field_after(json, rank_pos, "token_bytes") {
            Ok(pos) => pos,
            Err(_) if !out.is_empty() => break,
            Err(err) => return Err(err),
        };
        let token_bytes = parse_string_at_field(json, token_bytes_pos, "token_bytes")?;
        let token_bytes = decode_base64(&token_bytes)?;

        if rank >= out.len() {
            out.resize_with(rank + 1, || None);
        }
        out[rank] = Some(token_bytes);
        i = token_bytes_pos + "\"token_bytes\"".len();
    }

    if out.is_empty() {
        return Err(invalid_data("tekken.json did not contain any vocab ranks").into());
    }

    Ok(out)
}

fn parse_special_tokens(json: &str, ordinary_token_start: usize) -> Result<Vec<Option<String>>> {
    let mut out = vec![None; ordinary_token_start];
    let Some(mut i) = json.find("\"special_tokens\"") else {
        return Ok(out);
    };

    while let Some(rank_rel) = json[i..].find("\"rank\"") {
        let rank_pos = i + rank_rel;
        let rank = parse_usize_at_field(json, rank_pos, "rank")?;
        let token_str_pos = find_field_after(json, rank_pos, "token_str")?;
        let token = parse_string_at_field(json, token_str_pos, "token_str")?;
        if rank < ordinary_token_start {
            out[rank] = Some(token);
        }
        i = token_str_pos + "\"token_str\"".len();
    }

    Ok(out)
}

fn build_ordinary_token_lookup(
    token_bytes: &[Option<Vec<u8>>],
    ordinary_token_start: usize,
) -> HashMap<Vec<u8>, u32> {
    token_bytes
        .iter()
        .enumerate()
        .filter_map(|(rank, bytes)| {
            bytes
                .as_ref()
                .map(|bytes| (bytes.clone(), (rank + ordinary_token_start) as u32))
        })
        .collect()
}

pub fn tekken_pre_tokenize(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut i = 0;

    while i < text.len() {
        let start = i;
        let (ch, next) = next_char(text, i);

        if is_optional_word_prefix(ch) {
            if let Some((next_ch, after_next)) = next_char_opt(text, next) {
                if is_letterish(next_ch) {
                    i = take_word_regex_like(text, next);
                    pieces.push(&text[start..i]);
                    continue;
                }
                if ch == ' ' && is_token_punctuation(next_ch) {
                    i = take_punctuation_regex_like(text, after_next);
                    pieces.push(&text[start..i]);
                    continue;
                }
            }
        }

        if is_letterish(ch) {
            i = take_word_regex_like(text, start);
        } else if ch.is_numeric() {
            i = next;
        } else if is_token_punctuation(ch) {
            i = take_punctuation_regex_like(text, next);
        } else if ch.is_whitespace() {
            i = take_whitespace(text, next);
        } else {
            i = next;
        }
        pieces.push(&text[start..i]);
    }

    pieces
}

fn next_char(text: &str, start: usize) -> (char, usize) {
    let ch = text[start..]
        .chars()
        .next()
        .expect("next_char called at EOF");
    (ch, start + ch.len_utf8())
}

fn next_char_opt(text: &str, start: usize) -> Option<(char, usize)> {
    (start < text.len()).then(|| next_char(text, start))
}

fn take_word_regex_like(text: &str, start: usize) -> usize {
    let (first, after_first) = next_char(text, start);
    if !is_upper_letterish(first) {
        return take_lower_or_uncased_run(text, after_first);
    }

    let mut i = after_first;
    while let Some((ch, next)) = next_char_opt(text, i) {
        if !is_upper_letterish(ch) {
            break;
        }
        i = next;
    }

    while let Some((ch, next)) = next_char_opt(text, i) {
        if !is_lower_or_uncased_letterish(ch) {
            break;
        }
        i = next;
    }

    i
}

fn take_lower_or_uncased_run(text: &str, mut i: usize) -> usize {
    while let Some((ch, next)) = next_char_opt(text, i) {
        if !is_lower_or_uncased_letterish(ch) {
            break;
        }
        i = next;
    }
    i
}

fn take_punctuation_regex_like(text: &str, mut i: usize) -> usize {
    while let Some((ch, next)) = next_char_opt(text, i) {
        if !is_token_punctuation(ch) {
            break;
        }
        i = next;
    }
    while let Some((ch, next)) = next_char_opt(text, i) {
        if ch != '\r' && ch != '\n' && ch != '/' {
            break;
        }
        i = next;
    }
    i
}

fn take_whitespace(text: &str, mut i: usize) -> usize {
    while let Some((ch, next)) = next_char_opt(text, i) {
        if !ch.is_whitespace() {
            break;
        }
        i = next;
    }
    i
}

fn is_letterish(ch: char) -> bool {
    ch.is_alphabetic() || is_markish(ch)
}

fn is_optional_word_prefix(ch: char) -> bool {
    ch != '\r' && ch != '\n' && !is_letterish(ch) && !ch.is_numeric()
}

fn is_upper_letterish(ch: char) -> bool {
    ch.is_uppercase()
}

fn is_lower_or_uncased_letterish(ch: char) -> bool {
    is_letterish(ch) && !is_upper_letterish(ch)
}

fn is_token_punctuation(ch: char) -> bool {
    !ch.is_whitespace() && !is_letterish(ch) && !ch.is_numeric()
}

fn is_markish(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0300..=0x036F
            | 0x0483..=0x0489
            | 0x0591..=0x05BD
            | 0x05BF
            | 0x05C1..=0x05C2
            | 0x05C4..=0x05C5
            | 0x05C7
            | 0x0610..=0x061A
            | 0x064B..=0x065F
            | 0x0670
            | 0x06D6..=0x06DC
            | 0x06DF..=0x06E4
            | 0x06E7..=0x06E8
            | 0x06EA..=0x06ED
            | 0x0711
            | 0x0730..=0x074A
            | 0x07A6..=0x07B0
            | 0x07EB..=0x07F3
            | 0x0816..=0x0819
            | 0x081B..=0x0823
            | 0x0825..=0x0827
            | 0x0829..=0x082D
            | 0x0859..=0x085B
            | 0x08D3..=0x08E1
            | 0x08E3..=0x0903
            | 0x093A..=0x093C
            | 0x0941..=0x0948
            | 0x094D
            | 0x0951..=0x0957
            | 0x0962..=0x0963
            | 0x0981
            | 0x09BC
            | 0x09C1..=0x09C4
            | 0x09CD
            | 0x09E2..=0x09E3
            | 0x0A01..=0x0A02
            | 0x0A3C
            | 0x0A41..=0x0A42
            | 0x0A47..=0x0A48
            | 0x0A4B..=0x0A4D
            | 0x0A51
            | 0x0A70..=0x0A71
            | 0x0A75
            | 0x0A81..=0x0A82
            | 0x0ABC
            | 0x0AC1..=0x0AC5
            | 0x0AC7..=0x0AC8
            | 0x0ACD
            | 0x0AE2..=0x0AE3
            | 0x0B01
            | 0x0B3C
            | 0x0B3F
            | 0x0B41..=0x0B44
            | 0x0B4D
            | 0x0B56
            | 0x0B62..=0x0B63
            | 0x0B82
            | 0x0BC0
            | 0x0BCD
            | 0x0C00
            | 0x0C04
            | 0x0C3E..=0x0C40
            | 0x0C46..=0x0C48
            | 0x0C4A..=0x0C4D
            | 0x0C55..=0x0C56
            | 0x0C62..=0x0C63
            | 0x0C81
            | 0x0CBC
            | 0x0CBF
            | 0x0CC6
            | 0x0CCC..=0x0CCD
            | 0x0CE2..=0x0CE3
            | 0x0D00..=0x0D01
            | 0x0D3B..=0x0D3C
            | 0x0D41..=0x0D44
            | 0x0D4D
            | 0x0D62..=0x0D63
            | 0x0DCA
            | 0x0DD2..=0x0DD4
            | 0x0DD6
            | 0x0E31
            | 0x0E34..=0x0E3A
            | 0x0E47..=0x0E4E
            | 0x0EB1
            | 0x0EB4..=0x0EBC
            | 0x0EC8..=0x0ECD
            | 0x0F18..=0x0F19
            | 0x0F35
            | 0x0F37
            | 0x0F39
            | 0x0F71..=0x0F7E
            | 0x0F80..=0x0F84
            | 0x0F86..=0x0F87
            | 0x0F8D..=0x0F97
            | 0x0F99..=0x0FBC
            | 0x0FC6
            | 0x102D..=0x1030
            | 0x1032..=0x1037
            | 0x1039..=0x103A
            | 0x103D..=0x103E
            | 0x1058..=0x1059
            | 0x105E..=0x1060
            | 0x1071..=0x1074
            | 0x1082
            | 0x1085..=0x1086
            | 0x108D
            | 0x109D
            | 0x135D..=0x135F
            | 0x1712..=0x1714
            | 0x1732..=0x1734
            | 0x1752..=0x1753
            | 0x1772..=0x1773
            | 0x17B4..=0x17B5
            | 0x17B7..=0x17BD
            | 0x17C6
            | 0x17C9..=0x17D3
            | 0x17DD
            | 0x180B..=0x180D
            | 0x1885..=0x1886
            | 0x18A9
            | 0x1920..=0x1922
            | 0x1927..=0x1928
            | 0x1932
            | 0x1939..=0x193B
            | 0x1A17..=0x1A18
            | 0x1A1B
            | 0x1A56
            | 0x1A58..=0x1A5E
            | 0x1A60
            | 0x1A62
            | 0x1A65..=0x1A6C
            | 0x1A73..=0x1A7C
            | 0x1A7F
            | 0x1AB0..=0x1AFF
            | 0x1B00..=0x1B03
            | 0x1B34
            | 0x1B36..=0x1B3A
            | 0x1B3C
            | 0x1B42
            | 0x1B6B..=0x1B73
            | 0x1B80..=0x1B81
            | 0x1BA2..=0x1BA5
            | 0x1BA8..=0x1BA9
            | 0x1BAB..=0x1BAD
            | 0x1BE6
            | 0x1BE8..=0x1BE9
            | 0x1BED
            | 0x1BEF..=0x1BF1
            | 0x1C2C..=0x1C33
            | 0x1C36..=0x1C37
            | 0x1CD0..=0x1CD2
            | 0x1CD4..=0x1CE0
            | 0x1CE2..=0x1CE8
            | 0x1CED
            | 0x1CF4
            | 0x1CF8..=0x1CF9
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20FF
            | 0x2CEF..=0x2CF1
            | 0x2D7F
            | 0x2DE0..=0x2DFF
            | 0x302A..=0x302F
            | 0x3099..=0x309A
            | 0xA66F
            | 0xA674..=0xA67D
            | 0xA69E..=0xA69F
            | 0xA6F0..=0xA6F1
            | 0xA802
            | 0xA806
            | 0xA80B
            | 0xA825..=0xA826
            | 0xA82C
            | 0xA8C4..=0xA8C5
            | 0xA8E0..=0xA8F1
            | 0xA8FF
            | 0xA926..=0xA92D
            | 0xA947..=0xA951
            | 0xA980..=0xA982
            | 0xA9B3
            | 0xA9B6..=0xA9B9
            | 0xA9BC
            | 0xA9E5
            | 0xAA29..=0xAA2E
            | 0xAA31..=0xAA32
            | 0xAA35..=0xAA36
            | 0xAA43
            | 0xAA4C
            | 0xAA7C
            | 0xAAB0
            | 0xAAB2..=0xAAB4
            | 0xAAB7..=0xAAB8
            | 0xAABE..=0xAABF
            | 0xAAC1
            | 0xAAEC..=0xAAED
            | 0xAAF6
            | 0xABE5
            | 0xABE8
            | 0xABED
            | 0xFB1E
            | 0xFE00..=0xFE0F
            | 0xFE20..=0xFE2F
            | 0x101FD
            | 0x102E0
            | 0x10376..=0x1037A
            | 0x10A01..=0x10A03
            | 0x10A05..=0x10A06
            | 0x10A0C..=0x10A0F
            | 0x10A38..=0x10A3A
            | 0x10A3F
            | 0x10AE5..=0x10AE6
            | 0x10D24..=0x10D27
            | 0x10EAB..=0x10EAC
            | 0x10F46..=0x10F50
            | 0x10F82..=0x10F85
            | 0x11001
            | 0x11038..=0x11046
            | 0x11070
            | 0x11073..=0x11074
            | 0x1107F..=0x11081
            | 0x110B3..=0x110B6
            | 0x110B9..=0x110BA
            | 0x110C2
            | 0x11100..=0x11102
            | 0x11127..=0x1112B
            | 0x1112D..=0x11134
            | 0x11173
            | 0x11180..=0x11181
            | 0x111B6..=0x111BE
            | 0x111C9..=0x111CC
            | 0x111CF
            | 0x1122F..=0x11231
            | 0x11234
            | 0x11236..=0x11237
            | 0x1123E
            | 0x112DF
            | 0x112E3..=0x112EA
            | 0x11300..=0x11301
            | 0x1133B..=0x1133C
            | 0x11340
            | 0x11366..=0x1136C
            | 0x11370..=0x11374
            | 0x11438..=0x1143F
            | 0x11442..=0x11444
            | 0x11446
            | 0x1145E
            | 0x114B3..=0x114B8
            | 0x114BA
            | 0x114BF..=0x114C0
            | 0x114C2..=0x114C3
            | 0x115B2..=0x115B5
            | 0x115BC..=0x115BD
            | 0x115BF..=0x115C0
            | 0x115DC..=0x115DD
            | 0x11633..=0x1163A
            | 0x1163D
            | 0x1163F..=0x11640
            | 0x116AB
            | 0x116AD
            | 0x116B0..=0x116B5
            | 0x116B7
            | 0x1171D..=0x1171F
            | 0x11722..=0x11725
            | 0x11727..=0x1172B
            | 0x1182F..=0x11837
            | 0x11839..=0x1183A
            | 0x1193B..=0x1193C
            | 0x1193E
            | 0x11943
            | 0x119D4..=0x119D7
            | 0x119DA..=0x119DB
            | 0x119E0
            | 0x11A01..=0x11A0A
            | 0x11A33..=0x11A38
            | 0x11A3B..=0x11A3E
            | 0x11A47
            | 0x11A51..=0x11A56
            | 0x11A59..=0x11A5B
            | 0x11A8A..=0x11A96
            | 0x11A98..=0x11A99
            | 0x11C30..=0x11C36
            | 0x11C38..=0x11C3D
            | 0x11C3F
            | 0x11C92..=0x11CA7
            | 0x11CAA..=0x11CB0
            | 0x11CB2..=0x11CB3
            | 0x11CB5..=0x11CB6
            | 0x11D31..=0x11D36
            | 0x11D3A
            | 0x11D3C..=0x11D3D
            | 0x11D3F..=0x11D45
            | 0x11D47
            | 0x11D90..=0x11D91
            | 0x11D95
            | 0x11D97
            | 0x11EF3..=0x11EF4
            | 0x16AF0..=0x16AF4
            | 0x16B30..=0x16B36
            | 0x16F4F
            | 0x16F8F..=0x16F92
            | 0x16FE4
            | 0x1BC9D..=0x1BC9E
            | 0x1CF00..=0x1CF2D
            | 0x1CF30..=0x1CF46
            | 0x1D167..=0x1D169
            | 0x1D17B..=0x1D182
            | 0x1D185..=0x1D18B
            | 0x1D1AA..=0x1D1AD
            | 0x1D242..=0x1D244
            | 0x1DA00..=0x1DA36
            | 0x1DA3B..=0x1DA6C
            | 0x1DA75
            | 0x1DA84
            | 0x1DA9B..=0x1DA9F
            | 0x1DAA1..=0x1DAAF
            | 0x1E000..=0x1E006
            | 0x1E008..=0x1E018
            | 0x1E01B..=0x1E021
            | 0x1E023..=0x1E024
            | 0x1E026..=0x1E02A
            | 0x1E130..=0x1E136
            | 0x1E2AE
            | 0x1E2EC..=0x1E2EF
            | 0x1E8D0..=0x1E8D6
            | 0x1E944..=0x1E94A
            | 0xE0100..=0xE01EF
    )
}

fn parse_usize_field(json: &str, field: &str) -> Result<usize> {
    let pos = json
        .find(&format!("\"{field}\""))
        .ok_or_else(|| invalid_data(format!("missing JSON field {field}")))?;
    parse_usize_at_field(json, pos, field)
}

fn parse_usize_at_field(json: &str, field_pos: usize, field: &str) -> Result<usize> {
    let mut i = field_value_start(json, field_pos, field)?;
    let start = i;
    while i < json.len() && json.as_bytes()[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return Err(invalid_data(format!("expected number for JSON field {field}")).into());
    }

    Ok(json[start..i].parse()?)
}

fn parse_string_at_field(json: &str, field_pos: usize, field: &str) -> Result<String> {
    let i = field_value_start(json, field_pos, field)?;
    parse_json_string(json, i).map(|(value, _)| value)
}

fn find_field_after(json: &str, start: usize, field: &str) -> Result<usize> {
    let needle = format!("\"{field}\"");
    json[start..]
        .find(&needle)
        .map(|rel| start + rel)
        .ok_or_else(|| {
            invalid_data(format!("missing JSON field {field} after offset {start}")).into()
        })
}

fn field_value_start(json: &str, field_pos: usize, field: &str) -> Result<usize> {
    let needle_len = field.len() + 2;
    let mut i = skip_ws(json, field_pos + needle_len);
    expect_byte(json, i, b':')?;
    i = skip_ws(json, i + 1);
    Ok(i)
}

fn parse_json_string(input: &str, start: usize) -> Result<(String, usize)> {
    expect_byte(input, start, b'"')?;

    let mut out = String::new();
    let bytes = input.as_bytes();
    let mut i = start + 1;

    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Ok((out, i + 1)),
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    return Err(invalid_data("unterminated JSON string escape").into());
                }
                match bytes[i] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let end = i + 5;
                        if end > bytes.len() {
                            return Err(invalid_data("unterminated JSON unicode escape").into());
                        }
                        let value = u16::from_str_radix(&input[i + 1..end], 16)?;
                        let ch = char::from_u32(value as u32)
                            .ok_or_else(|| invalid_data("invalid JSON unicode escape"))?;
                        out.push(ch);
                        i += 4;
                    }
                    byte => {
                        return Err(invalid_data(format!(
                            "unsupported JSON string escape byte {byte}"
                        ))
                        .into());
                    }
                }
            }
            byte => out.push(byte as char),
        }
        i += 1;
    }

    Err(invalid_data("unterminated JSON string").into())
}

fn decode_base64(input: &str) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0u32;

    for byte in input.bytes() {
        if byte == b'=' {
            break;
        }
        if byte.is_ascii_whitespace() {
            continue;
        }

        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => {
                return Err(invalid_data(format!("invalid base64 byte {byte}")).into());
            }
        };

        buffer = (buffer << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buffer >> bits) & 0xff) as u8);
        }
    }

    Ok(out)
}

fn skip_ws(input: &str, mut i: usize) -> usize {
    while i < input.len() && input.as_bytes()[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

fn expect_byte(input: &str, offset: usize, expected: u8) -> Result<()> {
    match input.as_bytes().get(offset) {
        Some(actual) if *actual == expected => Ok(()),
        Some(actual) => Err(invalid_data(format!(
            "expected byte {}, got {} at offset {}",
            expected, actual, offset
        ))
        .into()),
        None => Err(invalid_data(format!(
            "expected byte {} at offset {}, got EOF",
            expected, offset
        ))
        .into()),
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}
