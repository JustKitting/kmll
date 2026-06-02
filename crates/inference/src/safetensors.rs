use std::{
    collections::HashMap,
    fs::File,
    io::{self, ErrorKind, Read},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use crate::dtypes::{Bf16, DType, TensorElement};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

impl TensorInfo {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_len(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }

    pub fn typed_element_count<T: TensorElement>(&self) -> Result<usize> {
        self.expect_dtype::<T>()?;
        Ok((self.byte_len() as usize) / tensor_element_byte_size::<T>())
    }

    pub fn expect_dtype<T: TensorElement>(&self) -> Result<()> {
        if self.dtype == T::DTYPE {
            return Ok(());
        }

        Err(invalid_data(format!(
            "tensor {} has dtype {}, expected {}",
            self.name,
            self.dtype.safetensors_name(),
            T::DTYPE.safetensors_name()
        ))
        .into())
    }
}

fn tensor_element_byte_size<T: TensorElement>() -> usize {
    T::DTYPE
        .size_in_bytes()
        .expect("TensorElement dtype must be byte-aligned")
}

pub struct SafetensorsFile {
    path: PathBuf,
    file: File,
    header_len: u64,
    data_start: u64,
    tensors: Vec<TensorInfo>,
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;

        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes);

        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)?;
        let header = String::from_utf8(header_bytes)?;
        let tensors = parse_header(&header)?;

        Ok(Self {
            path,
            file,
            header_len,
            data_start: 8 + header_len,
            tensors,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header_len(&self) -> u64 {
        self.header_len
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| invalid_data(format!("missing tensor {name}")).into())
    }

    pub fn first_existing_tensor(&self, names: &[&str]) -> Result<&TensorInfo> {
        names
            .iter()
            .find_map(|name| self.tensors.iter().find(|tensor| tensor.name == *name))
            .ok_or_else(|| invalid_data(format!("none of these tensors exist: {names:?}")).into())
    }

    pub fn read_tensor<T: TensorElement>(&self, tensor: &TensorInfo) -> Result<Vec<T>> {
        self.read_tensor_range(tensor, 0, tensor.typed_element_count::<T>()?)
    }

    pub fn read_tensor_range<T: TensorElement>(
        &self,
        tensor: &TensorInfo,
        element_start: usize,
        element_count: usize,
    ) -> Result<Vec<T>> {
        let elem_bytes = tensor_element_byte_size::<T>();
        let total_elements = tensor.typed_element_count::<T>()?;
        if element_start + element_count > total_elements {
            return Err(invalid_data(format!(
                "tensor {} range [{}..{}) exceeds {} elements",
                tensor.name,
                element_start,
                element_start + element_count,
                total_elements
            ))
            .into());
        }

        let byte_start =
            self.data_start + tensor.data_offsets[0] + element_start as u64 * elem_bytes as u64;
        let mut bytes = vec![0u8; element_count * elem_bytes];
        self.file.read_exact_at(&mut bytes, byte_start)?;

        Ok(bytes
            .chunks_exact(elem_bytes)
            .map(T::from_le_bytes)
            .collect())
    }

    pub fn read_bf16_tensor(&self, tensor: &TensorInfo) -> Result<Vec<Bf16>> {
        self.read_tensor(tensor)
    }

    pub fn read_bf16_range(
        &self,
        tensor: &TensorInfo,
        element_start: usize,
        element_count: usize,
    ) -> Result<Vec<Bf16>> {
        self.read_tensor_range(tensor, element_start, element_count)
    }
}

pub struct ModelTensor<'a> {
    file: &'a SafetensorsFile,
    tensor: &'a TensorInfo,
}

impl<'a> ModelTensor<'a> {
    pub fn info(&self) -> &'a TensorInfo {
        self.tensor
    }
}

impl std::ops::Deref for ModelTensor<'_> {
    type Target = TensorInfo;

    fn deref(&self) -> &Self::Target {
        self.tensor
    }
}

pub enum ModelWeights {
    Consolidated(SafetensorsFile),
    Sharded(ShardedSafetensors),
}

impl ModelWeights {
    pub fn open_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            return Self::open_sharded_model_dir(model_dir);
        }

        Self::open_consolidated_model_dir(model_dir)
    }

    pub fn open_consolidated_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let path = if default_consolidated_path(model_dir).exists() {
            default_consolidated_path(model_dir)
        } else {
            default_model_safetensors_path(model_dir)
        };
        Ok(Self::Consolidated(SafetensorsFile::open(path)?))
    }

    pub fn open_sharded_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let index_path = model_dir.join("model.safetensors.index.json");
        Ok(Self::Sharded(ShardedSafetensors::open(
            model_dir,
            &index_path,
        )?))
    }

    pub fn tensor(&self, name: &str) -> Result<ModelTensor<'_>> {
        self.tensor_opt(name)?
            .ok_or_else(|| invalid_data(format!("missing tensor {name}")).into())
    }

    pub fn tensor_opt(&self, name: &str) -> Result<Option<ModelTensor<'_>>> {
        match self {
            Self::Consolidated(file) => {
                Ok(resolve_file_tensor(file, name).map(|tensor| ModelTensor { file, tensor }))
            }
            Self::Sharded(sharded) => sharded.tensor_opt(name),
        }
    }

    pub fn source_path(&self) -> &Path {
        match self {
            Self::Consolidated(file) => file.path(),
            Self::Sharded(sharded) => sharded.index_path(),
        }
    }

    pub fn tensors(&self) -> Vec<&TensorInfo> {
        match self {
            Self::Consolidated(file) => file.tensors().iter().collect(),
            Self::Sharded(sharded) => sharded.tensors(),
        }
    }

    pub fn first_existing_tensor(&self, names: &[&str]) -> Result<ModelTensor<'_>> {
        for name in names {
            if let Ok(tensor) = self.tensor(name) {
                return Ok(tensor);
            }
        }

        Err(invalid_data(format!("none of these tensors exist: {names:?}")).into())
    }

    pub fn read_tensor<T: TensorElement>(&self, tensor: &ModelTensor<'_>) -> Result<Vec<T>> {
        tensor.file.read_tensor(tensor.info())
    }

    pub fn read_tensor_range<T: TensorElement>(
        &self,
        tensor: &ModelTensor<'_>,
        element_start: usize,
        element_count: usize,
    ) -> Result<Vec<T>> {
        tensor
            .file
            .read_tensor_range(tensor.info(), element_start, element_count)
    }

    pub fn read_bf16_tensor(&self, tensor: &ModelTensor<'_>) -> Result<Vec<Bf16>> {
        self.read_tensor(tensor)
    }

    pub fn read_bf16_range(
        &self,
        tensor: &ModelTensor<'_>,
        element_start: usize,
        element_count: usize,
    ) -> Result<Vec<Bf16>> {
        self.read_tensor_range(tensor, element_start, element_count)
    }
}

pub struct ShardedSafetensors {
    index_path: PathBuf,
    files: Vec<SafetensorsFile>,
    tensor_file_index: HashMap<String, usize>,
}

impl ShardedSafetensors {
    pub fn open(model_dir: &Path, index_path: &Path) -> Result<Self> {
        let index_json = std::fs::read_to_string(index_path)?;
        let weight_map = parse_weight_map(&index_json)?;
        if weight_map.is_empty() {
            return Err(invalid_data(format!(
                "safetensors index {} has an empty weight_map",
                index_path.display()
            ))
            .into());
        }

        let mut file_names = Vec::<String>::new();
        for (_, file_name) in &weight_map {
            if !file_names.iter().any(|existing| existing == file_name) {
                file_names.push(file_name.clone());
            }
        }
        file_names.sort();

        let mut file_index_by_name = HashMap::new();
        let mut files = Vec::with_capacity(file_names.len());
        for file_name in file_names {
            let file_index = files.len();
            file_index_by_name.insert(file_name.clone(), file_index);
            files.push(SafetensorsFile::open(model_dir.join(file_name))?);
        }

        let mut tensor_file_index = HashMap::with_capacity(weight_map.len());
        for (tensor_name, file_name) in weight_map {
            let file_index = *file_index_by_name.get(&file_name).ok_or_else(|| {
                invalid_data(format!(
                    "safetensors index references un-opened shard {file_name}"
                ))
            })?;
            files[file_index].tensor(&tensor_name)?;
            tensor_file_index.insert(tensor_name, file_index);
        }

        Ok(Self {
            index_path: index_path.to_path_buf(),
            files,
            tensor_file_index,
        })
    }

    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    pub fn tensors(&self) -> Vec<&TensorInfo> {
        let mut tensors = Vec::new();
        for file in &self.files {
            tensors.extend(file.tensors().iter());
        }
        tensors.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
        tensors
    }

    pub fn tensor(&self, name: &str) -> Result<ModelTensor<'_>> {
        self.tensor_opt(name)?
            .ok_or_else(|| invalid_data(format!("missing tensor {name}")).into())
    }

    pub fn tensor_opt(&self, name: &str) -> Result<Option<ModelTensor<'_>>> {
        let resolved_name = self.resolve_tensor_name(name);
        let Some(resolved_name) = resolved_name else {
            return Ok(None);
        };
        let file_index = self.tensor_file_index[resolved_name.as_str()];
        let file = &self.files[file_index];

        Ok(Some(ModelTensor {
            file,
            tensor: file.tensor(&resolved_name)?,
        }))
    }

    fn resolve_tensor_name(&self, name: &str) -> Option<String> {
        if self.tensor_file_index.contains_key(name) {
            return Some(name.to_string());
        }

        model_tensor_aliases(name)
            .into_iter()
            .find(|alias| self.tensor_file_index.contains_key(alias.as_str()))
    }
}

fn resolve_file_tensor<'a>(file: &'a SafetensorsFile, name: &str) -> Option<&'a TensorInfo> {
    file.tensors()
        .iter()
        .find(|tensor| tensor.name == name)
        .or_else(|| {
            let aliases = model_tensor_aliases(name);
            file.tensors()
                .iter()
                .find(|tensor| aliases.iter().any(|alias| alias == &tensor.name))
        })
}

pub fn default_consolidated_path(model_dir: impl AsRef<Path>) -> PathBuf {
    model_dir.as_ref().join("consolidated.safetensors")
}

pub fn default_model_safetensors_path(model_dir: impl AsRef<Path>) -> PathBuf {
    model_dir.as_ref().join("model.safetensors")
}

fn parse_weight_map(index_json: &str) -> Result<Vec<(String, String)>> {
    let mut i = skip_ws(index_json, field_value_start(index_json, "weight_map")?);
    expect_byte(index_json, i, b'{')?;
    i += 1;

    let mut entries = Vec::new();
    loop {
        i = skip_ws(index_json, i);
        if i >= index_json.len() {
            return Err(invalid_data("unterminated safetensors weight_map").into());
        }
        if index_json.as_bytes()[i] == b'}' {
            break;
        }

        let (tensor_name, next) = parse_json_string(index_json, i)?;
        i = skip_ws(index_json, next);
        expect_byte(index_json, i, b':')?;
        i = skip_ws(index_json, i + 1);
        let (file_name, next) = parse_json_string(index_json, i)?;
        entries.push((tensor_name, file_name));

        i = skip_ws(index_json, next);
        match index_json.as_bytes().get(i) {
            Some(b',') => i += 1,
            Some(b'}') => break,
            _ => {
                return Err(invalid_data("expected comma or } in safetensors weight_map").into());
            }
        }
    }

    Ok(entries)
}

pub fn model_tensor_alias(name: &str) -> Option<String> {
    model_tensor_aliases(name).into_iter().next()
}

pub fn model_tensor_aliases(name: &str) -> Vec<String> {
    let mut aliases = Vec::new();
    match name {
        "tok_embeddings.weight" => {
            aliases.push("language_model.model.embed_tokens.weight".to_string());
            aliases.push("model.embed_tokens.weight".to_string());
            aliases.push("model.language_model.embed_tokens.weight".to_string());
            return aliases;
        }
        "norm.weight" => {
            aliases.push("language_model.model.norm.weight".to_string());
            aliases.push("model.norm.weight".to_string());
            aliases.push("model.language_model.norm.weight".to_string());
            return aliases;
        }
        "output.weight" => {
            aliases.push("language_model.lm_head.weight".to_string());
            aliases.push("lm_head.weight".to_string());
            return aliases;
        }
        _ => {}
    }

    let Some(rest) = name.strip_prefix("layers.") else {
        return aliases;
    };
    let Some((layer, suffix)) = rest.split_once('.') else {
        return aliases;
    };
    let hf_suffix = match suffix {
        "attention_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attention.wq.weight" => "self_attn.q_proj.weight",
        "attention.wk.weight" => "self_attn.k_proj.weight",
        "attention.wv.weight" => "self_attn.v_proj.weight",
        "attention.wo.weight" => "self_attn.o_proj.weight",
        "attention.q_norm.weight" => "self_attn.q_norm.weight",
        "attention.k_norm.weight" => "self_attn.k_norm.weight",
        "feed_forward.w1.weight" => "mlp.gate_proj.weight",
        "feed_forward.w2.weight" => "mlp.down_proj.weight",
        "feed_forward.w3.weight" => "mlp.up_proj.weight",
        _ => return aliases,
    };

    aliases.push(format!("language_model.model.layers.{layer}.{hf_suffix}"));
    aliases.push(format!("model.layers.{layer}.{hf_suffix}"));
    aliases.push(format!("model.language_model.layers.{layer}.{hf_suffix}"));
    aliases
}

fn parse_header(header: &str) -> Result<Vec<TensorInfo>> {
    let mut tensors = Vec::new();
    let mut i = skip_ws(header, 0);

    expect_byte(header, i, b'{')?;
    i += 1;

    loop {
        i = skip_ws(header, i);
        if i >= header.len() {
            return Err(invalid_data("unterminated safetensors header").into());
        }
        if header.as_bytes()[i] == b'}' {
            break;
        }

        let (key, next) = parse_json_string(header, i)?;
        i = skip_ws(header, next);
        expect_byte(header, i, b':')?;
        i = skip_ws(header, i + 1);

        if i >= header.len() || header.as_bytes()[i] != b'{' {
            return Err(invalid_data(format!("field {key} is not an object")).into());
        }

        let object_end = find_matching_brace(header, i)?;
        let object = &header[i..=object_end];
        if key != "__metadata__" {
            tensors.push(parse_tensor_info(key, object)?);
        }

        i = skip_ws(header, object_end + 1);
        if i < header.len() && header.as_bytes()[i] == b',' {
            i += 1;
        }
    }

    Ok(tensors)
}

fn parse_tensor_info(name: String, object: &str) -> Result<TensorInfo> {
    let dtype_name = string_field(object, "dtype")?;
    let dtype = DType::from_safetensors_name(&dtype_name)
        .ok_or_else(|| invalid_data(format!("unsupported safetensors dtype {dtype_name}")))?;
    let shape = usize_array_field(object, "shape")?;
    let offsets = u64_array_field(object, "data_offsets")?;

    if offsets.len() != 2 {
        return Err(invalid_data(format!(
            "tensor {name} has {} data offsets, expected 2",
            offsets.len()
        ))
        .into());
    }

    Ok(TensorInfo {
        name,
        dtype,
        shape,
        data_offsets: [offsets[0], offsets[1]],
    })
}

fn string_field(object: &str, field: &str) -> Result<String> {
    let start = field_value_start(object, field)?;
    let start = skip_ws(object, start);
    let (value, _) = parse_json_string(object, start)?;
    Ok(value)
}

fn usize_array_field(object: &str, field: &str) -> Result<Vec<usize>> {
    let values = u64_array_field(object, field)?;
    values
        .into_iter()
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| invalid_data(format!("{field} value {value} does not fit usize")))
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn u64_array_field(object: &str, field: &str) -> Result<Vec<u64>> {
    let mut i = skip_ws(object, field_value_start(object, field)?);
    expect_byte(object, i, b'[')?;
    i += 1;

    let mut values = Vec::new();
    loop {
        i = skip_ws(object, i);
        if i >= object.len() {
            return Err(invalid_data(format!("unterminated array field {field}")).into());
        }
        if object.as_bytes()[i] == b']' {
            break;
        }

        let number_start = i;
        while i < object.len() && object.as_bytes()[i].is_ascii_digit() {
            i += 1;
        }
        if i == number_start {
            return Err(invalid_data(format!("expected number in array field {field}")).into());
        }

        values.push(object[number_start..i].parse()?);
        i = skip_ws(object, i);
        match object.as_bytes().get(i) {
            Some(b',') => i += 1,
            Some(b']') => break,
            _ => {
                return Err(
                    invalid_data(format!("expected comma or ] in array field {field}")).into(),
                );
            }
        }
    }

    Ok(values)
}

fn field_value_start(object: &str, field: &str) -> Result<usize> {
    let needle = format!("\"{field}\"");
    let field_pos = object
        .find(&needle)
        .ok_or_else(|| invalid_data(format!("missing field {field}")))?;
    let mut i = skip_ws(object, field_pos + needle.len());
    expect_byte(object, i, b':')?;
    i += 1;
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
                out.push(bytes[i] as char);
            }
            byte => out.push(byte as char),
        }
        i += 1;
    }

    Err(invalid_data("unterminated JSON string").into())
}

fn find_matching_brace(input: &str, start: usize) -> Result<usize> {
    expect_byte(input, start, b'{')?;

    let bytes = input.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (i, byte) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match *byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => {}
        }
    }

    Err(invalid_data("unterminated JSON object").into())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_tensor_alias_maps_internal_ministral_names_to_hf_names() {
        assert_eq!(
            model_tensor_alias("tok_embeddings.weight").as_deref(),
            Some("language_model.model.embed_tokens.weight")
        );
        assert_eq!(
            model_tensor_alias("norm.weight").as_deref(),
            Some("language_model.model.norm.weight")
        );
        assert_eq!(
            model_tensor_alias("output.weight").as_deref(),
            Some("language_model.lm_head.weight")
        );
        assert_eq!(
            model_tensor_alias("layers.12.attention.wq.weight").as_deref(),
            Some("language_model.model.layers.12.self_attn.q_proj.weight")
        );
        assert_eq!(
            model_tensor_alias("layers.12.attention.wk.weight").as_deref(),
            Some("language_model.model.layers.12.self_attn.k_proj.weight")
        );
        assert_eq!(
            model_tensor_alias("layers.12.attention.wv.weight").as_deref(),
            Some("language_model.model.layers.12.self_attn.v_proj.weight")
        );
        assert_eq!(
            model_tensor_alias("layers.12.attention.wo.weight").as_deref(),
            Some("language_model.model.layers.12.self_attn.o_proj.weight")
        );
        assert_eq!(
            model_tensor_alias("layers.12.attention_norm.weight").as_deref(),
            Some("language_model.model.layers.12.input_layernorm.weight")
        );
        assert_eq!(
            model_tensor_alias("layers.12.ffn_norm.weight").as_deref(),
            Some("language_model.model.layers.12.post_attention_layernorm.weight")
        );
        assert_eq!(
            model_tensor_alias("layers.12.feed_forward.w1.weight").as_deref(),
            Some("language_model.model.layers.12.mlp.gate_proj.weight")
        );
        assert_eq!(
            model_tensor_alias("layers.12.feed_forward.w2.weight").as_deref(),
            Some("language_model.model.layers.12.mlp.down_proj.weight")
        );
        assert_eq!(
            model_tensor_alias("layers.12.feed_forward.w3.weight").as_deref(),
            Some("language_model.model.layers.12.mlp.up_proj.weight")
        );
    }

    #[test]
    fn model_tensor_aliases_include_qwen_names() {
        assert!(
            model_tensor_aliases("tok_embeddings.weight")
                .iter()
                .any(|alias| alias == "model.embed_tokens.weight")
        );
        assert!(
            model_tensor_aliases("layers.2.attention.wq.weight")
                .iter()
                .any(|alias| alias == "model.layers.2.self_attn.q_proj.weight")
        );
        assert!(
            model_tensor_aliases("layers.2.attention.q_norm.weight")
                .iter()
                .any(|alias| alias == "model.layers.2.self_attn.q_norm.weight")
        );
        assert!(
            model_tensor_aliases("layers.2.attention.wq.weight")
                .iter()
                .any(|alias| alias == "model.language_model.layers.2.self_attn.q_proj.weight")
        );
        assert!(
            model_tensor_aliases("output.weight")
                .iter()
                .any(|alias| alias == "lm_head.weight")
        );
    }

    #[test]
    fn parses_safetensors_index_weight_map() {
        let index = r#"{
            "metadata": {"total_size": 10},
            "weight_map": {
                "language_model.lm_head.weight": "model-00002-of-00002.safetensors",
                "language_model.model.embed_tokens.weight": "model-00001-of-00002.safetensors"
            }
        }"#;

        let weight_map = parse_weight_map(index).expect("weight map parses");
        assert_eq!(
            weight_map,
            vec![
                (
                    "language_model.lm_head.weight".to_string(),
                    "model-00002-of-00002.safetensors".to_string()
                ),
                (
                    "language_model.model.embed_tokens.weight".to_string(),
                    "model-00001-of-00002.safetensors".to_string()
                ),
            ]
        );
    }
}
