use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelDataFormat {
    Safetensors,
    SafetensorsIndex,
    Onnx,
    Json,
    TokenizerJson,
    PytorchBinary,
    Numpy,
    Unknown,
}

impl ModelDataFormat {
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        detect_model_data_format(path)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
            Self::SafetensorsIndex => "safetensors-index",
            Self::Onnx => "onnx",
            Self::Json => "json",
            Self::TokenizerJson => "tokenizer-json",
            Self::PytorchBinary => "pytorch-binary",
            Self::Numpy => "numpy",
            Self::Unknown => "unknown",
        }
    }
}

pub fn detect_model_data_format(path: impl AsRef<Path>) -> ModelDataFormat {
    let path = path.as_ref();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let lower_name = file_name.to_ascii_lowercase();

    match lower_name.as_str() {
        "model.safetensors.index.json" => return ModelDataFormat::SafetensorsIndex,
        "tokenizer.json" => return ModelDataFormat::TokenizerJson,
        _ => {}
    }

    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("safetensors") => ModelDataFormat::Safetensors,
        Some("onnx") => ModelDataFormat::Onnx,
        Some("json") => ModelDataFormat::Json,
        Some("bin") | Some("pt") | Some("pth") => ModelDataFormat::PytorchBinary,
        Some("npy") | Some("npz") => ModelDataFormat::Numpy,
        _ => ModelDataFormat::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_common_model_data_formats() {
        assert_eq!(
            detect_model_data_format("model.safetensors"),
            ModelDataFormat::Safetensors
        );
        assert_eq!(
            detect_model_data_format("model.safetensors.index.json"),
            ModelDataFormat::SafetensorsIndex
        );
        assert_eq!(
            detect_model_data_format("decoder.onnx"),
            ModelDataFormat::Onnx
        );
        assert_eq!(
            detect_model_data_format("tokenizer.json"),
            ModelDataFormat::TokenizerJson
        );
        assert_eq!(
            detect_model_data_format("pytorch_model.bin"),
            ModelDataFormat::PytorchBinary
        );
        assert_eq!(
            detect_model_data_format("array.npy"),
            ModelDataFormat::Numpy
        );
        assert_eq!(
            detect_model_data_format("notes.txt"),
            ModelDataFormat::Unknown
        );
    }
}
