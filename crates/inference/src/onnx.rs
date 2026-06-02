use std::{
    fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
};

use crate::dtypes::DType;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone)]
pub struct OnnxFile {
    path: PathBuf,
    model: OnnxModel,
}

#[derive(Debug, Clone, Default)]
pub struct OnnxModel {
    pub ir_version: Option<i64>,
    pub producer_name: Option<String>,
    pub producer_version: Option<String>,
    pub domain: Option<String>,
    pub model_version: Option<i64>,
    pub doc_string: Option<String>,
    pub graph: OnnxGraph,
    pub opsets: Vec<OnnxOpset>,
    pub metadata: Vec<(String, String)>,
}

#[derive(Debug, Clone, Default)]
pub struct OnnxGraph {
    pub name: Option<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub nodes: Vec<OnnxNode>,
    pub initializers: Vec<OnnxTensorInfo>,
}

#[derive(Debug, Clone, Default)]
pub struct OnnxNode {
    pub op_type: Option<String>,
    pub name: Option<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxOpset {
    pub domain: String,
    pub version: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxTensorInfo {
    pub name: String,
    pub element_type: OnnxTensorElementType,
    pub shape: Vec<i64>,
    pub raw_data_offsets: Option<[usize; 2]>,
    pub external_data: Vec<(String, String)>,
    pub data_location: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxTensorElementType {
    DType(DType),
    String,
    Complex64,
    Complex128,
    Float8E4M3Fn,
    Float8E4M3FnUz,
    Float8E5M2,
    Float8E5M2FnUz,
    Uint4,
    Int4,
    Unsupported(i32),
}

impl OnnxFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let bytes = fs::read(&path)?;
        let model = parse_model(&bytes)?;
        Ok(Self { path, model })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn model(&self) -> &OnnxModel {
        &self.model
    }

    pub fn graph(&self) -> &OnnxGraph {
        &self.model.graph
    }

    pub fn initializers(&self) -> &[OnnxTensorInfo] {
        &self.model.graph.initializers
    }

    pub fn initializer(&self, name: &str) -> Result<&OnnxTensorInfo> {
        self.initializers()
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| invalid_data(format!("missing ONNX initializer {name}")).into())
    }
}

impl OnnxTensorElementType {
    pub fn from_onnx_code(code: i32) -> Self {
        match code {
            1 => Self::DType(DType::F32),
            2 => Self::DType(DType::U8),
            3 => Self::DType(DType::I8),
            4 => Self::DType(DType::U16),
            5 => Self::DType(DType::I16),
            6 => Self::DType(DType::I32),
            7 => Self::DType(DType::I64),
            8 => Self::String,
            9 => Self::DType(DType::Bool),
            10 => Self::DType(DType::F16),
            11 => Self::DType(DType::F64),
            12 => Self::DType(DType::U32),
            13 => Self::DType(DType::U64),
            14 => Self::Complex64,
            15 => Self::Complex128,
            16 => Self::DType(DType::Bf16),
            17 => Self::Float8E4M3Fn,
            18 => Self::Float8E4M3FnUz,
            19 => Self::Float8E5M2,
            20 => Self::Float8E5M2FnUz,
            21 => Self::Uint4,
            22 => Self::Int4,
            _ => Self::Unsupported(code),
        }
    }

    pub fn dtype(self) -> Option<DType> {
        match self {
            Self::DType(dtype) => Some(dtype),
            _ => None,
        }
    }

    pub fn onnx_name(self) -> &'static str {
        match self {
            Self::DType(DType::F32) => "FLOAT",
            Self::DType(DType::U8) => "UINT8",
            Self::DType(DType::I8) => "INT8",
            Self::DType(DType::U16) => "UINT16",
            Self::DType(DType::I16) => "INT16",
            Self::DType(DType::I32) => "INT32",
            Self::DType(DType::I64) => "INT64",
            Self::String => "STRING",
            Self::DType(DType::Bool) => "BOOL",
            Self::DType(DType::F16) => "FLOAT16",
            Self::DType(DType::F64) => "DOUBLE",
            Self::DType(DType::U32) => "UINT32",
            Self::DType(DType::U64) => "UINT64",
            Self::Complex64 => "COMPLEX64",
            Self::Complex128 => "COMPLEX128",
            Self::DType(DType::Bf16) => "BFLOAT16",
            Self::Float8E4M3Fn => "FLOAT8E4M3FN",
            Self::Float8E4M3FnUz => "FLOAT8E4M3FNUZ",
            Self::Float8E5M2 => "FLOAT8E5M2",
            Self::Float8E5M2FnUz => "FLOAT8E5M2FNUZ",
            Self::Uint4 => "UINT4",
            Self::Int4 => "INT4",
            Self::DType(_) => "SUPPORTED_DTYPE",
            Self::Unsupported(_) => "UNSUPPORTED",
        }
    }
}

fn parse_model(bytes: &[u8]) -> Result<OnnxModel> {
    let mut model = OnnxModel::default();
    let mut cursor = ProtoCursor::new(bytes, 0);

    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => model.ir_version = Some(field.expect_varint()? as i64),
            2 => model.producer_name = Some(field.expect_string()?),
            3 => model.producer_version = Some(field.expect_string()?),
            4 => model.domain = Some(field.expect_string()?),
            5 => model.model_version = Some(field.expect_varint()? as i64),
            6 => model.doc_string = Some(field.expect_string()?),
            7 => model.graph = parse_graph(field.expect_bytes()?)?,
            8 => model.opsets.push(parse_opset(field.expect_bytes()?)?),
            14 => model
                .metadata
                .push(parse_string_entry(field.expect_bytes()?)?),
            _ => {}
        }
    }

    Ok(model)
}

fn parse_graph(bytes: LocatedBytes<'_>) -> Result<OnnxGraph> {
    let mut graph = OnnxGraph::default();
    let mut cursor = ProtoCursor::from_located(bytes);

    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => graph.nodes.push(parse_node(field.expect_bytes()?)?),
            2 => graph.name = Some(field.expect_string()?),
            5 => graph
                .initializers
                .push(parse_tensor(field.expect_bytes()?)?),
            11 => {
                if let Some(name) = parse_value_info_name(field.expect_bytes()?)? {
                    graph.inputs.push(name);
                }
            }
            12 => {
                if let Some(name) = parse_value_info_name(field.expect_bytes()?)? {
                    graph.outputs.push(name);
                }
            }
            _ => {}
        }
    }

    Ok(graph)
}

fn parse_node(bytes: LocatedBytes<'_>) -> Result<OnnxNode> {
    let mut node = OnnxNode::default();
    let mut cursor = ProtoCursor::from_located(bytes);

    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => node.inputs.push(field.expect_string()?),
            2 => node.outputs.push(field.expect_string()?),
            3 => node.name = Some(field.expect_string()?),
            4 => node.op_type = Some(field.expect_string()?),
            _ => {}
        }
    }

    Ok(node)
}

fn parse_tensor(bytes: LocatedBytes<'_>) -> Result<OnnxTensorInfo> {
    let mut name = String::new();
    let mut element_type = OnnxTensorElementType::Unsupported(0);
    let mut shape = Vec::new();
    let mut raw_data_offsets = None;
    let mut external_data = Vec::new();
    let mut data_location = None;
    let mut cursor = ProtoCursor::from_located(bytes);

    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => shape.push(field.expect_varint()? as i64),
            2 => {
                element_type = OnnxTensorElementType::from_onnx_code(field.expect_varint()? as i32)
            }
            8 => name = field.expect_string()?,
            9 => {
                let raw = field.expect_bytes()?;
                raw_data_offsets = Some([raw.absolute_start, raw.absolute_end]);
            }
            13 => external_data.push(parse_string_entry(field.expect_bytes()?)?),
            14 => data_location = Some(field.expect_varint()? as i32),
            _ => {}
        }
    }

    Ok(OnnxTensorInfo {
        name,
        element_type,
        shape,
        raw_data_offsets,
        external_data,
        data_location,
    })
}

fn parse_opset(bytes: LocatedBytes<'_>) -> Result<OnnxOpset> {
    let mut domain = String::new();
    let mut version = 0;
    let mut cursor = ProtoCursor::from_located(bytes);

    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => domain = field.expect_string()?,
            2 => version = field.expect_varint()? as i64,
            _ => {}
        }
    }

    Ok(OnnxOpset { domain, version })
}

fn parse_value_info_name(bytes: LocatedBytes<'_>) -> Result<Option<String>> {
    let mut cursor = ProtoCursor::from_located(bytes);

    while let Some(field) = cursor.next_field()? {
        if field.number == 1 {
            return Ok(Some(field.expect_string()?));
        }
    }

    Ok(None)
}

fn parse_string_entry(bytes: LocatedBytes<'_>) -> Result<(String, String)> {
    let mut key = String::new();
    let mut value = String::new();
    let mut cursor = ProtoCursor::from_located(bytes);

    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => key = field.expect_string()?,
            2 => value = field.expect_string()?,
            _ => {}
        }
    }

    Ok((key, value))
}

#[derive(Clone, Copy)]
struct LocatedBytes<'a> {
    bytes: &'a [u8],
    absolute_start: usize,
    absolute_end: usize,
}

struct ProtoCursor<'a> {
    bytes: &'a [u8],
    absolute_base: usize,
    offset: usize,
}

struct ProtoField<'a> {
    number: u32,
    wire_type: u8,
    value: ProtoValue<'a>,
}

enum ProtoValue<'a> {
    Varint(u64),
    Fixed64,
    LengthDelimited(LocatedBytes<'a>),
    Fixed32,
}

impl<'a> ProtoCursor<'a> {
    fn new(bytes: &'a [u8], absolute_base: usize) -> Self {
        Self {
            bytes,
            absolute_base,
            offset: 0,
        }
    }

    fn from_located(bytes: LocatedBytes<'a>) -> Self {
        Self::new(bytes.bytes, bytes.absolute_start)
    }

    fn next_field(&mut self) -> Result<Option<ProtoField<'a>>> {
        if self.offset >= self.bytes.len() {
            return Ok(None);
        }

        let key = self.read_varint()?;
        let number = (key >> 3) as u32;
        let wire_type = (key & 0x07) as u8;
        let value = match wire_type {
            0 => ProtoValue::Varint(self.read_varint()?),
            1 => {
                self.skip_exact(8)?;
                ProtoValue::Fixed64
            }
            2 => {
                let len = usize::try_from(self.read_varint()?).map_err(|_| {
                    invalid_data("protobuf length-delimited field length does not fit usize")
                })?;
                let start = self.offset;
                self.skip_exact(len)?;
                ProtoValue::LengthDelimited(LocatedBytes {
                    bytes: &self.bytes[start..start + len],
                    absolute_start: self.absolute_base + start,
                    absolute_end: self.absolute_base + start + len,
                })
            }
            5 => {
                self.skip_exact(4)?;
                ProtoValue::Fixed32
            }
            _ => {
                return Err(
                    invalid_data(format!("unsupported protobuf wire type {wire_type}")).into(),
                );
            }
        };

        Ok(Some(ProtoField {
            number,
            wire_type,
            value,
        }))
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut out = 0u64;
        let mut shift = 0u32;

        loop {
            let byte = *self
                .bytes
                .get(self.offset)
                .ok_or_else(|| invalid_data("unterminated protobuf varint"))?;
            self.offset += 1;
            out |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
            if shift >= 64 {
                return Err(invalid_data("protobuf varint exceeds 64 bits").into());
            }
        }
    }

    fn skip_exact(&mut self, len: usize) -> Result<()> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| invalid_data("protobuf cursor overflow"))?;
        if end > self.bytes.len() {
            return Err(invalid_data("protobuf field overruns input").into());
        }
        self.offset = end;
        Ok(())
    }
}

impl<'a> ProtoField<'a> {
    fn expect_varint(self) -> Result<u64> {
        match self.value {
            ProtoValue::Varint(value) => Ok(value),
            _ => Err(invalid_data(format!(
                "protobuf field {} has wire type {}, expected varint",
                self.number, self.wire_type
            ))
            .into()),
        }
    }

    fn expect_bytes(self) -> Result<LocatedBytes<'a>> {
        match self.value {
            ProtoValue::LengthDelimited(bytes) => Ok(bytes),
            _ => Err(invalid_data(format!(
                "protobuf field {} has wire type {}, expected length-delimited",
                self.number, self.wire_type
            ))
            .into()),
        }
    }

    fn expect_string(self) -> Result<String> {
        let bytes = self.expect_bytes()?;
        Ok(String::from_utf8(bytes.bytes.to_vec())?)
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_graph_initializers_and_raw_offsets() {
        let tensor = message(vec![
            varint_field(1, 2),
            varint_field(1, 3),
            varint_field(2, 16),
            string_field(8, "weight"),
            bytes_field(9, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        ]);
        let graph = message(vec![
            string_field(2, "main"),
            bytes_field(5, &tensor),
            bytes_field(11, &message(vec![string_field(1, "input")])),
            bytes_field(12, &message(vec![string_field(1, "output")])),
        ]);
        let opset = message(vec![string_field(1, ""), varint_field(2, 18)]);
        let model_bytes = message(vec![
            varint_field(1, 9),
            string_field(2, "unit-test"),
            bytes_field(7, &graph),
            bytes_field(8, &opset),
        ]);

        let model = parse_model(&model_bytes).expect("ONNX model parses");
        assert_eq!(model.ir_version, Some(9));
        assert_eq!(model.producer_name.as_deref(), Some("unit-test"));
        assert_eq!(model.graph.name.as_deref(), Some("main"));
        assert_eq!(model.graph.inputs, vec!["input"]);
        assert_eq!(model.graph.outputs, vec!["output"]);
        assert_eq!(
            model.opsets,
            vec![OnnxOpset {
                domain: String::new(),
                version: 18,
            }]
        );

        let tensor = &model.graph.initializers[0];
        assert_eq!(tensor.name, "weight");
        assert_eq!(
            tensor.element_type,
            OnnxTensorElementType::DType(DType::Bf16)
        );
        assert_eq!(tensor.shape, vec![2, 3]);
        let offsets = tensor.raw_data_offsets.expect("raw data offsets");
        assert_eq!(
            &model_bytes[offsets[0]..offsets[1]],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
    }

    #[test]
    fn parses_external_initializer_metadata() {
        let external_location =
            message(vec![string_field(1, "location"), string_field(2, "w.bin")]);
        let tensor = message(vec![
            varint_field(2, 1),
            string_field(8, "external_weight"),
            bytes_field(13, &external_location),
            varint_field(14, 1),
        ]);
        let graph = message(vec![bytes_field(5, &tensor)]);
        let model_bytes = message(vec![bytes_field(7, &graph)]);

        let model = parse_model(&model_bytes).expect("ONNX model parses");
        let tensor = &model.graph.initializers[0];
        assert_eq!(tensor.name, "external_weight");
        assert_eq!(
            tensor.element_type,
            OnnxTensorElementType::DType(DType::F32)
        );
        assert_eq!(
            tensor.external_data,
            vec![("location".to_string(), "w.bin".to_string())]
        );
        assert_eq!(tensor.data_location, Some(1));
        assert_eq!(tensor.raw_data_offsets, None);
    }

    fn message(fields: Vec<Vec<u8>>) -> Vec<u8> {
        fields.into_iter().flatten().collect()
    }

    fn varint_field(number: u32, value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        push_varint(&mut out, u64::from(number << 3));
        push_varint(&mut out, value);
        out
    }

    fn string_field(number: u32, value: &str) -> Vec<u8> {
        bytes_field(number, value.as_bytes())
    }

    fn bytes_field(number: u32, value: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        push_varint(&mut out, u64::from((number << 3) | 2));
        push_varint(&mut out, value.len() as u64);
        out.extend_from_slice(value);
        out
    }

    fn push_varint(out: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            out.push((value as u8 & 0x7f) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }
}
