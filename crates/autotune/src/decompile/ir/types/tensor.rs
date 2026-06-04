use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassTensorMmaShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl SassTensorMmaShape {
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    pub fn parse_compact(raw: &str) -> Option<Self> {
        if !raw.chars().all(|ch| ch.is_ascii_digit()) {
            return None;
        }
        for m in [64, 32, 16, 8] {
            for n in [32, 16, 8] {
                for k in [256, 128, 64, 32, 16, 8, 4] {
                    if raw == format!("{m}{n}{k}") {
                        return Some(Self::new(m, n, k));
                    }
                }
            }
        }
        None
    }
}

impl fmt::Display for SassTensorMmaShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m{}n{}k{}", self.m, self.n, self.k)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassTensorElementType {
    Bit,
    Fp64,
    Fp32,
    Tf32,
    F16,
    Bf16,
    Half,
    Integer,
    Fp4,
    E2M1,
    Fp6,
    E2M3,
    E3M2,
    Fp8,
    E4M3,
    E5M2,
    Raw(String),
}

impl SassTensorElementType {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "bit" => Self::Bit,
            "fp64" => Self::Fp64,
            "fp32" => Self::Fp32,
            "tf32" => Self::Tf32,
            "f16" => Self::F16,
            "bf16" => Self::Bf16,
            "half" => Self::Half,
            "integer" => Self::Integer,
            "fp4" => Self::Fp4,
            "e2m1" => Self::E2M1,
            "fp6" => Self::Fp6,
            "e2m3" => Self::E2M3,
            "e3m2" => Self::E3M2,
            "fp8" => Self::Fp8,
            "e4m3" => Self::E4M3,
            "e5m2" => Self::E5M2,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Bit => "bit",
            Self::Fp64 => "fp64",
            Self::Fp32 => "fp32",
            Self::Tf32 => "tf32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Half => "half",
            Self::Integer => "integer",
            Self::Fp4 => "fp4",
            Self::E2M1 => "e2m1",
            Self::Fp6 => "fp6",
            Self::E2M3 => "e2m3",
            Self::E3M2 => "e3m2",
            Self::Fp8 => "fp8",
            Self::E4M3 => "e4m3",
            Self::E5M2 => "e5m2",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassTensorElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SassTensorMmaSignature {
    pub shape: Option<SassTensorMmaShape>,
    pub output_type: Option<SassTensorElementType>,
    pub lhs_type: Option<SassTensorElementType>,
    pub rhs_type: Option<SassTensorElementType>,
    pub accumulator_type: Option<SassTensorElementType>,
}

impl SassTensorMmaSignature {
    pub fn primary_element_type(&self) -> Option<SassTensorElementType> {
        self.lhs_type
            .clone()
            .or_else(|| self.rhs_type.clone())
            .or_else(|| self.accumulator_type.clone())
            .or_else(|| self.output_type.clone())
    }
}

impl fmt::Display for SassTensorMmaSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "shape={},output={},lhs={},rhs={},accumulator={}",
            option_display(self.shape.as_ref()),
            option_display(self.output_type.as_ref()),
            option_display(self.lhs_type.as_ref()),
            option_display(self.rhs_type.as_ref()),
            option_display(self.accumulator_type.as_ref())
        )
    }
}

fn option_display(value: Option<&impl fmt::Display>) -> String {
    value
        .map(ToString::to_string)
        .unwrap_or_else(|| "unknown".to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassTensorScope {
    Warp,
    WarpGroup,
    Uniform,
    Raw(String),
}

impl SassTensorScope {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        match raw.as_str() {
            "warp" => Self::Warp,
            "warpgroup" => Self::WarpGroup,
            "uniform" => Self::Uniform,
            _ => Self::Raw(raw),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Warp => "warp",
            Self::WarpGroup => "warpgroup",
            Self::Uniform => "uniform",
            Self::Raw(raw) => raw,
        }
    }
}

impl fmt::Display for SassTensorScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
