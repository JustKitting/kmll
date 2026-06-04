use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassArchitecture {
    sm: u16,
}

impl SassArchitecture {
    pub const fn sm(sm: u16) -> Self {
        Self { sm }
    }

    pub const fn sm_number(self) -> u16 {
        self.sm
    }

    pub fn parse(raw: &str) -> Option<Self> {
        let sm = raw
            .strip_prefix("sm_")
            .or_else(|| raw.strip_prefix("sm"))?
            .parse()
            .ok()?;
        Some(Self::sm(sm))
    }
}

impl fmt::Display for SassArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sm{}", self.sm)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassTarget {
    Architecture(SassArchitecture),
    Raw(String),
}

impl SassTarget {
    pub fn parse(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        SassArchitecture::parse(&raw)
            .map(Self::Architecture)
            .unwrap_or(Self::Raw(raw))
    }

    pub const fn architecture(&self) -> Option<SassArchitecture> {
        match self {
            Self::Architecture(architecture) => Some(*architecture),
            Self::Raw(_) => None,
        }
    }
}

impl fmt::Display for SassTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Architecture(architecture) => write!(f, "{architecture}"),
            Self::Raw(raw) => f.write_str(raw),
        }
    }
}
