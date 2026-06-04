use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SassArchitecture {
    sm: u16,
    suffix: Option<SassArchitectureSuffix>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassArchitectureSuffix {
    A,
}

impl SassArchitecture {
    pub const fn sm(sm: u16) -> Self {
        Self { sm, suffix: None }
    }

    pub const fn sm_a(sm: u16) -> Self {
        Self {
            sm,
            suffix: Some(SassArchitectureSuffix::A),
        }
    }

    pub const fn sm_number(self) -> u16 {
        self.sm
    }

    pub const fn suffix(self) -> Option<SassArchitectureSuffix> {
        self.suffix
    }

    pub fn parse(raw: &str) -> Option<Self> {
        let raw = raw.strip_prefix("sm_").or_else(|| raw.strip_prefix("sm"))?;
        let split_at = raw
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(raw.len());
        if split_at == 0 {
            return None;
        }
        let sm = raw[..split_at].parse().ok()?;
        let suffix = match &raw[split_at..] {
            "" => None,
            "a" | "A" => Some(SassArchitectureSuffix::A),
            _ => return None,
        };
        Some(Self { sm, suffix })
    }
}

impl fmt::Display for SassArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sm{}", self.sm)?;
        match self.suffix {
            Some(SassArchitectureSuffix::A) => f.write_str("a"),
            None => Ok(()),
        }
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
