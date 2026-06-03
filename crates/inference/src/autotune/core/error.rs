use super::super::*;

#[derive(Debug)]
pub enum KernelGenerationError {
    UnsupportedCandidate {
        family: String,
        generator: &'static str,
    },
    UnsupportedOperation {
        name: String,
        kind: OperationKind,
        reason: String,
    },
    NoOptimizationCandidate {
        name: String,
        kind: OperationKind,
    },
    MissingTransform {
        family: String,
        transform: &'static str,
    },
    InvalidSelection {
        reason: String,
    },
    Io(io::Error),
    Json(serde_json::Error),
}

impl fmt::Display for KernelGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCandidate { family, generator } => {
                write!(
                    f,
                    "candidate family {family:?} is not supported by generator {generator}"
                )
            }
            Self::UnsupportedOperation { name, kind, reason } => {
                write!(
                    f,
                    "operation {name:?} ({}) is not supported by inference autotune: {reason}",
                    kind.label()
                )
            }
            Self::NoOptimizationCandidate { name, kind } => {
                write!(
                    f,
                    "operation {name:?} ({}) did not produce an inference autotune candidate",
                    kind.label()
                )
            }
            Self::MissingTransform { family, transform } => {
                write!(
                    f,
                    "candidate family {family:?} is missing required {transform} transform"
                )
            }
            Self::InvalidSelection { reason } => {
                write!(
                    f,
                    "kernel optimization selection metadata is invalid: {reason}"
                )
            }
            Self::Io(error) => write!(f, "kernel artifact I/O failed: {error}"),
            Self::Json(error) => write!(f, "kernel artifact manifest JSON failed: {error}"),
        }
    }
}

impl std::error::Error for KernelGenerationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for KernelGenerationError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for KernelGenerationError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
