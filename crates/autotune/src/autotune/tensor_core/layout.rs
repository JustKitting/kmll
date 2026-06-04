use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorCoreOperandLayout {
    Row,
    Col,
}

impl TensorCoreOperandLayout {
    pub fn label(self) -> &'static str {
        match self {
            Self::Row => "row",
            Self::Col => "col",
        }
    }
}

impl fmt::Display for TensorCoreOperandLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorCoreOperandLayouts {
    pub lhs: TensorCoreOperandLayout,
    pub rhs: TensorCoreOperandLayout,
}

impl TensorCoreOperandLayouts {
    pub const ROW_COL: Self = Self {
        lhs: TensorCoreOperandLayout::Row,
        rhs: TensorCoreOperandLayout::Col,
    };
    pub const ROW_ROW: Self = Self {
        lhs: TensorCoreOperandLayout::Row,
        rhs: TensorCoreOperandLayout::Row,
    };
    pub const COL_ROW: Self = Self {
        lhs: TensorCoreOperandLayout::Col,
        rhs: TensorCoreOperandLayout::Row,
    };
    pub const COL_COL: Self = Self {
        lhs: TensorCoreOperandLayout::Col,
        rhs: TensorCoreOperandLayout::Col,
    };
}

impl fmt::Display for TensorCoreOperandLayouts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.lhs, self.rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorCoreSparsity {
    Structured2To4,
}

impl TensorCoreSparsity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Structured2To4 => "2:4",
        }
    }
}

impl fmt::Display for TensorCoreSparsity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorCoreBlockScale {
    Mx,
    Nvfp4,
    Mxfp4,
    Mxfp6,
    Mxfp8,
}

impl TensorCoreBlockScale {
    pub fn label(self) -> &'static str {
        match self {
            Self::Mx => "mx",
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp4 => "mxfp4",
            Self::Mxfp6 => "mxfp6",
            Self::Mxfp8 => "mxfp8",
        }
    }
}

impl fmt::Display for TensorCoreBlockScale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}
