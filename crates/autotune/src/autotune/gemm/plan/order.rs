#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmThreadOrder {
    NThenM,
    MThenN,
}

impl GemmThreadOrder {
    pub const fn symbol_suffix(self) -> &'static str {
        match self {
            Self::NThenM => "",
            Self::MThenN => "_sw01",
        }
    }

    pub const fn operation_suffix(self) -> &'static str {
        match self {
            Self::NThenM => "",
            Self::MThenN => "-sw01",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmATileLoadOrder {
    KContiguous,
    MContiguous,
}

impl GemmATileLoadOrder {
    pub const fn symbol_suffix(self) -> &'static str {
        match self {
            Self::KContiguous => "",
            Self::MContiguous => "_am",
        }
    }

    pub fn action_axes(self) -> Vec<u8> {
        match self {
            Self::KContiguous => vec![2, 0],
            Self::MContiguous => vec![0, 2],
        }
    }

    pub fn from_action_axes(axes: &[u8]) -> Option<Self> {
        match axes {
            [2, 0] => Some(Self::KContiguous),
            [0, 2] => Some(Self::MContiguous),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmBTileLoadOrder {
    TileLinear,
    KContiguous,
}

impl GemmBTileLoadOrder {
    pub const fn symbol_suffix(self) -> &'static str {
        match self {
            Self::TileLinear => "",
            Self::KContiguous => "_bk",
        }
    }

    pub fn action_axes(self) -> Vec<u8> {
        match self {
            Self::TileLinear => vec![1, 2],
            Self::KContiguous => vec![2, 1],
        }
    }

    pub fn from_action_axes(axes: &[u8]) -> Option<Self> {
        match axes {
            [1, 2] => Some(Self::TileLinear),
            [2, 1] => Some(Self::KContiguous),
            _ => None,
        }
    }
}
