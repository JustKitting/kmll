use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowMajorWarpRows {
    Rows1,
    Rows2,
    Rows4,
    Rows8,
}

impl RowMajorWarpRows {
    pub const ALL: [Self; 4] = [Self::Rows1, Self::Rows2, Self::Rows4, Self::Rows8];

    pub fn from_rows_per_block(rows_per_block: u32) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|plan| plan.rows_per_block() == rows_per_block)
    }

    pub fn rows_per_block(self) -> u32 {
        match self {
            Self::Rows1 => <RowMajorWarpRowMatvecPlan as MatvecKernelPlan>::rows_per_block(),
            Self::Rows2 => <RowMajorWarpRows2MatvecPlan as MatvecKernelPlan>::rows_per_block(),
            Self::Rows4 => <RowMajorWarpRows4MatvecPlan as MatvecKernelPlan>::rows_per_block(),
            Self::Rows8 => <RowMajorWarpRows8MatvecPlan as MatvecKernelPlan>::rows_per_block(),
        }
    }

    pub fn block_threads(self) -> u32 {
        match self {
            Self::Rows1 => <RowMajorWarpRowMatvecPlan as MatvecKernelPlan>::block_threads(),
            Self::Rows2 => <RowMajorWarpRows2MatvecPlan as MatvecKernelPlan>::block_threads(),
            Self::Rows4 => <RowMajorWarpRows4MatvecPlan as MatvecKernelPlan>::block_threads(),
            Self::Rows8 => <RowMajorWarpRows8MatvecPlan as MatvecKernelPlan>::block_threads(),
        }
    }

    pub fn grid_rows(self, rows: usize) -> u32 {
        match self {
            Self::Rows1 => <RowMajorWarpRowMatvecPlan as MatvecKernelPlan>::grid_rows(rows),
            Self::Rows2 => <RowMajorWarpRows2MatvecPlan as MatvecKernelPlan>::grid_rows(rows),
            Self::Rows4 => <RowMajorWarpRows4MatvecPlan as MatvecKernelPlan>::grid_rows(rows),
            Self::Rows8 => <RowMajorWarpRows8MatvecPlan as MatvecKernelPlan>::grid_rows(rows),
        }
    }

    pub fn plan_name(self) -> &'static str {
        match self {
            Self::Rows1 => RowMajorWarpRowMatvecPlan::NAME,
            Self::Rows2 => RowMajorWarpRows2MatvecPlan::NAME,
            Self::Rows4 => RowMajorWarpRows4MatvecPlan::NAME,
            Self::Rows8 => RowMajorWarpRows8MatvecPlan::NAME,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatvecRowSplit {
    rows_per_block: u32,
}

impl MatvecRowSplit {
    pub const LANES_PER_ROW: u32 = 32;
    pub const MAX_ROWS_PER_BLOCK: u32 = 32;

    pub fn new(rows_per_block: u32) -> Option<Self> {
        (rows_per_block > 0 && rows_per_block <= Self::MAX_ROWS_PER_BLOCK)
            .then_some(Self { rows_per_block })
    }

    pub const fn rows_per_block(self) -> u32 {
        self.rows_per_block
    }

    pub const fn block_threads(self) -> u32 {
        self.rows_per_block * Self::LANES_PER_ROW
    }

    pub fn grid_rows(self, rows: usize) -> u32 {
        (rows as u32).div_ceil(self.rows_per_block)
    }

    pub fn existing_plan(self) -> Option<RowMajorWarpRows> {
        RowMajorWarpRows::from_rows_per_block(self.rows_per_block)
    }

    pub fn plan_name(self) -> String {
        self.existing_plan()
            .map(RowMajorWarpRows::plan_name)
            .map(str::to_string)
            .unwrap_or_else(|| format!("row-major-warp-rows{}-matvec", self.rows_per_block))
    }
}

impl From<RowMajorWarpRows> for MatvecRowSplit {
    fn from(value: RowMajorWarpRows) -> Self {
        Self {
            rows_per_block: value.rows_per_block(),
        }
    }
}

impl PartialEq<RowMajorWarpRows> for MatvecRowSplit {
    fn eq(&self, other: &RowMajorWarpRows) -> bool {
        self.rows_per_block == other.rows_per_block()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatvecThreadGroup {
    lanes_per_row: u32,
}

impl MatvecThreadGroup {
    pub const DEFAULT_LANES_PER_ROW: u32 = 32;
    pub const SEARCH_LANES_PER_ROW: [u32; 4] = [2, 4, 8, 16];
    pub const SUPPORTED_LANES_PER_ROW: [u32; 5] = [2, 4, 8, 16, 32];

    pub fn new(lanes_per_row: u32) -> Option<Self> {
        Self::SUPPORTED_LANES_PER_ROW
            .contains(&lanes_per_row)
            .then_some(Self { lanes_per_row })
    }

    pub const fn default_group() -> Self {
        Self {
            lanes_per_row: Self::DEFAULT_LANES_PER_ROW,
        }
    }

    pub const fn lanes_per_row(self) -> u32 {
        self.lanes_per_row
    }

    pub const fn is_default(self) -> bool {
        self.lanes_per_row == Self::DEFAULT_LANES_PER_ROW
    }

    pub fn symbol_suffix(self) -> String {
        if self.is_default() {
            String::new()
        } else {
            format!("_tg{}", self.lanes_per_row)
        }
    }

    pub fn operation_suffix(self) -> String {
        if self.is_default() {
            String::new()
        } else {
            format!("-tg{}", self.lanes_per_row)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatvecRowUpcast {
    factor: u32,
}

impl MatvecRowUpcast {
    pub const DEFAULT_FACTOR: u32 = 1;
    pub const SEARCH_FACTORS: [u32; 3] = [2, 3, 4];

    pub fn new(factor: u32) -> Option<Self> {
        (factor >= Self::DEFAULT_FACTOR && factor <= 4).then_some(Self { factor })
    }

    pub const fn default_upcast() -> Self {
        Self {
            factor: Self::DEFAULT_FACTOR,
        }
    }

    pub const fn factor(self) -> u32 {
        self.factor
    }

    pub const fn is_default(self) -> bool {
        self.factor == Self::DEFAULT_FACTOR
    }

    pub fn symbol_suffix(self) -> String {
        if self.is_default() {
            String::new()
        } else {
            format!("_up{}", self.factor)
        }
    }

    pub fn operation_suffix(self) -> String {
        if self.is_default() {
            String::new()
        } else {
            format!("-up{}", self.factor)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatvecLoopOrder {
    RowThenReduction,
    ReductionThenRow,
}

impl MatvecLoopOrder {
    pub const DEFAULT: Self = Self::RowThenReduction;

    pub fn from_action_axes(axes: &[u8]) -> Option<Self> {
        match axes {
            [0, 1] => Some(Self::RowThenReduction),
            [1, 0] => Some(Self::ReductionThenRow),
            _ => None,
        }
    }

    pub const fn action_axes(self) -> &'static [u8] {
        match self {
            Self::RowThenReduction => &[0, 1],
            Self::ReductionThenRow => &[1, 0],
        }
    }

    pub const fn is_default(self) -> bool {
        matches!(self, Self::RowThenReduction)
    }

    pub const fn symbol_suffix(self) -> &'static str {
        match self {
            Self::RowThenReduction => "",
            Self::ReductionThenRow => "_rf",
        }
    }

    pub const fn operation_suffix(self) -> &'static str {
        match self {
            Self::RowThenReduction => "",
            Self::ReductionThenRow => "-rf",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatvecSchedulePlan {
    pub rows: MatvecRowSplit,
    pub row_upcast: MatvecRowUpcast,
    pub reduce_unroll: u32,
    pub reduce_group: u32,
    pub thread_group: MatvecThreadGroup,
    pub loop_order: MatvecLoopOrder,
}

impl MatvecSchedulePlan {
    pub const DEFAULT_REDUCE_UNROLL: u32 = 4;

    pub fn new(rows: impl Into<MatvecRowSplit>) -> Self {
        Self {
            rows: rows.into(),
            row_upcast: MatvecRowUpcast::default_upcast(),
            reduce_unroll: Self::DEFAULT_REDUCE_UNROLL,
            reduce_group: 0,
            thread_group: MatvecThreadGroup::default_group(),
            loop_order: MatvecLoopOrder::DEFAULT,
        }
    }

    pub const fn with_row_upcast(mut self, row_upcast: MatvecRowUpcast) -> Self {
        self.row_upcast = row_upcast;
        self
    }

    pub const fn with_reduce_unroll(mut self, factor: u32) -> Self {
        self.reduce_unroll = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_reduce_group(mut self, factor: u32) -> Self {
        self.reduce_group = factor;
        self
    }

    pub const fn with_thread_group(mut self, thread_group: MatvecThreadGroup) -> Self {
        self.thread_group = thread_group;
        self
    }

    pub const fn with_loop_order(mut self, loop_order: MatvecLoopOrder) -> Self {
        self.loop_order = loop_order;
        self
    }

    pub const fn reduce_group_size(self) -> u32 {
        self.reduce_group
    }

    pub const fn has_custom_reduce_group(self) -> bool {
        self.reduce_group != 0
    }

    pub const fn row_groups_per_block(self) -> u32 {
        self.rows
            .rows_per_block()
            .div_ceil(self.row_upcast.factor())
    }

    pub const fn block_threads(self) -> u32 {
        self.row_groups_per_block() * self.thread_group.lanes_per_row()
    }

    pub const fn normalized(self) -> Self {
        self.with_reduce_unroll(self.reduce_unroll)
            .with_reduce_group(self.reduce_group)
    }
}
