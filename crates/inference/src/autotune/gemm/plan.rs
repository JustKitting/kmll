use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmTileShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl GemmTileShape {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    pub fn with_axis(self, axis: u8, factor: u32) -> Option<Self> {
        match axis {
            0 => Some(Self {
                m: factor,
                n: self.n,
                k: self.k,
            }),
            1 => Some(Self {
                m: self.m,
                n: factor,
                k: self.k,
            }),
            2 => Some(Self {
                m: self.m,
                n: self.n,
                k: factor,
            }),
            _ => None,
        }
    }

    pub const fn axis_factor(self, axis: u8) -> Option<u32> {
        match axis {
            0 => Some(self.m),
            1 => Some(self.n),
            2 => Some(self.k),
            _ => None,
        }
    }

    pub const fn thread_count(self) -> u32 {
        self.m * self.n
    }

    pub const fn is_launchable_shape(self) -> bool {
        self.m > 0 && self.n > 0 && self.k > 0 && self.thread_count() <= 1024
    }

    pub fn block_dim(self) -> (u32, u32, u32) {
        (self.n, self.m, 1)
    }

    pub fn grid_dim(self, m: usize, n: usize) -> (u32, u32, u32) {
        ((n as u32).div_ceil(self.n), (m as u32).div_ceil(self.m), 1)
    }
}

impl From<GemmTileShape> for KernelTile3d {
    fn from(value: GemmTileShape) -> Self {
        Self::new(value.m, value.n, value.k)
    }
}

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
pub struct GemmSchedulePlan {
    pub tile: GemmTileShape,
    pub reduce_unroll: u32,
    pub reduce_group: u32,
    pub m_per_thread: u32,
    pub n_per_thread: u32,
    pub a_load_unroll: u32,
    pub b_load_unroll: u32,
    pub a_load_thread_group: u32,
    pub b_load_thread_group: u32,
    pub a_load_order: GemmATileLoadOrder,
    pub b_load_order: GemmBTileLoadOrder,
    pub thread_order: GemmThreadOrder,
}

impl GemmSchedulePlan {
    pub const fn new(tile: GemmTileShape) -> Self {
        Self {
            tile,
            reduce_unroll: 1,
            reduce_group: 0,
            m_per_thread: 1,
            n_per_thread: 1,
            a_load_unroll: 1,
            b_load_unroll: 1,
            a_load_thread_group: 0,
            b_load_thread_group: 0,
            a_load_order: GemmATileLoadOrder::KContiguous,
            b_load_order: GemmBTileLoadOrder::TileLinear,
            thread_order: GemmThreadOrder::NThenM,
        }
    }

    pub const fn with_reduce_unroll(mut self, factor: u32) -> Self {
        self.reduce_unroll = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_reduce_group(mut self, factor: u32) -> Self {
        self.reduce_group = factor;
        self
    }

    pub const fn with_m_per_thread(mut self, factor: u32) -> Self {
        self.m_per_thread = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_n_per_thread(mut self, factor: u32) -> Self {
        self.n_per_thread = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_a_load_unroll(mut self, factor: u32) -> Self {
        self.a_load_unroll = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_b_load_unroll(mut self, factor: u32) -> Self {
        self.b_load_unroll = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_a_load_thread_group(mut self, factor: u32) -> Self {
        self.a_load_thread_group = factor;
        self
    }

    pub const fn with_b_load_thread_group(mut self, factor: u32) -> Self {
        self.b_load_thread_group = factor;
        self
    }

    pub const fn with_tile(mut self, tile: GemmTileShape) -> Self {
        self.tile = tile;
        self
    }

    pub const fn with_a_load_order(mut self, order: GemmATileLoadOrder) -> Self {
        self.a_load_order = order;
        self
    }

    pub const fn with_b_load_order(mut self, order: GemmBTileLoadOrder) -> Self {
        self.b_load_order = order;
        self
    }

    pub const fn with_thread_order(mut self, order: GemmThreadOrder) -> Self {
        self.thread_order = order;
        self
    }

    pub const fn normalized(self) -> Self {
        self.with_reduce_unroll(self.reduce_unroll)
            .with_reduce_group(self.reduce_group)
            .with_m_per_thread(self.m_per_thread)
            .with_n_per_thread(self.n_per_thread)
            .with_a_load_unroll(self.a_load_unroll)
            .with_b_load_unroll(self.b_load_unroll)
    }

    pub fn block_dim(self) -> (u32, u32, u32) {
        let plan = self.normalized();
        let threads_n = plan.tile.n.div_ceil(plan.n_per_thread);
        let threads_m = plan.tile.m.div_ceil(plan.m_per_thread);
        match plan.thread_order {
            GemmThreadOrder::NThenM => (threads_n, threads_m, 1),
            GemmThreadOrder::MThenN => (threads_m, threads_n, 1),
        }
    }

    pub(in crate::autotune) fn thread_count(self) -> u32 {
        let plan = self.normalized();
        let threads_n = plan.tile.n.div_ceil(plan.n_per_thread);
        let threads_m = plan.tile.m.div_ceil(plan.m_per_thread);
        threads_m.saturating_mul(threads_n).max(1)
    }

    pub(in crate::autotune) fn shared_memory_bytes(self) -> u32 {
        const F32_BYTES: u32 = 4;
        let plan = self.normalized();
        plan.tile
            .m
            .saturating_mul(plan.tile.k)
            .saturating_add(plan.tile.k.saturating_mul(plan.tile.n))
            .saturating_mul(F32_BYTES)
    }

    pub(in crate::autotune) fn accumulator_elements_per_thread(self) -> u32 {
        let plan = self.normalized();
        plan.m_per_thread.saturating_mul(plan.n_per_thread).max(1)
    }

    pub(in crate::autotune) fn reduce_group_size(self) -> u32 {
        let plan = self.normalized();
        if plan.reduce_group == 0 {
            plan.tile.k
        } else {
            plan.reduce_group.clamp(1, plan.tile.k)
        }
    }

    pub(in crate::autotune) fn has_custom_reduce_group(self) -> bool {
        let plan = self.normalized();
        plan.reduce_group != 0 && plan.reduce_group_size() != plan.tile.k
    }

    pub(in crate::autotune) fn load_elements_per_block(self) -> u32 {
        let plan = self.normalized();
        plan.tile
            .m
            .saturating_mul(plan.tile.k)
            .saturating_add(plan.tile.k.saturating_mul(plan.tile.n))
    }

    pub(in crate::autotune) fn resource_usage(self) -> KernelResourceUsage {
        let accumulators = self.accumulator_elements_per_thread();
        KernelResourceUsage::new(
            self.thread_count(),
            self.shared_memory_bytes(),
            accumulators,
            accumulators,
            self.load_elements_per_block(),
        )
    }

    pub(in crate::autotune) fn a_load_rounds(self) -> u32 {
        let plan = self.normalized();
        plan.tile
            .m
            .saturating_mul(plan.tile.k)
            .div_ceil(plan.a_load_thread_count())
    }

    pub(in crate::autotune) fn b_load_rounds(self) -> u32 {
        let plan = self.normalized();
        plan.tile
            .k
            .saturating_mul(plan.tile.n)
            .div_ceil(plan.b_load_thread_count())
    }

    pub(in crate::autotune) fn a_load_thread_count(self) -> u32 {
        let thread_count = self.thread_count();
        if self.a_load_thread_group == 0 {
            thread_count
        } else {
            self.a_load_thread_group.clamp(1, thread_count)
        }
    }

    pub(in crate::autotune) fn b_load_thread_count(self) -> u32 {
        let thread_count = self.thread_count();
        if self.b_load_thread_group == 0 {
            thread_count
        } else {
            self.b_load_thread_group.clamp(1, thread_count)
        }
    }

    pub(in crate::autotune) fn has_custom_a_load_thread_group(self) -> bool {
        self.a_load_thread_group != 0 && self.a_load_thread_count() != self.thread_count()
    }

    pub(in crate::autotune) fn has_custom_b_load_thread_group(self) -> bool {
        self.b_load_thread_group != 0 && self.b_load_thread_count() != self.thread_count()
    }

    pub(in crate::autotune) fn per_thread_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.m_per_thread > 1 {
            write!(&mut suffix, "_mt{}", plan.m_per_thread).expect("write to string");
        }
        if plan.n_per_thread > 1 {
            write!(&mut suffix, "_nt{}", plan.n_per_thread).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn per_thread_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.m_per_thread > 1 {
            write!(&mut suffix, "-mt{}", plan.m_per_thread).expect("write to string");
        }
        if plan.n_per_thread > 1 {
            write!(&mut suffix, "-nt{}", plan.n_per_thread).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn load_unroll_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.a_load_unroll > 1 {
            write!(&mut suffix, "_au{}", plan.a_load_unroll).expect("write to string");
        }
        if plan.b_load_unroll > 1 {
            write!(&mut suffix, "_bu{}", plan.b_load_unroll).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn load_unroll_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.a_load_unroll > 1 {
            write!(&mut suffix, "-au{}", plan.a_load_unroll).expect("write to string");
        }
        if plan.b_load_unroll > 1 {
            write!(&mut suffix, "-bu{}", plan.b_load_unroll).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn reduce_group_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.has_custom_reduce_group() {
            write!(&mut suffix, "_kg{}", plan.reduce_group_size()).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn reduce_group_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.has_custom_reduce_group() {
            write!(&mut suffix, "-kg{}", plan.reduce_group_size()).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn load_thread_group_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.has_custom_a_load_thread_group() {
            write!(&mut suffix, "_atg{}", plan.a_load_thread_count()).expect("write to string");
        }
        if plan.has_custom_b_load_thread_group() {
            write!(&mut suffix, "_btg{}", plan.b_load_thread_count()).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn load_thread_group_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.has_custom_a_load_thread_group() {
            write!(&mut suffix, "-atg{}", plan.a_load_thread_count()).expect("write to string");
        }
        if plan.has_custom_b_load_thread_group() {
            write!(&mut suffix, "-btg{}", plan.b_load_thread_count()).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn thread_order_symbol_suffix(self) -> &'static str {
        self.normalized().thread_order.symbol_suffix()
    }

    pub(in crate::autotune) fn thread_order_operation_suffix(self) -> &'static str {
        self.normalized().thread_order.operation_suffix()
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
