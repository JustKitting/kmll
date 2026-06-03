use super::{GemmATileLoadOrder, GemmBTileLoadOrder, GemmThreadOrder, GemmTileShape};

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
}
