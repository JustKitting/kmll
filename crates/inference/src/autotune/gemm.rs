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

    fn thread_count(self) -> u32 {
        let plan = self.normalized();
        let threads_n = plan.tile.n.div_ceil(plan.n_per_thread);
        let threads_m = plan.tile.m.div_ceil(plan.m_per_thread);
        threads_m.saturating_mul(threads_n).max(1)
    }

    fn shared_memory_bytes(self) -> u32 {
        const F32_BYTES: u32 = 4;
        let plan = self.normalized();
        plan.tile
            .m
            .saturating_mul(plan.tile.k)
            .saturating_add(plan.tile.k.saturating_mul(plan.tile.n))
            .saturating_mul(F32_BYTES)
    }

    fn accumulator_elements_per_thread(self) -> u32 {
        let plan = self.normalized();
        plan.m_per_thread.saturating_mul(plan.n_per_thread).max(1)
    }

    fn load_elements_per_block(self) -> u32 {
        let plan = self.normalized();
        plan.tile
            .m
            .saturating_mul(plan.tile.k)
            .saturating_add(plan.tile.k.saturating_mul(plan.tile.n))
    }

    fn resource_usage(self) -> KernelResourceUsage {
        let accumulators = self.accumulator_elements_per_thread();
        KernelResourceUsage::new(
            self.thread_count(),
            self.shared_memory_bytes(),
            accumulators,
            accumulators,
            self.load_elements_per_block(),
        )
    }

    fn a_load_rounds(self) -> u32 {
        let plan = self.normalized();
        plan.tile
            .m
            .saturating_mul(plan.tile.k)
            .div_ceil(plan.a_load_thread_count())
    }

    fn b_load_rounds(self) -> u32 {
        let plan = self.normalized();
        plan.tile
            .k
            .saturating_mul(plan.tile.n)
            .div_ceil(plan.b_load_thread_count())
    }

    fn a_load_thread_count(self) -> u32 {
        let thread_count = self.thread_count();
        if self.a_load_thread_group == 0 {
            thread_count
        } else {
            self.a_load_thread_group.clamp(1, thread_count)
        }
    }

    fn b_load_thread_count(self) -> u32 {
        let thread_count = self.thread_count();
        if self.b_load_thread_group == 0 {
            thread_count
        } else {
            self.b_load_thread_group.clamp(1, thread_count)
        }
    }

    fn has_custom_a_load_thread_group(self) -> bool {
        self.a_load_thread_group != 0 && self.a_load_thread_count() != self.thread_count()
    }

    fn has_custom_b_load_thread_group(self) -> bool {
        self.b_load_thread_group != 0 && self.b_load_thread_count() != self.thread_count()
    }

    fn per_thread_symbol_suffix(self) -> String {
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

    fn per_thread_operation_suffix(self) -> String {
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

    fn load_unroll_symbol_suffix(self) -> String {
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

    fn load_unroll_operation_suffix(self) -> String {
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

    fn load_thread_group_symbol_suffix(self) -> String {
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

    fn load_thread_group_operation_suffix(self) -> String {
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

    fn thread_order_symbol_suffix(self) -> &'static str {
        self.normalized().thread_order.symbol_suffix()
    }

    fn thread_order_operation_suffix(self) -> &'static str {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmSearchProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_dtype: NumericKind,
    pub b_dtype: NumericKind,
    pub c_dtype: NumericKind,
    pub accumulator: NumericKind,
}

impl GemmSearchProblem {
    const EXISTING_TILE: GemmTileShape = GemmTileShape::new(16, 16, 16);
    const MAX_TILE_DIM: u32 = 32;
    const MAX_THREADS_PER_BLOCK: u32 = 1024;
    const MAX_SHARED_MEMORY_BYTES: u32 = 48 * 1024;
    const MAX_ACCUMULATOR_ELEMENTS_PER_THREAD: u32 = 16;
    const MAX_REDUCE_UNROLL_FACTOR: u32 = 32;
    const MAX_LOAD_UNROLL_FACTOR: u32 = 4;
    const LOAD_THREAD_GROUP_FACTORS: [u32; 4] = [32, 64, 128, 256];

    pub const fn f32_bf16_row_col_row(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            a_dtype: NumericKind::F32,
            b_dtype: NumericKind::Bf16,
            c_dtype: NumericKind::F32,
            accumulator: NumericKind::F32,
        }
    }

    pub fn candidate_for_tile(&self, tile: GemmTileShape) -> KernelCandidateMetadata {
        self.candidate_for_plan(GemmSchedulePlan::new(tile))
    }

    pub fn candidate_for_plan(&self, plan: GemmSchedulePlan) -> KernelCandidateMetadata {
        let plan = plan.normalized();
        let tile = plan.tile;
        let existing_tile = Self::is_existing_plan(plan);
        let a_order_suffix = plan.a_load_order.symbol_suffix();
        let b_order_suffix = plan.b_load_order.symbol_suffix();
        let per_thread_symbol_suffix = plan.per_thread_symbol_suffix();
        let per_thread_operation_suffix = plan.per_thread_operation_suffix();
        let load_unroll_symbol_suffix = plan.load_unroll_symbol_suffix();
        let load_unroll_operation_suffix = plan.load_unroll_operation_suffix();
        let load_thread_group_symbol_suffix = plan.load_thread_group_symbol_suffix();
        let load_thread_group_operation_suffix = plan.load_thread_group_operation_suffix();
        let thread_order_symbol_suffix = plan.thread_order_symbol_suffix();
        let thread_order_operation_suffix = plan.thread_order_operation_suffix();
        let symbol_hint = if plan.reduce_unroll == 1 {
            format!(
                "gemm_f32_bf16_tile_{}x{}x{}{}{}{}{}{}{}",
                tile.m,
                tile.n,
                tile.k,
                per_thread_symbol_suffix,
                load_unroll_symbol_suffix,
                load_thread_group_symbol_suffix,
                a_order_suffix,
                b_order_suffix,
                thread_order_symbol_suffix
            )
        } else {
            format!(
                "gemm_f32_bf16_tile_{}x{}x{}_u{}{}{}{}{}{}{}",
                tile.m,
                tile.n,
                tile.k,
                plan.reduce_unroll,
                per_thread_symbol_suffix,
                load_unroll_symbol_suffix,
                load_thread_group_symbol_suffix,
                a_order_suffix,
                b_order_suffix,
                thread_order_symbol_suffix
            )
        };
        let launch_kernel = if existing_tile {
            "gemm_f32_bf16_tiled_kernel".to_string()
        } else {
            symbol_hint.clone()
        };
        let launch = CudaLaunchSpec::new(
            launch_kernel,
            tile.grid_dim(self.m, self.n),
            plan.block_dim(),
            0,
        );
        let operation = TypedOperationSpec::new(
            if plan.reduce_unroll == 1 {
                format!(
                    "gemm-f32-bf16-{}x{}x{}{}{}{}{}{}{}",
                    tile.m,
                    tile.n,
                    tile.k,
                    per_thread_operation_suffix,
                    load_unroll_operation_suffix,
                    load_thread_group_operation_suffix,
                    a_order_suffix,
                    b_order_suffix,
                    thread_order_operation_suffix
                )
            } else {
                format!(
                    "gemm-f32-bf16-{}x{}x{}-u{}{}{}{}{}{}{}",
                    tile.m,
                    tile.n,
                    tile.k,
                    plan.reduce_unroll,
                    per_thread_operation_suffix,
                    load_unroll_operation_suffix,
                    load_thread_group_operation_suffix,
                    a_order_suffix,
                    b_order_suffix,
                    thread_order_operation_suffix
                )
            },
            OperationKind::Gemm,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(self.a_dtype, self.accumulator, [self.m, self.k])
                .with_layout("row-major"),
        )
        .with_input(
            TensorTypeSpec::new(self.b_dtype, self.accumulator, [self.k, self.n])
                .with_layout("column-major"),
        )
        .with_output(
            TensorTypeSpec::new(self.c_dtype, self.accumulator, [self.m, self.n])
                .with_layout("row-major"),
        )
        .with_launch(launch.clone());
        let materialization = if existing_tile {
            KernelMaterialization::Existing {
                symbol: "gemm_f32_bf16_tiled_kernel",
            }
        } else {
            KernelMaterialization::DeferredGenerated {
                symbol_hint,
                reason: "schedule descriptor has no emitted Rust CUDA kernel yet".to_string(),
            }
        };
        let mut schedule = KernelSchedule::new().with_transform(ScheduleTransform::TileGemm {
            m: tile.m,
            n: tile.n,
            k: tile.k,
        });
        if plan.reduce_unroll > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 2,
                factor: plan.reduce_unroll,
            });
        }
        if plan.m_per_thread > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Upcast {
                axis: 0,
                factor: plan.m_per_thread,
            });
        }
        if plan.n_per_thread > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Upcast {
                axis: 1,
                factor: plan.n_per_thread,
            });
        }
        if plan.a_load_unroll > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 3,
                factor: plan.a_load_unroll,
            });
        }
        if plan.b_load_unroll > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 4,
                factor: plan.b_load_unroll,
            });
        }
        if plan.has_custom_a_load_thread_group() {
            schedule = schedule.with_transform(ScheduleTransform::ThreadGroup {
                axis: 3,
                factor: plan.a_load_thread_count(),
            });
        }
        if plan.has_custom_b_load_thread_group() {
            schedule = schedule.with_transform(ScheduleTransform::ThreadGroup {
                axis: 4,
                factor: plan.b_load_thread_count(),
            });
        }
        if plan.a_load_order == GemmATileLoadOrder::MContiguous {
            schedule = schedule.with_transform(ScheduleTransform::StrideOrder { axes: vec![0, 2] });
        }
        if plan.b_load_order == GemmBTileLoadOrder::KContiguous {
            schedule = schedule.with_transform(ScheduleTransform::StrideOrder { axes: vec![2, 1] });
        }
        if plan.thread_order == GemmThreadOrder::MThenN {
            schedule = schedule.with_transform(ScheduleTransform::Swap {
                axis_a: 0,
                axis_b: 1,
            });
        }

        let mut candidate = KernelCandidateMetadata::new(
            "gemm-f32-bf16-row-col-row",
            self.axes(),
            schedule,
            "tiled-gemm-generator",
            materialization,
            launch,
            operation,
        );
        candidate.resources = Some(plan.resource_usage());
        candidate
    }

    fn axes(&self) -> Vec<KernelAxis> {
        vec![
            KernelAxis::spatial(0, "m", self.m, Some(self.k)),
            KernelAxis::spatial(1, "n", self.n, Some(1)),
            KernelAxis::reduction(2, "k", self.k, Some(1)),
        ]
    }

    fn is_existing_plan(plan: GemmSchedulePlan) -> bool {
        let plan = plan.normalized();
        plan.tile == Self::EXISTING_TILE
            && plan.reduce_unroll == 1
            && plan.m_per_thread == 1
            && plan.n_per_thread == 1
            && plan.a_load_unroll == 1
            && plan.b_load_unroll == 1
            && !plan.has_custom_a_load_thread_group()
            && !plan.has_custom_b_load_thread_group()
            && plan.a_load_order == GemmATileLoadOrder::KContiguous
            && plan.b_load_order == GemmBTileLoadOrder::TileLinear
            && plan.thread_order == GemmThreadOrder::NThenM
    }

    fn action_materialization_for_plan(plan: GemmSchedulePlan) -> KernelActionMaterialization {
        if Self::is_existing_plan(plan) {
            KernelActionMaterialization::Existing
        } else {
            KernelActionMaterialization::DeferredGenerated
        }
    }

    fn plan_within_resource_limits(plan: GemmSchedulePlan) -> bool {
        let plan = plan.normalized();
        let resources = plan.resource_usage();
        plan.tile.is_launchable_shape()
            && resources.threads_per_block <= Self::MAX_THREADS_PER_BLOCK
            && resources.shared_memory_bytes <= Self::MAX_SHARED_MEMORY_BYTES
            && resources.accumulator_elements_per_thread
                <= Self::MAX_ACCUMULATOR_ELEMENTS_PER_THREAD
    }

    fn candidate_for_checked_plan(
        &self,
        parent: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
        plan: GemmSchedulePlan,
    ) -> Option<KernelCandidateMetadata> {
        let plan = plan.normalized();
        Self::plan_within_resource_limits(plan)
            .then(|| candidate_with_action_trace(parent, action, self.candidate_for_plan(plan)))
    }

    fn reduce_unroll_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            factors.extend(Self::reduce_unroll_factors_for_tile(tile));
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn reduce_unroll_factors_for_tile(tile: GemmTileShape) -> Vec<u32> {
        bounded_unroll_factors(tile.k as usize, Self::MAX_REDUCE_UNROLL_FACTOR, Some(1))
    }

    fn m_per_thread_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            factors.extend(Self::m_per_thread_factors_for_tile(tile));
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn m_per_thread_factors_for_tile(tile: GemmTileShape) -> Vec<u32> {
        const MAX_M_PER_THREAD: u32 = 4;
        (2..=MAX_M_PER_THREAD)
            .filter(|factor| tile.m % *factor == 0)
            .collect()
    }

    fn n_per_thread_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            factors.extend(Self::n_per_thread_factors_for_tile(tile));
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn n_per_thread_factors_for_tile(tile: GemmTileShape) -> Vec<u32> {
        const MAX_N_PER_THREAD: u32 = 4;
        (2..=MAX_N_PER_THREAD)
            .filter(|factor| tile.n % *factor == 0)
            .collect()
    }

    fn per_thread_plans_for_tile(tile: GemmTileShape) -> Vec<GemmSchedulePlan> {
        let mut m_factors = vec![1];
        m_factors.extend(Self::m_per_thread_factors_for_tile(tile));
        let mut n_factors = vec![1];
        n_factors.extend(Self::n_per_thread_factors_for_tile(tile));

        let mut plans = Vec::new();
        for m_per_thread in m_factors {
            for n_per_thread in n_factors.iter().copied() {
                plans.push(
                    GemmSchedulePlan::new(tile)
                        .with_m_per_thread(m_per_thread)
                        .with_n_per_thread(n_per_thread),
                );
            }
        }
        plans
    }

    fn a_load_unroll_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            for plan in Self::per_thread_plans_for_tile(tile) {
                factors.extend(Self::a_load_unroll_factors_for_plan(plan));
            }
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn b_load_unroll_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            for plan in Self::per_thread_plans_for_tile(tile) {
                factors.extend(Self::b_load_unroll_factors_for_plan(plan));
            }
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn a_load_unroll_factors_for_plan(plan: GemmSchedulePlan) -> Vec<u32> {
        bounded_unroll_factors(
            plan.a_load_rounds() as usize,
            Self::MAX_LOAD_UNROLL_FACTOR,
            Some(1),
        )
    }

    fn b_load_unroll_factors_for_plan(plan: GemmSchedulePlan) -> Vec<u32> {
        bounded_unroll_factors(
            plan.b_load_rounds() as usize,
            Self::MAX_LOAD_UNROLL_FACTOR,
            Some(1),
        )
    }

    fn split_factors_for_axis(&self, axis: u8) -> Vec<u32> {
        match axis {
            0 => bounded_tile_factors(self.m, Self::MAX_TILE_DIM, None),
            1 => bounded_tile_factors(self.n, Self::MAX_TILE_DIM, None),
            2 => bounded_tile_factors(self.k, Self::MAX_TILE_DIM, None),
            _ => Vec::new(),
        }
    }

    fn split_action_variants(&self) -> Vec<KernelAxisFactorAction> {
        let mut variants = Vec::new();
        for axis in 0..=2 {
            variants.extend(self.split_factors_for_axis(axis).into_iter().map(|factor| {
                KernelAxisFactorAction::new(
                    axis,
                    factor,
                    KernelActionMaterialization::DeferredGenerated,
                )
            }));
        }
        variants
    }

    fn split_action_variants_for_plan(
        &self,
        plan: GemmSchedulePlan,
    ) -> Vec<KernelAxisFactorAction> {
        let mut variants = Vec::new();
        for axis in 0..=2 {
            let Some(current_factor) = plan.tile.axis_factor(axis) else {
                continue;
            };
            for factor in self.split_factors_for_axis(axis) {
                if factor == current_factor {
                    continue;
                }
                let Some(tile) = plan.tile.with_axis(axis, factor) else {
                    continue;
                };
                if !tile.is_launchable_shape() {
                    continue;
                }
                let next_plan = plan.with_tile(tile);
                if !Self::plan_within_resource_limits(next_plan) {
                    continue;
                }
                variants.push(KernelAxisFactorAction::new(
                    axis,
                    factor,
                    Self::action_materialization_for_plan(next_plan),
                ));
            }
        }
        variants
    }

    fn a_load_thread_group_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            for plan in Self::per_thread_plans_for_tile(tile) {
                factors.extend(Self::load_thread_group_factors_for_plan(plan));
            }
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn b_load_thread_group_factors(&self) -> Vec<u32> {
        self.a_load_thread_group_factors()
    }

    fn load_thread_group_factors_for_plan(plan: GemmSchedulePlan) -> Vec<u32> {
        let thread_count = plan.thread_count();
        Self::LOAD_THREAD_GROUP_FACTORS
            .into_iter()
            .filter(|factor| *factor < thread_count)
            .collect()
    }

    fn tile_shapes(&self) -> Vec<GemmTileShape> {
        let m_factors =
            bounded_tile_factors(self.m, Self::MAX_TILE_DIM, Some(Self::EXISTING_TILE.m));
        let n_factors =
            bounded_tile_factors(self.n, Self::MAX_TILE_DIM, Some(Self::EXISTING_TILE.n));
        let k_factors =
            bounded_tile_factors(self.k, Self::MAX_TILE_DIM, Some(Self::EXISTING_TILE.k));
        let mut tiles = Vec::new();
        for m in m_factors {
            for n in n_factors.iter().copied() {
                for k in k_factors.iter().copied() {
                    let tile = GemmTileShape::new(m, n, k);
                    if Self::plan_within_resource_limits(GemmSchedulePlan::new(tile)) {
                        tiles.push(tile);
                    }
                }
            }
        }
        tiles.sort_unstable_by_key(|tile| (tile.m, tile.n, tile.k));
        tiles.dedup();
        tiles
    }

    fn seed_tile_action_variants() -> Vec<KernelTile3dAction> {
        vec![KernelTile3dAction::new(
            Self::EXISTING_TILE.into(),
            KernelActionMaterialization::Existing,
        )]
    }

    fn stride_orders_for_plan(plan: GemmSchedulePlan) -> Vec<Vec<u8>> {
        let mut orders = Vec::new();
        if plan.a_load_order == GemmATileLoadOrder::KContiguous {
            orders.push(GemmATileLoadOrder::MContiguous.action_axes());
        }
        if plan.b_load_order == GemmBTileLoadOrder::TileLinear {
            orders.push(GemmBTileLoadOrder::KContiguous.action_axes());
        }
        orders
    }
}

impl KernelActionSearchProblem for GemmSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet {
        let split_variants = self.split_action_variants();
        let unroll_factors = self.reduce_unroll_factors();
        let m_per_thread_factors = self.m_per_thread_factors();
        let n_per_thread_factors = self.n_per_thread_factors();
        let a_load_unroll_factors = self.a_load_unroll_factors();
        let b_load_unroll_factors = self.b_load_unroll_factors();
        let a_load_thread_group_factors = self.a_load_thread_group_factors();
        let b_load_thread_group_factors = self.b_load_thread_group_factors();
        let stride_orders =
            Self::stride_orders_for_plan(GemmSchedulePlan::new(Self::EXISTING_TILE));
        let mut spaces = vec![
            KernelActionSpace::Split {
                variants: split_variants,
            },
            KernelActionSpace::TileGemm {
                variants: Self::seed_tile_action_variants(),
            },
            KernelActionSpace::Unroll {
                axis: 2,
                factors: unroll_factors,
            },
            KernelActionSpace::Upcast {
                axis: 0,
                factors: m_per_thread_factors,
            },
            KernelActionSpace::Upcast {
                axis: 1,
                factors: n_per_thread_factors,
            },
        ];
        if !a_load_unroll_factors.is_empty() {
            spaces.push(KernelActionSpace::Unroll {
                axis: 3,
                factors: a_load_unroll_factors,
            });
        }
        if !b_load_unroll_factors.is_empty() {
            spaces.push(KernelActionSpace::Unroll {
                axis: 4,
                factors: b_load_unroll_factors,
            });
        }
        if !a_load_thread_group_factors.is_empty() {
            spaces.push(KernelActionSpace::ThreadGroup {
                axis: 3,
                factors: a_load_thread_group_factors,
            });
        }
        if !b_load_thread_group_factors.is_empty() {
            spaces.push(KernelActionSpace::ThreadGroup {
                axis: 4,
                factors: b_load_thread_group_factors,
            });
        }
        spaces.push(KernelActionSpace::Swap {
            pairs: vec![(0, 1)],
        });
        spaces.push(KernelActionSpace::StrideOrder {
            orders: stride_orders,
        });
        KernelActionSpaceSet::new(spaces)
    }

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
        let Some(plan) = schedule_gemm_plan(&candidate.schedule) else {
            if !candidate.schedule.transforms.is_empty() {
                return KernelActionSpaceSet::default();
            }
            let split_variants =
                self.split_action_variants_for_plan(GemmSchedulePlan::new(Self::EXISTING_TILE));
            return KernelActionSpaceSet::new(vec![
                KernelActionSpace::Split {
                    variants: split_variants,
                },
                KernelActionSpace::TileGemm {
                    variants: Self::seed_tile_action_variants(),
                },
            ]);
        };

        let mut spaces = Vec::new();
        let split_variants = self.split_action_variants_for_plan(plan);
        if !split_variants.is_empty() {
            spaces.push(KernelActionSpace::Split {
                variants: split_variants,
            });
        }
        if plan.reduce_unroll == 1 {
            let factors = Self::reduce_unroll_factors_for_tile(plan.tile)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_reduce_unroll(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Unroll { axis: 2, factors });
            }
        }
        if plan.m_per_thread == 1 {
            let factors = Self::m_per_thread_factors_for_tile(plan.tile)
                .into_iter()
                .filter(|factor| Self::plan_within_resource_limits(plan.with_m_per_thread(*factor)))
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Upcast { axis: 0, factors });
            }
        }
        if plan.n_per_thread == 1 {
            let factors = Self::n_per_thread_factors_for_tile(plan.tile)
                .into_iter()
                .filter(|factor| Self::plan_within_resource_limits(plan.with_n_per_thread(*factor)))
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Upcast { axis: 1, factors });
            }
        }
        if plan.a_load_unroll == 1 {
            let factors = Self::a_load_unroll_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_a_load_unroll(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Unroll { axis: 3, factors });
            }
        }
        if plan.b_load_unroll == 1 {
            let factors = Self::b_load_unroll_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_b_load_unroll(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Unroll { axis: 4, factors });
            }
        }
        if !plan.has_custom_a_load_thread_group() {
            let factors = Self::load_thread_group_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_a_load_thread_group(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::ThreadGroup { axis: 3, factors });
            }
        }
        if !plan.has_custom_b_load_thread_group() {
            let factors = Self::load_thread_group_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_b_load_thread_group(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::ThreadGroup { axis: 4, factors });
            }
        }
        if plan.thread_order == GemmThreadOrder::NThenM
            && Self::plan_within_resource_limits(plan.with_thread_order(GemmThreadOrder::MThenN))
        {
            spaces.push(KernelActionSpace::Swap {
                pairs: vec![(0, 1)],
            });
        }
        let orders = Self::stride_orders_for_plan(plan)
            .into_iter()
            .filter(|axes| {
                GemmATileLoadOrder::from_action_axes(axes)
                    .map(|order| Self::plan_within_resource_limits(plan.with_a_load_order(order)))
                    .or_else(|| {
                        GemmBTileLoadOrder::from_action_axes(axes).map(|order| {
                            Self::plan_within_resource_limits(plan.with_b_load_order(order))
                        })
                    })
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        if !orders.is_empty() {
            spaces.push(KernelActionSpace::StrideOrder { orders });
        }
        KernelActionSpaceSet::new(spaces)
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata> {
        match action {
            KernelScheduleAction {
                op: KernelScheduleActionOp::TileGemm,
                axis: None,
                arg: KernelScheduleActionArg::Tile3d { m, n, k },
                materialization,
            } => {
                if schedule_gemm_plan(&candidate.schedule).is_some() {
                    return None;
                }
                let tile = GemmTileShape::new(*m, *n, *k);
                if !self.tile_shapes().contains(&tile) {
                    return None;
                }
                let plan = GemmSchedulePlan::new(tile);
                if *materialization != Self::action_materialization_for_plan(plan) {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan)
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Split,
                axis: Some(axis @ 0..=2),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization,
            } => {
                let plan = match schedule_gemm_plan(&candidate.schedule) {
                    Some(plan) => plan,
                    None if candidate.schedule.transforms.is_empty() => {
                        GemmSchedulePlan::new(Self::EXISTING_TILE)
                    }
                    None => return None,
                };
                let variants = self.split_action_variants_for_plan(plan);
                if !variants.contains(&KernelAxisFactorAction::new(
                    *axis,
                    *factor,
                    *materialization,
                )) {
                    return None;
                }
                let next_tile = plan.tile.with_axis(*axis, *factor)?;
                self.candidate_for_checked_plan(candidate, action, plan.with_tile(next_tile))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(2),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.reduce_unroll != 1
                    || !Self::reduce_unroll_factors_for_tile(plan.tile).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_reduce_unroll(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(3),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.a_load_unroll != 1
                    || !Self::a_load_unroll_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_a_load_unroll(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(4),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.b_load_unroll != 1
                    || !Self::b_load_unroll_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_b_load_unroll(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Upcast,
                axis: Some(0),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.m_per_thread != 1
                    || !Self::m_per_thread_factors_for_tile(plan.tile).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_m_per_thread(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Upcast,
                axis: Some(1),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.n_per_thread != 1
                    || !Self::n_per_thread_factors_for_tile(plan.tile).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_n_per_thread(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::ThreadGroup,
                axis: Some(3),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.has_custom_a_load_thread_group()
                    || !Self::load_thread_group_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(
                    candidate,
                    action,
                    plan.with_a_load_thread_group(*factor),
                )
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::ThreadGroup,
                axis: Some(4),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.has_custom_b_load_thread_group()
                    || !Self::load_thread_group_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(
                    candidate,
                    action,
                    plan.with_b_load_thread_group(*factor),
                )
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::StrideOrder,
                axis: None,
                arg: KernelScheduleActionArg::AxisOrder(axes),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if let Some(order) = GemmATileLoadOrder::from_action_axes(axes) {
                    if plan.a_load_order != GemmATileLoadOrder::KContiguous
                        || order == GemmATileLoadOrder::KContiguous
                    {
                        return None;
                    }
                    return self.candidate_for_checked_plan(
                        candidate,
                        action,
                        plan.with_a_load_order(order),
                    );
                }
                let order = GemmBTileLoadOrder::from_action_axes(axes)?;
                if plan.b_load_order != GemmBTileLoadOrder::TileLinear
                    || order == GemmBTileLoadOrder::TileLinear
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_b_load_order(order))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Swap,
                axis: None,
                arg: KernelScheduleActionArg::AxisPair { axis_a, axis_b },
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if (*axis_a, *axis_b) != (0, 1) || plan.thread_order != GemmThreadOrder::NThenM {
                    return None;
                }
                self.candidate_for_checked_plan(
                    candidate,
                    action,
                    plan.with_thread_order(GemmThreadOrder::MThenN),
                )
            }
            _ => None,
        }
    }
}

impl KernelMetadataSearchProblem for GemmSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new(
            "gemm_f32_bf16_tiled_kernel",
            TiledGemm16Plan::<RowMajor, ColumnMajor, RowMajor>::grid_dim(self.m, self.n),
            TiledGemm16Plan::<RowMajor, ColumnMajor, RowMajor>::block_dim(),
            0,
        );
        let operation = TypedOperationSpec::new(
            "gemm-f32-bf16-seed",
            OperationKind::Gemm,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(self.a_dtype, self.accumulator, [self.m, self.k])
                .with_layout("row-major"),
        )
        .with_input(
            TensorTypeSpec::new(self.b_dtype, self.accumulator, [self.k, self.n])
                .with_layout("column-major"),
        )
        .with_output(
            TensorTypeSpec::new(self.c_dtype, self.accumulator, [self.m, self.n])
                .with_layout("row-major"),
        )
        .with_launch(launch.clone());
        KernelCandidateMetadata::new(
            "gemm-f32-bf16-row-col-row",
            self.axes(),
            KernelSchedule::new(),
            "tiled-gemm-generator",
            KernelMaterialization::DeferredGenerated {
                symbol_hint: "gemm_f32_bf16_tiled_kernel".to_string(),
                reason: "seed descriptor has no concrete tile schedule yet".to_string(),
            },
            launch,
            operation,
        )
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        expand_with_schedule_actions(self, candidate)
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let plan = schedule_gemm_plan(&candidate.schedule)?;
        let tile = plan.tile;
        let m_per_thread = plan.m_per_thread.max(1);
        let n_per_thread = plan.n_per_thread.max(1);
        let tile_m = tile.m as usize;
        let tile_n = tile.n as usize;
        let tile_k = tile.k as usize;
        let padded_m = self.m.div_ceil(tile_m).checked_mul(tile_m)?;
        let padded_n = self.n.div_ceil(tile_n).checked_mul(tile_n)?;
        let padded_k = self.k.div_ceil(tile_k).checked_mul(tile_k)?;
        let padded_fma_ops = padded_m
            .checked_mul(padded_n)?
            .checked_mul(padded_k)?
            .checked_mul(2)? as f64;
        let block_count = self
            .m
            .div_ceil(tile_m)
            .checked_mul(self.n.div_ceil(tile_n))? as f64;
        let unroll = f64::from(plan.reduce_unroll.max(1));
        let per_thread_work = f64::from(m_per_thread * n_per_thread);
        let thread_count_u32 = plan.thread_count();
        let thread_count = f64::from(thread_count_u32);
        let a_load_unroll = plan.a_load_unroll.max(1);
        let b_load_unroll = plan.b_load_unroll.max(1);
        let a_load_threads = plan.a_load_thread_count();
        let b_load_threads = plan.b_load_thread_count();
        let a_load_loop_rounds = plan.a_load_rounds().div_ceil(a_load_unroll);
        let b_load_loop_rounds = plan.b_load_rounds().div_ceil(b_load_unroll);
        let loop_overhead = block_count * 4096.0 / unroll / per_thread_work.sqrt();
        let thread_overhead = block_count * thread_count * 16.0;
        let load_loop_overhead =
            block_count * f64::from(a_load_loop_rounds + b_load_loop_rounds) * 256.0;
        let average_load_threads = (a_load_threads + b_load_threads) / 2;
        let load_thread_penalty =
            block_count * f64::from(thread_count_u32.saturating_sub(average_load_threads)) * 8.0;
        let register_pressure =
            block_count * ((unroll - 1.0) * 256.0 + (per_thread_work - 1.0) * 1024.0);
        let load_unroll_pressure =
            block_count * f64::from(a_load_unroll + b_load_unroll - 2) * 128.0;
        let threads_m = tile.m.div_ceil(m_per_thread);
        let threads_n = tile.n.div_ceil(n_per_thread);
        let thread_order_penalty = match plan.thread_order {
            GemmThreadOrder::NThenM if threads_n < 16 && threads_m >= 16 => block_count * 128.0,
            GemmThreadOrder::MThenN if threads_n >= 16 => block_count * 96.0,
            _ => 0.0,
        };
        let a_load_penalty = match plan.a_load_order {
            GemmATileLoadOrder::KContiguous => block_count * 64.0,
            GemmATileLoadOrder::MContiguous => block_count * 512.0,
        };
        let b_load_penalty = match plan.b_load_order {
            GemmBTileLoadOrder::TileLinear => block_count * 512.0,
            GemmBTileLoadOrder::KContiguous => block_count * 64.0,
        };
        SearchScore::heuristic(
            padded_fma_ops
                + loop_overhead
                + thread_overhead
                + load_loop_overhead
                + load_thread_penalty
                + register_pressure
                + load_unroll_pressure
                + thread_order_penalty
                + a_load_penalty
                + b_load_penalty,
        )
    }
}
