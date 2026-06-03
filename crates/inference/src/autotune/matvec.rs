use super::{metadata::*, *};

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
pub struct MatvecSchedulePlan {
    pub rows: MatvecRowSplit,
    pub row_upcast: MatvecRowUpcast,
    pub reduce_unroll: u32,
    pub thread_group: MatvecThreadGroup,
}

impl MatvecSchedulePlan {
    pub const DEFAULT_REDUCE_UNROLL: u32 = 4;

    pub fn new(rows: impl Into<MatvecRowSplit>) -> Self {
        Self {
            rows: rows.into(),
            row_upcast: MatvecRowUpcast::default_upcast(),
            reduce_unroll: Self::DEFAULT_REDUCE_UNROLL,
            thread_group: MatvecThreadGroup::default_group(),
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

    pub const fn with_thread_group(mut self, thread_group: MatvecThreadGroup) -> Self {
        self.thread_group = thread_group;
        self
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
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatvecSearchProblem {
    pub rows: usize,
    pub cols: usize,
    pub input_dtype: NumericKind,
    pub weight_dtype: NumericKind,
    pub accumulator: NumericKind,
}

impl MatvecSearchProblem {
    const MAX_ROWS_PER_BLOCK: u32 = MatvecRowSplit::MAX_ROWS_PER_BLOCK;
    const MAX_REDUCE_UNROLL_FACTOR: u32 = 32;

    pub const fn bf16_row_major(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            input_dtype: NumericKind::F32,
            weight_dtype: NumericKind::Bf16,
            accumulator: NumericKind::F32,
        }
    }

    pub fn candidate_for_rows(&self, plan: RowMajorWarpRows) -> KernelCandidateMetadata {
        self.candidate_for_plan_with_materialization(
            MatvecSchedulePlan::new(plan),
            "matvec_bf16_kernel".to_string(),
            KernelMaterialization::Existing {
                symbol: "matvec_bf16_kernel",
            },
        )
    }

    pub fn generated_candidate_for_rows(&self, plan: RowMajorWarpRows) -> KernelCandidateMetadata {
        self.generated_candidate_for_plan(MatvecSchedulePlan::new(plan))
    }

    pub fn generated_candidate_for_row_split(
        &self,
        rows: MatvecRowSplit,
    ) -> KernelCandidateMetadata {
        self.generated_candidate_for_plan(MatvecSchedulePlan::new(rows))
    }

    pub fn generated_candidate_for_plan(
        &self,
        plan: MatvecSchedulePlan,
    ) -> KernelCandidateMetadata {
        let plan = plan.normalized();
        let symbol_hint = matvec_symbol_hint(plan);
        self.candidate_for_plan_with_materialization(
            plan,
            symbol_hint.clone(),
            KernelMaterialization::DeferredGenerated {
                symbol_hint,
                reason: "row split descriptor has no emitted Rust CUDA kernel yet".to_string(),
            },
        )
    }

    fn candidate_for_plan_with_materialization(
        &self,
        plan: MatvecSchedulePlan,
        launch_kernel: String,
        materialization: KernelMaterialization,
    ) -> KernelCandidateMetadata {
        let plan = plan.normalized();
        let rows = plan.rows;
        let rows_per_block = rows.rows_per_block();
        let mut schedule = KernelSchedule::new()
            .with_transform(ScheduleTransform::Split {
                axis: 0,
                factor: rows_per_block,
            })
            .with_transform(ScheduleTransform::ThreadGroup {
                axis: 0,
                factor: plan.block_threads(),
            });
        if !plan.row_upcast.is_default() {
            schedule = schedule.with_transform(ScheduleTransform::Upcast {
                axis: 0,
                factor: plan.row_upcast.factor(),
            });
        }
        if !plan.thread_group.is_default() {
            schedule = schedule.with_transform(ScheduleTransform::ThreadGroup {
                axis: 1,
                factor: plan.thread_group.lanes_per_row(),
            });
        }
        if plan.reduce_unroll != MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 1,
                factor: plan.reduce_unroll,
            });
        }
        let launch = CudaLaunchSpec::new(
            launch_kernel,
            (rows.grid_rows(self.rows), 1, 1),
            (plan.block_threads(), 1, 1),
            0,
        );
        let operation = TypedOperationSpec::new(
            matvec_operation_name(plan),
            OperationKind::Matvec,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(self.input_dtype, self.accumulator, [self.cols])
                .with_layout("contiguous"),
        )
        .with_input(
            TensorTypeSpec::new(self.weight_dtype, self.accumulator, [self.rows, self.cols])
                .with_layout("row-major"),
        )
        .with_output(
            TensorTypeSpec::new(self.accumulator, self.accumulator, [self.rows])
                .with_layout("contiguous"),
        )
        .with_launch(launch.clone());

        KernelCandidateMetadata::new(
            "matvec-bf16-row-major",
            self.axes(),
            schedule,
            "row-major-matvec-generator",
            materialization,
            launch,
            operation,
        )
    }

    fn axes(&self) -> Vec<KernelAxis> {
        vec![
            KernelAxis::spatial(0, "row", self.rows, Some(self.cols)),
            KernelAxis::reduction(1, "col", self.cols, Some(1)),
        ]
    }

    fn reduce_unroll_factors(&self) -> Vec<u32> {
        bounded_unroll_factors(
            self.cols,
            Self::MAX_REDUCE_UNROLL_FACTOR,
            Some(MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL),
        )
    }

    fn thread_group_factors(&self) -> Vec<u32> {
        KernelScheduleActionTemplate::INFERENCE_DEFAULT.legal_thread_group_factors(|factor| {
            MatvecThreadGroup::SEARCH_LANES_PER_ROW.contains(&factor)
        })
    }

    fn row_upcast_factors(&self) -> Vec<u32> {
        KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .legal_upcast_factors(|factor| MatvecRowUpcast::SEARCH_FACTORS.contains(&factor))
    }

    fn row_upcast_factors_for_rows(&self, rows: MatvecRowSplit) -> Vec<u32> {
        self.row_upcast_factors()
            .into_iter()
            .filter(|factor| *factor <= rows.rows_per_block())
            .collect()
    }

    fn deferred_row_split_factors(&self) -> Vec<u32> {
        bounded_unroll_factors(self.rows, Self::MAX_ROWS_PER_BLOCK, None)
    }

    fn split_variants(&self) -> Vec<KernelAxisFactorAction> {
        let mut variants = Vec::new();
        variants.extend(RowMajorWarpRows::ALL.into_iter().map(|plan| {
            KernelAxisFactorAction::new(
                0,
                plan.rows_per_block(),
                KernelActionMaterialization::Existing,
            )
        }));
        variants.extend(self.deferred_row_split_factors().into_iter().map(|factor| {
            KernelAxisFactorAction::new(0, factor, KernelActionMaterialization::DeferredGenerated)
        }));
        variants
    }
}

impl KernelActionSearchProblem for MatvecSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet {
        let split_variants = self.split_variants();
        let row_upcast_factors = self.row_upcast_factors();
        let unroll_factors = self.reduce_unroll_factors();
        let thread_group_factors = self.thread_group_factors();
        KernelActionSpaceSet::new(vec![
            KernelActionSpace::Split {
                variants: split_variants,
            },
            KernelActionSpace::Upcast {
                axis: 0,
                factors: row_upcast_factors,
            },
            KernelActionSpace::Unroll {
                axis: 1,
                factors: unroll_factors,
            },
            KernelActionSpace::ThreadGroup {
                axis: 1,
                factors: thread_group_factors,
            },
        ])
    }

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
        if candidate.family != "matvec-bf16-row-major" {
            return KernelActionSpaceSet::default();
        }
        if candidate.schedule.depth() == 0 {
            return KernelActionSpaceSet::new(vec![KernelActionSpace::Split {
                variants: self.split_variants(),
            }]);
        }
        let Some(plan) = schedule_matvec_plan(&candidate.schedule) else {
            return KernelActionSpaceSet::default();
        };
        if candidate.is_launchable() {
            return KernelActionSpaceSet::default();
        }
        let mut spaces = Vec::new();
        if plan.row_upcast.is_default() {
            let factors = self.row_upcast_factors_for_rows(plan.rows);
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Upcast { axis: 0, factors });
            }
        }
        if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
            spaces.push(KernelActionSpace::Unroll {
                axis: 1,
                factors: self.reduce_unroll_factors(),
            });
        }
        if plan.thread_group.is_default() {
            spaces.push(KernelActionSpace::ThreadGroup {
                axis: 1,
                factors: self.thread_group_factors(),
            });
        }
        KernelActionSpaceSet::new(spaces)
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata> {
        if candidate.family != "matvec-bf16-row-major" {
            return None;
        }
        match action {
            KernelScheduleAction {
                op: KernelScheduleActionOp::Split,
                axis: Some(0),
                arg: KernelScheduleActionArg::Factor(rows_per_block),
                materialization,
            } => {
                if candidate.schedule.depth() > 0 {
                    return None;
                }
                let next = match materialization {
                    KernelActionMaterialization::Existing => {
                        let rows = RowMajorWarpRows::from_rows_per_block(*rows_per_block)?;
                        self.candidate_for_rows(rows)
                    }
                    KernelActionMaterialization::DeferredGenerated => {
                        if !self.deferred_row_split_factors().contains(rows_per_block) {
                            return None;
                        }
                        let rows = MatvecRowSplit::new(*rows_per_block)?;
                        self.generated_candidate_for_row_split(rows)
                    }
                };
                Some(candidate_with_action_trace(candidate, action, next))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Upcast,
                axis: Some(0),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.is_launchable() {
                    return None;
                }
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if !plan.row_upcast.is_default()
                    || !self.row_upcast_factors_for_rows(plan.rows).contains(factor)
                {
                    return None;
                }
                let row_upcast = MatvecRowUpcast::new(*factor)?;
                let next = self.generated_candidate_for_plan(plan.with_row_upcast(row_upcast));
                Some(candidate_with_action_trace(candidate, action, next))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(1),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.is_launchable() || !self.reduce_unroll_factors().contains(factor) {
                    return None;
                }
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if plan.reduce_unroll != MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
                    return None;
                }
                let next = self.generated_candidate_for_plan(plan.with_reduce_unroll(*factor));
                Some(candidate_with_action_trace(candidate, action, next))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::ThreadGroup,
                axis: Some(1),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.is_launchable() || !self.thread_group_factors().contains(factor) {
                    return None;
                }
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if !plan.thread_group.is_default() {
                    return None;
                }
                let thread_group = MatvecThreadGroup::new(*factor)?;
                let next = self.generated_candidate_for_plan(plan.with_thread_group(thread_group));
                Some(candidate_with_action_trace(candidate, action, next))
            }
            _ => None,
        }
    }
}

impl KernelMetadataSearchProblem for MatvecSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new("matvec_bf16_kernel", (1, 1, 1), (32, 1, 1), 0);
        let operation = TypedOperationSpec::new(
            "matvec-bf16-row-major-seed",
            OperationKind::Matvec,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(self.input_dtype, self.accumulator, [self.cols])
                .with_layout("contiguous"),
        )
        .with_input(
            TensorTypeSpec::new(self.weight_dtype, self.accumulator, [self.rows, self.cols])
                .with_layout("row-major"),
        )
        .with_output(
            TensorTypeSpec::new(self.accumulator, self.accumulator, [self.rows])
                .with_layout("contiguous"),
        )
        .with_launch(launch.clone());
        KernelCandidateMetadata::new(
            "matvec-bf16-row-major",
            self.axes(),
            KernelSchedule::new(),
            "row-major-matvec-generator",
            KernelMaterialization::DeferredGenerated {
                symbol_hint: "matvec_bf16_kernel".to_string(),
                reason: "seed descriptor has no concrete schedule yet".to_string(),
            },
            launch,
            operation,
        )
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        expand_with_schedule_actions(self, candidate)
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let plan = schedule_matvec_plan(&candidate.schedule)?;
        let rows_per_block = plan.rows.rows_per_block() as usize;
        let lanes_per_row = plan.thread_group.lanes_per_row() as usize;
        let row_upcast = plan.row_upcast.factor() as usize;
        let blocks = self.rows.div_ceil(rows_per_block);
        let padded_rows = blocks * rows_per_block;
        let useful_fma_ops = self.rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let wasted_rows = padded_rows.saturating_sub(self.rows);
        let wasted_fma_ops = wasted_rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let block_overhead = blocks as f64 * 2048.0;
        let unroll = f64::from(plan.reduce_unroll.max(1));
        let loop_overhead = blocks as f64
            * row_upcast as f64
            * (self.cols as f64 / lanes_per_row as f64).ceil()
            * 64.0
            / unroll;
        let thread_overhead = blocks as f64 * f64::from(plan.block_threads()) * 8.0;
        let subgroup_pressure =
            blocks as f64 * (32.0_f64 / lanes_per_row as f64 - 1.0_f64).max(0.0_f64) * 256.0;
        let row_upcast_pressure =
            blocks as f64 * (row_upcast as f64 - 1.0_f64).max(0.0_f64) * 384.0;
        let register_pressure = blocks as f64 * (unroll - 1.0_f64).max(0.0_f64) * 32.0;
        let generic_runtime_penalty = if candidate.is_launchable() {
            blocks as f64 * 64.0
        } else {
            0.0
        };
        SearchScore::heuristic(
            useful_fma_ops
                + wasted_fma_ops * 8.0
                + block_overhead
                + loop_overhead
                + thread_overhead
                + subgroup_pressure
                + row_upcast_pressure
                + register_pressure
                + generic_runtime_penalty,
        )
    }
}
