use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KernelActionSpaceSet {
    pub spaces: Vec<KernelActionSpace>,
}

impl KernelActionSpaceSet {
    pub fn new(spaces: impl Into<Vec<KernelActionSpace>>) -> Self {
        Self {
            spaces: spaces.into(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.spaces.is_empty()
    }

    pub fn actions(&self) -> Vec<KernelScheduleAction> {
        self.spaces
            .iter()
            .flat_map(KernelActionSpace::actions)
            .collect()
    }

    pub fn optimization_spec(&self) -> ProfilingActionSpaceSet {
        ProfilingActionSpaceSet::new(
            self.spaces
                .iter()
                .map(KernelActionSpace::optimization_spec)
                .collect::<Vec<_>>(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelActionSpace {
    Split {
        variants: Vec<KernelAxisFactorAction>,
    },
    Upcast {
        axis: u8,
        factors: Vec<u32>,
    },
    Unroll {
        axis: u8,
        factors: Vec<u32>,
    },
    LocalTile {
        axis: u8,
        factors: Vec<u32>,
    },
    GroupTop {
        axis: u8,
        factors: Vec<u32>,
    },
    Group {
        axis: u8,
        factors: Vec<u32>,
    },
    ThreadGroup {
        axis: u8,
        factors: Vec<u32>,
    },
    TileGemm {
        variants: Vec<KernelTile3dAction>,
    },
    StrideOrder {
        orders: Vec<Vec<u8>>,
    },
    Swap {
        pairs: Vec<(u8, u8)>,
    },
}

impl KernelActionSpace {
    pub fn actions(&self) -> Vec<KernelScheduleAction> {
        match self {
            Self::Split { variants } => variants
                .iter()
                .map(|variant| {
                    KernelScheduleAction::split(
                        variant.axis,
                        variant.factor,
                        variant.materialization,
                    )
                })
                .collect(),
            Self::Upcast { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::upcast(*axis, factor))
                .collect(),
            Self::Unroll { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::unroll(*axis, factor))
                .collect(),
            Self::LocalTile { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::local_tile(*axis, factor))
                .collect(),
            Self::GroupTop { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::group_top(*axis, factor))
                .collect(),
            Self::Group { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::group(*axis, factor))
                .collect(),
            Self::ThreadGroup { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::thread_group(*axis, factor))
                .collect(),
            Self::TileGemm { variants } => variants
                .iter()
                .map(|variant| {
                    KernelScheduleAction::tile_gemm(
                        variant.tile.m,
                        variant.tile.n,
                        variant.tile.k,
                        variant.materialization,
                    )
                })
                .collect(),
            Self::StrideOrder { orders } => orders
                .iter()
                .cloned()
                .map(KernelScheduleAction::stride_order)
                .collect(),
            Self::Swap { pairs } => pairs
                .iter()
                .copied()
                .map(|(axis_a, axis_b)| KernelScheduleAction::swap(axis_a, axis_b))
                .collect(),
        }
    }

    fn optimization_spec(&self) -> ProfilingActionSpace {
        match self {
            Self::Split { variants } => ProfilingActionSpace::Split {
                variants: variants
                    .iter()
                    .map(|variant| {
                        ProfilingAxisFactorChoice::new(
                            variant.axis,
                            variant.factor,
                            variant.materialization,
                        )
                    })
                    .collect(),
            },
            Self::Upcast { axis, factors } => ProfilingActionSpace::Upcast {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::Unroll { axis, factors } => ProfilingActionSpace::Unroll {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::LocalTile { axis, factors } => ProfilingActionSpace::LocalTile {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::GroupTop { axis, factors } => ProfilingActionSpace::GroupTop {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::Group { axis, factors } => ProfilingActionSpace::Group {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::ThreadGroup { axis, factors } => ProfilingActionSpace::ThreadGroup {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::TileGemm { variants } => ProfilingActionSpace::TileGemm {
                variants: variants
                    .iter()
                    .map(|variant| {
                        ProfilingTile3dChoice::new(
                            variant.tile.m,
                            variant.tile.n,
                            variant.tile.k,
                            variant.materialization,
                        )
                    })
                    .collect(),
            },
            Self::StrideOrder { orders } => ProfilingActionSpace::StrideOrder {
                orders: orders.clone(),
            },
            Self::Swap { pairs } => ProfilingActionSpace::Swap {
                pairs: pairs.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelAxisFactorAction {
    pub axis: u8,
    pub factor: u32,
    pub materialization: KernelActionMaterialization,
}

impl KernelAxisFactorAction {
    pub const fn new(axis: u8, factor: u32, materialization: KernelActionMaterialization) -> Self {
        Self {
            axis,
            factor,
            materialization,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelTile3d {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl KernelTile3d {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelTile3dAction {
    pub tile: KernelTile3d,
    pub materialization: KernelActionMaterialization,
}

impl KernelTile3dAction {
    pub const fn new(tile: KernelTile3d, materialization: KernelActionMaterialization) -> Self {
        Self {
            tile,
            materialization,
        }
    }
}

pub trait KernelMetadataSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata;
    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata>;
    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore>;
}

pub trait KernelActionSearchProblem: KernelMetadataSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet;

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet;

    fn schedule_actions(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelScheduleAction> {
        self.action_spaces(candidate).actions()
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata>;
}
