#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelScheduleActionTemplate {
    pub split_factors: &'static [u32],
    pub upcast_factors: &'static [u32],
    pub unroll_factors: &'static [u32],
    pub local_tile_factors: &'static [u32],
    pub group_top_factors: &'static [u32],
    pub group_factors: &'static [u32],
    pub thread_group_factors: &'static [u32],
}

impl KernelScheduleActionTemplate {
    pub const TINYGRAD_LIKE: Self = Self {
        split_factors: &[8, 13, 16, 24, 32],
        upcast_factors: &[2, 3, 4, 5, 7],
        unroll_factors: &[4, 7],
        local_tile_factors: &[2, 3, 4, 8, 13, 16, 29, 32],
        group_top_factors: &[13, 16, 28, 29, 32, 49, 64, 256],
        group_factors: &[4, 8, 16],
        thread_group_factors: &[2, 3, 4, 5, 8, 12, 16, 24, 32, 64],
    };

    pub const INFERENCE_DEFAULT: Self = Self {
        split_factors: &[8, 13, 16, 24, 32],
        upcast_factors: &[2, 3, 4, 5, 7],
        unroll_factors: &[4, 7],
        local_tile_factors: &[2, 3, 4, 8, 13, 16, 24, 29, 32],
        group_top_factors: &[13, 16, 28, 29, 32, 49, 64, 256],
        group_factors: &[4, 8, 16],
        thread_group_factors: &[2, 3, 4, 5, 8, 12, 16, 24, 32, 64, 128, 256],
    };

    pub fn bounded_split_factors(
        self,
        extent: usize,
        max_factor: u32,
        required_factor: Option<u32>,
    ) -> Vec<u32> {
        let upper = extent.min(max_factor as usize) as u32;
        let mut factors = self
            .split_factors
            .iter()
            .copied()
            .filter(|factor| *factor <= upper)
            .collect::<Vec<_>>();
        if upper > 0 {
            factors.push(upper);
        }
        if let Some(required) = required_factor {
            factors.push(required);
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub fn exhaustive_unroll_factors(
        self,
        extent: usize,
        max_factor: u32,
        excluded_factor: Option<u32>,
    ) -> Vec<u32> {
        let upper = extent.min(max_factor as usize) as u32;
        (1..=upper)
            .filter(|factor| Some(*factor) != excluded_factor)
            .collect()
    }

    pub fn legal_upcast_factors(self, mut accepts: impl FnMut(u32) -> bool) -> Vec<u32> {
        self.upcast_factors
            .iter()
            .copied()
            .filter(|factor| accepts(*factor))
            .collect()
    }

    pub fn legal_thread_group_factors(self, mut accepts: impl FnMut(u32) -> bool) -> Vec<u32> {
        self.thread_group_factors
            .iter()
            .copied()
            .filter(|factor| accepts(*factor))
            .collect()
    }
}

pub(in crate::autotune) fn bounded_unroll_factors(
    extent: usize,
    max_factor: u32,
    excluded_factor: Option<u32>,
) -> Vec<u32> {
    KernelScheduleActionTemplate::INFERENCE_DEFAULT.exhaustive_unroll_factors(
        extent,
        max_factor,
        excluded_factor,
    )
}
