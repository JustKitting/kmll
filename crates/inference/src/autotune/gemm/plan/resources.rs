use crate::autotune::{GemmSchedulePlan, KernelResourceUsage};

impl GemmSchedulePlan {
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
}
