use super::*;

impl MatvecSearchProblem {
    pub(in crate::autotune::matvec::problem) fn reduce_unroll_factors(&self) -> Vec<u32> {
        bounded_unroll_factors(
            self.cols,
            Self::MAX_REDUCE_UNROLL_FACTOR,
            Some(MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL),
        )
    }

    pub(in crate::autotune::matvec::problem) fn reduce_group_top_factors(&self) -> Vec<u32> {
        Self::reduce_group_top_factors_for_plan(
            MatvecSchedulePlan::new(RowMajorWarpRows::Rows1),
            self.cols,
        )
    }

    pub(in crate::autotune::matvec::problem) fn reduce_group_top_factors_for_plan(
        plan: MatvecSchedulePlan,
        cols: usize,
    ) -> Vec<u32> {
        let min_factor = plan.normalized().reduce_unroll.max(1);
        KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .group_top_factors
            .iter()
            .copied()
            .filter(|factor| *factor >= min_factor && (*factor as usize) < cols)
            .collect()
    }

    pub(in crate::autotune::matvec::problem) fn thread_group_factors(&self) -> Vec<u32> {
        KernelScheduleActionTemplate::INFERENCE_DEFAULT.legal_thread_group_factors(|factor| {
            MatvecThreadGroup::SEARCH_LANES_PER_ROW.contains(&factor)
        })
    }

    pub(in crate::autotune::matvec::problem) fn group_factors(&self) -> Vec<u32> {
        KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .group_factors
            .iter()
            .copied()
            .filter(|factor| MatvecThreadGroup::SEARCH_LANES_PER_ROW.contains(factor))
            .collect()
    }

    pub(in crate::autotune::matvec::problem) fn thread_group_only_factors(&self) -> Vec<u32> {
        let group_factors = self.group_factors();
        self.thread_group_factors()
            .into_iter()
            .filter(|factor| !group_factors.contains(factor))
            .collect()
    }

    pub(in crate::autotune::matvec::problem) fn row_upcast_factors(&self) -> Vec<u32> {
        KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .legal_upcast_factors(|factor| MatvecRowUpcast::SEARCH_FACTORS.contains(&factor))
    }

    pub(in crate::autotune::matvec::problem) fn row_upcast_factors_for_rows(
        &self,
        rows: MatvecRowSplit,
    ) -> Vec<u32> {
        self.row_upcast_factors()
            .into_iter()
            .filter(|factor| *factor <= rows.rows_per_block())
            .collect()
    }

    pub(in crate::autotune::matvec::problem) fn local_tile_factors(&self) -> Vec<u32> {
        let upper = self.rows.min(MatvecRowSplit::MAX_ROWS_PER_BLOCK as usize) as u32;
        KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .local_tile_factors
            .iter()
            .copied()
            .filter(|factor| *factor <= upper)
            .filter(|factor| MatvecRowSplit::new(*factor).is_some())
            .collect()
    }

    pub(in crate::autotune::matvec::problem) fn local_tile_factors_for_plan(
        plan: MatvecSchedulePlan,
        rows: usize,
    ) -> Vec<u32> {
        let plan = plan.normalized();
        let upper = rows.min(MatvecRowSplit::MAX_ROWS_PER_BLOCK as usize) as u32;
        KernelScheduleActionTemplate::INFERENCE_DEFAULT
            .local_tile_factors
            .iter()
            .copied()
            .filter(|factor| *factor <= upper)
            .filter(|factor| *factor != plan.rows.rows_per_block())
            .filter(|factor| *factor >= plan.row_upcast.factor())
            .filter(|factor| MatvecRowSplit::new(*factor).is_some())
            .collect()
    }

    pub(in crate::autotune::matvec::problem) fn stride_orders_for_plan(
        plan: MatvecSchedulePlan,
    ) -> Vec<Vec<u8>> {
        let plan = plan.normalized();
        if plan.row_upcast.is_default() || !plan.loop_order.is_default() {
            Vec::new()
        } else {
            vec![MatvecLoopOrder::ReductionThenRow.action_axes().to_vec()]
        }
    }

    pub(in crate::autotune::matvec::problem) fn deferred_row_split_factors(&self) -> Vec<u32> {
        bounded_unroll_factors(self.rows, Self::MAX_ROWS_PER_BLOCK, None)
    }

    pub(in crate::autotune::matvec::problem) fn group_top_factors(&self) -> Vec<u32> {
        self.deferred_row_split_factors()
    }

    pub(in crate::autotune::matvec::problem) fn split_variants(
        &self,
    ) -> Vec<KernelAxisFactorAction> {
        let mut variants = Vec::new();
        variants.extend(RowMajorWarpRows::ALL.into_iter().map(|plan| {
            KernelAxisFactorAction::new(
                0,
                plan.rows_per_block(),
                KernelActionMaterialization::Existing,
            )
        }));
        variants
    }
}
