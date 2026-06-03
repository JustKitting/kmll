use super::super::*;

pub(in crate::autotune) fn schedule_rows_per_block(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Split { axis: 0, factor }
            | ScheduleTransform::GroupTop { axis: 0, factor } => Some(*factor),
            _ => None,
        })
}

pub(in crate::autotune) fn schedule_matvec_reduce_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 1, factor } => Some(*factor),
            _ => None,
        })
}

pub(in crate::autotune) fn schedule_matvec_reduce_group(schedule: &KernelSchedule) -> u32 {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::GroupTop { axis: 1, factor } => Some(*factor),
            _ => None,
        })
        .unwrap_or(0)
}

pub(in crate::autotune) fn schedule_matvec_row_upcast(
    schedule: &KernelSchedule,
) -> Option<MatvecRowUpcast> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Upcast { axis: 0, factor } => MatvecRowUpcast::new(*factor),
            _ => None,
        })
        .or_else(|| Some(MatvecRowUpcast::default_upcast()))
}

pub(in crate::autotune) fn schedule_matvec_thread_group(
    schedule: &KernelSchedule,
) -> Option<MatvecThreadGroup> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::ThreadGroup { axis: 1, factor }
            | ScheduleTransform::Group { axis: 1, factor } => MatvecThreadGroup::new(*factor),
            _ => None,
        })
        .or_else(|| Some(MatvecThreadGroup::default_group()))
}

pub(in crate::autotune) fn schedule_matvec_loop_order(
    schedule: &KernelSchedule,
) -> MatvecLoopOrder {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::StrideOrder { axes } => MatvecLoopOrder::from_action_axes(axes),
            _ => None,
        })
        .unwrap_or(MatvecLoopOrder::DEFAULT)
}

pub(in crate::autotune) fn schedule_matvec_plan(
    schedule: &KernelSchedule,
) -> Option<MatvecSchedulePlan> {
    let rows_per_block = schedule_rows_per_block(schedule)?;
    let rows = MatvecRowSplit::new(rows_per_block)?;
    Some(MatvecSchedulePlan {
        rows,
        row_upcast: schedule_matvec_row_upcast(schedule)?,
        reduce_unroll: schedule_matvec_reduce_unroll(schedule)
            .unwrap_or(MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL),
        reduce_group: schedule_matvec_reduce_group(schedule),
        thread_group: schedule_matvec_thread_group(schedule)?,
        loop_order: schedule_matvec_loop_order(schedule),
    })
}

pub(in crate::autotune) fn matvec_symbol_hint(plan: MatvecSchedulePlan) -> String {
    let plan = plan.normalized();
    let mut base = format!("matvec_bf16_rows{}", plan.rows.rows_per_block());
    base.push_str(&plan.row_upcast.symbol_suffix());
    let reduce_group_suffix = if plan.has_custom_reduce_group() {
        format!("_cg{}", plan.reduce_group_size())
    } else {
        String::new()
    };
    if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
        base.push_str(&reduce_group_suffix);
        base.push_str(&plan.thread_group.symbol_suffix());
        base.push_str(plan.loop_order.symbol_suffix());
        base
    } else {
        write!(
            &mut base,
            "_u{}{}{}{}",
            plan.reduce_unroll,
            reduce_group_suffix,
            plan.thread_group.symbol_suffix(),
            plan.loop_order.symbol_suffix()
        )
        .expect("write to string");
        base
    }
}

pub(in crate::autotune) fn matvec_operation_name(plan: MatvecSchedulePlan) -> String {
    let plan = plan.normalized();
    let plan_name = plan.rows.plan_name();
    let row_upcast_suffix = plan.row_upcast.operation_suffix();
    let reduce_group_suffix = if plan.has_custom_reduce_group() {
        format!("-cg{}", plan.reduce_group_size())
    } else {
        String::new()
    };
    if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
        format!(
            "{plan_name}::bf16{}{}{}{}",
            row_upcast_suffix,
            reduce_group_suffix,
            plan.thread_group.operation_suffix(),
            plan.loop_order.operation_suffix()
        )
    } else {
        format!(
            "{plan_name}::bf16{}-u{}{}{}{}",
            row_upcast_suffix,
            plan.reduce_unroll,
            reduce_group_suffix,
            plan.thread_group.operation_suffix(),
            plan.loop_order.operation_suffix()
        )
    }
}

pub(in crate::autotune) fn schedule_gemm_tile(schedule: &KernelSchedule) -> Option<GemmTileShape> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::TileGemm { m, n, k } => Some(GemmTileShape::new(*m, *n, *k)),
            _ => None,
        })
}

pub(in crate::autotune) fn schedule_gemm_reduce_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 2, factor } => Some(*factor),
            _ => None,
        })
}

pub(in crate::autotune) fn schedule_gemm_reduce_group(schedule: &KernelSchedule) -> u32 {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::GroupTop { axis: 2, factor } => Some(*factor),
            _ => None,
        })
        .unwrap_or(0)
}

pub(in crate::autotune) fn schedule_gemm_m_per_thread(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Upcast { axis: 0, factor } => Some(*factor),
            _ => None,
        })
}

pub(in crate::autotune) fn schedule_gemm_n_per_thread(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Upcast { axis: 1, factor } => Some(*factor),
            _ => None,
        })
}

pub(in crate::autotune) fn schedule_gemm_a_load_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 3, factor } => Some(*factor),
            _ => None,
        })
}

pub(in crate::autotune) fn schedule_gemm_b_load_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 4, factor } => Some(*factor),
            _ => None,
        })
}

pub(in crate::autotune) fn schedule_gemm_a_load_thread_group(schedule: &KernelSchedule) -> u32 {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::ThreadGroup { axis: 3, factor }
            | ScheduleTransform::Group { axis: 3, factor } => Some(*factor),
            _ => None,
        })
        .unwrap_or(0)
}

pub(in crate::autotune) fn schedule_gemm_b_load_thread_group(schedule: &KernelSchedule) -> u32 {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::ThreadGroup { axis: 4, factor }
            | ScheduleTransform::Group { axis: 4, factor } => Some(*factor),
            _ => None,
        })
        .unwrap_or(0)
}

pub(in crate::autotune) fn schedule_gemm_b_load_order(
    schedule: &KernelSchedule,
) -> GemmBTileLoadOrder {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::StrideOrder { axes } if axes.as_slice() == [2, 1] => {
                Some(GemmBTileLoadOrder::KContiguous)
            }
            _ => None,
        })
        .unwrap_or(GemmBTileLoadOrder::TileLinear)
}

pub(in crate::autotune) fn schedule_gemm_a_load_order(
    schedule: &KernelSchedule,
) -> GemmATileLoadOrder {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::StrideOrder { axes } if axes.as_slice() == [0, 2] => {
                Some(GemmATileLoadOrder::MContiguous)
            }
            _ => None,
        })
        .unwrap_or(GemmATileLoadOrder::KContiguous)
}

pub(in crate::autotune) fn schedule_gemm_thread_order(
    schedule: &KernelSchedule,
) -> GemmThreadOrder {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Swap {
                axis_a: 0,
                axis_b: 1,
            } => Some(GemmThreadOrder::MThenN),
            _ => None,
        })
        .unwrap_or(GemmThreadOrder::NThenM)
}

pub(in crate::autotune) fn schedule_gemm_plan(
    schedule: &KernelSchedule,
) -> Option<GemmSchedulePlan> {
    let tile = schedule_gemm_tile(schedule)?;
    Some(GemmSchedulePlan {
        tile,
        reduce_unroll: schedule_gemm_reduce_unroll(schedule).unwrap_or(1),
        reduce_group: schedule_gemm_reduce_group(schedule),
        m_per_thread: schedule_gemm_m_per_thread(schedule).unwrap_or(1),
        n_per_thread: schedule_gemm_n_per_thread(schedule).unwrap_or(1),
        a_load_unroll: schedule_gemm_a_load_unroll(schedule).unwrap_or(1),
        b_load_unroll: schedule_gemm_b_load_unroll(schedule).unwrap_or(1),
        a_load_thread_group: schedule_gemm_a_load_thread_group(schedule),
        b_load_thread_group: schedule_gemm_b_load_thread_group(schedule),
        a_load_order: schedule_gemm_a_load_order(schedule),
        b_load_order: schedule_gemm_b_load_order(schedule),
        thread_order: schedule_gemm_thread_order(schedule),
    })
}
