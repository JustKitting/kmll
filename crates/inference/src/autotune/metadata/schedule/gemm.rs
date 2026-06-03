use super::*;

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
