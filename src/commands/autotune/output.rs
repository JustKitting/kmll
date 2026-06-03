use nn_rust_inference::autotune::{
    EmittedKernelOptimizationSelection, KernelCandidateMetadata, KernelExpansionPolicy,
    KernelMaterialization, KernelOptimizationCacheKey, KernelScheduleAction,
    KernelScheduleActionArg, ScheduleTransform, SearchScore, SearchScoreSource,
    SelectionCacheStatus,
};
use nn_rust_profiling::{MAX_OPTIMIZATION_SETUP_SEGMENTS, OptimizationTimingSegment};

use super::options::format_optional_u32_cap;

pub(super) fn print_kernel_expansion_policy(policy: KernelExpansionPolicy) {
    println!(
        "search_policy require_launchable={} max_threads_per_block={} max_shared_memory_bytes={} max_accumulator_elements_per_thread={} max_output_elements_per_thread={} max_load_elements_per_block={}",
        policy.require_launchable,
        format_optional_u32_cap(policy.max_threads_per_block),
        format_optional_u32_cap(policy.max_shared_memory_bytes),
        format_optional_u32_cap(policy.max_accumulator_elements_per_thread),
        format_optional_u32_cap(policy.max_output_elements_per_thread),
        format_optional_u32_cap(policy.max_load_elements_per_block),
    );
}

pub(super) fn print_selection_cache_status(
    cache_key: &KernelOptimizationCacheKey,
    status: &SelectionCacheStatus,
) {
    match status {
        SelectionCacheStatus::Stale { reason } => {
            println!(
                "selection_cache cache_key={} status={} reason={}",
                cache_key.hex(),
                status.label(),
                reason
            );
        }
        _ => {
            println!(
                "selection_cache cache_key={} status={}",
                cache_key.hex(),
                status.label()
            );
        }
    }
}

pub(super) fn print_selection_cache_write(
    cache_key: &KernelOptimizationCacheKey,
    emitted: &EmittedKernelOptimizationSelection,
) {
    println!(
        "selection_cache_write cache_key={} artifact_key={} selection_path={} selection_bytes={}",
        cache_key.hex(),
        emitted.artifact_key,
        emitted.selection_path.display(),
        emitted.selection_bytes
    );
}

pub(super) fn print_kernel_candidate(rank: usize, candidate: &KernelCandidateMetadata) {
    let materialization = match &candidate.generated.materialization {
        KernelMaterialization::Existing { symbol } => format!("existing:{symbol}"),
        KernelMaterialization::Generated { symbol } => format!("generated:{symbol}"),
        KernelMaterialization::DeferredGenerated {
            symbol_hint,
            reason,
        } => format!("deferred-generated:{symbol_hint}:{reason}"),
    };
    let score = candidate
        .score
        .map(format_search_score)
        .unwrap_or_else(|| "none".to_string());
    println!(
        "candidate rank={rank} key={} family={} launchable={} score={} generator={} materialization={} kernel={} grid={}x{}x{} block={}x{}x{} schedule={} actions={}",
        candidate.artifact_key().hex(),
        candidate.family,
        candidate.is_launchable(),
        score,
        candidate.generated.generator,
        materialization,
        candidate.launch.kernel,
        candidate.launch.grid_dim.x,
        candidate.launch.grid_dim.y,
        candidate.launch.grid_dim.z,
        candidate.launch.block_dim.x,
        candidate.launch.block_dim.y,
        candidate.launch.block_dim.z,
        format_schedule(&candidate.schedule.transforms),
        format_action_trace(&candidate.action_trace)
    );
}

fn format_search_score(score: SearchScore) -> String {
    match score.source {
        SearchScoreSource::Heuristic => format!("{:.3}:{}", score.value, score.source.label()),
        SearchScoreSource::Measured => {
            let timing = score
                .timing
                .map(|timing| {
                    let setup = format_setup_segments(&timing.setup_segments);
                    format!(
                        ":source={} samples={} warmup={} min={:.9}s max={:.9}s{}",
                        timing.source.label(),
                        timing.samples.count,
                        timing.warmup_count,
                        timing.samples.min,
                        timing.samples.max,
                        setup
                    )
                })
                .unwrap_or_default();
            format!("{:.9}s:{}{}", score.value, score.source.label(), timing)
        }
    }
}

fn format_setup_segments(
    segments: &[Option<OptimizationTimingSegment>; MAX_OPTIMIZATION_SETUP_SEGMENTS],
) -> String {
    let parts = segments
        .iter()
        .flatten()
        .map(|segment| format!("{}={:.6}s", segment.name, segment.duration.as_seconds_f64()))
        .collect::<Vec<_>>();
    if parts.is_empty() {
        String::new()
    } else {
        format!(" setup=[{}]", parts.join(","))
    }
}

fn format_schedule(transforms: &[ScheduleTransform]) -> String {
    if transforms.is_empty() {
        return "[]".to_string();
    }
    let parts = transforms
        .iter()
        .map(|transform| match transform {
            ScheduleTransform::Split { axis, factor } => {
                format!("split(axis={axis},factor={factor})")
            }
            ScheduleTransform::Upcast { axis, factor } => {
                format!("upcast(axis={axis},factor={factor})")
            }
            ScheduleTransform::Unroll { axis, factor } => {
                format!("unroll(axis={axis},factor={factor})")
            }
            ScheduleTransform::LocalTile { axis, factor } => {
                format!("local_tile(axis={axis},factor={factor})")
            }
            ScheduleTransform::GroupTop { axis, factor } => {
                format!("group_top(axis={axis},factor={factor})")
            }
            ScheduleTransform::Group { axis, factor } => {
                format!("group(axis={axis},factor={factor})")
            }
            ScheduleTransform::ThreadGroup { axis, factor } => {
                format!("thread_group(axis={axis},factor={factor})")
            }
            ScheduleTransform::TileGemm { m, n, k } => format!("tile_gemm(m={m},n={n},k={k})"),
            ScheduleTransform::StrideOrder { axes } => format!("stride_order(axes={axes:?})"),
            ScheduleTransform::Swap { axis_a, axis_b } => {
                format!("swap(axis_a={axis_a},axis_b={axis_b})")
            }
        })
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(","))
}

fn format_action_trace(actions: &[KernelScheduleAction]) -> String {
    if actions.is_empty() {
        return "[]".to_string();
    }
    let parts = actions
        .iter()
        .map(|action| {
            let axis = action
                .axis
                .map(|axis| axis.to_string())
                .unwrap_or_else(|| "none".to_string());
            format!(
                "{}(axis={},arg={},materialization={})",
                action.op.label(),
                axis,
                format_action_arg(&action.arg),
                action.materialization.label()
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(","))
}

fn format_action_arg(arg: &KernelScheduleActionArg) -> String {
    match arg {
        KernelScheduleActionArg::Factor(factor) => format!("factor:{factor}"),
        KernelScheduleActionArg::Tile3d { m, n, k } => format!("tile3d:{m}x{n}x{k}"),
        KernelScheduleActionArg::AxisOrder(axes) => format!("axis_order:{axes:?}"),
        KernelScheduleActionArg::AxisPair { axis_a, axis_b } => {
            format!("axis_pair:{axis_a}<->{axis_b}")
        }
    }
}
