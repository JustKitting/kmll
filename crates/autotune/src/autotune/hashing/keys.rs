use super::{action_space::*, candidate::*, primitives::*, *};

pub(in crate::autotune) fn metadata_key(
    family: &str,
    axes: &[KernelAxis],
    schedule: &KernelSchedule,
    launch: &CudaLaunchSpec,
) -> KernelMetadataKey {
    let mut state = FNV_OFFSET;
    state = hash_str(state, family);
    for axis in axes {
        state = hash_u64(state, axis.id as u64);
        state = hash_str(state, axis.name);
        state = hash_u64(state, axis.extent as u64);
        state = hash_u64(
            state,
            match axis.kind {
                KernelAxisKind::Spatial => 1,
                KernelAxisKind::Reduction => 2,
            },
        );
        state = hash_u64(state, axis.stride.unwrap_or(usize::MAX) as u64);
    }
    for transform in &schedule.transforms {
        state = hash_transform(state, transform);
    }
    state = hash_str(state, &launch.kernel);
    state = hash_u64(state, launch.grid_dim.x as u64);
    state = hash_u64(state, launch.grid_dim.y as u64);
    state = hash_u64(state, launch.grid_dim.z as u64);
    state = hash_u64(state, launch.block_dim.x as u64);
    state = hash_u64(state, launch.block_dim.y as u64);
    state = hash_u64(state, launch.block_dim.z as u64);
    state = hash_u64(state, launch.shared_mem_bytes as u64);
    KernelMetadataKey(state)
}

pub(in crate::autotune) fn implementation_key(
    family: &str,
    schedule: &KernelSchedule,
    launch: &CudaLaunchSpec,
    generated: &GeneratedKernelMetadata,
) -> KernelImplementationKey {
    let mut state = FNV_OFFSET;
    state = hash_str(state, "implementation-key-v1");
    state = hash_str(state, family);
    state = hash_str(state, generated.generator);
    state = match &generated.materialization {
        KernelMaterialization::Existing { symbol } => {
            let state = hash_str(state, "existing");
            hash_str(state, symbol)
        }
        KernelMaterialization::Generated { symbol } => {
            let state = hash_str(state, "generated");
            hash_str(state, symbol)
        }
        KernelMaterialization::DeferredGenerated { symbol_hint, .. } => {
            let state = hash_str(state, "deferred-generated");
            hash_str(state, symbol_hint)
        }
    };
    for transform in &schedule.transforms {
        state = hash_transform(state, transform);
    }
    state = hash_str(state, &launch.kernel);
    state = hash_u64(state, launch.block_dim.x as u64);
    state = hash_u64(state, launch.block_dim.y as u64);
    state = hash_u64(state, launch.block_dim.z as u64);
    state = hash_u64(state, launch.shared_mem_bytes as u64);
    KernelImplementationKey(state)
}

pub(in crate::autotune) fn search_report_key(
    report: &OptimizationSearchReport,
) -> KernelMetadataKey {
    let mut state = FNV_OFFSET;
    state = hash_str(state, "optimization-search-report");
    state = hash_str(state, &report.family);
    state = hash_u64(state, report.config.beam_width as u64);
    state = hash_u64(state, report.config.max_depth as u64);
    state = hash_u64(state, u64::from(report.config.require_launchable));
    state = hash_optional_profiling_action_space_set(state, report.action_space.as_ref());
    state = hash_u64(state, report.explored as u64);
    state = hash_u64(state, report.rejected as u64);
    state = hash_u64(state, report.duplicates as u64);
    if let Some(best) = &report.best {
        state = hash_optimization_candidate(state, best);
    } else {
        state = hash_str(state, "no-best");
    }
    for candidate in &report.beam {
        state = hash_optimization_candidate(state, candidate);
    }
    KernelMetadataKey(state)
}

pub(in crate::autotune) fn auto_search_report_key(
    report: &AutoOptimizationSearchReport,
) -> KernelMetadataKey {
    let mut state = FNV_OFFSET;
    state = hash_str(state, "auto-optimization-search-report");
    state = hash_str(state, &report.family);
    state = hash_u64(state, report.config.beam_width as u64);
    state = hash_u64(state, report.config.max_steps as u64);
    state = hash_u64(state, u64::from(report.config.require_launchable));
    state = hash_u64(state, report.config.min_score_improvement.to_bits());
    state = hash_optional_profiling_action_space_set(state, report.action_space.as_ref());
    state = hash_u64(state, report.explored as u64);
    state = hash_u64(state, report.rejected as u64);
    state = hash_u64(state, report.duplicates as u64);
    state = hash_str(state, report.exit_reason.label());
    if let ProfilingAutoOptimizationExitReason::NoImprovement { best_delta } = report.exit_reason {
        state = hash_u64(state, best_delta.to_bits());
    }
    for step in &report.steps {
        state = hash_u64(state, step.depth as u64);
        state = hash_u64(state, step.input_beam_len as u64);
        state = hash_u64(state, step.generated as u64);
        state = hash_u64(state, step.accepted as u64);
        state = hash_u64(state, step.rejected as u64);
        state = hash_u64(state, step.duplicates as u64);
        state = hash_optional_score(state, step.best_before);
        state = hash_optional_score(state, step.best_after);
        if let Some(best_candidate) = &step.best_candidate {
            state = hash_optimization_candidate(state, best_candidate);
        } else {
            state = hash_str(state, "no-step-best-candidate");
        }
        if let Some(improvement) = step.improvement {
            state = hash_str(state, "improvement");
            state = hash_u64(state, improvement.to_bits());
        } else {
            state = hash_str(state, "no-improvement-value");
        }
    }
    if let Some(best) = &report.best {
        state = hash_optimization_candidate(state, best);
    } else {
        state = hash_str(state, "no-best");
    }
    for candidate in &report.beam {
        state = hash_optimization_candidate(state, candidate);
    }
    KernelMetadataKey(state)
}
