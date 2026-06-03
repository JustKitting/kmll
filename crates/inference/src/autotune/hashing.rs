fn sanitize_path_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "kernel".to_string()
    } else {
        sanitized
    }
}

fn sanitize_identifier(value: &str) -> String {
    let mut sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        sanitized.push_str("kernel");
    }
    if sanitized
        .as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_digit())
    {
        sanitized.insert_str(0, "k_");
    }
    sanitized
}

fn metadata_key(
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

fn search_report_key(report: &OptimizationSearchReport) -> KernelMetadataKey {
    let mut state = FNV_OFFSET;
    state = hash_str(state, "optimization-search-report");
    state = hash_str(state, &report.family);
    state = hash_u64(state, report.config.beam_width as u64);
    state = hash_u64(state, report.config.max_depth as u64);
    state = hash_u64(state, u64::from(report.config.require_launchable));
    state = hash_optional_profiling_action_space_set(state, report.action_space.as_ref());
    state = hash_u64(state, report.explored as u64);
    state = hash_u64(state, report.rejected as u64);
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

fn auto_search_report_key(report: &AutoOptimizationSearchReport) -> KernelMetadataKey {
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

fn hash_optional_profiling_action_space_set(
    state: u64,
    action_space: Option<&ProfilingActionSpaceSet>,
) -> u64 {
    if let Some(action_space) = action_space {
        hash_profiling_action_space_set(state, action_space)
    } else {
        hash_str(state, "no-action-space")
    }
}

fn hash_profiling_action_space_set(mut state: u64, action_space: &ProfilingActionSpaceSet) -> u64 {
    state = hash_str(state, "profiling-action-space-set");
    state = hash_u64(state, action_space.spaces.len() as u64);
    for space in &action_space.spaces {
        state = hash_profiling_action_space(state, space);
    }
    state
}

fn hash_profiling_action_space(mut state: u64, action_space: &ProfilingActionSpace) -> u64 {
    match action_space {
        ProfilingActionSpace::Split { variants } => {
            state = hash_str(state, "split");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.axis as u64);
                state = hash_u64(state, variant.factor as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        ProfilingActionSpace::Upcast { axis, factors } => {
            state = hash_str(state, "upcast");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::Unroll { axis, factors } => {
            state = hash_str(state, "unroll");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::LocalTile { axis, factors } => {
            state = hash_str(state, "local-tile");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::ThreadGroup { axis, factors } => {
            state = hash_str(state, "thread-group");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::TileGemm { variants } => {
            state = hash_str(state, "tile-gemm");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.m as u64);
                state = hash_u64(state, variant.n as u64);
                state = hash_u64(state, variant.k as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        ProfilingActionSpace::StrideOrder { orders } => {
            state = hash_str(state, "stride-order");
            state = hash_u64(state, orders.len() as u64);
            for order in orders {
                state = hash_u64(state, order.len() as u64);
                for axis in order {
                    state = hash_u64(state, *axis as u64);
                }
            }
            state
        }
        ProfilingActionSpace::Swap { pairs } => {
            state = hash_str(state, "swap");
            state = hash_u64(state, pairs.len() as u64);
            for (axis_a, axis_b) in pairs {
                state = hash_u64(state, *axis_a as u64);
                state = hash_u64(state, *axis_b as u64);
            }
            state
        }
    }
}

fn hash_optional_score(mut state: u64, score: Option<SearchScore>) -> u64 {
    if let Some(score) = score {
        state = hash_str(state, "score");
        state = hash_str(state, score.source.label());
        hash_u64(state, score.value.to_bits())
    } else {
        hash_str(state, "no-score")
    }
}

fn hash_action_space_set(mut state: u64, action_space: &KernelActionSpaceSet) -> u64 {
    state = hash_str(state, "action-space-set");
    state = hash_u64(state, action_space.spaces.len() as u64);
    for space in &action_space.spaces {
        state = hash_action_space(state, space);
    }
    state
}

fn hash_action_space(mut state: u64, action_space: &KernelActionSpace) -> u64 {
    match action_space {
        KernelActionSpace::Split { variants } => {
            state = hash_str(state, "split");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.axis as u64);
                state = hash_u64(state, variant.factor as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        KernelActionSpace::Upcast { axis, factors } => {
            state = hash_str(state, "upcast");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        KernelActionSpace::Unroll { axis, factors } => {
            state = hash_str(state, "unroll");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        KernelActionSpace::LocalTile { axis, factors } => {
            state = hash_str(state, "local-tile");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        KernelActionSpace::ThreadGroup { axis, factors } => {
            state = hash_str(state, "thread-group");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        KernelActionSpace::TileGemm { variants } => {
            state = hash_str(state, "tile-gemm");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.tile.m as u64);
                state = hash_u64(state, variant.tile.n as u64);
                state = hash_u64(state, variant.tile.k as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        KernelActionSpace::StrideOrder { orders } => {
            state = hash_str(state, "stride-order");
            state = hash_u64(state, orders.len() as u64);
            for order in orders {
                state = hash_u64(state, order.len() as u64);
                for axis in order {
                    state = hash_u64(state, *axis as u64);
                }
            }
            state
        }
        KernelActionSpace::Swap { pairs } => {
            state = hash_str(state, "swap");
            state = hash_u64(state, pairs.len() as u64);
            for (axis_a, axis_b) in pairs {
                state = hash_u64(state, *axis_a as u64);
                state = hash_u64(state, *axis_b as u64);
            }
            state
        }
    }
}

fn hash_optimization_candidate(mut state: u64, candidate: &OptimizationCandidateSpec) -> u64 {
    state = hash_str(state, &candidate.family);
    state = hash_str(state, &candidate.artifact_key);
    state = hash_str(state, &candidate.generator);
    state = hash_u64(state, u64::from(candidate.launchable));
    state = hash_str(state, &candidate.launch.kernel);
    state = hash_u64(state, candidate.launch.grid_dim.x as u64);
    state = hash_u64(state, candidate.launch.grid_dim.y as u64);
    state = hash_u64(state, candidate.launch.grid_dim.z as u64);
    state = hash_u64(state, candidate.launch.block_dim.x as u64);
    state = hash_u64(state, candidate.launch.block_dim.y as u64);
    state = hash_u64(state, candidate.launch.block_dim.z as u64);
    state = hash_u64(state, candidate.launch.shared_mem_bytes as u64);
    state = hash_operation_spec(state, &candidate.operation);
    for action in &candidate.action_trace {
        state = hash_str(state, action.op.label());
        state = hash_u64(state, action.axis.unwrap_or(u8::MAX) as u64);
        state = match &action.arg {
            KernelScheduleActionArg::Factor(factor) => {
                let state = hash_str(state, "factor");
                hash_u64(state, *factor as u64)
            }
            KernelScheduleActionArg::Tile3d { m, n, k } => {
                let mut state = hash_str(state, "tile-3d");
                state = hash_u64(state, *m as u64);
                state = hash_u64(state, *n as u64);
                hash_u64(state, *k as u64)
            }
            KernelScheduleActionArg::AxisOrder(axes) => {
                let mut state = hash_str(state, "axis-order");
                for axis in axes {
                    state = hash_u64(state, *axis as u64);
                }
                state
            }
            KernelScheduleActionArg::AxisPair { axis_a, axis_b } => {
                let mut state = hash_str(state, "axis-pair");
                state = hash_u64(state, *axis_a as u64);
                hash_u64(state, *axis_b as u64)
            }
        };
        state = hash_str(state, action.materialization.label());
    }
    if let Some(score) = candidate.score {
        state = hash_str(state, score.source.label());
        state = hash_u64(state, score.value.to_bits());
        if let Some(timing) = score.timing {
            state = hash_str(state, timing.source.label());
            state = hash_u64(state, timing.warmup_count as u64);
            state = hash_u64(
                state,
                timing.selected.as_nanos_u128().min(u64::MAX as u128) as u64,
            );
            state = hash_u64(state, timing.samples.count as u64);
            state = hash_u64(state, timing.samples.mean.to_bits());
            state = hash_u64(state, timing.samples.median.to_bits());
            state = hash_u64(state, timing.samples.min.to_bits());
            state = hash_u64(state, timing.samples.max.to_bits());
            for segment in timing.setup_segments.iter().flatten() {
                state = hash_str(state, segment.name);
                state = hash_str(state, segment.source.label());
                state = hash_u64(
                    state,
                    segment.duration.as_nanos_u128().min(u64::MAX as u128) as u64,
                );
            }
        }
    } else {
        state = hash_str(state, "no-score");
    }
    state
}

fn hash_operation_spec(mut state: u64, operation: &TypedOperationSpec) -> u64 {
    state = hash_str(state, &operation.name);
    state = hash_str(state, operation.kind.label());
    state = hash_str(state, operation.route.label());
    state = hash_tensor_specs(state, "inputs", &operation.inputs);
    state = hash_tensor_specs(state, "outputs", &operation.outputs);
    if let Some(launch) = &operation.launch {
        state = hash_str(state, "operation-launch");
        state = hash_str(state, &launch.kernel);
        state = hash_u64(state, launch.grid_dim.x as u64);
        state = hash_u64(state, launch.grid_dim.y as u64);
        state = hash_u64(state, launch.grid_dim.z as u64);
        state = hash_u64(state, launch.block_dim.x as u64);
        state = hash_u64(state, launch.block_dim.y as u64);
        state = hash_u64(state, launch.block_dim.z as u64);
        state = hash_u64(state, launch.shared_mem_bytes as u64);
    } else {
        state = hash_str(state, "no-operation-launch");
    }
    state
}

fn hash_tensor_specs(mut state: u64, label: &str, specs: &[TensorTypeSpec]) -> u64 {
    state = hash_str(state, label);
    state = hash_u64(state, specs.len() as u64);
    for spec in specs {
        state = hash_tensor_spec(state, spec);
    }
    state
}

fn hash_tensor_spec(mut state: u64, spec: &TensorTypeSpec) -> u64 {
    state = hash_str(state, spec.dtype.label());
    state = hash_str(state, spec.accumulator.label());
    state = hash_u64(state, spec.shape.len() as u64);
    for dim in &spec.shape {
        state = hash_u64(state, *dim as u64);
    }
    if let Some(layout) = &spec.layout {
        state = hash_str(state, layout);
    } else {
        state = hash_str(state, "no-layout");
    }
    state
}

fn hash_transform(mut state: u64, transform: &ScheduleTransform) -> u64 {
    match transform {
        ScheduleTransform::Split { axis, factor } => {
            state = hash_str(state, "split");
            state = hash_u64(state, *axis as u64);
            hash_u64(state, *factor as u64)
        }
        ScheduleTransform::Upcast { axis, factor } => {
            state = hash_str(state, "upcast");
            state = hash_u64(state, *axis as u64);
            hash_u64(state, *factor as u64)
        }
        ScheduleTransform::Unroll { axis, factor } => {
            state = hash_str(state, "unroll");
            state = hash_u64(state, *axis as u64);
            hash_u64(state, *factor as u64)
        }
        ScheduleTransform::LocalTile { axis, factor } => {
            state = hash_str(state, "local-tile");
            state = hash_u64(state, *axis as u64);
            hash_u64(state, *factor as u64)
        }
        ScheduleTransform::ThreadGroup { axis, factor } => {
            state = hash_str(state, "thread-group");
            state = hash_u64(state, *axis as u64);
            hash_u64(state, *factor as u64)
        }
        ScheduleTransform::TileGemm { m, n, k } => {
            state = hash_str(state, "tile-gemm");
            state = hash_u64(state, *m as u64);
            state = hash_u64(state, *n as u64);
            hash_u64(state, *k as u64)
        }
        ScheduleTransform::StrideOrder { axes } => {
            state = hash_str(state, "stride-order");
            for axis in axes {
                state = hash_u64(state, *axis as u64);
            }
            state
        }
        ScheduleTransform::Swap { axis_a, axis_b } => {
            state = hash_str(state, "swap");
            state = hash_u64(state, *axis_a as u64);
            hash_u64(state, *axis_b as u64)
        }
    }
}

fn hash_str(state: u64, value: &str) -> u64 {
    hash_bytes(state, value.as_bytes())
}

fn hash_u64(state: u64, value: u64) -> u64 {
    hash_bytes(state, &value.to_le_bytes())
}

fn hash_bytes(mut state: u64, value: &[u8]) -> u64 {
    for byte in value {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(FNV_PRIME);
    }
    state
}
