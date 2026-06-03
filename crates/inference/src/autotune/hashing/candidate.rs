use super::primitives::*;
use super::*;

pub(in crate::autotune) fn hash_optional_score(mut state: u64, score: Option<SearchScore>) -> u64 {
    if let Some(score) = score {
        state = hash_str(state, "score");
        state = hash_str(state, score.source.label());
        hash_u64(state, score.value.to_bits())
    } else {
        hash_str(state, "no-score")
    }
}

pub(in crate::autotune) fn hash_optimization_candidate(
    mut state: u64,
    candidate: &OptimizationCandidateSpec,
) -> u64 {
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

pub(in crate::autotune) fn hash_operation_spec(
    mut state: u64,
    operation: &TypedOperationSpec,
) -> u64 {
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

pub(in crate::autotune) fn hash_tensor_specs(
    mut state: u64,
    label: &str,
    specs: &[TensorTypeSpec],
) -> u64 {
    state = hash_str(state, label);
    state = hash_u64(state, specs.len() as u64);
    for spec in specs {
        state = hash_tensor_spec(state, spec);
    }
    state
}

pub(in crate::autotune) fn hash_tensor_spec(mut state: u64, spec: &TensorTypeSpec) -> u64 {
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

pub(in crate::autotune) fn hash_transform(mut state: u64, transform: &ScheduleTransform) -> u64 {
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
