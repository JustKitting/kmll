use super::super::super::*;

pub(in crate::autotune) fn generated_kernel_manifest(candidate: &KernelCandidateMetadata) -> Value {
    json!({
        "schema_version": 1,
        "artifact_key": candidate.artifact_key().hex(),
        "family": &candidate.family,
        "generator": candidate.generated.generator,
        "materialization": materialization_json(&candidate.generated.materialization),
        "launch": launch_json(&candidate.launch),
        "operation": operation_json(&candidate.operation),
        "axes": candidate.axes.iter().map(axis_json).collect::<Vec<_>>(),
        "schedule": candidate
            .schedule
            .transforms
            .iter()
            .map(transform_json)
            .collect::<Vec<_>>(),
        "action_trace": candidate
            .action_trace
            .iter()
            .map(action_json)
            .collect::<Vec<_>>(),
        "resources": candidate.resources.map(resource_usage_json),
        "score": candidate.score.map(score_json),
    })
}

pub(in crate::autotune) fn selection_json(selection: &KernelOptimizationSelection) -> Value {
    json!({
        "schema_version": 1,
        "family": &selection.family,
        "artifact_key": &selection.artifact_key,
        "generator": &selection.generator,
        "launchable": selection.launchable,
        "materialization": selection
            .materialization
            .as_ref()
            .map(materialization_descriptor_json),
        "action_trace": selection
            .action_trace
            .iter()
            .map(action_json)
            .collect::<Vec<_>>(),
        "score": selection.score.map(score_json),
    })
}

pub(in crate::autotune) fn score_record_json(record: &KernelOptimizationScoreRecord) -> Value {
    json!({
        "schema_version": 1,
        "score_namespace": &record.score_namespace,
        "family": &record.family,
        "artifact_key": &record.artifact_key,
        "generator": &record.generator,
        "launchable": record.launchable,
        "materialization": record
            .materialization
            .as_ref()
            .map(materialization_descriptor_json),
        "action_trace": record
            .action_trace
            .iter()
            .map(action_json)
            .collect::<Vec<_>>(),
        "score": score_json(record.score),
    })
}

pub(in crate::autotune) fn materialization_json(materialization: &KernelMaterialization) -> Value {
    match materialization {
        KernelMaterialization::Existing { symbol } => {
            json!({"kind": "existing", "symbol": symbol})
        }
        KernelMaterialization::Generated { symbol } => {
            json!({"kind": "generated", "symbol": symbol})
        }
        KernelMaterialization::DeferredGenerated {
            symbol_hint,
            reason,
        } => json!({
            "kind": "deferred-generated",
            "symbol_hint": symbol_hint,
            "reason": reason,
        }),
    }
}

pub(in crate::autotune) fn materialization_descriptor_json(
    materialization: &KernelMaterializationDescriptor,
) -> Value {
    match materialization {
        KernelMaterializationDescriptor::Existing { symbol } => {
            json!({"kind": "existing", "symbol": symbol})
        }
        KernelMaterializationDescriptor::Generated { symbol } => {
            json!({"kind": "generated", "symbol": symbol})
        }
        KernelMaterializationDescriptor::DeferredGenerated {
            symbol_hint,
            reason,
        } => json!({
            "kind": "deferred-generated",
            "symbol_hint": symbol_hint,
            "reason": reason,
        }),
    }
}

pub(in crate::autotune) fn launch_json(launch: &CudaLaunchSpec) -> Value {
    json!({
        "kernel": &launch.kernel,
        "grid_dim": [launch.grid_dim.x, launch.grid_dim.y, launch.grid_dim.z],
        "block_dim": [launch.block_dim.x, launch.block_dim.y, launch.block_dim.z],
        "shared_mem_bytes": launch.shared_mem_bytes,
    })
}

pub(in crate::autotune) fn operation_json(operation: &TypedOperationSpec) -> Value {
    json!({
        "name": &operation.name,
        "kind": operation.kind.label(),
        "route": operation.route.label(),
        "inputs": operation.inputs.iter().map(tensor_json).collect::<Vec<_>>(),
        "outputs": operation.outputs.iter().map(tensor_json).collect::<Vec<_>>(),
    })
}

pub(in crate::autotune) fn tensor_json(tensor: &TensorTypeSpec) -> Value {
    json!({
        "dtype": tensor.dtype.label(),
        "dtype_bits": tensor.dtype.bits(),
        "accumulator": tensor.accumulator.label(),
        "accumulator_bits": tensor.accumulator.bits(),
        "shape": &tensor.shape,
        "layout": &tensor.layout,
    })
}

pub(in crate::autotune) fn axis_json(axis: &KernelAxis) -> Value {
    json!({
        "id": axis.id,
        "name": axis.name,
        "extent": axis.extent,
        "kind": match axis.kind {
            KernelAxisKind::Spatial => "spatial",
            KernelAxisKind::Reduction => "reduction",
        },
        "stride": axis.stride,
    })
}

pub(in crate::autotune) fn transform_json(transform: &ScheduleTransform) -> Value {
    match transform {
        ScheduleTransform::Split { axis, factor } => {
            json!({"op": "split", "axis": axis, "factor": factor})
        }
        ScheduleTransform::Upcast { axis, factor } => {
            json!({"op": "upcast", "axis": axis, "factor": factor})
        }
        ScheduleTransform::Unroll { axis, factor } => {
            json!({"op": "unroll", "axis": axis, "factor": factor})
        }
        ScheduleTransform::LocalTile { axis, factor } => {
            json!({"op": "local-tile", "axis": axis, "factor": factor})
        }
        ScheduleTransform::GroupTop { axis, factor } => {
            json!({"op": "group-top", "axis": axis, "factor": factor})
        }
        ScheduleTransform::Group { axis, factor } => {
            json!({"op": "group", "axis": axis, "factor": factor})
        }
        ScheduleTransform::ThreadGroup { axis, factor } => {
            json!({"op": "thread-group", "axis": axis, "factor": factor})
        }
        ScheduleTransform::TileGemm { m, n, k } => {
            json!({"op": "tile-gemm", "m": m, "n": n, "k": k})
        }
        ScheduleTransform::StrideOrder { axes } => {
            json!({"op": "stride-order", "axes": axes})
        }
        ScheduleTransform::Swap { axis_a, axis_b } => {
            json!({"op": "swap", "axis_a": axis_a, "axis_b": axis_b})
        }
    }
}

pub(in crate::autotune) fn action_json(action: &KernelScheduleAction) -> Value {
    json!({
        "op": action.op.label(),
        "axis": action.axis,
        "arg": action_arg_json(&action.arg),
        "materialization": action.materialization.label(),
    })
}

pub(in crate::autotune) fn action_arg_json(arg: &KernelScheduleActionArg) -> Value {
    match arg {
        KernelScheduleActionArg::Factor(factor) => json!({"kind": "factor", "value": factor}),
        KernelScheduleActionArg::Tile3d { m, n, k } => {
            json!({"kind": "tile-3d", "m": m, "n": n, "k": k})
        }
        KernelScheduleActionArg::AxisOrder(axes) => {
            json!({"kind": "axis-order", "axes": axes})
        }
        KernelScheduleActionArg::AxisPair { axis_a, axis_b } => {
            json!({"kind": "axis-pair", "axis_a": axis_a, "axis_b": axis_b})
        }
    }
}

pub(in crate::autotune) fn resource_usage_json(resources: KernelResourceUsage) -> Value {
    json!({
        "threads_per_block": resources.threads_per_block,
        "shared_memory_bytes": resources.shared_memory_bytes,
        "accumulator_elements_per_thread": resources.accumulator_elements_per_thread,
        "output_elements_per_thread": resources.output_elements_per_thread,
        "load_elements_per_block": resources.load_elements_per_block,
    })
}

pub(in crate::autotune) fn score_json(score: SearchScore) -> Value {
    json!({
        "value": score.value,
        "source": match score.source {
            SearchScoreSource::Heuristic => "heuristic",
            SearchScoreSource::Measured => "measured",
        },
        "timing": score.timing.map(timing_json),
    })
}

pub(in crate::autotune) fn timing_json(timing: OptimizationTiming) -> Value {
    let setup_segments = timing
        .setup_segments
        .iter()
        .flatten()
        .map(|segment| {
            json!({
                "name": segment.name,
                "source": segment.source.label(),
                "duration_seconds": segment.duration.as_seconds_f64(),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "source": timing.source.label(),
        "warmup_count": timing.warmup_count,
        "selected_seconds": timing.selected.as_seconds_f64(),
        "samples": {
            "count": timing.samples.count,
            "mean_seconds": timing.samples.mean,
            "median_seconds": timing.samples.median,
            "min_seconds": timing.samples.min,
            "max_seconds": timing.samples.max,
        },
        "setup_segments": setup_segments,
    })
}
