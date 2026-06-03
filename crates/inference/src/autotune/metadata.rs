fn compare_candidates(a: &KernelCandidateMetadata, b: &KernelCandidateMetadata) -> Ordering {
    let a_score = a.score.map(|score| score.value).unwrap_or(f64::INFINITY);
    let b_score = b.score.map(|score| score.value).unwrap_or(f64::INFINITY);
    a_score
        .partial_cmp(&b_score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.artifact_key().cmp(&b.artifact_key()))
}

fn schedule_rows_per_block(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Split { axis: 0, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_matvec_reduce_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 1, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_matvec_row_upcast(schedule: &KernelSchedule) -> Option<MatvecRowUpcast> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Upcast { axis: 0, factor } => MatvecRowUpcast::new(*factor),
            _ => None,
        })
        .or_else(|| Some(MatvecRowUpcast::default_upcast()))
}

fn schedule_matvec_thread_group(schedule: &KernelSchedule) -> Option<MatvecThreadGroup> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::ThreadGroup { axis: 1, factor } => MatvecThreadGroup::new(*factor),
            _ => None,
        })
        .or_else(|| Some(MatvecThreadGroup::default_group()))
}

fn schedule_matvec_plan(schedule: &KernelSchedule) -> Option<MatvecSchedulePlan> {
    let rows_per_block = schedule_rows_per_block(schedule)?;
    let rows = MatvecRowSplit::new(rows_per_block)?;
    Some(MatvecSchedulePlan {
        rows,
        row_upcast: schedule_matvec_row_upcast(schedule)?,
        reduce_unroll: schedule_matvec_reduce_unroll(schedule)
            .unwrap_or(MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL),
        thread_group: schedule_matvec_thread_group(schedule)?,
    })
}

fn matvec_symbol_hint(plan: MatvecSchedulePlan) -> String {
    let plan = plan.normalized();
    let mut base = format!("matvec_bf16_rows{}", plan.rows.rows_per_block());
    base.push_str(&plan.row_upcast.symbol_suffix());
    if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
        base.push_str(&plan.thread_group.symbol_suffix());
        base
    } else {
        write!(
            &mut base,
            "_u{}{}",
            plan.reduce_unroll,
            plan.thread_group.symbol_suffix()
        )
        .expect("write to string");
        base
    }
}

fn matvec_operation_name(plan: MatvecSchedulePlan) -> String {
    let plan = plan.normalized();
    let plan_name = plan.rows.plan_name();
    let row_upcast_suffix = plan.row_upcast.operation_suffix();
    if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
        format!(
            "{plan_name}::bf16{}{}",
            row_upcast_suffix,
            plan.thread_group.operation_suffix()
        )
    } else {
        format!(
            "{plan_name}::bf16{}-u{}{}",
            row_upcast_suffix,
            plan.reduce_unroll,
            plan.thread_group.operation_suffix()
        )
    }
}

fn schedule_gemm_tile(schedule: &KernelSchedule) -> Option<GemmTileShape> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::TileGemm { m, n, k } => Some(GemmTileShape::new(*m, *n, *k)),
            _ => None,
        })
}

fn schedule_gemm_reduce_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 2, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_gemm_m_per_thread(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Upcast { axis: 0, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_gemm_n_per_thread(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Upcast { axis: 1, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_gemm_a_load_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 3, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_gemm_b_load_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 4, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_gemm_a_load_thread_group(schedule: &KernelSchedule) -> u32 {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::ThreadGroup { axis: 3, factor } => Some(*factor),
            _ => None,
        })
        .unwrap_or(0)
}

fn schedule_gemm_b_load_thread_group(schedule: &KernelSchedule) -> u32 {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::ThreadGroup { axis: 4, factor } => Some(*factor),
            _ => None,
        })
        .unwrap_or(0)
}

fn schedule_gemm_b_load_order(schedule: &KernelSchedule) -> GemmBTileLoadOrder {
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

fn schedule_gemm_a_load_order(schedule: &KernelSchedule) -> GemmATileLoadOrder {
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

fn schedule_gemm_thread_order(schedule: &KernelSchedule) -> GemmThreadOrder {
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

fn schedule_gemm_plan(schedule: &KernelSchedule) -> Option<GemmSchedulePlan> {
    let tile = schedule_gemm_tile(schedule)?;
    Some(GemmSchedulePlan {
        tile,
        reduce_unroll: schedule_gemm_reduce_unroll(schedule).unwrap_or(1),
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

fn generated_kernel_manifest(candidate: &KernelCandidateMetadata) -> Value {
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

fn selection_json(selection: &KernelOptimizationSelection) -> Value {
    json!({
        "schema_version": 1,
        "family": &selection.family,
        "artifact_key": &selection.artifact_key,
        "generator": &selection.generator,
        "launchable": selection.launchable,
        "action_trace": selection
            .action_trace
            .iter()
            .map(action_json)
            .collect::<Vec<_>>(),
        "score": selection.score.map(score_json),
    })
}

fn score_record_json(record: &KernelOptimizationScoreRecord) -> Value {
    json!({
        "schema_version": 1,
        "score_namespace": &record.score_namespace,
        "family": &record.family,
        "artifact_key": &record.artifact_key,
        "generator": &record.generator,
        "launchable": record.launchable,
        "action_trace": record
            .action_trace
            .iter()
            .map(action_json)
            .collect::<Vec<_>>(),
        "score": score_json(record.score),
    })
}

fn parse_selection_json(
    value: &Value,
) -> Result<KernelOptimizationSelection, KernelGenerationError> {
    let schema_version = required_u64(value, "schema_version")?;
    if schema_version != 1 {
        return Err(invalid_selection(format!(
            "unsupported schema_version {schema_version}"
        )));
    }
    Ok(KernelOptimizationSelection {
        family: required_str(value, "family")?.to_string(),
        artifact_key: required_str(value, "artifact_key")?.to_string(),
        generator: required_str(value, "generator")?.to_string(),
        launchable: required_bool(value, "launchable")?,
        action_trace: parse_action_trace(required_array(value, "action_trace")?)?,
        score: parse_optional_score(value.get("score").unwrap_or(&Value::Null))?,
    })
}

fn parse_score_record_json(
    value: &Value,
) -> Result<KernelOptimizationScoreRecord, KernelGenerationError> {
    let schema_version = required_u64(value, "schema_version")?;
    if schema_version != 1 {
        return Err(invalid_selection(format!(
            "unsupported score schema_version {schema_version}"
        )));
    }
    let score = parse_optional_score(required_field(value, "score")?)?
        .ok_or_else(|| invalid_selection("score cache record must contain a score"))?;
    Ok(KernelOptimizationScoreRecord {
        score_namespace: required_str(value, "score_namespace")?.to_string(),
        family: required_str(value, "family")?.to_string(),
        artifact_key: required_str(value, "artifact_key")?.to_string(),
        generator: required_str(value, "generator")?.to_string(),
        launchable: required_bool(value, "launchable")?,
        action_trace: parse_action_trace(required_array(value, "action_trace")?)?,
        score,
    })
}

fn parse_action_trace(
    actions: &[Value],
) -> Result<Vec<KernelScheduleAction>, KernelGenerationError> {
    actions
        .iter()
        .enumerate()
        .map(|(index, action)| parse_action_json(action, index))
        .collect()
}

fn parse_action_json(
    value: &Value,
    index: usize,
) -> Result<KernelScheduleAction, KernelGenerationError> {
    let op = match required_str(value, "op")? {
        "split" => KernelScheduleActionOp::Split,
        "upcast" => KernelScheduleActionOp::Upcast,
        "unroll" => KernelScheduleActionOp::Unroll,
        "local-tile" => KernelScheduleActionOp::LocalTile,
        "thread-group" => KernelScheduleActionOp::ThreadGroup,
        "tile-gemm" => KernelScheduleActionOp::TileGemm,
        "stride-order" => KernelScheduleActionOp::StrideOrder,
        "swap" => KernelScheduleActionOp::Swap,
        op => {
            return Err(invalid_selection(format!(
                "action_trace[{index}] has unsupported op {op:?}"
            )));
        }
    };
    let materialization = match required_str(value, "materialization")? {
        "existing" => KernelActionMaterialization::Existing,
        "deferred-generated" => KernelActionMaterialization::DeferredGenerated,
        materialization => {
            return Err(invalid_selection(format!(
                "action_trace[{index}] has unsupported materialization {materialization:?}"
            )));
        }
    };
    Ok(KernelScheduleAction {
        op,
        axis: optional_u8(value, "axis")?,
        arg: parse_action_arg_json(required_field(value, "arg")?, index)?,
        materialization,
    })
}

fn parse_action_arg_json(
    value: &Value,
    index: usize,
) -> Result<KernelScheduleActionArg, KernelGenerationError> {
    match required_str(value, "kind")? {
        "factor" => Ok(KernelScheduleActionArg::Factor(required_u32(
            value, "value",
        )?)),
        "tile-3d" => Ok(KernelScheduleActionArg::Tile3d {
            m: required_u32(value, "m")?,
            n: required_u32(value, "n")?,
            k: required_u32(value, "k")?,
        }),
        "axis-order" => Ok(KernelScheduleActionArg::AxisOrder(
            required_array(value, "axes")?
                .iter()
                .enumerate()
                .map(|(axis_index, axis)| {
                    value_as_u8(axis).ok_or_else(|| {
                        invalid_selection(format!(
                            "action_trace[{index}].arg.axes[{axis_index}] must be a u8"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        "axis-pair" => Ok(KernelScheduleActionArg::AxisPair {
            axis_a: required_u8(value, "axis_a")?,
            axis_b: required_u8(value, "axis_b")?,
        }),
        kind => Err(invalid_selection(format!(
            "action_trace[{index}] has unsupported arg kind {kind:?}"
        ))),
    }
}

fn parse_optional_score(value: &Value) -> Result<Option<SearchScore>, KernelGenerationError> {
    if value.is_null() {
        return Ok(None);
    }
    let score_value = required_f64(value, "value")?;
    if !score_value.is_finite() {
        return Err(invalid_selection("score.value must be finite"));
    }
    let source = match required_str(value, "source")? {
        "heuristic" => SearchScoreSource::Heuristic,
        "measured" => SearchScoreSource::Measured,
        source => {
            return Err(invalid_selection(format!(
                "score.source is unsupported: {source:?}"
            )));
        }
    };
    Ok(Some(SearchScore {
        value: score_value,
        source,
        timing: parse_optional_timing(value.get("timing").unwrap_or(&Value::Null))?,
    }))
}

fn parse_optional_timing(
    value: &Value,
) -> Result<Option<OptimizationTiming>, KernelGenerationError> {
    if value.is_null() {
        return Ok(None);
    }
    let source = parse_profile_time_source(required_str(value, "source")?, "score.timing.source")?;
    let selected_seconds = required_f64(value, "selected_seconds")?;
    let selected = ProfileDuration::from_seconds_f64(selected_seconds).ok_or_else(|| {
        invalid_selection("score.timing.selected_seconds must be nonnegative and finite")
    })?;
    let samples = required_field(value, "samples")?;
    let sample_stats = SampleStats {
        count: required_usize(samples, "count")?,
        mean: required_f64(samples, "mean_seconds")?,
        median: required_f64(samples, "median_seconds")?,
        min: required_f64(samples, "min_seconds")?,
        max: required_f64(samples, "max_seconds")?,
    };
    if sample_stats.count == 0
        || !sample_stats.mean.is_finite()
        || !sample_stats.median.is_finite()
        || !sample_stats.min.is_finite()
        || !sample_stats.max.is_finite()
    {
        return Err(invalid_selection(
            "score.timing.samples must contain nonzero finite statistics",
        ));
    }
    let setup_segments =
        parse_timing_segments(value.get("setup_segments").unwrap_or(&Value::Null))?;
    Ok(Some(
        OptimizationTiming::new(
            source,
            required_usize(value, "warmup_count")?,
            sample_stats,
            selected,
        )
        .with_setup_segments(&setup_segments),
    ))
}

fn parse_timing_segments(
    value: &Value,
) -> Result<Vec<OptimizationTimingSegment>, KernelGenerationError> {
    if value.is_null() {
        return Ok(Vec::new());
    }
    let values = value
        .as_array()
        .ok_or_else(|| invalid_selection("score.timing.setup_segments must be an array"))?;
    if values.len() > MAX_OPTIMIZATION_SETUP_SEGMENTS {
        return Err(invalid_selection(format!(
            "score.timing.setup_segments has {} entries, maximum is {MAX_OPTIMIZATION_SETUP_SEGMENTS}",
            values.len()
        )));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            let duration_seconds = required_f64(segment, "duration_seconds")?;
            let duration = ProfileDuration::from_seconds_f64(duration_seconds).ok_or_else(|| {
                invalid_selection(format!(
                    "score.timing.setup_segments[{index}].duration_seconds must be nonnegative and finite"
                ))
            })?;
            Ok(OptimizationTimingSegment::new(
                parse_timing_segment_name(required_str(segment, "name")?, index)?,
                parse_profile_time_source(
                    required_str(segment, "source")?,
                    "score.timing.setup_segments[].source",
                )?,
                duration,
            ))
        })
        .collect()
}

fn parse_profile_time_source(
    source: &str,
    field_name: &str,
) -> Result<ProfileTimeSource, KernelGenerationError> {
    match source {
        "wall-clock" => Ok(ProfileTimeSource::WallClock),
        "cuda-event" => Ok(ProfileTimeSource::CudaEvent),
        "host-self-time" => Ok(ProfileTimeSource::SelfTimeAccounting),
        source => Err(invalid_selection(format!(
            "{field_name} is unsupported: {source:?}"
        ))),
    }
}

fn parse_timing_segment_name(
    name: &str,
    index: usize,
) -> Result<&'static str, KernelGenerationError> {
    match name {
        "emit-standalone-crate" => Ok("emit-standalone-crate"),
        "compile-standalone-crate" => Ok("compile-standalone-crate"),
        "load-generated-module" => Ok("load-generated-module"),
        "load-generated-symbol" => Ok("load-generated-symbol"),
        "cleanup-compile-scratch" => Ok("cleanup-compile-scratch"),
        name => Err(invalid_selection(format!(
            "score.timing.setup_segments[{index}].name is unsupported: {name:?}"
        ))),
    }
}

fn required_field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, KernelGenerationError> {
    value
        .get(name)
        .ok_or_else(|| invalid_selection(format!("missing field {name:?}")))
}

fn required_str<'a>(value: &'a Value, name: &str) -> Result<&'a str, KernelGenerationError> {
    required_field(value, name)?
        .as_str()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be a string")))
}

fn required_bool(value: &Value, name: &str) -> Result<bool, KernelGenerationError> {
    required_field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be a bool")))
}

fn required_array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value], KernelGenerationError> {
    required_field(value, name)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be an array")))
}

fn required_u64(value: &Value, name: &str) -> Result<u64, KernelGenerationError> {
    required_field(value, name)?
        .as_u64()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be a u64")))
}

fn required_usize(value: &Value, name: &str) -> Result<usize, KernelGenerationError> {
    usize::try_from(required_u64(value, name)?)
        .map_err(|_| invalid_selection(format!("field {name:?} exceeds usize")))
}

fn required_u32(value: &Value, name: &str) -> Result<u32, KernelGenerationError> {
    u32::try_from(required_u64(value, name)?)
        .map_err(|_| invalid_selection(format!("field {name:?} exceeds u32")))
}

fn required_u8(value: &Value, name: &str) -> Result<u8, KernelGenerationError> {
    value_as_u8(required_field(value, name)?)
        .ok_or_else(|| invalid_selection(format!("field {name:?} must fit in u8")))
}

fn optional_u8(value: &Value, name: &str) -> Result<Option<u8>, KernelGenerationError> {
    match value.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value_as_u8(value)
            .ok_or_else(|| invalid_selection(format!("field {name:?} must be null or a u8")))
            .map(Some),
    }
}

fn value_as_u8(value: &Value) -> Option<u8> {
    value.as_u64().and_then(|value| u8::try_from(value).ok())
}

fn required_f64(value: &Value, name: &str) -> Result<f64, KernelGenerationError> {
    required_field(value, name)?
        .as_f64()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be an f64")))
}

fn invalid_selection(reason: impl Into<String>) -> KernelGenerationError {
    KernelGenerationError::InvalidSelection {
        reason: reason.into(),
    }
}

fn materialization_json(materialization: &KernelMaterialization) -> Value {
    match materialization {
        KernelMaterialization::Existing { symbol } => {
            json!({"kind": "existing", "symbol": symbol})
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

fn launch_json(launch: &CudaLaunchSpec) -> Value {
    json!({
        "kernel": &launch.kernel,
        "grid_dim": [launch.grid_dim.x, launch.grid_dim.y, launch.grid_dim.z],
        "block_dim": [launch.block_dim.x, launch.block_dim.y, launch.block_dim.z],
        "shared_mem_bytes": launch.shared_mem_bytes,
    })
}

fn operation_json(operation: &TypedOperationSpec) -> Value {
    json!({
        "name": &operation.name,
        "kind": operation.kind.label(),
        "route": operation.route.label(),
        "inputs": operation.inputs.iter().map(tensor_json).collect::<Vec<_>>(),
        "outputs": operation.outputs.iter().map(tensor_json).collect::<Vec<_>>(),
    })
}

fn tensor_json(tensor: &TensorTypeSpec) -> Value {
    json!({
        "dtype": tensor.dtype.label(),
        "dtype_bits": tensor.dtype.bits(),
        "accumulator": tensor.accumulator.label(),
        "accumulator_bits": tensor.accumulator.bits(),
        "shape": &tensor.shape,
        "layout": &tensor.layout,
    })
}

fn axis_json(axis: &KernelAxis) -> Value {
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

fn transform_json(transform: &ScheduleTransform) -> Value {
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

fn action_json(action: &KernelScheduleAction) -> Value {
    json!({
        "op": action.op.label(),
        "axis": action.axis,
        "arg": action_arg_json(&action.arg),
        "materialization": action.materialization.label(),
    })
}

fn action_arg_json(arg: &KernelScheduleActionArg) -> Value {
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

fn resource_usage_json(resources: KernelResourceUsage) -> Value {
    json!({
        "threads_per_block": resources.threads_per_block,
        "shared_memory_bytes": resources.shared_memory_bytes,
        "accumulator_elements_per_thread": resources.accumulator_elements_per_thread,
        "output_elements_per_thread": resources.output_elements_per_thread,
        "load_elements_per_block": resources.load_elements_per_block,
    })
}

fn score_json(score: SearchScore) -> Value {
    json!({
        "value": score.value,
        "source": match score.source {
            SearchScoreSource::Heuristic => "heuristic",
            SearchScoreSource::Measured => "measured",
        },
        "timing": score.timing.map(timing_json),
    })
}

fn timing_json(timing: OptimizationTiming) -> Value {
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
