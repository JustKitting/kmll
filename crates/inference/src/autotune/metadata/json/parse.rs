use super::{super::super::*, fields::*};

pub(in crate::autotune) fn parse_selection_json(
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
        materialization: parse_optional_materialization_descriptor(
            value.get("materialization").unwrap_or(&Value::Null),
            "materialization",
        )?,
        action_trace: parse_action_trace(required_array(value, "action_trace")?)?,
        score: parse_optional_score(value.get("score").unwrap_or(&Value::Null))?,
    })
}

pub(in crate::autotune) fn parse_score_record_json(
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
        materialization: parse_optional_materialization_descriptor(
            value.get("materialization").unwrap_or(&Value::Null),
            "materialization",
        )?,
        action_trace: parse_action_trace(required_array(value, "action_trace")?)?,
        score,
    })
}

pub(in crate::autotune) fn parse_optional_materialization_descriptor(
    value: &Value,
    field_name: &str,
) -> Result<Option<KernelMaterializationDescriptor>, KernelGenerationError> {
    if value.is_null() {
        return Ok(None);
    }
    let kind = required_str(value, "kind")?;
    match kind {
        "existing" => Ok(Some(KernelMaterializationDescriptor::Existing {
            symbol: required_str(value, "symbol")?.to_string(),
        })),
        "generated" => Ok(Some(KernelMaterializationDescriptor::Generated {
            symbol: required_str(value, "symbol")?.to_string(),
        })),
        "deferred-generated" => Ok(Some(KernelMaterializationDescriptor::DeferredGenerated {
            symbol_hint: required_str(value, "symbol_hint")?.to_string(),
            reason: required_str(value, "reason")?.to_string(),
        })),
        kind => Err(invalid_selection(format!(
            "{field_name}.kind is unsupported: {kind:?}"
        ))),
    }
}

pub(in crate::autotune) fn parse_action_trace(
    actions: &[Value],
) -> Result<Vec<KernelScheduleAction>, KernelGenerationError> {
    actions
        .iter()
        .enumerate()
        .map(|(index, action)| parse_action_json(action, index))
        .collect()
}

pub(in crate::autotune) fn parse_action_json(
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

pub(in crate::autotune) fn parse_action_arg_json(
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

pub(in crate::autotune) fn parse_optional_score(
    value: &Value,
) -> Result<Option<SearchScore>, KernelGenerationError> {
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

pub(in crate::autotune) fn parse_optional_timing(
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

pub(in crate::autotune) fn parse_timing_segments(
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

pub(in crate::autotune) fn parse_profile_time_source(
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

pub(in crate::autotune) fn parse_timing_segment_name(
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
