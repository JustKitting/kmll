use crate::autotune::{
    AutoOptimizeResult, KernelScheduleAction, KernelScheduleActionArg, SearchScore,
};

use super::{super::super::DecompiledAutotuneEvidence, types::DecompileAutotuneOverview};

pub(super) fn resolved_source_score(
    overview: &DecompileAutotuneOverview<'_>,
) -> Option<SearchScore> {
    overview.source_score.or_else(|| {
        overview
            .optimization
            .steps
            .first()
            .and_then(|step| step.best_before)
    })
}

pub(super) fn positive_step_count(optimization: &AutoOptimizeResult) -> usize {
    optimization
        .steps
        .iter()
        .filter(|step| {
            step.improvement
                .is_some_and(|improvement| improvement > 0.0)
        })
        .count()
}

pub(super) fn evidence_summary(evidence: &DecompiledAutotuneEvidence) -> String {
    let mut items = Vec::new();
    if evidence.has_bf16_descriptor_load {
        items.push("bf16 desc load");
    }
    if evidence.has_f32_descriptor_load {
        items.push("f32 desc load");
    }
    if evidence.has_bf16_widen {
        items.push("bf16 widen");
    }
    if evidence.has_f32_mul_add {
        items.push("f32 mul+add");
    }
    if evidence.has_f32_fused_multiply_add {
        items.push("f32 fma");
    }
    if evidence.has_shared_load {
        items.push("shared load");
    }
    if evidence.has_shared_store {
        items.push("shared store");
    }
    if evidence.has_barrier {
        items.push("barrier");
    }
    if evidence.has_descriptor_store {
        items.push("desc store");
    }
    if evidence.has_warp_reduce_sum {
        items.push("warp reduce");
    }
    if items.is_empty() {
        "none".to_string()
    } else {
        items.join(", ")
    }
}

pub(super) fn pattern_category_list(evidence: &DecompiledAutotuneEvidence) -> String {
    if evidence.pattern_categories.is_empty() {
        return "none".to_string();
    }
    evidence
        .pattern_categories
        .iter()
        .map(|category| category.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) fn format_action_trace(actions: &[KernelScheduleAction]) -> String {
    if actions.is_empty() {
        return "none".to_string();
    }
    actions
        .iter()
        .map(|action| format!("{}({})", action.op.label(), format_action_arg(action)))
        .collect::<Vec<_>>()
        .join(" -> ")
}

pub(super) fn format_action_arg(action: &KernelScheduleAction) -> String {
    match &action.arg {
        KernelScheduleActionArg::Factor(factor) => {
            format!(
                "axis={} factor={factor}",
                action
                    .axis
                    .map(|axis| axis.to_string())
                    .unwrap_or_else(|| "-".to_string())
            )
        }
        KernelScheduleActionArg::Tile3d { m, n, k } => format!("tile={m}x{n}x{k}"),
        KernelScheduleActionArg::AxisOrder(axes) => format!("axes={}", format_axes(axes)),
        KernelScheduleActionArg::AxisPair { axis_a, axis_b } => {
            format!("axes={axis_a},{axis_b}")
        }
    }
}

pub(super) fn format_score(score: Option<SearchScore>) -> String {
    score
        .map(|score| format!("{:.12} ({})", score.value, score.source.label()))
        .unwrap_or_else(|| "not-scored".to_string())
}

pub(super) fn md_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

pub(super) fn mermaid_label(value: &str) -> String {
    value
        .replace('"', "'")
        .replace('[', "(")
        .replace(']', ")")
        .replace('\n', "<br/>")
}

fn format_axes(axes: &[u8]) -> String {
    axes.iter().map(u8::to_string).collect::<Vec<_>>().join(",")
}
