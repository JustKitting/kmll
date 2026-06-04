use std::{fmt::Write as _, path::Path};

use crate::autotune::KernelScheduleAction;

use super::{
    format::{
        format_action_arg, format_action_trace, format_score, md_cell, pattern_category_list,
        resolved_source_score,
    },
    types::{DecompileAutotuneOverview, DecompileAutotuneOverviewSide},
};

pub(super) fn render_markdown_overview(
    overview: &DecompileAutotuneOverview<'_>,
    graph: &str,
) -> String {
    let mut out = String::new();
    push_summary_table(&mut out, overview);
    push_graph(&mut out, graph);
    push_metric_diff_table(&mut out, overview);
    push_evidence_table(&mut out, overview);
    push_positive_steps_table(&mut out, overview);
    push_action_trace_table(&mut out, overview.best.action_trace.as_slice());
    push_artifact_table(&mut out, overview);
    out
}

fn push_summary_table(out: &mut String, overview: &DecompileAutotuneOverview<'_>) {
    let source_score = resolved_source_score(overview);
    let best_score = overview.best.score;
    let score_delta = source_score
        .zip(best_score)
        .map(|(source, best)| source.value - best.value);

    writeln!(out, "# {}", overview.title).expect("write to string should not fail");
    writeln!(out).expect("write to string should not fail");
    writeln!(out, "| field | value |").expect("write to string should not fail");
    writeln!(out, "| --- | --- |").expect("write to string should not fail");
    writeln!(out, "| shape | `{}` |", md_cell(&overview.shape))
        .expect("write to string should not fail");
    writeln!(
        out,
        "| operation | `{}` |",
        md_cell(overview.operation_name)
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        "| source symbol | `{}` |",
        md_cell(overview.source.symbol)
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        "| best symbol | `{}` |",
        md_cell(overview.optimized.symbol)
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        "| autotune config | beam_width={} max_steps={} require_launchable={} min_score_improvement={} |",
        overview.config.beam_width,
        overview.config.max_steps,
        overview.config.require_launchable,
        overview.config.min_score_improvement
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        "| search result | explored={} rejected={} duplicates={} exit={} |",
        overview.optimization.explored,
        overview.optimization.rejected,
        overview.optimization.duplicates,
        overview.optimization.exit_reason.label()
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        "| source score | {} |",
        md_cell(&format_score(source_score))
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        "| best score | {} |",
        md_cell(&format_score(best_score))
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        "| source_to_best_delta | {} |",
        score_delta
            .map(|delta| format!("{delta:.12}"))
            .unwrap_or_else(|| "not-scored".to_string())
    )
    .expect("write to string should not fail");
    writeln!(out, "| lower_score_is_better | true |").expect("write to string should not fail");
    writeln!(out).expect("write to string should not fail");
}

fn push_graph(out: &mut String, graph: &str) {
    writeln!(out, "## Pipeline").expect("write to string should not fail");
    writeln!(out).expect("write to string should not fail");
    writeln!(out, "```mermaid").expect("write to string should not fail");
    out.push_str(graph);
    if !graph.ends_with('\n') {
        out.push('\n');
    }
    writeln!(out, "```").expect("write to string should not fail");
    writeln!(out).expect("write to string should not fail");
}

fn push_metric_diff_table(out: &mut String, overview: &DecompileAutotuneOverview<'_>) {
    writeln!(out, "## Source vs Optimized").expect("write to string should not fail");
    writeln!(out).expect("write to string should not fail");
    writeln!(out, "| metric | source | optimized | delta |")
        .expect("write to string should not fail");
    writeln!(out, "| --- | ---: | ---: | ---: |").expect("write to string should not fail");
    push_usize_delta_row(
        out,
        "parsed_instructions",
        overview.source.parsed_instruction_count,
        overview.optimized.parsed_instruction_count,
    );
    push_usize_delta_row(
        out,
        "unsupported_instructions",
        overview.source.unsupported_instruction_count,
        overview.optimized.unsupported_instruction_count,
    );
    push_usize_delta_row(
        out,
        "semantic_patterns",
        overview.source.semantic_pattern_count,
        overview.optimized.semantic_pattern_count,
    );
    writeln!(out).expect("write to string should not fail");
}

fn push_evidence_table(out: &mut String, overview: &DecompileAutotuneOverview<'_>) {
    writeln!(out, "## Evidence Diff").expect("write to string should not fail");
    writeln!(out).expect("write to string should not fail");
    writeln!(out, "| evidence | source | optimized | changed |")
        .expect("write to string should not fail");
    writeln!(out, "| --- | --- | --- | --- |").expect("write to string should not fail");
    push_bool_diff_row(
        out,
        "bf16_descriptor_load",
        overview.source.evidence.has_bf16_descriptor_load,
        overview.optimized.evidence.has_bf16_descriptor_load,
    );
    push_bool_diff_row(
        out,
        "f32_descriptor_load",
        overview.source.evidence.has_f32_descriptor_load,
        overview.optimized.evidence.has_f32_descriptor_load,
    );
    push_bool_diff_row(
        out,
        "bf16_widen",
        overview.source.evidence.has_bf16_widen,
        overview.optimized.evidence.has_bf16_widen,
    );
    push_bool_diff_row(
        out,
        "f32_mul_add",
        overview.source.evidence.has_f32_mul_add,
        overview.optimized.evidence.has_f32_mul_add,
    );
    push_bool_diff_row(
        out,
        "f32_fused_multiply_add",
        overview.source.evidence.has_f32_fused_multiply_add,
        overview.optimized.evidence.has_f32_fused_multiply_add,
    );
    push_bool_diff_row(
        out,
        "shared_load",
        overview.source.evidence.has_shared_load,
        overview.optimized.evidence.has_shared_load,
    );
    push_bool_diff_row(
        out,
        "shared_store",
        overview.source.evidence.has_shared_store,
        overview.optimized.evidence.has_shared_store,
    );
    push_bool_diff_row(
        out,
        "barrier",
        overview.source.evidence.has_barrier,
        overview.optimized.evidence.has_barrier,
    );
    push_bool_diff_row(
        out,
        "descriptor_store",
        overview.source.evidence.has_descriptor_store,
        overview.optimized.evidence.has_descriptor_store,
    );
    push_bool_diff_row(
        out,
        "warp_reduce_sum",
        overview.source.evidence.has_warp_reduce_sum,
        overview.optimized.evidence.has_warp_reduce_sum,
    );
    writeln!(
        out,
        "| pattern_categories | `{}` | `{}` | {} |",
        md_cell(&pattern_category_list(overview.source.evidence)),
        md_cell(&pattern_category_list(overview.optimized.evidence)),
        overview.source.evidence.pattern_categories
            != overview.optimized.evidence.pattern_categories
    )
    .expect("write to string should not fail");
    writeln!(out).expect("write to string should not fail");
}

fn push_positive_steps_table(out: &mut String, overview: &DecompileAutotuneOverview<'_>) {
    writeln!(out, "## Positive Search Steps").expect("write to string should not fail");
    writeln!(out).expect("write to string should not fail");
    writeln!(
        out,
        "| depth | generated | accepted | rejected | best_before | best_after | improvement | best_candidate_actions |"
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        "| ---: | ---: | ---: | ---: | --- | --- | ---: | --- |"
    )
    .expect("write to string should not fail");
    let mut wrote_any = false;
    for step in &overview.optimization.steps {
        let Some(improvement) = step.improvement else {
            continue;
        };
        if improvement <= 0.0 {
            continue;
        }
        wrote_any = true;
        writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {:.12} | `{}` |",
            step.depth,
            step.generated,
            step.accepted,
            step.rejected,
            md_cell(&format_score(step.best_before)),
            md_cell(&format_score(step.best_after)),
            improvement,
            md_cell(
                &step
                    .best_candidate
                    .as_ref()
                    .map(|candidate| format_action_trace(candidate.action_trace.as_slice()))
                    .unwrap_or_else(|| "none".to_string())
            )
        )
        .expect("write to string should not fail");
    }
    if !wrote_any {
        writeln!(
            out,
            "| - | - | - | - | - | - | - | no positive score delta recorded |"
        )
        .expect("write to string should not fail");
    }
    writeln!(out).expect("write to string should not fail");
}

fn push_action_trace_table(out: &mut String, actions: &[KernelScheduleAction]) {
    writeln!(out, "## Best Action Trace").expect("write to string should not fail");
    writeln!(out).expect("write to string should not fail");
    writeln!(out, "| index | op | axis | arg | materialization |")
        .expect("write to string should not fail");
    writeln!(out, "| ---: | --- | --- | --- | --- |").expect("write to string should not fail");
    if actions.is_empty() {
        writeln!(out, "| - | none | - | - | - |").expect("write to string should not fail");
    } else {
        for (index, action) in actions.iter().enumerate() {
            writeln!(
                out,
                "| {} | `{}` | {} | `{}` | `{}` |",
                index,
                action.op.label(),
                action
                    .axis
                    .map(|axis| axis.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                md_cell(&format_action_arg(action)),
                action.materialization.label()
            )
            .expect("write to string should not fail");
        }
    }
    writeln!(out).expect("write to string should not fail");
}

fn push_artifact_table(out: &mut String, overview: &DecompileAutotuneOverview<'_>) {
    writeln!(out, "## Artifacts").expect("write to string should not fail");
    writeln!(out).expect("write to string should not fail");
    writeln!(out, "| artifact | path |").expect("write to string should not fail");
    writeln!(out, "| --- | --- |").expect("write to string should not fail");
    push_path_row(out, "auto_report", Some(overview.auto_report_path));
    push_side_paths(out, overview.source);
    push_side_paths(out, overview.optimized);
    writeln!(out).expect("write to string should not fail");
}

fn push_side_paths(out: &mut String, side: DecompileAutotuneOverviewSide<'_>) {
    push_path_row(out, &format!("{}_source", side.label), side.source_path);
    push_path_row(out, &format!("{}_ptx", side.label), side.ptx_path);
    push_path_row(out, &format!("{}_cubin", side.label), side.cubin_path);
    push_path_row(out, &format!("{}_sass", side.label), Some(side.sass_path));
    push_path_row(out, &format!("{}_ir", side.label), Some(side.ir_path));
    push_path_row(
        out,
        &format!("{}_lifted_value_ir", side.label),
        Some(side.lifted_ir_path),
    );
    push_path_row(
        out,
        &format!("{}_analysis", side.label),
        Some(side.analysis_path),
    );
    push_path_row(
        out,
        &format!("{}_patterns", side.label),
        Some(side.pattern_path),
    );
    push_path_row(
        out,
        &format!("{}_side_by_side", side.label),
        Some(side.side_by_side_path),
    );
}

fn push_path_row(out: &mut String, label: &str, path: Option<&Path>) {
    writeln!(
        out,
        "| `{}` | `{}` |",
        md_cell(label),
        md_cell(
            &path
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "-".to_string())
        )
    )
    .expect("write to string should not fail");
}

fn push_usize_delta_row(out: &mut String, label: &str, source: usize, optimized: usize) {
    writeln!(
        out,
        "| `{}` | {} | {} | {} |",
        label,
        source,
        optimized,
        optimized as isize - source as isize
    )
    .expect("write to string should not fail");
}

fn push_bool_diff_row(out: &mut String, label: &str, source: bool, optimized: bool) {
    writeln!(
        out,
        "| `{}` | {} | {} | {} |",
        label,
        source,
        optimized,
        source != optimized
    )
    .expect("write to string should not fail");
}
