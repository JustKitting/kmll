use std::fmt::Write as _;

use super::{
    format::{
        evidence_summary, format_action_arg, format_score, mermaid_label, positive_step_count,
    },
    types::DecompileAutotuneOverview,
};

pub(super) fn render_mermaid_graph(overview: &DecompileAutotuneOverview<'_>) -> String {
    let mut out = String::new();
    writeln!(out, "flowchart LR").expect("write to string should not fail");
    push_node(
        &mut out,
        "source_sass",
        &format!(
            "source SASS<br/>symbol: {}<br/>instructions: {}",
            overview.source.symbol, overview.source.parsed_instruction_count
        ),
    );
    push_node(
        &mut out,
        "source_ir",
        &format!(
            "lifted IR<br/>unsupported: {}<br/>patterns: {}",
            overview.source.unsupported_instruction_count, overview.source.semantic_pattern_count
        ),
    );
    push_node(
        &mut out,
        "evidence",
        &format!(
            "typed evidence<br/>{}",
            evidence_summary(overview.source.evidence)
        ),
    );
    push_node(
        &mut out,
        "search",
        &format!(
            "autotune search<br/>explored: {} rejected: {}<br/>positive steps: {}",
            overview.optimization.explored,
            overview.optimization.rejected,
            positive_step_count(overview.optimization)
        ),
    );
    push_node(
        &mut out,
        "best",
        &format!(
            "best candidate<br/>symbol: {}<br/>score: {}",
            overview.optimized.symbol,
            format_score(overview.best.score)
        ),
    );
    push_node(
        &mut out,
        "optimized_sass",
        &format!(
            "optimized SASS<br/>instructions: {}",
            overview.optimized.parsed_instruction_count
        ),
    );
    push_node(
        &mut out,
        "optimized_ir",
        &format!(
            "optimized IR<br/>unsupported: {}<br/>patterns: {}",
            overview.optimized.unsupported_instruction_count,
            overview.optimized.semantic_pattern_count
        ),
    );
    writeln!(
        out,
        "    source_sass --> source_ir --> evidence --> search --> best --> optimized_sass --> optimized_ir"
    )
    .expect("write to string should not fail");

    if overview.best.action_trace.is_empty() {
        writeln!(out, "    search --> no_actions[\"no schedule actions\"]")
            .expect("write to string should not fail");
        return out;
    }

    writeln!(out, "    search --> action_0").expect("write to string should not fail");
    for (index, action) in overview.best.action_trace.iter().enumerate() {
        push_node(
            &mut out,
            &format!("action_{index}"),
            &format!(
                "{}: {}<br/>{}",
                index,
                action.op.label(),
                format_action_arg(action)
            ),
        );
        if index + 1 < overview.best.action_trace.len() {
            writeln!(out, "    action_{index} --> action_{}", index + 1)
                .expect("write to string should not fail");
        } else {
            writeln!(out, "    action_{index} --> best").expect("write to string should not fail");
        }
    }
    out
}

fn push_node(out: &mut String, id: &str, label: &str) {
    writeln!(out, "    {id}[\"{}\"]", mermaid_label(label))
        .expect("write to string should not fail");
}
