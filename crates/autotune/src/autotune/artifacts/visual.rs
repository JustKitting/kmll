use std::{fmt::Write as _, fs, path::Path};

use nn_rust_profiling::{
    LaunchDimensions, OptimizationActionArg, OptimizationActionSpaceSet, OptimizationActionSpec,
    OptimizationCandidateMaterialization, OptimizationScore, OptimizationScoreSource,
};

use super::super::*;
use super::types::EmittedSearchReport;

pub(super) fn write_auto_search_report_visuals(
    report_path: &Path,
    report: &AutoOptimizationSearchReport,
    emitted: &mut EmittedSearchReport,
) -> Result<(), KernelGenerationError> {
    let svg_path = report_path.with_extension("svg");
    let html_path = report_path.with_extension("html");
    let svg = render_auto_search_report_svg(report);
    let html = render_auto_search_report_html(
        &svg_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("report.svg"),
        &report_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("report.json"),
    );
    fs::write(&svg_path, svg.as_bytes())?;
    fs::write(&html_path, html.as_bytes())?;
    emitted.visual_svg_path = Some(svg_path);
    emitted.visual_svg_bytes = Some(svg.len());
    emitted.visual_html_path = Some(html_path);
    emitted.visual_html_bytes = Some(html.len());
    Ok(())
}

fn render_auto_search_report_svg(report: &AutoOptimizationSearchReport) -> String {
    let beam_len = report.beam.len().min(6);
    let step_len = report.steps.len().min(8);
    let height = 640 + beam_len as u32 * 68 + step_len as u32 * 34;
    let best_score = report.best.as_ref().and_then(|candidate| candidate.score);
    let max_score = report
        .beam
        .iter()
        .filter_map(|candidate| candidate.score.map(|score| score.value))
        .filter(|value| value.is_finite() && *value >= 0.0)
        .fold(0.0_f64, f64::max);
    let action_count = report
        .action_space
        .as_ref()
        .map(OptimizationActionSpaceSet::action_count);
    let mut out = String::new();
    writeln!(
        out,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" viewBox="0 0 1280 {height}" role="img" aria-labelledby="title desc">"#
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        r#"<title id="title">Autotune Search Report</title><desc id="desc">Generated visualization for an autotune search report.</desc>"#
    )
    .expect("write to string should not fail");
    out.push_str(STYLE);
    writeln!(
        out,
        r#"<rect class="bg" x="0" y="0" width="1280" height="{height}"/>"#
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        r#"<text class="title" x="54" y="64">Autotune Search Report</text>"#
    )
    .expect("write to string should not fail");
    writeln!(
        out,
        r#"<text class="subtitle" x="54" y="94">{}</text>"#,
        xml_escape(&report.family)
    )
    .expect("write to string should not fail");

    panel(&mut out, 54, 122, 1172, 162);
    label_value(&mut out, 78, 154, "exit", report.exit_reason.label());
    label_value(&mut out, 260, 154, "explored", &report.explored.to_string());
    label_value(&mut out, 442, 154, "rejected", &report.rejected.to_string());
    label_value(
        &mut out,
        624,
        154,
        "duplicates",
        &report.duplicates.to_string(),
    );
    label_value(
        &mut out,
        806,
        154,
        "action space",
        &action_count
            .map(|count| count.to_string())
            .unwrap_or_else(|| "not emitted".to_string()),
    );
    text(&mut out, "small", 78, 214, "config");
    text(
        &mut out,
        "mono",
        78,
        239,
        &format!(
            "beam={} max_steps={} require_launchable={}",
            report.config.beam_width, report.config.max_steps, report.config.require_launchable
        ),
    );
    text(
        &mut out,
        "mono",
        78,
        262,
        &format!(
            "min_delta={}",
            compact_f64(report.config.min_score_improvement)
        ),
    );
    label_value(
        &mut out,
        806,
        214,
        "best score",
        &best_score
            .map(format_score)
            .unwrap_or_else(|| "none".to_string()),
    );

    panel(&mut out, 54, 336, 1172, 118);
    if let Some(best) = &report.best {
        text(&mut out, "label", 78, 368, "Best Candidate");
        text(
            &mut out,
            "mono",
            78,
            394,
            &format!(
                "{} {}",
                best.launch.kernel,
                materialization_label(best.materialization)
            ),
        );
        text(
            &mut out,
            "small",
            78,
            419,
            &format!(
                "grid={} block={} shared={}B launchable={} actions={}",
                dim3(&best.launch.grid_dim),
                dim3(&best.launch.block_dim),
                best.launch.shared_mem_bytes,
                best.launchable,
                action_trace_summary(&best.action_trace)
            ),
        );
        score_bar(&mut out, 760, 378, 360, best.score, max_score);
    } else {
        text(&mut out, "label", 78, 388, "No best candidate was selected");
        text(
            &mut out,
            "small",
            78,
            416,
            "The search report did not contain a best candidate.",
        );
    }

    let mut y = 496_u32;
    text(&mut out, "section", 54, y, "Beam Candidates");
    y += 20;
    if report.beam.is_empty() {
        panel(&mut out, 54, y, 1172, 48);
        text(
            &mut out,
            "small",
            78,
            y + 30,
            "No beam candidates were recorded.",
        );
        y += 68;
    } else {
        for (index, candidate) in report.beam.iter().take(beam_len).enumerate() {
            panel(&mut out, 54, y, 1172, 56);
            text(
                &mut out,
                "label",
                78,
                y + 23,
                &format!("#{} {}", index, candidate.launch.kernel),
            );
            text(
                &mut out,
                "small",
                78,
                y + 45,
                &format!(
                    "{} score={} actions={}",
                    materialization_label(candidate.materialization),
                    candidate
                        .score
                        .map(format_score)
                        .unwrap_or_else(|| "none".to_string()),
                    action_trace_summary(&candidate.action_trace)
                ),
            );
            score_bar(&mut out, 760, y + 19, 360, candidate.score, max_score);
            y += 68;
        }
    }

    y += 28;
    text(&mut out, "section", 54, y, "Search Steps");
    y += 20;
    if report.steps.is_empty() {
        panel(&mut out, 54, y, 1172, 48);
        text(
            &mut out,
            "small",
            78,
            y + 30,
            "No incremental search steps were recorded.",
        );
    } else {
        for step in report.steps.iter().take(step_len) {
            writeln!(
                out,
                r#"<rect class="row" x="54" y="{y}" width="1172" height="26" rx="5"/>"#
            )
            .expect("write to string should not fail");
            text(
                &mut out,
                "small",
                78,
                y + 18,
                &format!(
                    "depth={} input={} generated={} accepted={} rejected={} duplicates={} before={} after={} improvement={}",
                    step.depth,
                    step.input_beam_len,
                    step.generated,
                    step.accepted,
                    step.rejected,
                    step.duplicates,
                    step.best_before
                        .map(format_score)
                        .unwrap_or_else(|| "none".to_string()),
                    step.best_after
                        .map(format_score)
                        .unwrap_or_else(|| "none".to_string()),
                    step.improvement
                        .map(compact_f64)
                        .unwrap_or_else(|| "none".to_string())
                ),
            );
            y += 34;
        }
    }

    writeln!(
        out,
        r#"<text class="foot" x="54" y="{}">Generated from AutoOptimizationSearchReport. Lower score is better.</text>"#,
        height - 22
    )
    .expect("write to string should not fail");
    out.push_str("</svg>\n");
    out
}

fn render_auto_search_report_html(svg_file: &str, json_file: &str) -> String {
    format!(
        r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Autotune Search Report</title>
    <style>
      body {{ margin: 0; background: #f7f8fb; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
      main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
      img {{ display: block; width: 100%; height: auto; border: 1px solid #d9dee8; background: #fff; }}
      p {{ color: #4a5568; line-height: 1.45; }}
      a {{ color: #1f6feb; }}
    </style>
  </head>
  <body>
    <main>
      <img src="{}" alt="Autotune search report visualization">
      <p>Generated from <a href="{}">{}</a>.</p>
    </main>
  </body>
</html>
"#,
        html_escape(svg_file),
        html_escape(json_file),
        html_escape(json_file)
    )
}

fn panel(out: &mut String, x: u32, y: u32, width: u32, height: u32) {
    writeln!(
        out,
        r#"<rect class="panel" x="{x}" y="{y}" width="{width}" height="{height}" rx="8"/>"#
    )
    .expect("write to string should not fail");
}

fn label_value(out: &mut String, x: u32, y: u32, label: &str, value: &str) {
    text(out, "small", x, y, label);
    text(out, "metric", x, y + 28, value);
}

fn text(out: &mut String, class: &str, x: u32, y: u32, value: &str) {
    writeln!(
        out,
        r#"<text class="{class}" x="{x}" y="{y}">{}</text>"#,
        xml_escape(value)
    )
    .expect("write to string should not fail");
}

fn score_bar(
    out: &mut String,
    x: u32,
    y: u32,
    width: u32,
    score: Option<OptimizationScore>,
    max_score: f64,
) {
    writeln!(
        out,
        r#"<rect class="bar-track" x="{x}" y="{y}" width="{width}" height="16" rx="4"/>"#
    )
    .expect("write to string should not fail");
    if let Some(score) = score {
        let bar_width = if max_score > 0.0 {
            ((score.value.max(0.0) / max_score).min(1.0) * f64::from(width)).max(2.0)
        } else {
            f64::from(width)
        };
        writeln!(
            out,
            r#"<rect class="bar" x="{x}" y="{y}" width="{bar_width:.1}" height="16" rx="4"/>"#
        )
        .expect("write to string should not fail");
        text(out, "small", x + width + 14, y + 13, &format_score(score));
    } else {
        text(out, "small", x + width + 14, y + 13, "unscored");
    }
}

fn format_score(score: OptimizationScore) -> String {
    match score.source {
        OptimizationScoreSource::Measured => format_seconds(score.value),
        OptimizationScoreSource::Heuristic => format!("{} heuristic", compact_f64(score.value)),
    }
}

fn format_seconds(seconds: f64) -> String {
    if seconds < 0.001 {
        format!("{:.3} us", seconds * 1_000_000.0)
    } else if seconds < 1.0 {
        format!("{:.3} ms", seconds * 1_000.0)
    } else {
        format!("{:.6} s", seconds)
    }
}

fn compact_f64(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else if value.abs() >= 1000.0 || value.abs() < 0.001 {
        format!("{value:.6e}")
    } else {
        let formatted = format!("{value:.6}");
        formatted
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

fn materialization_label(materialization: OptimizationCandidateMaterialization) -> &'static str {
    materialization.label()
}

fn action_trace_summary(actions: &[OptimizationActionSpec]) -> String {
    if actions.is_empty() {
        return "none".to_string();
    }
    let mut parts = actions
        .iter()
        .take(4)
        .map(action_summary)
        .collect::<Vec<_>>();
    if actions.len() > parts.len() {
        parts.push(format!("+{} more", actions.len() - parts.len()));
    }
    parts.join(" -> ")
}

fn action_summary(action: &OptimizationActionSpec) -> String {
    let axis = action
        .axis
        .map(|axis| format!("axis={axis} "))
        .unwrap_or_default();
    format!(
        "{}({}{})",
        action.op.label(),
        axis,
        action_arg_summary(&action.arg)
    )
}

fn action_arg_summary(arg: &OptimizationActionArg) -> String {
    match arg {
        OptimizationActionArg::Factor(factor) => format!("factor={factor}"),
        OptimizationActionArg::Tile3d { m, n, k } => format!("{m}x{n}x{k}"),
        OptimizationActionArg::AxisOrder(axes) => format!("order={axes:?}"),
        OptimizationActionArg::AxisPair { axis_a, axis_b } => {
            format!("axis_a={axis_a} axis_b={axis_b}")
        }
    }
}

fn dim3(dim: &LaunchDimensions) -> String {
    format!("{}x{}x{}", dim.x, dim.y, dim.z)
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn html_escape(value: &str) -> String {
    xml_escape(value)
}

const STYLE: &str = r#"<defs>
  <style>
    .bg { fill: #f7f8fb; }
    .panel { fill: #ffffff; stroke: #d9dee8; stroke-width: 1.2; }
    .row { fill: #ffffff; stroke: #e2e8f0; stroke-width: 1; }
    .title { fill: #18202f; font: 700 34px system-ui, -apple-system, Segoe UI, sans-serif; }
    .subtitle { fill: #4a5568; font: 500 16px system-ui, -apple-system, Segoe UI, sans-serif; }
    .section { fill: #18202f; font: 800 18px system-ui, -apple-system, Segoe UI, sans-serif; }
    .label { fill: #2d3748; font: 700 15px system-ui, -apple-system, Segoe UI, sans-serif; }
    .small { fill: #4a5568; font: 500 13px system-ui, -apple-system, Segoe UI, sans-serif; }
    .mono { fill: #223046; font: 600 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .metric { fill: #111827; font: 800 20px system-ui, -apple-system, Segoe UI, sans-serif; }
    .foot { fill: #718096; font: 500 12px system-ui, -apple-system, Segoe UI, sans-serif; }
    .bar-track { fill: #e8edf5; }
    .bar { fill: #1f6feb; }
  </style>
</defs>
"#;
