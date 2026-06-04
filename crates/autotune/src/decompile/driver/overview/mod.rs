use std::{error::Error, fs, path::Path};

mod format;
mod graph;
mod markdown;
mod types;

#[cfg(test)]
mod tests;

pub(super) use types::{DecompileAutotuneOverview, DecompileAutotuneOverviewSide};

use self::types::DecompileAutotuneOverviewPaths;

pub(super) fn write_decompile_autotune_overview(
    auto_report_path: &Path,
    overview: &DecompileAutotuneOverview<'_>,
) -> Result<DecompileAutotuneOverviewPaths, Box<dyn Error>> {
    let paths = overview_paths(auto_report_path);
    fs::create_dir_all(
        paths
            .markdown_path
            .parent()
            .expect("overview path should have a parent"),
    )?;
    let graph = graph::render_mermaid_graph(overview);
    fs::write(&paths.graph_path, graph.as_bytes())?;
    fs::write(
        &paths.markdown_path,
        markdown::render_markdown_overview(overview, &graph).as_bytes(),
    )?;
    Ok(paths)
}

fn overview_paths(auto_report_path: &Path) -> DecompileAutotuneOverviewPaths {
    let parent = auto_report_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = auto_report_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("auto-search-report");
    DecompileAutotuneOverviewPaths {
        markdown_path: parent.join(format!("{stem}.overview.md")),
        graph_path: parent.join(format!("{stem}.overview.mmd")),
    }
}
