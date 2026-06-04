use std::{
    env,
    error::Error,
    fmt::Write as _,
    io,
    path::{Path, PathBuf},
    process,
};

use super::{KernelIrModule, SimpleKernelFixture};

pub(super) fn run_checked(
    command: &str,
    args: &[String],
    current_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let output = process::Command::new(command)
        .args(args)
        .current_dir(current_dir)
        .output()?;
    if output.status.success() {
        return Ok(());
    }
    Err(Box::new(io::Error::other(format!(
        "{command} failed with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))))
}

pub(super) fn run_capture(
    command: &str,
    args: &[String],
    current_dir: &Path,
) -> Result<String, Box<dyn Error>> {
    let output = process::Command::new(command)
        .args(args)
        .current_dir(current_dir)
        .output()?;
    if !output.status.success() {
        return Err(Box::new(io::Error::other(format!(
            "{command} failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))));
    }
    Ok(String::from_utf8(output.stdout)?)
}

pub(super) fn absolute_path(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

pub fn render_side_by_side(
    fixture: &SimpleKernelFixture,
    sass: &str,
    ir: &KernelIrModule,
) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# fixture={} symbol={} behavior={}",
        fixture.kind.name(),
        fixture.symbol,
        fixture.behavior
    )
    .expect("write to string");
    writeln!(out).expect("write to string");
    writeln!(out, "## source").expect("write to string");
    writeln!(out, "```rust").expect("write to string");
    writeln!(out, "{}", fixture.source.trim_end()).expect("write to string");
    writeln!(out, "```").expect("write to string");
    writeln!(out).expect("write to string");
    writeln!(out, "## sass").expect("write to string");
    writeln!(out, "```sass").expect("write to string");
    writeln!(out, "{}", sass.trim_end()).expect("write to string");
    writeln!(out, "```").expect("write to string");
    writeln!(out).expect("write to string");
    writeln!(out, "## project-ir").expect("write to string");
    writeln!(out, "```text").expect("write to string");
    writeln!(out, "{}", ir.to_text().trim_end()).expect("write to string");
    writeln!(out, "```").expect("write to string");
    out
}

pub fn render_sass_file_side_by_side(
    sass_path: &Path,
    source: Option<(&Path, &str)>,
    sass: &str,
    ir: &KernelIrModule,
) -> String {
    let mut out = String::new();
    writeln!(out, "# sass={}", sass_path.display()).expect("write to string");
    if let Some((source_path, _)) = source {
        writeln!(out, "# source={}", source_path.display()).expect("write to string");
    }
    writeln!(out).expect("write to string");
    if let Some((_, source_text)) = source {
        writeln!(out, "## source").expect("write to string");
        writeln!(out, "```rust").expect("write to string");
        writeln!(out, "{}", source_text.trim_end()).expect("write to string");
        writeln!(out, "```").expect("write to string");
        writeln!(out).expect("write to string");
    }
    writeln!(out, "## sass").expect("write to string");
    writeln!(out, "```sass").expect("write to string");
    writeln!(out, "{}", sass.trim_end()).expect("write to string");
    writeln!(out, "```").expect("write to string");
    writeln!(out).expect("write to string");
    writeln!(out, "## project-ir").expect("write to string");
    writeln!(out, "```text").expect("write to string");
    writeln!(out, "{}", ir.to_text().trim_end()).expect("write to string");
    writeln!(out, "```").expect("write to string");
    out
}
