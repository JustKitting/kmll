use std::{
    error::Error,
    fmt::{self, Write as _},
    fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
    process,
};

use crate::{
    autotune::{compile_standalone_kernel_crate, standalone_cargo_toml, standalone_main_source},
    runtime,
};

mod fixtures;
mod ir;
mod sass;

pub use self::{
    fixtures::{SimpleKernelFixture, SimpleKernelFixtureKind, simple_kernel_fixtures},
    ir::{
        KernelIrFunction, KernelIrModule, KernelIrOp, KernelIrOpKind, SassMappingConfidence,
        lower_sass_module,
    },
    sass::{
        RegisterClass, SassFunction, SassInstruction, SassModule, SassOperand, SassOperandKind,
        SassParseError, SassPredicate, SassRegister, parse_nvdisasm_sass,
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureOptions {
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub fixtures: Vec<SimpleKernelFixtureKind>,
}

impl DecompileFixtureOptions {
    pub fn sm120_default() -> Self {
        Self {
            artifact_root: runtime::default_artifact_dir().join("decompile-fixtures"),
            compile_arch: "sm_120".to_string(),
            fixtures: vec![SimpleKernelFixtureKind::I32Add],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureReport {
    pub fixture: SimpleKernelFixtureKind,
    pub symbol: String,
    pub crate_dir: PathBuf,
    pub source_path: PathBuf,
    pub ptx_path: PathBuf,
    pub cubin_path: PathBuf,
    pub sass_path: PathBuf,
    pub ir_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassFileDecompileOptions {
    pub sass_path: PathBuf,
    pub source_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassFileDecompileReport {
    pub sass_path: PathBuf,
    pub source_path: Option<PathBuf>,
    pub ir_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
}

pub fn run_decompile_fixtures(
    options: &DecompileFixtureOptions,
) -> Result<Vec<DecompileFixtureReport>, Box<dyn Error>> {
    let mut reports = Vec::new();
    let fixtures = simple_kernel_fixtures();
    for requested in &options.fixtures {
        let fixture = fixtures
            .iter()
            .find(|fixture| fixture.kind == *requested)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("unknown decompile fixture {}", requested.name()),
                )
            })?;
        reports.push(run_decompile_fixture(options, fixture)?);
    }
    Ok(reports)
}

pub fn run_sass_file_decompile(
    options: &SassFileDecompileOptions,
) -> Result<SassFileDecompileReport, Box<dyn Error>> {
    let sass = fs::read_to_string(&options.sass_path)?;
    let source = match &options.source_path {
        Some(path) => Some((path.clone(), fs::read_to_string(path)?)),
        None => None,
    };
    let parsed = parse_nvdisasm_sass(&sass)?;
    let lowered = lower_sass_module(&parsed);
    let output_dir = options.output_dir.clone().unwrap_or_else(|| {
        options
            .sass_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    });
    fs::create_dir_all(&output_dir)?;
    let stem = options
        .sass_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("sass");
    let ir_path = output_dir.join(format!("{stem}.lifted.ir.txt"));
    fs::write(&ir_path, lowered.to_text().as_bytes())?;
    let side_by_side_path = output_dir.join(format!("{stem}.source-sass-ir.txt"));
    let side_by_side = render_sass_file_side_by_side(
        options.sass_path.as_path(),
        source
            .as_ref()
            .map(|(path, text)| (path.as_path(), text.as_str())),
        &sass,
        &lowered,
    );
    fs::write(&side_by_side_path, side_by_side.as_bytes())?;

    Ok(SassFileDecompileReport {
        sass_path: options.sass_path.clone(),
        source_path: options.source_path.clone(),
        ir_path,
        side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        unsupported_instruction_count: lowered.unsupported_instruction_count(),
    })
}

fn run_decompile_fixture(
    options: &DecompileFixtureOptions,
    fixture: &SimpleKernelFixture,
) -> Result<DecompileFixtureReport, Box<dyn Error>> {
    let fixture_dir = options.artifact_root.join(fixture.kind.name());
    let crate_dir = fixture_dir.join("standalone-crate");
    let source_path = crate_dir.join("src").join("main.rs");
    let cargo_toml_path = crate_dir.join("Cargo.toml");
    let package_stem = format!(
        "nn_rust_sass_fixture_{}",
        fixture.kind.name().replace('-', "_")
    );
    fs::create_dir_all(
        source_path
            .parent()
            .expect("source path should have a parent"),
    )?;
    fs::write(&cargo_toml_path, standalone_cargo_toml(&package_stem))?;
    fs::write(&source_path, standalone_main_source(fixture.source))?;

    let ptx_output_dir = fixture_dir.join("ptx");
    let target_dir = options.artifact_root.join("standalone-target");
    let compiled = compile_standalone_kernel_crate(
        &crate_dir,
        &ptx_output_dir,
        &package_stem,
        Some(&options.compile_arch),
        Some(&target_dir),
    )?;

    let cubin_path = fixture_dir.join(format!("{}.{}.cubin", fixture.symbol, options.compile_arch));
    run_checked(
        "ptxas",
        &[
            format!("-arch={}", options.compile_arch),
            "-o".to_string(),
            cubin_path.display().to_string(),
            compiled.ptx_path.display().to_string(),
        ],
        &fixture_dir,
    )?;

    let sass_path = fixture_dir.join(format!(
        "{}.{}.nvdisasm.sass",
        fixture.symbol, options.compile_arch
    ));
    let sass = run_capture(
        "nvdisasm",
        &[cubin_path.display().to_string()],
        &fixture_dir,
    )?;
    fs::write(&sass_path, sass.as_bytes())?;

    let parsed = parse_nvdisasm_sass(&sass)?;
    let lowered = lower_sass_module(&parsed);
    let ir_text = lowered.to_text();
    let ir_path = fixture_dir.join("lifted.ir.txt");
    fs::write(&ir_path, ir_text.as_bytes())?;

    let side_by_side = render_side_by_side(fixture, &sass, &lowered);
    let side_by_side_path = fixture_dir.join("source-sass-ir.txt");
    fs::write(&side_by_side_path, side_by_side.as_bytes())?;

    Ok(DecompileFixtureReport {
        fixture: fixture.kind,
        symbol: fixture.symbol.to_string(),
        crate_dir,
        source_path,
        ptx_path: compiled.ptx_path,
        cubin_path,
        sass_path,
        ir_path,
        side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        unsupported_instruction_count: lowered.unsupported_instruction_count(),
    })
}

fn run_checked(command: &str, args: &[String], current_dir: &Path) -> Result<(), Box<dyn Error>> {
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

fn run_capture(
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

impl fmt::Display for SimpleKernelFixtureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests;
