use std::{
    error::Error,
    fmt::{self, Write as _},
    fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
    process,
};

use nn_rust_inference::runtime;

use crate::autotune::{
    compile_standalone_kernel_crate, standalone_cargo_toml, standalone_main_source,
};

mod analysis;
mod coverage;
mod coverage_compare;
mod fixtures;
mod ir;
mod known_opcodes;
mod lifted;
mod patterns;
mod ptx_probes;
mod sass;

pub use self::{
    analysis::{
        SassAnalysisFunction, SassAnalysisModule, SassBasicBlock, SassBlockTerminator, SassCfgEdge,
        SassCfgEdgeKind, SassDataflowOp, SassDefUseEdge, SassDominatorBlock, SassLiveRange,
        SassMemoryAccess, SassMemoryAccessKind, SassNaturalLoop, SassReachingUse, SassRegion,
        SassRegionKind, SassRegionPath, SassSsaValue, SassValueOp, analyze_sass_ir,
    },
    coverage::{
        SassCoverageBasicBlock, SassCoverageCfgEdge, SassCoverageDataflowOp,
        SassCoverageDefUseEdge, SassCoverageDominatorBlock, SassCoverageFileReport,
        SassCoverageLiveRange, SassCoverageMemoryAccess, SassCoverageNaturalLoop,
        SassCoverageOptions, SassCoverageReachingUse, SassCoverageRegion, SassCoverageReport,
        SassCoverageSemanticPattern, SassCoverageSsaValue, SassCoverageValueOp,
        SassOpcodeCatalogEntry, SassOpcodeCount, SassOpcodeProbeTarget, SassUnsupportedInstruction,
        run_sass_coverage_scan,
    },
    coverage_compare::{
        SassCoverageComparisonOptions, SassCoverageComparisonReport, SassCoverageOpcodeDelta,
        SassCoverageProbeTargetDelta, run_sass_coverage_comparison,
    },
    fixtures::{
        SimpleKernelFixture, SimpleKernelFixtureKind, all_simple_kernel_fixture_kinds,
        simple_kernel_fixtures,
    },
    ir::{
        AggregateOperand, AggregateOperandKind, ControlTarget, ControlTargetKind, ImmediateValue,
        KernelIrFunction, KernelIrModule, KernelIrOp, KernelIrOpKind, MemoryAccessInfo,
        MemoryAddress, MemoryAddressBase, MemoryAddressImmediate, MemoryAddressImmediateKind,
        MemoryAddressKind, MemorySpace, PredicateCondition, PredicateConditionKind, RegisterRef,
        RegisterRefKind, SassMappingConfidence, ScalarOperand, ScalarOperandKind, lift_sass_module,
    },
    known_opcodes::{KnownSassOpcode, known_sass_opcodes},
    lifted::{
        SassLiftedFunction, SassLiftedModule, SassLiftedOp, SassLiftedOpClass, SassLiftedOpKind,
        SassLiftedSemantics, SassLiftedValueRef, lift_sass_value_ir,
    },
    patterns::{
        SassPatternConfidence, SassPatternFunction, SassPatternModule, SassSemanticPattern,
        SassSemanticPatternKind, recover_sass_patterns,
    },
    ptx_probes::{
        PtxDecompileProbe, PtxDecompileProbeKind, all_ptx_decompile_probe_kinds,
        ptx_decompile_probes,
    },
    sass::{
        RegisterClass, SassFunction, SassInstruction, SassModule, SassOperand, SassOperandKind,
        SassParseError, SassPredicate, SassRegister, SassSourcePosition, parse_nvidia_sass,
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
pub struct DecompileFixtureCoverageOptions {
    pub fixture_options: DecompileFixtureOptions,
    pub coverage_output_dir: PathBuf,
}

impl DecompileFixtureCoverageOptions {
    pub fn sm120_all_default() -> Self {
        let artifact_root = runtime::default_artifact_dir().join("decompile-fixtures");
        Self {
            fixture_options: DecompileFixtureOptions {
                artifact_root: artifact_root.clone(),
                compile_arch: "sm_120".to_string(),
                fixtures: all_simple_kernel_fixture_kinds(),
            },
            coverage_output_dir: artifact_root.join("coverage"),
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
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub cfg_block_count: usize,
    pub cfg_edge_count: usize,
    pub dominator_block_count: usize,
    pub natural_loop_count: usize,
    pub region_count: usize,
    pub reaching_use_count: usize,
    pub ssa_value_count: usize,
    pub def_use_edge_count: usize,
    pub value_op_count: usize,
    pub lifted_op_count: usize,
    pub live_range_count: usize,
    pub memory_access_count: usize,
    pub semantic_pattern_count: usize,
    pub unsupported_instruction_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureCoverageReport {
    pub fixture_reports: Vec<DecompileFixtureReport>,
    pub coverage_report: SassCoverageReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompilePtxProbeOptions {
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub probes: Vec<PtxDecompileProbeKind>,
}

impl DecompilePtxProbeOptions {
    pub fn sm120_default() -> Self {
        Self {
            artifact_root: runtime::default_artifact_dir().join("decompile-probes"),
            compile_arch: "sm_120".to_string(),
            probes: vec![
                PtxDecompileProbeKind::TensorCoreHmma,
                PtxDecompileProbeKind::TensorCoreImma,
                PtxDecompileProbeKind::TensorCoreDmma,
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompilePtxProbeReport {
    pub probe: PtxDecompileProbeKind,
    pub symbol: String,
    pub probe_dir: PathBuf,
    pub ptx_path: PathBuf,
    pub cubin_path: PathBuf,
    pub nvdisasm_sass_path: PathBuf,
    pub cuobjdump_sass_path: PathBuf,
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
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub cfg_block_count: usize,
    pub cfg_edge_count: usize,
    pub dominator_block_count: usize,
    pub natural_loop_count: usize,
    pub region_count: usize,
    pub reaching_use_count: usize,
    pub ssa_value_count: usize,
    pub def_use_edge_count: usize,
    pub value_op_count: usize,
    pub lifted_op_count: usize,
    pub live_range_count: usize,
    pub memory_access_count: usize,
    pub semantic_pattern_count: usize,
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

pub fn run_decompile_fixture_coverage(
    options: &DecompileFixtureCoverageOptions,
) -> Result<DecompileFixtureCoverageReport, Box<dyn Error>> {
    let fixture_reports = run_decompile_fixtures(&options.fixture_options)?;
    let coverage_report = run_sass_coverage_scan(&SassCoverageOptions {
        root: options.fixture_options.artifact_root.clone(),
        output_dir: options.coverage_output_dir.clone(),
    })?;
    Ok(DecompileFixtureCoverageReport {
        fixture_reports,
        coverage_report,
    })
}

pub fn run_decompile_ptx_probes(
    options: &DecompilePtxProbeOptions,
) -> Result<Vec<DecompilePtxProbeReport>, Box<dyn Error>> {
    let mut reports = Vec::new();
    let probes = ptx_decompile_probes();
    for requested in &options.probes {
        let probe = probes
            .iter()
            .find(|probe| probe.kind == *requested)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("unknown PTX decompile probe {}", requested.name()),
                )
            })?;
        reports.push(run_decompile_ptx_probe(options, probe)?);
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
    let parsed = parse_nvidia_sass(&sass)?;
    let project_ir = lift_sass_module(&parsed);
    let analysis = analyze_sass_ir(&project_ir);
    let lifted = lift_sass_value_ir(&project_ir, &analysis);
    let patterns = recover_sass_patterns(&project_ir);
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
    fs::write(&ir_path, project_ir.to_text().as_bytes())?;
    let lifted_ir_path = output_dir.join(format!("{stem}.lifted-value-ir.txt"));
    fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;
    let analysis_path = output_dir.join(format!("{stem}.analysis.txt"));
    fs::write(&analysis_path, analysis.to_text().as_bytes())?;
    let pattern_path = output_dir.join(format!("{stem}.patterns.txt"));
    fs::write(&pattern_path, patterns.to_text().as_bytes())?;
    let side_by_side_path = output_dir.join(format!("{stem}.source-sass-ir.txt"));
    let side_by_side = render_sass_file_side_by_side(
        options.sass_path.as_path(),
        source
            .as_ref()
            .map(|(path, text)| (path.as_path(), text.as_str())),
        &sass,
        &project_ir,
    );
    fs::write(&side_by_side_path, side_by_side.as_bytes())?;

    Ok(SassFileDecompileReport {
        sass_path: options.sass_path.clone(),
        source_path: options.source_path.clone(),
        ir_path,
        lifted_ir_path,
        analysis_path,
        pattern_path,
        side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        cfg_block_count: analysis.block_count(),
        cfg_edge_count: analysis.edge_count(),
        dominator_block_count: analysis.dominator_block_count(),
        natural_loop_count: analysis.natural_loop_count(),
        region_count: analysis.region_count(),
        reaching_use_count: analysis.reaching_use_count(),
        ssa_value_count: analysis.ssa_value_count(),
        def_use_edge_count: analysis.def_use_edge_count(),
        value_op_count: analysis.value_op_count(),
        lifted_op_count: lifted.op_count(),
        live_range_count: analysis.live_range_count(),
        memory_access_count: analysis.memory_access_count(),
        semantic_pattern_count: patterns.pattern_count(),
        unsupported_instruction_count: project_ir.unsupported_instruction_count(),
    })
}

fn run_decompile_ptx_probe(
    options: &DecompilePtxProbeOptions,
    probe: &PtxDecompileProbe,
) -> Result<DecompilePtxProbeReport, Box<dyn Error>> {
    let probe_dir = options.artifact_root.join(probe.kind.name());
    fs::create_dir_all(&probe_dir)?;
    let ptx_path = probe_dir.join(format!("{}.ptx", probe.symbol));
    fs::write(&ptx_path, probe.source.as_bytes())?;

    let cubin_path = probe_dir.join(format!("{}.{}.cubin", probe.symbol, options.compile_arch));
    run_checked(
        "ptxas",
        &[
            format!("-arch={}", options.compile_arch),
            "-o".to_string(),
            cubin_path.display().to_string(),
            ptx_path.display().to_string(),
        ],
        &probe_dir,
    )?;

    let nvdisasm_sass_path = probe_dir.join(format!(
        "{}.{}.nvdisasm.sass",
        probe.symbol, options.compile_arch
    ));
    let nvdisasm_sass = run_capture("nvdisasm", &[cubin_path.display().to_string()], &probe_dir)?;
    fs::write(&nvdisasm_sass_path, nvdisasm_sass.as_bytes())?;

    let cuobjdump_sass_path = probe_dir.join(format!(
        "{}.{}.cuobjdump.sass",
        probe.symbol, options.compile_arch
    ));
    let cuobjdump_sass = run_capture(
        "cuobjdump",
        &["--dump-sass".to_string(), cubin_path.display().to_string()],
        &probe_dir,
    )?;
    fs::write(&cuobjdump_sass_path, cuobjdump_sass.as_bytes())?;

    let nvdisasm_module = parse_nvidia_sass(&nvdisasm_sass)?;
    let cuobjdump_module = parse_nvidia_sass(&cuobjdump_sass)?;
    let nvdisasm_ir = lift_sass_module(&nvdisasm_module);
    let cuobjdump_ir = lift_sass_module(&cuobjdump_module);

    Ok(DecompilePtxProbeReport {
        probe: probe.kind,
        symbol: probe.symbol.to_string(),
        probe_dir,
        ptx_path,
        cubin_path,
        nvdisasm_sass_path,
        cuobjdump_sass_path,
        parsed_instruction_count: nvdisasm_module.instruction_count()
            + cuobjdump_module.instruction_count(),
        unsupported_instruction_count: nvdisasm_ir.unsupported_instruction_count()
            + cuobjdump_ir.unsupported_instruction_count(),
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

    let parsed = parse_nvidia_sass(&sass)?;
    let project_ir = lift_sass_module(&parsed);
    let analysis = analyze_sass_ir(&project_ir);
    let lifted = lift_sass_value_ir(&project_ir, &analysis);
    let patterns = recover_sass_patterns(&project_ir);
    let ir_text = project_ir.to_text();
    let ir_path = fixture_dir.join("lifted.ir.txt");
    fs::write(&ir_path, ir_text.as_bytes())?;
    let lifted_ir_path = fixture_dir.join("lifted-value-ir.txt");
    fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;
    let analysis_path = fixture_dir.join("analysis.txt");
    fs::write(&analysis_path, analysis.to_text().as_bytes())?;
    let pattern_path = fixture_dir.join("patterns.txt");
    fs::write(&pattern_path, patterns.to_text().as_bytes())?;

    let side_by_side = render_side_by_side(fixture, &sass, &project_ir);
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
        lifted_ir_path,
        analysis_path,
        pattern_path,
        side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        cfg_block_count: analysis.block_count(),
        cfg_edge_count: analysis.edge_count(),
        dominator_block_count: analysis.dominator_block_count(),
        natural_loop_count: analysis.natural_loop_count(),
        region_count: analysis.region_count(),
        reaching_use_count: analysis.reaching_use_count(),
        ssa_value_count: analysis.ssa_value_count(),
        def_use_edge_count: analysis.def_use_edge_count(),
        value_op_count: analysis.value_op_count(),
        lifted_op_count: lifted.op_count(),
        live_range_count: analysis.live_range_count(),
        memory_access_count: analysis.memory_access_count(),
        semantic_pattern_count: patterns.pattern_count(),
        unsupported_instruction_count: project_ir.unsupported_instruction_count(),
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
