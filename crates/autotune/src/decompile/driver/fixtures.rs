use std::{
    error::Error,
    fs,
    io::{self, ErrorKind},
};

use crate::autotune::{
    compile_standalone_kernel_crate, standalone_cargo_toml, standalone_main_source,
};

use super::super::{
    SassCoverageOptions, SimpleKernelFixture, analyze_sass_ir, driver_support::render_side_by_side,
    driver_support::run_capture, driver_support::run_checked, lift_sass_module, lift_sass_value_ir,
    parse_nvidia_sass, recover_sass_patterns, run_sass_coverage_scan, simple_kernel_fixtures,
};
use super::types::{
    DecompileFixtureCoverageOptions, DecompileFixtureCoverageReport, DecompileFixtureOptions,
    DecompileFixtureReport,
};

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
