use std::path::PathBuf;

use nn_rust_inference::{
    decompile::{
        DecompileFixtureOptions, SassFileDecompileOptions, SimpleKernelFixtureKind,
        run_decompile_fixtures, run_sass_file_decompile,
    },
    runtime,
};

use crate::{AppResult, invalid_input, parse_required_flag_value};

const DECOMPILE_FIXTURES_USAGE: &str =
    "kernel-decompile-fixtures [--fixture NAME|all] [--artifact-root PATH] [--compile-arch sm_120]";
const DECOMPILE_SASS_USAGE: &str =
    "kernel-decompile-sass SASS_PATH [--source PATH] [--out-dir PATH]";

pub(crate) fn run_kernel_decompile_sass(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    if index >= args.len() {
        return Err(invalid_input(format!(
            "kernel-decompile-sass requires SASS_PATH; usage: {DECOMPILE_SASS_USAGE}"
        )));
    }
    let sass_path = PathBuf::from(&args[index]);
    index += 1;
    let mut source_path = None;
    let mut output_dir = None;

    while index < args.len() {
        match args[index].as_str() {
            "--source" => {
                source_path = Some(PathBuf::from(parse_required_flag_value(
                    args, &mut index, "--source",
                )?));
            }
            "--out-dir" => {
                output_dir = Some(PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--out-dir",
                )?));
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-sass unknown argument {flag:?}; usage: {DECOMPILE_SASS_USAGE}"
                )));
            }
            extra => {
                return Err(invalid_input(format!(
                    "kernel-decompile-sass unexpected argument {extra:?}; usage: {DECOMPILE_SASS_USAGE}"
                )));
            }
        }
    }

    let report = run_sass_file_decompile(&SassFileDecompileOptions {
        sass_path,
        source_path,
        output_dir,
    })?;
    println!(
        "kernel_decompile_sass parsed_instructions={} unsupported_instructions={} sass_path={} ir_path={} side_by_side_path={}",
        report.parsed_instruction_count,
        report.unsupported_instruction_count,
        report.sass_path.display(),
        report.ir_path.display(),
        report.side_by_side_path.display(),
    );
    Ok(())
}

pub(crate) fn run_kernel_decompile_fixtures(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let mut artifact_root = runtime::default_artifact_dir().join("decompile-fixtures");
    let mut compile_arch = "sm_120".to_string();
    let mut fixtures = Vec::new();

    while index < args.len() {
        match args[index].as_str() {
            "--artifact-root" => {
                artifact_root = PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--artifact-root",
                )?);
            }
            "--compile-arch" => {
                compile_arch =
                    parse_required_flag_value(args, &mut index, "--compile-arch")?.to_string();
            }
            "--fixture" => {
                let value = parse_required_flag_value(args, &mut index, "--fixture")?;
                push_fixture_arg(value, &mut fixtures)?;
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-decompile-fixtures unknown argument {flag:?}; usage: {DECOMPILE_FIXTURES_USAGE}"
                )));
            }
            value => {
                push_fixture_arg(value, &mut fixtures)?;
                index += 1;
            }
        }
    }

    if fixtures.is_empty() {
        fixtures.push(SimpleKernelFixtureKind::I32Add);
    }

    let options = DecompileFixtureOptions {
        artifact_root,
        compile_arch,
        fixtures,
    };
    let reports = run_decompile_fixtures(&options)?;
    for report in reports {
        println!(
            "kernel_decompile_fixture fixture={} symbol={} parsed_instructions={} unsupported_instructions={} source_path={} ptx_path={} cubin_path={} sass_path={} ir_path={} side_by_side_path={}",
            report.fixture.name(),
            report.symbol,
            report.parsed_instruction_count,
            report.unsupported_instruction_count,
            report.source_path.display(),
            report.ptx_path.display(),
            report.cubin_path.display(),
            report.sass_path.display(),
            report.ir_path.display(),
            report.side_by_side_path.display(),
        );
    }
    Ok(())
}

fn push_fixture_arg(value: &str, fixtures: &mut Vec<SimpleKernelFixtureKind>) -> AppResult<()> {
    if value == "all" {
        for fixture in [
            SimpleKernelFixtureKind::I32Add,
            SimpleKernelFixtureKind::F32Add,
            SimpleKernelFixtureKind::F32Mul,
            SimpleKernelFixtureKind::F32Fma,
            SimpleKernelFixtureKind::LoadStore,
            SimpleKernelFixtureKind::PredicateBranch,
            SimpleKernelFixtureKind::ThreadIndexRead,
            SimpleKernelFixtureKind::Bf16ToF32,
            SimpleKernelFixtureKind::F16Ops,
        ] {
            push_unique_fixture(fixtures, fixture);
        }
        return Ok(());
    }
    let fixture = SimpleKernelFixtureKind::parse(value).ok_or_else(|| {
        invalid_input(format!(
            "unknown decompile fixture {value:?}; usage: {DECOMPILE_FIXTURES_USAGE}"
        ))
    })?;
    push_unique_fixture(fixtures, fixture);
    Ok(())
}

fn push_unique_fixture(
    fixtures: &mut Vec<SimpleKernelFixtureKind>,
    fixture: SimpleKernelFixtureKind,
) {
    if !fixtures.contains(&fixture) {
        fixtures.push(fixture);
    }
}
