use std::{
    error::Error,
    fs,
    io::{self, ErrorKind},
};

use super::super::{
    PtxDecompileProbe, driver_support::absolute_path, driver_support::run_capture,
    driver_support::run_checked, lift_sass_module, parse_nvidia_sass, ptx_decompile_probes,
};
use super::types::{DecompilePtxProbeOptions, DecompilePtxProbeReport};

pub fn run_decompile_ptx_probes(
    options: &DecompilePtxProbeOptions,
) -> Result<Vec<DecompilePtxProbeReport>, Box<dyn Error>> {
    let mut reports = Vec::new();
    let artifact_root = absolute_path(&options.artifact_root)?;
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
        reports.push(run_decompile_ptx_probe(options, &artifact_root, probe)?);
    }
    Ok(reports)
}

fn run_decompile_ptx_probe(
    options: &DecompilePtxProbeOptions,
    artifact_root: &std::path::Path,
    probe: &PtxDecompileProbe,
) -> Result<DecompilePtxProbeReport, Box<dyn Error>> {
    let probe_dir = artifact_root.join(probe.kind.name());
    fs::create_dir_all(&probe_dir)?;
    let ptx_path = probe_dir.join(format!("{}.ptx", probe.symbol));
    fs::write(&ptx_path, probe.source.as_bytes())?;
    let compile_arch = probe.compile_arch_for(&options.compile_arch);

    let cubin_path = probe_dir.join(format!("{}.{}.cubin", probe.symbol, compile_arch));
    run_checked(
        "ptxas",
        &[
            format!("-arch={compile_arch}"),
            "-o".to_string(),
            cubin_path.display().to_string(),
            ptx_path.display().to_string(),
        ],
        &probe_dir,
    )?;

    let nvdisasm_sass_path =
        probe_dir.join(format!("{}.{}.nvdisasm.sass", probe.symbol, compile_arch));
    let nvdisasm_sass = run_capture("nvdisasm", &[cubin_path.display().to_string()], &probe_dir)?;
    fs::write(&nvdisasm_sass_path, nvdisasm_sass.as_bytes())?;

    let cuobjdump_sass_path =
        probe_dir.join(format!("{}.{}.cuobjdump.sass", probe.symbol, compile_arch));
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
        compile_arch: compile_arch.to_string(),
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
