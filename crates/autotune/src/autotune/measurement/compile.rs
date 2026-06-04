use std::{
    fs,
    path::{Path, PathBuf},
    process,
};

use nn_rust_profiling::{
    OptimizationTimingSegment, ProfileDuration, ProfileTimeSource, ProfileTimer,
};

use super::super::{
    EmittedStandaloneKernelCrate, KernelArtifactStore, KernelCandidateMetadata,
    KernelSourceGenerator,
};
use super::{KernelAutotuneMeasureResult, invalid_input};

pub(in crate::autotune::measurement) fn emit_and_compile_generated_kernel_scratch<G>(
    store: &KernelArtifactStore,
    candidate: &KernelCandidateMetadata,
    generator: &G,
    arch: Option<&str>,
) -> KernelAutotuneMeasureResult<(
    EmittedStandaloneKernelCrate,
    CompiledStandaloneKernelCrate,
    ScratchKernelBuild,
    Vec<OptimizationTimingSegment>,
)>
where
    G: KernelSourceGenerator,
{
    let scratch = ScratchKernelBuild::new(
        store
            .compile_scratch_root()
            .join(candidate.artifact_key().hex()),
    )?;

    let timer = ProfileTimer::start();
    let emitted = store.emit_standalone_crate_to_dir(candidate, generator, scratch.crate_dir())?;
    let emit_duration = timer.elapsed();

    let output_dir = scratch.output_dir();
    let timer = ProfileTimer::start();
    let compiled = compile_standalone_kernel_crate(
        &emitted.paths.crate_dir,
        &output_dir,
        &emitted.package_name,
        arch,
        Some(&store.standalone_target_root()),
    )?;
    let compile_duration = timer.elapsed();

    Ok((
        emitted,
        compiled,
        scratch,
        vec![
            OptimizationTimingSegment::new(
                "emit-standalone-crate",
                ProfileTimeSource::WallClock,
                emit_duration,
            ),
            OptimizationTimingSegment::new(
                "compile-standalone-crate",
                ProfileTimeSource::WallClock,
                compile_duration,
            ),
        ],
    ))
}

pub(in crate::autotune::measurement) struct ScratchKernelBuild {
    root: PathBuf,
    cleaned: bool,
}

impl ScratchKernelBuild {
    fn new(root: PathBuf) -> KernelAutotuneMeasureResult<Self> {
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            cleaned: false,
        })
    }

    fn crate_dir(&self) -> PathBuf {
        self.root.join("standalone-crate")
    }

    fn output_dir(&self) -> PathBuf {
        self.root.join("ptx")
    }

    pub(in crate::autotune::measurement) fn cleanup(
        mut self,
    ) -> KernelAutotuneMeasureResult<ProfileDuration> {
        let timer = ProfileTimer::start();
        if self.root.exists() {
            fs::remove_dir_all(&self.root)?;
        }
        self.cleaned = true;
        Ok(timer.elapsed())
    }
}

impl Drop for ScratchKernelBuild {
    fn drop(&mut self) {
        if !self.cleaned {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

pub fn compile_standalone_kernel_crate(
    crate_dir: &Path,
    output_dir: &Path,
    ptx_stem: &str,
    arch: Option<&str>,
    target_dir: Option<&Path>,
) -> KernelAutotuneMeasureResult<CompiledStandaloneKernelCrate> {
    fs::create_dir_all(output_dir)?;
    let crate_dir = crate_dir.canonicalize()?;
    let output_dir = output_dir.canonicalize()?;
    let mut command = process::Command::new("cargo");
    command
        .arg("oxide")
        .arg("build")
        .current_dir(&crate_dir)
        .env("CUDA_OXIDE_PTX_DIR", &output_dir);
    if let Some(target_dir) = target_dir {
        fs::create_dir_all(target_dir)?;
        command.env("CARGO_TARGET_DIR", target_dir.canonicalize()?);
    }
    if let Some(arch) = arch {
        command.arg("--arch").arg(arch);
    }
    let output = command.output().map_err(|error| {
        invalid_input(format!(
            "failed to run cargo oxide build in {}: {error}",
            crate_dir.display()
        ))
    })?;
    if !output.status.success() {
        return Err(invalid_input(format!(
            "cargo oxide build failed in {} with status {}\nstdout:\n{}\nstderr:\n{}",
            crate_dir.display(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let ptx_path = output_dir.join(format!("{ptx_stem}.ptx"));
    if !ptx_path.exists() {
        return Err(invalid_input(format!(
            "cargo oxide build succeeded in {} but expected generated PTX {} was not written",
            crate_dir.display(),
            ptx_path.display()
        )));
    }
    Ok(CompiledStandaloneKernelCrate {
        crate_dir,
        output_dir,
        ptx_path,
        stdout_bytes: output.stdout.len(),
        stderr_bytes: output.stderr.len(),
    })
}

pub struct CompiledStandaloneKernelCrate {
    pub crate_dir: PathBuf,
    pub output_dir: PathBuf,
    pub ptx_path: PathBuf,
    pub stdout_bytes: usize,
    pub stderr_bytes: usize,
}
