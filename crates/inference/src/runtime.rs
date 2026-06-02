use std::{
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use cuda_core::{CudaContext, CudaModule};
use cuda_host::LtoirError;

pub const ARTIFACT_DIR_ENV: &str = "NN_RUST_CUDA_ARTIFACT_DIR";
const DEFAULT_ARTIFACT_RELATIVE_DIR: &str = "target/cuda-oxide/inference";

pub fn default_artifact_name() -> &'static str {
    "nn_rust_inference"
}

pub fn default_artifact_dir() -> PathBuf {
    env::var_os(ARTIFACT_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join(DEFAULT_ARTIFACT_RELATIVE_DIR))
}

pub fn load_default_module(ctx: &Arc<CudaContext>) -> Result<Arc<CudaModule>, LtoirError> {
    load_kernel_artifact(ctx, default_artifact_name())
}

fn load_kernel_artifact(ctx: &Arc<CudaContext>, name: &str) -> Result<Arc<CudaModule>, LtoirError> {
    let dir = default_artifact_dir();
    let mut artifacts = Vec::new();

    for kind in [ArtifactKind::Cubin, ArtifactKind::Ptx, ArtifactKind::Ll] {
        let path = dir.join(format!("{name}.{}", kind.extension()));
        if let Some(modified) = modified_time(&path) {
            artifacts.push((modified, kind, path));
        }
    }

    let Some((_, kind, path)) = artifacts
        .into_iter()
        .max_by_key(|(modified, _, _)| *modified)
    else {
        return Err(LtoirError::NoArtifact {
            name: name.to_string(),
            dir,
        });
    };

    match kind {
        ArtifactKind::Ll => {
            let arch = cuda_host::ltoir::target_arch();
            let cubin = cuda_host::ltoir::build_cubin_from_ll(&path, &arch)?;
            load_module_file(ctx, &cubin)
        }
        ArtifactKind::Cubin | ArtifactKind::Ptx => load_module_file(ctx, &path),
    }
}

fn load_module_file(ctx: &Arc<CudaContext>, path: &Path) -> Result<Arc<CudaModule>, LtoirError> {
    Ok(ctx.load_module_from_file(
        path.to_str()
            .expect("kernel artifact path is not valid UTF-8"),
    )?)
}

fn modified_time(path: &Path) -> Option<SystemTime> {
    path.metadata().ok()?.modified().ok()
}

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or(manifest_dir)
}

#[derive(Debug, Clone, Copy)]
enum ArtifactKind {
    Cubin,
    Ptx,
    Ll,
}

impl ArtifactKind {
    fn extension(self) -> &'static str {
        match self {
            Self::Cubin => "cubin",
            Self::Ptx => "ptx",
            Self::Ll => "ll",
        }
    }
}
