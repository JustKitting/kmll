use super::*;

pub(in crate::autotune) fn standalone_package_name(candidate: &KernelCandidateMetadata) -> String {
    format!("nn_rust_kernel_{}", candidate.artifact_key().hex())
}

pub(in crate::autotune) fn standalone_cargo_toml(package_name: &str) -> String {
    let mut manifest = String::new();
    let cuda_oxide_root = standalone_cuda_oxide_checkout_root();
    writeln!(manifest, "[package]").expect("write to string");
    writeln!(manifest, "name = \"{package_name}\"").expect("write to string");
    writeln!(manifest, "version = \"0.1.0\"").expect("write to string");
    writeln!(manifest, "edition = \"2024\"").expect("write to string");
    writeln!(manifest).expect("write to string");
    writeln!(manifest, "[workspace]").expect("write to string");
    writeln!(manifest).expect("write to string");
    writeln!(manifest, "[dependencies]").expect("write to string");
    write_cuda_oxide_dependency(&mut manifest, "cuda-device", cuda_oxide_root.as_deref());
    write_cuda_oxide_dependency(&mut manifest, "cuda-host", cuda_oxide_root.as_deref());
    manifest
}

pub(in crate::autotune) fn write_cuda_oxide_dependency(
    manifest: &mut String,
    crate_name: &str,
    root: Option<&Path>,
) {
    if let Some(root) = root {
        let path = root.join("crates").join(crate_name);
        writeln!(
            manifest,
            "{crate_name} = {{ path = \"{}\" }}",
            toml_string(&path.to_string_lossy())
        )
        .expect("write to string");
    } else {
        writeln!(
            manifest,
            "{crate_name} = {{ git = \"https://github.com/NVlabs/cuda-oxide.git\", tag = \"v0.1.0\" }}"
        )
        .expect("write to string");
    }
}

pub(in crate::autotune) fn standalone_cuda_oxide_checkout_root() -> Option<PathBuf> {
    let configured = env::var_os("NN_RUST_CUDA_OXIDE_ROOT")
        .map(PathBuf::from)
        .filter(|path| cuda_oxide_checkout_has_kernel_crates(path));
    if configured.is_some() {
        return configured;
    }

    let cargo_home = env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))?;
    let checkouts = cargo_home.join("git").join("checkouts");
    let mut candidates = Vec::new();
    let entries = fs::read_dir(checkouts).ok()?;
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("cuda-oxide-") {
            continue;
        }
        let Ok(revisions) = fs::read_dir(entry.path()) else {
            continue;
        };
        for revision in revisions.flatten() {
            let path = revision.path();
            if cuda_oxide_checkout_has_kernel_crates(&path) {
                candidates.push(path);
            }
        }
    }
    candidates.sort();
    candidates.pop()
}

pub(in crate::autotune) fn cuda_oxide_checkout_has_kernel_crates(path: &Path) -> bool {
    path.join("crates")
        .join("cuda-device")
        .join("Cargo.toml")
        .is_file()
        && path
            .join("crates")
            .join("cuda-host")
            .join("Cargo.toml")
            .is_file()
}

pub(in crate::autotune) fn toml_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

pub(in crate::autotune) fn standalone_main_source(kernel_source: &str) -> String {
    let mut source = String::new();
    source.push_str(kernel_source);
    if !source.ends_with('\n') {
        source.push('\n');
    }
    writeln!(source).expect("write to string");
    writeln!(source, "fn main() {{}}").expect("write to string");
    source
}
