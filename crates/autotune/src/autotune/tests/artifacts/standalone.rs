use super::*;

#[test]
fn artifact_store_writes_standalone_crate_for_generated_matvec_kernel() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_rows(RowMajorWarpRows::Rows8);

    let emitted = store
        .emit_standalone_crate(&candidate, &MatvecRustCudaGenerator)
        .expect("artifact store should write standalone generated matvec kernel crate");

    assert_eq!(emitted.artifact_key, candidate.artifact_key());
    assert_eq!(emitted.symbol, "matvec_bf16_rows8");
    assert!(emitted.paths.crate_dir.starts_with(store.root()));
    assert!(emitted.paths.cargo_toml_path.starts_with(store.root()));
    assert!(emitted.paths.source_path.starts_with(store.root()));

    let source = fs::read_to_string(&emitted.paths.source_path)
        .expect("standalone matvec main.rs should be readable");
    assert!(source.contains("pub fn matvec_bf16_rows8("));
    assert!(source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
    assert!(source.contains("fn main() {}"));

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_writes_standalone_crate_to_scratch_dir_without_persistent_source() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_rows(RowMajorWarpRows::Rows8);
    let persistent_paths = store.standalone_crate_paths_for(&candidate);
    let scratch_root = root
        .join("compile-scratch")
        .join(candidate.artifact_key().hex());
    let scratch_crate_dir = scratch_root.join("standalone-crate");

    let emitted = store
        .emit_standalone_crate_to_dir(&candidate, &MatvecRustCudaGenerator, &scratch_crate_dir)
        .expect("artifact store should write scratch standalone generated matvec crate");

    assert_eq!(emitted.artifact_key, candidate.artifact_key());
    assert_eq!(emitted.symbol, "matvec_bf16_rows8");
    assert_eq!(emitted.paths.crate_dir, scratch_crate_dir);
    assert!(emitted.paths.cargo_toml_path.starts_with(&scratch_root));
    assert!(emitted.paths.source_path.starts_with(&scratch_root));
    assert!(!persistent_paths.crate_dir.exists());
    assert!(!persistent_paths.cargo_toml_path.exists());
    assert!(!persistent_paths.source_path.exists());

    let source = fs::read_to_string(&emitted.paths.source_path)
        .expect("scratch standalone matvec main.rs should be readable");
    assert!(source.contains("pub fn matvec_bf16_rows8("));

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_removes_stale_compile_scratch_root() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let stale_source_path = store
        .compile_scratch_root()
        .join("stale-candidate")
        .join("standalone-crate")
        .join("src")
        .join("main.rs");
    fs::create_dir_all(
        stale_source_path
            .parent()
            .expect("stale scratch path should have a parent"),
    )
    .expect("stale scratch directory should be creatable");
    fs::write(&stale_source_path, "fn main() {}\n")
        .expect("stale scratch source should be writable");

    store
        .remove_compile_scratch()
        .expect("stale compile scratch root should be removable");

    assert!(!store.compile_scratch_root().exists());
    assert_eq!(
        store.standalone_target_root(),
        root.join("standalone-target")
    );
    assert!(root.exists());

    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_compile_scratch_cleanup_is_idempotent() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);

    store
        .remove_compile_scratch()
        .expect("missing compile scratch root should be accepted");

    fs::create_dir_all(&root).expect("test root should be creatable");
    remove_test_generated_root(&root);
}

#[test]
fn artifact_store_writes_standalone_crate_for_generated_kernel() {
    let root = test_generated_root();
    let store = KernelArtifactStore::new(&root);
    let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
    let candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));

    let emitted = store
        .emit_standalone_crate(&candidate, &GemmRustCudaGenerator)
        .expect("artifact store should write standalone generated kernel crate");

    assert_eq!(emitted.artifact_key, candidate.artifact_key());
    assert_eq!(emitted.package_name, "nn_rust_kernel_172b6682003af067");
    assert_eq!(emitted.symbol, "gemm_f32_bf16_tile_16x32x16");
    assert!(emitted.paths.crate_dir.starts_with(store.root()));
    assert!(emitted.paths.cargo_toml_path.starts_with(store.root()));
    assert!(emitted.paths.source_path.starts_with(store.root()));

    let cargo_toml = fs::read_to_string(&emitted.paths.cargo_toml_path)
        .expect("standalone Cargo.toml should be readable");
    assert!(cargo_toml.contains("name = \"nn_rust_kernel_172b6682003af067\""));
    assert!(cargo_toml.contains("cuda-device"));
    assert!(cargo_toml.contains("cuda-host"));

    let source = fs::read_to_string(&emitted.paths.source_path)
        .expect("standalone main.rs should be readable");
    assert!(source.contains("pub fn gemm_f32_bf16_tile_16x32x16("));
    assert!(source.contains("pub struct Bf16(u16);"));
    assert!(source.contains("fn main() {}"));

    remove_test_generated_root(&root);
}

#[test]
fn standalone_manifest_can_render_cuda_oxide_path_dependencies() {
    let mut manifest = String::new();

    write_cuda_oxide_dependency(
        &mut manifest,
        "cuda-device",
        Some(Path::new("/tmp/cuda oxide/root")),
    );
    write_cuda_oxide_dependency(&mut manifest, "cuda-host", None);

    assert!(
        manifest.contains("cuda-device = { path = \"/tmp/cuda oxide/root/crates/cuda-device\" }")
    );
    assert!(manifest.contains(
        "cuda-host = { git = \"https://github.com/NVlabs/cuda-oxide.git\", tag = \"v0.1.0\" }"
    ));
}

#[test]
fn cuda_oxide_checkout_detection_requires_kernel_crates() {
    let root = test_generated_root();
    let checkout = root.join("cuda-oxide");
    fs::create_dir_all(checkout.join("crates").join("cuda-device"))
        .expect("cuda-device directory should be creatable");
    fs::write(
        checkout
            .join("crates")
            .join("cuda-device")
            .join("Cargo.toml"),
        "[package]\nname = \"cuda-device\"\n",
    )
    .expect("cuda-device manifest should be writable");

    assert!(!cuda_oxide_checkout_has_kernel_crates(&checkout));

    fs::create_dir_all(checkout.join("crates").join("cuda-host"))
        .expect("cuda-host directory should be creatable");
    fs::write(
        checkout.join("crates").join("cuda-host").join("Cargo.toml"),
        "[package]\nname = \"cuda-host\"\n",
    )
    .expect("cuda-host manifest should be writable");

    assert!(cuda_oxide_checkout_has_kernel_crates(&checkout));

    remove_test_generated_root(&root);
}
