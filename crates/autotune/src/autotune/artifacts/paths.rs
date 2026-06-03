use super::{
    super::{hashing::*, *},
    store::KernelArtifactStore,
    types::*,
};

impl KernelArtifactStore {
    pub fn managed() -> Self {
        Self::new(runtime::default_artifact_dir().join("generated"))
    }

    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn paths_for(&self, candidate: &KernelCandidateMetadata) -> KernelArtifactPaths {
        let directory = self
            .root
            .join(sanitize_path_component(&candidate.family))
            .join(candidate.artifact_key().hex());
        KernelArtifactPaths {
            directory: directory.clone(),
            manifest_path: directory.join("manifest.json"),
        }
    }

    pub fn standalone_crate_paths_for(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> StandaloneKernelCratePaths {
        let crate_dir = self.paths_for(candidate).directory.join("standalone-crate");
        standalone_crate_paths(crate_dir)
    }

    pub fn compile_scratch_root(&self) -> PathBuf {
        self.root.join("compile-scratch")
    }

    pub fn standalone_target_root(&self) -> PathBuf {
        self.root.join("standalone-target")
    }

    pub fn search_report_path_for(&self, report: &OptimizationSearchReport) -> PathBuf {
        self.root
            .join("search-reports")
            .join(sanitize_path_component(&report.family))
            .join(format!("{}.json", search_report_key(report).hex()))
    }

    pub fn auto_search_report_path_for(&self, report: &AutoOptimizationSearchReport) -> PathBuf {
        self.root
            .join("auto-search-reports")
            .join(sanitize_path_component(&report.family))
            .join(format!("{}.json", auto_search_report_key(report).hex()))
    }

    pub fn selection_path_for(&self, selection: &KernelOptimizationSelection) -> PathBuf {
        self.root
            .join("selections")
            .join(sanitize_path_component(&selection.family))
            .join(format!("{}.json", selection.artifact_key))
    }

    pub fn selection_cache_path_for(&self, cache_key: &KernelOptimizationCacheKey) -> PathBuf {
        self.root
            .join("selection-cache")
            .join(sanitize_path_component(&cache_key.family))
            .join(format!("{}.json", cache_key.key.hex()))
    }

    pub fn score_cache_path_for(
        &self,
        score_namespace: &str,
        candidate: &KernelCandidateMetadata,
    ) -> PathBuf {
        self.root
            .join("score-cache")
            .join(sanitize_path_component(&candidate.family))
            .join(sanitize_path_component(score_namespace))
            .join(format!("{}.json", candidate.artifact_key().hex()))
    }
}

pub(super) fn standalone_crate_paths(crate_dir: PathBuf) -> StandaloneKernelCratePaths {
    StandaloneKernelCratePaths {
        cargo_toml_path: crate_dir.join("Cargo.toml"),
        source_path: crate_dir.join("src").join("main.rs"),
        crate_dir,
    }
}
