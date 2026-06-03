use super::{codegen::*, hashing::*, metadata::*, *};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArtifactStore {
    root: PathBuf,
}

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

    pub fn remove_compile_scratch(&self) -> Result<(), KernelGenerationError> {
        match fs::remove_dir_all(self.compile_scratch_root()) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
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

    pub fn emit_search_report(
        &self,
        report: &OptimizationSearchReport,
    ) -> Result<EmittedSearchReport, KernelGenerationError> {
        let path = self.search_report_path_for(report);
        fs::create_dir_all(
            path.parent()
                .expect("search report path should have a parent directory"),
        )?;
        let report_json = report.to_json_string();
        fs::write(&path, report_json.as_bytes())?;
        Ok(EmittedSearchReport {
            report_key: search_report_key(report),
            report_path: path,
            report_bytes: report_json.len(),
        })
    }

    pub fn emit_auto_search_report(
        &self,
        report: &AutoOptimizationSearchReport,
    ) -> Result<EmittedSearchReport, KernelGenerationError> {
        let path = self.auto_search_report_path_for(report);
        fs::create_dir_all(
            path.parent()
                .expect("auto search report path should have a parent directory"),
        )?;
        let report_json = report.to_json_string();
        fs::write(&path, report_json.as_bytes())?;
        Ok(EmittedSearchReport {
            report_key: auto_search_report_key(report),
            report_path: path,
            report_bytes: report_json.len(),
        })
    }

    pub fn emit_selection(
        &self,
        selection: &KernelOptimizationSelection,
    ) -> Result<EmittedKernelOptimizationSelection, KernelGenerationError> {
        let path = self.selection_path_for(selection);
        fs::create_dir_all(
            path.parent()
                .expect("selection path should have a parent directory"),
        )?;
        let selection_json = serde_json::to_vec_pretty(&selection_json(selection))?;
        fs::write(&path, &selection_json)?;
        Ok(EmittedKernelOptimizationSelection {
            artifact_key: selection.artifact_key.clone(),
            selection_path: path,
            selection_bytes: selection_json.len(),
        })
    }

    pub fn emit_selection_for_candidate(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<EmittedKernelOptimizationSelection, KernelGenerationError> {
        self.emit_selection(&KernelOptimizationSelection::from_candidate(candidate))
    }

    pub fn emit_selection_cache(
        &self,
        cache_key: &KernelOptimizationCacheKey,
        selection: &KernelOptimizationSelection,
    ) -> Result<EmittedKernelOptimizationSelection, KernelGenerationError> {
        let path = self.selection_cache_path_for(cache_key);
        fs::create_dir_all(
            path.parent()
                .expect("selection cache path should have a parent directory"),
        )?;
        let selection_json = serde_json::to_vec_pretty(&selection_json(selection))?;
        fs::write(&path, &selection_json)?;
        Ok(EmittedKernelOptimizationSelection {
            artifact_key: selection.artifact_key.clone(),
            selection_path: path,
            selection_bytes: selection_json.len(),
        })
    }

    pub fn emit_selection_cache_for_candidate(
        &self,
        cache_key: &KernelOptimizationCacheKey,
        candidate: &KernelCandidateMetadata,
    ) -> Result<EmittedKernelOptimizationSelection, KernelGenerationError> {
        self.emit_selection_cache(
            cache_key,
            &KernelOptimizationSelection::from_candidate(candidate),
        )
    }

    pub fn read_selection(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<KernelOptimizationSelection, KernelGenerationError> {
        let selection_text = fs::read_to_string(path)?;
        let selection_json: Value = serde_json::from_str(&selection_text)?;
        parse_selection_json(&selection_json)
    }

    pub fn read_selection_cache(
        &self,
        cache_key: &KernelOptimizationCacheKey,
    ) -> Result<Option<KernelOptimizationSelection>, KernelGenerationError> {
        let path = self.selection_cache_path_for(cache_key);
        let selection_text = match fs::read_to_string(path) {
            Ok(selection_text) => selection_text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let selection_json: Value = serde_json::from_str(&selection_text)?;
        parse_selection_json(&selection_json).map(Some)
    }

    pub fn emit_score_cache_for_candidate(
        &self,
        score_namespace: &str,
        candidate: &KernelCandidateMetadata,
    ) -> Result<Option<EmittedKernelOptimizationScore>, KernelGenerationError> {
        let Some(record) =
            KernelOptimizationScoreRecord::from_candidate(score_namespace, candidate)
        else {
            return Ok(None);
        };
        let path = self.score_cache_path_for(score_namespace, candidate);
        fs::create_dir_all(
            path.parent()
                .expect("score cache path should have a parent directory"),
        )?;
        let score_json = serde_json::to_vec_pretty(&score_record_json(&record))?;
        fs::write(&path, &score_json)?;
        Ok(Some(EmittedKernelOptimizationScore {
            artifact_key: record.artifact_key,
            score_path: path,
            score_bytes: score_json.len(),
        }))
    }

    pub fn read_score_cache_for_candidate(
        &self,
        score_namespace: &str,
        candidate: &KernelCandidateMetadata,
    ) -> Result<Option<SearchScore>, KernelGenerationError> {
        let path = self.score_cache_path_for(score_namespace, candidate);
        let score_text = match fs::read_to_string(path) {
            Ok(score_text) => score_text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let score_json: Value = serde_json::from_str(&score_text)?;
        let record = parse_score_record_json(&score_json)?;
        if record.matches_candidate(score_namespace, candidate) {
            Ok(Some(record.score))
        } else {
            Ok(None)
        }
    }

    pub fn emit_metadata(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<EmittedKernelMetadata, KernelGenerationError> {
        let paths = self.paths_for(candidate);
        fs::create_dir_all(&paths.directory)?;
        let manifest = generated_kernel_manifest(candidate);
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
        fs::write(&paths.manifest_path, &manifest_bytes)?;
        Ok(EmittedKernelMetadata {
            artifact_key: candidate.artifact_key(),
            paths,
            manifest_bytes: manifest_bytes.len(),
        })
    }

    pub fn emit_standalone_crate<G>(
        &self,
        candidate: &KernelCandidateMetadata,
        generator: &G,
    ) -> Result<EmittedStandaloneKernelCrate, KernelGenerationError>
    where
        G: KernelSourceGenerator,
    {
        let paths = self.standalone_crate_paths_for(candidate);
        self.emit_standalone_crate_to_paths(candidate, generator, paths)
    }

    pub fn emit_standalone_crate_to_dir<G>(
        &self,
        candidate: &KernelCandidateMetadata,
        generator: &G,
        crate_dir: impl Into<PathBuf>,
    ) -> Result<EmittedStandaloneKernelCrate, KernelGenerationError>
    where
        G: KernelSourceGenerator,
    {
        let paths = standalone_crate_paths(crate_dir.into());
        self.emit_standalone_crate_to_paths(candidate, generator, paths)
    }

    fn emit_standalone_crate_to_paths<G>(
        &self,
        candidate: &KernelCandidateMetadata,
        generator: &G,
        paths: StandaloneKernelCratePaths,
    ) -> Result<EmittedStandaloneKernelCrate, KernelGenerationError>
    where
        G: KernelSourceGenerator,
    {
        let generated = generator.source_for(candidate)?;
        let package_name = standalone_package_name(candidate);
        let cargo_toml = standalone_cargo_toml(&package_name);
        let source = standalone_main_source(&generated.source);
        fs::create_dir_all(
            paths
                .source_path
                .parent()
                .expect("standalone source path should have a parent directory"),
        )?;
        fs::write(&paths.cargo_toml_path, cargo_toml.as_bytes())?;
        fs::write(&paths.source_path, source.as_bytes())?;
        Ok(EmittedStandaloneKernelCrate {
            artifact_key: candidate.artifact_key(),
            package_name,
            symbol: generated.symbol,
            paths,
            cargo_toml_bytes: cargo_toml.len(),
            source_bytes: source.len(),
        })
    }
}

fn standalone_crate_paths(crate_dir: PathBuf) -> StandaloneKernelCratePaths {
    StandaloneKernelCratePaths {
        cargo_toml_path: crate_dir.join("Cargo.toml"),
        source_path: crate_dir.join("src").join("main.rs"),
        crate_dir,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArtifactPaths {
    pub directory: PathBuf,
    pub manifest_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandaloneKernelCratePaths {
    pub crate_dir: PathBuf,
    pub cargo_toml_path: PathBuf,
    pub source_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedKernelMetadata {
    pub artifact_key: KernelMetadataKey,
    pub paths: KernelArtifactPaths,
    pub manifest_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedSearchReport {
    pub report_key: KernelMetadataKey,
    pub report_path: PathBuf,
    pub report_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelOptimizationCacheKey {
    pub family: String,
    pub key: KernelMetadataKey,
}

impl KernelOptimizationCacheKey {
    pub fn hex(&self) -> String {
        self.key.hex()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelOptimizationSelection {
    pub family: String,
    pub artifact_key: String,
    pub generator: String,
    pub launchable: bool,
    pub action_trace: Vec<KernelScheduleAction>,
    pub score: Option<SearchScore>,
}

impl KernelOptimizationSelection {
    pub fn from_candidate(candidate: &KernelCandidateMetadata) -> Self {
        Self {
            family: candidate.family.clone(),
            artifact_key: candidate.artifact_key().hex(),
            generator: candidate.generated.generator.to_string(),
            launchable: candidate.is_launchable(),
            action_trace: candidate.action_trace.clone(),
            score: candidate.score,
        }
    }

    pub fn replay<P>(&self, problem: &P) -> Result<KernelCandidateMetadata, KernelActionReplayError>
    where
        P: KernelActionSearchProblem,
    {
        let candidate = replay_schedule_actions(problem, &self.action_trace)?;
        if candidate.family != self.family {
            return Err(KernelActionReplayError::FamilyMismatch {
                expected: self.family.clone(),
                actual: candidate.family,
            });
        }
        let actual_key = candidate.artifact_key().hex();
        if actual_key != self.artifact_key {
            return Err(KernelActionReplayError::ArtifactKeyMismatch {
                expected: self.artifact_key.clone(),
                actual: actual_key,
            });
        }
        if candidate.generated.generator != self.generator {
            return Err(KernelActionReplayError::GeneratorMismatch {
                expected: self.generator.clone(),
                actual: candidate.generated.generator.to_string(),
            });
        }
        let actual_launchable = candidate.is_launchable();
        if actual_launchable != self.launchable {
            return Err(KernelActionReplayError::LaunchabilityMismatch {
                expected: self.launchable,
                actual: actual_launchable,
            });
        }
        Ok(candidate)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelOptimizationScoreRecord {
    pub score_namespace: String,
    pub family: String,
    pub artifact_key: String,
    pub generator: String,
    pub launchable: bool,
    pub action_trace: Vec<KernelScheduleAction>,
    pub score: SearchScore,
}

impl KernelOptimizationScoreRecord {
    pub fn from_candidate(
        score_namespace: &str,
        candidate: &KernelCandidateMetadata,
    ) -> Option<Self> {
        Some(Self {
            score_namespace: score_namespace.to_string(),
            family: candidate.family.clone(),
            artifact_key: candidate.artifact_key().hex(),
            generator: candidate.generated.generator.to_string(),
            launchable: candidate.is_launchable(),
            action_trace: candidate.action_trace.clone(),
            score: candidate.score?,
        })
    }

    pub fn matches_candidate(
        &self,
        score_namespace: &str,
        candidate: &KernelCandidateMetadata,
    ) -> bool {
        self.score_namespace == score_namespace
            && self.family == candidate.family
            && self.artifact_key == candidate.artifact_key().hex()
            && self.generator == candidate.generated.generator
            && self.launchable == candidate.is_launchable()
            && self.action_trace == candidate.action_trace
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedKernelOptimizationSelection {
    pub artifact_key: String,
    pub selection_path: PathBuf,
    pub selection_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedKernelOptimizationScore {
    pub artifact_key: String,
    pub score_path: PathBuf,
    pub score_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedStandaloneKernelCrate {
    pub artifact_key: KernelMetadataKey,
    pub package_name: String,
    pub symbol: String,
    pub paths: StandaloneKernelCratePaths,
    pub cargo_toml_bytes: usize,
    pub source_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MatvecRustCudaGenerator;

impl KernelSourceGenerator for MatvecRustCudaGenerator {
    fn name(&self) -> &'static str {
        "matvec-rust-cuda-source-generator"
    }

    fn source_for(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<GeneratedKernelSource, KernelGenerationError> {
        if candidate.family != "matvec-bf16-row-major" {
            return Err(KernelGenerationError::UnsupportedCandidate {
                family: candidate.family.clone(),
                generator: self.name(),
            });
        }
        let plan = schedule_matvec_plan(&candidate.schedule).ok_or_else(|| {
            KernelGenerationError::MissingTransform {
                family: candidate.family.clone(),
                transform: "Split",
            }
        })?;
        let symbol = match &candidate.generated.materialization {
            KernelMaterialization::Existing { symbol } => (*symbol).to_string(),
            KernelMaterialization::DeferredGenerated { symbol_hint, .. } => {
                sanitize_identifier(symbol_hint)
            }
        };
        Ok(GeneratedKernelSource {
            symbol: symbol.clone(),
            source: render_bf16_matvec_source(&symbol, plan),
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GemmRustCudaGenerator;

impl KernelSourceGenerator for GemmRustCudaGenerator {
    fn name(&self) -> &'static str {
        "gemm-rust-cuda-source-generator"
    }

    fn source_for(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<GeneratedKernelSource, KernelGenerationError> {
        if candidate.family != "gemm-f32-bf16-row-col-row" {
            return Err(KernelGenerationError::UnsupportedCandidate {
                family: candidate.family.clone(),
                generator: self.name(),
            });
        }
        let plan = schedule_gemm_plan(&candidate.schedule).ok_or_else(|| {
            KernelGenerationError::MissingTransform {
                family: candidate.family.clone(),
                transform: "TileGemm",
            }
        })?;
        let symbol = match &candidate.generated.materialization {
            KernelMaterialization::Existing { symbol } => (*symbol).to_string(),
            KernelMaterialization::DeferredGenerated { symbol_hint, .. } => {
                sanitize_identifier(symbol_hint)
            }
        };
        Ok(GeneratedKernelSource {
            symbol: symbol.clone(),
            source: render_f32_bf16_gemm_source(&symbol, plan),
        })
    }
}
