use super::{
    super::{codegen::*, hashing::*, metadata::*, *},
    paths::standalone_crate_paths,
    types::*,
    visual::write_auto_search_report_visuals,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArtifactStore {
    pub(super) root: PathBuf,
}

impl KernelArtifactStore {
    pub fn remove_compile_scratch(&self) -> Result<(), KernelGenerationError> {
        match fs::remove_dir_all(self.compile_scratch_root()) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
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
            visual_svg_path: None,
            visual_svg_bytes: None,
            visual_html_path: None,
            visual_html_bytes: None,
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
        let mut emitted = EmittedSearchReport {
            report_key: auto_search_report_key(report),
            report_path: path,
            report_bytes: report_json.len(),
            visual_svg_path: None,
            visual_svg_bytes: None,
            visual_html_path: None,
            visual_html_bytes: None,
        };
        let report_path = emitted.report_path.clone();
        write_auto_search_report_visuals(&report_path, report, &mut emitted)?;
        Ok(emitted)
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
