use std::{
    cmp::Ordering,
    collections::HashSet,
    fmt::{self, Write as _},
    fs, io,
    path::{Path, PathBuf},
};

pub use nn_rust_profiling::{
    AutoOptimizationExitReason as ProfilingAutoOptimizationExitReason,
    AutoOptimizationSearchConfig, AutoOptimizationSearchReport, AutoOptimizationSearchStep,
    OptimizationActionArg as KernelScheduleActionArg,
    OptimizationActionMaterialization as KernelActionMaterialization,
    OptimizationActionOp as KernelScheduleActionOp, OptimizationActionSpec as KernelScheduleAction,
    OptimizationCandidateSpec, OptimizationScore as SearchScore,
    OptimizationScoreSource as SearchScoreSource, OptimizationSearchConfig,
    OptimizationSearchReport, OptimizationTiming,
};
use nn_rust_profiling::{
    CudaLaunchSpec, NumericKind, OperationKind, OperationRoute, ProfileDuration, ProfileTimeSource,
    SampleStats, TensorTypeSpec, TypedOperationSpec,
};
use serde_json::{Value, json};

use crate::{
    layout::{
        ColumnMajor, GemmKernelPlan, MatvecKernelPlan, RowMajor, RowMajorWarpRowMatvecPlan,
        RowMajorWarpRows2MatvecPlan, RowMajorWarpRows4MatvecPlan, RowMajorWarpRows8MatvecPlan,
        TiledGemm16Plan,
    },
    runtime,
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn bounded_unroll_factors(
    extent: usize,
    max_factor: u32,
    excluded_factor: Option<u32>,
) -> Vec<u32> {
    let upper = extent.min(max_factor as usize) as u32;
    (1..=upper)
        .filter(|factor| Some(*factor) != excluded_factor)
        .collect()
}

fn bounded_tile_factors(extent: usize, max_factor: u32, required_factor: Option<u32>) -> Vec<u32> {
    const FACTORS: [u32; 5] = [8, 13, 16, 24, 32];

    let upper = extent.min(max_factor as usize) as u32;
    let mut factors = FACTORS
        .into_iter()
        .filter(|factor| *factor <= upper)
        .collect::<Vec<_>>();
    if upper > 0 {
        factors.push(upper);
    }
    if let Some(required) = required_factor {
        factors.push(required);
    }
    factors.sort_unstable();
    factors.dedup();
    factors
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelAxisKind {
    Spatial,
    Reduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelAxis {
    pub id: u8,
    pub name: &'static str,
    pub extent: usize,
    pub kind: KernelAxisKind,
    pub stride: Option<usize>,
}

impl KernelAxis {
    pub const fn spatial(id: u8, name: &'static str, extent: usize, stride: Option<usize>) -> Self {
        Self {
            id,
            name,
            extent,
            kind: KernelAxisKind::Spatial,
            stride,
        }
    }

    pub const fn reduction(
        id: u8,
        name: &'static str,
        extent: usize,
        stride: Option<usize>,
    ) -> Self {
        Self {
            id,
            name,
            extent,
            kind: KernelAxisKind::Reduction,
            stride,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScheduleTransform {
    Split { axis: u8, factor: u32 },
    Unroll { axis: u8, factor: u32 },
    LocalTile { axis: u8, factor: u32 },
    ThreadGroup { axis: u8, factor: u32 },
    TileGemm { m: u32, n: u32, k: u32 },
    StrideOrder { axes: Vec<u8> },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct KernelSchedule {
    pub transforms: Vec<ScheduleTransform>,
}

impl KernelSchedule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_transform(mut self, transform: ScheduleTransform) -> Self {
        self.transforms.push(transform);
        self
    }

    pub fn depth(&self) -> usize {
        self.transforms.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelMetadataKey(u64);

impl KernelMetadataKey {
    pub const fn raw(self) -> u64 {
        self.0
    }

    pub fn hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelMaterialization {
    Existing { symbol: &'static str },
    DeferredGenerated { symbol_hint: String, reason: String },
}

impl KernelMaterialization {
    pub const fn is_launchable(&self) -> bool {
        matches!(self, Self::Existing { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedKernelMetadata {
    pub generator: &'static str,
    pub artifact_key: KernelMetadataKey,
    pub materialization: KernelMaterialization,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelCandidateMetadata {
    pub family: String,
    pub axes: Vec<KernelAxis>,
    pub schedule: KernelSchedule,
    pub action_trace: Vec<KernelScheduleAction>,
    pub generated: GeneratedKernelMetadata,
    pub launch: CudaLaunchSpec,
    pub operation: TypedOperationSpec,
    pub score: Option<SearchScore>,
}

impl KernelCandidateMetadata {
    pub fn is_launchable(&self) -> bool {
        self.generated.materialization.is_launchable()
    }

    pub fn artifact_key(&self) -> KernelMetadataKey {
        self.generated.artifact_key
    }

    pub fn optimization_spec(&self) -> OptimizationCandidateSpec {
        OptimizationCandidateSpec::new(
            self.family.clone(),
            self.artifact_key().hex(),
            self.generated.generator,
            self.launch.clone(),
            self.operation.clone(),
        )
        .with_launchable(self.is_launchable())
        .with_action_trace(self.action_trace.clone())
        .with_score(self.score)
    }

    pub fn new(
        family: impl Into<String>,
        axes: Vec<KernelAxis>,
        schedule: KernelSchedule,
        generator: &'static str,
        materialization: KernelMaterialization,
        launch: CudaLaunchSpec,
        operation: TypedOperationSpec,
    ) -> Self {
        let family = family.into();
        let artifact_key = metadata_key(&family, &axes, &schedule, &launch);
        Self {
            family,
            axes,
            schedule,
            action_trace: Vec::new(),
            generated: GeneratedKernelMetadata {
                generator,
                artifact_key,
                materialization,
            },
            launch,
            operation,
            score: None,
        }
    }
}

fn candidate_with_action_trace(
    parent: &KernelCandidateMetadata,
    action: &KernelScheduleAction,
    mut candidate: KernelCandidateMetadata,
) -> KernelCandidateMetadata {
    candidate.action_trace = parent.action_trace.clone();
    candidate.action_trace.push(action.clone());
    candidate
}

#[derive(Debug)]
pub enum KernelGenerationError {
    UnsupportedCandidate {
        family: String,
        generator: &'static str,
    },
    MissingTransform {
        family: String,
        transform: &'static str,
    },
    InvalidSelection {
        reason: String,
    },
    Io(io::Error),
    Json(serde_json::Error),
}

impl fmt::Display for KernelGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCandidate { family, generator } => {
                write!(
                    f,
                    "candidate family {family:?} is not supported by generator {generator}"
                )
            }
            Self::MissingTransform { family, transform } => {
                write!(
                    f,
                    "candidate family {family:?} is missing required {transform} transform"
                )
            }
            Self::InvalidSelection { reason } => {
                write!(
                    f,
                    "kernel optimization selection metadata is invalid: {reason}"
                )
            }
            Self::Io(error) => write!(f, "kernel artifact I/O failed: {error}"),
            Self::Json(error) => write!(f, "kernel artifact manifest JSON failed: {error}"),
        }
    }
}

impl std::error::Error for KernelGenerationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for KernelGenerationError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for KernelGenerationError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedKernelSource {
    pub symbol: String,
    pub source: String,
}

pub trait KernelSourceGenerator {
    fn name(&self) -> &'static str;
    fn source_for(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<GeneratedKernelSource, KernelGenerationError>;
}

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
        StandaloneKernelCratePaths {
            cargo_toml_path: crate_dir.join("Cargo.toml"),
            source_path: crate_dir.join("src").join("main.rs"),
            crate_dir,
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
        let generated = generator.source_for(candidate)?;
        let paths = self.standalone_crate_paths_for(candidate);
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedKernelOptimizationSelection {
    pub artifact_key: String,
    pub selection_path: PathBuf,
    pub selection_bytes: usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeamSearchConfig {
    pub beam_width: usize,
    pub max_depth: usize,
    pub require_launchable: bool,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_depth: 2,
            require_launchable: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoOptimizeConfig {
    pub beam_width: usize,
    pub max_steps: usize,
    pub require_launchable: bool,
    pub min_score_improvement: f64,
}

impl AutoOptimizeConfig {
    pub const fn from_beam_search_config(config: BeamSearchConfig) -> Self {
        Self {
            beam_width: config.beam_width,
            max_steps: config.max_depth,
            require_launchable: config.require_launchable,
            min_score_improvement: 0.0,
        }
    }

    pub const fn as_beam_search_config(self) -> BeamSearchConfig {
        BeamSearchConfig {
            beam_width: self.beam_width,
            max_depth: self.max_steps,
            require_launchable: self.require_launchable,
        }
    }
}

impl Default for AutoOptimizeConfig {
    fn default() -> Self {
        Self::from_beam_search_config(BeamSearchConfig::default())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeamSearchResult {
    pub best: Option<KernelCandidateMetadata>,
    pub beam: Vec<KernelCandidateMetadata>,
    pub explored: usize,
    pub rejected: usize,
}

impl BeamSearchResult {
    pub fn optimization_report(
        &self,
        family: impl Into<String>,
        config: BeamSearchConfig,
    ) -> OptimizationSearchReport {
        OptimizationSearchReport::new(
            family,
            OptimizationSearchConfig::new(
                config.beam_width,
                config.max_depth,
                config.require_launchable,
            ),
            self.explored,
            self.rejected,
            self.best
                .as_ref()
                .map(KernelCandidateMetadata::optimization_spec),
            self.beam
                .iter()
                .map(KernelCandidateMetadata::optimization_spec)
                .collect(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutoOptimizeExitReason {
    CacheHit,
    MaxSteps,
    NoCandidates,
    NoImprovement { best_delta: f64 },
}

impl AutoOptimizeExitReason {
    pub const fn label(self) -> &'static str {
        match self {
            Self::CacheHit => "cache-hit",
            Self::MaxSteps => "max-steps",
            Self::NoCandidates => "no-candidates",
            Self::NoImprovement { .. } => "no-improvement",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AutoOptimizeStep {
    pub depth: usize,
    pub input_beam_len: usize,
    pub generated: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub best_before: Option<SearchScore>,
    pub best_after: Option<SearchScore>,
    pub improvement: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AutoOptimizeResult {
    pub best: Option<KernelCandidateMetadata>,
    pub beam: Vec<KernelCandidateMetadata>,
    pub explored: usize,
    pub rejected: usize,
    pub steps: Vec<AutoOptimizeStep>,
    pub exit_reason: AutoOptimizeExitReason,
}

impl AutoOptimizeResult {
    pub fn as_beam_search_result(&self) -> BeamSearchResult {
        BeamSearchResult {
            best: self.best.clone(),
            beam: self.beam.clone(),
            explored: self.explored,
            rejected: self.rejected,
        }
    }

    pub fn optimization_report(
        &self,
        family: impl Into<String>,
        config: AutoOptimizeConfig,
    ) -> OptimizationSearchReport {
        self.as_beam_search_result()
            .optimization_report(family, config.as_beam_search_config())
    }

    pub fn auto_optimization_report(
        &self,
        family: impl Into<String>,
        config: AutoOptimizeConfig,
    ) -> AutoOptimizationSearchReport {
        AutoOptimizationSearchReport::new(
            family,
            AutoOptimizationSearchConfig::new(
                config.beam_width,
                config.max_steps,
                config.require_launchable,
                config.min_score_improvement,
            ),
            self.explored,
            self.rejected,
            profiling_auto_exit_reason(self.exit_reason),
            self.steps.iter().map(profiling_auto_search_step).collect(),
            self.best
                .as_ref()
                .map(KernelCandidateMetadata::optimization_spec),
            self.beam
                .iter()
                .map(KernelCandidateMetadata::optimization_spec)
                .collect(),
        )
    }
}

fn profiling_auto_exit_reason(
    reason: AutoOptimizeExitReason,
) -> ProfilingAutoOptimizationExitReason {
    match reason {
        AutoOptimizeExitReason::CacheHit => ProfilingAutoOptimizationExitReason::CacheHit,
        AutoOptimizeExitReason::MaxSteps => ProfilingAutoOptimizationExitReason::MaxSteps,
        AutoOptimizeExitReason::NoCandidates => ProfilingAutoOptimizationExitReason::NoCandidates,
        AutoOptimizeExitReason::NoImprovement { best_delta } => {
            ProfilingAutoOptimizationExitReason::NoImprovement { best_delta }
        }
    }
}

fn profiling_auto_search_step(step: &AutoOptimizeStep) -> AutoOptimizationSearchStep {
    AutoOptimizationSearchStep {
        depth: step.depth,
        input_beam_len: step.input_beam_len,
        generated: step.generated,
        accepted: step.accepted,
        rejected: step.rejected,
        best_before: step.best_before,
        best_after: step.best_after,
        improvement: step.improvement,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionCacheStatus {
    Hit,
    Miss,
    Stale { reason: String },
}

impl SelectionCacheStatus {
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::Stale { .. } => "stale",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedBeamSearchResult {
    pub result: BeamSearchResult,
    pub cache_key: KernelOptimizationCacheKey,
    pub cache_status: SelectionCacheStatus,
    pub cache_write: Option<EmittedKernelOptimizationSelection>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedAutoOptimizeResult {
    pub result: AutoOptimizeResult,
    pub cache_key: KernelOptimizationCacheKey,
    pub cache_status: SelectionCacheStatus,
    pub cache_write: Option<EmittedKernelOptimizationSelection>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KernelActionSpaceSet {
    pub spaces: Vec<KernelActionSpace>,
}

impl KernelActionSpaceSet {
    pub fn new(spaces: impl Into<Vec<KernelActionSpace>>) -> Self {
        Self {
            spaces: spaces.into(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.spaces.is_empty()
    }

    pub fn actions(&self) -> Vec<KernelScheduleAction> {
        self.spaces
            .iter()
            .flat_map(KernelActionSpace::actions)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelActionSpace {
    Split {
        variants: Vec<KernelAxisFactorAction>,
    },
    Unroll {
        axis: u8,
        factors: Vec<u32>,
    },
    TileGemm {
        variants: Vec<KernelTile3dAction>,
    },
    StrideOrder {
        orders: Vec<Vec<u8>>,
    },
}

impl KernelActionSpace {
    pub fn actions(&self) -> Vec<KernelScheduleAction> {
        match self {
            Self::Split { variants } => variants
                .iter()
                .map(|variant| {
                    KernelScheduleAction::split(
                        variant.axis,
                        variant.factor,
                        variant.materialization,
                    )
                })
                .collect(),
            Self::Unroll { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::unroll(*axis, factor))
                .collect(),
            Self::TileGemm { variants } => variants
                .iter()
                .map(|variant| {
                    KernelScheduleAction::tile_gemm(
                        variant.tile.m,
                        variant.tile.n,
                        variant.tile.k,
                        variant.materialization,
                    )
                })
                .collect(),
            Self::StrideOrder { orders } => orders
                .iter()
                .cloned()
                .map(KernelScheduleAction::stride_order)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelAxisFactorAction {
    pub axis: u8,
    pub factor: u32,
    pub materialization: KernelActionMaterialization,
}

impl KernelAxisFactorAction {
    pub const fn new(axis: u8, factor: u32, materialization: KernelActionMaterialization) -> Self {
        Self {
            axis,
            factor,
            materialization,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelTile3d {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl KernelTile3d {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelTile3dAction {
    pub tile: KernelTile3d,
    pub materialization: KernelActionMaterialization,
}

impl KernelTile3dAction {
    pub const fn new(tile: KernelTile3d, materialization: KernelActionMaterialization) -> Self {
        Self {
            tile,
            materialization,
        }
    }
}

pub trait KernelMetadataSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata;
    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata>;
    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore>;
}

pub trait KernelActionSearchProblem: KernelMetadataSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet;

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet;

    fn schedule_actions(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelScheduleAction> {
        self.action_spaces(candidate).actions()
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelActionReplayError {
    InvalidAction {
        index: usize,
        action: KernelScheduleAction,
    },
    FamilyMismatch {
        expected: String,
        actual: String,
    },
    GeneratorMismatch {
        expected: String,
        actual: String,
    },
    ArtifactKeyMismatch {
        expected: String,
        actual: String,
    },
    LaunchabilityMismatch {
        expected: bool,
        actual: bool,
    },
}

impl fmt::Display for KernelActionReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAction { index, action } => {
                write!(f, "invalid schedule action at index {index}: {action:?}")
            }
            Self::FamilyMismatch { expected, actual } => {
                write!(
                    f,
                    "replayed candidate family {actual:?} did not match expected {expected:?}"
                )
            }
            Self::GeneratorMismatch { expected, actual } => {
                write!(
                    f,
                    "replayed candidate generator {actual:?} did not match expected {expected:?}"
                )
            }
            Self::ArtifactKeyMismatch { expected, actual } => {
                write!(
                    f,
                    "replayed candidate artifact key {actual} did not match expected {expected}"
                )
            }
            Self::LaunchabilityMismatch { expected, actual } => {
                write!(
                    f,
                    "replayed candidate launchable={actual} did not match expected launchable={expected}"
                )
            }
        }
    }
}

impl std::error::Error for KernelActionReplayError {}

pub fn expand_with_schedule_actions<P>(
    problem: &P,
    candidate: &KernelCandidateMetadata,
) -> Vec<KernelCandidateMetadata>
where
    P: KernelActionSearchProblem,
{
    problem
        .schedule_actions(candidate)
        .into_iter()
        .filter_map(|action| problem.apply_schedule_action(candidate, &action))
        .collect()
}

pub fn replay_schedule_actions<P>(
    problem: &P,
    actions: &[KernelScheduleAction],
) -> Result<KernelCandidateMetadata, KernelActionReplayError>
where
    P: KernelActionSearchProblem,
{
    let mut candidate = problem.seed();
    for (index, action) in actions.iter().enumerate() {
        candidate = problem
            .apply_schedule_action(&candidate, action)
            .ok_or_else(|| KernelActionReplayError::InvalidAction {
                index,
                action: action.clone(),
            })?;
    }
    Ok(candidate)
}

pub fn replay_optimization_candidate_spec<P>(
    problem: &P,
    spec: &OptimizationCandidateSpec,
) -> Result<KernelCandidateMetadata, KernelActionReplayError>
where
    P: KernelActionSearchProblem,
{
    let candidate = replay_schedule_actions(problem, &spec.action_trace)?;
    if candidate.family != spec.family {
        return Err(KernelActionReplayError::FamilyMismatch {
            expected: spec.family.clone(),
            actual: candidate.family,
        });
    }
    let actual_key = candidate.artifact_key().hex();
    if actual_key != spec.artifact_key {
        return Err(KernelActionReplayError::ArtifactKeyMismatch {
            expected: spec.artifact_key.clone(),
            actual: actual_key,
        });
    }
    Ok(candidate)
}

pub fn beam_search_metadata<P>(problem: &P, config: BeamSearchConfig) -> BeamSearchResult
where
    P: KernelMetadataSearchProblem,
{
    beam_search_metadata_with_scorer(problem, config, |candidate| problem.score(candidate))
}

pub fn beam_search_metadata_with_scorer<P, F>(
    problem: &P,
    config: BeamSearchConfig,
    mut score_candidate: F,
) -> BeamSearchResult
where
    P: KernelMetadataSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    assert!(config.beam_width > 0, "beam width must be nonzero");

    let mut seed = problem.seed();
    seed.score = score_candidate(&seed);
    let mut seen = HashSet::new();
    seen.insert(seed.artifact_key());
    let mut beam = vec![seed];
    let mut explored = 0;
    let mut rejected = 0;

    for _ in 0..config.max_depth {
        let mut candidates = Vec::new();
        for candidate in &beam {
            for mut next in problem.expand(candidate) {
                if !seen.insert(next.artifact_key()) {
                    continue;
                }
                explored += 1;
                if config.require_launchable && !next.is_launchable() {
                    rejected += 1;
                    continue;
                }
                match score_candidate(&next) {
                    Some(score) => {
                        next.score = Some(score);
                        candidates.push(next);
                    }
                    None => rejected += 1,
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        candidates.sort_by(compare_candidates);
        beam = candidates.into_iter().take(config.beam_width).collect();
    }

    let best = beam.first().cloned();
    BeamSearchResult {
        best,
        beam,
        explored,
        rejected,
    }
}

pub fn auto_optimize_metadata<P>(problem: &P, config: AutoOptimizeConfig) -> AutoOptimizeResult
where
    P: KernelMetadataSearchProblem,
{
    auto_optimize_metadata_with_scorer(problem, config, |candidate| problem.score(candidate))
}

pub fn auto_optimize_metadata_with_scorer<P, F>(
    problem: &P,
    config: AutoOptimizeConfig,
    mut score_candidate: F,
) -> AutoOptimizeResult
where
    P: KernelMetadataSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    assert!(config.beam_width > 0, "beam width must be nonzero");
    assert!(
        config.min_score_improvement.is_finite() && config.min_score_improvement >= 0.0,
        "minimum score improvement must be finite and nonnegative"
    );

    let mut seed = problem.seed();
    seed.score = score_candidate(&seed);
    let mut seen = HashSet::new();
    seen.insert(seed.artifact_key());
    let mut beam = vec![seed];
    let mut explored = 0;
    let mut rejected = 0;
    let mut steps = Vec::new();
    let mut exit_reason = AutoOptimizeExitReason::MaxSteps;

    for depth in 0..config.max_steps {
        let input_beam_len = beam.len();
        let best_before = beam.first().and_then(|candidate| candidate.score);
        let mut candidates = Vec::new();
        let mut generated = 0;
        let mut step_rejected = 0;

        for candidate in &beam {
            for mut next in problem.expand(candidate) {
                if !seen.insert(next.artifact_key()) {
                    continue;
                }
                explored += 1;
                generated += 1;
                if config.require_launchable && !next.is_launchable() {
                    rejected += 1;
                    step_rejected += 1;
                    continue;
                }
                match score_candidate(&next) {
                    Some(score) => {
                        next.score = Some(score);
                        candidates.push(next);
                    }
                    None => {
                        rejected += 1;
                        step_rejected += 1;
                    }
                }
            }
        }

        if candidates.is_empty() {
            steps.push(AutoOptimizeStep {
                depth,
                input_beam_len,
                generated,
                accepted: 0,
                rejected: step_rejected,
                best_before,
                best_after: None,
                improvement: None,
            });
            exit_reason = AutoOptimizeExitReason::NoCandidates;
            break;
        }

        candidates.sort_by(compare_candidates);
        let accepted = candidates.len().min(config.beam_width);
        let next_beam = candidates
            .into_iter()
            .take(config.beam_width)
            .collect::<Vec<_>>();
        let best_after = next_beam.first().and_then(|candidate| candidate.score);
        let improvement =
            best_before.and_then(|before| best_after.map(|after| before.value - after.value));
        let stop_for_no_improvement = improvement
            .map(|delta| delta <= config.min_score_improvement)
            .unwrap_or(false);

        steps.push(AutoOptimizeStep {
            depth,
            input_beam_len,
            generated,
            accepted,
            rejected: step_rejected,
            best_before,
            best_after,
            improvement,
        });

        if stop_for_no_improvement {
            if improvement.is_some_and(|delta| delta > 0.0)
                && let Some(best_next) = next_beam.first()
            {
                beam = vec![best_next.clone()];
            }
            exit_reason = AutoOptimizeExitReason::NoImprovement {
                best_delta: improvement.unwrap_or(0.0),
            };
            break;
        }

        beam = next_beam;
    }

    let best = beam.first().cloned();
    AutoOptimizeResult {
        best,
        beam,
        explored,
        rejected,
        steps,
        exit_reason,
    }
}

pub fn beam_search_metadata_with_selection_cache<P, F>(
    store: &KernelArtifactStore,
    problem: &P,
    config: BeamSearchConfig,
    score_namespace: &str,
    mut score_candidate: F,
) -> Result<CachedBeamSearchResult, KernelGenerationError>
where
    P: KernelActionSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    let cache_key = optimization_selection_cache_key(problem, config, score_namespace);
    let cache_status = match store.read_selection_cache(&cache_key)? {
        Some(selection) => match selection.replay(problem) {
            Ok(mut candidate) => {
                candidate.score = selection.score;
                return Ok(CachedBeamSearchResult {
                    result: BeamSearchResult {
                        best: Some(candidate.clone()),
                        beam: vec![candidate],
                        explored: 0,
                        rejected: 0,
                    },
                    cache_key,
                    cache_status: SelectionCacheStatus::Hit,
                    cache_write: None,
                });
            }
            Err(error) => SelectionCacheStatus::Stale {
                reason: error.to_string(),
            },
        },
        None => SelectionCacheStatus::Miss,
    };
    let result =
        beam_search_metadata_with_scorer(problem, config, |candidate| score_candidate(candidate));
    let cache_write = result
        .best
        .as_ref()
        .map(|candidate| store.emit_selection_cache_for_candidate(&cache_key, candidate))
        .transpose()?;
    Ok(CachedBeamSearchResult {
        result,
        cache_key,
        cache_status,
        cache_write,
    })
}

pub fn auto_optimize_metadata_with_selection_cache<P, F>(
    store: &KernelArtifactStore,
    problem: &P,
    config: AutoOptimizeConfig,
    score_namespace: &str,
    mut score_candidate: F,
) -> Result<CachedAutoOptimizeResult, KernelGenerationError>
where
    P: KernelActionSearchProblem,
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    let cache_key = auto_optimization_selection_cache_key(problem, config, score_namespace);
    let cache_status = match store.read_selection_cache(&cache_key)? {
        Some(selection) => match selection.replay(problem) {
            Ok(mut candidate) => {
                candidate.score = selection.score;
                return Ok(CachedAutoOptimizeResult {
                    result: AutoOptimizeResult {
                        best: Some(candidate.clone()),
                        beam: vec![candidate],
                        explored: 0,
                        rejected: 0,
                        steps: Vec::new(),
                        exit_reason: AutoOptimizeExitReason::CacheHit,
                    },
                    cache_key,
                    cache_status: SelectionCacheStatus::Hit,
                    cache_write: None,
                });
            }
            Err(error) => SelectionCacheStatus::Stale {
                reason: error.to_string(),
            },
        },
        None => SelectionCacheStatus::Miss,
    };
    let result =
        auto_optimize_metadata_with_scorer(problem, config, |candidate| score_candidate(candidate));
    let cache_write = result
        .best
        .as_ref()
        .map(|candidate| store.emit_selection_cache_for_candidate(&cache_key, candidate))
        .transpose()?;
    Ok(CachedAutoOptimizeResult {
        result,
        cache_key,
        cache_status,
        cache_write,
    })
}

pub fn optimization_selection_cache_key<P>(
    problem: &P,
    config: BeamSearchConfig,
    score_namespace: &str,
) -> KernelOptimizationCacheKey
where
    P: KernelActionSearchProblem,
{
    let seed = problem.seed();
    let mut state = FNV_OFFSET;
    state = hash_str(state, "optimization-selection-cache");
    state = hash_str(state, score_namespace);
    state = hash_u64(state, config.beam_width as u64);
    state = hash_u64(state, config.max_depth as u64);
    state = hash_u64(state, u64::from(config.require_launchable));
    state = hash_optimization_candidate(state, &seed.optimization_spec());
    state = hash_action_space_set(state, &problem.search_space());
    KernelOptimizationCacheKey {
        family: seed.family,
        key: KernelMetadataKey(state),
    }
}

pub fn auto_optimization_selection_cache_key<P>(
    problem: &P,
    config: AutoOptimizeConfig,
    score_namespace: &str,
) -> KernelOptimizationCacheKey
where
    P: KernelActionSearchProblem,
{
    let seed = problem.seed();
    let mut state = FNV_OFFSET;
    state = hash_str(state, "auto-optimization-selection-cache");
    state = hash_str(state, score_namespace);
    state = hash_u64(state, config.beam_width as u64);
    state = hash_u64(state, config.max_steps as u64);
    state = hash_u64(state, u64::from(config.require_launchable));
    state = hash_u64(state, config.min_score_improvement.to_bits());
    state = hash_optimization_candidate(state, &seed.optimization_spec());
    state = hash_action_space_set(state, &problem.search_space());
    KernelOptimizationCacheKey {
        family: seed.family,
        key: KernelMetadataKey(state),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowMajorWarpRows {
    Rows1,
    Rows2,
    Rows4,
    Rows8,
}

impl RowMajorWarpRows {
    pub const ALL: [Self; 4] = [Self::Rows1, Self::Rows2, Self::Rows4, Self::Rows8];

    pub fn from_rows_per_block(rows_per_block: u32) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|plan| plan.rows_per_block() == rows_per_block)
    }

    pub fn rows_per_block(self) -> u32 {
        match self {
            Self::Rows1 => <RowMajorWarpRowMatvecPlan as MatvecKernelPlan>::rows_per_block(),
            Self::Rows2 => <RowMajorWarpRows2MatvecPlan as MatvecKernelPlan>::rows_per_block(),
            Self::Rows4 => <RowMajorWarpRows4MatvecPlan as MatvecKernelPlan>::rows_per_block(),
            Self::Rows8 => <RowMajorWarpRows8MatvecPlan as MatvecKernelPlan>::rows_per_block(),
        }
    }

    pub fn block_threads(self) -> u32 {
        match self {
            Self::Rows1 => <RowMajorWarpRowMatvecPlan as MatvecKernelPlan>::block_threads(),
            Self::Rows2 => <RowMajorWarpRows2MatvecPlan as MatvecKernelPlan>::block_threads(),
            Self::Rows4 => <RowMajorWarpRows4MatvecPlan as MatvecKernelPlan>::block_threads(),
            Self::Rows8 => <RowMajorWarpRows8MatvecPlan as MatvecKernelPlan>::block_threads(),
        }
    }

    pub fn grid_rows(self, rows: usize) -> u32 {
        match self {
            Self::Rows1 => <RowMajorWarpRowMatvecPlan as MatvecKernelPlan>::grid_rows(rows),
            Self::Rows2 => <RowMajorWarpRows2MatvecPlan as MatvecKernelPlan>::grid_rows(rows),
            Self::Rows4 => <RowMajorWarpRows4MatvecPlan as MatvecKernelPlan>::grid_rows(rows),
            Self::Rows8 => <RowMajorWarpRows8MatvecPlan as MatvecKernelPlan>::grid_rows(rows),
        }
    }

    pub fn plan_name(self) -> &'static str {
        match self {
            Self::Rows1 => RowMajorWarpRowMatvecPlan::NAME,
            Self::Rows2 => RowMajorWarpRows2MatvecPlan::NAME,
            Self::Rows4 => RowMajorWarpRows4MatvecPlan::NAME,
            Self::Rows8 => RowMajorWarpRows8MatvecPlan::NAME,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatvecRowSplit {
    rows_per_block: u32,
}

impl MatvecRowSplit {
    pub const LANES_PER_ROW: u32 = 32;
    pub const MAX_ROWS_PER_BLOCK: u32 = 32;

    pub fn new(rows_per_block: u32) -> Option<Self> {
        (rows_per_block > 0 && rows_per_block <= Self::MAX_ROWS_PER_BLOCK)
            .then_some(Self { rows_per_block })
    }

    pub const fn rows_per_block(self) -> u32 {
        self.rows_per_block
    }

    pub const fn block_threads(self) -> u32 {
        self.rows_per_block * Self::LANES_PER_ROW
    }

    pub fn grid_rows(self, rows: usize) -> u32 {
        (rows as u32).div_ceil(self.rows_per_block)
    }

    pub fn existing_plan(self) -> Option<RowMajorWarpRows> {
        RowMajorWarpRows::from_rows_per_block(self.rows_per_block)
    }

    pub fn plan_name(self) -> String {
        self.existing_plan()
            .map(RowMajorWarpRows::plan_name)
            .map(str::to_string)
            .unwrap_or_else(|| format!("row-major-warp-rows{}-matvec", self.rows_per_block))
    }
}

impl From<RowMajorWarpRows> for MatvecRowSplit {
    fn from(value: RowMajorWarpRows) -> Self {
        Self {
            rows_per_block: value.rows_per_block(),
        }
    }
}

impl PartialEq<RowMajorWarpRows> for MatvecRowSplit {
    fn eq(&self, other: &RowMajorWarpRows) -> bool {
        self.rows_per_block == other.rows_per_block()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatvecSchedulePlan {
    pub rows: MatvecRowSplit,
    pub reduce_unroll: u32,
}

impl MatvecSchedulePlan {
    pub const DEFAULT_REDUCE_UNROLL: u32 = 4;

    pub fn new(rows: impl Into<MatvecRowSplit>) -> Self {
        Self {
            rows: rows.into(),
            reduce_unroll: Self::DEFAULT_REDUCE_UNROLL,
        }
    }

    pub const fn with_reduce_unroll(mut self, factor: u32) -> Self {
        self.reduce_unroll = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn normalized(self) -> Self {
        self.with_reduce_unroll(self.reduce_unroll)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatvecSearchProblem {
    pub rows: usize,
    pub cols: usize,
    pub input_dtype: NumericKind,
    pub weight_dtype: NumericKind,
    pub accumulator: NumericKind,
}

impl MatvecSearchProblem {
    const MAX_ROWS_PER_BLOCK: u32 = MatvecRowSplit::MAX_ROWS_PER_BLOCK;
    const MAX_REDUCE_UNROLL_FACTOR: u32 = 32;

    pub const fn bf16_row_major(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            input_dtype: NumericKind::F32,
            weight_dtype: NumericKind::Bf16,
            accumulator: NumericKind::F32,
        }
    }

    pub fn candidate_for_rows(&self, plan: RowMajorWarpRows) -> KernelCandidateMetadata {
        self.candidate_for_plan_with_materialization(
            MatvecSchedulePlan::new(plan),
            "matvec_bf16_kernel".to_string(),
            KernelMaterialization::Existing {
                symbol: "matvec_bf16_kernel",
            },
        )
    }

    pub fn generated_candidate_for_rows(&self, plan: RowMajorWarpRows) -> KernelCandidateMetadata {
        self.generated_candidate_for_plan(MatvecSchedulePlan::new(plan))
    }

    pub fn generated_candidate_for_row_split(
        &self,
        rows: MatvecRowSplit,
    ) -> KernelCandidateMetadata {
        self.generated_candidate_for_plan(MatvecSchedulePlan::new(rows))
    }

    pub fn generated_candidate_for_plan(
        &self,
        plan: MatvecSchedulePlan,
    ) -> KernelCandidateMetadata {
        let plan = plan.normalized();
        let symbol_hint = matvec_symbol_hint(plan);
        self.candidate_for_plan_with_materialization(
            plan,
            symbol_hint.clone(),
            KernelMaterialization::DeferredGenerated {
                symbol_hint,
                reason: "row split descriptor has no emitted Rust CUDA kernel yet".to_string(),
            },
        )
    }

    fn candidate_for_plan_with_materialization(
        &self,
        plan: MatvecSchedulePlan,
        launch_kernel: String,
        materialization: KernelMaterialization,
    ) -> KernelCandidateMetadata {
        let plan = plan.normalized();
        let rows = plan.rows;
        let rows_per_block = rows.rows_per_block();
        let mut schedule = KernelSchedule::new()
            .with_transform(ScheduleTransform::Split {
                axis: 0,
                factor: rows_per_block,
            })
            .with_transform(ScheduleTransform::ThreadGroup {
                axis: 0,
                factor: rows.block_threads(),
            });
        if plan.reduce_unroll != MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 1,
                factor: plan.reduce_unroll,
            });
        }
        let launch = CudaLaunchSpec::new(
            launch_kernel,
            (rows.grid_rows(self.rows), 1, 1),
            (rows.block_threads(), 1, 1),
            0,
        );
        let operation = TypedOperationSpec::new(
            matvec_operation_name(plan),
            OperationKind::Matvec,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(self.input_dtype, self.accumulator, [self.cols])
                .with_layout("contiguous"),
        )
        .with_input(
            TensorTypeSpec::new(self.weight_dtype, self.accumulator, [self.rows, self.cols])
                .with_layout("row-major"),
        )
        .with_output(
            TensorTypeSpec::new(self.accumulator, self.accumulator, [self.rows])
                .with_layout("contiguous"),
        )
        .with_launch(launch.clone());

        KernelCandidateMetadata::new(
            "matvec-bf16-row-major",
            self.axes(),
            schedule,
            "row-major-matvec-generator",
            materialization,
            launch,
            operation,
        )
    }

    fn axes(&self) -> Vec<KernelAxis> {
        vec![
            KernelAxis::spatial(0, "row", self.rows, Some(self.cols)),
            KernelAxis::reduction(1, "col", self.cols, Some(1)),
        ]
    }

    fn reduce_unroll_factors(&self) -> Vec<u32> {
        bounded_unroll_factors(
            self.cols,
            Self::MAX_REDUCE_UNROLL_FACTOR,
            Some(MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL),
        )
    }

    fn deferred_row_split_factors(&self) -> Vec<u32> {
        bounded_unroll_factors(self.rows, Self::MAX_ROWS_PER_BLOCK, None)
    }

    fn split_variants(&self) -> Vec<KernelAxisFactorAction> {
        let mut variants = Vec::new();
        variants.extend(RowMajorWarpRows::ALL.into_iter().map(|plan| {
            KernelAxisFactorAction::new(
                0,
                plan.rows_per_block(),
                KernelActionMaterialization::Existing,
            )
        }));
        variants.extend(self.deferred_row_split_factors().into_iter().map(|factor| {
            KernelAxisFactorAction::new(0, factor, KernelActionMaterialization::DeferredGenerated)
        }));
        variants
    }
}

impl KernelActionSearchProblem for MatvecSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet {
        let split_variants = self.split_variants();
        let unroll_factors = self.reduce_unroll_factors();
        KernelActionSpaceSet::new(vec![
            KernelActionSpace::Split {
                variants: split_variants,
            },
            KernelActionSpace::Unroll {
                axis: 1,
                factors: unroll_factors,
            },
        ])
    }

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
        if candidate.family != "matvec-bf16-row-major" {
            return KernelActionSpaceSet::default();
        }
        if candidate.schedule.depth() == 0 {
            return KernelActionSpaceSet::new(vec![KernelActionSpace::Split {
                variants: self.split_variants(),
            }]);
        }
        let Some(plan) = schedule_matvec_plan(&candidate.schedule) else {
            return KernelActionSpaceSet::default();
        };
        if candidate.is_launchable()
            || plan.reduce_unroll != MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL
        {
            return KernelActionSpaceSet::default();
        }
        let factors = self.reduce_unroll_factors();
        KernelActionSpaceSet::new(vec![KernelActionSpace::Unroll { axis: 1, factors }])
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata> {
        if candidate.family != "matvec-bf16-row-major" {
            return None;
        }
        match action {
            KernelScheduleAction {
                op: KernelScheduleActionOp::Split,
                axis: Some(0),
                arg: KernelScheduleActionArg::Factor(rows_per_block),
                materialization,
            } => {
                if candidate.schedule.depth() > 0 {
                    return None;
                }
                let next = match materialization {
                    KernelActionMaterialization::Existing => {
                        let rows = RowMajorWarpRows::from_rows_per_block(*rows_per_block)?;
                        self.candidate_for_rows(rows)
                    }
                    KernelActionMaterialization::DeferredGenerated => {
                        if !self.deferred_row_split_factors().contains(rows_per_block) {
                            return None;
                        }
                        let rows = MatvecRowSplit::new(*rows_per_block)?;
                        self.generated_candidate_for_row_split(rows)
                    }
                };
                Some(candidate_with_action_trace(candidate, action, next))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(1),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.is_launchable() || !self.reduce_unroll_factors().contains(factor) {
                    return None;
                }
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if plan.reduce_unroll != MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
                    return None;
                }
                let next = self.generated_candidate_for_plan(plan.with_reduce_unroll(*factor));
                Some(candidate_with_action_trace(candidate, action, next))
            }
            _ => None,
        }
    }
}

impl KernelMetadataSearchProblem for MatvecSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new("matvec_bf16_kernel", (1, 1, 1), (32, 1, 1), 0);
        let operation = TypedOperationSpec::new(
            "matvec-bf16-row-major-seed",
            OperationKind::Matvec,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(self.input_dtype, self.accumulator, [self.cols])
                .with_layout("contiguous"),
        )
        .with_input(
            TensorTypeSpec::new(self.weight_dtype, self.accumulator, [self.rows, self.cols])
                .with_layout("row-major"),
        )
        .with_output(
            TensorTypeSpec::new(self.accumulator, self.accumulator, [self.rows])
                .with_layout("contiguous"),
        )
        .with_launch(launch.clone());
        KernelCandidateMetadata::new(
            "matvec-bf16-row-major",
            self.axes(),
            KernelSchedule::new(),
            "row-major-matvec-generator",
            KernelMaterialization::DeferredGenerated {
                symbol_hint: "matvec_bf16_kernel".to_string(),
                reason: "seed descriptor has no concrete schedule yet".to_string(),
            },
            launch,
            operation,
        )
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        expand_with_schedule_actions(self, candidate)
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let plan = schedule_matvec_plan(&candidate.schedule)?;
        let rows_per_block = plan.rows.rows_per_block() as usize;
        let blocks = self.rows.div_ceil(rows_per_block);
        let padded_rows = blocks * rows_per_block;
        let useful_fma_ops = self.rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let wasted_rows = padded_rows.saturating_sub(self.rows);
        let wasted_fma_ops = wasted_rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let block_overhead = blocks as f64 * 2048.0;
        let unroll = f64::from(plan.reduce_unroll.max(1));
        let loop_overhead = blocks as f64 * (self.cols as f64 / 32.0).ceil() * 64.0 / unroll;
        let register_pressure = blocks as f64 * (unroll - 1.0).max(0.0) * 32.0;
        let generic_runtime_penalty = if candidate.is_launchable() {
            blocks as f64 * 64.0
        } else {
            0.0
        };
        SearchScore::heuristic(
            useful_fma_ops
                + wasted_fma_ops * 8.0
                + block_overhead
                + loop_overhead
                + register_pressure
                + generic_runtime_penalty,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmTileShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl GemmTileShape {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    pub const fn thread_count(self) -> u32 {
        self.m * self.n
    }

    pub const fn is_launchable_shape(self) -> bool {
        self.m > 0 && self.n > 0 && self.k > 0 && self.thread_count() <= 1024
    }

    pub fn block_dim(self) -> (u32, u32, u32) {
        (self.n, self.m, 1)
    }

    pub fn grid_dim(self, m: usize, n: usize) -> (u32, u32, u32) {
        ((n as u32).div_ceil(self.n), (m as u32).div_ceil(self.m), 1)
    }
}

impl From<GemmTileShape> for KernelTile3d {
    fn from(value: GemmTileShape) -> Self {
        Self::new(value.m, value.n, value.k)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmSchedulePlan {
    pub tile: GemmTileShape,
    pub reduce_unroll: u32,
    pub b_load_order: GemmBTileLoadOrder,
}

impl GemmSchedulePlan {
    pub const fn new(tile: GemmTileShape) -> Self {
        Self {
            tile,
            reduce_unroll: 1,
            b_load_order: GemmBTileLoadOrder::TileLinear,
        }
    }

    pub const fn with_reduce_unroll(mut self, factor: u32) -> Self {
        self.reduce_unroll = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_b_load_order(mut self, order: GemmBTileLoadOrder) -> Self {
        self.b_load_order = order;
        self
    }

    pub const fn normalized(self) -> Self {
        self.with_reduce_unroll(self.reduce_unroll)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmBTileLoadOrder {
    TileLinear,
    KContiguous,
}

impl GemmBTileLoadOrder {
    pub const fn symbol_suffix(self) -> &'static str {
        match self {
            Self::TileLinear => "",
            Self::KContiguous => "_bk",
        }
    }

    pub fn action_axes(self) -> Vec<u8> {
        match self {
            Self::TileLinear => vec![1, 2],
            Self::KContiguous => vec![2, 1],
        }
    }

    pub fn from_action_axes(axes: &[u8]) -> Option<Self> {
        match axes {
            [1, 2] => Some(Self::TileLinear),
            [2, 1] => Some(Self::KContiguous),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmSearchProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_dtype: NumericKind,
    pub b_dtype: NumericKind,
    pub c_dtype: NumericKind,
    pub accumulator: NumericKind,
}

impl GemmSearchProblem {
    const EXISTING_TILE: GemmTileShape = GemmTileShape::new(16, 16, 16);
    const MAX_TILE_DIM: u32 = 32;
    const MAX_REDUCE_UNROLL_FACTOR: u32 = 32;
    const B_LOAD_ORDERS: [GemmBTileLoadOrder; 1] = [GemmBTileLoadOrder::KContiguous];

    pub const fn f32_bf16_row_col_row(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            a_dtype: NumericKind::F32,
            b_dtype: NumericKind::Bf16,
            c_dtype: NumericKind::F32,
            accumulator: NumericKind::F32,
        }
    }

    pub fn candidate_for_tile(&self, tile: GemmTileShape) -> KernelCandidateMetadata {
        self.candidate_for_plan(GemmSchedulePlan::new(tile))
    }

    pub fn candidate_for_plan(&self, plan: GemmSchedulePlan) -> KernelCandidateMetadata {
        let plan = plan.normalized();
        let tile = plan.tile;
        let existing_tile = Self::is_existing_plan(plan);
        let b_order_suffix = plan.b_load_order.symbol_suffix();
        let symbol_hint = if plan.reduce_unroll == 1 {
            format!(
                "gemm_f32_bf16_tile_{}x{}x{}{}",
                tile.m, tile.n, tile.k, b_order_suffix
            )
        } else {
            format!(
                "gemm_f32_bf16_tile_{}x{}x{}_u{}{}",
                tile.m, tile.n, tile.k, plan.reduce_unroll, b_order_suffix
            )
        };
        let launch_kernel = if existing_tile {
            "gemm_f32_bf16_tiled_kernel".to_string()
        } else {
            symbol_hint.clone()
        };
        let launch = CudaLaunchSpec::new(
            launch_kernel,
            tile.grid_dim(self.m, self.n),
            tile.block_dim(),
            0,
        );
        let operation = TypedOperationSpec::new(
            if plan.reduce_unroll == 1 {
                format!(
                    "gemm-f32-bf16-{}x{}x{}{}",
                    tile.m, tile.n, tile.k, b_order_suffix
                )
            } else {
                format!(
                    "gemm-f32-bf16-{}x{}x{}-u{}{}",
                    tile.m, tile.n, tile.k, plan.reduce_unroll, b_order_suffix
                )
            },
            OperationKind::Gemm,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(self.a_dtype, self.accumulator, [self.m, self.k])
                .with_layout("row-major"),
        )
        .with_input(
            TensorTypeSpec::new(self.b_dtype, self.accumulator, [self.k, self.n])
                .with_layout("column-major"),
        )
        .with_output(
            TensorTypeSpec::new(self.c_dtype, self.accumulator, [self.m, self.n])
                .with_layout("row-major"),
        )
        .with_launch(launch.clone());
        let materialization = if existing_tile {
            KernelMaterialization::Existing {
                symbol: "gemm_f32_bf16_tiled_kernel",
            }
        } else {
            KernelMaterialization::DeferredGenerated {
                symbol_hint,
                reason: "schedule descriptor has no emitted Rust CUDA kernel yet".to_string(),
            }
        };
        let mut schedule = KernelSchedule::new().with_transform(ScheduleTransform::TileGemm {
            m: tile.m,
            n: tile.n,
            k: tile.k,
        });
        if plan.reduce_unroll > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 2,
                factor: plan.reduce_unroll,
            });
        }
        if plan.b_load_order == GemmBTileLoadOrder::KContiguous {
            schedule = schedule.with_transform(ScheduleTransform::StrideOrder { axes: vec![2, 1] });
        }

        KernelCandidateMetadata::new(
            "gemm-f32-bf16-row-col-row",
            self.axes(),
            schedule,
            "tiled-gemm-generator",
            materialization,
            launch,
            operation,
        )
    }

    fn axes(&self) -> Vec<KernelAxis> {
        vec![
            KernelAxis::spatial(0, "m", self.m, Some(self.k)),
            KernelAxis::spatial(1, "n", self.n, Some(1)),
            KernelAxis::reduction(2, "k", self.k, Some(1)),
        ]
    }

    fn is_existing_plan(plan: GemmSchedulePlan) -> bool {
        let plan = plan.normalized();
        plan.tile == Self::EXISTING_TILE
            && plan.reduce_unroll == 1
            && plan.b_load_order == GemmBTileLoadOrder::TileLinear
    }

    fn action_materialization_for_plan(plan: GemmSchedulePlan) -> KernelActionMaterialization {
        if Self::is_existing_plan(plan) {
            KernelActionMaterialization::Existing
        } else {
            KernelActionMaterialization::DeferredGenerated
        }
    }

    fn reduce_unroll_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            factors.extend(Self::reduce_unroll_factors_for_tile(tile));
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn reduce_unroll_factors_for_tile(tile: GemmTileShape) -> Vec<u32> {
        bounded_unroll_factors(tile.k as usize, Self::MAX_REDUCE_UNROLL_FACTOR, Some(1))
    }

    fn tile_shapes(&self) -> Vec<GemmTileShape> {
        let m_factors =
            bounded_tile_factors(self.m, Self::MAX_TILE_DIM, Some(Self::EXISTING_TILE.m));
        let n_factors =
            bounded_tile_factors(self.n, Self::MAX_TILE_DIM, Some(Self::EXISTING_TILE.n));
        let k_factors =
            bounded_tile_factors(self.k, Self::MAX_TILE_DIM, Some(Self::EXISTING_TILE.k));
        let mut tiles = Vec::new();
        for m in m_factors {
            for n in n_factors.iter().copied() {
                for k in k_factors.iter().copied() {
                    let tile = GemmTileShape::new(m, n, k);
                    if tile.is_launchable_shape() {
                        tiles.push(tile);
                    }
                }
            }
        }
        tiles.sort_unstable_by_key(|tile| (tile.m, tile.n, tile.k));
        tiles.dedup();
        tiles
    }

    fn tile_action_variants(&self) -> Vec<KernelTile3dAction> {
        self.tile_shapes()
            .into_iter()
            .map(|tile| {
                KernelTile3dAction::new(
                    tile.into(),
                    Self::action_materialization_for_plan(GemmSchedulePlan::new(tile)),
                )
            })
            .collect()
    }
}

impl KernelActionSearchProblem for GemmSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet {
        let tile_variants = self.tile_action_variants();
        let unroll_factors = self.reduce_unroll_factors();
        let stride_orders = Self::B_LOAD_ORDERS
            .into_iter()
            .map(GemmBTileLoadOrder::action_axes)
            .collect();
        KernelActionSpaceSet::new(vec![
            KernelActionSpace::TileGemm {
                variants: tile_variants,
            },
            KernelActionSpace::Unroll {
                axis: 2,
                factors: unroll_factors,
            },
            KernelActionSpace::StrideOrder {
                orders: stride_orders,
            },
        ])
    }

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
        let Some(plan) = schedule_gemm_plan(&candidate.schedule) else {
            return KernelActionSpaceSet::new(vec![KernelActionSpace::TileGemm {
                variants: self.tile_action_variants(),
            }]);
        };

        if plan.reduce_unroll == 1 {
            let mut spaces = Vec::new();
            let factors = Self::reduce_unroll_factors_for_tile(plan.tile);
            spaces.push(KernelActionSpace::Unroll { axis: 2, factors });
            if plan.b_load_order == GemmBTileLoadOrder::TileLinear {
                let orders = Self::B_LOAD_ORDERS
                    .into_iter()
                    .map(GemmBTileLoadOrder::action_axes)
                    .collect();
                spaces.push(KernelActionSpace::StrideOrder { orders });
            }
            return KernelActionSpaceSet::new(spaces);
        }

        if plan.b_load_order == GemmBTileLoadOrder::TileLinear {
            let orders = Self::B_LOAD_ORDERS
                .into_iter()
                .map(GemmBTileLoadOrder::action_axes)
                .collect();
            return KernelActionSpaceSet::new(vec![KernelActionSpace::StrideOrder { orders }]);
        }

        KernelActionSpaceSet::default()
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata> {
        match action {
            KernelScheduleAction {
                op: KernelScheduleActionOp::TileGemm,
                axis: None,
                arg: KernelScheduleActionArg::Tile3d { m, n, k },
                materialization,
            } => {
                if schedule_gemm_plan(&candidate.schedule).is_some() {
                    return None;
                }
                let tile = GemmTileShape::new(*m, *n, *k);
                if !self.tile_shapes().contains(&tile) {
                    return None;
                }
                let plan = GemmSchedulePlan::new(tile);
                if *materialization != Self::action_materialization_for_plan(plan) {
                    return None;
                }
                Some(candidate_with_action_trace(
                    candidate,
                    action,
                    self.candidate_for_plan(plan),
                ))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(2),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.reduce_unroll != 1
                    || !Self::reduce_unroll_factors_for_tile(plan.tile).contains(factor)
                {
                    return None;
                }
                Some(candidate_with_action_trace(
                    candidate,
                    action,
                    self.candidate_for_plan(plan.with_reduce_unroll(*factor)),
                ))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::StrideOrder,
                axis: None,
                arg: KernelScheduleActionArg::AxisOrder(axes),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.b_load_order != GemmBTileLoadOrder::TileLinear {
                    return None;
                }
                let order = GemmBTileLoadOrder::from_action_axes(axes)?;
                if order == GemmBTileLoadOrder::TileLinear {
                    return None;
                }
                Some(candidate_with_action_trace(
                    candidate,
                    action,
                    self.candidate_for_plan(plan.with_b_load_order(order)),
                ))
            }
            _ => None,
        }
    }
}

impl KernelMetadataSearchProblem for GemmSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        let launch = CudaLaunchSpec::new(
            "gemm_f32_bf16_tiled_kernel",
            TiledGemm16Plan::<RowMajor, ColumnMajor, RowMajor>::grid_dim(self.m, self.n),
            TiledGemm16Plan::<RowMajor, ColumnMajor, RowMajor>::block_dim(),
            0,
        );
        let operation = TypedOperationSpec::new(
            "gemm-f32-bf16-seed",
            OperationKind::Gemm,
            OperationRoute::CudaKernel,
        )
        .with_input(
            TensorTypeSpec::new(self.a_dtype, self.accumulator, [self.m, self.k])
                .with_layout("row-major"),
        )
        .with_input(
            TensorTypeSpec::new(self.b_dtype, self.accumulator, [self.k, self.n])
                .with_layout("column-major"),
        )
        .with_output(
            TensorTypeSpec::new(self.c_dtype, self.accumulator, [self.m, self.n])
                .with_layout("row-major"),
        )
        .with_launch(launch.clone());
        KernelCandidateMetadata::new(
            "gemm-f32-bf16-row-col-row",
            self.axes(),
            KernelSchedule::new(),
            "tiled-gemm-generator",
            KernelMaterialization::DeferredGenerated {
                symbol_hint: "gemm_f32_bf16_tiled_kernel".to_string(),
                reason: "seed descriptor has no concrete tile schedule yet".to_string(),
            },
            launch,
            operation,
        )
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        expand_with_schedule_actions(self, candidate)
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let plan = schedule_gemm_plan(&candidate.schedule)?;
        let tile = plan.tile;
        let tile_m = tile.m as usize;
        let tile_n = tile.n as usize;
        let tile_k = tile.k as usize;
        let padded_m = self.m.div_ceil(tile_m).checked_mul(tile_m)?;
        let padded_n = self.n.div_ceil(tile_n).checked_mul(tile_n)?;
        let padded_k = self.k.div_ceil(tile_k).checked_mul(tile_k)?;
        let padded_fma_ops = padded_m
            .checked_mul(padded_n)?
            .checked_mul(padded_k)?
            .checked_mul(2)? as f64;
        let block_count = self
            .m
            .div_ceil(tile_m)
            .checked_mul(self.n.div_ceil(tile_n))? as f64;
        let unroll = f64::from(plan.reduce_unroll.max(1));
        let loop_overhead = block_count * 4096.0 / unroll;
        let register_pressure = block_count * (unroll - 1.0) * 256.0;
        let b_load_penalty = match plan.b_load_order {
            GemmBTileLoadOrder::TileLinear => block_count * 512.0,
            GemmBTileLoadOrder::KContiguous => block_count * 64.0,
        };
        SearchScore::heuristic(padded_fma_ops + loop_overhead + register_pressure + b_load_penalty)
    }
}

fn compare_candidates(a: &KernelCandidateMetadata, b: &KernelCandidateMetadata) -> Ordering {
    let a_score = a.score.map(|score| score.value).unwrap_or(f64::INFINITY);
    let b_score = b.score.map(|score| score.value).unwrap_or(f64::INFINITY);
    a_score
        .partial_cmp(&b_score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.artifact_key().cmp(&b.artifact_key()))
}

fn schedule_rows_per_block(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Split { axis: 0, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_matvec_reduce_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 1, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_matvec_plan(schedule: &KernelSchedule) -> Option<MatvecSchedulePlan> {
    let rows_per_block = schedule_rows_per_block(schedule)?;
    let rows = MatvecRowSplit::new(rows_per_block)?;
    Some(MatvecSchedulePlan {
        rows,
        reduce_unroll: schedule_matvec_reduce_unroll(schedule)
            .unwrap_or(MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL),
    })
}

fn matvec_symbol_hint(plan: MatvecSchedulePlan) -> String {
    let plan = plan.normalized();
    let base = format!("matvec_bf16_rows{}", plan.rows.rows_per_block());
    if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
        base
    } else {
        format!("{base}_u{}", plan.reduce_unroll)
    }
}

fn matvec_operation_name(plan: MatvecSchedulePlan) -> String {
    let plan = plan.normalized();
    let plan_name = plan.rows.plan_name();
    if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
        format!("{plan_name}::bf16")
    } else {
        format!("{plan_name}::bf16-u{}", plan.reduce_unroll)
    }
}

fn schedule_gemm_tile(schedule: &KernelSchedule) -> Option<GemmTileShape> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::TileGemm { m, n, k } => Some(GemmTileShape::new(*m, *n, *k)),
            _ => None,
        })
}

fn schedule_gemm_reduce_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 2, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_gemm_b_load_order(schedule: &KernelSchedule) -> GemmBTileLoadOrder {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::StrideOrder { axes } if axes.as_slice() == [2, 1] => {
                Some(GemmBTileLoadOrder::KContiguous)
            }
            _ => None,
        })
        .unwrap_or(GemmBTileLoadOrder::TileLinear)
}

fn schedule_gemm_plan(schedule: &KernelSchedule) -> Option<GemmSchedulePlan> {
    let tile = schedule_gemm_tile(schedule)?;
    Some(GemmSchedulePlan {
        tile,
        reduce_unroll: schedule_gemm_reduce_unroll(schedule).unwrap_or(1),
        b_load_order: schedule_gemm_b_load_order(schedule),
    })
}

fn generated_kernel_manifest(candidate: &KernelCandidateMetadata) -> Value {
    json!({
        "schema_version": 1,
        "artifact_key": candidate.artifact_key().hex(),
        "family": &candidate.family,
        "generator": candidate.generated.generator,
        "materialization": materialization_json(&candidate.generated.materialization),
        "launch": launch_json(&candidate.launch),
        "operation": operation_json(&candidate.operation),
        "axes": candidate.axes.iter().map(axis_json).collect::<Vec<_>>(),
        "schedule": candidate
            .schedule
            .transforms
            .iter()
            .map(transform_json)
            .collect::<Vec<_>>(),
        "action_trace": candidate
            .action_trace
            .iter()
            .map(action_json)
            .collect::<Vec<_>>(),
        "score": candidate.score.map(score_json),
    })
}

fn selection_json(selection: &KernelOptimizationSelection) -> Value {
    json!({
        "schema_version": 1,
        "family": &selection.family,
        "artifact_key": &selection.artifact_key,
        "generator": &selection.generator,
        "launchable": selection.launchable,
        "action_trace": selection
            .action_trace
            .iter()
            .map(action_json)
            .collect::<Vec<_>>(),
        "score": selection.score.map(score_json),
    })
}

fn parse_selection_json(
    value: &Value,
) -> Result<KernelOptimizationSelection, KernelGenerationError> {
    let schema_version = required_u64(value, "schema_version")?;
    if schema_version != 1 {
        return Err(invalid_selection(format!(
            "unsupported schema_version {schema_version}"
        )));
    }
    Ok(KernelOptimizationSelection {
        family: required_str(value, "family")?.to_string(),
        artifact_key: required_str(value, "artifact_key")?.to_string(),
        generator: required_str(value, "generator")?.to_string(),
        launchable: required_bool(value, "launchable")?,
        action_trace: parse_action_trace(required_array(value, "action_trace")?)?,
        score: parse_optional_score(value.get("score").unwrap_or(&Value::Null))?,
    })
}

fn parse_action_trace(
    actions: &[Value],
) -> Result<Vec<KernelScheduleAction>, KernelGenerationError> {
    actions
        .iter()
        .enumerate()
        .map(|(index, action)| parse_action_json(action, index))
        .collect()
}

fn parse_action_json(
    value: &Value,
    index: usize,
) -> Result<KernelScheduleAction, KernelGenerationError> {
    let op = match required_str(value, "op")? {
        "split" => KernelScheduleActionOp::Split,
        "unroll" => KernelScheduleActionOp::Unroll,
        "tile-gemm" => KernelScheduleActionOp::TileGemm,
        "stride-order" => KernelScheduleActionOp::StrideOrder,
        op => {
            return Err(invalid_selection(format!(
                "action_trace[{index}] has unsupported op {op:?}"
            )));
        }
    };
    let materialization = match required_str(value, "materialization")? {
        "existing" => KernelActionMaterialization::Existing,
        "deferred-generated" => KernelActionMaterialization::DeferredGenerated,
        materialization => {
            return Err(invalid_selection(format!(
                "action_trace[{index}] has unsupported materialization {materialization:?}"
            )));
        }
    };
    Ok(KernelScheduleAction {
        op,
        axis: optional_u8(value, "axis")?,
        arg: parse_action_arg_json(required_field(value, "arg")?, index)?,
        materialization,
    })
}

fn parse_action_arg_json(
    value: &Value,
    index: usize,
) -> Result<KernelScheduleActionArg, KernelGenerationError> {
    match required_str(value, "kind")? {
        "factor" => Ok(KernelScheduleActionArg::Factor(required_u32(
            value, "value",
        )?)),
        "tile-3d" => Ok(KernelScheduleActionArg::Tile3d {
            m: required_u32(value, "m")?,
            n: required_u32(value, "n")?,
            k: required_u32(value, "k")?,
        }),
        "axis-order" => Ok(KernelScheduleActionArg::AxisOrder(
            required_array(value, "axes")?
                .iter()
                .enumerate()
                .map(|(axis_index, axis)| {
                    value_as_u8(axis).ok_or_else(|| {
                        invalid_selection(format!(
                            "action_trace[{index}].arg.axes[{axis_index}] must be a u8"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        kind => Err(invalid_selection(format!(
            "action_trace[{index}] has unsupported arg kind {kind:?}"
        ))),
    }
}

fn parse_optional_score(value: &Value) -> Result<Option<SearchScore>, KernelGenerationError> {
    if value.is_null() {
        return Ok(None);
    }
    let score_value = required_f64(value, "value")?;
    if !score_value.is_finite() {
        return Err(invalid_selection("score.value must be finite"));
    }
    let source = match required_str(value, "source")? {
        "heuristic" => SearchScoreSource::Heuristic,
        "measured" => SearchScoreSource::Measured,
        source => {
            return Err(invalid_selection(format!(
                "score.source is unsupported: {source:?}"
            )));
        }
    };
    Ok(Some(SearchScore {
        value: score_value,
        source,
        timing: parse_optional_timing(value.get("timing").unwrap_or(&Value::Null))?,
    }))
}

fn parse_optional_timing(
    value: &Value,
) -> Result<Option<OptimizationTiming>, KernelGenerationError> {
    if value.is_null() {
        return Ok(None);
    }
    let source = match required_str(value, "source")? {
        "wall-clock" => ProfileTimeSource::WallClock,
        "cuda-event" => ProfileTimeSource::CudaEvent,
        "host-self-time" => ProfileTimeSource::SelfTimeAccounting,
        source => {
            return Err(invalid_selection(format!(
                "score.timing.source is unsupported: {source:?}"
            )));
        }
    };
    let selected_seconds = required_f64(value, "selected_seconds")?;
    let selected = ProfileDuration::from_seconds_f64(selected_seconds).ok_or_else(|| {
        invalid_selection("score.timing.selected_seconds must be nonnegative and finite")
    })?;
    let samples = required_field(value, "samples")?;
    let sample_stats = SampleStats {
        count: required_usize(samples, "count")?,
        mean: required_f64(samples, "mean_seconds")?,
        median: required_f64(samples, "median_seconds")?,
        min: required_f64(samples, "min_seconds")?,
        max: required_f64(samples, "max_seconds")?,
    };
    if sample_stats.count == 0
        || !sample_stats.mean.is_finite()
        || !sample_stats.median.is_finite()
        || !sample_stats.min.is_finite()
        || !sample_stats.max.is_finite()
    {
        return Err(invalid_selection(
            "score.timing.samples must contain nonzero finite statistics",
        ));
    }
    Ok(Some(OptimizationTiming::new(
        source,
        required_usize(value, "warmup_count")?,
        sample_stats,
        selected,
    )))
}

fn required_field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, KernelGenerationError> {
    value
        .get(name)
        .ok_or_else(|| invalid_selection(format!("missing field {name:?}")))
}

fn required_str<'a>(value: &'a Value, name: &str) -> Result<&'a str, KernelGenerationError> {
    required_field(value, name)?
        .as_str()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be a string")))
}

fn required_bool(value: &Value, name: &str) -> Result<bool, KernelGenerationError> {
    required_field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be a bool")))
}

fn required_array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value], KernelGenerationError> {
    required_field(value, name)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be an array")))
}

fn required_u64(value: &Value, name: &str) -> Result<u64, KernelGenerationError> {
    required_field(value, name)?
        .as_u64()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be a u64")))
}

fn required_usize(value: &Value, name: &str) -> Result<usize, KernelGenerationError> {
    usize::try_from(required_u64(value, name)?)
        .map_err(|_| invalid_selection(format!("field {name:?} exceeds usize")))
}

fn required_u32(value: &Value, name: &str) -> Result<u32, KernelGenerationError> {
    u32::try_from(required_u64(value, name)?)
        .map_err(|_| invalid_selection(format!("field {name:?} exceeds u32")))
}

fn optional_u8(value: &Value, name: &str) -> Result<Option<u8>, KernelGenerationError> {
    match value.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value_as_u8(value)
            .ok_or_else(|| invalid_selection(format!("field {name:?} must be null or a u8")))
            .map(Some),
    }
}

fn value_as_u8(value: &Value) -> Option<u8> {
    value.as_u64().and_then(|value| u8::try_from(value).ok())
}

fn required_f64(value: &Value, name: &str) -> Result<f64, KernelGenerationError> {
    required_field(value, name)?
        .as_f64()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be an f64")))
}

fn invalid_selection(reason: impl Into<String>) -> KernelGenerationError {
    KernelGenerationError::InvalidSelection {
        reason: reason.into(),
    }
}

fn materialization_json(materialization: &KernelMaterialization) -> Value {
    match materialization {
        KernelMaterialization::Existing { symbol } => {
            json!({"kind": "existing", "symbol": symbol})
        }
        KernelMaterialization::DeferredGenerated {
            symbol_hint,
            reason,
        } => json!({
            "kind": "deferred-generated",
            "symbol_hint": symbol_hint,
            "reason": reason,
        }),
    }
}

fn launch_json(launch: &CudaLaunchSpec) -> Value {
    json!({
        "kernel": &launch.kernel,
        "grid_dim": [launch.grid_dim.x, launch.grid_dim.y, launch.grid_dim.z],
        "block_dim": [launch.block_dim.x, launch.block_dim.y, launch.block_dim.z],
        "shared_mem_bytes": launch.shared_mem_bytes,
    })
}

fn operation_json(operation: &TypedOperationSpec) -> Value {
    json!({
        "name": &operation.name,
        "kind": operation.kind.label(),
        "route": operation.route.label(),
        "inputs": operation.inputs.iter().map(tensor_json).collect::<Vec<_>>(),
        "outputs": operation.outputs.iter().map(tensor_json).collect::<Vec<_>>(),
    })
}

fn tensor_json(tensor: &TensorTypeSpec) -> Value {
    json!({
        "dtype": tensor.dtype.label(),
        "dtype_bits": tensor.dtype.bits(),
        "accumulator": tensor.accumulator.label(),
        "accumulator_bits": tensor.accumulator.bits(),
        "shape": &tensor.shape,
        "layout": &tensor.layout,
    })
}

fn axis_json(axis: &KernelAxis) -> Value {
    json!({
        "id": axis.id,
        "name": axis.name,
        "extent": axis.extent,
        "kind": match axis.kind {
            KernelAxisKind::Spatial => "spatial",
            KernelAxisKind::Reduction => "reduction",
        },
        "stride": axis.stride,
    })
}

fn transform_json(transform: &ScheduleTransform) -> Value {
    match transform {
        ScheduleTransform::Split { axis, factor } => {
            json!({"op": "split", "axis": axis, "factor": factor})
        }
        ScheduleTransform::Unroll { axis, factor } => {
            json!({"op": "unroll", "axis": axis, "factor": factor})
        }
        ScheduleTransform::LocalTile { axis, factor } => {
            json!({"op": "local-tile", "axis": axis, "factor": factor})
        }
        ScheduleTransform::ThreadGroup { axis, factor } => {
            json!({"op": "thread-group", "axis": axis, "factor": factor})
        }
        ScheduleTransform::TileGemm { m, n, k } => {
            json!({"op": "tile-gemm", "m": m, "n": n, "k": k})
        }
        ScheduleTransform::StrideOrder { axes } => {
            json!({"op": "stride-order", "axes": axes})
        }
    }
}

fn action_json(action: &KernelScheduleAction) -> Value {
    json!({
        "op": action.op.label(),
        "axis": action.axis,
        "arg": action_arg_json(&action.arg),
        "materialization": action.materialization.label(),
    })
}

fn action_arg_json(arg: &KernelScheduleActionArg) -> Value {
    match arg {
        KernelScheduleActionArg::Factor(factor) => json!({"kind": "factor", "value": factor}),
        KernelScheduleActionArg::Tile3d { m, n, k } => {
            json!({"kind": "tile-3d", "m": m, "n": n, "k": k})
        }
        KernelScheduleActionArg::AxisOrder(axes) => {
            json!({"kind": "axis-order", "axes": axes})
        }
    }
}

fn score_json(score: SearchScore) -> Value {
    json!({
        "value": score.value,
        "source": match score.source {
            SearchScoreSource::Heuristic => "heuristic",
            SearchScoreSource::Measured => "measured",
        },
        "timing": score.timing.map(timing_json),
    })
}

fn timing_json(timing: OptimizationTiming) -> Value {
    json!({
        "source": timing.source.label(),
        "warmup_count": timing.warmup_count,
        "selected_seconds": timing.selected.as_seconds_f64(),
        "samples": {
            "count": timing.samples.count,
            "mean_seconds": timing.samples.mean,
            "median_seconds": timing.samples.median,
            "min_seconds": timing.samples.min,
            "max_seconds": timing.samples.max,
        },
    })
}

fn render_bf16_matvec_source(symbol: &str, plan: MatvecSchedulePlan) -> String {
    let plan = plan.normalized();
    let rows_per_block = plan.rows.rows_per_block().max(1);
    let reduce_unroll = plan.reduce_unroll.max(1);
    let mut source = String::new();
    writeln!(
        source,
        "use cuda_device::{{DisjointSlice, kernel, thread, warp}};"
    )
    .expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "#[repr(transparent)]").expect("write to string");
    writeln!(source, "#[derive(Clone, Copy, Default)]").expect("write to string");
    writeln!(source, "pub struct Bf16(u16);").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "impl Bf16 {{").expect("write to string");
    writeln!(source, "    #[inline(always)]").expect("write to string");
    writeln!(source, "    pub fn to_f32(self) -> f32 {{").expect("write to string");
    writeln!(source, "        f32::from_bits((self.0 as u32) << 16)").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source, "}}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "const LANES_PER_ROW: u32 = 32;").expect("write to string");
    writeln!(source, "const ROWS_PER_BLOCK: u32 = {rows_per_block};").expect("write to string");
    writeln!(source, "const REDUCE_UNROLL: u32 = {reduce_unroll};").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "#[inline(always)]").expect("write to string");
    writeln!(source, "fn warp_reduce_sum(mut acc: f32) -> f32 {{").expect("write to string");
    writeln!(source, "    acc += warp::shuffle_down_f32(acc, 16);").expect("write to string");
    writeln!(source, "    acc += warp::shuffle_down_f32(acc, 8);").expect("write to string");
    writeln!(source, "    acc += warp::shuffle_down_f32(acc, 4);").expect("write to string");
    writeln!(source, "    acc += warp::shuffle_down_f32(acc, 2);").expect("write to string");
    writeln!(source, "    acc += warp::shuffle_down_f32(acc, 1);").expect("write to string");
    writeln!(source, "    acc").expect("write to string");
    writeln!(source, "}}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "#[kernel]").expect("write to string");
    writeln!(source, "pub fn {symbol}(").expect("write to string");
    writeln!(source, "    input: &[f32],").expect("write to string");
    writeln!(source, "    weight: &[Bf16],").expect("write to string");
    writeln!(source, "    rows: u32,").expect("write to string");
    writeln!(source, "    cols: u32,").expect("write to string");
    writeln!(source, "    row_stride: u32,").expect("write to string");
    writeln!(source, "    col_stride: u32,").expect("write to string");
    writeln!(source, "    _rows_per_block: u32,").expect("write to string");
    writeln!(source, "    mut out: DisjointSlice<f32>,").expect("write to string");
    writeln!(source, ") {{").expect("write to string");
    writeln!(source, "    let thread_x = thread::threadIdx_x();").expect("write to string");
    writeln!(source, "    let row_in_block = thread_x / LANES_PER_ROW;").expect("write to string");
    writeln!(source, "    if row_in_block >= ROWS_PER_BLOCK {{").expect("write to string");
    writeln!(source, "        return;").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(
        source,
        "    let row = (thread::blockIdx_x() * ROWS_PER_BLOCK + row_in_block) as usize;"
    )
    .expect("write to string");
    writeln!(source, "    if row >= rows as usize {{").expect("write to string");
    writeln!(source, "        return;").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "    let lane = warp::lane_id();").expect("write to string");
    writeln!(source, "    let cols = cols as usize;").expect("write to string");
    writeln!(source, "    let row_stride = row_stride as usize;").expect("write to string");
    writeln!(source, "    let col_stride = col_stride as usize;").expect("write to string");
    writeln!(source, "    let row_base = row * row_stride;").expect("write to string");
    writeln!(source, "    let mut acc = 0.0_f32;").expect("write to string");
    writeln!(source, "    let mut col = lane as usize;").expect("write to string");
    writeln!(source).expect("write to string");
    if reduce_unroll > 1 {
        let last_offset = (reduce_unroll - 1) * 32;
        let stride = reduce_unroll * 32;
        writeln!(source, "    while col + {last_offset} < cols {{").expect("write to string");
        for offset in 0..reduce_unroll {
            let col_expr = if offset == 0 {
                "col".to_string()
            } else {
                format!("col + {}", offset * 32)
            };
            writeln!(source, "        let col{offset} = {col_expr};").expect("write to string");
        }
        for offset in 0..reduce_unroll {
            writeln!(
                source,
                "        acc += weight[row_base + col{offset} * col_stride].to_f32() * input[col{offset}];"
            )
            .expect("write to string");
        }
        writeln!(source, "        col += {stride};").expect("write to string");
        writeln!(source, "    }}").expect("write to string");
        writeln!(source).expect("write to string");
    }
    writeln!(source, "    while col < cols {{").expect("write to string");
    writeln!(
        source,
        "        acc += weight[row_base + col * col_stride].to_f32() * input[col];"
    )
    .expect("write to string");
    writeln!(source, "        col += 32;").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "    let acc = warp_reduce_sum(acc);").expect("write to string");
    writeln!(source, "    if lane == 0 {{").expect("write to string");
    writeln!(source, "        unsafe {{").expect("write to string");
    writeln!(source, "            *out.get_unchecked_mut(row) = acc;").expect("write to string");
    writeln!(source, "        }}").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source, "}}").expect("write to string");
    source
}

fn render_f32_bf16_gemm_source(symbol: &str, plan: GemmSchedulePlan) -> String {
    let tile = plan.tile;
    let reduce_unroll = plan.reduce_unroll.max(1);
    let k_contiguous_b_load = plan.b_load_order == GemmBTileLoadOrder::KContiguous;
    let mut source = String::new();
    writeln!(
        source,
        "use cuda_device::{{DisjointSlice, SharedArray, kernel, thread}};"
    )
    .expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "#[repr(transparent)]").expect("write to string");
    writeln!(source, "#[derive(Clone, Copy, Default)]").expect("write to string");
    writeln!(source, "pub struct Bf16(u16);").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "impl Bf16 {{").expect("write to string");
    writeln!(source, "    #[inline(always)]").expect("write to string");
    writeln!(source, "    pub fn to_f32(self) -> f32 {{").expect("write to string");
    writeln!(source, "        f32::from_bits((self.0 as u32) << 16)").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source, "}}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "const TILE_M: usize = {};", tile.m).expect("write to string");
    writeln!(source, "const TILE_N: usize = {};", tile.n).expect("write to string");
    writeln!(source, "const TILE_K: usize = {};", tile.k).expect("write to string");
    writeln!(source, "const REDUCE_UNROLL: usize = {reduce_unroll};").expect("write to string");
    writeln!(source, "const TILE_A_ELEMS: usize = TILE_M * TILE_K;").expect("write to string");
    writeln!(source, "const TILE_B_ELEMS: usize = TILE_K * TILE_N;").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "#[kernel]").expect("write to string");
    writeln!(source, "pub fn {symbol}(").expect("write to string");
    writeln!(source, "    a: &[f32],").expect("write to string");
    writeln!(source, "    b: &[Bf16],").expect("write to string");
    writeln!(source, "    m: u32,").expect("write to string");
    writeln!(source, "    n: u32,").expect("write to string");
    writeln!(source, "    k: u32,").expect("write to string");
    writeln!(source, "    a_row_stride: u32,").expect("write to string");
    writeln!(source, "    a_col_stride: u32,").expect("write to string");
    writeln!(source, "    b_row_stride: u32,").expect("write to string");
    writeln!(source, "    b_col_stride: u32,").expect("write to string");
    writeln!(source, "    c_row_stride: u32,").expect("write to string");
    writeln!(source, "    c_col_stride: u32,").expect("write to string");
    writeln!(source, "    alpha: f32,").expect("write to string");
    writeln!(source, "    beta: f32,").expect("write to string");
    writeln!(source, "    mut c: DisjointSlice<f32>,").expect("write to string");
    writeln!(source, ") {{").expect("write to string");
    writeln!(
        source,
        "    static mut TILE_A: SharedArray<f32, TILE_A_ELEMS> = SharedArray::UNINIT;"
    )
    .expect("write to string");
    writeln!(
        source,
        "    static mut TILE_B: SharedArray<f32, TILE_B_ELEMS> = SharedArray::UNINIT;"
    )
    .expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "    let tx = thread::threadIdx_x() as usize;").expect("write to string");
    writeln!(source, "    let ty = thread::threadIdx_y() as usize;").expect("write to string");
    writeln!(source, "    if tx >= TILE_N || ty >= TILE_M {{").expect("write to string");
    writeln!(source, "        return;").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(
        source,
        "    let row = thread::blockIdx_y() as usize * TILE_M + ty;"
    )
    .expect("write to string");
    writeln!(
        source,
        "    let col = thread::blockIdx_x() as usize * TILE_N + tx;"
    )
    .expect("write to string");
    writeln!(source, "    let tid = ty * TILE_N + tx;").expect("write to string");
    writeln!(source, "    let thread_count = TILE_M * TILE_N;").expect("write to string");
    writeln!(source, "    let m = m as usize;").expect("write to string");
    writeln!(source, "    let n = n as usize;").expect("write to string");
    writeln!(source, "    let k = k as usize;").expect("write to string");
    writeln!(source, "    let a_row_stride = a_row_stride as usize;").expect("write to string");
    writeln!(source, "    let a_col_stride = a_col_stride as usize;").expect("write to string");
    writeln!(source, "    let b_row_stride = b_row_stride as usize;").expect("write to string");
    writeln!(source, "    let b_col_stride = b_col_stride as usize;").expect("write to string");
    writeln!(source, "    let c_row_stride = c_row_stride as usize;").expect("write to string");
    writeln!(source, "    let c_col_stride = c_col_stride as usize;").expect("write to string");
    writeln!(source, "    let mut acc = 0.0_f32;").expect("write to string");
    writeln!(source, "    let mut k_base = 0;").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "    while k_base < k {{").expect("write to string");
    writeln!(source, "        let mut load = tid;").expect("write to string");
    writeln!(source, "        while load < TILE_A_ELEMS {{").expect("write to string");
    writeln!(source, "            let tile_row = load / TILE_K;").expect("write to string");
    writeln!(source, "            let tile_col = load % TILE_K;").expect("write to string");
    writeln!(
        source,
        "            let global_row = thread::blockIdx_y() as usize * TILE_M + tile_row;"
    )
    .expect("write to string");
    writeln!(source, "            let global_col = k_base + tile_col;").expect("write to string");
    writeln!(source, "            unsafe {{").expect("write to string");
    writeln!(
        source,
        "                TILE_A[load] = if global_row < m && global_col < k {{"
    )
    .expect("write to string");
    writeln!(
        source,
        "                    a[global_row * a_row_stride + global_col * a_col_stride]"
    )
    .expect("write to string");
    writeln!(source, "                }} else {{").expect("write to string");
    writeln!(source, "                    0.0").expect("write to string");
    writeln!(source, "                }};").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "            load += thread_count;").expect("write to string");
    writeln!(source, "        }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "        load = tid;").expect("write to string");
    writeln!(source, "        while load < TILE_B_ELEMS {{").expect("write to string");
    if k_contiguous_b_load {
        writeln!(source, "            let tile_row = load % TILE_K;").expect("write to string");
        writeln!(source, "            let tile_col = load / TILE_K;").expect("write to string");
    } else {
        writeln!(source, "            let tile_row = load / TILE_N;").expect("write to string");
        writeln!(source, "            let tile_col = load % TILE_N;").expect("write to string");
    }
    writeln!(
        source,
        "            let b_smem_index = tile_row * TILE_N + tile_col;"
    )
    .expect("write to string");
    writeln!(source, "            let global_row = k_base + tile_row;").expect("write to string");
    writeln!(
        source,
        "            let global_col = thread::blockIdx_x() as usize * TILE_N + tile_col;"
    )
    .expect("write to string");
    writeln!(source, "            unsafe {{").expect("write to string");
    writeln!(
        source,
        "                TILE_B[b_smem_index] = if global_row < k && global_col < n {{"
    )
    .expect("write to string");
    writeln!(
        source,
        "                    b[global_row * b_row_stride + global_col * b_col_stride].to_f32()"
    )
    .expect("write to string");
    writeln!(source, "                }} else {{").expect("write to string");
    writeln!(source, "                    0.0").expect("write to string");
    writeln!(source, "                }};").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "            load += thread_count;").expect("write to string");
    writeln!(source, "        }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "        thread::sync_threads();").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "        unsafe {{").expect("write to string");
    writeln!(source, "            let mut kk = 0;").expect("write to string");
    writeln!(source, "            while kk + REDUCE_UNROLL <= TILE_K {{").expect("write to string");
    for offset in 0..reduce_unroll {
        let k_expr = if offset == 0 {
            "kk".to_string()
        } else {
            format!("kk + {offset}")
        };
        writeln!(
            source,
            "                acc += TILE_A[ty * TILE_K + {k_expr}] * TILE_B[({k_expr}) * TILE_N + tx];"
        )
        .expect("write to string");
    }
    writeln!(source, "                kk += REDUCE_UNROLL;").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "            while kk < TILE_K {{").expect("write to string");
    writeln!(
        source,
        "                acc += TILE_A[ty * TILE_K + kk] * TILE_B[kk * TILE_N + tx];"
    )
    .expect("write to string");
    writeln!(source, "                kk += 1;").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "        }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "        thread::sync_threads();").expect("write to string");
    writeln!(source, "        k_base += TILE_K;").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "    if row < m && col < n {{").expect("write to string");
    writeln!(
        source,
        "        let c_offset = row * c_row_stride + col * c_col_stride;"
    )
    .expect("write to string");
    writeln!(source, "        unsafe {{").expect("write to string");
    writeln!(
        source,
        "            let c_elem = c.get_unchecked_mut(c_offset);"
    )
    .expect("write to string");
    writeln!(source, "            let current = *c_elem;").expect("write to string");
    writeln!(
        source,
        "            *c_elem = alpha * acc + beta * current;"
    )
    .expect("write to string");
    writeln!(source, "        }}").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source, "}}").expect("write to string");
    source
}

fn standalone_package_name(candidate: &KernelCandidateMetadata) -> String {
    format!("nn_rust_kernel_{}", candidate.artifact_key().hex())
}

fn standalone_cargo_toml(package_name: &str) -> String {
    let mut manifest = String::new();
    writeln!(manifest, "[package]").expect("write to string");
    writeln!(manifest, "name = \"{package_name}\"").expect("write to string");
    writeln!(manifest, "version = \"0.1.0\"").expect("write to string");
    writeln!(manifest, "edition = \"2024\"").expect("write to string");
    writeln!(manifest).expect("write to string");
    writeln!(manifest, "[workspace]").expect("write to string");
    writeln!(manifest).expect("write to string");
    writeln!(manifest, "[dependencies]").expect("write to string");
    writeln!(
        manifest,
        "cuda-device = {{ git = \"https://github.com/NVlabs/cuda-oxide.git\", tag = \"v0.1.0\" }}"
    )
    .expect("write to string");
    writeln!(
        manifest,
        "cuda-host = {{ git = \"https://github.com/NVlabs/cuda-oxide.git\", tag = \"v0.1.0\" }}"
    )
    .expect("write to string");
    manifest
}

fn standalone_main_source(kernel_source: &str) -> String {
    let mut source = String::new();
    source.push_str(kernel_source);
    if !source.ends_with('\n') {
        source.push('\n');
    }
    writeln!(source).expect("write to string");
    writeln!(source, "fn main() {{}}").expect("write to string");
    source
}

fn sanitize_path_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "kernel".to_string()
    } else {
        sanitized
    }
}

fn sanitize_identifier(value: &str) -> String {
    let mut sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        sanitized.push_str("kernel");
    }
    if sanitized
        .as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_digit())
    {
        sanitized.insert_str(0, "k_");
    }
    sanitized
}

fn metadata_key(
    family: &str,
    axes: &[KernelAxis],
    schedule: &KernelSchedule,
    launch: &CudaLaunchSpec,
) -> KernelMetadataKey {
    let mut state = FNV_OFFSET;
    state = hash_str(state, family);
    for axis in axes {
        state = hash_u64(state, axis.id as u64);
        state = hash_str(state, axis.name);
        state = hash_u64(state, axis.extent as u64);
        state = hash_u64(
            state,
            match axis.kind {
                KernelAxisKind::Spatial => 1,
                KernelAxisKind::Reduction => 2,
            },
        );
        state = hash_u64(state, axis.stride.unwrap_or(usize::MAX) as u64);
    }
    for transform in &schedule.transforms {
        state = hash_transform(state, transform);
    }
    state = hash_str(state, &launch.kernel);
    state = hash_u64(state, launch.grid_dim.x as u64);
    state = hash_u64(state, launch.grid_dim.y as u64);
    state = hash_u64(state, launch.grid_dim.z as u64);
    state = hash_u64(state, launch.block_dim.x as u64);
    state = hash_u64(state, launch.block_dim.y as u64);
    state = hash_u64(state, launch.block_dim.z as u64);
    state = hash_u64(state, launch.shared_mem_bytes as u64);
    KernelMetadataKey(state)
}

fn search_report_key(report: &OptimizationSearchReport) -> KernelMetadataKey {
    let mut state = FNV_OFFSET;
    state = hash_str(state, "optimization-search-report");
    state = hash_str(state, &report.family);
    state = hash_u64(state, report.config.beam_width as u64);
    state = hash_u64(state, report.config.max_depth as u64);
    state = hash_u64(state, u64::from(report.config.require_launchable));
    state = hash_u64(state, report.explored as u64);
    state = hash_u64(state, report.rejected as u64);
    if let Some(best) = &report.best {
        state = hash_optimization_candidate(state, best);
    } else {
        state = hash_str(state, "no-best");
    }
    for candidate in &report.beam {
        state = hash_optimization_candidate(state, candidate);
    }
    KernelMetadataKey(state)
}

fn auto_search_report_key(report: &AutoOptimizationSearchReport) -> KernelMetadataKey {
    let mut state = FNV_OFFSET;
    state = hash_str(state, "auto-optimization-search-report");
    state = hash_str(state, &report.family);
    state = hash_u64(state, report.config.beam_width as u64);
    state = hash_u64(state, report.config.max_steps as u64);
    state = hash_u64(state, u64::from(report.config.require_launchable));
    state = hash_u64(state, report.config.min_score_improvement.to_bits());
    state = hash_u64(state, report.explored as u64);
    state = hash_u64(state, report.rejected as u64);
    state = hash_str(state, report.exit_reason.label());
    if let ProfilingAutoOptimizationExitReason::NoImprovement { best_delta } = report.exit_reason {
        state = hash_u64(state, best_delta.to_bits());
    }
    for step in &report.steps {
        state = hash_u64(state, step.depth as u64);
        state = hash_u64(state, step.input_beam_len as u64);
        state = hash_u64(state, step.generated as u64);
        state = hash_u64(state, step.accepted as u64);
        state = hash_u64(state, step.rejected as u64);
        state = hash_optional_score(state, step.best_before);
        state = hash_optional_score(state, step.best_after);
        if let Some(improvement) = step.improvement {
            state = hash_str(state, "improvement");
            state = hash_u64(state, improvement.to_bits());
        } else {
            state = hash_str(state, "no-improvement-value");
        }
    }
    if let Some(best) = &report.best {
        state = hash_optimization_candidate(state, best);
    } else {
        state = hash_str(state, "no-best");
    }
    for candidate in &report.beam {
        state = hash_optimization_candidate(state, candidate);
    }
    KernelMetadataKey(state)
}

fn hash_optional_score(mut state: u64, score: Option<SearchScore>) -> u64 {
    if let Some(score) = score {
        state = hash_str(state, "score");
        state = hash_str(state, score.source.label());
        hash_u64(state, score.value.to_bits())
    } else {
        hash_str(state, "no-score")
    }
}

fn hash_action_space_set(mut state: u64, action_space: &KernelActionSpaceSet) -> u64 {
    state = hash_str(state, "action-space-set");
    state = hash_u64(state, action_space.spaces.len() as u64);
    for space in &action_space.spaces {
        state = hash_action_space(state, space);
    }
    state
}

fn hash_action_space(mut state: u64, action_space: &KernelActionSpace) -> u64 {
    match action_space {
        KernelActionSpace::Split { variants } => {
            state = hash_str(state, "split");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.axis as u64);
                state = hash_u64(state, variant.factor as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        KernelActionSpace::Unroll { axis, factors } => {
            state = hash_str(state, "unroll");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        KernelActionSpace::TileGemm { variants } => {
            state = hash_str(state, "tile-gemm");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.tile.m as u64);
                state = hash_u64(state, variant.tile.n as u64);
                state = hash_u64(state, variant.tile.k as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        KernelActionSpace::StrideOrder { orders } => {
            state = hash_str(state, "stride-order");
            state = hash_u64(state, orders.len() as u64);
            for order in orders {
                state = hash_u64(state, order.len() as u64);
                for axis in order {
                    state = hash_u64(state, *axis as u64);
                }
            }
            state
        }
    }
}

fn hash_optimization_candidate(mut state: u64, candidate: &OptimizationCandidateSpec) -> u64 {
    state = hash_str(state, &candidate.family);
    state = hash_str(state, &candidate.artifact_key);
    state = hash_str(state, &candidate.generator);
    state = hash_u64(state, u64::from(candidate.launchable));
    state = hash_str(state, &candidate.launch.kernel);
    state = hash_u64(state, candidate.launch.grid_dim.x as u64);
    state = hash_u64(state, candidate.launch.grid_dim.y as u64);
    state = hash_u64(state, candidate.launch.grid_dim.z as u64);
    state = hash_u64(state, candidate.launch.block_dim.x as u64);
    state = hash_u64(state, candidate.launch.block_dim.y as u64);
    state = hash_u64(state, candidate.launch.block_dim.z as u64);
    state = hash_u64(state, candidate.launch.shared_mem_bytes as u64);
    state = hash_operation_spec(state, &candidate.operation);
    for action in &candidate.action_trace {
        state = hash_str(state, action.op.label());
        state = hash_u64(state, action.axis.unwrap_or(u8::MAX) as u64);
        state = match &action.arg {
            KernelScheduleActionArg::Factor(factor) => {
                let state = hash_str(state, "factor");
                hash_u64(state, *factor as u64)
            }
            KernelScheduleActionArg::Tile3d { m, n, k } => {
                let mut state = hash_str(state, "tile-3d");
                state = hash_u64(state, *m as u64);
                state = hash_u64(state, *n as u64);
                hash_u64(state, *k as u64)
            }
            KernelScheduleActionArg::AxisOrder(axes) => {
                let mut state = hash_str(state, "axis-order");
                for axis in axes {
                    state = hash_u64(state, *axis as u64);
                }
                state
            }
        };
        state = hash_str(state, action.materialization.label());
    }
    if let Some(score) = candidate.score {
        state = hash_str(state, score.source.label());
        state = hash_u64(state, score.value.to_bits());
        if let Some(timing) = score.timing {
            state = hash_str(state, timing.source.label());
            state = hash_u64(state, timing.warmup_count as u64);
            state = hash_u64(
                state,
                timing.selected.as_nanos_u128().min(u64::MAX as u128) as u64,
            );
            state = hash_u64(state, timing.samples.count as u64);
            state = hash_u64(state, timing.samples.mean.to_bits());
            state = hash_u64(state, timing.samples.median.to_bits());
            state = hash_u64(state, timing.samples.min.to_bits());
            state = hash_u64(state, timing.samples.max.to_bits());
        }
    } else {
        state = hash_str(state, "no-score");
    }
    state
}

fn hash_operation_spec(mut state: u64, operation: &TypedOperationSpec) -> u64 {
    state = hash_str(state, &operation.name);
    state = hash_str(state, operation.kind.label());
    state = hash_str(state, operation.route.label());
    state = hash_tensor_specs(state, "inputs", &operation.inputs);
    state = hash_tensor_specs(state, "outputs", &operation.outputs);
    if let Some(launch) = &operation.launch {
        state = hash_str(state, "operation-launch");
        state = hash_str(state, &launch.kernel);
        state = hash_u64(state, launch.grid_dim.x as u64);
        state = hash_u64(state, launch.grid_dim.y as u64);
        state = hash_u64(state, launch.grid_dim.z as u64);
        state = hash_u64(state, launch.block_dim.x as u64);
        state = hash_u64(state, launch.block_dim.y as u64);
        state = hash_u64(state, launch.block_dim.z as u64);
        state = hash_u64(state, launch.shared_mem_bytes as u64);
    } else {
        state = hash_str(state, "no-operation-launch");
    }
    state
}

fn hash_tensor_specs(mut state: u64, label: &str, specs: &[TensorTypeSpec]) -> u64 {
    state = hash_str(state, label);
    state = hash_u64(state, specs.len() as u64);
    for spec in specs {
        state = hash_tensor_spec(state, spec);
    }
    state
}

fn hash_tensor_spec(mut state: u64, spec: &TensorTypeSpec) -> u64 {
    state = hash_str(state, spec.dtype.label());
    state = hash_str(state, spec.accumulator.label());
    state = hash_u64(state, spec.shape.len() as u64);
    for dim in &spec.shape {
        state = hash_u64(state, *dim as u64);
    }
    if let Some(layout) = &spec.layout {
        state = hash_str(state, layout);
    } else {
        state = hash_str(state, "no-layout");
    }
    state
}

fn hash_transform(mut state: u64, transform: &ScheduleTransform) -> u64 {
    match transform {
        ScheduleTransform::Split { axis, factor } => {
            state = hash_str(state, "split");
            state = hash_u64(state, *axis as u64);
            hash_u64(state, *factor as u64)
        }
        ScheduleTransform::Unroll { axis, factor } => {
            state = hash_str(state, "unroll");
            state = hash_u64(state, *axis as u64);
            hash_u64(state, *factor as u64)
        }
        ScheduleTransform::LocalTile { axis, factor } => {
            state = hash_str(state, "local-tile");
            state = hash_u64(state, *axis as u64);
            hash_u64(state, *factor as u64)
        }
        ScheduleTransform::ThreadGroup { axis, factor } => {
            state = hash_str(state, "thread-group");
            state = hash_u64(state, *axis as u64);
            hash_u64(state, *factor as u64)
        }
        ScheduleTransform::TileGemm { m, n, k } => {
            state = hash_str(state, "tile-gemm");
            state = hash_u64(state, *m as u64);
            state = hash_u64(state, *n as u64);
            hash_u64(state, *k as u64)
        }
        ScheduleTransform::StrideOrder { axes } => {
            state = hash_str(state, "stride-order");
            for axis in axes {
                state = hash_u64(state, *axis as u64);
            }
            state
        }
    }
}

fn hash_str(state: u64, value: &str) -> u64 {
    hash_bytes(state, value.as_bytes())
}

fn hash_u64(state: u64, value: u64) -> u64 {
    hash_bytes(state, &value.to_le_bytes())
}

fn hash_bytes(mut state: u64, value: &[u8]) -> u64 {
    for byte in value {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(FNV_PRIME);
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_generated_root() -> PathBuf {
        runtime::default_artifact_dir()
            .join("test-generated")
            .join(format!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system clock should be after unix epoch")
                    .as_nanos()
            ))
    }

    fn remove_test_generated_root(root: &Path) {
        fs::remove_dir_all(root).expect("test-generated kernel artifact directory should clean up");
        if let Some(parent) = root.parent() {
            let _ = fs::remove_dir(parent);
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct MatvecWithoutUnrollSpace(MatvecSearchProblem);

    impl KernelMetadataSearchProblem for MatvecWithoutUnrollSpace {
        fn seed(&self) -> KernelCandidateMetadata {
            self.0.seed()
        }

        fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
            expand_with_schedule_actions(self, candidate)
        }

        fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
            self.0.score(candidate)
        }
    }

    impl KernelActionSearchProblem for MatvecWithoutUnrollSpace {
        fn search_space(&self) -> KernelActionSpaceSet {
            let mut spaces = self.0.search_space();
            spaces
                .spaces
                .retain(|space| !matches!(space, KernelActionSpace::Unroll { .. }));
            spaces
        }

        fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
            let mut spaces = self.0.action_spaces(candidate);
            spaces
                .spaces
                .retain(|space| !matches!(space, KernelActionSpace::Unroll { .. }));
            spaces
        }

        fn apply_schedule_action(
            &self,
            candidate: &KernelCandidateMetadata,
            action: &KernelScheduleAction,
        ) -> Option<KernelCandidateMetadata> {
            if matches!(action.op, KernelScheduleActionOp::Unroll) {
                return None;
            }
            self.0.apply_schedule_action(candidate, action)
        }
    }

    #[test]
    fn matvec_search_keeps_only_metadata_for_rows_per_block_variants() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let result = beam_search_metadata(
            &problem,
            BeamSearchConfig {
                beam_width: 2,
                max_depth: 1,
                require_launchable: true,
            },
        );
        let best = result
            .best
            .expect("matvec search should produce a candidate");
        assert_eq!(result.explored, 36);
        assert_eq!(result.rejected, 32);
        assert!(best.is_launchable());
        assert_eq!(best.launch.kernel, "matvec_bf16_kernel");
        assert_eq!(best.launch.grid_dim.x, 512);
        assert_eq!(best.launch.block_dim.x, 256);
        assert_eq!(schedule_rows_per_block(&best.schedule), Some(8));
    }

    #[test]
    fn matvec_search_can_prefer_smaller_row_groups_for_tiny_outputs() {
        let problem = MatvecSearchProblem::bf16_row_major(10, 4096);
        let result = beam_search_metadata(
            &problem,
            BeamSearchConfig {
                beam_width: 1,
                max_depth: 1,
                require_launchable: true,
            },
        );
        let best = result
            .best
            .expect("matvec search should produce a candidate");
        assert_eq!(schedule_rows_per_block(&best.schedule), Some(2));
    }

    #[test]
    fn matvec_search_exposes_generated_row_split_metadata() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let seed = problem.seed();
        let candidates = problem.expand(&seed);
        assert_eq!(candidates.len(), 36);

        let generated = candidates
            .iter()
            .find(|candidate| {
                !candidate.is_launchable()
                    && schedule_rows_per_block(&candidate.schedule) == Some(13)
            })
            .expect("matvec search should expose arbitrary generated rows-per-block metadata");
        assert_eq!(generated.launch.kernel, "matvec_bf16_rows13");
        assert_eq!(generated.launch.grid_dim.x, 316);
        assert_eq!(generated.launch.block_dim.x, 416);
        assert!(matches!(
            generated.generated.materialization,
            KernelMaterialization::DeferredGenerated { .. }
        ));
        assert_eq!(generated.generated.generator, "row-major-matvec-generator");
    }

    #[test]
    fn matvec_action_space_exposes_existing_and_deferred_row_splits() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let seed = problem.seed();
        let spaces = problem.action_spaces(&seed);
        let actions = problem.schedule_actions(&seed);

        assert_eq!(spaces.spaces.len(), 1);
        assert_eq!(spaces.actions(), actions);
        let KernelActionSpace::Split { variants } = &spaces.spaces[0] else {
            panic!("matvec seed should expose split action-space metadata");
        };
        assert_eq!(variants.len(), 36);
        assert!(variants.contains(&KernelAxisFactorAction::new(
            0,
            8,
            KernelActionMaterialization::Existing
        )));
        assert!(variants.contains(&KernelAxisFactorAction::new(
            0,
            13,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(variants.contains(&KernelAxisFactorAction::new(
            0,
            32,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(!variants.contains(&KernelAxisFactorAction::new(
            0,
            32,
            KernelActionMaterialization::Existing
        )));
        let complete_space = problem.search_space();
        assert_eq!(complete_space.spaces.len(), 2);
        assert!(matches!(
            complete_space.spaces[0],
            KernelActionSpace::Split { .. }
        ));
        assert!(matches!(
            complete_space.spaces[1],
            KernelActionSpace::Unroll { .. }
        ));
        assert_eq!(actions.len(), 36);
        assert!(actions.contains(&KernelScheduleAction::split(
            0,
            8,
            KernelActionMaterialization::Existing
        )));
        assert!(actions.contains(&KernelScheduleAction::split(
            0,
            13,
            KernelActionMaterialization::DeferredGenerated
        )));

        let generated = problem
            .apply_schedule_action(
                &seed,
                &KernelScheduleAction::split(0, 13, KernelActionMaterialization::DeferredGenerated),
            )
            .expect("row split action should produce candidate metadata");
        assert_eq!(generated.launch.kernel, "matvec_bf16_rows13");
        assert_eq!(schedule_rows_per_block(&generated.schedule), Some(13));
        assert_eq!(
            generated.action_trace,
            vec![KernelScheduleAction::split(
                0,
                13,
                KernelActionMaterialization::DeferredGenerated
            )]
        );
        assert!(matches!(
            generated.generated.materialization,
            KernelMaterialization::DeferredGenerated { .. }
        ));
    }

    #[test]
    fn matvec_generated_row_split_exposes_reduce_unroll_actions() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let seed = problem.seed();
        let rows8 = problem
            .apply_schedule_action(
                &seed,
                &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            )
            .expect("row split action should produce generated candidate metadata");
        let spaces = problem.action_spaces(&rows8);
        let actions = problem.schedule_actions(&rows8);

        assert_eq!(spaces.spaces.len(), 1);
        assert_eq!(spaces.actions(), actions);
        let KernelActionSpace::Unroll { axis, factors } = &spaces.spaces[0] else {
            panic!("generated matvec split should expose unroll action-space metadata");
        };
        assert_eq!(*axis, 1);
        assert_eq!(factors.len(), 31);
        assert_eq!(factors.first().copied(), Some(1));
        assert_eq!(factors.last().copied(), Some(32));
        assert!(!factors.contains(&MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL));
        assert!(factors.contains(&7));
        assert!(actions.contains(&KernelScheduleAction::unroll(1, 1)));
        assert!(actions.contains(&KernelScheduleAction::unroll(1, 2)));
        assert!(actions.contains(&KernelScheduleAction::unroll(1, 7)));
        assert!(actions.contains(&KernelScheduleAction::unroll(1, 32)));

        let unrolled = problem
            .apply_schedule_action(&rows8, &KernelScheduleAction::unroll(1, 7))
            .expect("reduce unroll action should produce generated candidate metadata");
        assert_eq!(unrolled.launch.kernel, "matvec_bf16_rows8_u7");
        assert_eq!(schedule_matvec_reduce_unroll(&unrolled.schedule), Some(7));
        assert_eq!(
            unrolled.action_trace,
            vec![
                KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
                KernelScheduleAction::unroll(1, 7),
            ]
        );
        assert_ne!(rows8.artifact_key(), unrolled.artifact_key());
    }

    #[test]
    fn matvec_unroll_action_space_is_bounded_by_problem_shape() {
        let problem = MatvecSearchProblem::bf16_row_major(16, 3);
        let seed = problem.seed();
        let rows8 = problem
            .apply_schedule_action(
                &seed,
                &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            )
            .expect("row split action should produce generated candidate metadata");
        let KernelActionSpace::Unroll { factors, .. } = &problem.action_spaces(&rows8).spaces[0]
        else {
            panic!("generated matvec split should expose unroll metadata");
        };

        assert_eq!(factors.as_slice(), &[1, 2, 3]);
        assert!(
            problem
                .apply_schedule_action(&rows8, &KernelScheduleAction::unroll(1, 3))
                .is_some()
        );
        assert!(
            problem
                .apply_schedule_action(&rows8, &KernelScheduleAction::unroll(1, 5))
                .is_none()
        );
    }

    #[test]
    fn matvec_search_can_rank_reduce_unroll_variants_when_allowed() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let result = beam_search_metadata_with_scorer(
            &problem,
            BeamSearchConfig {
                beam_width: 40,
                max_depth: 2,
                require_launchable: false,
            },
            |candidate| {
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if plan.rows.rows_per_block() == 13 && plan.reduce_unroll == 7 {
                    SearchScore::measured(0.0)
                } else {
                    SearchScore::measured(
                        100.0
                            + f64::from(plan.rows.rows_per_block())
                            + f64::from(plan.reduce_unroll),
                    )
                }
            },
        );
        let best = result
            .best
            .expect("matvec search should produce an unrolled generated candidate");
        let plan = schedule_matvec_plan(&best.schedule).expect("best candidate should have plan");

        assert_eq!(plan.rows.rows_per_block(), 13);
        assert_eq!(plan.reduce_unroll, 7);
        assert_eq!(best.launch.kernel, "matvec_bf16_rows13_u7");
        assert_eq!(
            best.score.map(|score| score.source),
            Some(SearchScoreSource::Measured)
        );
    }

    #[test]
    fn auto_optimize_preserves_parent_when_children_do_not_improve() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let result = auto_optimize_metadata_with_scorer(
            &problem,
            AutoOptimizeConfig {
                beam_width: 8,
                max_steps: 2,
                require_launchable: false,
                min_score_improvement: 0.0,
            },
            |candidate| {
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
                    SearchScore::measured(1.0)
                } else {
                    SearchScore::measured(10.0)
                }
            },
        );
        let best = result
            .best
            .expect("auto optimize should keep the best parent candidate");
        let best_plan = schedule_matvec_plan(&best.schedule).expect("best candidate should plan");

        assert_eq!(
            result.exit_reason,
            AutoOptimizeExitReason::NoImprovement { best_delta: -9.0 }
        );
        assert_eq!(result.steps.len(), 2);
        assert_eq!(
            best_plan.reduce_unroll,
            MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL
        );
        assert_eq!(best.score.and_then(|score| Some(score.value)), Some(1.0));
    }

    #[test]
    fn auto_optimize_accepts_improving_generated_action() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let result = auto_optimize_metadata_with_scorer(
            &problem,
            AutoOptimizeConfig {
                beam_width: 8,
                max_steps: 2,
                require_launchable: false,
                min_score_improvement: 0.0,
            },
            |candidate| {
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                match plan.reduce_unroll {
                    7 => SearchScore::measured(1.0),
                    factor if factor == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL => {
                        SearchScore::measured(10.0)
                    }
                    _ => SearchScore::measured(5.0),
                }
            },
        );
        let best = result
            .best
            .expect("auto optimize should accept the improving generated candidate");
        let best_plan = schedule_matvec_plan(&best.schedule).expect("best candidate should plan");

        assert_eq!(result.exit_reason, AutoOptimizeExitReason::MaxSteps);
        assert_eq!(result.steps.len(), 2);
        assert_eq!(result.steps[1].improvement, Some(9.0));
        assert_eq!(best_plan.reduce_unroll, 7);
        assert_eq!(best.score.and_then(|score| Some(score.value)), Some(1.0));
    }

    #[test]
    fn candidate_projects_to_profiling_optimization_spec() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let seed = problem.seed();
        let mut candidate = problem
            .apply_schedule_action(
                &seed,
                &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            )
            .expect("row split action should produce candidate metadata");
        candidate.score = SearchScore::heuristic(1.0);

        let spec = candidate.optimization_spec();

        assert_eq!(spec.family, "matvec-bf16-row-major");
        assert_eq!(spec.artifact_key, candidate.artifact_key().hex());
        assert_eq!(spec.generator, "row-major-matvec-generator");
        assert!(!spec.launchable);
        assert_eq!(spec.launch.kernel, "matvec_bf16_rows8");
        assert_eq!(spec.operation.kind, OperationKind::Matvec);
        assert_eq!(spec.action_trace, candidate.action_trace);
        assert_eq!(spec.score, candidate.score);
    }

    #[test]
    fn action_trace_replay_reconstructs_generated_matvec_candidate() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let actions = vec![
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::unroll(1, 8),
        ];

        let candidate = replay_schedule_actions(&problem, &actions)
            .expect("valid matvec action trace should replay into candidate metadata");

        assert_eq!(candidate.family, "matvec-bf16-row-major");
        assert_eq!(candidate.action_trace, actions);
        assert_eq!(candidate.launch.kernel, "matvec_bf16_rows8_u8");
        assert_eq!(schedule_rows_per_block(&candidate.schedule), Some(8));
        assert_eq!(schedule_matvec_reduce_unroll(&candidate.schedule), Some(8));
        assert!(!candidate.is_launchable());

        let generated = MatvecRustCudaGenerator
            .source_for(&candidate)
            .expect("replayed generated candidate should render source on demand");
        assert_eq!(generated.symbol, "matvec_bf16_rows8_u8");
        assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
        assert!(generated.source.contains("const REDUCE_UNROLL: u32 = 8;"));
    }

    #[test]
    fn optimization_candidate_spec_replay_checks_artifact_key() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let candidate = replay_schedule_actions(
            &problem,
            &[
                KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
                KernelScheduleAction::unroll(1, 8),
            ],
        )
        .expect("valid matvec action trace should replay into candidate metadata");
        let spec = candidate.optimization_spec();

        let replayed = replay_optimization_candidate_spec(&problem, &spec)
            .expect("matching optimization spec should replay");
        assert_eq!(replayed.artifact_key(), candidate.artifact_key());

        let mut stale_spec = spec;
        stale_spec.artifact_key = "0000000000000000".to_string();
        assert_eq!(
            replay_optimization_candidate_spec(&problem, &stale_spec),
            Err(KernelActionReplayError::ArtifactKeyMismatch {
                expected: "0000000000000000".to_string(),
                actual: candidate.artifact_key().hex(),
            })
        );
    }

    #[test]
    fn action_trace_replay_rejects_invalid_action_sequence() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let action = KernelScheduleAction::unroll(1, 8);

        assert_eq!(
            replay_schedule_actions(&problem, std::slice::from_ref(&action)),
            Err(KernelActionReplayError::InvalidAction { index: 0, action })
        );
    }

    #[test]
    fn action_trace_replay_reconstructs_generated_gemm_candidate() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let actions = vec![
            KernelScheduleAction::tile_gemm(
                16,
                32,
                16,
                KernelActionMaterialization::DeferredGenerated,
            ),
            KernelScheduleAction::unroll(2, 4),
            KernelScheduleAction::stride_order(vec![2, 1]),
        ];

        let candidate = replay_schedule_actions(&problem, &actions)
            .expect("valid GEMM action trace should replay into candidate metadata");
        let plan = schedule_gemm_plan(&candidate.schedule).expect("replayed GEMM should have plan");

        assert_eq!(candidate.family, "gemm-f32-bf16-row-col-row");
        assert_eq!(candidate.action_trace, actions);
        assert_eq!(candidate.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u4_bk");
        assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
        assert_eq!(plan.reduce_unroll, 4);
        assert_eq!(plan.b_load_order, GemmBTileLoadOrder::KContiguous);
        assert!(!candidate.is_launchable());

        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("replayed generated GEMM candidate should render source on demand");
        assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_u4_bk");
        assert!(generated.source.contains("const TILE_M: usize = 16;"));
        assert!(generated.source.contains("const TILE_N: usize = 32;"));
        assert!(generated.source.contains("const REDUCE_UNROLL: usize = 4;"));
        assert!(generated.source.contains("let tile_row = load % TILE_K;"));
    }

    #[test]
    fn matvec_generator_renders_rows_per_block_source_on_demand() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let candidate = problem.generated_candidate_for_row_split(
            MatvecRowSplit::new(13).expect("rows13 should be a legal generated split"),
        );
        let generated = MatvecRustCudaGenerator
            .source_for(&candidate)
            .expect("matvec generator should render rows-per-block source");

        assert_eq!(generated.symbol, "matvec_bf16_rows13");
        assert!(generated.source.contains("#[kernel]"));
        assert!(generated.source.contains("pub fn matvec_bf16_rows13("));
        assert!(generated.source.contains("pub struct Bf16(u16);"));
        assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 13;"));
        assert!(generated.source.contains("const REDUCE_UNROLL: u32 = 4;"));
        assert!(
            generated
                .source
                .contains("let row_in_block = thread_x / LANES_PER_ROW;")
        );
        assert!(generated.source.contains("while col + 96 < cols"));
        assert!(generated.source.contains("warp::shuffle_down_f32(acc, 16)"));
    }

    #[test]
    fn matvec_generator_renders_reduce_unroll_source_on_demand() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let candidate = problem.generated_candidate_for_plan(
            MatvecSchedulePlan::new(RowMajorWarpRows::Rows8).with_reduce_unroll(8),
        );
        let generated = MatvecRustCudaGenerator
            .source_for(&candidate)
            .expect("matvec generator should render reduce-unrolled source");

        assert_eq!(generated.symbol, "matvec_bf16_rows8_u8");
        assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
        assert!(generated.source.contains("const REDUCE_UNROLL: u32 = 8;"));
        assert!(generated.source.contains("while col + 224 < cols"));
        assert!(generated.source.contains("let col7 = col + 224;"));
        assert!(
            generated
                .source
                .contains("acc += weight[row_base + col7 * col_stride].to_f32() * input[col7];")
        );
        assert!(generated.source.contains("col += 256;"));
    }

    #[test]
    fn matvec_search_accepts_external_measured_scores_for_generated_candidate() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let result = beam_search_metadata_with_scorer(
            &problem,
            BeamSearchConfig {
                beam_width: 1,
                max_depth: 1,
                require_launchable: false,
            },
            |candidate| {
                let rows_per_block = schedule_rows_per_block(&candidate.schedule)?;
                let generated_bonus = if candidate.is_launchable() {
                    100.0
                } else {
                    0.0
                };
                SearchScore::measured(generated_bonus + (32.0 - f64::from(rows_per_block)))
            },
        );
        let best = result
            .best
            .expect("matvec search should keep externally best generated candidate");

        assert!(!best.is_launchable());
        assert_eq!(best.launch.kernel, "matvec_bf16_rows32");
        assert_eq!(schedule_rows_per_block(&best.schedule), Some(32));
        assert_eq!(
            best.score.map(|score| score.source),
            Some(SearchScoreSource::Measured)
        );
    }

    #[test]
    fn metadata_key_changes_when_schedule_changes() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let rows4 = problem.candidate_for_rows(RowMajorWarpRows::Rows4);
        let rows8 = problem.candidate_for_rows(RowMajorWarpRows::Rows8);
        assert_ne!(rows4.artifact_key(), rows8.artifact_key());
        assert_ne!(
            rows4.generated.artifact_key.hex(),
            rows8.generated.artifact_key.hex()
        );
    }

    #[test]
    fn gemm_describes_deferred_tiles_without_storing_kernel_payloads() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let seed = problem.seed();
        let candidates = problem.expand(&seed);
        assert_eq!(candidates.len(), 125);
        let deferred = candidates
            .iter()
            .find(|candidate| {
                schedule_gemm_tile(&candidate.schedule) == Some(GemmTileShape::new(13, 24, 13))
            })
            .expect("GEMM search should expose arbitrary deferred generated tile metadata");
        assert!(!deferred.is_launchable());
        assert_eq!(deferred.launch.kernel, "gemm_f32_bf16_tile_13x24x13");
        assert!(matches!(
            deferred.generated.materialization,
            KernelMaterialization::DeferredGenerated { .. }
        ));
        assert_eq!(deferred.generated.generator, "tiled-gemm-generator");
    }

    #[test]
    fn gemm_action_space_exposes_tile_unroll_and_stride_metadata() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let seed = problem.seed();
        let full_space = problem.search_space();
        assert_eq!(full_space.spaces.len(), 3);
        assert!(matches!(
            full_space.spaces[0],
            KernelActionSpace::TileGemm { .. }
        ));
        assert!(matches!(
            full_space.spaces[1],
            KernelActionSpace::Unroll { .. }
        ));
        assert!(matches!(
            full_space.spaces[2],
            KernelActionSpace::StrideOrder { .. }
        ));

        let tile_spaces = problem.action_spaces(&seed);
        let tile_actions = problem.schedule_actions(&seed);
        let tile_action = KernelScheduleAction::tile_gemm(
            13,
            24,
            13,
            KernelActionMaterialization::DeferredGenerated,
        );

        assert_eq!(tile_spaces.spaces.len(), 1);
        assert_eq!(tile_spaces.actions(), tile_actions);
        let KernelActionSpace::TileGemm { variants } = &tile_spaces.spaces[0] else {
            panic!("GEMM seed should expose tile action-space metadata");
        };
        assert_eq!(variants.len(), 125);
        assert!(variants.contains(&KernelTile3dAction::new(
            KernelTile3d::new(16, 16, 16),
            KernelActionMaterialization::Existing
        )));
        assert!(variants.contains(&KernelTile3dAction::new(
            KernelTile3d::new(13, 24, 13),
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(variants.contains(&KernelTile3dAction::new(
            KernelTile3d::new(32, 32, 32),
            KernelActionMaterialization::DeferredGenerated
        )));
        assert_eq!(tile_actions.len(), 125);
        assert!(tile_actions.contains(&KernelScheduleAction::tile_gemm(
            16,
            16,
            16,
            KernelActionMaterialization::Existing
        )));
        assert!(tile_actions.contains(&tile_action));

        let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
        let schedule_spaces = problem.action_spaces(&tile_candidate);
        let schedule_actions = problem.schedule_actions(&tile_candidate);
        assert_eq!(schedule_spaces.spaces.len(), 2);
        assert_eq!(schedule_spaces.actions(), schedule_actions);
        assert!(matches!(
            schedule_spaces.spaces[0],
            KernelActionSpace::Unroll { .. }
        ));
        assert!(matches!(
            schedule_spaces.spaces[1],
            KernelActionSpace::StrideOrder { .. }
        ));
        assert_eq!(schedule_actions.len(), 16);
        assert!(schedule_actions.contains(&KernelScheduleAction::unroll(2, 7)));
        assert!(schedule_actions.contains(&KernelScheduleAction::unroll(2, 16)));
        assert!(!schedule_actions.contains(&KernelScheduleAction::unroll(2, 1)));
        assert!(schedule_actions.contains(&KernelScheduleAction::stride_order(vec![2, 1])));

        let unrolled = problem
            .apply_schedule_action(&tile_candidate, &KernelScheduleAction::unroll(2, 7))
            .expect("unroll action should produce candidate metadata");
        let unrolled_plan =
            schedule_gemm_plan(&unrolled.schedule).expect("unrolled candidate should have plan");
        assert_eq!(unrolled_plan.reduce_unroll, 7);
        assert_eq!(unrolled.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u7");

        let traced_tile = problem
            .apply_schedule_action(&seed, &tile_action)
            .expect("tile action should produce candidate metadata");
        let traced_unrolled = problem
            .apply_schedule_action(&traced_tile, &KernelScheduleAction::unroll(2, 7))
            .expect("unroll action should extend candidate action trace");
        assert_eq!(
            traced_unrolled.action_trace,
            vec![tile_action, KernelScheduleAction::unroll(2, 7)]
        );
        let traced_unrolled_plan = schedule_gemm_plan(&traced_unrolled.schedule)
            .expect("traced unrolled candidate should have plan");
        assert_eq!(traced_unrolled_plan.tile, GemmTileShape::new(13, 24, 13));
        assert_eq!(traced_unrolled_plan.reduce_unroll, 7);
        assert_eq!(
            traced_unrolled.launch.kernel,
            "gemm_f32_bf16_tile_13x24x13_u7"
        );

        let reordered = problem
            .apply_schedule_action(
                &tile_candidate,
                &KernelScheduleAction::stride_order(vec![2, 1]),
            )
            .expect("stride-order action should produce candidate metadata");
        let reordered_plan =
            schedule_gemm_plan(&reordered.schedule).expect("reordered candidate should have plan");
        assert_eq!(reordered_plan.b_load_order, GemmBTileLoadOrder::KContiguous);
        assert_eq!(reordered.launch.kernel, "gemm_f32_bf16_tile_16x32x16_bk");
    }

    #[test]
    fn gemm_generator_renders_tile_specific_source_on_demand() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = problem.candidate_for_tile(GemmTileShape::new(13, 24, 13));
        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("GEMM generator should render deferred tile source");

        assert_eq!(generated.symbol, "gemm_f32_bf16_tile_13x24x13");
        assert!(generated.source.contains("#[kernel]"));
        assert!(
            generated
                .source
                .contains("pub fn gemm_f32_bf16_tile_13x24x13(")
        );
        assert!(generated.source.contains("pub struct Bf16(u16);"));
        assert!(generated.source.contains(".to_f32()"));
        assert!(generated.source.contains("const TILE_M: usize = 13;"));
        assert!(generated.source.contains("const TILE_N: usize = 24;"));
        assert!(generated.source.contains("const TILE_K: usize = 13;"));
        assert!(generated.source.contains("while load < TILE_A_ELEMS"));
        assert!(generated.source.contains("while load < TILE_B_ELEMS"));
    }

    #[test]
    fn gemm_search_expands_tile_metadata_into_reduce_unroll_variants() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
        let candidates = problem.expand(&tile_candidate);
        assert_eq!(candidates.len(), 16);

        let unroll7 = candidates
            .iter()
            .find(|candidate| schedule_gemm_reduce_unroll(&candidate.schedule) == Some(7))
            .expect("GEMM search should expose a reduce unroll factor 7 descriptor");
        assert_eq!(
            schedule_gemm_tile(&unroll7.schedule),
            Some(GemmTileShape::new(16, 32, 16))
        );
        assert_ne!(tile_candidate.artifact_key(), unroll7.artifact_key());
        assert_eq!(unroll7.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u7");
        assert!(!unroll7.is_launchable());
        assert!(matches!(
            unroll7.generated.materialization,
            KernelMaterialization::DeferredGenerated { .. }
        ));
    }

    #[test]
    fn gemm_search_expands_tile_metadata_into_b_load_stride_order() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
        let candidates = problem.expand(&tile_candidate);

        let k_contiguous = candidates
            .iter()
            .find(|candidate| {
                schedule_gemm_b_load_order(&candidate.schedule) == GemmBTileLoadOrder::KContiguous
            })
            .expect("GEMM search should expose K-contiguous B load order metadata");
        let plan =
            schedule_gemm_plan(&k_contiguous.schedule).expect("candidate should have GEMM plan");
        assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
        assert_eq!(plan.reduce_unroll, 1);
        assert_eq!(plan.b_load_order, GemmBTileLoadOrder::KContiguous);
        assert_ne!(tile_candidate.artifact_key(), k_contiguous.artifact_key());
        assert_eq!(k_contiguous.launch.kernel, "gemm_f32_bf16_tile_16x32x16_bk");
        assert!(k_contiguous.schedule.transforms.iter().any(|transform| {
            matches!(transform, ScheduleTransform::StrideOrder { axes } if axes == &[2, 1])
        }));
    }

    #[test]
    fn gemm_generator_renders_reduce_unroll_source_on_demand() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = problem.candidate_for_plan(
            GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16)).with_reduce_unroll(7),
        );
        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("GEMM generator should render reduce-unrolled source");

        assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_u7");
        assert!(generated.source.contains("const REDUCE_UNROLL: usize = 7;"));
        assert!(
            generated
                .source
                .contains("while kk + REDUCE_UNROLL <= TILE_K")
        );
        assert!(generated.source.contains("TILE_A[ty * TILE_K + kk + 6]"));
        assert!(generated.source.contains("TILE_B[(kk + 6) * TILE_N + tx]"));
        assert!(generated.source.contains("while kk < TILE_K"));
    }

    #[test]
    fn gemm_generator_renders_b_load_stride_order_source_on_demand() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = problem.candidate_for_plan(
            GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
                .with_b_load_order(GemmBTileLoadOrder::KContiguous),
        );
        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("GEMM generator should render stride-order source");

        assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_bk");
        assert!(generated.source.contains("let tile_row = load % TILE_K;"));
        assert!(generated.source.contains("let tile_col = load / TILE_K;"));
        assert!(
            generated
                .source
                .contains("let b_smem_index = tile_row * TILE_N + tile_col;")
        );
        assert!(generated.source.contains("TILE_B[b_smem_index] = if"));
    }

    #[test]
    fn artifact_store_writes_metadata_manifest_without_kernel_source() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));

        let emitted = store
            .emit_metadata(&candidate)
            .expect("artifact store should write generated metadata manifest");

        assert_eq!(emitted.artifact_key, candidate.artifact_key());
        assert!(emitted.paths.directory.starts_with(store.root()));
        assert!(emitted.paths.manifest_path.starts_with(store.root()));
        assert!(emitted.manifest_bytes > 0);
        assert!(
            !emitted.paths.directory.join("kernel.rs").exists(),
            "metadata emission should not persist generated kernel source"
        );

        let manifest_text = fs::read_to_string(&emitted.paths.manifest_path)
            .expect("generated manifest should be readable");
        let manifest: Value =
            serde_json::from_str(&manifest_text).expect("manifest should be valid JSON");
        let artifact_key = candidate.artifact_key().hex();
        assert_eq!(
            manifest["artifact_key"].as_str(),
            Some(artifact_key.as_str())
        );
        assert_eq!(
            manifest["family"].as_str(),
            Some("gemm-f32-bf16-row-col-row")
        );
        assert_eq!(manifest["generator"].as_str(), Some("tiled-gemm-generator"));
        assert_eq!(
            manifest["materialization"]["kind"].as_str(),
            Some("deferred-generated")
        );
        assert_eq!(
            manifest["materialization"]["symbol_hint"].as_str(),
            Some("gemm_f32_bf16_tile_16x32x16")
        );
        assert_eq!(manifest["schedule"][0]["op"].as_str(), Some("tile-gemm"));
        assert_eq!(manifest["schedule"][0]["n"].as_u64(), Some(32));
        assert_eq!(manifest["action_trace"].as_array().map(Vec::len), Some(0));

        remove_test_generated_root(&root);
    }

    #[test]
    fn artifact_store_records_action_trace_metadata_without_changing_kernel_key() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let seed = problem.seed();
        let tile_action = KernelScheduleAction::tile_gemm(
            16,
            32,
            16,
            KernelActionMaterialization::DeferredGenerated,
        );
        let tile_candidate = problem
            .apply_schedule_action(&seed, &tile_action)
            .expect("tile action should produce candidate metadata");
        let candidate = problem
            .apply_schedule_action(&tile_candidate, &KernelScheduleAction::unroll(2, 4))
            .expect("unroll action should produce candidate metadata");
        let direct_candidate = problem.candidate_for_plan(
            GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16)).with_reduce_unroll(4),
        );

        assert_eq!(candidate.artifact_key(), direct_candidate.artifact_key());
        assert_eq!(
            candidate.action_trace,
            vec![tile_action, KernelScheduleAction::unroll(2, 4)]
        );

        let emitted = store
            .emit_metadata(&candidate)
            .expect("artifact store should write action trace metadata manifest");
        let manifest_text = fs::read_to_string(&emitted.paths.manifest_path)
            .expect("generated manifest should be readable");
        let manifest: Value =
            serde_json::from_str(&manifest_text).expect("manifest should be valid JSON");

        assert_eq!(manifest["action_trace"].as_array().map(Vec::len), Some(2));
        assert_eq!(
            manifest["action_trace"][0]["op"].as_str(),
            Some("tile-gemm")
        );
        assert_eq!(
            manifest["action_trace"][0]["materialization"].as_str(),
            Some("deferred-generated")
        );
        assert_eq!(
            manifest["action_trace"][0]["arg"]["kind"].as_str(),
            Some("tile-3d")
        );
        assert_eq!(manifest["action_trace"][0]["arg"]["n"].as_u64(), Some(32));
        assert_eq!(manifest["action_trace"][1]["op"].as_str(), Some("unroll"));
        assert_eq!(manifest["action_trace"][1]["axis"].as_u64(), Some(2));
        assert_eq!(
            manifest["action_trace"][1]["arg"]["value"].as_u64(),
            Some(4)
        );

        remove_test_generated_root(&root);
    }

    #[test]
    fn artifact_store_records_measured_timing_metadata() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let seed = problem.seed();
        let mut candidate = problem
            .apply_schedule_action(
                &seed,
                &KernelScheduleAction::split(0, 8, KernelActionMaterialization::Existing),
            )
            .expect("row split action should produce candidate metadata");
        let samples =
            nn_rust_profiling::SampleStats::from_finite_samples(&[0.000002, 0.000003, 0.000004])
                .expect("sample stats should accept finite samples");
        let selected = nn_rust_profiling::ProfileDuration::from_seconds_f64(samples.median)
            .expect("median should be a valid duration");
        let timing = OptimizationTiming::new(
            nn_rust_profiling::ProfileTimeSource::CudaEvent,
            1,
            samples,
            selected,
        );
        candidate.score = SearchScore::measured_with_timing(samples.median, timing);

        let emitted = store
            .emit_metadata(&candidate)
            .expect("artifact store should write measured timing metadata manifest");
        let manifest_text = fs::read_to_string(&emitted.paths.manifest_path)
            .expect("generated manifest should be readable");
        let manifest: Value =
            serde_json::from_str(&manifest_text).expect("manifest should be valid JSON");

        assert_eq!(manifest["score"]["source"].as_str(), Some("measured"));
        assert_eq!(
            manifest["score"]["timing"]["source"].as_str(),
            Some("cuda-event")
        );
        assert_eq!(
            manifest["score"]["timing"]["warmup_count"].as_u64(),
            Some(1)
        );
        assert_eq!(
            manifest["score"]["timing"]["samples"]["count"].as_u64(),
            Some(3)
        );

        remove_test_generated_root(&root);
    }

    #[test]
    fn artifact_store_writes_selection_metadata_and_replays_action_trace() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let mut candidate = replay_schedule_actions(
            &problem,
            &[
                KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
                KernelScheduleAction::unroll(1, 8),
            ],
        )
        .expect("valid matvec action trace should replay into candidate metadata");
        let samples = SampleStats::from_finite_samples(&[0.000002, 0.000003, 0.000004])
            .expect("sample stats should accept finite samples");
        let timing = OptimizationTiming::new(
            ProfileTimeSource::CudaEvent,
            1,
            samples,
            ProfileDuration::from_seconds_f64(samples.median)
                .expect("median should be a valid duration"),
        );
        candidate.score = SearchScore::measured_with_timing(samples.median, timing);

        let emitted = store
            .emit_selection_for_candidate(&candidate)
            .expect("artifact store should write selected action metadata");

        assert!(emitted.selection_path.starts_with(store.root()));
        assert!(
            emitted
                .selection_path
                .components()
                .any(|component| component.as_os_str() == "selections")
        );
        assert!(emitted.selection_bytes > 0);

        let selection_text =
            fs::read_to_string(&emitted.selection_path).expect("selection should be readable");
        assert!(selection_text.contains("\"action_trace\""));
        assert!(selection_text.contains("\"op\": \"split\""));
        assert!(selection_text.contains("\"op\": \"unroll\""));
        assert!(selection_text.contains("\"source\": \"measured\""));
        assert!(!selection_text.contains("#[kernel]"));
        assert!(!selection_text.contains("pub fn matvec_bf16"));

        let selection = store
            .read_selection(&emitted.selection_path)
            .expect("selection should parse back from JSON");
        assert_eq!(selection.family, "matvec-bf16-row-major");
        assert_eq!(selection.artifact_key, candidate.artifact_key().hex());
        assert_eq!(selection.generator, "row-major-matvec-generator");
        assert!(!selection.launchable);
        assert_eq!(selection.action_trace, candidate.action_trace);
        assert_eq!(
            selection
                .score
                .and_then(|score| score.timing)
                .map(|timing| timing.samples.count),
            Some(3)
        );

        let replayed = selection
            .replay(&problem)
            .expect("selection should replay into selected candidate");
        assert_eq!(replayed.artifact_key(), candidate.artifact_key());
        assert_eq!(replayed.launch.kernel, "matvec_bf16_rows8_u8");

        remove_test_generated_root(&root);
    }

    #[test]
    fn selection_cache_key_tracks_problem_config_and_score_namespace() {
        let problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let same_problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let different_problem = MatvecSearchProblem::bf16_row_major(128, 512);
        let config = BeamSearchConfig {
            beam_width: 4,
            max_depth: 2,
            require_launchable: false,
        };

        let key = optimization_selection_cache_key(&problem, config, "heuristic");

        assert_eq!(
            key,
            optimization_selection_cache_key(&same_problem, config, "heuristic")
        );
        assert_ne!(
            key,
            optimization_selection_cache_key(&different_problem, config, "heuristic")
        );
        assert_ne!(
            key,
            optimization_selection_cache_key(
                &problem,
                BeamSearchConfig {
                    beam_width: 8,
                    ..config
                },
                "heuristic"
            )
        );
        assert_ne!(
            key,
            optimization_selection_cache_key(&problem, config, "measured-cuda-event-r5-w2")
        );
        assert_ne!(
            key,
            optimization_selection_cache_key(
                &MatvecWithoutUnrollSpace(problem),
                config,
                "heuristic"
            )
        );
        assert_eq!(key.family, "matvec-bf16-row-major");
        assert_eq!(key.hex().len(), 16);
    }

    #[test]
    fn auto_selection_cache_key_tracks_auto_config_and_action_space() {
        let problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let config = AutoOptimizeConfig {
            beam_width: 4,
            max_steps: 2,
            require_launchable: false,
            min_score_improvement: 0.0,
        };

        let key = auto_optimization_selection_cache_key(&problem, config, "heuristic");

        assert_eq!(
            key,
            auto_optimization_selection_cache_key(&problem, config, "heuristic")
        );
        assert_ne!(
            key,
            auto_optimization_selection_cache_key(
                &problem,
                AutoOptimizeConfig {
                    min_score_improvement: 0.5,
                    ..config
                },
                "heuristic"
            )
        );
        assert_ne!(
            key,
            auto_optimization_selection_cache_key(
                &problem,
                AutoOptimizeConfig {
                    max_steps: 3,
                    ..config
                },
                "heuristic"
            )
        );
        assert_ne!(
            key,
            auto_optimization_selection_cache_key(
                &MatvecWithoutUnrollSpace(problem),
                config,
                "heuristic"
            )
        );
        assert_eq!(key.family, "matvec-bf16-row-major");
        assert_eq!(key.hex().len(), 16);
    }

    #[test]
    fn cached_auto_optimize_replays_hit_without_expanding_beam() {
        let root = test_generated_root();
        fs::create_dir_all(&root).expect("test-generated root should be creatable");
        let store = KernelArtifactStore::new(&root);
        let problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let config = AutoOptimizeConfig {
            beam_width: 4,
            max_steps: 2,
            require_launchable: false,
            min_score_improvement: 0.0,
        };

        let first = auto_optimize_metadata_with_selection_cache(
            &store,
            &problem,
            config,
            "heuristic",
            |candidate| problem.score(candidate),
        )
        .expect("cache miss should run auto optimize");
        assert_eq!(first.cache_status, SelectionCacheStatus::Miss);
        assert!(first.cache_write.is_some());
        assert!(first.result.explored > 0);
        let first_best = first
            .result
            .best
            .as_ref()
            .expect("auto optimize should find a best candidate")
            .artifact_key();

        let second = auto_optimize_metadata_with_selection_cache(
            &store,
            &problem,
            config,
            "heuristic",
            |candidate| problem.score(candidate),
        )
        .expect("cache hit should replay selected auto-optimized candidate");
        let second_best = second
            .result
            .best
            .as_ref()
            .expect("cache hit should have selected candidate");

        assert_eq!(second.cache_status, SelectionCacheStatus::Hit);
        assert!(second.cache_write.is_none());
        assert_eq!(second.result.exit_reason, AutoOptimizeExitReason::CacheHit);
        assert_eq!(second.result.explored, 0);
        assert_eq!(second.result.rejected, 0);
        assert!(second.result.steps.is_empty());
        assert_eq!(second.result.beam.len(), 1);
        assert_eq!(second_best.artifact_key(), first_best);

        remove_test_generated_root(&root);
    }

    #[test]
    fn artifact_store_writes_and_reads_selection_cache() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let config = BeamSearchConfig {
            beam_width: 4,
            max_depth: 3,
            require_launchable: false,
        };
        let cache_key = optimization_selection_cache_key(&problem, config, "heuristic");
        let candidate = replay_schedule_actions(
            &problem,
            &[
                KernelScheduleAction::tile_gemm(
                    16,
                    32,
                    16,
                    KernelActionMaterialization::DeferredGenerated,
                ),
                KernelScheduleAction::unroll(2, 4),
                KernelScheduleAction::stride_order(vec![2, 1]),
            ],
        )
        .expect("valid GEMM action trace should replay into candidate metadata");

        assert!(
            store
                .read_selection_cache(&cache_key)
                .expect("missing selection cache should not be an error")
                .is_none()
        );

        let emitted = store
            .emit_selection_cache_for_candidate(&cache_key, &candidate)
            .expect("artifact store should write selection cache metadata");
        assert!(emitted.selection_path.starts_with(store.root()));
        assert!(
            emitted
                .selection_path
                .components()
                .any(|component| component.as_os_str() == "selection-cache")
        );
        assert_eq!(
            emitted
                .selection_path
                .file_stem()
                .and_then(|stem| stem.to_str()),
            Some(cache_key.hex().as_str())
        );

        let selection = store
            .read_selection_cache(&cache_key)
            .expect("selection cache should parse")
            .expect("selection cache should exist after write");
        let replayed = selection
            .replay(&problem)
            .expect("cached selection should replay into selected candidate");
        assert_eq!(selection.artifact_key, candidate.artifact_key().hex());
        assert_eq!(replayed.artifact_key(), candidate.artifact_key());
        assert_eq!(replayed.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u4_bk");

        let cache_text = fs::read_to_string(&emitted.selection_path)
            .expect("selection cache should be readable");
        assert!(cache_text.contains("\"action_trace\""));
        assert!(!cache_text.contains("#[kernel]"));
        assert!(!cache_text.contains("pub fn gemm_f32_bf16"));

        remove_test_generated_root(&root);
    }

    #[test]
    fn cached_beam_search_reports_miss_and_runs_search() {
        let root = test_generated_root();
        fs::create_dir_all(&root).expect("test-generated root should be creatable");
        let store = KernelArtifactStore::new(&root);
        let problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let config = BeamSearchConfig {
            beam_width: 4,
            max_depth: 2,
            require_launchable: false,
        };

        let cached = beam_search_metadata_with_selection_cache(
            &store,
            &problem,
            config,
            "heuristic",
            |candidate| problem.score(candidate),
        )
        .expect("cache miss should still run beam search");

        assert_eq!(cached.cache_status, SelectionCacheStatus::Miss);
        assert!(cached.result.explored > 0);
        assert!(cached.result.best.is_some());
        assert!(cached.cache_write.is_some());
        assert_eq!(
            cached.cache_key,
            optimization_selection_cache_key(&problem, config, "heuristic")
        );
        let cached_selection = store
            .read_selection_cache(&cached.cache_key)
            .expect("selection cache read should succeed")
            .expect("miss fallback should write selection cache metadata");
        let replayed = cached_selection
            .replay(&problem)
            .expect("written cache selection should replay");
        assert_eq!(
            replayed.artifact_key(),
            cached
                .result
                .best
                .as_ref()
                .expect("miss fallback should have a best candidate")
                .artifact_key()
        );

        remove_test_generated_root(&root);
    }

    #[test]
    fn cached_beam_search_replays_hit_without_expanding_beam() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let config = BeamSearchConfig {
            beam_width: 4,
            max_depth: 2,
            require_launchable: false,
        };
        let cache_key = optimization_selection_cache_key(&problem, config, "heuristic");
        let mut candidate = replay_schedule_actions(
            &problem,
            &[
                KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
                KernelScheduleAction::unroll(1, 8),
            ],
        )
        .expect("valid matvec action trace should replay into candidate metadata");
        candidate.score = SearchScore::heuristic(12.0);
        store
            .emit_selection_cache_for_candidate(&cache_key, &candidate)
            .expect("artifact store should write selection cache metadata");

        let cached = beam_search_metadata_with_selection_cache(
            &store,
            &problem,
            config,
            "heuristic",
            |candidate| problem.score(candidate),
        )
        .expect("cache hit should replay selected candidate");
        let best = cached
            .result
            .best
            .expect("cache hit should produce selected candidate");

        assert_eq!(cached.cache_status, SelectionCacheStatus::Hit);
        assert!(cached.cache_write.is_none());
        assert_eq!(cached.result.explored, 0);
        assert_eq!(cached.result.rejected, 0);
        assert_eq!(cached.result.beam.len(), 1);
        assert_eq!(best.artifact_key(), candidate.artifact_key());
        assert_eq!(best.score, SearchScore::heuristic(12.0));

        remove_test_generated_root(&root);
    }

    #[test]
    fn cached_beam_search_falls_back_on_stale_selection() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let config = BeamSearchConfig {
            beam_width: 4,
            max_depth: 2,
            require_launchable: false,
        };
        let cache_key = optimization_selection_cache_key(&problem, config, "heuristic");
        let candidate = replay_schedule_actions(
            &problem,
            &[KernelScheduleAction::split(
                0,
                8,
                KernelActionMaterialization::DeferredGenerated,
            )],
        )
        .expect("valid matvec action trace should replay into candidate metadata");
        let mut stale_selection = KernelOptimizationSelection::from_candidate(&candidate);
        stale_selection.artifact_key = "0000000000000000".to_string();
        store
            .emit_selection_cache(&cache_key, &stale_selection)
            .expect("artifact store should write stale selection cache metadata");

        let cached = beam_search_metadata_with_selection_cache(
            &store,
            &problem,
            config,
            "heuristic",
            |candidate| problem.score(candidate),
        )
        .expect("stale cache should fall back to beam search");

        assert!(matches!(
            cached.cache_status,
            SelectionCacheStatus::Stale { .. }
        ));
        assert!(cached.result.explored > 0);
        assert!(cached.result.best.is_some());
        assert!(cached.cache_write.is_some());
        let refreshed = store
            .read_selection_cache(&cached.cache_key)
            .expect("selection cache read should succeed after stale fallback")
            .expect("stale fallback should refresh selection cache metadata");
        let refreshed_candidate = refreshed
            .replay(&problem)
            .expect("refreshed cache selection should replay");
        assert_eq!(
            refreshed_candidate.artifact_key(),
            cached
                .result
                .best
                .as_ref()
                .expect("stale fallback should have a best candidate")
                .artifact_key()
        );

        remove_test_generated_root(&root);
    }

    #[test]
    fn artifact_store_writes_search_report_metadata_without_kernel_source() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let config = BeamSearchConfig {
            beam_width: 4,
            max_depth: 2,
            require_launchable: false,
        };
        let result = beam_search_metadata_with_scorer(&problem, config, |candidate| {
            let plan = schedule_matvec_plan(&candidate.schedule)?;
            let score = (8.0 - f64::from(plan.rows.rows_per_block())) * 10.0
                + (64.0 - f64::from(plan.reduce_unroll));
            SearchScore::measured(score)
        });
        let report = result.optimization_report("matvec-bf16-row-major", config);

        let emitted = store
            .emit_search_report(&report)
            .expect("artifact store should write compact search report");

        assert!(emitted.report_path.starts_with(store.root()));
        assert!(
            emitted
                .report_path
                .components()
                .any(|component| component.as_os_str() == "search-reports")
        );
        assert!(emitted.report_bytes > 0);
        assert!(
            !emitted
                .report_path
                .with_file_name("standalone-crate")
                .exists()
        );

        let report_text =
            fs::read_to_string(&emitted.report_path).expect("search report should be readable");
        let report_json: Value =
            serde_json::from_str(&report_text).expect("search report should be valid JSON");

        assert_eq!(
            report_json["family"].as_str(),
            Some("matvec-bf16-row-major")
        );
        assert_eq!(report_json["config"]["beam_width"].as_u64(), Some(4));
        assert_eq!(
            report_json["config"]["require_launchable"].as_bool(),
            Some(false)
        );
        assert_eq!(report_json["best"]["launchable"].as_bool(), Some(false));
        assert_eq!(
            report_json["best"]["launch"]["kernel"].as_str(),
            Some("matvec_bf16_rows32_u32")
        );
        assert_eq!(
            report_json["best"]["action_trace"].as_array().map(Vec::len),
            Some(2)
        );
        assert_eq!(
            report_json["best"]["action_trace"][1]["op"].as_str(),
            Some("unroll")
        );
        assert_eq!(
            report_json["best"]["score"]["source"].as_str(),
            Some("measured")
        );
        assert!(
            report_json["beam"]
                .as_array()
                .is_some_and(|beam| !beam.is_empty())
        );
        assert!(!report_text.contains("#[kernel]"));
        assert!(!report_text.contains("pub fn matvec_bf16"));
        assert!(!report_text.contains("pub struct Bf16"));

        remove_test_generated_root(&root);
    }

    #[test]
    fn artifact_store_writes_auto_search_report_with_step_metadata() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = MatvecSearchProblem::bf16_row_major(128, 256);
        let config = AutoOptimizeConfig {
            beam_width: 4,
            max_steps: 2,
            require_launchable: false,
            min_score_improvement: 0.0,
        };
        let result = auto_optimize_metadata_with_scorer(&problem, config, |candidate| {
            let plan = schedule_matvec_plan(&candidate.schedule)?;
            if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
                SearchScore::heuristic(1.0)
            } else {
                SearchScore::heuristic(10.0)
            }
        });
        let report = result.auto_optimization_report("matvec-bf16-row-major", config);

        let emitted = store
            .emit_auto_search_report(&report)
            .expect("artifact store should write compact auto-search report");

        assert!(emitted.report_path.starts_with(store.root()));
        assert!(
            emitted
                .report_path
                .components()
                .any(|component| component.as_os_str() == "auto-search-reports")
        );
        assert!(emitted.report_bytes > 0);

        let report_text = fs::read_to_string(&emitted.report_path)
            .expect("auto-search report should be readable");
        let report_json: Value =
            serde_json::from_str(&report_text).expect("auto-search report should be valid JSON");

        assert_eq!(
            report_json["family"].as_str(),
            Some("matvec-bf16-row-major")
        );
        assert_eq!(report_json["config"]["beam_width"].as_u64(), Some(4));
        assert_eq!(report_json["config"]["max_steps"].as_u64(), Some(2));
        assert_eq!(
            report_json["exit_reason"]["label"].as_str(),
            Some("no-improvement")
        );
        assert!(
            report_json["steps"]
                .as_array()
                .is_some_and(|steps| !steps.is_empty())
        );
        let steps = report_json["steps"]
            .as_array()
            .expect("steps should be serialized as an array");
        assert!(
            steps
                .iter()
                .any(|step| step["best_before"].is_object() && step["best_after"].is_object())
        );
        assert_eq!(
            report_json["best"]["action_trace"][0]["op"].as_str(),
            Some("split")
        );
        assert!(!report_text.contains("#[kernel]"));
        assert!(!report_text.contains("pub fn matvec_bf16"));
        assert!(!report_text.contains("pub struct Bf16"));

        remove_test_generated_root(&root);
    }

    #[test]
    fn artifact_store_records_b_load_stride_order_metadata() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = problem.candidate_for_plan(
            GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
                .with_b_load_order(GemmBTileLoadOrder::KContiguous),
        );

        let emitted = store
            .emit_metadata(&candidate)
            .expect("artifact store should write stride-order metadata manifest");
        let manifest_text = fs::read_to_string(&emitted.paths.manifest_path)
            .expect("generated manifest should be readable");
        let manifest: Value =
            serde_json::from_str(&manifest_text).expect("manifest should be valid JSON");

        assert_eq!(
            manifest["materialization"]["symbol_hint"].as_str(),
            Some("gemm_f32_bf16_tile_16x32x16_bk")
        );
        assert_eq!(manifest["schedule"][1]["op"].as_str(), Some("stride-order"));
        assert_eq!(manifest["schedule"][1]["axes"][0].as_u64(), Some(2));
        assert_eq!(manifest["schedule"][1]["axes"][1].as_u64(), Some(1));

        remove_test_generated_root(&root);
    }

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
    fn gemm_search_can_rank_deferred_generated_descriptors_when_allowed() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let result = beam_search_metadata(
            &problem,
            BeamSearchConfig {
                beam_width: 1,
                max_depth: 1,
                require_launchable: false,
            },
        );
        let best = result
            .best
            .expect("GEMM search should keep a generated descriptor when allowed");

        assert_eq!(
            schedule_gemm_tile(&best.schedule),
            Some(GemmTileShape::new(32, 32, 32))
        );
        assert!(!best.is_launchable());
        assert!(matches!(
            best.generated.materialization,
            KernelMaterialization::DeferredGenerated { .. }
        ));
    }

    #[test]
    fn gemm_search_accepts_external_measured_scores_across_unroll_depth() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let result = beam_search_metadata_with_scorer(
            &problem,
            BeamSearchConfig {
                beam_width: 125,
                max_depth: 2,
                require_launchable: false,
            },
            |candidate| {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                let target = GemmTileShape::new(13, 24, 13);
                let score = if plan.tile == target && plan.reduce_unroll == 7 {
                    0.0
                } else if plan.tile == target {
                    10.0
                } else {
                    1000.0
                        + f64::from(plan.tile.m + plan.tile.n + plan.tile.k)
                        + f64::from(plan.reduce_unroll)
                };
                SearchScore::measured(score)
            },
        );
        let best = result
            .best
            .expect("GEMM search should keep the externally best unroll descriptor");
        let plan = schedule_gemm_plan(&best.schedule).expect("best candidate should have a plan");

        assert_eq!(plan.tile, GemmTileShape::new(13, 24, 13));
        assert_eq!(plan.reduce_unroll, 7);
        assert_eq!(
            best.score.map(|score| score.source),
            Some(SearchScoreSource::Measured)
        );
    }

    #[test]
    fn gemm_search_accepts_external_measured_scores_across_stride_order_depth() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let result = beam_search_metadata_with_scorer(
            &problem,
            BeamSearchConfig {
                beam_width: 125,
                max_depth: 3,
                require_launchable: false,
            },
            |candidate| {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                let target = GemmTileShape::new(13, 24, 13);
                let score = if plan.tile == target
                    && plan.reduce_unroll == 7
                    && plan.b_load_order == GemmBTileLoadOrder::KContiguous
                {
                    0.0
                } else if plan.tile == target
                    && (plan.reduce_unroll == 7
                        || plan.b_load_order == GemmBTileLoadOrder::KContiguous)
                {
                    10.0
                } else if plan.tile == target {
                    20.0
                } else {
                    1000.0
                        + f64::from(plan.tile.m + plan.tile.n + plan.tile.k)
                        + f64::from(plan.reduce_unroll)
                };
                SearchScore::measured(score)
            },
        );
        let best = result
            .best
            .expect("GEMM search should keep the externally best stride-order descriptor");
        let plan = schedule_gemm_plan(&best.schedule).expect("best candidate should have a plan");

        assert_eq!(plan.tile, GemmTileShape::new(13, 24, 13));
        assert_eq!(plan.reduce_unroll, 7);
        assert_eq!(plan.b_load_order, GemmBTileLoadOrder::KContiguous);
        assert_eq!(best.launch.kernel, "gemm_f32_bf16_tile_13x24x13_u7_bk");
    }

    #[test]
    fn gemm_search_accepts_external_measured_scores() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let result = beam_search_metadata_with_scorer(
            &problem,
            BeamSearchConfig {
                beam_width: 1,
                max_depth: 1,
                require_launchable: false,
            },
            |candidate| {
                let tile = schedule_gemm_tile(&candidate.schedule)?;
                if tile == GemmTileShape::new(13, 24, 13) {
                    SearchScore::measured(0.0)
                } else {
                    SearchScore::measured((tile.m + tile.n + tile.k) as f64)
                }
            },
        );
        let best = result
            .best
            .expect("GEMM search should accept externally scored candidates");

        assert_eq!(
            schedule_gemm_tile(&best.schedule),
            Some(GemmTileShape::new(13, 24, 13))
        );
        assert_eq!(
            best.score.map(|score| score.source),
            Some(SearchScoreSource::Measured)
        );
    }

    #[test]
    fn gemm_default_search_keeps_only_currently_launchable_tile() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let result = beam_search_metadata(&problem, BeamSearchConfig::default());
        let best = result
            .best
            .expect("GEMM search should keep the existing tile");
        assert_eq!(result.explored, 141);
        assert_eq!(result.rejected, 140);
        assert_eq!(
            schedule_gemm_tile(&best.schedule),
            Some(GemmTileShape::new(16, 16, 16))
        );
        assert!(best.is_launchable());
    }
}
