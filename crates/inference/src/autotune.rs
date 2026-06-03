use std::{
    cmp::Ordering,
    collections::HashSet,
    env,
    fmt::{self, Write as _},
    fs, io,
    path::{Path, PathBuf},
};

pub use nn_rust_profiling::{
    AutoOptimizationExitReason as ProfilingAutoOptimizationExitReason,
    AutoOptimizationSearchConfig, AutoOptimizationSearchReport, AutoOptimizationSearchStep,
    OptimizationActionArg as KernelScheduleActionArg,
    OptimizationActionMaterialization as KernelActionMaterialization,
    OptimizationActionOp as KernelScheduleActionOp,
    OptimizationActionSpace as ProfilingActionSpace,
    OptimizationActionSpaceSet as ProfilingActionSpaceSet,
    OptimizationActionSpec as KernelScheduleAction,
    OptimizationAxisFactorChoice as ProfilingAxisFactorChoice, OptimizationCandidateSpec,
    OptimizationResourceUsage as KernelResourceUsage, OptimizationScore as SearchScore,
    OptimizationScoreSource as SearchScoreSource, OptimizationSearchConfig,
    OptimizationSearchReport, OptimizationTile3dChoice as ProfilingTile3dChoice,
    OptimizationTiming,
};
use nn_rust_profiling::{
    CudaLaunchSpec, MAX_OPTIMIZATION_SETUP_SEGMENTS, NumericKind, OperationKind, OperationRoute,
    OptimizationTimingSegment, ProfileDuration, ProfileTimeSource, SampleStats, TensorTypeSpec,
    TypedOperationSpec,
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
    Upcast { axis: u8, factor: u32 },
    Unroll { axis: u8, factor: u32 },
    LocalTile { axis: u8, factor: u32 },
    ThreadGroup { axis: u8, factor: u32 },
    TileGemm { m: u32, n: u32, k: u32 },
    StrideOrder { axes: Vec<u8> },
    Swap { axis_a: u8, axis_b: u8 },
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
    pub resources: Option<KernelResourceUsage>,
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
        .with_resource_usage(self.resources)
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
            resources: None,
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

    pub fn optimization_report_with_action_space(
        &self,
        family: impl Into<String>,
        config: BeamSearchConfig,
        action_space: &KernelActionSpaceSet,
    ) -> OptimizationSearchReport {
        self.optimization_report(family, config)
            .with_action_space(action_space.optimization_spec())
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

    pub fn optimization_report_with_action_space(
        &self,
        family: impl Into<String>,
        config: AutoOptimizeConfig,
        action_space: &KernelActionSpaceSet,
    ) -> OptimizationSearchReport {
        self.as_beam_search_result()
            .optimization_report_with_action_space(
                family,
                config.as_beam_search_config(),
                action_space,
            )
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

    pub fn auto_optimization_report_with_action_space(
        &self,
        family: impl Into<String>,
        config: AutoOptimizeConfig,
        action_space: &KernelActionSpaceSet,
    ) -> AutoOptimizationSearchReport {
        self.auto_optimization_report(family, config)
            .with_action_space(action_space.optimization_spec())
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

    pub fn optimization_spec(&self) -> ProfilingActionSpaceSet {
        ProfilingActionSpaceSet::new(
            self.spaces
                .iter()
                .map(KernelActionSpace::optimization_spec)
                .collect::<Vec<_>>(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelActionSpace {
    Split {
        variants: Vec<KernelAxisFactorAction>,
    },
    Upcast {
        axis: u8,
        factors: Vec<u32>,
    },
    Unroll {
        axis: u8,
        factors: Vec<u32>,
    },
    LocalTile {
        axis: u8,
        factors: Vec<u32>,
    },
    ThreadGroup {
        axis: u8,
        factors: Vec<u32>,
    },
    TileGemm {
        variants: Vec<KernelTile3dAction>,
    },
    StrideOrder {
        orders: Vec<Vec<u8>>,
    },
    Swap {
        pairs: Vec<(u8, u8)>,
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
            Self::Upcast { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::upcast(*axis, factor))
                .collect(),
            Self::Unroll { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::unroll(*axis, factor))
                .collect(),
            Self::LocalTile { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::local_tile(*axis, factor))
                .collect(),
            Self::ThreadGroup { axis, factors } => factors
                .iter()
                .copied()
                .map(|factor| KernelScheduleAction::thread_group(*axis, factor))
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
            Self::Swap { pairs } => pairs
                .iter()
                .copied()
                .map(|(axis_a, axis_b)| KernelScheduleAction::swap(axis_a, axis_b))
                .collect(),
        }
    }

    fn optimization_spec(&self) -> ProfilingActionSpace {
        match self {
            Self::Split { variants } => ProfilingActionSpace::Split {
                variants: variants
                    .iter()
                    .map(|variant| {
                        ProfilingAxisFactorChoice::new(
                            variant.axis,
                            variant.factor,
                            variant.materialization,
                        )
                    })
                    .collect(),
            },
            Self::Upcast { axis, factors } => ProfilingActionSpace::Upcast {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::Unroll { axis, factors } => ProfilingActionSpace::Unroll {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::LocalTile { axis, factors } => ProfilingActionSpace::LocalTile {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::ThreadGroup { axis, factors } => ProfilingActionSpace::ThreadGroup {
                axis: *axis,
                factors: factors.clone(),
            },
            Self::TileGemm { variants } => ProfilingActionSpace::TileGemm {
                variants: variants
                    .iter()
                    .map(|variant| {
                        ProfilingTile3dChoice::new(
                            variant.tile.m,
                            variant.tile.n,
                            variant.tile.k,
                            variant.materialization,
                        )
                    })
                    .collect(),
            },
            Self::StrideOrder { orders } => ProfilingActionSpace::StrideOrder {
                orders: orders.clone(),
            },
            Self::Swap { pairs } => ProfilingActionSpace::Swap {
                pairs: pairs.clone(),
            },
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
pub struct MatvecThreadGroup {
    lanes_per_row: u32,
}

impl MatvecThreadGroup {
    pub const DEFAULT_LANES_PER_ROW: u32 = 32;
    pub const SEARCH_LANES_PER_ROW: [u32; 4] = [2, 4, 8, 16];
    pub const SUPPORTED_LANES_PER_ROW: [u32; 5] = [2, 4, 8, 16, 32];

    pub fn new(lanes_per_row: u32) -> Option<Self> {
        Self::SUPPORTED_LANES_PER_ROW
            .contains(&lanes_per_row)
            .then_some(Self { lanes_per_row })
    }

    pub const fn default_group() -> Self {
        Self {
            lanes_per_row: Self::DEFAULT_LANES_PER_ROW,
        }
    }

    pub const fn lanes_per_row(self) -> u32 {
        self.lanes_per_row
    }

    pub const fn is_default(self) -> bool {
        self.lanes_per_row == Self::DEFAULT_LANES_PER_ROW
    }

    pub fn symbol_suffix(self) -> String {
        if self.is_default() {
            String::new()
        } else {
            format!("_tg{}", self.lanes_per_row)
        }
    }

    pub fn operation_suffix(self) -> String {
        if self.is_default() {
            String::new()
        } else {
            format!("-tg{}", self.lanes_per_row)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatvecRowUpcast {
    factor: u32,
}

impl MatvecRowUpcast {
    pub const DEFAULT_FACTOR: u32 = 1;
    pub const SEARCH_FACTORS: [u32; 3] = [2, 3, 4];

    pub fn new(factor: u32) -> Option<Self> {
        (factor >= Self::DEFAULT_FACTOR && factor <= 4).then_some(Self { factor })
    }

    pub const fn default_upcast() -> Self {
        Self {
            factor: Self::DEFAULT_FACTOR,
        }
    }

    pub const fn factor(self) -> u32 {
        self.factor
    }

    pub const fn is_default(self) -> bool {
        self.factor == Self::DEFAULT_FACTOR
    }

    pub fn symbol_suffix(self) -> String {
        if self.is_default() {
            String::new()
        } else {
            format!("_up{}", self.factor)
        }
    }

    pub fn operation_suffix(self) -> String {
        if self.is_default() {
            String::new()
        } else {
            format!("-up{}", self.factor)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatvecSchedulePlan {
    pub rows: MatvecRowSplit,
    pub row_upcast: MatvecRowUpcast,
    pub reduce_unroll: u32,
    pub thread_group: MatvecThreadGroup,
}

impl MatvecSchedulePlan {
    pub const DEFAULT_REDUCE_UNROLL: u32 = 4;

    pub fn new(rows: impl Into<MatvecRowSplit>) -> Self {
        Self {
            rows: rows.into(),
            row_upcast: MatvecRowUpcast::default_upcast(),
            reduce_unroll: Self::DEFAULT_REDUCE_UNROLL,
            thread_group: MatvecThreadGroup::default_group(),
        }
    }

    pub const fn with_row_upcast(mut self, row_upcast: MatvecRowUpcast) -> Self {
        self.row_upcast = row_upcast;
        self
    }

    pub const fn with_reduce_unroll(mut self, factor: u32) -> Self {
        self.reduce_unroll = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_thread_group(mut self, thread_group: MatvecThreadGroup) -> Self {
        self.thread_group = thread_group;
        self
    }

    pub const fn row_groups_per_block(self) -> u32 {
        self.rows
            .rows_per_block()
            .div_ceil(self.row_upcast.factor())
    }

    pub const fn block_threads(self) -> u32 {
        self.row_groups_per_block() * self.thread_group.lanes_per_row()
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
                factor: plan.block_threads(),
            });
        if !plan.row_upcast.is_default() {
            schedule = schedule.with_transform(ScheduleTransform::Upcast {
                axis: 0,
                factor: plan.row_upcast.factor(),
            });
        }
        if !plan.thread_group.is_default() {
            schedule = schedule.with_transform(ScheduleTransform::ThreadGroup {
                axis: 1,
                factor: plan.thread_group.lanes_per_row(),
            });
        }
        if plan.reduce_unroll != MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 1,
                factor: plan.reduce_unroll,
            });
        }
        let launch = CudaLaunchSpec::new(
            launch_kernel,
            (rows.grid_rows(self.rows), 1, 1),
            (plan.block_threads(), 1, 1),
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

    fn thread_group_factors(&self) -> Vec<u32> {
        MatvecThreadGroup::SEARCH_LANES_PER_ROW.to_vec()
    }

    fn row_upcast_factors(&self) -> Vec<u32> {
        MatvecRowUpcast::SEARCH_FACTORS.to_vec()
    }

    fn row_upcast_factors_for_rows(&self, rows: MatvecRowSplit) -> Vec<u32> {
        self.row_upcast_factors()
            .into_iter()
            .filter(|factor| *factor <= rows.rows_per_block())
            .collect()
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
        let row_upcast_factors = self.row_upcast_factors();
        let unroll_factors = self.reduce_unroll_factors();
        let thread_group_factors = self.thread_group_factors();
        KernelActionSpaceSet::new(vec![
            KernelActionSpace::Split {
                variants: split_variants,
            },
            KernelActionSpace::Upcast {
                axis: 0,
                factors: row_upcast_factors,
            },
            KernelActionSpace::Unroll {
                axis: 1,
                factors: unroll_factors,
            },
            KernelActionSpace::ThreadGroup {
                axis: 1,
                factors: thread_group_factors,
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
        if candidate.is_launchable() {
            return KernelActionSpaceSet::default();
        }
        let mut spaces = Vec::new();
        if plan.row_upcast.is_default() {
            let factors = self.row_upcast_factors_for_rows(plan.rows);
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Upcast { axis: 0, factors });
            }
        }
        if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
            spaces.push(KernelActionSpace::Unroll {
                axis: 1,
                factors: self.reduce_unroll_factors(),
            });
        }
        if plan.thread_group.is_default() {
            spaces.push(KernelActionSpace::ThreadGroup {
                axis: 1,
                factors: self.thread_group_factors(),
            });
        }
        KernelActionSpaceSet::new(spaces)
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
                op: KernelScheduleActionOp::Upcast,
                axis: Some(0),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.is_launchable() {
                    return None;
                }
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if !plan.row_upcast.is_default()
                    || !self.row_upcast_factors_for_rows(plan.rows).contains(factor)
                {
                    return None;
                }
                let row_upcast = MatvecRowUpcast::new(*factor)?;
                let next = self.generated_candidate_for_plan(plan.with_row_upcast(row_upcast));
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
            KernelScheduleAction {
                op: KernelScheduleActionOp::ThreadGroup,
                axis: Some(1),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                if candidate.is_launchable() || !self.thread_group_factors().contains(factor) {
                    return None;
                }
                let plan = schedule_matvec_plan(&candidate.schedule)?;
                if !plan.thread_group.is_default() {
                    return None;
                }
                let thread_group = MatvecThreadGroup::new(*factor)?;
                let next = self.generated_candidate_for_plan(plan.with_thread_group(thread_group));
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
        let lanes_per_row = plan.thread_group.lanes_per_row() as usize;
        let row_upcast = plan.row_upcast.factor() as usize;
        let blocks = self.rows.div_ceil(rows_per_block);
        let padded_rows = blocks * rows_per_block;
        let useful_fma_ops = self.rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let wasted_rows = padded_rows.saturating_sub(self.rows);
        let wasted_fma_ops = wasted_rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let block_overhead = blocks as f64 * 2048.0;
        let unroll = f64::from(plan.reduce_unroll.max(1));
        let loop_overhead = blocks as f64
            * row_upcast as f64
            * (self.cols as f64 / lanes_per_row as f64).ceil()
            * 64.0
            / unroll;
        let thread_overhead = blocks as f64 * f64::from(plan.block_threads()) * 8.0;
        let subgroup_pressure =
            blocks as f64 * (32.0 / lanes_per_row as f64 - 1.0).max(0.0) * 256.0;
        let row_upcast_pressure = blocks as f64 * (row_upcast as f64 - 1.0).max(0.0) * 384.0;
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
                + thread_overhead
                + subgroup_pressure
                + row_upcast_pressure
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

    pub fn with_axis(self, axis: u8, factor: u32) -> Option<Self> {
        match axis {
            0 => Some(Self {
                m: factor,
                n: self.n,
                k: self.k,
            }),
            1 => Some(Self {
                m: self.m,
                n: factor,
                k: self.k,
            }),
            2 => Some(Self {
                m: self.m,
                n: self.n,
                k: factor,
            }),
            _ => None,
        }
    }

    pub const fn axis_factor(self, axis: u8) -> Option<u32> {
        match axis {
            0 => Some(self.m),
            1 => Some(self.n),
            2 => Some(self.k),
            _ => None,
        }
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
pub enum GemmThreadOrder {
    NThenM,
    MThenN,
}

impl GemmThreadOrder {
    pub const fn symbol_suffix(self) -> &'static str {
        match self {
            Self::NThenM => "",
            Self::MThenN => "_sw01",
        }
    }

    pub const fn operation_suffix(self) -> &'static str {
        match self {
            Self::NThenM => "",
            Self::MThenN => "-sw01",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmSchedulePlan {
    pub tile: GemmTileShape,
    pub reduce_unroll: u32,
    pub m_per_thread: u32,
    pub n_per_thread: u32,
    pub a_load_unroll: u32,
    pub b_load_unroll: u32,
    pub a_load_thread_group: u32,
    pub b_load_thread_group: u32,
    pub a_load_order: GemmATileLoadOrder,
    pub b_load_order: GemmBTileLoadOrder,
    pub thread_order: GemmThreadOrder,
}

impl GemmSchedulePlan {
    pub const fn new(tile: GemmTileShape) -> Self {
        Self {
            tile,
            reduce_unroll: 1,
            m_per_thread: 1,
            n_per_thread: 1,
            a_load_unroll: 1,
            b_load_unroll: 1,
            a_load_thread_group: 0,
            b_load_thread_group: 0,
            a_load_order: GemmATileLoadOrder::KContiguous,
            b_load_order: GemmBTileLoadOrder::TileLinear,
            thread_order: GemmThreadOrder::NThenM,
        }
    }

    pub const fn with_reduce_unroll(mut self, factor: u32) -> Self {
        self.reduce_unroll = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_m_per_thread(mut self, factor: u32) -> Self {
        self.m_per_thread = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_n_per_thread(mut self, factor: u32) -> Self {
        self.n_per_thread = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_a_load_unroll(mut self, factor: u32) -> Self {
        self.a_load_unroll = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_b_load_unroll(mut self, factor: u32) -> Self {
        self.b_load_unroll = if factor == 0 { 1 } else { factor };
        self
    }

    pub const fn with_a_load_thread_group(mut self, factor: u32) -> Self {
        self.a_load_thread_group = factor;
        self
    }

    pub const fn with_b_load_thread_group(mut self, factor: u32) -> Self {
        self.b_load_thread_group = factor;
        self
    }

    pub const fn with_tile(mut self, tile: GemmTileShape) -> Self {
        self.tile = tile;
        self
    }

    pub const fn with_a_load_order(mut self, order: GemmATileLoadOrder) -> Self {
        self.a_load_order = order;
        self
    }

    pub const fn with_b_load_order(mut self, order: GemmBTileLoadOrder) -> Self {
        self.b_load_order = order;
        self
    }

    pub const fn with_thread_order(mut self, order: GemmThreadOrder) -> Self {
        self.thread_order = order;
        self
    }

    pub const fn normalized(self) -> Self {
        self.with_reduce_unroll(self.reduce_unroll)
            .with_m_per_thread(self.m_per_thread)
            .with_n_per_thread(self.n_per_thread)
            .with_a_load_unroll(self.a_load_unroll)
            .with_b_load_unroll(self.b_load_unroll)
    }

    pub fn block_dim(self) -> (u32, u32, u32) {
        let plan = self.normalized();
        let threads_n = plan.tile.n.div_ceil(plan.n_per_thread);
        let threads_m = plan.tile.m.div_ceil(plan.m_per_thread);
        match plan.thread_order {
            GemmThreadOrder::NThenM => (threads_n, threads_m, 1),
            GemmThreadOrder::MThenN => (threads_m, threads_n, 1),
        }
    }

    fn thread_count(self) -> u32 {
        let plan = self.normalized();
        let threads_n = plan.tile.n.div_ceil(plan.n_per_thread);
        let threads_m = plan.tile.m.div_ceil(plan.m_per_thread);
        threads_m.saturating_mul(threads_n).max(1)
    }

    fn shared_memory_bytes(self) -> u32 {
        const F32_BYTES: u32 = 4;
        let plan = self.normalized();
        plan.tile
            .m
            .saturating_mul(plan.tile.k)
            .saturating_add(plan.tile.k.saturating_mul(plan.tile.n))
            .saturating_mul(F32_BYTES)
    }

    fn accumulator_elements_per_thread(self) -> u32 {
        let plan = self.normalized();
        plan.m_per_thread.saturating_mul(plan.n_per_thread).max(1)
    }

    fn load_elements_per_block(self) -> u32 {
        let plan = self.normalized();
        plan.tile
            .m
            .saturating_mul(plan.tile.k)
            .saturating_add(plan.tile.k.saturating_mul(plan.tile.n))
    }

    fn resource_usage(self) -> KernelResourceUsage {
        let accumulators = self.accumulator_elements_per_thread();
        KernelResourceUsage::new(
            self.thread_count(),
            self.shared_memory_bytes(),
            accumulators,
            accumulators,
            self.load_elements_per_block(),
        )
    }

    fn a_load_rounds(self) -> u32 {
        let plan = self.normalized();
        plan.tile
            .m
            .saturating_mul(plan.tile.k)
            .div_ceil(plan.a_load_thread_count())
    }

    fn b_load_rounds(self) -> u32 {
        let plan = self.normalized();
        plan.tile
            .k
            .saturating_mul(plan.tile.n)
            .div_ceil(plan.b_load_thread_count())
    }

    fn a_load_thread_count(self) -> u32 {
        let thread_count = self.thread_count();
        if self.a_load_thread_group == 0 {
            thread_count
        } else {
            self.a_load_thread_group.clamp(1, thread_count)
        }
    }

    fn b_load_thread_count(self) -> u32 {
        let thread_count = self.thread_count();
        if self.b_load_thread_group == 0 {
            thread_count
        } else {
            self.b_load_thread_group.clamp(1, thread_count)
        }
    }

    fn has_custom_a_load_thread_group(self) -> bool {
        self.a_load_thread_group != 0 && self.a_load_thread_count() != self.thread_count()
    }

    fn has_custom_b_load_thread_group(self) -> bool {
        self.b_load_thread_group != 0 && self.b_load_thread_count() != self.thread_count()
    }

    fn per_thread_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.m_per_thread > 1 {
            write!(&mut suffix, "_mt{}", plan.m_per_thread).expect("write to string");
        }
        if plan.n_per_thread > 1 {
            write!(&mut suffix, "_nt{}", plan.n_per_thread).expect("write to string");
        }
        suffix
    }

    fn per_thread_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.m_per_thread > 1 {
            write!(&mut suffix, "-mt{}", plan.m_per_thread).expect("write to string");
        }
        if plan.n_per_thread > 1 {
            write!(&mut suffix, "-nt{}", plan.n_per_thread).expect("write to string");
        }
        suffix
    }

    fn load_unroll_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.a_load_unroll > 1 {
            write!(&mut suffix, "_au{}", plan.a_load_unroll).expect("write to string");
        }
        if plan.b_load_unroll > 1 {
            write!(&mut suffix, "_bu{}", plan.b_load_unroll).expect("write to string");
        }
        suffix
    }

    fn load_unroll_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.a_load_unroll > 1 {
            write!(&mut suffix, "-au{}", plan.a_load_unroll).expect("write to string");
        }
        if plan.b_load_unroll > 1 {
            write!(&mut suffix, "-bu{}", plan.b_load_unroll).expect("write to string");
        }
        suffix
    }

    fn load_thread_group_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.has_custom_a_load_thread_group() {
            write!(&mut suffix, "_atg{}", plan.a_load_thread_count()).expect("write to string");
        }
        if plan.has_custom_b_load_thread_group() {
            write!(&mut suffix, "_btg{}", plan.b_load_thread_count()).expect("write to string");
        }
        suffix
    }

    fn load_thread_group_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.has_custom_a_load_thread_group() {
            write!(&mut suffix, "-atg{}", plan.a_load_thread_count()).expect("write to string");
        }
        if plan.has_custom_b_load_thread_group() {
            write!(&mut suffix, "-btg{}", plan.b_load_thread_count()).expect("write to string");
        }
        suffix
    }

    fn thread_order_symbol_suffix(self) -> &'static str {
        self.normalized().thread_order.symbol_suffix()
    }

    fn thread_order_operation_suffix(self) -> &'static str {
        self.normalized().thread_order.operation_suffix()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmATileLoadOrder {
    KContiguous,
    MContiguous,
}

impl GemmATileLoadOrder {
    pub const fn symbol_suffix(self) -> &'static str {
        match self {
            Self::KContiguous => "",
            Self::MContiguous => "_am",
        }
    }

    pub fn action_axes(self) -> Vec<u8> {
        match self {
            Self::KContiguous => vec![2, 0],
            Self::MContiguous => vec![0, 2],
        }
    }

    pub fn from_action_axes(axes: &[u8]) -> Option<Self> {
        match axes {
            [2, 0] => Some(Self::KContiguous),
            [0, 2] => Some(Self::MContiguous),
            _ => None,
        }
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
    const MAX_THREADS_PER_BLOCK: u32 = 1024;
    const MAX_SHARED_MEMORY_BYTES: u32 = 48 * 1024;
    const MAX_ACCUMULATOR_ELEMENTS_PER_THREAD: u32 = 16;
    const MAX_REDUCE_UNROLL_FACTOR: u32 = 32;
    const MAX_LOAD_UNROLL_FACTOR: u32 = 4;
    const LOAD_THREAD_GROUP_FACTORS: [u32; 4] = [32, 64, 128, 256];

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
        let a_order_suffix = plan.a_load_order.symbol_suffix();
        let b_order_suffix = plan.b_load_order.symbol_suffix();
        let per_thread_symbol_suffix = plan.per_thread_symbol_suffix();
        let per_thread_operation_suffix = plan.per_thread_operation_suffix();
        let load_unroll_symbol_suffix = plan.load_unroll_symbol_suffix();
        let load_unroll_operation_suffix = plan.load_unroll_operation_suffix();
        let load_thread_group_symbol_suffix = plan.load_thread_group_symbol_suffix();
        let load_thread_group_operation_suffix = plan.load_thread_group_operation_suffix();
        let thread_order_symbol_suffix = plan.thread_order_symbol_suffix();
        let thread_order_operation_suffix = plan.thread_order_operation_suffix();
        let symbol_hint = if plan.reduce_unroll == 1 {
            format!(
                "gemm_f32_bf16_tile_{}x{}x{}{}{}{}{}{}{}",
                tile.m,
                tile.n,
                tile.k,
                per_thread_symbol_suffix,
                load_unroll_symbol_suffix,
                load_thread_group_symbol_suffix,
                a_order_suffix,
                b_order_suffix,
                thread_order_symbol_suffix
            )
        } else {
            format!(
                "gemm_f32_bf16_tile_{}x{}x{}_u{}{}{}{}{}{}{}",
                tile.m,
                tile.n,
                tile.k,
                plan.reduce_unroll,
                per_thread_symbol_suffix,
                load_unroll_symbol_suffix,
                load_thread_group_symbol_suffix,
                a_order_suffix,
                b_order_suffix,
                thread_order_symbol_suffix
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
            plan.block_dim(),
            0,
        );
        let operation = TypedOperationSpec::new(
            if plan.reduce_unroll == 1 {
                format!(
                    "gemm-f32-bf16-{}x{}x{}{}{}{}{}{}{}",
                    tile.m,
                    tile.n,
                    tile.k,
                    per_thread_operation_suffix,
                    load_unroll_operation_suffix,
                    load_thread_group_operation_suffix,
                    a_order_suffix,
                    b_order_suffix,
                    thread_order_operation_suffix
                )
            } else {
                format!(
                    "gemm-f32-bf16-{}x{}x{}-u{}{}{}{}{}{}{}",
                    tile.m,
                    tile.n,
                    tile.k,
                    plan.reduce_unroll,
                    per_thread_operation_suffix,
                    load_unroll_operation_suffix,
                    load_thread_group_operation_suffix,
                    a_order_suffix,
                    b_order_suffix,
                    thread_order_operation_suffix
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
        if plan.m_per_thread > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Upcast {
                axis: 0,
                factor: plan.m_per_thread,
            });
        }
        if plan.n_per_thread > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Upcast {
                axis: 1,
                factor: plan.n_per_thread,
            });
        }
        if plan.a_load_unroll > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 3,
                factor: plan.a_load_unroll,
            });
        }
        if plan.b_load_unroll > 1 {
            schedule = schedule.with_transform(ScheduleTransform::Unroll {
                axis: 4,
                factor: plan.b_load_unroll,
            });
        }
        if plan.has_custom_a_load_thread_group() {
            schedule = schedule.with_transform(ScheduleTransform::ThreadGroup {
                axis: 3,
                factor: plan.a_load_thread_count(),
            });
        }
        if plan.has_custom_b_load_thread_group() {
            schedule = schedule.with_transform(ScheduleTransform::ThreadGroup {
                axis: 4,
                factor: plan.b_load_thread_count(),
            });
        }
        if plan.a_load_order == GemmATileLoadOrder::MContiguous {
            schedule = schedule.with_transform(ScheduleTransform::StrideOrder { axes: vec![0, 2] });
        }
        if plan.b_load_order == GemmBTileLoadOrder::KContiguous {
            schedule = schedule.with_transform(ScheduleTransform::StrideOrder { axes: vec![2, 1] });
        }
        if plan.thread_order == GemmThreadOrder::MThenN {
            schedule = schedule.with_transform(ScheduleTransform::Swap {
                axis_a: 0,
                axis_b: 1,
            });
        }

        let mut candidate = KernelCandidateMetadata::new(
            "gemm-f32-bf16-row-col-row",
            self.axes(),
            schedule,
            "tiled-gemm-generator",
            materialization,
            launch,
            operation,
        );
        candidate.resources = Some(plan.resource_usage());
        candidate
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
            && plan.m_per_thread == 1
            && plan.n_per_thread == 1
            && plan.a_load_unroll == 1
            && plan.b_load_unroll == 1
            && !plan.has_custom_a_load_thread_group()
            && !plan.has_custom_b_load_thread_group()
            && plan.a_load_order == GemmATileLoadOrder::KContiguous
            && plan.b_load_order == GemmBTileLoadOrder::TileLinear
            && plan.thread_order == GemmThreadOrder::NThenM
    }

    fn action_materialization_for_plan(plan: GemmSchedulePlan) -> KernelActionMaterialization {
        if Self::is_existing_plan(plan) {
            KernelActionMaterialization::Existing
        } else {
            KernelActionMaterialization::DeferredGenerated
        }
    }

    fn plan_within_resource_limits(plan: GemmSchedulePlan) -> bool {
        let plan = plan.normalized();
        let resources = plan.resource_usage();
        plan.tile.is_launchable_shape()
            && resources.threads_per_block <= Self::MAX_THREADS_PER_BLOCK
            && resources.shared_memory_bytes <= Self::MAX_SHARED_MEMORY_BYTES
            && resources.accumulator_elements_per_thread
                <= Self::MAX_ACCUMULATOR_ELEMENTS_PER_THREAD
    }

    fn candidate_for_checked_plan(
        &self,
        parent: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
        plan: GemmSchedulePlan,
    ) -> Option<KernelCandidateMetadata> {
        let plan = plan.normalized();
        Self::plan_within_resource_limits(plan)
            .then(|| candidate_with_action_trace(parent, action, self.candidate_for_plan(plan)))
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

    fn m_per_thread_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            factors.extend(Self::m_per_thread_factors_for_tile(tile));
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn m_per_thread_factors_for_tile(tile: GemmTileShape) -> Vec<u32> {
        const MAX_M_PER_THREAD: u32 = 4;
        (2..=MAX_M_PER_THREAD)
            .filter(|factor| tile.m % *factor == 0)
            .collect()
    }

    fn n_per_thread_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            factors.extend(Self::n_per_thread_factors_for_tile(tile));
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn n_per_thread_factors_for_tile(tile: GemmTileShape) -> Vec<u32> {
        const MAX_N_PER_THREAD: u32 = 4;
        (2..=MAX_N_PER_THREAD)
            .filter(|factor| tile.n % *factor == 0)
            .collect()
    }

    fn per_thread_plans_for_tile(tile: GemmTileShape) -> Vec<GemmSchedulePlan> {
        let mut m_factors = vec![1];
        m_factors.extend(Self::m_per_thread_factors_for_tile(tile));
        let mut n_factors = vec![1];
        n_factors.extend(Self::n_per_thread_factors_for_tile(tile));

        let mut plans = Vec::new();
        for m_per_thread in m_factors {
            for n_per_thread in n_factors.iter().copied() {
                plans.push(
                    GemmSchedulePlan::new(tile)
                        .with_m_per_thread(m_per_thread)
                        .with_n_per_thread(n_per_thread),
                );
            }
        }
        plans
    }

    fn a_load_unroll_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            for plan in Self::per_thread_plans_for_tile(tile) {
                factors.extend(Self::a_load_unroll_factors_for_plan(plan));
            }
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn b_load_unroll_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            for plan in Self::per_thread_plans_for_tile(tile) {
                factors.extend(Self::b_load_unroll_factors_for_plan(plan));
            }
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn a_load_unroll_factors_for_plan(plan: GemmSchedulePlan) -> Vec<u32> {
        bounded_unroll_factors(
            plan.a_load_rounds() as usize,
            Self::MAX_LOAD_UNROLL_FACTOR,
            Some(1),
        )
    }

    fn b_load_unroll_factors_for_plan(plan: GemmSchedulePlan) -> Vec<u32> {
        bounded_unroll_factors(
            plan.b_load_rounds() as usize,
            Self::MAX_LOAD_UNROLL_FACTOR,
            Some(1),
        )
    }

    fn split_factors_for_axis(&self, axis: u8) -> Vec<u32> {
        match axis {
            0 => bounded_tile_factors(self.m, Self::MAX_TILE_DIM, None),
            1 => bounded_tile_factors(self.n, Self::MAX_TILE_DIM, None),
            2 => bounded_tile_factors(self.k, Self::MAX_TILE_DIM, None),
            _ => Vec::new(),
        }
    }

    fn split_action_variants(&self) -> Vec<KernelAxisFactorAction> {
        let mut variants = Vec::new();
        for axis in 0..=2 {
            variants.extend(self.split_factors_for_axis(axis).into_iter().map(|factor| {
                KernelAxisFactorAction::new(
                    axis,
                    factor,
                    KernelActionMaterialization::DeferredGenerated,
                )
            }));
        }
        variants
    }

    fn split_action_variants_for_plan(
        &self,
        plan: GemmSchedulePlan,
    ) -> Vec<KernelAxisFactorAction> {
        let mut variants = Vec::new();
        for axis in 0..=2 {
            let Some(current_factor) = plan.tile.axis_factor(axis) else {
                continue;
            };
            for factor in self.split_factors_for_axis(axis) {
                if factor == current_factor {
                    continue;
                }
                let Some(tile) = plan.tile.with_axis(axis, factor) else {
                    continue;
                };
                if !tile.is_launchable_shape() {
                    continue;
                }
                let next_plan = plan.with_tile(tile);
                if !Self::plan_within_resource_limits(next_plan) {
                    continue;
                }
                variants.push(KernelAxisFactorAction::new(
                    axis,
                    factor,
                    Self::action_materialization_for_plan(next_plan),
                ));
            }
        }
        variants
    }

    fn a_load_thread_group_factors(&self) -> Vec<u32> {
        let mut factors = Vec::new();
        for tile in self.tile_shapes() {
            for plan in Self::per_thread_plans_for_tile(tile) {
                factors.extend(Self::load_thread_group_factors_for_plan(plan));
            }
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    fn b_load_thread_group_factors(&self) -> Vec<u32> {
        self.a_load_thread_group_factors()
    }

    fn load_thread_group_factors_for_plan(plan: GemmSchedulePlan) -> Vec<u32> {
        let thread_count = plan.thread_count();
        Self::LOAD_THREAD_GROUP_FACTORS
            .into_iter()
            .filter(|factor| *factor < thread_count)
            .collect()
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
                    if Self::plan_within_resource_limits(GemmSchedulePlan::new(tile)) {
                        tiles.push(tile);
                    }
                }
            }
        }
        tiles.sort_unstable_by_key(|tile| (tile.m, tile.n, tile.k));
        tiles.dedup();
        tiles
    }

    fn seed_tile_action_variants() -> Vec<KernelTile3dAction> {
        vec![KernelTile3dAction::new(
            Self::EXISTING_TILE.into(),
            KernelActionMaterialization::Existing,
        )]
    }

    fn stride_orders_for_plan(plan: GemmSchedulePlan) -> Vec<Vec<u8>> {
        let mut orders = Vec::new();
        if plan.a_load_order == GemmATileLoadOrder::KContiguous {
            orders.push(GemmATileLoadOrder::MContiguous.action_axes());
        }
        if plan.b_load_order == GemmBTileLoadOrder::TileLinear {
            orders.push(GemmBTileLoadOrder::KContiguous.action_axes());
        }
        orders
    }
}

impl KernelActionSearchProblem for GemmSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet {
        let split_variants = self.split_action_variants();
        let unroll_factors = self.reduce_unroll_factors();
        let m_per_thread_factors = self.m_per_thread_factors();
        let n_per_thread_factors = self.n_per_thread_factors();
        let a_load_unroll_factors = self.a_load_unroll_factors();
        let b_load_unroll_factors = self.b_load_unroll_factors();
        let a_load_thread_group_factors = self.a_load_thread_group_factors();
        let b_load_thread_group_factors = self.b_load_thread_group_factors();
        let stride_orders =
            Self::stride_orders_for_plan(GemmSchedulePlan::new(Self::EXISTING_TILE));
        let mut spaces = vec![
            KernelActionSpace::Split {
                variants: split_variants,
            },
            KernelActionSpace::TileGemm {
                variants: Self::seed_tile_action_variants(),
            },
            KernelActionSpace::Unroll {
                axis: 2,
                factors: unroll_factors,
            },
            KernelActionSpace::Upcast {
                axis: 0,
                factors: m_per_thread_factors,
            },
            KernelActionSpace::Upcast {
                axis: 1,
                factors: n_per_thread_factors,
            },
        ];
        if !a_load_unroll_factors.is_empty() {
            spaces.push(KernelActionSpace::Unroll {
                axis: 3,
                factors: a_load_unroll_factors,
            });
        }
        if !b_load_unroll_factors.is_empty() {
            spaces.push(KernelActionSpace::Unroll {
                axis: 4,
                factors: b_load_unroll_factors,
            });
        }
        if !a_load_thread_group_factors.is_empty() {
            spaces.push(KernelActionSpace::ThreadGroup {
                axis: 3,
                factors: a_load_thread_group_factors,
            });
        }
        if !b_load_thread_group_factors.is_empty() {
            spaces.push(KernelActionSpace::ThreadGroup {
                axis: 4,
                factors: b_load_thread_group_factors,
            });
        }
        spaces.push(KernelActionSpace::Swap {
            pairs: vec![(0, 1)],
        });
        spaces.push(KernelActionSpace::StrideOrder {
            orders: stride_orders,
        });
        KernelActionSpaceSet::new(spaces)
    }

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
        let Some(plan) = schedule_gemm_plan(&candidate.schedule) else {
            if !candidate.schedule.transforms.is_empty() {
                return KernelActionSpaceSet::default();
            }
            let split_variants =
                self.split_action_variants_for_plan(GemmSchedulePlan::new(Self::EXISTING_TILE));
            return KernelActionSpaceSet::new(vec![
                KernelActionSpace::Split {
                    variants: split_variants,
                },
                KernelActionSpace::TileGemm {
                    variants: Self::seed_tile_action_variants(),
                },
            ]);
        };

        let mut spaces = Vec::new();
        let split_variants = self.split_action_variants_for_plan(plan);
        if !split_variants.is_empty() {
            spaces.push(KernelActionSpace::Split {
                variants: split_variants,
            });
        }
        if plan.reduce_unroll == 1 {
            let factors = Self::reduce_unroll_factors_for_tile(plan.tile)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_reduce_unroll(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Unroll { axis: 2, factors });
            }
        }
        if plan.m_per_thread == 1 {
            let factors = Self::m_per_thread_factors_for_tile(plan.tile)
                .into_iter()
                .filter(|factor| Self::plan_within_resource_limits(plan.with_m_per_thread(*factor)))
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Upcast { axis: 0, factors });
            }
        }
        if plan.n_per_thread == 1 {
            let factors = Self::n_per_thread_factors_for_tile(plan.tile)
                .into_iter()
                .filter(|factor| Self::plan_within_resource_limits(plan.with_n_per_thread(*factor)))
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Upcast { axis: 1, factors });
            }
        }
        if plan.a_load_unroll == 1 {
            let factors = Self::a_load_unroll_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_a_load_unroll(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Unroll { axis: 3, factors });
            }
        }
        if plan.b_load_unroll == 1 {
            let factors = Self::b_load_unroll_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_b_load_unroll(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::Unroll { axis: 4, factors });
            }
        }
        if !plan.has_custom_a_load_thread_group() {
            let factors = Self::load_thread_group_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_a_load_thread_group(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::ThreadGroup { axis: 3, factors });
            }
        }
        if !plan.has_custom_b_load_thread_group() {
            let factors = Self::load_thread_group_factors_for_plan(plan)
                .into_iter()
                .filter(|factor| {
                    Self::plan_within_resource_limits(plan.with_b_load_thread_group(*factor))
                })
                .collect::<Vec<_>>();
            if !factors.is_empty() {
                spaces.push(KernelActionSpace::ThreadGroup { axis: 4, factors });
            }
        }
        if plan.thread_order == GemmThreadOrder::NThenM
            && Self::plan_within_resource_limits(plan.with_thread_order(GemmThreadOrder::MThenN))
        {
            spaces.push(KernelActionSpace::Swap {
                pairs: vec![(0, 1)],
            });
        }
        let orders = Self::stride_orders_for_plan(plan)
            .into_iter()
            .filter(|axes| {
                GemmATileLoadOrder::from_action_axes(axes)
                    .map(|order| Self::plan_within_resource_limits(plan.with_a_load_order(order)))
                    .or_else(|| {
                        GemmBTileLoadOrder::from_action_axes(axes).map(|order| {
                            Self::plan_within_resource_limits(plan.with_b_load_order(order))
                        })
                    })
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        if !orders.is_empty() {
            spaces.push(KernelActionSpace::StrideOrder { orders });
        }
        KernelActionSpaceSet::new(spaces)
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
                self.candidate_for_checked_plan(candidate, action, plan)
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Split,
                axis: Some(axis @ 0..=2),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization,
            } => {
                let plan = match schedule_gemm_plan(&candidate.schedule) {
                    Some(plan) => plan,
                    None if candidate.schedule.transforms.is_empty() => {
                        GemmSchedulePlan::new(Self::EXISTING_TILE)
                    }
                    None => return None,
                };
                let variants = self.split_action_variants_for_plan(plan);
                if !variants.contains(&KernelAxisFactorAction::new(
                    *axis,
                    *factor,
                    *materialization,
                )) {
                    return None;
                }
                let next_tile = plan.tile.with_axis(*axis, *factor)?;
                self.candidate_for_checked_plan(candidate, action, plan.with_tile(next_tile))
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
                self.candidate_for_checked_plan(candidate, action, plan.with_reduce_unroll(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(3),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.a_load_unroll != 1
                    || !Self::a_load_unroll_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_a_load_unroll(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Unroll,
                axis: Some(4),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.b_load_unroll != 1
                    || !Self::b_load_unroll_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_b_load_unroll(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Upcast,
                axis: Some(0),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.m_per_thread != 1
                    || !Self::m_per_thread_factors_for_tile(plan.tile).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_m_per_thread(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Upcast,
                axis: Some(1),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.n_per_thread != 1
                    || !Self::n_per_thread_factors_for_tile(plan.tile).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_n_per_thread(*factor))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::ThreadGroup,
                axis: Some(3),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.has_custom_a_load_thread_group()
                    || !Self::load_thread_group_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(
                    candidate,
                    action,
                    plan.with_a_load_thread_group(*factor),
                )
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::ThreadGroup,
                axis: Some(4),
                arg: KernelScheduleActionArg::Factor(factor),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if plan.has_custom_b_load_thread_group()
                    || !Self::load_thread_group_factors_for_plan(plan).contains(factor)
                {
                    return None;
                }
                self.candidate_for_checked_plan(
                    candidate,
                    action,
                    plan.with_b_load_thread_group(*factor),
                )
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::StrideOrder,
                axis: None,
                arg: KernelScheduleActionArg::AxisOrder(axes),
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if let Some(order) = GemmATileLoadOrder::from_action_axes(axes) {
                    if plan.a_load_order != GemmATileLoadOrder::KContiguous
                        || order == GemmATileLoadOrder::KContiguous
                    {
                        return None;
                    }
                    return self.candidate_for_checked_plan(
                        candidate,
                        action,
                        plan.with_a_load_order(order),
                    );
                }
                let order = GemmBTileLoadOrder::from_action_axes(axes)?;
                if plan.b_load_order != GemmBTileLoadOrder::TileLinear
                    || order == GemmBTileLoadOrder::TileLinear
                {
                    return None;
                }
                self.candidate_for_checked_plan(candidate, action, plan.with_b_load_order(order))
            }
            KernelScheduleAction {
                op: KernelScheduleActionOp::Swap,
                axis: None,
                arg: KernelScheduleActionArg::AxisPair { axis_a, axis_b },
                materialization: KernelActionMaterialization::DeferredGenerated,
            } => {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                if (*axis_a, *axis_b) != (0, 1) || plan.thread_order != GemmThreadOrder::NThenM {
                    return None;
                }
                self.candidate_for_checked_plan(
                    candidate,
                    action,
                    plan.with_thread_order(GemmThreadOrder::MThenN),
                )
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
        let m_per_thread = plan.m_per_thread.max(1);
        let n_per_thread = plan.n_per_thread.max(1);
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
        let per_thread_work = f64::from(m_per_thread * n_per_thread);
        let thread_count_u32 = plan.thread_count();
        let thread_count = f64::from(thread_count_u32);
        let a_load_unroll = plan.a_load_unroll.max(1);
        let b_load_unroll = plan.b_load_unroll.max(1);
        let a_load_threads = plan.a_load_thread_count();
        let b_load_threads = plan.b_load_thread_count();
        let a_load_loop_rounds = plan.a_load_rounds().div_ceil(a_load_unroll);
        let b_load_loop_rounds = plan.b_load_rounds().div_ceil(b_load_unroll);
        let loop_overhead = block_count * 4096.0 / unroll / per_thread_work.sqrt();
        let thread_overhead = block_count * thread_count * 16.0;
        let load_loop_overhead =
            block_count * f64::from(a_load_loop_rounds + b_load_loop_rounds) * 256.0;
        let average_load_threads = (a_load_threads + b_load_threads) / 2;
        let load_thread_penalty =
            block_count * f64::from(thread_count_u32.saturating_sub(average_load_threads)) * 8.0;
        let register_pressure =
            block_count * ((unroll - 1.0) * 256.0 + (per_thread_work - 1.0) * 1024.0);
        let load_unroll_pressure =
            block_count * f64::from(a_load_unroll + b_load_unroll - 2) * 128.0;
        let threads_m = tile.m.div_ceil(m_per_thread);
        let threads_n = tile.n.div_ceil(n_per_thread);
        let thread_order_penalty = match plan.thread_order {
            GemmThreadOrder::NThenM if threads_n < 16 && threads_m >= 16 => block_count * 128.0,
            GemmThreadOrder::MThenN if threads_n >= 16 => block_count * 96.0,
            _ => 0.0,
        };
        let a_load_penalty = match plan.a_load_order {
            GemmATileLoadOrder::KContiguous => block_count * 64.0,
            GemmATileLoadOrder::MContiguous => block_count * 512.0,
        };
        let b_load_penalty = match plan.b_load_order {
            GemmBTileLoadOrder::TileLinear => block_count * 512.0,
            GemmBTileLoadOrder::KContiguous => block_count * 64.0,
        };
        SearchScore::heuristic(
            padded_fma_ops
                + loop_overhead
                + thread_overhead
                + load_loop_overhead
                + load_thread_penalty
                + register_pressure
                + load_unroll_pressure
                + thread_order_penalty
                + a_load_penalty
                + b_load_penalty,
        )
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

fn schedule_matvec_row_upcast(schedule: &KernelSchedule) -> Option<MatvecRowUpcast> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Upcast { axis: 0, factor } => MatvecRowUpcast::new(*factor),
            _ => None,
        })
        .or_else(|| Some(MatvecRowUpcast::default_upcast()))
}

fn schedule_matvec_thread_group(schedule: &KernelSchedule) -> Option<MatvecThreadGroup> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::ThreadGroup { axis: 1, factor } => MatvecThreadGroup::new(*factor),
            _ => None,
        })
        .or_else(|| Some(MatvecThreadGroup::default_group()))
}

fn schedule_matvec_plan(schedule: &KernelSchedule) -> Option<MatvecSchedulePlan> {
    let rows_per_block = schedule_rows_per_block(schedule)?;
    let rows = MatvecRowSplit::new(rows_per_block)?;
    Some(MatvecSchedulePlan {
        rows,
        row_upcast: schedule_matvec_row_upcast(schedule)?,
        reduce_unroll: schedule_matvec_reduce_unroll(schedule)
            .unwrap_or(MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL),
        thread_group: schedule_matvec_thread_group(schedule)?,
    })
}

fn matvec_symbol_hint(plan: MatvecSchedulePlan) -> String {
    let plan = plan.normalized();
    let mut base = format!("matvec_bf16_rows{}", plan.rows.rows_per_block());
    base.push_str(&plan.row_upcast.symbol_suffix());
    if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
        base.push_str(&plan.thread_group.symbol_suffix());
        base
    } else {
        write!(
            &mut base,
            "_u{}{}",
            plan.reduce_unroll,
            plan.thread_group.symbol_suffix()
        )
        .expect("write to string");
        base
    }
}

fn matvec_operation_name(plan: MatvecSchedulePlan) -> String {
    let plan = plan.normalized();
    let plan_name = plan.rows.plan_name();
    let row_upcast_suffix = plan.row_upcast.operation_suffix();
    if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL {
        format!(
            "{plan_name}::bf16{}{}",
            row_upcast_suffix,
            plan.thread_group.operation_suffix()
        )
    } else {
        format!(
            "{plan_name}::bf16{}-u{}{}",
            row_upcast_suffix,
            plan.reduce_unroll,
            plan.thread_group.operation_suffix()
        )
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

fn schedule_gemm_m_per_thread(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Upcast { axis: 0, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_gemm_n_per_thread(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Upcast { axis: 1, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_gemm_a_load_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 3, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_gemm_b_load_unroll(schedule: &KernelSchedule) -> Option<u32> {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Unroll { axis: 4, factor } => Some(*factor),
            _ => None,
        })
}

fn schedule_gemm_a_load_thread_group(schedule: &KernelSchedule) -> u32 {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::ThreadGroup { axis: 3, factor } => Some(*factor),
            _ => None,
        })
        .unwrap_or(0)
}

fn schedule_gemm_b_load_thread_group(schedule: &KernelSchedule) -> u32 {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::ThreadGroup { axis: 4, factor } => Some(*factor),
            _ => None,
        })
        .unwrap_or(0)
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

fn schedule_gemm_a_load_order(schedule: &KernelSchedule) -> GemmATileLoadOrder {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::StrideOrder { axes } if axes.as_slice() == [0, 2] => {
                Some(GemmATileLoadOrder::MContiguous)
            }
            _ => None,
        })
        .unwrap_or(GemmATileLoadOrder::KContiguous)
}

fn schedule_gemm_thread_order(schedule: &KernelSchedule) -> GemmThreadOrder {
    schedule
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ScheduleTransform::Swap {
                axis_a: 0,
                axis_b: 1,
            } => Some(GemmThreadOrder::MThenN),
            _ => None,
        })
        .unwrap_or(GemmThreadOrder::NThenM)
}

fn schedule_gemm_plan(schedule: &KernelSchedule) -> Option<GemmSchedulePlan> {
    let tile = schedule_gemm_tile(schedule)?;
    Some(GemmSchedulePlan {
        tile,
        reduce_unroll: schedule_gemm_reduce_unroll(schedule).unwrap_or(1),
        m_per_thread: schedule_gemm_m_per_thread(schedule).unwrap_or(1),
        n_per_thread: schedule_gemm_n_per_thread(schedule).unwrap_or(1),
        a_load_unroll: schedule_gemm_a_load_unroll(schedule).unwrap_or(1),
        b_load_unroll: schedule_gemm_b_load_unroll(schedule).unwrap_or(1),
        a_load_thread_group: schedule_gemm_a_load_thread_group(schedule),
        b_load_thread_group: schedule_gemm_b_load_thread_group(schedule),
        a_load_order: schedule_gemm_a_load_order(schedule),
        b_load_order: schedule_gemm_b_load_order(schedule),
        thread_order: schedule_gemm_thread_order(schedule),
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
        "resources": candidate.resources.map(resource_usage_json),
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
        "upcast" => KernelScheduleActionOp::Upcast,
        "unroll" => KernelScheduleActionOp::Unroll,
        "local-tile" => KernelScheduleActionOp::LocalTile,
        "thread-group" => KernelScheduleActionOp::ThreadGroup,
        "tile-gemm" => KernelScheduleActionOp::TileGemm,
        "stride-order" => KernelScheduleActionOp::StrideOrder,
        "swap" => KernelScheduleActionOp::Swap,
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
        "axis-pair" => Ok(KernelScheduleActionArg::AxisPair {
            axis_a: required_u8(value, "axis_a")?,
            axis_b: required_u8(value, "axis_b")?,
        }),
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
    let source = parse_profile_time_source(required_str(value, "source")?, "score.timing.source")?;
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
    let setup_segments =
        parse_timing_segments(value.get("setup_segments").unwrap_or(&Value::Null))?;
    Ok(Some(
        OptimizationTiming::new(
            source,
            required_usize(value, "warmup_count")?,
            sample_stats,
            selected,
        )
        .with_setup_segments(&setup_segments),
    ))
}

fn parse_timing_segments(
    value: &Value,
) -> Result<Vec<OptimizationTimingSegment>, KernelGenerationError> {
    if value.is_null() {
        return Ok(Vec::new());
    }
    let values = value
        .as_array()
        .ok_or_else(|| invalid_selection("score.timing.setup_segments must be an array"))?;
    if values.len() > MAX_OPTIMIZATION_SETUP_SEGMENTS {
        return Err(invalid_selection(format!(
            "score.timing.setup_segments has {} entries, maximum is {MAX_OPTIMIZATION_SETUP_SEGMENTS}",
            values.len()
        )));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            let duration_seconds = required_f64(segment, "duration_seconds")?;
            let duration = ProfileDuration::from_seconds_f64(duration_seconds).ok_or_else(|| {
                invalid_selection(format!(
                    "score.timing.setup_segments[{index}].duration_seconds must be nonnegative and finite"
                ))
            })?;
            Ok(OptimizationTimingSegment::new(
                parse_timing_segment_name(required_str(segment, "name")?, index)?,
                parse_profile_time_source(
                    required_str(segment, "source")?,
                    "score.timing.setup_segments[].source",
                )?,
                duration,
            ))
        })
        .collect()
}

fn parse_profile_time_source(
    source: &str,
    field_name: &str,
) -> Result<ProfileTimeSource, KernelGenerationError> {
    match source {
        "wall-clock" => Ok(ProfileTimeSource::WallClock),
        "cuda-event" => Ok(ProfileTimeSource::CudaEvent),
        "host-self-time" => Ok(ProfileTimeSource::SelfTimeAccounting),
        source => Err(invalid_selection(format!(
            "{field_name} is unsupported: {source:?}"
        ))),
    }
}

fn parse_timing_segment_name(
    name: &str,
    index: usize,
) -> Result<&'static str, KernelGenerationError> {
    match name {
        "emit-standalone-crate" => Ok("emit-standalone-crate"),
        "compile-standalone-crate" => Ok("compile-standalone-crate"),
        "load-generated-module" => Ok("load-generated-module"),
        "load-generated-symbol" => Ok("load-generated-symbol"),
        "cleanup-compile-scratch" => Ok("cleanup-compile-scratch"),
        name => Err(invalid_selection(format!(
            "score.timing.setup_segments[{index}].name is unsupported: {name:?}"
        ))),
    }
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

fn required_u8(value: &Value, name: &str) -> Result<u8, KernelGenerationError> {
    value_as_u8(required_field(value, name)?)
        .ok_or_else(|| invalid_selection(format!("field {name:?} must fit in u8")))
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
        ScheduleTransform::Upcast { axis, factor } => {
            json!({"op": "upcast", "axis": axis, "factor": factor})
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
        ScheduleTransform::Swap { axis_a, axis_b } => {
            json!({"op": "swap", "axis_a": axis_a, "axis_b": axis_b})
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
        KernelScheduleActionArg::AxisPair { axis_a, axis_b } => {
            json!({"kind": "axis-pair", "axis_a": axis_a, "axis_b": axis_b})
        }
    }
}

fn resource_usage_json(resources: KernelResourceUsage) -> Value {
    json!({
        "threads_per_block": resources.threads_per_block,
        "shared_memory_bytes": resources.shared_memory_bytes,
        "accumulator_elements_per_thread": resources.accumulator_elements_per_thread,
        "output_elements_per_thread": resources.output_elements_per_thread,
        "load_elements_per_block": resources.load_elements_per_block,
    })
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
    let setup_segments = timing
        .setup_segments
        .iter()
        .flatten()
        .map(|segment| {
            json!({
                "name": segment.name,
                "source": segment.source.label(),
                "duration_seconds": segment.duration.as_seconds_f64(),
            })
        })
        .collect::<Vec<_>>();
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
        "setup_segments": setup_segments,
    })
}

fn render_bf16_matvec_source(symbol: &str, plan: MatvecSchedulePlan) -> String {
    let plan = plan.normalized();
    let rows_per_block = plan.rows.rows_per_block().max(1);
    let rows_per_upcast = plan.row_upcast.factor().max(1);
    let row_groups_per_block = rows_per_block.div_ceil(rows_per_upcast);
    let lanes_per_row = plan.thread_group.lanes_per_row().max(1);
    let reduce_unroll = plan.reduce_unroll.max(1);
    let reduce_offsets = warp_subgroup_reduce_offsets(lanes_per_row);
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
    writeln!(source, "const LANES_PER_ROW: u32 = {lanes_per_row};").expect("write to string");
    writeln!(source, "const ROWS_PER_BLOCK: u32 = {rows_per_block};").expect("write to string");
    writeln!(source, "const ROWS_PER_UPCAST: u32 = {rows_per_upcast};").expect("write to string");
    writeln!(
        source,
        "const ROW_GROUPS_PER_BLOCK: u32 = {row_groups_per_block};"
    )
    .expect("write to string");
    writeln!(source, "const REDUCE_UNROLL: u32 = {reduce_unroll};").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "#[inline(always)]").expect("write to string");
    writeln!(source, "fn warp_reduce_sum(mut acc: f32) -> f32 {{").expect("write to string");
    for offset in reduce_offsets {
        writeln!(source, "    acc += warp::shuffle_down_f32(acc, {offset});")
            .expect("write to string");
    }
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
    writeln!(
        source,
        "    let row_group_in_block = thread_x / LANES_PER_ROW;"
    )
    .expect("write to string");
    writeln!(
        source,
        "    if row_group_in_block >= ROW_GROUPS_PER_BLOCK {{"
    )
    .expect("write to string");
    writeln!(source, "        return;").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "    let lane = warp::lane_id() % LANES_PER_ROW;").expect("write to string");
    writeln!(source, "    let cols = cols as usize;").expect("write to string");
    writeln!(source, "    let row_stride = row_stride as usize;").expect("write to string");
    writeln!(source, "    let col_stride = col_stride as usize;").expect("write to string");
    writeln!(
        source,
        "    let row_in_block_base = row_group_in_block * ROWS_PER_UPCAST;"
    )
    .expect("write to string");
    writeln!(source).expect("write to string");
    for row_offset in 0..rows_per_upcast {
        render_matvec_upcast_row_body(
            &mut source,
            row_offset,
            reduce_unroll,
            lanes_per_row,
            "    ",
        );
    }
    writeln!(source, "}}").expect("write to string");
    source
}

fn render_matvec_upcast_row_body(
    source: &mut String,
    row_offset: u32,
    reduce_unroll: u32,
    lanes_per_row: u32,
    indent: &str,
) {
    let inner = format!("{indent}    ");
    let last_offset = (reduce_unroll - 1) * lanes_per_row;
    let stride = reduce_unroll * lanes_per_row;
    writeln!(
        source,
        "{indent}let row{row_offset}_in_block = row_in_block_base + {row_offset};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}let row{row_offset} = (thread::blockIdx_x() * ROWS_PER_BLOCK + row{row_offset}_in_block) as usize;"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}if row{row_offset}_in_block < ROWS_PER_BLOCK && row{row_offset} < rows as usize {{"
    )
    .expect("write to string");
    writeln!(
        source,
        "{inner}let row_base = row{row_offset} * row_stride;"
    )
    .expect("write to string");
    writeln!(source, "{inner}let mut acc = 0.0_f32;").expect("write to string");
    writeln!(source, "{inner}let mut col = lane as usize;").expect("write to string");
    if reduce_unroll > 1 {
        writeln!(source, "{inner}while col + {last_offset} < cols {{").expect("write to string");
        for offset in 0..reduce_unroll {
            let col_expr = if offset == 0 {
                "col".to_string()
            } else {
                format!("col + {}", offset * lanes_per_row)
            };
            writeln!(source, "{inner}    let col{offset} = {col_expr};").expect("write to string");
        }
        for offset in 0..reduce_unroll {
            writeln!(
                source,
                "{inner}    acc += weight[row_base + col{offset} * col_stride].to_f32() * input[col{offset}];"
            )
            .expect("write to string");
        }
        writeln!(source, "{inner}    col += {stride};").expect("write to string");
        writeln!(source, "{inner}}}").expect("write to string");
    }
    writeln!(source, "{inner}while col < cols {{").expect("write to string");
    writeln!(
        source,
        "{inner}    acc += weight[row_base + col * col_stride].to_f32() * input[col];"
    )
    .expect("write to string");
    writeln!(source, "{inner}    col += LANES_PER_ROW as usize;").expect("write to string");
    writeln!(source, "{inner}}}").expect("write to string");
    writeln!(source, "{inner}let acc = warp_reduce_sum(acc);").expect("write to string");
    writeln!(source, "{inner}if lane == 0 {{").expect("write to string");
    writeln!(source, "{inner}    unsafe {{").expect("write to string");
    writeln!(
        source,
        "{inner}        *out.get_unchecked_mut(row{row_offset}) = acc;"
    )
    .expect("write to string");
    writeln!(source, "{inner}    }}").expect("write to string");
    writeln!(source, "{inner}}}").expect("write to string");
    writeln!(source, "{indent}}}").expect("write to string");
    writeln!(source).expect("write to string");
}

fn warp_subgroup_reduce_offsets(lanes_per_row: u32) -> Vec<u32> {
    let mut offset = lanes_per_row / 2;
    let mut offsets = Vec::new();
    while offset > 0 {
        offsets.push(offset);
        offset /= 2;
    }
    offsets
}

fn render_gemm_a_load_body(
    source: &mut String,
    load_name: &str,
    suffix: u32,
    indent: &str,
    m_contiguous_a_load: bool,
) {
    if m_contiguous_a_load {
        writeln!(
            source,
            "{indent}let a_tile_row{suffix} = {load_name} % TILE_M;"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let a_tile_col{suffix} = {load_name} / TILE_M;"
        )
        .expect("write to string");
    } else {
        writeln!(
            source,
            "{indent}let a_tile_row{suffix} = {load_name} / TILE_K;"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let a_tile_col{suffix} = {load_name} % TILE_K;"
        )
        .expect("write to string");
    }
    writeln!(
        source,
        "{indent}let a_smem_index{suffix} = a_tile_row{suffix} * TILE_K + a_tile_col{suffix};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}let a_global_row{suffix} = thread::blockIdx_y() as usize * TILE_M + a_tile_row{suffix};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}let a_global_col{suffix} = k_base + a_tile_col{suffix};"
    )
    .expect("write to string");
    writeln!(source, "{indent}unsafe {{").expect("write to string");
    writeln!(
        source,
        "{indent}    TILE_A[a_smem_index{suffix}] = if a_global_row{suffix} < m && a_global_col{suffix} < k {{"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}        a[a_global_row{suffix} * a_row_stride + a_global_col{suffix} * a_col_stride]"
    )
    .expect("write to string");
    writeln!(source, "{indent}    }} else {{").expect("write to string");
    writeln!(source, "{indent}        0.0").expect("write to string");
    writeln!(source, "{indent}    }};").expect("write to string");
    writeln!(source, "{indent}}}").expect("write to string");
}

fn render_gemm_b_load_body(
    source: &mut String,
    load_name: &str,
    suffix: u32,
    indent: &str,
    k_contiguous_b_load: bool,
) {
    if k_contiguous_b_load {
        writeln!(
            source,
            "{indent}let b_tile_row{suffix} = {load_name} % TILE_K;"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let b_tile_col{suffix} = {load_name} / TILE_K;"
        )
        .expect("write to string");
    } else {
        writeln!(
            source,
            "{indent}let b_tile_row{suffix} = {load_name} / TILE_N;"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let b_tile_col{suffix} = {load_name} % TILE_N;"
        )
        .expect("write to string");
    }
    writeln!(
        source,
        "{indent}let b_smem_index{suffix} = b_tile_row{suffix} * TILE_N + b_tile_col{suffix};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}let b_global_row{suffix} = k_base + b_tile_row{suffix};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}let b_global_col{suffix} = thread::blockIdx_x() as usize * TILE_N + b_tile_col{suffix};"
    )
    .expect("write to string");
    writeln!(source, "{indent}unsafe {{").expect("write to string");
    writeln!(
        source,
        "{indent}    TILE_B[b_smem_index{suffix}] = if b_global_row{suffix} < k && b_global_col{suffix} < n {{"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}        b[b_global_row{suffix} * b_row_stride + b_global_col{suffix} * b_col_stride].to_f32()"
    )
    .expect("write to string");
    writeln!(source, "{indent}    }} else {{").expect("write to string");
    writeln!(source, "{indent}        0.0").expect("write to string");
    writeln!(source, "{indent}    }};").expect("write to string");
    writeln!(source, "{indent}}}").expect("write to string");
}

fn render_gemm_a_load_unrolled(
    source: &mut String,
    a_load_unroll: u32,
    m_contiguous_a_load: bool,
    load_thread_const: &str,
) {
    for offset in 0..a_load_unroll {
        let load_name = format!("a_load{offset}");
        if offset == 0 {
            writeln!(source, "            let {load_name} = load;").expect("write to string");
            render_gemm_a_load_body(
                source,
                &load_name,
                offset,
                "            ",
                m_contiguous_a_load,
            );
        } else {
            let load_expr = if offset == 1 {
                format!("load + {load_thread_const}")
            } else {
                format!("load + {load_thread_const} * {offset}")
            };
            writeln!(source, "            let {load_name} = {load_expr};")
                .expect("write to string");
            writeln!(source, "            if {load_name} < TILE_A_ELEMS {{")
                .expect("write to string");
            render_gemm_a_load_body(
                source,
                &load_name,
                offset,
                "                ",
                m_contiguous_a_load,
            );
            writeln!(source, "            }}").expect("write to string");
        }
    }
    writeln!(
        source,
        "            load += {load_thread_const} * A_LOAD_UNROLL;"
    )
    .expect("write to string");
}

fn render_gemm_b_load_unrolled(
    source: &mut String,
    b_load_unroll: u32,
    k_contiguous_b_load: bool,
    load_thread_const: &str,
) {
    for offset in 0..b_load_unroll {
        let load_name = format!("b_load{offset}");
        if offset == 0 {
            writeln!(source, "            let {load_name} = load;").expect("write to string");
            render_gemm_b_load_body(
                source,
                &load_name,
                offset,
                "            ",
                k_contiguous_b_load,
            );
        } else {
            let load_expr = if offset == 1 {
                format!("load + {load_thread_const}")
            } else {
                format!("load + {load_thread_const} * {offset}")
            };
            writeln!(source, "            let {load_name} = {load_expr};")
                .expect("write to string");
            writeln!(source, "            if {load_name} < TILE_B_ELEMS {{")
                .expect("write to string");
            render_gemm_b_load_body(
                source,
                &load_name,
                offset,
                "                ",
                k_contiguous_b_load,
            );
            writeln!(source, "            }}").expect("write to string");
        }
    }
    writeln!(
        source,
        "            load += {load_thread_const} * B_LOAD_UNROLL;"
    )
    .expect("write to string");
}

fn render_f32_bf16_gemm_source(symbol: &str, plan: GemmSchedulePlan) -> String {
    let plan = plan.normalized();
    let tile = plan.tile;
    let reduce_unroll = plan.reduce_unroll.max(1);
    let m_per_thread = plan.m_per_thread.max(1);
    let n_per_thread = plan.n_per_thread.max(1);
    let a_load_unroll = plan.a_load_unroll.max(1);
    let b_load_unroll = plan.b_load_unroll.max(1);
    let a_load_threads = plan.a_load_thread_count();
    let b_load_threads = plan.b_load_thread_count();
    let m_contiguous_a_load = plan.a_load_order == GemmATileLoadOrder::MContiguous;
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
    writeln!(source, "const THREAD_TILE_M: usize = {m_per_thread};").expect("write to string");
    writeln!(source, "const THREAD_TILE_N: usize = {n_per_thread};").expect("write to string");
    writeln!(source, "const A_LOAD_UNROLL: usize = {a_load_unroll};").expect("write to string");
    writeln!(source, "const B_LOAD_UNROLL: usize = {b_load_unroll};").expect("write to string");
    writeln!(source, "const A_LOAD_THREADS: usize = {a_load_threads};").expect("write to string");
    writeln!(source, "const B_LOAD_THREADS: usize = {b_load_threads};").expect("write to string");
    writeln!(
        source,
        "const THREADS_M: usize = (TILE_M + THREAD_TILE_M - 1) / THREAD_TILE_M;"
    )
    .expect("write to string");
    writeln!(
        source,
        "const THREADS_N: usize = (TILE_N + THREAD_TILE_N - 1) / THREAD_TILE_N;"
    )
    .expect("write to string");
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
    match plan.thread_order {
        GemmThreadOrder::NThenM => {
            writeln!(source, "    let thread_n = thread::threadIdx_x() as usize;")
                .expect("write to string");
            writeln!(source, "    let thread_m = thread::threadIdx_y() as usize;")
                .expect("write to string");
            writeln!(
                source,
                "    if thread_n >= THREADS_N || thread_m >= THREADS_M {{"
            )
            .expect("write to string");
            writeln!(source, "        return;").expect("write to string");
            writeln!(source, "    }}").expect("write to string");
            writeln!(source, "    let tid = thread_m * THREADS_N + thread_n;")
                .expect("write to string");
        }
        GemmThreadOrder::MThenN => {
            writeln!(source, "    let thread_m = thread::threadIdx_x() as usize;")
                .expect("write to string");
            writeln!(source, "    let thread_n = thread::threadIdx_y() as usize;")
                .expect("write to string");
            writeln!(
                source,
                "    if thread_m >= THREADS_M || thread_n >= THREADS_N {{"
            )
            .expect("write to string");
            writeln!(source, "        return;").expect("write to string");
            writeln!(source, "    }}").expect("write to string");
            writeln!(source, "    let tid = thread_n * THREADS_M + thread_m;")
                .expect("write to string");
        }
    }
    writeln!(source).expect("write to string");
    for row_output in 0..m_per_thread {
        if row_output == 0 {
            writeln!(source, "    let tile_row0 = thread_m * THREAD_TILE_M;")
                .expect("write to string");
        } else {
            writeln!(
                source,
                "    let tile_row{row_output} = thread_m * THREAD_TILE_M + {row_output};"
            )
            .expect("write to string");
        }
        writeln!(
            source,
            "    let row{row_output} = thread::blockIdx_y() as usize * TILE_M + tile_row{row_output};"
        )
        .expect("write to string");
    }
    for output in 0..n_per_thread {
        if output == 0 {
            writeln!(source, "    let tile_col0 = thread_n * THREAD_TILE_N;")
                .expect("write to string");
        } else {
            writeln!(
                source,
                "    let tile_col{output} = thread_n * THREAD_TILE_N + {output};"
            )
            .expect("write to string");
        }
        writeln!(
            source,
            "    let col{output} = thread::blockIdx_x() as usize * TILE_N + tile_col{output};"
        )
        .expect("write to string");
    }
    writeln!(source, "    let m = m as usize;").expect("write to string");
    writeln!(source, "    let n = n as usize;").expect("write to string");
    writeln!(source, "    let k = k as usize;").expect("write to string");
    writeln!(source, "    let a_row_stride = a_row_stride as usize;").expect("write to string");
    writeln!(source, "    let a_col_stride = a_col_stride as usize;").expect("write to string");
    writeln!(source, "    let b_row_stride = b_row_stride as usize;").expect("write to string");
    writeln!(source, "    let b_col_stride = b_col_stride as usize;").expect("write to string");
    writeln!(source, "    let c_row_stride = c_row_stride as usize;").expect("write to string");
    writeln!(source, "    let c_col_stride = c_col_stride as usize;").expect("write to string");
    for row_output in 0..m_per_thread {
        for col_output in 0..n_per_thread {
            let acc = row_output * n_per_thread + col_output;
            writeln!(source, "    let mut acc{acc} = 0.0_f32;").expect("write to string");
        }
    }
    writeln!(source, "    let mut k_base = 0;").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "    while k_base < k {{").expect("write to string");
    writeln!(
        source,
        "        let mut load = if tid < A_LOAD_THREADS {{ tid }} else {{ TILE_A_ELEMS }};"
    )
    .expect("write to string");
    writeln!(source, "        while load < TILE_A_ELEMS {{").expect("write to string");
    if a_load_unroll > 1 {
        render_gemm_a_load_unrolled(
            &mut source,
            a_load_unroll,
            m_contiguous_a_load,
            "A_LOAD_THREADS",
        );
    } else {
        if m_contiguous_a_load {
            writeln!(source, "            let tile_row = load % TILE_M;").expect("write to string");
            writeln!(source, "            let tile_col = load / TILE_M;").expect("write to string");
        } else {
            writeln!(source, "            let tile_row = load / TILE_K;").expect("write to string");
            writeln!(source, "            let tile_col = load % TILE_K;").expect("write to string");
        }
        writeln!(
            source,
            "            let a_smem_index = tile_row * TILE_K + tile_col;"
        )
        .expect("write to string");
        writeln!(
            source,
            "            let global_row = thread::blockIdx_y() as usize * TILE_M + tile_row;"
        )
        .expect("write to string");
        writeln!(source, "            let global_col = k_base + tile_col;")
            .expect("write to string");
        writeln!(source, "            unsafe {{").expect("write to string");
        writeln!(
            source,
            "                TILE_A[a_smem_index] = if global_row < m && global_col < k {{"
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
        writeln!(source, "            load += A_LOAD_THREADS;").expect("write to string");
    }
    writeln!(source, "        }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(
        source,
        "        load = if tid < B_LOAD_THREADS {{ tid }} else {{ TILE_B_ELEMS }};"
    )
    .expect("write to string");
    writeln!(source, "        while load < TILE_B_ELEMS {{").expect("write to string");
    if b_load_unroll > 1 {
        render_gemm_b_load_unrolled(
            &mut source,
            b_load_unroll,
            k_contiguous_b_load,
            "B_LOAD_THREADS",
        );
    } else {
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
        writeln!(source, "            let global_row = k_base + tile_row;")
            .expect("write to string");
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
        writeln!(source, "            load += B_LOAD_THREADS;").expect("write to string");
    }
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
        for row_output in 0..m_per_thread {
            for col_output in 0..n_per_thread {
                let acc = row_output * n_per_thread + col_output;
                writeln!(
                    source,
                    "                if tile_row{row_output} < TILE_M && tile_col{col_output} < TILE_N {{"
                )
                .expect("write to string");
                writeln!(
                    source,
                    "                    acc{acc} += TILE_A[tile_row{row_output} * TILE_K + {k_expr}] * TILE_B[({k_expr}) * TILE_N + tile_col{col_output}];"
                )
                .expect("write to string");
                writeln!(source, "                }}").expect("write to string");
            }
        }
    }
    writeln!(source, "                kk += REDUCE_UNROLL;").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "            while kk < TILE_K {{").expect("write to string");
    for row_output in 0..m_per_thread {
        for col_output in 0..n_per_thread {
            let acc = row_output * n_per_thread + col_output;
            writeln!(
                source,
                "                if tile_row{row_output} < TILE_M && tile_col{col_output} < TILE_N {{"
            )
            .expect("write to string");
            writeln!(
                source,
                "                    acc{acc} += TILE_A[tile_row{row_output} * TILE_K + kk] * TILE_B[kk * TILE_N + tile_col{col_output}];"
            )
            .expect("write to string");
            writeln!(source, "                }}").expect("write to string");
        }
    }
    writeln!(source, "                kk += 1;").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "        }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "        thread::sync_threads();").expect("write to string");
    writeln!(source, "        k_base += TILE_K;").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source).expect("write to string");
    for row_output in 0..m_per_thread {
        for col_output in 0..n_per_thread {
            let acc = row_output * n_per_thread + col_output;
            writeln!(
                source,
                "    if row{row_output} < m && col{col_output} < n {{"
            )
            .expect("write to string");
            writeln!(
                source,
                "        let c_offset = row{row_output} * c_row_stride + col{col_output} * c_col_stride;"
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
                "            *c_elem = alpha * acc{acc} + beta * current;"
            )
            .expect("write to string");
            writeln!(source, "        }}").expect("write to string");
            writeln!(source, "    }}").expect("write to string");
        }
    }
    writeln!(source, "}}").expect("write to string");
    source
}

fn standalone_package_name(candidate: &KernelCandidateMetadata) -> String {
    format!("nn_rust_kernel_{}", candidate.artifact_key().hex())
}

fn standalone_cargo_toml(package_name: &str) -> String {
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

fn write_cuda_oxide_dependency(manifest: &mut String, crate_name: &str, root: Option<&Path>) {
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

fn standalone_cuda_oxide_checkout_root() -> Option<PathBuf> {
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

fn cuda_oxide_checkout_has_kernel_crates(path: &Path) -> bool {
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

fn toml_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
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
    state = hash_optional_profiling_action_space_set(state, report.action_space.as_ref());
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
    state = hash_optional_profiling_action_space_set(state, report.action_space.as_ref());
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

fn hash_optional_profiling_action_space_set(
    state: u64,
    action_space: Option<&ProfilingActionSpaceSet>,
) -> u64 {
    if let Some(action_space) = action_space {
        hash_profiling_action_space_set(state, action_space)
    } else {
        hash_str(state, "no-action-space")
    }
}

fn hash_profiling_action_space_set(mut state: u64, action_space: &ProfilingActionSpaceSet) -> u64 {
    state = hash_str(state, "profiling-action-space-set");
    state = hash_u64(state, action_space.spaces.len() as u64);
    for space in &action_space.spaces {
        state = hash_profiling_action_space(state, space);
    }
    state
}

fn hash_profiling_action_space(mut state: u64, action_space: &ProfilingActionSpace) -> u64 {
    match action_space {
        ProfilingActionSpace::Split { variants } => {
            state = hash_str(state, "split");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.axis as u64);
                state = hash_u64(state, variant.factor as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        ProfilingActionSpace::Upcast { axis, factors } => {
            state = hash_str(state, "upcast");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::Unroll { axis, factors } => {
            state = hash_str(state, "unroll");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::LocalTile { axis, factors } => {
            state = hash_str(state, "local-tile");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::ThreadGroup { axis, factors } => {
            state = hash_str(state, "thread-group");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::TileGemm { variants } => {
            state = hash_str(state, "tile-gemm");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.m as u64);
                state = hash_u64(state, variant.n as u64);
                state = hash_u64(state, variant.k as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        ProfilingActionSpace::StrideOrder { orders } => {
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
        ProfilingActionSpace::Swap { pairs } => {
            state = hash_str(state, "swap");
            state = hash_u64(state, pairs.len() as u64);
            for (axis_a, axis_b) in pairs {
                state = hash_u64(state, *axis_a as u64);
                state = hash_u64(state, *axis_b as u64);
            }
            state
        }
    }
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
        KernelActionSpace::Upcast { axis, factors } => {
            state = hash_str(state, "upcast");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
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
        KernelActionSpace::LocalTile { axis, factors } => {
            state = hash_str(state, "local-tile");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        KernelActionSpace::ThreadGroup { axis, factors } => {
            state = hash_str(state, "thread-group");
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
        KernelActionSpace::Swap { pairs } => {
            state = hash_str(state, "swap");
            state = hash_u64(state, pairs.len() as u64);
            for (axis_a, axis_b) in pairs {
                state = hash_u64(state, *axis_a as u64);
                state = hash_u64(state, *axis_b as u64);
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
            KernelScheduleActionArg::AxisPair { axis_a, axis_b } => {
                let mut state = hash_str(state, "axis-pair");
                state = hash_u64(state, *axis_a as u64);
                hash_u64(state, *axis_b as u64)
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
            for segment in timing.setup_segments.iter().flatten() {
                state = hash_str(state, segment.name);
                state = hash_str(state, segment.source.label());
                state = hash_u64(
                    state,
                    segment.duration.as_nanos_u128().min(u64::MAX as u128) as u64,
                );
            }
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
        ScheduleTransform::Upcast { axis, factor } => {
            state = hash_str(state, "upcast");
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
        ScheduleTransform::Swap { axis_a, axis_b } => {
            state = hash_str(state, "swap");
            state = hash_u64(state, *axis_a as u64);
            hash_u64(state, *axis_b as u64)
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
        assert_eq!(complete_space.spaces.len(), 4);
        assert!(matches!(
            complete_space.spaces[0],
            KernelActionSpace::Split { .. }
        ));
        assert!(matches!(
            complete_space.spaces[1],
            KernelActionSpace::Upcast { .. }
        ));
        assert!(matches!(
            complete_space.spaces[2],
            KernelActionSpace::Unroll { .. }
        ));
        assert!(matches!(
            complete_space.spaces[3],
            KernelActionSpace::ThreadGroup { .. }
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

        assert_eq!(spaces.spaces.len(), 3);
        assert_eq!(spaces.actions(), actions);
        let KernelActionSpace::Upcast { axis, factors } = &spaces.spaces[0] else {
            panic!("generated matvec split should expose upcast action-space metadata");
        };
        assert_eq!(*axis, 0);
        assert_eq!(
            factors.as_slice(),
            MatvecRowUpcast::SEARCH_FACTORS.as_slice()
        );
        assert!(actions.contains(&KernelScheduleAction::upcast(0, 2)));
        assert!(actions.contains(&KernelScheduleAction::upcast(0, 4)));

        let KernelActionSpace::Unroll { axis, factors } = &spaces.spaces[1] else {
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
        let KernelActionSpace::ThreadGroup { axis, factors } = &spaces.spaces[2] else {
            panic!("generated matvec split should expose thread-group action-space metadata");
        };
        assert_eq!(*axis, 1);
        assert_eq!(
            factors.as_slice(),
            MatvecThreadGroup::SEARCH_LANES_PER_ROW.as_slice()
        );
        assert!(actions.contains(&KernelScheduleAction::thread_group(1, 16)));

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

        let upcast = problem
            .apply_schedule_action(&rows8, &KernelScheduleAction::upcast(0, 2))
            .expect("row upcast action should produce generated candidate metadata");
        assert_eq!(upcast.launch.kernel, "matvec_bf16_rows8_up2");
        assert_eq!(upcast.launch.block_dim.x, 128);
        assert_eq!(
            schedule_matvec_row_upcast(&upcast.schedule).map(MatvecRowUpcast::factor),
            Some(2)
        );
        assert_eq!(
            upcast.action_trace,
            vec![
                KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
                KernelScheduleAction::upcast(0, 2),
            ]
        );
        assert_ne!(rows8.artifact_key(), upcast.artifact_key());

        let grouped = problem
            .apply_schedule_action(&rows8, &KernelScheduleAction::thread_group(1, 16))
            .expect("thread-group action should produce generated candidate metadata");
        assert_eq!(grouped.launch.kernel, "matvec_bf16_rows8_tg16");
        assert_eq!(grouped.launch.block_dim.x, 128);
        assert_eq!(
            schedule_matvec_thread_group(&grouped.schedule).map(MatvecThreadGroup::lanes_per_row),
            Some(16)
        );
        assert_eq!(
            grouped.action_trace,
            vec![
                KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
                KernelScheduleAction::thread_group(1, 16),
            ]
        );
        assert_ne!(rows8.artifact_key(), grouped.artifact_key());
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
        let spaces = problem.action_spaces(&rows8);
        let KernelActionSpace::Unroll { factors, .. } = spaces
            .spaces
            .iter()
            .find(|space| matches!(space, KernelActionSpace::Unroll { .. }))
            .expect("generated matvec split should expose unroll metadata")
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
                if plan.reduce_unroll == MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL
                    && plan.row_upcast.is_default()
                    && plan.thread_group.is_default()
                {
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
    fn action_trace_replay_reconstructs_2d_upcast_gemm_candidate() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let actions = vec![
            KernelScheduleAction::tile_gemm(
                16,
                32,
                16,
                KernelActionMaterialization::DeferredGenerated,
            ),
            KernelScheduleAction::upcast(0, 2),
            KernelScheduleAction::upcast(1, 2),
        ];

        let candidate = replay_schedule_actions(&problem, &actions)
            .expect("valid GEMM 2D upcast trace should replay into candidate metadata");
        let plan = schedule_gemm_plan(&candidate.schedule).expect("upcast GEMM should have plan");

        assert_eq!(candidate.family, "gemm-f32-bf16-row-col-row");
        assert_eq!(candidate.action_trace, actions);
        assert_eq!(
            candidate.launch.kernel,
            "gemm_f32_bf16_tile_16x32x16_mt2_nt2"
        );
        assert_eq!(candidate.launch.block_dim.x, 16);
        assert_eq!(candidate.launch.block_dim.y, 8);
        assert_eq!(candidate.launch.block_dim.z, 1);
        assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
        assert_eq!(plan.m_per_thread, 2);
        assert_eq!(plan.n_per_thread, 2);
        assert!(candidate.schedule.transforms.iter().any(|transform| {
            matches!(transform, ScheduleTransform::Upcast { axis: 0, factor: 2 })
        }));
        assert!(candidate.schedule.transforms.iter().any(|transform| {
            matches!(transform, ScheduleTransform::Upcast { axis: 1, factor: 2 })
        }));

        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("2D upcast generated GEMM candidate should render source");
        assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_mt2_nt2");
        assert!(generated.source.contains("const THREAD_TILE_M: usize = 2;"));
        assert!(generated.source.contains("const THREAD_TILE_N: usize = 2;"));
        assert!(
            generated
                .source
                .contains("let tile_row1 = thread_m * THREAD_TILE_M + 1;")
        );
        assert!(
            generated
                .source
                .contains("let tile_col1 = thread_n * THREAD_TILE_N + 1;")
        );
        assert!(generated.source.contains("let mut acc3 = 0.0_f32;"));
        assert!(generated.source.contains("TILE_A[tile_row1 * TILE_K + kk]"));
        assert!(generated.source.contains("TILE_B[kk * TILE_N + tile_col1]"));
        assert!(
            generated
                .source
                .contains("*c_elem = alpha * acc3 + beta * current;")
        );
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
        assert!(generated.source.contains("const ROWS_PER_UPCAST: u32 = 1;"));
        assert!(
            generated
                .source
                .contains("const ROW_GROUPS_PER_BLOCK: u32 = 13;")
        );
        assert!(generated.source.contains("const REDUCE_UNROLL: u32 = 4;"));
        assert!(
            generated
                .source
                .contains("let row_group_in_block = thread_x / LANES_PER_ROW;")
        );
        assert!(
            generated
                .source
                .contains("let row0_in_block = row_in_block_base + 0;")
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
    fn matvec_generator_renders_row_upcast_source_on_demand() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let candidate = problem.generated_candidate_for_plan(
            MatvecSchedulePlan::new(RowMajorWarpRows::Rows8).with_row_upcast(
                MatvecRowUpcast::new(2).expect("2 rows should be a supported upcast"),
            ),
        );
        let generated = MatvecRustCudaGenerator
            .source_for(&candidate)
            .expect("matvec generator should render row-upcast source");

        assert_eq!(generated.symbol, "matvec_bf16_rows8_up2");
        assert_eq!(candidate.launch.block_dim.x, 128);
        assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
        assert!(generated.source.contains("const ROWS_PER_UPCAST: u32 = 2;"));
        assert!(
            generated
                .source
                .contains("const ROW_GROUPS_PER_BLOCK: u32 = 4;")
        );
        assert!(
            generated
                .source
                .contains("let row1_in_block = row_in_block_base + 1;")
        );
        assert!(
            generated
                .source
                .contains("if row1_in_block < ROWS_PER_BLOCK && row1 < rows as usize")
        );
        assert!(
            generated
                .source
                .contains("*out.get_unchecked_mut(row1) = acc;")
        );
    }

    #[test]
    fn matvec_generator_renders_thread_group_source_on_demand() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let candidate = problem.generated_candidate_for_plan(
            MatvecSchedulePlan::new(RowMajorWarpRows::Rows8)
                .with_reduce_unroll(8)
                .with_thread_group(
                    MatvecThreadGroup::new(16).expect("16 lanes should be supported"),
                ),
        );
        let generated = MatvecRustCudaGenerator
            .source_for(&candidate)
            .expect("matvec generator should render thread-grouped source");

        assert_eq!(generated.symbol, "matvec_bf16_rows8_u8_tg16");
        assert_eq!(candidate.launch.block_dim.x, 128);
        assert!(generated.source.contains("const LANES_PER_ROW: u32 = 16;"));
        assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
        assert!(generated.source.contains("const REDUCE_UNROLL: u32 = 8;"));
        assert!(generated.source.contains("warp::shuffle_down_f32(acc, 8)"));
        assert!(!generated.source.contains("warp::shuffle_down_f32(acc, 16)"));
        assert!(
            generated
                .source
                .contains("let lane = warp::lane_id() % LANES_PER_ROW;")
        );
        assert!(generated.source.contains("while col + 112 < cols"));
        assert!(generated.source.contains("let col7 = col + 112;"));
        assert!(generated.source.contains("col += 128;"));
        assert!(generated.source.contains("col += LANES_PER_ROW as usize;"));
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
        let actions = vec![
            KernelScheduleAction::split(0, 13, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::split(1, 24, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::split(2, 13, KernelActionMaterialization::DeferredGenerated),
        ];
        let deferred = replay_schedule_actions(&problem, &actions)
            .expect("GEMM split trace should expose arbitrary deferred generated tile metadata");
        assert_eq!(
            schedule_gemm_tile(&deferred.schedule),
            Some(GemmTileShape::new(13, 24, 13))
        );
        assert!(!deferred.is_launchable());
        assert_eq!(deferred.launch.kernel, "gemm_f32_bf16_tile_13x24x13");
        assert!(matches!(
            deferred.generated.materialization,
            KernelMaterialization::DeferredGenerated { .. }
        ));
        assert_eq!(deferred.action_trace, actions);
        assert_eq!(deferred.generated.generator, "tiled-gemm-generator");
    }

    #[test]
    fn gemm_action_space_exposes_tile_unroll_upcast_and_stride_metadata() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let seed = problem.seed();
        let full_space = problem.search_space();
        assert_eq!(full_space.spaces.len(), 11);
        let KernelActionSpace::Split { variants } = &full_space.spaces[0] else {
            panic!("GEMM global action space should expose per-axis split metadata");
        };
        assert_eq!(variants.len(), 15);
        assert!(variants.contains(&KernelAxisFactorAction::new(
            0,
            13,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(variants.contains(&KernelAxisFactorAction::new(
            1,
            24,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(variants.contains(&KernelAxisFactorAction::new(
            2,
            32,
            KernelActionMaterialization::DeferredGenerated
        )));
        let KernelActionSpace::TileGemm { variants } = &full_space.spaces[1] else {
            panic!("GEMM global action space should expose existing tile materialization metadata");
        };
        assert_eq!(
            variants,
            &[KernelTile3dAction::new(
                KernelTile3d::new(16, 16, 16),
                KernelActionMaterialization::Existing
            )]
        );
        assert!(matches!(
            full_space.spaces[2],
            KernelActionSpace::Unroll { .. }
        ));
        assert!(matches!(
            full_space.spaces[3],
            KernelActionSpace::Upcast { .. }
        ));
        assert!(matches!(
            full_space.spaces[4],
            KernelActionSpace::Upcast { .. }
        ));
        let KernelActionSpace::Unroll {
            axis: a_load_axis,
            factors: a_load_factors,
        } = &full_space.spaces[5]
        else {
            panic!("GEMM global action space should expose A shared-load unroll metadata");
        };
        assert_eq!(*a_load_axis, 3);
        assert_eq!(a_load_factors, &[2, 3, 4]);
        let KernelActionSpace::Unroll {
            axis: b_load_axis,
            factors: b_load_factors,
        } = &full_space.spaces[6]
        else {
            panic!("GEMM global action space should expose B shared-load unroll metadata");
        };
        assert_eq!(*b_load_axis, 4);
        assert_eq!(b_load_factors, &[2, 3, 4]);
        let KernelActionSpace::ThreadGroup {
            axis: a_load_thread_axis,
            factors: a_load_thread_factors,
        } = &full_space.spaces[7]
        else {
            panic!("GEMM global action space should expose A shared-load thread-group metadata");
        };
        assert_eq!(*a_load_thread_axis, 3);
        assert_eq!(a_load_thread_factors, &[32, 64, 128, 256]);
        let KernelActionSpace::ThreadGroup {
            axis: b_load_thread_axis,
            factors: b_load_thread_factors,
        } = &full_space.spaces[8]
        else {
            panic!("GEMM global action space should expose B shared-load thread-group metadata");
        };
        assert_eq!(*b_load_thread_axis, 4);
        assert_eq!(b_load_thread_factors, &[32, 64, 128, 256]);
        assert!(matches!(
            full_space.spaces[9],
            KernelActionSpace::Swap { .. }
        ));
        assert!(matches!(
            full_space.spaces[10],
            KernelActionSpace::StrideOrder { .. }
        ));

        let tile_spaces = problem.action_spaces(&seed);
        let tile_actions = problem.schedule_actions(&seed);
        let deferred_tile_action = KernelScheduleAction::tile_gemm(
            13,
            24,
            13,
            KernelActionMaterialization::DeferredGenerated,
        );

        assert_eq!(tile_spaces.spaces.len(), 2);
        assert_eq!(tile_spaces.actions(), tile_actions);
        let KernelActionSpace::Split { variants } = &tile_spaces.spaces[0] else {
            panic!("GEMM seed should expose one-axis split action-space metadata");
        };
        assert_eq!(variants.len(), 12);
        assert!(variants.contains(&KernelAxisFactorAction::new(
            0,
            13,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(variants.contains(&KernelAxisFactorAction::new(
            1,
            24,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(variants.contains(&KernelAxisFactorAction::new(
            2,
            32,
            KernelActionMaterialization::DeferredGenerated
        )));
        let KernelActionSpace::TileGemm { variants } = &tile_spaces.spaces[1] else {
            panic!("GEMM seed should expose existing tile materialization metadata");
        };
        assert_eq!(
            variants,
            &[KernelTile3dAction::new(
                KernelTile3d::new(16, 16, 16),
                KernelActionMaterialization::Existing
            )]
        );
        assert_eq!(tile_actions.len(), 13);
        assert!(tile_actions.contains(&KernelScheduleAction::split(
            0,
            13,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(tile_actions.contains(&KernelScheduleAction::tile_gemm(
            16,
            16,
            16,
            KernelActionMaterialization::Existing
        )));
        assert!(!tile_actions.contains(&deferred_tile_action));
        let direct_tile = problem
            .apply_schedule_action(&seed, &deferred_tile_action)
            .expect("direct tile-gemm action should remain replay-compatible");
        assert_eq!(
            schedule_gemm_tile(&direct_tile.schedule),
            Some(GemmTileShape::new(13, 24, 13))
        );

        let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
        let schedule_spaces = problem.action_spaces(&tile_candidate);
        let schedule_actions = problem.schedule_actions(&tile_candidate);
        assert_eq!(schedule_spaces.spaces.len(), 8);
        assert_eq!(schedule_spaces.actions(), schedule_actions);
        let KernelActionSpace::Split { variants } = &schedule_spaces.spaces[0] else {
            panic!("GEMM tile should expose one-axis retile split metadata");
        };
        assert_eq!(variants.len(), 12);
        assert!(variants.contains(&KernelAxisFactorAction::new(
            0,
            24,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(variants.contains(&KernelAxisFactorAction::new(
            1,
            16,
            KernelActionMaterialization::Existing
        )));
        assert!(variants.contains(&KernelAxisFactorAction::new(
            2,
            32,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(!variants.contains(&KernelAxisFactorAction::new(
            0,
            16,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(matches!(
            schedule_spaces.spaces[1],
            KernelActionSpace::Unroll { .. }
        ));
        let KernelActionSpace::Upcast {
            axis: m_axis,
            factors: m_factors,
        } = &schedule_spaces.spaces[2]
        else {
            panic!("GEMM tile should expose M-axis upcast metadata");
        };
        assert_eq!(*m_axis, 0);
        assert_eq!(m_factors, &[2, 4]);
        let KernelActionSpace::Upcast {
            axis: n_axis,
            factors: n_factors,
        } = &schedule_spaces.spaces[3]
        else {
            panic!("GEMM tile should expose N-axis upcast metadata");
        };
        assert_eq!(*n_axis, 1);
        assert_eq!(n_factors, &[2, 4]);
        let KernelActionSpace::ThreadGroup {
            axis: a_load_thread_axis,
            factors: a_load_thread_factors,
        } = &schedule_spaces.spaces[4]
        else {
            panic!("GEMM tile should expose A shared-load thread-group metadata");
        };
        assert_eq!(*a_load_thread_axis, 3);
        assert_eq!(a_load_thread_factors, &[32, 64, 128, 256]);
        let KernelActionSpace::ThreadGroup {
            axis: b_load_thread_axis,
            factors: b_load_thread_factors,
        } = &schedule_spaces.spaces[5]
        else {
            panic!("GEMM tile should expose B shared-load thread-group metadata");
        };
        assert_eq!(*b_load_thread_axis, 4);
        assert_eq!(b_load_thread_factors, &[32, 64, 128, 256]);
        assert!(matches!(
            schedule_spaces.spaces[6],
            KernelActionSpace::Swap { .. }
        ));
        assert!(matches!(
            schedule_spaces.spaces[7],
            KernelActionSpace::StrideOrder { .. }
        ));
        assert_eq!(schedule_actions.len(), 42);
        assert!(schedule_actions.contains(&KernelScheduleAction::split(
            0,
            24,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(schedule_actions.contains(&KernelScheduleAction::split(
            1,
            16,
            KernelActionMaterialization::Existing
        )));
        assert!(schedule_actions.contains(&KernelScheduleAction::split(
            2,
            32,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(schedule_actions.contains(&KernelScheduleAction::unroll(2, 7)));
        assert!(schedule_actions.contains(&KernelScheduleAction::unroll(2, 16)));
        assert!(!schedule_actions.contains(&KernelScheduleAction::unroll(2, 1)));
        assert!(schedule_actions.contains(&KernelScheduleAction::upcast(0, 2)));
        assert!(schedule_actions.contains(&KernelScheduleAction::upcast(0, 4)));
        assert!(!schedule_actions.contains(&KernelScheduleAction::upcast(0, 3)));
        assert!(schedule_actions.contains(&KernelScheduleAction::upcast(1, 2)));
        assert!(schedule_actions.contains(&KernelScheduleAction::upcast(1, 4)));
        assert!(!schedule_actions.contains(&KernelScheduleAction::upcast(1, 3)));
        assert!(schedule_actions.contains(&KernelScheduleAction::stride_order(vec![0, 2])));
        assert!(schedule_actions.contains(&KernelScheduleAction::stride_order(vec![2, 1])));
        assert!(schedule_actions.contains(&KernelScheduleAction::swap(0, 1)));
        assert!(schedule_actions.contains(&KernelScheduleAction::thread_group(3, 64)));
        assert!(schedule_actions.contains(&KernelScheduleAction::thread_group(4, 64)));
        assert!(!schedule_actions.contains(&KernelScheduleAction::thread_group(3, 16)));

        let m_split = problem
            .apply_schedule_action(
                &tile_candidate,
                &KernelScheduleAction::split(0, 24, KernelActionMaterialization::DeferredGenerated),
            )
            .expect("M-axis split action should retile candidate metadata");
        let m_split_plan =
            schedule_gemm_plan(&m_split.schedule).expect("M split candidate should have plan");
        assert_eq!(m_split_plan.tile, GemmTileShape::new(24, 32, 16));
        assert_eq!(m_split.launch.kernel, "gemm_f32_bf16_tile_24x32x16");
        assert_eq!(
            m_split.action_trace,
            vec![KernelScheduleAction::split(
                0,
                24,
                KernelActionMaterialization::DeferredGenerated
            )]
        );

        let existing_split = problem
            .apply_schedule_action(
                &tile_candidate,
                &KernelScheduleAction::split(1, 16, KernelActionMaterialization::Existing),
            )
            .expect("N-axis split to the existing tile should produce existing candidate metadata");
        let existing_split_plan = schedule_gemm_plan(&existing_split.schedule)
            .expect("existing split candidate should have plan");
        assert_eq!(existing_split_plan.tile, GemmTileShape::new(16, 16, 16));
        assert_eq!(existing_split.launch.kernel, "gemm_f32_bf16_tiled_kernel");
        assert!(existing_split.is_launchable());
        assert!(matches!(
            existing_split.generated.materialization,
            KernelMaterialization::Existing { .. }
        ));
        assert!(
            problem
                .apply_schedule_action(
                    &tile_candidate,
                    &KernelScheduleAction::split(
                        1,
                        16,
                        KernelActionMaterialization::DeferredGenerated,
                    ),
                )
                .is_none()
        );

        let unrolled = problem
            .apply_schedule_action(&tile_candidate, &KernelScheduleAction::unroll(2, 7))
            .expect("unroll action should produce candidate metadata");
        let unrolled_plan =
            schedule_gemm_plan(&unrolled.schedule).expect("unrolled candidate should have plan");
        assert_eq!(unrolled_plan.reduce_unroll, 7);
        assert_eq!(unrolled.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u7");

        let m_upcast = problem
            .apply_schedule_action(&tile_candidate, &KernelScheduleAction::upcast(0, 2))
            .expect("M upcast action should produce candidate metadata");
        let m_upcast_plan =
            schedule_gemm_plan(&m_upcast.schedule).expect("M upcast candidate should have plan");
        assert_eq!(m_upcast_plan.m_per_thread, 2);
        assert_eq!(m_upcast.launch.kernel, "gemm_f32_bf16_tile_16x32x16_mt2");
        assert_eq!(m_upcast.launch.block_dim.x, 32);
        assert_eq!(m_upcast.launch.block_dim.y, 8);
        assert_eq!(m_upcast.launch.block_dim.z, 1);

        let n_upcast = problem
            .apply_schedule_action(&tile_candidate, &KernelScheduleAction::upcast(1, 2))
            .expect("N upcast action should produce candidate metadata");
        let n_upcast_plan =
            schedule_gemm_plan(&n_upcast.schedule).expect("N upcast candidate should have plan");
        assert_eq!(n_upcast_plan.n_per_thread, 2);
        assert_eq!(n_upcast.launch.kernel, "gemm_f32_bf16_tile_16x32x16_nt2");
        assert_eq!(n_upcast.launch.block_dim.x, 16);
        assert_eq!(n_upcast.launch.block_dim.y, 16);
        assert_eq!(n_upcast.launch.block_dim.z, 1);

        let a_load_thread_group = problem
            .apply_schedule_action(&tile_candidate, &KernelScheduleAction::thread_group(3, 64))
            .expect("A shared-load thread-group action should produce candidate metadata");
        let a_load_thread_group_plan = schedule_gemm_plan(&a_load_thread_group.schedule)
            .expect("A shared-load thread-group candidate should have plan");
        assert_eq!(a_load_thread_group_plan.a_load_thread_count(), 64);
        assert_eq!(a_load_thread_group_plan.b_load_thread_count(), 512);
        assert_eq!(
            a_load_thread_group.launch.kernel,
            "gemm_f32_bf16_tile_16x32x16_atg64"
        );
        assert!(
            a_load_thread_group
                .schedule
                .transforms
                .iter()
                .any(|transform| {
                    matches!(
                        transform,
                        ScheduleTransform::ThreadGroup {
                            axis: 3,
                            factor: 64
                        }
                    )
                })
        );

        let b_load_thread_group = problem
            .apply_schedule_action(&tile_candidate, &KernelScheduleAction::thread_group(4, 64))
            .expect("B shared-load thread-group action should produce candidate metadata");
        let b_load_thread_group_plan = schedule_gemm_plan(&b_load_thread_group.schedule)
            .expect("B shared-load thread-group candidate should have plan");
        assert_eq!(b_load_thread_group_plan.a_load_thread_count(), 512);
        assert_eq!(b_load_thread_group_plan.b_load_thread_count(), 64);
        assert_eq!(
            b_load_thread_group.launch.kernel,
            "gemm_f32_bf16_tile_16x32x16_btg64"
        );
        assert!(
            b_load_thread_group
                .schedule
                .transforms
                .iter()
                .any(|transform| {
                    matches!(
                        transform,
                        ScheduleTransform::ThreadGroup {
                            axis: 4,
                            factor: 64
                        }
                    )
                })
        );

        let traced_tile = problem
            .apply_schedule_action(&seed, &deferred_tile_action)
            .expect("tile action should produce candidate metadata");
        let traced_unrolled = problem
            .apply_schedule_action(&traced_tile, &KernelScheduleAction::unroll(2, 7))
            .expect("unroll action should extend candidate action trace");
        assert_eq!(
            traced_unrolled.action_trace,
            vec![deferred_tile_action, KernelScheduleAction::unroll(2, 7)]
        );
        let traced_unrolled_plan = schedule_gemm_plan(&traced_unrolled.schedule)
            .expect("traced unrolled candidate should have plan");
        assert_eq!(traced_unrolled_plan.tile, GemmTileShape::new(13, 24, 13));
        assert_eq!(traced_unrolled_plan.reduce_unroll, 7);
        assert_eq!(
            traced_unrolled.launch.kernel,
            "gemm_f32_bf16_tile_13x24x13_u7"
        );

        let a_reordered = problem
            .apply_schedule_action(
                &tile_candidate,
                &KernelScheduleAction::stride_order(vec![0, 2]),
            )
            .expect("A stride-order action should produce candidate metadata");
        let a_reordered_plan =
            schedule_gemm_plan(&a_reordered.schedule).expect("A reordered candidate should plan");
        assert_eq!(
            a_reordered_plan.a_load_order,
            GemmATileLoadOrder::MContiguous
        );
        assert_eq!(
            a_reordered_plan.b_load_order,
            GemmBTileLoadOrder::TileLinear
        );
        assert_eq!(a_reordered.launch.kernel, "gemm_f32_bf16_tile_16x32x16_am");

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

        let swapped = problem
            .apply_schedule_action(&tile_candidate, &KernelScheduleAction::swap(0, 1))
            .expect("swap action should produce candidate metadata");
        let swapped_plan =
            schedule_gemm_plan(&swapped.schedule).expect("swapped candidate should have plan");
        assert_eq!(swapped_plan.thread_order, GemmThreadOrder::MThenN);
        assert_eq!(swapped.launch.kernel, "gemm_f32_bf16_tile_16x32x16_sw01");
        assert_eq!(swapped.launch.block_dim.x, 16);
        assert_eq!(swapped.launch.block_dim.y, 32);
        assert_eq!(swapped.launch.block_dim.z, 1);
        assert!(swapped.schedule.transforms.iter().any(|transform| {
            matches!(
                transform,
                ScheduleTransform::Swap {
                    axis_a: 0,
                    axis_b: 1
                }
            )
        }));
    }

    #[test]
    fn gemm_action_space_exposes_shared_load_unroll_after_local_tiling() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let upcast_candidate = problem.candidate_for_plan(
            GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
                .with_m_per_thread(2)
                .with_n_per_thread(2),
        );
        let spaces = problem.action_spaces(&upcast_candidate);
        let actions = spaces.actions();

        assert_eq!(spaces.spaces.len(), 8);
        let KernelActionSpace::Split { variants } = &spaces.spaces[0] else {
            panic!("upcast GEMM should still expose one-axis retile split metadata");
        };
        assert_eq!(variants.len(), 12);
        assert!(variants.contains(&KernelAxisFactorAction::new(
            0,
            24,
            KernelActionMaterialization::DeferredGenerated
        )));
        let KernelActionSpace::Unroll {
            axis: reduce_axis, ..
        } = &spaces.spaces[1]
        else {
            panic!("upcast GEMM should still expose reduce unroll metadata");
        };
        assert_eq!(*reduce_axis, 2);
        let KernelActionSpace::Unroll {
            axis: a_load_axis,
            factors: a_load_factors,
        } = &spaces.spaces[2]
        else {
            panic!("upcast GEMM should expose A shared-load unroll metadata");
        };
        assert_eq!(*a_load_axis, 3);
        assert_eq!(a_load_factors, &[2]);
        let KernelActionSpace::Unroll {
            axis: b_load_axis,
            factors: b_load_factors,
        } = &spaces.spaces[3]
        else {
            panic!("upcast GEMM should expose B shared-load unroll metadata");
        };
        assert_eq!(*b_load_axis, 4);
        assert_eq!(b_load_factors, &[2, 3, 4]);
        let KernelActionSpace::ThreadGroup {
            axis: a_load_thread_axis,
            factors: a_load_thread_factors,
        } = &spaces.spaces[4]
        else {
            panic!("upcast GEMM should expose A shared-load thread-group metadata");
        };
        assert_eq!(*a_load_thread_axis, 3);
        assert_eq!(a_load_thread_factors, &[32, 64]);
        let KernelActionSpace::ThreadGroup {
            axis: b_load_thread_axis,
            factors: b_load_thread_factors,
        } = &spaces.spaces[5]
        else {
            panic!("upcast GEMM should expose B shared-load thread-group metadata");
        };
        assert_eq!(*b_load_thread_axis, 4);
        assert_eq!(b_load_thread_factors, &[32, 64]);
        assert!(matches!(spaces.spaces[6], KernelActionSpace::Swap { .. }));
        assert!(matches!(
            spaces.spaces[7],
            KernelActionSpace::StrideOrder { .. }
        ));

        assert!(actions.contains(&KernelScheduleAction::split(
            0,
            24,
            KernelActionMaterialization::DeferredGenerated
        )));
        assert!(actions.contains(&KernelScheduleAction::unroll(3, 2)));
        assert!(!actions.contains(&KernelScheduleAction::unroll(3, 3)));
        assert!(actions.contains(&KernelScheduleAction::unroll(4, 4)));
        assert!(actions.contains(&KernelScheduleAction::thread_group(3, 32)));
        assert!(actions.contains(&KernelScheduleAction::thread_group(4, 64)));
        assert!(!actions.contains(&KernelScheduleAction::thread_group(3, 128)));
        assert!(actions.contains(&KernelScheduleAction::swap(0, 1)));

        let a_unrolled = problem
            .apply_schedule_action(&upcast_candidate, &KernelScheduleAction::unroll(3, 2))
            .expect("A shared-load unroll should produce candidate metadata");
        let a_unrolled_plan = schedule_gemm_plan(&a_unrolled.schedule)
            .expect("A load-unrolled candidate should plan");
        assert_eq!(a_unrolled_plan.a_load_unroll, 2);
        assert_eq!(a_unrolled_plan.b_load_unroll, 1);
        assert_eq!(
            a_unrolled.launch.kernel,
            "gemm_f32_bf16_tile_16x32x16_mt2_nt2_au2"
        );

        let b_unrolled = problem
            .apply_schedule_action(&upcast_candidate, &KernelScheduleAction::unroll(4, 4))
            .expect("B shared-load unroll should produce candidate metadata");
        let b_unrolled_plan = schedule_gemm_plan(&b_unrolled.schedule)
            .expect("B load-unrolled candidate should plan");
        assert_eq!(b_unrolled_plan.a_load_unroll, 1);
        assert_eq!(b_unrolled_plan.b_load_unroll, 4);
        assert_eq!(
            b_unrolled.launch.kernel,
            "gemm_f32_bf16_tile_16x32x16_mt2_nt2_bu4"
        );

        let a_thread_grouped = problem
            .apply_schedule_action(
                &upcast_candidate,
                &KernelScheduleAction::thread_group(3, 32),
            )
            .expect("A shared-load thread-group should produce candidate metadata");
        let a_thread_grouped_plan = schedule_gemm_plan(&a_thread_grouped.schedule)
            .expect("A shared-load thread-grouped candidate should plan");
        assert_eq!(a_thread_grouped_plan.a_load_thread_count(), 32);
        assert_eq!(a_thread_grouped_plan.b_load_thread_count(), 128);
        assert_eq!(
            a_thread_grouped.launch.kernel,
            "gemm_f32_bf16_tile_16x32x16_mt2_nt2_atg32"
        );

        let b_thread_grouped = problem
            .apply_schedule_action(
                &upcast_candidate,
                &KernelScheduleAction::thread_group(4, 64),
            )
            .expect("B shared-load thread-group should produce candidate metadata");
        let b_thread_grouped_plan = schedule_gemm_plan(&b_thread_grouped.schedule)
            .expect("B shared-load thread-grouped candidate should plan");
        assert_eq!(b_thread_grouped_plan.a_load_thread_count(), 128);
        assert_eq!(b_thread_grouped_plan.b_load_thread_count(), 64);
        assert_eq!(
            b_thread_grouped.launch.kernel,
            "gemm_f32_bf16_tile_16x32x16_mt2_nt2_btg64"
        );
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
        assert_eq!(candidates.len(), 42);

        let m_split = candidates
            .iter()
            .find(|candidate| {
                schedule_gemm_tile(&candidate.schedule) == Some(GemmTileShape::new(24, 32, 16))
            })
            .expect("GEMM search should expose an M-axis split descriptor");
        assert_eq!(m_split.launch.kernel, "gemm_f32_bf16_tile_24x32x16");
        assert_eq!(
            m_split.action_trace,
            vec![KernelScheduleAction::split(
                0,
                24,
                KernelActionMaterialization::DeferredGenerated
            )]
        );

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
    fn gemm_search_expands_tile_metadata_into_2d_upcast_variants() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
        let candidates = problem.expand(&tile_candidate);

        let m_upcast = candidates
            .iter()
            .find(|candidate| schedule_gemm_m_per_thread(&candidate.schedule) == Some(2))
            .expect("GEMM search should expose an M upcast factor 2 descriptor");
        let m_plan =
            schedule_gemm_plan(&m_upcast.schedule).expect("candidate should have GEMM plan");
        assert_eq!(m_plan.tile, GemmTileShape::new(16, 32, 16));
        assert_eq!(m_plan.m_per_thread, 2);
        assert_eq!(m_plan.n_per_thread, 1);
        assert_ne!(tile_candidate.artifact_key(), m_upcast.artifact_key());
        assert_eq!(m_upcast.launch.kernel, "gemm_f32_bf16_tile_16x32x16_mt2");
        assert_eq!(m_upcast.launch.block_dim.x, 32);
        assert_eq!(m_upcast.launch.block_dim.y, 8);
        assert_eq!(m_upcast.launch.block_dim.z, 1);
        assert!(m_upcast.schedule.transforms.iter().any(|transform| {
            matches!(transform, ScheduleTransform::Upcast { axis: 0, factor: 2 })
        }));

        let n_upcast = candidates
            .iter()
            .find(|candidate| schedule_gemm_n_per_thread(&candidate.schedule) == Some(2))
            .expect("GEMM search should expose an N upcast factor 2 descriptor");
        let plan = schedule_gemm_plan(&n_upcast.schedule).expect("candidate should have GEMM plan");
        assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
        assert_eq!(plan.m_per_thread, 1);
        assert_eq!(plan.n_per_thread, 2);
        assert_eq!(plan.reduce_unroll, 1);
        assert_ne!(tile_candidate.artifact_key(), n_upcast.artifact_key());
        assert_eq!(n_upcast.launch.kernel, "gemm_f32_bf16_tile_16x32x16_nt2");
        assert_eq!(n_upcast.launch.block_dim.x, 16);
        assert_eq!(n_upcast.launch.block_dim.y, 16);
        assert_eq!(n_upcast.launch.block_dim.z, 1);
        assert!(n_upcast.schedule.transforms.iter().any(|transform| {
            matches!(transform, ScheduleTransform::Upcast { axis: 1, factor: 2 })
        }));
    }

    #[test]
    fn gemm_search_expands_upcast_metadata_into_shared_load_unroll_variants() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let upcast_candidate = problem.candidate_for_plan(
            GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
                .with_m_per_thread(2)
                .with_n_per_thread(2),
        );
        let candidates = problem.expand(&upcast_candidate);

        let a_load_unrolled = candidates
            .iter()
            .find(|candidate| schedule_gemm_a_load_unroll(&candidate.schedule) == Some(2))
            .expect("GEMM search should expose A shared-load unroll factor 2 metadata");
        let a_plan = schedule_gemm_plan(&a_load_unrolled.schedule)
            .expect("A load-unrolled candidate should have GEMM plan");
        assert_eq!(a_plan.tile, GemmTileShape::new(16, 32, 16));
        assert_eq!(a_plan.m_per_thread, 2);
        assert_eq!(a_plan.n_per_thread, 2);
        assert_eq!(a_plan.a_load_unroll, 2);
        assert_eq!(a_plan.b_load_unroll, 1);
        assert_eq!(
            a_load_unrolled.launch.kernel,
            "gemm_f32_bf16_tile_16x32x16_mt2_nt2_au2"
        );
        assert!(a_load_unrolled.schedule.transforms.iter().any(|transform| {
            matches!(transform, ScheduleTransform::Unroll { axis: 3, factor: 2 })
        }));

        let b_load_unrolled = candidates
            .iter()
            .find(|candidate| schedule_gemm_b_load_unroll(&candidate.schedule) == Some(4))
            .expect("GEMM search should expose B shared-load unroll factor 4 metadata");
        let b_plan = schedule_gemm_plan(&b_load_unrolled.schedule)
            .expect("B load-unrolled candidate should have GEMM plan");
        assert_eq!(b_plan.tile, GemmTileShape::new(16, 32, 16));
        assert_eq!(b_plan.m_per_thread, 2);
        assert_eq!(b_plan.n_per_thread, 2);
        assert_eq!(b_plan.a_load_unroll, 1);
        assert_eq!(b_plan.b_load_unroll, 4);
        assert_eq!(
            b_load_unrolled.launch.kernel,
            "gemm_f32_bf16_tile_16x32x16_mt2_nt2_bu4"
        );
        assert!(b_load_unrolled.schedule.transforms.iter().any(|transform| {
            matches!(transform, ScheduleTransform::Unroll { axis: 4, factor: 4 })
        }));
    }

    #[test]
    fn gemm_search_expands_tile_metadata_into_a_load_stride_order() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
        let candidates = problem.expand(&tile_candidate);

        let m_contiguous = candidates
            .iter()
            .find(|candidate| {
                schedule_gemm_plan(&candidate.schedule)
                    .map(|plan| plan.a_load_order == GemmATileLoadOrder::MContiguous)
                    .unwrap_or(false)
            })
            .expect("GEMM search should expose M-contiguous A load order metadata");
        let plan =
            schedule_gemm_plan(&m_contiguous.schedule).expect("candidate should have GEMM plan");
        assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
        assert_eq!(plan.reduce_unroll, 1);
        assert_eq!(plan.a_load_order, GemmATileLoadOrder::MContiguous);
        assert_eq!(plan.b_load_order, GemmBTileLoadOrder::TileLinear);
        assert_ne!(tile_candidate.artifact_key(), m_contiguous.artifact_key());
        assert_eq!(m_contiguous.launch.kernel, "gemm_f32_bf16_tile_16x32x16_am");
        assert!(m_contiguous.schedule.transforms.iter().any(|transform| {
            matches!(transform, ScheduleTransform::StrideOrder { axes } if axes == &[0, 2])
        }));
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
        assert!(
            generated
                .source
                .contains("TILE_A[tile_row0 * TILE_K + kk + 6]")
        );
        assert!(
            generated
                .source
                .contains("TILE_B[(kk + 6) * TILE_N + tile_col0]")
        );
        assert!(generated.source.contains("while kk < TILE_K"));
    }

    #[test]
    fn gemm_generator_renders_shared_load_unroll_source_on_demand() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = replay_schedule_actions(
            &problem,
            &[
                KernelScheduleAction::tile_gemm(
                    16,
                    32,
                    16,
                    KernelActionMaterialization::DeferredGenerated,
                ),
                KernelScheduleAction::upcast(0, 2),
                KernelScheduleAction::upcast(1, 2),
                KernelScheduleAction::unroll(3, 2),
                KernelScheduleAction::unroll(4, 2),
            ],
        )
        .expect("valid shared-load-unroll GEMM action trace should replay");
        let plan =
            schedule_gemm_plan(&candidate.schedule).expect("load-unrolled candidate should plan");
        assert_eq!(plan.m_per_thread, 2);
        assert_eq!(plan.n_per_thread, 2);
        assert_eq!(plan.a_load_unroll, 2);
        assert_eq!(plan.b_load_unroll, 2);

        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("GEMM generator should render shared-load-unrolled source");

        assert_eq!(
            generated.symbol,
            "gemm_f32_bf16_tile_16x32x16_mt2_nt2_au2_bu2"
        );
        assert!(generated.source.contains("const A_LOAD_UNROLL: usize = 2;"));
        assert!(generated.source.contains("const B_LOAD_UNROLL: usize = 2;"));
        assert!(
            generated
                .source
                .contains("let a_load1 = load + A_LOAD_THREADS;")
        );
        assert!(generated.source.contains("if a_load1 < TILE_A_ELEMS"));
        assert!(
            generated
                .source
                .contains("load += A_LOAD_THREADS * A_LOAD_UNROLL;")
        );
        assert!(
            generated
                .source
                .contains("let b_load1 = load + B_LOAD_THREADS;")
        );
        assert!(generated.source.contains("if b_load1 < TILE_B_ELEMS"));
        assert!(
            generated
                .source
                .contains("load += B_LOAD_THREADS * B_LOAD_UNROLL;")
        );
    }

    #[test]
    fn gemm_generator_renders_shared_load_thread_group_source_on_demand() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = replay_schedule_actions(
            &problem,
            &[
                KernelScheduleAction::tile_gemm(
                    16,
                    32,
                    16,
                    KernelActionMaterialization::DeferredGenerated,
                ),
                KernelScheduleAction::upcast(0, 2),
                KernelScheduleAction::upcast(1, 2),
                KernelScheduleAction::thread_group(3, 32),
                KernelScheduleAction::thread_group(4, 64),
            ],
        )
        .expect("valid shared-load-thread-group GEMM action trace should replay");
        let plan = schedule_gemm_plan(&candidate.schedule)
            .expect("load-thread-grouped candidate should plan");
        assert_eq!(plan.a_load_thread_count(), 32);
        assert_eq!(plan.b_load_thread_count(), 64);

        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("GEMM generator should render shared-load-thread-grouped source");

        assert_eq!(
            generated.symbol,
            "gemm_f32_bf16_tile_16x32x16_mt2_nt2_atg32_btg64"
        );
        assert!(
            generated
                .source
                .contains("const A_LOAD_THREADS: usize = 32;")
        );
        assert!(
            generated
                .source
                .contains("const B_LOAD_THREADS: usize = 64;")
        );
        assert!(
            generated
                .source
                .contains("let mut load = if tid < A_LOAD_THREADS { tid } else { TILE_A_ELEMS };")
        );
        assert!(
            generated
                .source
                .contains("load = if tid < B_LOAD_THREADS { tid } else { TILE_B_ELEMS };")
        );
        assert!(generated.source.contains("load += A_LOAD_THREADS;"));
        assert!(generated.source.contains("load += B_LOAD_THREADS;"));
    }

    #[test]
    fn gemm_generator_renders_a_load_stride_order_source_on_demand() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = problem.candidate_for_plan(
            GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16))
                .with_a_load_order(GemmATileLoadOrder::MContiguous),
        );
        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("GEMM generator should render A stride-order source");

        assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_am");
        assert!(generated.source.contains("let tile_row = load % TILE_M;"));
        assert!(generated.source.contains("let tile_col = load / TILE_M;"));
        assert!(
            generated
                .source
                .contains("let a_smem_index = tile_row * TILE_K + tile_col;")
        );
        assert!(generated.source.contains("TILE_A[a_smem_index] = if"));
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
    fn gemm_generator_renders_thread_axis_swap_source_on_demand() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = replay_schedule_actions(
            &problem,
            &[
                KernelScheduleAction::tile_gemm(
                    16,
                    32,
                    16,
                    KernelActionMaterialization::DeferredGenerated,
                ),
                KernelScheduleAction::swap(0, 1),
            ],
        )
        .expect("valid swapped GEMM action trace should replay");
        let plan = schedule_gemm_plan(&candidate.schedule).expect("swapped candidate should plan");
        assert_eq!(plan.thread_order, GemmThreadOrder::MThenN);
        assert_eq!(candidate.launch.block_dim.x, 16);
        assert_eq!(candidate.launch.block_dim.y, 32);

        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("GEMM generator should render thread-axis-swapped source");

        assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_sw01");
        assert!(
            generated
                .source
                .contains("let thread_m = thread::threadIdx_x() as usize;")
        );
        assert!(
            generated
                .source
                .contains("let thread_n = thread::threadIdx_y() as usize;")
        );
        assert!(
            generated
                .source
                .contains("if thread_m >= THREADS_M || thread_n >= THREADS_N")
        );
        assert!(
            generated
                .source
                .contains("let tid = thread_n * THREADS_M + thread_m;")
        );
        assert!(
            generated
                .source
                .contains("let tile_row0 = thread_m * THREAD_TILE_M;")
        );
        assert!(
            generated
                .source
                .contains("let tile_col0 = thread_n * THREAD_TILE_N;")
        );
    }

    #[test]
    fn artifact_store_writes_metadata_manifest_without_kernel_source() {
        let root = test_generated_root();
        let store = KernelArtifactStore::new(&root);
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
        let optimization_spec = candidate.optimization_spec();

        assert_eq!(optimization_spec.resources, candidate.resources);

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
        assert_eq!(
            manifest["resources"]["threads_per_block"].as_u64(),
            Some(512)
        );
        assert_eq!(
            manifest["resources"]["shared_memory_bytes"].as_u64(),
            Some(3072)
        );
        assert_eq!(
            manifest["resources"]["accumulator_elements_per_thread"].as_u64(),
            Some(1)
        );
        assert_eq!(
            manifest["resources"]["load_elements_per_block"].as_u64(),
            Some(768)
        );

        remove_test_generated_root(&root);
    }

    #[test]
    fn gemm_resource_limits_reject_overbudget_plan_metadata() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let seed = problem.seed();
        let action = KernelScheduleAction::tile_gemm(
            128,
            128,
            64,
            KernelActionMaterialization::DeferredGenerated,
        );
        let overbudget_plan = GemmSchedulePlan::new(GemmTileShape::new(128, 128, 64));

        assert_eq!(overbudget_plan.resource_usage().threads_per_block, 16_384);
        assert_eq!(overbudget_plan.resource_usage().shared_memory_bytes, 65_536);
        assert!(!GemmSearchProblem::plan_within_resource_limits(
            overbudget_plan
        ));
        assert!(
            problem
                .candidate_for_checked_plan(&seed, &action, overbudget_plan)
                .is_none()
        );
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
        )
        .with_setup_segment(OptimizationTimingSegment::new(
            "compile-standalone-crate",
            ProfileTimeSource::WallClock,
            ProfileDuration::from_seconds_f64(0.125)
                .expect("setup segment duration should be valid"),
        ))
        .with_setup_segment(OptimizationTimingSegment::new(
            "cleanup-compile-scratch",
            ProfileTimeSource::WallClock,
            ProfileDuration::from_seconds_f64(0.015625)
                .expect("cleanup segment duration should be valid"),
        ));
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
        assert_eq!(
            manifest["score"]["timing"]["setup_segments"][0]["name"].as_str(),
            Some("compile-standalone-crate")
        );
        assert_eq!(
            manifest["score"]["timing"]["setup_segments"][0]["source"].as_str(),
            Some("wall-clock")
        );
        assert_eq!(
            manifest["score"]["timing"]["setup_segments"][1]["name"].as_str(),
            Some("cleanup-compile-scratch")
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
        )
        .with_setup_segment(OptimizationTimingSegment::new(
            "load-generated-module",
            ProfileTimeSource::WallClock,
            ProfileDuration::from_seconds_f64(0.03125)
                .expect("setup segment duration should be valid"),
        ));
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
        assert!(selection_text.contains("\"setup_segments\""));
        assert!(selection_text.contains("\"load-generated-module\""));
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
        assert_eq!(
            selection
                .score
                .and_then(|score| score.timing)
                .and_then(|timing| timing.setup_segments[0])
                .map(|segment| segment.name),
            Some("load-generated-module")
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
        let report = result.optimization_report_with_action_space(
            "matvec-bf16-row-major",
            config,
            &problem.search_space(),
        );

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
        assert_eq!(
            report_json["action_space"]["total_actions"].as_u64(),
            Some(problem.search_space().actions().len() as u64)
        );
        assert_eq!(
            report_json["action_space"]["spaces"][0]["op"].as_str(),
            Some("split")
        );
        assert_eq!(
            report_json["action_space"]["spaces"][1]["op"].as_str(),
            Some("upcast")
        );
        assert_eq!(
            report_json["action_space"]["spaces"][2]["op"].as_str(),
            Some("unroll")
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
        let report = result.auto_optimization_report_with_action_space(
            "matvec-bf16-row-major",
            config,
            &problem.search_space(),
        );

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
            report_json["action_space"]["total_actions"].as_u64(),
            Some(problem.search_space().actions().len() as u64)
        );
        assert_eq!(
            report_json["action_space"]["spaces"][0]["variants"][0]["materialization"].as_str(),
            Some("existing")
        );
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
            manifest
                .contains("cuda-device = { path = \"/tmp/cuda oxide/root/crates/cuda-device\" }")
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

        assert_ne!(
            schedule_gemm_tile(&best.schedule),
            Some(GemmSearchProblem::EXISTING_TILE)
        );
        assert!(!best.is_launchable());
        assert!(matches!(
            best.action_trace.as_slice(),
            [KernelScheduleAction {
                op: KernelScheduleActionOp::Split,
                ..
            }]
        ));
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
                beam_width: 64,
                max_depth: 4,
                require_launchable: false,
            },
            |candidate| {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                let target = GemmTileShape::new(13, 24, 13);
                let tile_distance = plan.tile.m.abs_diff(target.m)
                    + plan.tile.n.abs_diff(target.n)
                    + plan.tile.k.abs_diff(target.k);
                let score = f64::from(tile_distance) * 1000.0
                    + if plan.reduce_unroll == 7 {
                        0.0
                    } else {
                        100.0 + f64::from(plan.reduce_unroll)
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
                beam_width: 96,
                max_depth: 5,
                require_launchable: false,
            },
            |candidate| {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                let target = GemmTileShape::new(13, 24, 13);
                let tile_distance = plan.tile.m.abs_diff(target.m)
                    + plan.tile.n.abs_diff(target.n)
                    + plan.tile.k.abs_diff(target.k);
                let score = f64::from(tile_distance) * 10_000.0
                    + if plan.reduce_unroll == 7 {
                        0.0
                    } else {
                        1000.0 + f64::from(plan.reduce_unroll)
                    }
                    + if plan.b_load_order == GemmBTileLoadOrder::KContiguous {
                        0.0
                    } else {
                        100.0
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
                beam_width: 8,
                max_depth: 3,
                require_launchable: false,
            },
            |candidate| {
                let tile = schedule_gemm_tile(&candidate.schedule)?;
                let target = GemmTileShape::new(13, 24, 13);
                let distance = tile.m.abs_diff(target.m)
                    + tile.n.abs_diff(target.n)
                    + tile.k.abs_diff(target.k);
                SearchScore::measured(f64::from(distance))
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
        assert_eq!(result.explored, 41);
        assert_eq!(result.rejected, 40);
        assert_eq!(
            schedule_gemm_tile(&best.schedule),
            Some(GemmTileShape::new(16, 16, 16))
        );
        assert!(best.is_launchable());
    }
}
