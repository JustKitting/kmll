use std::{
    cmp::Ordering,
    collections::HashSet,
    fmt::{self, Write as _},
    fs, io,
    path::{Path, PathBuf},
};

use nn_rust_profiling::{
    CudaLaunchSpec, NumericKind, OperationKind, OperationRoute, TensorTypeSpec, TypedOperationSpec,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelScheduleActionOp {
    Split,
    Unroll,
    TileGemm,
    StrideOrder,
}

impl KernelScheduleActionOp {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Split => "split",
            Self::Unroll => "unroll",
            Self::TileGemm => "tile-gemm",
            Self::StrideOrder => "stride-order",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelActionMaterialization {
    Existing,
    DeferredGenerated,
}

impl KernelActionMaterialization {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Existing => "existing",
            Self::DeferredGenerated => "deferred-generated",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KernelScheduleActionArg {
    Factor(u32),
    Tile3d { m: u32, n: u32, k: u32 },
    AxisOrder(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelScheduleAction {
    pub op: KernelScheduleActionOp,
    pub axis: Option<u8>,
    pub arg: KernelScheduleActionArg,
    pub materialization: KernelActionMaterialization,
}

impl KernelScheduleAction {
    pub const fn split(
        axis: u8,
        factor: u32,
        materialization: KernelActionMaterialization,
    ) -> Self {
        Self {
            op: KernelScheduleActionOp::Split,
            axis: Some(axis),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization,
        }
    }

    pub const fn unroll(axis: u8, factor: u32) -> Self {
        Self {
            op: KernelScheduleActionOp::Unroll,
            axis: Some(axis),
            arg: KernelScheduleActionArg::Factor(factor),
            materialization: KernelActionMaterialization::DeferredGenerated,
        }
    }

    pub const fn tile_gemm(
        m: u32,
        n: u32,
        k: u32,
        materialization: KernelActionMaterialization,
    ) -> Self {
        Self {
            op: KernelScheduleActionOp::TileGemm,
            axis: None,
            arg: KernelScheduleActionArg::Tile3d { m, n, k },
            materialization,
        }
    }

    pub fn stride_order(axes: Vec<u8>) -> Self {
        Self {
            op: KernelScheduleActionOp::StrideOrder,
            axis: None,
            arg: KernelScheduleActionArg::AxisOrder(axes),
            materialization: KernelActionMaterialization::DeferredGenerated,
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchScoreSource {
    Heuristic,
    Measured,
}

impl SearchScoreSource {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Heuristic => "heuristic",
            Self::Measured => "measured",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchScore {
    pub value: f64,
    pub source: SearchScoreSource,
}

impl SearchScore {
    pub fn heuristic(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self {
            value,
            source: SearchScoreSource::Heuristic,
        })
    }

    pub fn measured(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self {
            value,
            source: SearchScoreSource::Measured,
        })
    }
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
        let rows_per_block = schedule_rows_per_block(&candidate.schedule).ok_or_else(|| {
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
            source: render_bf16_matvec_source(&symbol, rows_per_block),
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

#[derive(Debug, Clone, PartialEq)]
pub struct BeamSearchResult {
    pub best: Option<KernelCandidateMetadata>,
    pub beam: Vec<KernelCandidateMetadata>,
    pub explored: usize,
    pub rejected: usize,
}

pub trait KernelMetadataSearchProblem {
    fn seed(&self) -> KernelCandidateMetadata;
    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata>;
    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore>;
}

pub trait KernelActionSearchProblem: KernelMetadataSearchProblem {
    fn schedule_actions(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelScheduleAction>;

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata>;
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatvecSearchProblem {
    pub rows: usize,
    pub cols: usize,
    pub input_dtype: NumericKind,
    pub weight_dtype: NumericKind,
    pub accumulator: NumericKind,
}

impl MatvecSearchProblem {
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
        self.candidate_for_rows_with_materialization(
            plan,
            "matvec_bf16_kernel".to_string(),
            KernelMaterialization::Existing {
                symbol: "matvec_bf16_kernel",
            },
        )
    }

    pub fn generated_candidate_for_rows(&self, plan: RowMajorWarpRows) -> KernelCandidateMetadata {
        let symbol_hint = format!("matvec_bf16_rows{}", plan.rows_per_block());
        self.candidate_for_rows_with_materialization(
            plan,
            symbol_hint.clone(),
            KernelMaterialization::DeferredGenerated {
                symbol_hint,
                reason: "row split descriptor has no emitted Rust CUDA kernel yet".to_string(),
            },
        )
    }

    fn candidate_for_rows_with_materialization(
        &self,
        plan: RowMajorWarpRows,
        launch_kernel: String,
        materialization: KernelMaterialization,
    ) -> KernelCandidateMetadata {
        let rows_per_block = plan.rows_per_block();
        let schedule = KernelSchedule::new()
            .with_transform(ScheduleTransform::Split {
                axis: 0,
                factor: rows_per_block,
            })
            .with_transform(ScheduleTransform::ThreadGroup {
                axis: 0,
                factor: plan.block_threads(),
            });
        let launch = CudaLaunchSpec::new(
            launch_kernel,
            (plan.grid_rows(self.rows), 1, 1),
            (plan.block_threads(), 1, 1),
            0,
        );
        let operation = TypedOperationSpec::new(
            format!("{}::bf16", plan.plan_name()),
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
}

impl KernelActionSearchProblem for MatvecSearchProblem {
    fn schedule_actions(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelScheduleAction> {
        if candidate.schedule.depth() > 0 {
            return Vec::new();
        }
        RowMajorWarpRows::ALL
            .into_iter()
            .flat_map(|plan| {
                let rows_per_block = plan.rows_per_block();
                [
                    KernelScheduleAction::split(
                        0,
                        rows_per_block,
                        KernelActionMaterialization::Existing,
                    ),
                    KernelScheduleAction::split(
                        0,
                        rows_per_block,
                        KernelActionMaterialization::DeferredGenerated,
                    ),
                ]
            })
            .collect()
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata> {
        if candidate.schedule.depth() > 0 {
            return None;
        }
        let KernelScheduleAction {
            op: KernelScheduleActionOp::Split,
            axis: Some(0),
            arg: KernelScheduleActionArg::Factor(rows_per_block),
            materialization,
        } = action
        else {
            return None;
        };
        let plan = RowMajorWarpRows::from_rows_per_block(*rows_per_block)?;

        let next = match materialization {
            KernelActionMaterialization::Existing => self.candidate_for_rows(plan),
            KernelActionMaterialization::DeferredGenerated => {
                self.generated_candidate_for_rows(plan)
            }
        };
        Some(candidate_with_action_trace(candidate, action, next))
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
        let rows_per_block = schedule_rows_per_block(&candidate.schedule)? as usize;
        let blocks = self.rows.div_ceil(rows_per_block);
        let padded_rows = blocks * rows_per_block;
        let useful_fma_ops = self.rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let wasted_rows = padded_rows.saturating_sub(self.rows);
        let wasted_fma_ops = wasted_rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let block_overhead = blocks as f64 * 2048.0;
        let generic_runtime_penalty = if candidate.is_launchable() {
            blocks as f64 * 64.0
        } else {
            0.0
        };
        SearchScore::heuristic(
            useful_fma_ops + wasted_fma_ops * 8.0 + block_overhead + generic_runtime_penalty,
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

    pub fn block_dim(self) -> (u32, u32, u32) {
        (self.n, self.m, 1)
    }

    pub fn grid_dim(self, m: usize, n: usize) -> (u32, u32, u32) {
        ((n as u32).div_ceil(self.n), (m as u32).div_ceil(self.m), 1)
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
    const TILE_SHAPES: [GemmTileShape; 4] = [
        GemmTileShape::new(8, 16, 16),
        GemmTileShape::new(16, 16, 16),
        GemmTileShape::new(16, 32, 16),
        GemmTileShape::new(32, 16, 16),
    ];
    const REDUCE_UNROLL_FACTORS: [u32; 3] = [2, 4, 8];
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
        plan.tile == GemmTileShape::new(16, 16, 16)
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
}

impl KernelActionSearchProblem for GemmSearchProblem {
    fn schedule_actions(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelScheduleAction> {
        let Some(plan) = schedule_gemm_plan(&candidate.schedule) else {
            return Self::TILE_SHAPES
                .into_iter()
                .map(|tile| {
                    KernelScheduleAction::tile_gemm(
                        tile.m,
                        tile.n,
                        tile.k,
                        Self::action_materialization_for_plan(GemmSchedulePlan::new(tile)),
                    )
                })
                .collect();
        };

        if plan.reduce_unroll == 1 {
            let mut actions = Self::REDUCE_UNROLL_FACTORS
                .into_iter()
                .filter(|factor| *factor <= plan.tile.k && plan.tile.k % *factor == 0)
                .map(|factor| KernelScheduleAction::unroll(2, factor))
                .collect::<Vec<_>>();
            if plan.b_load_order == GemmBTileLoadOrder::TileLinear {
                actions.extend(
                    Self::B_LOAD_ORDERS
                        .into_iter()
                        .map(|order| KernelScheduleAction::stride_order(order.action_axes())),
                );
            }
            return actions;
        }

        if plan.b_load_order == GemmBTileLoadOrder::TileLinear {
            return Self::B_LOAD_ORDERS
                .into_iter()
                .map(|order| KernelScheduleAction::stride_order(order.action_axes()))
                .collect();
        }

        Vec::new()
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
                let plan = GemmSchedulePlan::new(GemmTileShape::new(*m, *n, *k));
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
                if plan.reduce_unroll != 1 || *factor > plan.tile.k || plan.tile.k % *factor != 0 {
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
    })
}

fn render_bf16_matvec_source(symbol: &str, rows_per_block: u32) -> String {
    let rows_per_block = rows_per_block.max(1);
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
    writeln!(source, "    while col + 96 < cols {{").expect("write to string");
    writeln!(source, "        let col0 = col;").expect("write to string");
    writeln!(source, "        let col1 = col + 32;").expect("write to string");
    writeln!(source, "        let col2 = col + 64;").expect("write to string");
    writeln!(source, "        let col3 = col + 96;").expect("write to string");
    writeln!(
        source,
        "        acc += weight[row_base + col0 * col_stride].to_f32() * input[col0];"
    )
    .expect("write to string");
    writeln!(
        source,
        "        acc += weight[row_base + col1 * col_stride].to_f32() * input[col1];"
    )
    .expect("write to string");
    writeln!(
        source,
        "        acc += weight[row_base + col2 * col_stride].to_f32() * input[col2];"
    )
    .expect("write to string");
    writeln!(
        source,
        "        acc += weight[row_base + col3 * col_stride].to_f32() * input[col3];"
    )
    .expect("write to string");
    writeln!(source, "        col += 128;").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source).expect("write to string");
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
        assert_eq!(result.explored, 8);
        assert_eq!(result.rejected, 4);
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
        assert_eq!(candidates.len(), 8);

        let generated = candidates
            .iter()
            .find(|candidate| {
                !candidate.is_launchable()
                    && schedule_rows_per_block(&candidate.schedule) == Some(8)
            })
            .expect("matvec search should expose generated rows-per-block metadata");
        assert_eq!(generated.launch.kernel, "matvec_bf16_rows8");
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
        let actions = problem.schedule_actions(&seed);

        assert_eq!(actions.len(), 8);
        assert!(actions.contains(&KernelScheduleAction::split(
            0,
            8,
            KernelActionMaterialization::Existing
        )));
        assert!(actions.contains(&KernelScheduleAction::split(
            0,
            8,
            KernelActionMaterialization::DeferredGenerated
        )));

        let generated = problem
            .apply_schedule_action(
                &seed,
                &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            )
            .expect("row split action should produce candidate metadata");
        assert_eq!(generated.launch.kernel, "matvec_bf16_rows8");
        assert_eq!(schedule_rows_per_block(&generated.schedule), Some(8));
        assert_eq!(
            generated.action_trace,
            vec![KernelScheduleAction::split(
                0,
                8,
                KernelActionMaterialization::DeferredGenerated
            )]
        );
        assert!(matches!(
            generated.generated.materialization,
            KernelMaterialization::DeferredGenerated { .. }
        ));
    }

    #[test]
    fn matvec_generator_renders_rows_per_block_source_on_demand() {
        let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
        let candidate = problem.generated_candidate_for_rows(RowMajorWarpRows::Rows8);
        let generated = MatvecRustCudaGenerator
            .source_for(&candidate)
            .expect("matvec generator should render rows-per-block source");

        assert_eq!(generated.symbol, "matvec_bf16_rows8");
        assert!(generated.source.contains("#[kernel]"));
        assert!(generated.source.contains("pub fn matvec_bf16_rows8("));
        assert!(generated.source.contains("pub struct Bf16(u16);"));
        assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
        assert!(
            generated
                .source
                .contains("let row_in_block = thread_x / LANES_PER_ROW;")
        );
        assert!(generated.source.contains("while col + 96 < cols"));
        assert!(generated.source.contains("warp::shuffle_down_f32(acc, 16)"));
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
                SearchScore::measured(generated_bonus + f64::from(8 - rows_per_block))
            },
        );
        let best = result
            .best
            .expect("matvec search should keep externally best generated candidate");

        assert!(!best.is_launchable());
        assert_eq!(best.launch.kernel, "matvec_bf16_rows8");
        assert_eq!(schedule_rows_per_block(&best.schedule), Some(8));
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
        assert_eq!(candidates.len(), 4);
        let deferred = candidates
            .iter()
            .find(|candidate| {
                schedule_gemm_tile(&candidate.schedule) == Some(GemmTileShape::new(16, 32, 16))
            })
            .expect("GEMM search should expose deferred generated tile metadata");
        assert!(!deferred.is_launchable());
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
        let tile_actions = problem.schedule_actions(&seed);
        let tile_action = KernelScheduleAction::tile_gemm(
            16,
            32,
            16,
            KernelActionMaterialization::DeferredGenerated,
        );

        assert_eq!(tile_actions.len(), 4);
        assert!(tile_actions.contains(&KernelScheduleAction::tile_gemm(
            16,
            16,
            16,
            KernelActionMaterialization::Existing
        )));
        assert!(tile_actions.contains(&tile_action));

        let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
        let schedule_actions = problem.schedule_actions(&tile_candidate);
        assert_eq!(schedule_actions.len(), 4);
        assert!(schedule_actions.contains(&KernelScheduleAction::unroll(2, 4)));
        assert!(schedule_actions.contains(&KernelScheduleAction::stride_order(vec![2, 1])));

        let unrolled = problem
            .apply_schedule_action(&tile_candidate, &KernelScheduleAction::unroll(2, 4))
            .expect("unroll action should produce candidate metadata");
        let unrolled_plan =
            schedule_gemm_plan(&unrolled.schedule).expect("unrolled candidate should have plan");
        assert_eq!(unrolled_plan.reduce_unroll, 4);
        assert_eq!(unrolled.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u4");

        let traced_tile = problem
            .apply_schedule_action(&seed, &tile_action)
            .expect("tile action should produce candidate metadata");
        let traced_unrolled = problem
            .apply_schedule_action(&traced_tile, &KernelScheduleAction::unroll(2, 4))
            .expect("unroll action should extend candidate action trace");
        assert_eq!(
            traced_unrolled.action_trace,
            vec![tile_action, KernelScheduleAction::unroll(2, 4)]
        );
        assert_eq!(traced_unrolled.artifact_key(), unrolled.artifact_key());

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
        let candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("GEMM generator should render deferred tile source");

        assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16");
        assert!(generated.source.contains("#[kernel]"));
        assert!(
            generated
                .source
                .contains("pub fn gemm_f32_bf16_tile_16x32x16(")
        );
        assert!(generated.source.contains("pub struct Bf16(u16);"));
        assert!(generated.source.contains(".to_f32()"));
        assert!(generated.source.contains("const TILE_M: usize = 16;"));
        assert!(generated.source.contains("const TILE_N: usize = 32;"));
        assert!(generated.source.contains("const TILE_K: usize = 16;"));
        assert!(generated.source.contains("while load < TILE_A_ELEMS"));
        assert!(generated.source.contains("while load < TILE_B_ELEMS"));
    }

    #[test]
    fn gemm_search_expands_tile_metadata_into_reduce_unroll_variants() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let tile_candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));
        let candidates = problem.expand(&tile_candidate);
        assert_eq!(candidates.len(), 4);

        let unroll4 = candidates
            .iter()
            .find(|candidate| schedule_gemm_reduce_unroll(&candidate.schedule) == Some(4))
            .expect("GEMM search should expose a reduce unroll factor 4 descriptor");
        assert_eq!(
            schedule_gemm_tile(&unroll4.schedule),
            Some(GemmTileShape::new(16, 32, 16))
        );
        assert_ne!(tile_candidate.artifact_key(), unroll4.artifact_key());
        assert_eq!(unroll4.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u4");
        assert!(!unroll4.is_launchable());
        assert!(matches!(
            unroll4.generated.materialization,
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
            GemmSchedulePlan::new(GemmTileShape::new(16, 32, 16)).with_reduce_unroll(4),
        );
        let generated = GemmRustCudaGenerator
            .source_for(&candidate)
            .expect("GEMM generator should render reduce-unrolled source");

        assert_eq!(generated.symbol, "gemm_f32_bf16_tile_16x32x16_u4");
        assert!(generated.source.contains("const REDUCE_UNROLL: usize = 4;"));
        assert!(
            generated
                .source
                .contains("while kk + REDUCE_UNROLL <= TILE_K")
        );
        assert!(generated.source.contains("TILE_A[ty * TILE_K + kk + 3]"));
        assert!(generated.source.contains("TILE_B[(kk + 3) * TILE_N + tx]"));
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
            Some(GemmTileShape::new(16, 32, 16))
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
                beam_width: 4,
                max_depth: 2,
                require_launchable: false,
            },
            |candidate| {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                let score =
                    if plan.tile == GemmTileShape::new(16, 32, 16) && plan.reduce_unroll == 4 {
                        0.0
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

        assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
        assert_eq!(plan.reduce_unroll, 4);
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
                beam_width: 4,
                max_depth: 3,
                require_launchable: false,
            },
            |candidate| {
                let plan = schedule_gemm_plan(&candidate.schedule)?;
                let score = if plan.tile == GemmTileShape::new(16, 32, 16)
                    && plan.reduce_unroll == 4
                    && plan.b_load_order == GemmBTileLoadOrder::KContiguous
                {
                    0.0
                } else if plan.tile == GemmTileShape::new(16, 32, 16)
                    && (plan.reduce_unroll == 4
                        || plan.b_load_order == GemmBTileLoadOrder::KContiguous)
                {
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
            .expect("GEMM search should keep the externally best stride-order descriptor");
        let plan = schedule_gemm_plan(&best.schedule).expect("best candidate should have a plan");

        assert_eq!(plan.tile, GemmTileShape::new(16, 32, 16));
        assert_eq!(plan.reduce_unroll, 4);
        assert_eq!(plan.b_load_order, GemmBTileLoadOrder::KContiguous);
        assert_eq!(best.launch.kernel, "gemm_f32_bf16_tile_16x32x16_u4_bk");
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
                SearchScore::measured((tile.m + tile.n + tile.k) as f64)
            },
        );
        let best = result
            .best
            .expect("GEMM search should accept externally scored candidates");

        assert_eq!(
            schedule_gemm_tile(&best.schedule),
            Some(GemmTileShape::new(8, 16, 16))
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
        assert_eq!(result.explored, 8);
        assert_eq!(result.rejected, 7);
        assert_eq!(
            schedule_gemm_tile(&best.schedule),
            Some(GemmTileShape::new(16, 16, 16))
        );
        assert!(best.is_launchable());
    }
}
