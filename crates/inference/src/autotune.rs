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
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelCandidateMetadata {
    pub family: String,
    pub axes: Vec<KernelAxis>,
    pub schedule: KernelSchedule,
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
            source_path: directory.join("kernel.rs"),
            manifest_path: directory.join("manifest.json"),
        }
    }

    pub fn emit<G>(
        &self,
        candidate: &KernelCandidateMetadata,
        generator: &G,
    ) -> Result<EmittedKernelArtifact, KernelGenerationError>
    where
        G: KernelSourceGenerator,
    {
        let generated = generator.source_for(candidate)?;
        let paths = self.paths_for(candidate);
        fs::create_dir_all(&paths.directory)?;
        fs::write(&paths.source_path, generated.source.as_bytes())?;
        let manifest = generated_kernel_manifest(candidate, generator.name(), &generated);
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
        fs::write(&paths.manifest_path, &manifest_bytes)?;
        Ok(EmittedKernelArtifact {
            artifact_key: candidate.artifact_key(),
            symbol: generated.symbol,
            paths,
            source_bytes: generated.source.len(),
            manifest_bytes: manifest_bytes.len(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArtifactPaths {
    pub directory: PathBuf,
    pub source_path: PathBuf,
    pub manifest_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedKernelArtifact {
    pub artifact_key: KernelMetadataKey,
    pub symbol: String,
    pub paths: KernelArtifactPaths,
    pub source_bytes: usize,
    pub manifest_bytes: usize,
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
        let tile = schedule_gemm_tile(&candidate.schedule).ok_or_else(|| {
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
            source: render_f32_bf16_gemm_source(&symbol, tile),
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

pub fn beam_search_metadata<P>(problem: &P, config: BeamSearchConfig) -> BeamSearchResult
where
    P: KernelMetadataSearchProblem,
{
    assert!(config.beam_width > 0, "beam width must be nonzero");

    let mut seed = problem.seed();
    seed.score = problem.score(&seed);
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
                match problem.score(&next) {
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
            "matvec_bf16_kernel",
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
            KernelMaterialization::Existing {
                symbol: "matvec_bf16_kernel",
            },
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
        if candidate.schedule.depth() > 0 {
            return Vec::new();
        }
        RowMajorWarpRows::ALL
            .into_iter()
            .map(|plan| self.candidate_for_rows(plan))
            .collect()
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let rows_per_block = schedule_rows_per_block(&candidate.schedule)? as usize;
        let blocks = self.rows.div_ceil(rows_per_block);
        let padded_rows = blocks * rows_per_block;
        let useful_fma_ops = self.rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let wasted_rows = padded_rows.saturating_sub(self.rows);
        let wasted_fma_ops = wasted_rows.checked_mul(self.cols)?.checked_mul(2)? as f64;
        let block_overhead = blocks as f64 * 2048.0;
        SearchScore::heuristic(useful_fma_ops + wasted_fma_ops * 8.0 + block_overhead)
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
        let existing_tile = tile == GemmTileShape::new(16, 16, 16);
        let symbol_hint = format!("gemm_f32_bf16_tile_{}x{}x{}", tile.m, tile.n, tile.k);
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
            format!("gemm-f32-bf16-{}x{}x{}", tile.m, tile.n, tile.k),
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
                reason: "tile descriptor has no emitted Rust CUDA kernel yet".to_string(),
            }
        };

        KernelCandidateMetadata::new(
            "gemm-f32-bf16-row-col-row",
            self.axes(),
            KernelSchedule::new().with_transform(ScheduleTransform::TileGemm {
                m: tile.m,
                n: tile.n,
                k: tile.k,
            }),
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
        if candidate.schedule.depth() > 0 {
            return Vec::new();
        }
        [
            GemmTileShape::new(8, 16, 16),
            GemmTileShape::new(16, 16, 16),
            GemmTileShape::new(16, 32, 16),
            GemmTileShape::new(32, 16, 16),
        ]
        .into_iter()
        .map(|tile| self.candidate_for_tile(tile))
        .collect()
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        let tile = schedule_gemm_tile(&candidate.schedule)?;
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
        SearchScore::heuristic(padded_fma_ops + block_count * 4096.0)
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

fn generated_kernel_manifest(
    candidate: &KernelCandidateMetadata,
    generator: &'static str,
    source: &GeneratedKernelSource,
) -> Value {
    json!({
        "schema_version": 1,
        "artifact_key": candidate.artifact_key().hex(),
        "family": &candidate.family,
        "generator": generator,
        "source_symbol": &source.symbol,
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

fn score_json(score: SearchScore) -> Value {
    json!({
        "value": score.value,
        "source": match score.source {
            SearchScoreSource::Heuristic => "heuristic",
            SearchScoreSource::Measured => "measured",
        },
    })
}

fn render_f32_bf16_gemm_source(symbol: &str, tile: GemmTileShape) -> String {
    let mut source = String::new();
    writeln!(
        source,
        "use cuda_device::{{DisjointSlice, SharedArray, kernel, thread}};"
    )
    .expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "use crate::dtypes::{{AccumulateToF32, Bf16}};").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "const TILE_M: usize = {};", tile.m).expect("write to string");
    writeln!(source, "const TILE_N: usize = {};", tile.n).expect("write to string");
    writeln!(source, "const TILE_K: usize = {};", tile.k).expect("write to string");
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
    writeln!(source, "            let tile_row = load / TILE_N;").expect("write to string");
    writeln!(source, "            let tile_col = load % TILE_N;").expect("write to string");
    writeln!(source, "            let global_row = k_base + tile_row;").expect("write to string");
    writeln!(
        source,
        "            let global_col = thread::blockIdx_x() as usize * TILE_N + tile_col;"
    )
    .expect("write to string");
    writeln!(source, "            unsafe {{").expect("write to string");
    writeln!(
        source,
        "                TILE_B[load] = if global_row < k && global_col < n {{"
    )
    .expect("write to string");
    writeln!(source, "                    b[global_row * b_row_stride + global_col * b_col_stride].to_f32_accumulator()").expect("write to string");
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
        assert_eq!(result.explored, 4);
        assert_eq!(result.rejected, 0);
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
        assert!(generated.source.contains("const TILE_M: usize = 16;"));
        assert!(generated.source.contains("const TILE_N: usize = 32;"));
        assert!(generated.source.contains("const TILE_K: usize = 16;"));
        assert!(generated.source.contains("while load < TILE_A_ELEMS"));
        assert!(generated.source.contains("while load < TILE_B_ELEMS"));
    }

    #[test]
    fn artifact_store_writes_source_and_manifest_under_managed_style_root() {
        let root = runtime::default_artifact_dir()
            .join("test-generated")
            .join(format!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system clock should be after unix epoch")
                    .as_nanos()
            ));
        let store = KernelArtifactStore::new(&root);
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let candidate = problem.candidate_for_tile(GemmTileShape::new(16, 32, 16));

        let emitted = store
            .emit(&candidate, &GemmRustCudaGenerator)
            .expect("artifact store should write generated source and manifest");

        assert_eq!(emitted.artifact_key, candidate.artifact_key());
        assert_eq!(emitted.symbol, "gemm_f32_bf16_tile_16x32x16");
        assert!(emitted.paths.directory.starts_with(store.root()));
        assert!(emitted.paths.source_path.starts_with(store.root()));
        assert!(emitted.paths.manifest_path.starts_with(store.root()));
        assert!(emitted.source_bytes > 0);
        assert!(emitted.manifest_bytes > 0);

        let source = fs::read_to_string(&emitted.paths.source_path)
            .expect("generated source should be readable");
        assert!(source.contains("pub fn gemm_f32_bf16_tile_16x32x16("));

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
        assert_eq!(
            manifest["source_symbol"].as_str(),
            Some("gemm_f32_bf16_tile_16x32x16")
        );
        assert_eq!(
            manifest["materialization"]["kind"].as_str(),
            Some("deferred-generated")
        );
        assert_eq!(manifest["schedule"][0]["op"].as_str(), Some("tile-gemm"));
        assert_eq!(manifest["schedule"][0]["n"].as_u64(), Some(32));

        fs::remove_dir_all(&root)
            .expect("test-generated kernel artifact directory should clean up");
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
    fn gemm_default_search_keeps_only_currently_launchable_tile() {
        let problem = GemmSearchProblem::f32_bf16_row_col_row(128, 128, 256);
        let result = beam_search_metadata(&problem, BeamSearchConfig::default());
        let best = result
            .best
            .expect("GEMM search should keep the existing tile");
        assert_eq!(result.explored, 4);
        assert_eq!(result.rejected, 3);
        assert_eq!(
            schedule_gemm_tile(&best.schedule),
            Some(GemmTileShape::new(16, 16, 16))
        );
        assert!(best.is_launchable());
    }
}
