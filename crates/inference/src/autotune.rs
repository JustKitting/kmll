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

// The autotune implementation is intentionally split by ownership while staying in
// one Rust module. That preserves the existing public API and private helper
// boundaries without leaving a 9k-line source file.
include!("autotune/core.rs");
include!("autotune/artifacts.rs");
include!("autotune/search.rs");
include!("autotune/matvec.rs");
include!("autotune/gemm.rs");
include!("autotune/metadata.rs");
include!("autotune/codegen.rs");
include!("autotune/hashing.rs");
#[cfg(test)]
include!("autotune/tests.rs");
