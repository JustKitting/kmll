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
    OptimizationAxisFactorChoice as ProfilingAxisFactorChoice,
    OptimizationCandidateMaterialization as ProfilingCandidateMaterialization,
    OptimizationCandidateSpec, OptimizationResourceUsage as KernelResourceUsage,
    OptimizationScore as SearchScore, OptimizationScoreSource as SearchScoreSource,
    OptimizationSearchConfig, OptimizationSearchReport,
    OptimizationTile3dChoice as ProfilingTile3dChoice, OptimizationTiming,
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

mod artifacts;
mod codegen;
mod core;
mod gemm;
mod hashing;
mod matvec;
mod measurement;
mod metadata;
mod problem;
mod search;

pub use self::{artifacts::*, core::*, gemm::*, matvec::*, measurement::*, problem::*, search::*};

#[cfg(test)]
mod tests;
