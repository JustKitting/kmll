use std::path::{Path, PathBuf};

use crate::autotune::{
    AutoOptimizeConfig, AutoOptimizeResult, KernelCandidateMetadata, SearchScore,
};

use super::super::super::DecompiledAutotuneEvidence;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::decompile::driver) struct DecompileAutotuneOverviewPaths {
    pub markdown_path: PathBuf,
    pub graph_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::decompile::driver) struct DecompileAutotuneOverviewSide<'a> {
    pub label: &'static str,
    pub symbol: &'a str,
    pub source_path: Option<&'a Path>,
    pub ptx_path: Option<&'a Path>,
    pub cubin_path: Option<&'a Path>,
    pub sass_path: &'a Path,
    pub ir_path: &'a Path,
    pub lifted_ir_path: &'a Path,
    pub analysis_path: &'a Path,
    pub pattern_path: &'a Path,
    pub side_by_side_path: &'a Path,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
    pub semantic_pattern_count: usize,
    pub evidence: &'a DecompiledAutotuneEvidence,
}

#[derive(Debug, Clone)]
pub(in crate::decompile::driver) struct DecompileAutotuneOverview<'a> {
    pub title: &'static str,
    pub shape: String,
    pub operation_name: &'a str,
    pub config: AutoOptimizeConfig,
    pub source: DecompileAutotuneOverviewSide<'a>,
    pub optimized: DecompileAutotuneOverviewSide<'a>,
    pub optimization: &'a AutoOptimizeResult,
    pub best: &'a KernelCandidateMetadata,
    pub source_score: Option<SearchScore>,
    pub auto_report_path: &'a Path,
}
