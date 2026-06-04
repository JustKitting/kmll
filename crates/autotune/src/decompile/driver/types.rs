use std::{fmt, path::PathBuf};

use nn_rust_inference::runtime;

use crate::autotune::AutoOptimizeConfig;

use super::super::{
    DecompiledAutotuneEvidence, DecompiledAutotuneShape, PtxDecompileProbeKind, SassCoverageReport,
    SimpleKernelFixtureKind, all_simple_kernel_fixture_kinds,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureOptions {
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub fixtures: Vec<SimpleKernelFixtureKind>,
}

impl DecompileFixtureOptions {
    pub fn sm120_default() -> Self {
        Self {
            artifact_root: runtime::default_artifact_dir().join("decompile-fixtures"),
            compile_arch: "sm_120".to_string(),
            fixtures: vec![SimpleKernelFixtureKind::I32Add],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureCoverageOptions {
    pub fixture_options: DecompileFixtureOptions,
    pub coverage_output_dir: PathBuf,
}

impl DecompileFixtureCoverageOptions {
    pub fn sm120_all_default() -> Self {
        let artifact_root = runtime::default_artifact_dir().join("decompile-fixtures");
        Self {
            fixture_options: DecompileFixtureOptions {
                artifact_root: artifact_root.clone(),
                compile_arch: "sm_120".to_string(),
                fixtures: all_simple_kernel_fixture_kinds(),
            },
            coverage_output_dir: artifact_root.join("coverage"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureReport {
    pub fixture: SimpleKernelFixtureKind,
    pub symbol: String,
    pub crate_dir: PathBuf,
    pub source_path: PathBuf,
    pub ptx_path: PathBuf,
    pub cubin_path: PathBuf,
    pub sass_path: PathBuf,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub cfg_block_count: usize,
    pub cfg_edge_count: usize,
    pub dominator_block_count: usize,
    pub natural_loop_count: usize,
    pub region_count: usize,
    pub reaching_use_count: usize,
    pub ssa_value_count: usize,
    pub def_use_edge_count: usize,
    pub value_op_count: usize,
    pub lifted_op_count: usize,
    pub live_range_count: usize,
    pub memory_access_count: usize,
    pub semantic_pattern_count: usize,
    pub unsupported_instruction_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompileFixtureCoverageReport {
    pub fixture_reports: Vec<DecompileFixtureReport>,
    pub coverage_report: SassCoverageReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompilePtxProbeOptions {
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub probes: Vec<PtxDecompileProbeKind>,
}

impl DecompilePtxProbeOptions {
    pub fn sm120_default() -> Self {
        Self {
            artifact_root: runtime::default_artifact_dir().join("decompile-probes"),
            compile_arch: "sm_120".to_string(),
            probes: vec![
                PtxDecompileProbeKind::TensorCoreHmma,
                PtxDecompileProbeKind::TensorCoreImma,
                PtxDecompileProbeKind::TensorCoreDmma,
                PtxDecompileProbeKind::ScalarMemoryLogic,
                PtxDecompileProbeKind::ScalarMemoryAtomic,
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecompilePtxProbeReport {
    pub probe: PtxDecompileProbeKind,
    pub symbol: String,
    pub probe_dir: PathBuf,
    pub ptx_path: PathBuf,
    pub cubin_path: PathBuf,
    pub nvdisasm_sass_path: PathBuf,
    pub cuobjdump_sass_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneMatvecOptions {
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub rows: usize,
    pub cols: usize,
    pub config: AutoOptimizeConfig,
}

impl DecompileAutotuneMatvecOptions {
    pub fn sm120_default(rows: usize, cols: usize) -> Self {
        Self {
            artifact_root: runtime::default_artifact_dir()
                .join("decompile-autotune")
                .join("matvec-bf16-row-major"),
            compile_arch: "sm_120".to_string(),
            rows,
            cols,
            config: AutoOptimizeConfig {
                beam_width: 4,
                max_steps: 2,
                require_launchable: false,
                min_score_improvement: 0.0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneMatvecReport {
    pub rows: usize,
    pub cols: usize,
    pub naive_symbol: String,
    pub source_path: PathBuf,
    pub ptx_path: PathBuf,
    pub cubin_path: PathBuf,
    pub sass_path: PathBuf,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
    pub semantic_pattern_count: usize,
    pub evidence: DecompiledAutotuneEvidence,
    pub operation_name: String,
    pub auto_report_path: PathBuf,
    pub optimized_source_path: PathBuf,
    pub optimized_ptx_path: PathBuf,
    pub optimized_cubin_path: PathBuf,
    pub optimized_sass_path: PathBuf,
    pub optimized_ir_path: PathBuf,
    pub optimized_lifted_ir_path: PathBuf,
    pub optimized_analysis_path: PathBuf,
    pub optimized_pattern_path: PathBuf,
    pub optimized_side_by_side_path: PathBuf,
    pub optimized_parsed_instruction_count: usize,
    pub optimized_unsupported_instruction_count: usize,
    pub optimized_semantic_pattern_count: usize,
    pub optimized_evidence: DecompiledAutotuneEvidence,
    pub best_symbol: String,
    pub best_action_ops: Vec<String>,
    pub best_action_count: usize,
    pub best_score: Option<f64>,
    pub explored: usize,
    pub rejected: usize,
    pub improving_step_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneGemmOptions {
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub config: AutoOptimizeConfig,
}

impl DecompileAutotuneGemmOptions {
    pub fn sm120_default(m: usize, n: usize, k: usize) -> Self {
        Self {
            artifact_root: runtime::default_artifact_dir()
                .join("decompile-autotune")
                .join("gemm-f32-bf16-row-col-row"),
            compile_arch: "sm_120".to_string(),
            m,
            n,
            k,
            config: AutoOptimizeConfig {
                beam_width: 4,
                max_steps: 2,
                require_launchable: false,
                min_score_improvement: 0.0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneGemmReport {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub source_symbol: String,
    pub source_path: PathBuf,
    pub ptx_path: PathBuf,
    pub cubin_path: PathBuf,
    pub sass_path: PathBuf,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
    pub semantic_pattern_count: usize,
    pub evidence: DecompiledAutotuneEvidence,
    pub operation_name: String,
    pub auto_report_path: PathBuf,
    pub optimized_source_path: PathBuf,
    pub optimized_ptx_path: PathBuf,
    pub optimized_cubin_path: PathBuf,
    pub optimized_sass_path: PathBuf,
    pub optimized_ir_path: PathBuf,
    pub optimized_lifted_ir_path: PathBuf,
    pub optimized_analysis_path: PathBuf,
    pub optimized_pattern_path: PathBuf,
    pub optimized_side_by_side_path: PathBuf,
    pub optimized_parsed_instruction_count: usize,
    pub optimized_unsupported_instruction_count: usize,
    pub optimized_semantic_pattern_count: usize,
    pub optimized_evidence: DecompiledAutotuneEvidence,
    pub best_symbol: String,
    pub best_action_ops: Vec<String>,
    pub best_action_count: usize,
    pub best_score: Option<f64>,
    pub explored: usize,
    pub rejected: usize,
    pub improving_step_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneSassOptions {
    pub sass_path: PathBuf,
    pub source_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub artifact_root: PathBuf,
    pub compile_arch: String,
    pub function_symbol: Option<String>,
    pub shape: DecompiledAutotuneShape,
    pub config: AutoOptimizeConfig,
}

impl DecompileAutotuneSassOptions {
    pub fn sm120_default(sass_path: PathBuf, shape: DecompiledAutotuneShape) -> Self {
        Self {
            sass_path,
            source_path: None,
            output_dir: None,
            artifact_root: runtime::default_artifact_dir()
                .join("decompile-autotune")
                .join("input-sass"),
            compile_arch: "sm_120".to_string(),
            function_symbol: None,
            shape,
            config: AutoOptimizeConfig {
                beam_width: 4,
                max_steps: 2,
                require_launchable: false,
                min_score_improvement: 0.0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecompileAutotuneSassReport {
    pub sass_path: PathBuf,
    pub source_path: Option<PathBuf>,
    pub function_symbol: String,
    pub output_dir: PathBuf,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub unsupported_instruction_count: usize,
    pub semantic_pattern_count: usize,
    pub evidence: DecompiledAutotuneEvidence,
    pub operation_name: String,
    pub auto_report_path: PathBuf,
    pub optimized_source_path: PathBuf,
    pub optimized_ptx_path: PathBuf,
    pub optimized_cubin_path: PathBuf,
    pub optimized_sass_path: PathBuf,
    pub optimized_ir_path: PathBuf,
    pub optimized_lifted_ir_path: PathBuf,
    pub optimized_analysis_path: PathBuf,
    pub optimized_pattern_path: PathBuf,
    pub optimized_side_by_side_path: PathBuf,
    pub optimized_parsed_instruction_count: usize,
    pub optimized_unsupported_instruction_count: usize,
    pub optimized_semantic_pattern_count: usize,
    pub optimized_evidence: DecompiledAutotuneEvidence,
    pub best_symbol: String,
    pub best_action_ops: Vec<String>,
    pub best_action_count: usize,
    pub best_score: Option<f64>,
    pub explored: usize,
    pub rejected: usize,
    pub improving_step_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassFileDecompileOptions {
    pub sass_path: PathBuf,
    pub source_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassFileDecompileReport {
    pub sass_path: PathBuf,
    pub source_path: Option<PathBuf>,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
    pub parsed_instruction_count: usize,
    pub cfg_block_count: usize,
    pub cfg_edge_count: usize,
    pub dominator_block_count: usize,
    pub natural_loop_count: usize,
    pub region_count: usize,
    pub reaching_use_count: usize,
    pub ssa_value_count: usize,
    pub def_use_edge_count: usize,
    pub value_op_count: usize,
    pub lifted_op_count: usize,
    pub live_range_count: usize,
    pub memory_access_count: usize,
    pub semantic_pattern_count: usize,
    pub unsupported_instruction_count: usize,
}

impl fmt::Display for SimpleKernelFixtureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
