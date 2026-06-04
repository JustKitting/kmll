use std::{
    error::Error,
    fs,
    io::{self, ErrorKind},
};

use crate::autotune::{
    InferenceKernelRustCudaGenerator, KernelArtifactStore, auto_optimize_inference_kernel,
    compile_standalone_kernel_crate,
};

use super::super::{
    DecompiledAutotuneShape, decompiled_autotune_operation_with_module,
    driver_support::absolute_path,
};
use super::artifacts::{DecompiledSassArtifacts, disassemble_ptx_to_sass};
use super::overview::{
    DecompileAutotuneOverview, DecompileAutotuneOverviewSide, write_decompile_autotune_overview,
};
use super::types::{DecompileAutotuneSassOptions, DecompileAutotuneSassReport};

pub fn run_decompile_autotune_sass(
    options: &DecompileAutotuneSassOptions,
) -> Result<DecompileAutotuneSassReport, Box<dyn Error>> {
    validate_decompiled_shape(options.shape)?;

    let sass_path = absolute_path(&options.sass_path)?;
    let source_path = options
        .source_path
        .as_ref()
        .map(|path| absolute_path(path))
        .transpose()?;
    let artifact_root = absolute_path(&options.artifact_root)?;
    let sass = fs::read_to_string(&sass_path)?;
    let source = match &source_path {
        Some(path) => Some((path.clone(), fs::read_to_string(path)?)),
        None => None,
    };
    let output_dir = match &options.output_dir {
        Some(path) => absolute_path(path)?,
        None => artifact_root.join("input").join(
            sass_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("sass"),
        ),
    };
    let (parsed, input_artifacts) = DecompiledSassArtifacts::parse_and_write(
        output_dir.clone(),
        sass_path.clone(),
        source.clone(),
        sass,
    )?;

    let function = match &options.function_symbol {
        Some(symbol) => input_artifacts
            .ir
            .functions
            .iter()
            .find(|function| function.name.as_str() == symbol)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "SASS function {symbol:?} was not found in {}",
                        sass_path.display()
                    ),
                )
            })?,
        None => input_artifacts.ir.functions.first().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "input SASS did not contain any functions",
            )
        })?,
    };
    let routed =
        decompiled_autotune_operation_with_module(&input_artifacts.ir, function, options.shape)
            .map_err(|error| {
                io::Error::new(
                    ErrorKind::InvalidData,
                    format!("input SASS did not provide supported autotune evidence: {error:?}"),
                )
            })?;

    let optimization = auto_optimize_inference_kernel(&routed.operation, options.config)?;
    let best = optimization.best_candidate().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            "decompiled SASS autotune did not produce any candidates",
        )
    })?;
    let generated_store = KernelArtifactStore::new(artifact_root.join("generated"));
    let auto_report = optimization.auto_optimization_report(options.config);
    let emitted_report = generated_store.emit_auto_search_report(&auto_report)?;
    let emitted_optimized =
        generated_store.emit_standalone_crate(best, &InferenceKernelRustCudaGenerator)?;
    let optimized_output_dir = generated_store.paths_for(best).directory.join("ptx");
    let compiled_optimized = compile_standalone_kernel_crate(
        &emitted_optimized.paths.crate_dir,
        &optimized_output_dir,
        &emitted_optimized.package_name,
        Some(&options.compile_arch),
        Some(&generated_store.standalone_target_root()),
    )?;
    let optimized_sass_dir = generated_store.paths_for(best).directory.join("sass");
    let optimized_disassembly = disassemble_ptx_to_sass(
        compiled_optimized.ptx_path.clone(),
        optimized_sass_dir.clone(),
        &emitted_optimized.symbol,
        &options.compile_arch,
    )?;
    let optimized_source_text = fs::read_to_string(&emitted_optimized.paths.source_path)?;
    let (optimized_parsed, optimized_artifacts) = DecompiledSassArtifacts::parse_and_write(
        optimized_sass_dir,
        optimized_disassembly.sass_path.clone(),
        Some((
            emitted_optimized.paths.source_path.clone(),
            optimized_source_text,
        )),
        optimized_disassembly.sass,
    )?;
    let optimized_function = optimized_artifacts
        .ir
        .functions
        .iter()
        .find(|function| function.name.as_str() == emitted_optimized.symbol)
        .or_else(|| optimized_artifacts.ir.functions.first())
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "compiled optimized SASS did not contain any functions",
            )
        })?;
    let optimized_routed = decompiled_autotune_operation_with_module(
        &optimized_artifacts.ir,
        optimized_function,
        options.shape,
    )
    .map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "compiled optimized SASS did not provide supported autotune evidence: {error:?}"
            ),
        )
    })?;
    let best_action_ops = best
        .action_trace
        .iter()
        .map(|action| action.op.label().to_string())
        .collect::<Vec<_>>();
    let improving_step_count = optimization
        .result
        .steps
        .iter()
        .filter(|step| {
            step.improvement
                .is_some_and(|improvement| improvement > 0.0)
        })
        .count();
    let shape_label = match options.shape {
        DecompiledAutotuneShape::MatvecBf16RowMajor { rows, cols } => {
            format!("matvec-bf16-row-major rows={rows} cols={cols}")
        }
        DecompiledAutotuneShape::GemmF32Bf16RowColRow { m, n, k } => {
            format!("gemm-f32-bf16-row-col-row m={m} n={n} k={k}")
        }
    };
    let overview_paths = write_decompile_autotune_overview(
        &emitted_report.report_path,
        &DecompileAutotuneOverview {
            title: "Decompile Autotune External SASS Overview",
            shape: shape_label,
            operation_name: &routed.operation.name,
            config: options.config,
            source: DecompileAutotuneOverviewSide {
                label: "source",
                symbol: function.name.as_str(),
                source_path: source_path.as_deref(),
                ptx_path: None,
                cubin_path: None,
                sass_path: &sass_path,
                ir_path: &input_artifacts.ir_path,
                lifted_ir_path: &input_artifacts.lifted_ir_path,
                analysis_path: &input_artifacts.analysis_path,
                pattern_path: &input_artifacts.pattern_path,
                side_by_side_path: &input_artifacts.side_by_side_path,
                parsed_instruction_count: parsed.instruction_count(),
                unsupported_instruction_count: input_artifacts.ir.unsupported_instruction_count(),
                semantic_pattern_count: input_artifacts.patterns.pattern_count(),
                evidence: &routed.evidence,
            },
            optimized: DecompileAutotuneOverviewSide {
                label: "optimized",
                symbol: &emitted_optimized.symbol,
                source_path: Some(&emitted_optimized.paths.source_path),
                ptx_path: Some(&compiled_optimized.ptx_path),
                cubin_path: Some(&optimized_disassembly.cubin_path),
                sass_path: &optimized_disassembly.sass_path,
                ir_path: &optimized_artifacts.ir_path,
                lifted_ir_path: &optimized_artifacts.lifted_ir_path,
                analysis_path: &optimized_artifacts.analysis_path,
                pattern_path: &optimized_artifacts.pattern_path,
                side_by_side_path: &optimized_artifacts.side_by_side_path,
                parsed_instruction_count: optimized_parsed.instruction_count(),
                unsupported_instruction_count: optimized_artifacts
                    .ir
                    .unsupported_instruction_count(),
                semantic_pattern_count: optimized_artifacts.patterns.pattern_count(),
                evidence: &optimized_routed.evidence,
            },
            optimization: &optimization.result,
            best,
            source_score: None,
            auto_report_path: &emitted_report.report_path,
        },
    )?;

    Ok(DecompileAutotuneSassReport {
        sass_path,
        source_path,
        function_symbol: function.name.as_str().to_string(),
        output_dir,
        ir_path: input_artifacts.ir_path,
        lifted_ir_path: input_artifacts.lifted_ir_path,
        analysis_path: input_artifacts.analysis_path,
        pattern_path: input_artifacts.pattern_path,
        side_by_side_path: input_artifacts.side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        unsupported_instruction_count: input_artifacts.ir.unsupported_instruction_count(),
        semantic_pattern_count: input_artifacts.patterns.pattern_count(),
        evidence: routed.evidence,
        operation_name: routed.operation.name,
        auto_report_path: emitted_report.report_path,
        overview_path: overview_paths.markdown_path,
        overview_graph_path: overview_paths.graph_path,
        optimized_source_path: emitted_optimized.paths.source_path,
        optimized_ptx_path: compiled_optimized.ptx_path,
        optimized_cubin_path: optimized_disassembly.cubin_path,
        optimized_sass_path: optimized_disassembly.sass_path,
        optimized_ir_path: optimized_artifacts.ir_path,
        optimized_lifted_ir_path: optimized_artifacts.lifted_ir_path,
        optimized_analysis_path: optimized_artifacts.analysis_path,
        optimized_pattern_path: optimized_artifacts.pattern_path,
        optimized_side_by_side_path: optimized_artifacts.side_by_side_path,
        optimized_parsed_instruction_count: optimized_parsed.instruction_count(),
        optimized_unsupported_instruction_count: optimized_artifacts
            .ir
            .unsupported_instruction_count(),
        optimized_semantic_pattern_count: optimized_artifacts.patterns.pattern_count(),
        optimized_evidence: optimized_routed.evidence,
        best_symbol: emitted_optimized.symbol,
        best_action_count: best_action_ops.len(),
        best_action_ops,
        best_score: best.score.map(|score| score.value),
        explored: optimization.result.explored,
        rejected: optimization.result.rejected,
        improving_step_count,
    })
}

fn validate_decompiled_shape(shape: DecompiledAutotuneShape) -> Result<(), Box<dyn Error>> {
    let invalid = match shape {
        DecompiledAutotuneShape::MatvecBf16RowMajor { rows, cols } => rows == 0 || cols == 0,
        DecompiledAutotuneShape::GemmF32Bf16RowColRow { m, n, k } => m == 0 || n == 0 || k == 0,
    };
    if invalid {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "decompile autotune SASS dimensions must be nonzero",
        )));
    }
    Ok(())
}
