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
    DecompiledAutotuneShape, analyze_sass_ir, decompiled_autotune_operation_with_module,
    driver_support::{absolute_path, render_sass_file_side_by_side, run_capture, run_checked},
    lift_sass_module, lift_sass_value_ir, parse_nvidia_sass, recover_sass_patterns,
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
    let parsed = parse_nvidia_sass(&sass)?;
    let project_ir = lift_sass_module(&parsed);
    let analysis = analyze_sass_ir(&project_ir);
    let lifted = lift_sass_value_ir(&project_ir, &analysis);
    let patterns = recover_sass_patterns(&project_ir);
    let output_dir = match &options.output_dir {
        Some(path) => absolute_path(path)?,
        None => artifact_root.join("input").join(
            sass_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("sass"),
        ),
    };
    fs::create_dir_all(&output_dir)?;
    let ir_path = output_dir.join("lifted.ir.txt");
    fs::write(&ir_path, project_ir.to_text().as_bytes())?;
    let lifted_ir_path = output_dir.join("lifted-value-ir.txt");
    fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;
    let analysis_path = output_dir.join("analysis.txt");
    fs::write(&analysis_path, analysis.to_text().as_bytes())?;
    let pattern_path = output_dir.join("patterns.txt");
    fs::write(&pattern_path, patterns.to_text().as_bytes())?;
    let side_by_side = render_sass_file_side_by_side(
        &sass_path,
        source
            .as_ref()
            .map(|(path, text)| (path.as_path(), text.as_str())),
        &sass,
        &project_ir,
    );
    let side_by_side_path = output_dir.join("source-sass-ir.txt");
    fs::write(&side_by_side_path, side_by_side.as_bytes())?;

    let function = match &options.function_symbol {
        Some(symbol) => project_ir
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
        None => project_ir.functions.first().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "input SASS did not contain any functions",
            )
        })?,
    };
    let routed = decompiled_autotune_operation_with_module(&project_ir, function, options.shape)
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
    fs::create_dir_all(&optimized_sass_dir)?;
    let optimized_cubin_path = optimized_sass_dir.join(format!(
        "{}.{}.cubin",
        emitted_optimized.symbol, options.compile_arch
    ));
    run_checked(
        "ptxas",
        &[
            format!("-arch={}", options.compile_arch),
            "-o".to_string(),
            optimized_cubin_path.display().to_string(),
            compiled_optimized.ptx_path.display().to_string(),
        ],
        &optimized_sass_dir,
    )?;
    let optimized_sass_path = optimized_sass_dir.join(format!(
        "{}.{}.nvdisasm.sass",
        emitted_optimized.symbol, options.compile_arch
    ));
    let optimized_sass = run_capture(
        "nvdisasm",
        &[optimized_cubin_path.display().to_string()],
        &optimized_sass_dir,
    )?;
    fs::write(&optimized_sass_path, optimized_sass.as_bytes())?;
    let optimized_parsed = parse_nvidia_sass(&optimized_sass)?;
    let optimized_ir = lift_sass_module(&optimized_parsed);
    let optimized_analysis = analyze_sass_ir(&optimized_ir);
    let optimized_lifted = lift_sass_value_ir(&optimized_ir, &optimized_analysis);
    let optimized_patterns = recover_sass_patterns(&optimized_ir);
    let optimized_ir_path = optimized_sass_dir.join("lifted.ir.txt");
    fs::write(&optimized_ir_path, optimized_ir.to_text().as_bytes())?;
    let optimized_lifted_ir_path = optimized_sass_dir.join("lifted-value-ir.txt");
    fs::write(
        &optimized_lifted_ir_path,
        optimized_lifted.to_text().as_bytes(),
    )?;
    let optimized_analysis_path = optimized_sass_dir.join("analysis.txt");
    fs::write(
        &optimized_analysis_path,
        optimized_analysis.to_text().as_bytes(),
    )?;
    let optimized_pattern_path = optimized_sass_dir.join("patterns.txt");
    fs::write(
        &optimized_pattern_path,
        optimized_patterns.to_text().as_bytes(),
    )?;
    let optimized_source_text = fs::read_to_string(&emitted_optimized.paths.source_path)?;
    let optimized_side_by_side = render_sass_file_side_by_side(
        &optimized_sass_path,
        Some((
            emitted_optimized.paths.source_path.as_path(),
            optimized_source_text.as_str(),
        )),
        &optimized_sass,
        &optimized_ir,
    );
    let optimized_side_by_side_path = optimized_sass_dir.join("source-sass-ir.txt");
    fs::write(
        &optimized_side_by_side_path,
        optimized_side_by_side.as_bytes(),
    )?;
    let optimized_function = optimized_ir
        .functions
        .iter()
        .find(|function| function.name.as_str() == emitted_optimized.symbol)
        .or_else(|| optimized_ir.functions.first())
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "compiled optimized SASS did not contain any functions",
            )
        })?;
    let optimized_routed = decompiled_autotune_operation_with_module(
        &optimized_ir,
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

    Ok(DecompileAutotuneSassReport {
        sass_path,
        source_path,
        function_symbol: function.name.as_str().to_string(),
        output_dir,
        ir_path,
        lifted_ir_path,
        analysis_path,
        pattern_path,
        side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        unsupported_instruction_count: project_ir.unsupported_instruction_count(),
        semantic_pattern_count: patterns.pattern_count(),
        evidence: routed.evidence,
        operation_name: routed.operation.name,
        auto_report_path: emitted_report.report_path,
        optimized_source_path: emitted_optimized.paths.source_path,
        optimized_ptx_path: compiled_optimized.ptx_path,
        optimized_cubin_path,
        optimized_sass_path,
        optimized_ir_path,
        optimized_lifted_ir_path,
        optimized_analysis_path,
        optimized_pattern_path,
        optimized_side_by_side_path,
        optimized_parsed_instruction_count: optimized_parsed.instruction_count(),
        optimized_unsupported_instruction_count: optimized_ir.unsupported_instruction_count(),
        optimized_semantic_pattern_count: optimized_patterns.pattern_count(),
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
