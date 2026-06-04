use std::{
    error::Error,
    fs,
    io::{self, ErrorKind},
};

use crate::autotune::{
    InferenceKernelRustCudaGenerator, KernelArtifactStore, KernelSourceGenerator,
    MatvecRustCudaGenerator, MatvecSearchProblem, auto_optimize_inference_kernel,
    compile_standalone_kernel_crate, standalone_cargo_toml, standalone_main_source,
};

use super::super::{
    DecompiledAutotuneShape, decompiled_autotune_operation_with_module,
    driver_support::absolute_path,
};
use super::artifacts::{DecompiledSassArtifacts, disassemble_ptx_to_sass};
use super::types::{DecompileAutotuneMatvecOptions, DecompileAutotuneMatvecReport};

pub fn run_decompile_autotune_matvec(
    options: &DecompileAutotuneMatvecOptions,
) -> Result<DecompileAutotuneMatvecReport, Box<dyn Error>> {
    if options.rows == 0 || options.cols == 0 {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "decompile autotune matvec dimensions must be nonzero",
        )));
    }

    let problem = MatvecSearchProblem::bf16_row_major(options.rows, options.cols);
    let naive_candidate = problem.generated_naive_candidate();
    let naive_source = MatvecRustCudaGenerator.source_for(&naive_candidate)?;
    let artifact_root = absolute_path(&options.artifact_root)?;
    let run_dir = artifact_root
        .join(format!("{}x{}", options.rows, options.cols))
        .join("naive");
    let crate_dir = run_dir.join("standalone-crate");
    let source_path = crate_dir.join("src").join("main.rs");
    let cargo_toml_path = crate_dir.join("Cargo.toml");
    let package_stem = format!(
        "nn_rust_decompile_autotune_matvec_{}x{}_naive",
        options.rows, options.cols
    );
    fs::create_dir_all(
        source_path
            .parent()
            .expect("source path should have a parent"),
    )?;
    fs::write(&cargo_toml_path, standalone_cargo_toml(&package_stem))?;
    fs::write(&source_path, standalone_main_source(&naive_source.source))?;

    let ptx_output_dir = run_dir.join("ptx");
    let target_dir = artifact_root.join("standalone-target");
    let compiled = compile_standalone_kernel_crate(
        &crate_dir,
        &ptx_output_dir,
        &package_stem,
        Some(&options.compile_arch),
        Some(&target_dir),
    )?;

    let source_disassembly = disassemble_ptx_to_sass(
        compiled.ptx_path.clone(),
        run_dir.clone(),
        &naive_source.symbol,
        &options.compile_arch,
    )?;
    let (parsed, source_artifacts) = DecompiledSassArtifacts::parse_and_write(
        run_dir.clone(),
        source_disassembly.sass_path.clone(),
        Some((source_path.clone(), naive_source.source.clone())),
        source_disassembly.sass,
    )?;

    let function = source_artifacts
        .ir
        .functions
        .iter()
        .find(|function| function.name.as_str() == naive_source.symbol)
        .or_else(|| source_artifacts.ir.functions.first())
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "compiled naive matvec SASS did not contain any functions",
            )
        })?;
    let routed = decompiled_autotune_operation_with_module(
        &source_artifacts.ir,
        function,
        DecompiledAutotuneShape::MatvecBf16RowMajor {
            rows: options.rows,
            cols: options.cols,
        },
    )
    .map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "compiled naive matvec SASS did not provide supported autotune evidence: {error:?}"
            ),
        )
    })?;

    let optimization = auto_optimize_inference_kernel(&routed.operation, options.config)?;
    let best = optimization.best_candidate().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            "decompiled matvec autotune did not produce any candidates",
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
                "compiled optimized matvec SASS did not contain any functions",
            )
        })?;
    let optimized_routed = decompiled_autotune_operation_with_module(
        &optimized_artifacts.ir,
        optimized_function,
        DecompiledAutotuneShape::MatvecBf16RowMajor {
            rows: options.rows,
            cols: options.cols,
        },
    )
    .map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "compiled optimized matvec SASS did not provide supported autotune evidence: {error:?}"
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

    Ok(DecompileAutotuneMatvecReport {
        rows: options.rows,
        cols: options.cols,
        naive_symbol: naive_source.symbol,
        source_path,
        ptx_path: compiled.ptx_path,
        cubin_path: source_disassembly.cubin_path,
        sass_path: source_disassembly.sass_path,
        ir_path: source_artifacts.ir_path,
        lifted_ir_path: source_artifacts.lifted_ir_path,
        analysis_path: source_artifacts.analysis_path,
        pattern_path: source_artifacts.pattern_path,
        side_by_side_path: source_artifacts.side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        unsupported_instruction_count: source_artifacts.ir.unsupported_instruction_count(),
        semantic_pattern_count: source_artifacts.patterns.pattern_count(),
        evidence: routed.evidence,
        operation_name: routed.operation.name,
        auto_report_path: emitted_report.report_path,
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
