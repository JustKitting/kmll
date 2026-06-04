use std::{
    error::Error,
    fs,
    io::{self, ErrorKind},
};

use crate::autotune::{
    GemmRustCudaGenerator, GemmSearchProblem, GemmTileShape, InferenceKernelRustCudaGenerator,
    KernelArtifactStore, KernelSourceGenerator, auto_optimize_inference_kernel,
    compile_standalone_kernel_crate, standalone_cargo_toml, standalone_main_source,
};

use super::super::{
    DecompiledAutotuneShape, decompiled_autotune_operation_with_module,
    driver_support::absolute_path,
};
use super::artifacts::{DecompiledSassArtifacts, disassemble_ptx_to_sass};
use super::overview::{
    DecompileAutotuneOverview, DecompileAutotuneOverviewSide, write_decompile_autotune_overview,
};
use super::types::{DecompileAutotuneGemmOptions, DecompileAutotuneGemmReport};

pub fn run_decompile_autotune_gemm(
    options: &DecompileAutotuneGemmOptions,
) -> Result<DecompileAutotuneGemmReport, Box<dyn Error>> {
    if options.m == 0 || options.n == 0 || options.k == 0 {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "decompile autotune GEMM dimensions must be nonzero",
        )));
    }

    let problem = GemmSearchProblem::f32_bf16_row_col_row(options.m, options.n, options.k);
    let source_tile = GemmTileShape::new(16, 16, 8);
    let source_candidate = problem.candidate_for_tile(source_tile);
    let source_kernel = GemmRustCudaGenerator.source_for(&source_candidate)?;
    let artifact_root = absolute_path(&options.artifact_root)?;
    let run_dir = artifact_root
        .join(format!("{}x{}x{}", options.m, options.n, options.k))
        .join(format!(
            "source-tile-{}x{}x{}",
            source_tile.m, source_tile.n, source_tile.k
        ));
    let crate_dir = run_dir.join("standalone-crate");
    let source_path = crate_dir.join("src").join("main.rs");
    let cargo_toml_path = crate_dir.join("Cargo.toml");
    let package_stem = format!(
        "nn_rust_decompile_autotune_gemm_{}x{}x{}_source",
        options.m, options.n, options.k
    );
    fs::create_dir_all(
        source_path
            .parent()
            .expect("source path should have a parent"),
    )?;
    fs::write(&cargo_toml_path, standalone_cargo_toml(&package_stem))?;
    fs::write(&source_path, standalone_main_source(&source_kernel.source))?;

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
        &source_kernel.symbol,
        &options.compile_arch,
    )?;
    let (parsed, source_artifacts) = DecompiledSassArtifacts::parse_and_write(
        run_dir.clone(),
        source_disassembly.sass_path.clone(),
        Some((source_path.clone(), source_kernel.source.clone())),
        source_disassembly.sass,
    )?;

    let function = source_artifacts
        .ir
        .functions
        .iter()
        .find(|function| function.name.as_str() == source_kernel.symbol)
        .or_else(|| source_artifacts.ir.functions.first())
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "compiled source GEMM SASS did not contain any functions",
            )
        })?;
    let routed = decompiled_autotune_operation_with_module(
        &source_artifacts.ir,
        function,
        DecompiledAutotuneShape::GemmF32Bf16RowColRow {
            m: options.m,
            n: options.n,
            k: options.k,
        },
    )
    .map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "compiled source GEMM SASS did not provide supported autotune evidence: {error:?}"
            ),
        )
    })?;

    let optimization = auto_optimize_inference_kernel(&routed.operation, options.config)?;
    let best = optimization.best_candidate().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            "decompiled GEMM autotune did not produce any candidates",
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
                "compiled optimized GEMM SASS did not contain any functions",
            )
        })?;
    let optimized_routed = decompiled_autotune_operation_with_module(
        &optimized_artifacts.ir,
        optimized_function,
        DecompiledAutotuneShape::GemmF32Bf16RowColRow {
            m: options.m,
            n: options.n,
            k: options.k,
        },
    )
    .map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "compiled optimized GEMM SASS did not provide supported autotune evidence: {error:?}"
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
    let overview_paths = write_decompile_autotune_overview(
        &emitted_report.report_path,
        &DecompileAutotuneOverview {
            title: "Decompile Autotune GEMM Overview",
            shape: format!(
                "gemm-f32-bf16-row-col-row m={} n={} k={}",
                options.m, options.n, options.k
            ),
            operation_name: &routed.operation.name,
            config: options.config,
            source: DecompileAutotuneOverviewSide {
                label: "source",
                symbol: &source_kernel.symbol,
                source_path: Some(&source_path),
                ptx_path: Some(&compiled.ptx_path),
                cubin_path: Some(&source_disassembly.cubin_path),
                sass_path: &source_disassembly.sass_path,
                ir_path: &source_artifacts.ir_path,
                lifted_ir_path: &source_artifacts.lifted_ir_path,
                analysis_path: &source_artifacts.analysis_path,
                pattern_path: &source_artifacts.pattern_path,
                side_by_side_path: &source_artifacts.side_by_side_path,
                parsed_instruction_count: parsed.instruction_count(),
                unsupported_instruction_count: source_artifacts.ir.unsupported_instruction_count(),
                semantic_pattern_count: source_artifacts.patterns.pattern_count(),
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
    let auto_report_visual_svg_path = emitted_report.visual_svg_path.clone().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            "auto-search report did not emit an SVG visualization",
        )
    })?;
    let auto_report_visual_html_path =
        emitted_report.visual_html_path.clone().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "auto-search report did not emit an HTML visualization",
            )
        })?;

    Ok(DecompileAutotuneGemmReport {
        m: options.m,
        n: options.n,
        k: options.k,
        source_symbol: source_kernel.symbol,
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
        auto_report_visual_svg_path,
        auto_report_visual_html_path,
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
