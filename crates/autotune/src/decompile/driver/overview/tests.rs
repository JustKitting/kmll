use std::{fs, path::Path, process, time::SystemTime};

use nn_rust_profiling::{
    CudaLaunchSpec, OperationKind, OperationRoute, OptimizationActionMaterialization,
    OptimizationActionSpec, OptimizationScore, TypedOperationSpec,
};

use crate::{
    autotune::{
        AutoOptimizeConfig, AutoOptimizeExitReason, AutoOptimizeResult, AutoOptimizeStep,
        KernelCandidateMetadata, KernelMaterialization,
    },
    decompile::{DecompiledAutotuneEvidence, SassSemanticPatternCategory},
};

use super::{
    graph::render_mermaid_graph,
    markdown::render_markdown_overview,
    types::{DecompileAutotuneOverview, DecompileAutotuneOverviewSide},
    write_decompile_autotune_overview,
};

#[test]
fn overview_marks_only_positive_steps_and_renders_diff() {
    let evidence = source_evidence();
    let optimized_evidence = DecompiledAutotuneEvidence {
        has_warp_reduce_sum: true,
        ..evidence.clone()
    };
    let mut best = KernelCandidateMetadata::new(
        "matvec-bf16-row-major",
        Vec::new(),
        Default::default(),
        "test-generator",
        KernelMaterialization::Generated {
            symbol: "best_kernel".to_string(),
        },
        CudaLaunchSpec::new("best_kernel", (1, 1, 1), (32, 1, 1), 0),
        TypedOperationSpec::new("test-op", OperationKind::Matvec, OperationRoute::CudaKernel),
    );
    best.action_trace = vec![OptimizationActionSpec::split(
        0,
        32,
        OptimizationActionMaterialization::DeferredGenerated,
    )];
    best.score = OptimizationScore::heuristic(2.0);
    let optimization = AutoOptimizeResult {
        best: Some(best.clone()),
        beam: vec![best.clone()],
        explored: 4,
        rejected: 1,
        duplicates: 0,
        steps: vec![
            AutoOptimizeStep {
                depth: 0,
                input_beam_len: 1,
                generated: 4,
                accepted: 2,
                rejected: 1,
                duplicates: 0,
                best_before: OptimizationScore::heuristic(3.0),
                best_after: OptimizationScore::heuristic(2.0),
                best_candidate: Some(best.clone()),
                improvement: Some(1.0),
            },
            AutoOptimizeStep {
                depth: 1,
                input_beam_len: 2,
                generated: 2,
                accepted: 1,
                rejected: 0,
                duplicates: 0,
                best_before: OptimizationScore::heuristic(2.0),
                best_after: OptimizationScore::heuristic(2.5),
                best_candidate: Some(best.clone()),
                improvement: Some(-0.5),
            },
        ],
        exit_reason: AutoOptimizeExitReason::MaxSteps,
    };
    let source = overview_side("source", "source_kernel", &evidence, 10, 1);
    let optimized = overview_side("optimized", "best_kernel", &optimized_evidence, 12, 2);
    let root = overview_test_root();
    let report_path = root.join("reports").join("report.json");
    let overview = DecompileAutotuneOverview {
        title: "Test Overview",
        shape: "matvec rows=4 cols=8".to_string(),
        operation_name: "test-op",
        config: AutoOptimizeConfig {
            beam_width: 2,
            max_steps: 2,
            require_launchable: false,
            min_score_improvement: 0.0,
        },
        source,
        optimized,
        optimization: &optimization,
        best: &best,
        source_score: OptimizationScore::heuristic(3.0),
        auto_report_path: &report_path,
    };

    let graph = render_mermaid_graph(&overview);
    let markdown = render_markdown_overview(&overview, &graph);
    let paths = write_decompile_autotune_overview(&report_path, &overview)
        .expect("overview writer should emit managed artifacts");

    assert!(graph.contains("positive steps: 1"));
    assert!(markdown.contains("| `parsed_instructions` | 10 | 12 | 2 |"));
    assert!(markdown.contains("| `warp_reduce_sum` | false | true | true |"));
    assert!(markdown.contains("| 0 | 4 | 2 | 1 | 3.000000000000 (heuristic) | 2.000000000000 (heuristic) | 1.000000000000 |"));
    assert!(!markdown.contains("| 1 | 2 | 1 | 0 | 2.000000000000 (heuristic) | 2.500000000000 (heuristic) | -0.500000000000 |"));
    assert!(markdown.contains("| 0 | `split` | 0 | `axis=0 factor=32` | `deferred-generated` |"));
    assert!(paths.markdown_path.exists());
    assert!(paths.graph_path.exists());
    assert!(
        fs::read_to_string(&paths.markdown_path)
            .expect("overview markdown should be readable")
            .contains("## Positive Search Steps")
    );
    assert!(
        fs::read_to_string(&paths.graph_path)
            .expect("overview graph should be readable")
            .contains("flowchart LR")
    );
    fs::remove_dir_all(root).expect("overview test artifacts should clean up");
}

fn source_evidence() -> DecompiledAutotuneEvidence {
    DecompiledAutotuneEvidence {
        pattern_categories: vec![SassSemanticPatternCategory::Bf16WidenBits],
        has_bf16_descriptor_load: true,
        has_f32_descriptor_load: false,
        has_bf16_widen: true,
        has_f32_mul_add: true,
        has_f32_fused_multiply_add: false,
        has_shared_load: false,
        has_shared_store: false,
        has_barrier: false,
        has_descriptor_store: false,
        has_warp_reduce_sum: false,
    }
}

fn overview_side<'a>(
    label: &'static str,
    symbol: &'a str,
    evidence: &'a DecompiledAutotuneEvidence,
    parsed_instruction_count: usize,
    semantic_pattern_count: usize,
) -> DecompileAutotuneOverviewSide<'a> {
    DecompileAutotuneOverviewSide {
        label,
        symbol,
        source_path: None,
        ptx_path: None,
        cubin_path: None,
        sass_path: Path::new("kernel.sass"),
        ir_path: Path::new("kernel.ir"),
        lifted_ir_path: Path::new("kernel.lifted"),
        analysis_path: Path::new("kernel.analysis"),
        pattern_path: Path::new("kernel.patterns"),
        side_by_side_path: Path::new("kernel.side"),
        parsed_instruction_count,
        unsupported_instruction_count: 0,
        semantic_pattern_count,
        evidence,
    }
}

fn overview_test_root() -> std::path::PathBuf {
    nn_rust_inference::runtime::default_artifact_dir()
        .join("test-decompile-overview")
        .join(format!(
            "{}-{}",
            process::id(),
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        ))
}
