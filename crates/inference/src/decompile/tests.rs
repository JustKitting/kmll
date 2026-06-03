use super::*;
use std::{env, fs, process, time::SystemTime};

const SIMPLE_SASS: &str = r#"
        .target sm_120

        .section .text.sass_fixture_i32_add,"ax",@progbits
        .global sass_fixture_i32_add
sass_fixture_i32_add:
.text.sass_fixture_i32_add:
        /*0000*/                   S2R R0, SR_TID.X ;                            /* 0x0 */
        /*0010*/                   LD.E R2, desc[UR4][R0.64] ;                   /* 0x0 */
        /*0020*/                   LD.E R3, desc[UR6][R0.64+0x4] ;               /* 0x0 */
        /*0030*/                   IADD R4, R2, R3 ;                             /* 0x0 */
        /*0040*/                   ST.E desc[UR8][R0.64], R4 ;                   /* 0x0 */
        /*0050*/                   PRMT R5, R4, 0x7610, RZ ;                     /* 0x0 */
        /*0060*/                   EXIT ;                                        /* 0x0 */
.L_x_0:
        /*0070*/                   BRA `(.L_x_0);                                /* 0x0 */
"#;

const ROWS17_SLICE: &str = r#"
        .target sm_120

        .section .text.matvec_bf16_rows17,"ax",@progbits
        .global matvec_bf16_rows17
matvec_bf16_rows17:
.text.matvec_bf16_rows17:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                       /* 0x0 */
        /*0010*/                   S2R R2, SR_TID.X ;                            /* 0x0 */
        /*0020*/                   SHF.R.U32.HI R2, RZ, 0x5, R2 ;                /* 0x0 */
        /*0030*/                   ISETP.GT.U32.AND P0, PT, R2, 0x10, PT ;       /* 0x0 */
        /*0040*/               @P0 EXIT ;                                        /* 0x0 */
        /*0050*/                   BSSY.RECONVERGENT B0, `(.L_x_0) ;             /* 0x0 */
        /*0060*/                   LD.E.U16 R23, desc[UR10][R24.64] ;            /* 0x0 */
        /*0070*/                   HFMA2 R3, -RZ, RZ, 0, 0 ;                     /* 0x0 */
        /*0080*/                   IMAD.U32 R23, R23, 0x10000, RZ ;              /* 0x0 */
        /*0090*/                   FMUL R23, R23, R32 ;                          /* 0x0 */
        /*00a0*/                   FADD R22, R23, R22 ;                          /* 0x0 */
        /*00b0*/                   HADD2 R5, R6.H0_H0, R5.H0_H0 ;                /* 0x0 */
        /*00c0*/                   HMUL2 R5, R5.H0_H0, 0.5, 0.5 ;                /* 0x0 */
        /*00d0*/               @P0 BRA `(.L_x_1) ;                               /* 0x0 */
.L_x_0:
        /*00e0*/                   BSYNC B0 ;                                    /* 0x0 */
        /*00f0*/                   CALL.REL.NOINC `($helper) ;                   /* 0x0 */
$helper:
        /*0100*/                   SHFL.DOWN PT, R5, R22, 0x10, 0x1f ;           /* 0x0 */
        /*0110*/                   FADD R6, R5, R22 ;                            /* 0x0 */
        /*0120*/                   SHFL.DOWN PT, R5, R6, 0x8, 0x1f ;             /* 0x0 */
        /*0130*/                   FADD R7, R6, R5 ;                             /* 0x0 */
        /*0140*/                   RET.REL.NODEC R4 `(matvec_bf16_rows17) ;       /* 0x0 */
"#;

const UNSUPPORTED_SASS: &str = r#"
        .target sm_120

        .section .text.unsupported_fixture,"ax",@progbits
        .global unsupported_fixture
unsupported_fixture:
.text.unsupported_fixture:
        /*0000*/                   MYSTERY R0, R1 ;                             /* 0x0 */
        /*0010*/                   EXIT ;                                        /* 0x0 */
"#;

const CFG_SASS: &str = r#"
        .target sm_120

        .section .text.cfg_fixture,"ax",@progbits
        .global cfg_fixture
cfg_fixture:
.text.cfg_fixture:
        /*0000*/                   ISETP.GT.U32.AND P0, PT, R0, R1, PT ;        /* 0x0 */
        /*0010*/               @P0 BRA `(.L_then) ;                              /* 0x0 */
        /*0020*/                   IADD R2, R0, R1 ;                             /* 0x0 */
.L_then:
        /*0030*/                   IADD R3, R2, R1 ;                             /* 0x0 */
        /*0040*/                   EXIT ;                                        /* 0x0 */
"#;

const PREDICATED_EXIT_SASS: &str = r#"
        .target sm_120

        .section .text.predicated_exit_fixture,"ax",@progbits
        .global predicated_exit_fixture
predicated_exit_fixture:
.text.predicated_exit_fixture:
        /*0000*/                   ISETP.GT.U32.AND P0, PT, R0, R1, PT ;        /* 0x0 */
        /*0010*/               @P0 EXIT ;                                        /* 0x0 */
        /*0020*/                   IADD R2, R0, R1 ;                             /* 0x0 */
        /*0030*/                   EXIT ;                                        /* 0x0 */
"#;

#[test]
fn parse_nvdisasm_sass_captures_function_and_operands() {
    let module = parse_nvdisasm_sass(SIMPLE_SASS).expect("fixture SASS should parse");

    assert_eq!(module.target.as_deref(), Some("sm_120"));
    assert_eq!(module.functions.len(), 1);
    let function = &module.functions[0];
    assert_eq!(function.name, "sass_fixture_i32_add");
    assert_eq!(function.instructions.len(), 8);
    assert_eq!(function.instructions[0].opcode, "S2R");
    assert_eq!(function.instructions[2].modifiers, ["E"]);
    assert_eq!(
        function.instructions[2].operands[1].kind,
        SassOperandKind::DescriptorMemory {
            descriptor: "UR6".to_string(),
            address: "R0".to_string(),
            address_width: Some(64),
            offset: Some("0x4".to_string())
        }
    );
    assert_eq!(function.instructions[7].label.as_deref(), Some(".L_x_0"));
}

#[test]
fn lower_simple_sass_maps_observed_core_ops() {
    let module = parse_nvdisasm_sass(SIMPLE_SASS).expect("fixture SASS should parse");
    let ir = lower_sass_module(&module);
    let kinds = ir.functions[0]
        .ops
        .iter()
        .map(|op| &op.kind)
        .collect::<Vec<_>>();

    assert!(matches!(
        kinds[0],
        KernelIrOpKind::ReadSpecialRegister { dst, special }
            if dst == "R0" && special == "SR_TID.X"
    ));
    assert!(matches!(kinds[1], KernelIrOpKind::Load { .. }));
    assert!(matches!(
        kinds[3],
        KernelIrOpKind::IntegerAdd {
            dst,
            inputs,
            width_bits: None
        } if dst == "R4" && inputs == &vec!["R2".to_string(), "R3".to_string()]
    ));
    assert!(matches!(kinds[4], KernelIrOpKind::Store { .. }));
    assert!(matches!(
        kinds[5],
        KernelIrOpKind::Permute { dst, inputs } if dst == "R5" && inputs.len() == 3
    ));
    assert!(matches!(kinds[6], KernelIrOpKind::Exit { condition: None }));
    assert!(matches!(
        kinds[7],
        KernelIrOpKind::Branch {
            target: Some(_),
            condition: None
        }
    ));
    assert_eq!(ir.unsupported_instruction_count(), 0);
}

#[test]
fn lower_rows17_slice_keeps_predicates_and_half_fma_visible() {
    let module = parse_nvdisasm_sass(ROWS17_SLICE).expect("rows17 slice should parse");
    let ir = lower_sass_module(&module);
    let ops = &ir.functions[0].ops;

    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::Exit {
            condition: Some(ref condition)
        } if condition == "P0"
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::FusedMultiplyAdd {
            lane_bits: Some(16),
            ..
        }
    )));
    assert!(
        ops.iter()
            .any(|op| matches!(op.kind, KernelIrOpKind::PackedHalfAdd { lanes: 2, .. }))
    );
    assert!(
        ops.iter()
            .any(|op| matches!(op.kind, KernelIrOpKind::PackedHalfMul { lanes: 2, .. }))
    );
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::Branch {
            target: Some(ref target),
            condition: Some(ref condition)
        } if target == ".L_x_1" && condition == "P0"
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::Call {
            target: Some(ref target),
            ..
        } if target == "$helper"
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::WarpShuffle {
            mode: Some(ref mode),
            offset: ref shuffle_offset,
            ..
        } if mode == "DOWN" && shuffle_offset == "0x10"
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::Return {
            target: Some(ref target),
            ..
        } if target == "matvec_bf16_rows17"
    )));
    assert_eq!(ir.unsupported_instruction_count(), 0);
}

#[test]
fn semantic_patterns_recover_bf16_widen_and_warp_reduce() {
    let module = parse_nvdisasm_sass(ROWS17_SLICE).expect("rows17 slice should parse");
    let ir = lower_sass_module(&module);
    let patterns = recover_sass_patterns(&ir);
    let flat = patterns
        .functions
        .iter()
        .flat_map(|function| function.patterns.iter())
        .collect::<Vec<_>>();

    assert!(flat.iter().any(|pattern| matches!(
        pattern.kind,
        SassSemanticPatternKind::Bf16WidenBits {
            ref src,
            ref dst,
            ..
        } if src == "R23" && dst == "R23"
    )));
    assert!(flat.iter().any(|pattern| matches!(
        pattern.kind,
        SassSemanticPatternKind::F32MulAddPair {
            ref mul_dst,
            ref add_dst,
            ..
        } if mul_dst == "R23" && add_dst == "R22"
    )));
    assert!(flat.iter().any(|pattern| matches!(
        pattern.kind,
        SassSemanticPatternKind::WarpReduceSum {
            ref input,
            ref output,
            ref offsets,
            ..
        } if input == "R22" && output == "R7" && offsets == &vec!["0x10".to_string(), "0x8".to_string()]
    )));
    let text = patterns.to_text();
    assert!(text.contains("bf16-widen-bits"));
    assert!(text.contains("warp-reduce-sum"));
}

#[test]
fn analysis_recovers_cfg_edges_and_register_dataflow() {
    let module = parse_nvdisasm_sass(CFG_SASS).expect("cfg fixture should parse");
    let ir = lower_sass_module(&module);
    let analysis = analyze_sass_ir(&ir);
    let function = &analysis.functions[0];

    assert_eq!(function.blocks.len(), 3);
    assert!(function.blocks.iter().any(|block| {
        block.id == 0
            && block.start_address == 0x0
            && block.end_address == 0x10
            && block.terminator == SassBlockTerminator::Branch
    }));
    assert!(function.edges.iter().any(|edge| {
        edge.from_block == 0
            && edge.to_block == Some(2)
            && edge.kind == SassCfgEdgeKind::Branch
            && edge.condition.as_deref() == Some("P0")
            && edge.target.as_deref() == Some(".L_then")
    }));
    assert!(function.edges.iter().any(|edge| {
        edge.from_block == 0
            && edge.to_block == Some(1)
            && edge.kind == SassCfgEdgeKind::Fallthrough
    }));
    assert!(function.edges.iter().any(|edge| {
        edge.from_block == 1
            && edge.to_block == Some(2)
            && edge.kind == SassCfgEdgeKind::Fallthrough
    }));
    assert!(function.edges.iter().any(|edge| {
        edge.from_block == 2 && edge.to_block.is_none() && edge.kind == SassCfgEdgeKind::Exit
    }));

    let compare = function
        .dataflow
        .iter()
        .find(|op| op.address == 0x0)
        .expect("compare dataflow should exist");
    assert_eq!(compare.defines, ["P0"]);
    assert_eq!(compare.uses, ["R0", "R1"]);
    let add = function
        .dataflow
        .iter()
        .find(|op| op.address == 0x20)
        .expect("add dataflow should exist");
    assert_eq!(add.defines, ["R2"]);
    assert_eq!(add.uses, ["R0", "R1"]);

    let text = analysis.to_text();
    assert!(text.contains("b0 -> b2 [branch condition=P0 target=.L_then]"));
    assert!(text.contains("0x0020: def=[R2] use=[R0,R1]"));
}

#[test]
fn analysis_keeps_fallthrough_after_predicated_exit() {
    let module =
        parse_nvdisasm_sass(PREDICATED_EXIT_SASS).expect("predicated exit fixture should parse");
    let ir = lower_sass_module(&module);
    let analysis = analyze_sass_ir(&ir);
    let function = &analysis.functions[0];

    assert_eq!(function.blocks.len(), 2);
    assert!(function.edges.iter().any(|edge| {
        edge.from_block == 0
            && edge.to_block.is_none()
            && edge.kind == SassCfgEdgeKind::Exit
            && edge.condition.as_deref() == Some("P0")
    }));
    assert!(function.edges.iter().any(|edge| {
        edge.from_block == 0
            && edge.to_block == Some(1)
            && edge.kind == SassCfgEdgeKind::Fallthrough
    }));
}

#[test]
fn side_by_side_dump_contains_source_sass_and_ir_sections() {
    let fixture = simple_kernel_fixtures()
        .into_iter()
        .find(|fixture| fixture.kind == SimpleKernelFixtureKind::I32Add)
        .expect("i32 add fixture should exist");
    let module = parse_nvdisasm_sass(SIMPLE_SASS).expect("fixture SASS should parse");
    let ir = lower_sass_module(&module);
    let dump = render_side_by_side(&fixture, SIMPLE_SASS, &ir);

    assert!(dump.contains("## source"));
    assert!(dump.contains("## sass"));
    assert!(dump.contains("## project-ir"));
    assert!(dump.contains("IntegerAdd"));
}

#[test]
fn coverage_scan_reports_opcode_counts_and_unsupported_instructions() {
    let root = unique_test_dir("coverage");
    let input = root.join("input");
    let nested = input.join("nested");
    let output = root.join("output");
    fs::create_dir_all(&nested).expect("test input dir should be created");
    fs::write(input.join("simple.nvdisasm.sass"), SIMPLE_SASS)
        .expect("simple test SASS should be written");
    fs::write(nested.join("unsupported.nvdisasm.sass"), UNSUPPORTED_SASS)
        .expect("unsupported test SASS should be written");

    let report = run_sass_coverage_scan(&SassCoverageOptions {
        root: input,
        output_dir: output,
    })
    .expect("coverage scan should complete");

    assert_eq!(report.files.len(), 2);
    assert_eq!(report.parsed_file_count, 2);
    assert_eq!(report.parse_error_count, 0);
    assert_eq!(report.unsupported_instruction_count, 1);
    assert!(
        report
            .opcode_counts
            .iter()
            .any(|count| count.opcode == "MYSTERY" && count.count == 1)
    );
    assert!(
        report
            .unsupported_instructions
            .iter()
            .any(|instruction| instruction.opcode == "MYSTERY")
    );
    assert!(report.summary_path.exists());
    assert!(report.files_path.exists());
    assert!(report.opcode_frequency_path.exists());
    assert!(report.opcode_signature_frequency_path.exists());
    assert!(report.semantic_patterns_path.exists());
    assert!(report.semantic_pattern_frequency_path.exists());
    assert!(report.cfg_blocks_path.exists());
    assert!(report.cfg_edges_path.exists());
    assert!(report.dataflow_path.exists());
    assert!(report.unsupported_instructions_path.exists());
    assert!(report.cfg_block_count > 0);
    assert!(report.cfg_edge_count > 0);
    assert!(report.dataflow_op_count > 0);
    assert!(
        report
            .files
            .iter()
            .filter_map(|file| file.ir_path.as_ref())
            .all(|path| path.exists())
    );
    assert!(
        report
            .files
            .iter()
            .filter_map(|file| file.analysis_path.as_ref())
            .all(|path| path.exists())
    );
    assert!(
        report
            .files
            .iter()
            .filter_map(|file| file.pattern_path.as_ref())
            .all(|path| path.exists())
    );
}

fn unique_test_dir(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    env::temp_dir().join(format!(
        "nn_rust_decompile_{name}_{}_{}",
        process::id(),
        nanos
    ))
}
