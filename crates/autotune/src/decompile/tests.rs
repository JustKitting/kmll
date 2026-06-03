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

const CUOBJDUMP_SASS: &str = r#"
	code for sm_120
	.target	sm_120

		Function : cuobjdump_fixture
	.headerflags	@"EF_CUDA_SM120"
        /*0000*/                   S2R R0, SR_TID.X ;                            /* 0x0 */
        /*0010*/                   IADD R1, R0, 0x1 ;                            /* 0x0 */
        /*0020*/                   EXIT ;                                        /* 0x0 */
"#;

const CUOBJDUMP_BRANCH_SASS: &str = r#"
	code for sm_120
	.target	sm_120

		Function : cuobjdump_branch_fixture
	.headerflags	@"EF_CUDA_SM120"
        /*0000*/                   MOV R0, RZ ;                                  /* 0x0 */
        /*0010*/                   IADD R0, R0, 0x1 ;                            /* 0x0 */
        /*0020*/                   BRA 0x10 ;                                    /* 0x0 */
        /*0030*/                   EXIT ;                                        /* 0x0 */
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

const PREDICATED_WRITE_SASS: &str = r#"
        .target sm_120

        .section .text.predicated_write_fixture,"ax",@progbits
        .global predicated_write_fixture
predicated_write_fixture:
.text.predicated_write_fixture:
        /*0000*/                   IADD R2, R0, R1 ;                             /* 0x0 */
        /*0010*/                   ISETP.GT.U32.AND P0, PT, R0, R1, PT ;         /* 0x0 */
        /*0020*/               @P0 IADD R2, R2, R1 ;                             /* 0x0 */
        /*0030*/                   IADD R3, R2, R1 ;                             /* 0x0 */
        /*0040*/                   EXIT ;                                        /* 0x0 */
"#;

const LOOP_SASS: &str = r#"
        .target sm_120

        .section .text.loop_fixture,"ax",@progbits
        .global loop_fixture
loop_fixture:
.text.loop_fixture:
        /*0000*/                   MOV R2, RZ ;                                  /* 0x0 */
.L_loop:
        /*0010*/                   IADD R2, R2, 0x1 ;                            /* 0x0 */
        /*0020*/                   ISETP.LT.U32.AND P0, PT, R2, R0, PT ;         /* 0x0 */
        /*0030*/              @!P0 BRA `(.L_done) ;                              /* 0x0 */
        /*0040*/                   IADD R3, R2, R1 ;                             /* 0x0 */
        /*0050*/                   BRA `(.L_loop) ;                              /* 0x0 */
.L_done:
        /*0060*/                   EXIT ;                                        /* 0x0 */
"#;

#[test]
fn parse_nvidia_sass_captures_nvdisasm_function_and_operands() {
    let module = parse_nvidia_sass(SIMPLE_SASS).expect("fixture SASS should parse");

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
fn parse_nvidia_sass_captures_cuobjdump_function_header() {
    let module = parse_nvidia_sass(CUOBJDUMP_SASS).expect("cuobjdump SASS should parse");

    assert_eq!(module.target.as_deref(), Some("sm_120"));
    assert_eq!(module.functions.len(), 1);
    let function = &module.functions[0];
    assert_eq!(function.name, "cuobjdump_fixture");
    assert_eq!(function.section.as_deref(), Some("cuobjdump_fixture"));
    assert_eq!(function.instructions.len(), 3);
    assert_eq!(function.instructions[0].opcode, "S2R");
    assert_eq!(function.instructions[1].opcode, "IADD");
    assert_eq!(function.instructions[2].opcode, "EXIT");
}

#[test]
fn analysis_resolves_cuobjdump_numeric_branch_targets() {
    let module = parse_nvidia_sass(CUOBJDUMP_BRANCH_SASS).expect("cuobjdump SASS should parse");
    let ir = lift_sass_module(&module);
    let branch = ir.functions[0]
        .ops
        .iter()
        .find(|op| op.address == 0x20)
        .expect("branch op should exist");
    assert!(matches!(
        &branch.kind,
        KernelIrOpKind::Branch {
            target: Some(target),
            condition: None
        } if target == "0x10"
    ));

    let analysis = analyze_sass_ir(&ir);
    let function = &analysis.functions[0];
    let loop_block = function
        .blocks
        .iter()
        .find(|block| block.start_address == 0x10)
        .expect("numeric branch target should start a block");
    assert!(function.edges.iter().any(|edge| {
        edge.kind == SassCfgEdgeKind::Branch
            && edge.from_block == loop_block.id
            && edge.to_block == Some(loop_block.id)
    }));
    assert!(function.natural_loops.iter().any(|natural_loop| {
        natural_loop.header_block == loop_block.id && natural_loop.latch_block == loop_block.id
    }));
}

#[test]
fn lift_simple_sass_maps_observed_core_ops() {
    let module = parse_nvidia_sass(SIMPLE_SASS).expect("fixture SASS should parse");
    let ir = lift_sass_module(&module);
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
fn analysis_recovers_structured_memory_accesses() {
    let module = parse_nvidia_sass(SIMPLE_SASS).expect("fixture SASS should parse");
    let ir = lift_sass_module(&module);
    let analysis = analyze_sass_ir(&ir);
    let function = &analysis.functions[0];

    assert_eq!(function.memory_accesses.len(), 3);
    let first_load = function
        .memory_accesses
        .iter()
        .find(|access| access.address == 0x10)
        .expect("first descriptor load should be recovered");
    assert_eq!(first_load.kind, SassMemoryAccessKind::Load);
    assert_eq!(first_load.space, MemorySpace::Descriptor);
    assert_eq!(first_load.value_register, "R2");
    assert_eq!(first_load.address_expr, "desc[UR4][R0.64]");
    assert_eq!(first_load.address_registers, ["UR4", "R0"]);
    assert_eq!(first_load.address_base.as_deref(), Some("UR4"));
    assert_eq!(first_load.offset, None);

    let offset_load = function
        .memory_accesses
        .iter()
        .find(|access| access.address == 0x20)
        .expect("offset descriptor load should be recovered");
    assert_eq!(offset_load.address_base.as_deref(), Some("UR6"));
    assert_eq!(offset_load.offset.as_deref(), Some("0x4"));

    let store = function
        .memory_accesses
        .iter()
        .find(|access| access.address == 0x40)
        .expect("descriptor store should be recovered");
    assert_eq!(store.kind, SassMemoryAccessKind::Store);
    assert_eq!(store.space, MemorySpace::Descriptor);
    assert_eq!(store.value_register, "R4");
    assert_eq!(store.address_registers, ["UR8", "R0"]);

    let text = analysis.to_text();
    assert!(text.contains("memory_accesses"));
    assert!(text.contains("0x0010: load descriptor value=R2 addr=desc[UR4][R0.64]"));
    assert!(text.contains("base=UR4"));
}

#[test]
fn lifted_value_ir_classifies_ops_and_keeps_ssa_refs() {
    let module = parse_nvidia_sass(SIMPLE_SASS).expect("fixture SASS should parse");
    let ir = lift_sass_module(&module);
    let analysis = analyze_sass_ir(&ir);
    let lifted = lift_sass_value_ir(&ir, &analysis);
    let function = &lifted.functions[0];

    assert_eq!(lifted.op_count(), ir.functions[0].ops.len());

    let load = function
        .ops
        .iter()
        .find(|op| op.address == 0x10)
        .expect("descriptor load should lift");
    assert_eq!(load.class, SassLiftedOpClass::Memory);
    assert_eq!(load.kind, SassLiftedOpKind::Load);
    assert!(matches!(
        &load.semantics,
        SassLiftedSemantics::Load {
            space: MemorySpace::Descriptor,
            dst,
            address
        } if dst == "R2" && address == "desc[UR4][R0.64]"
    ));
    assert!(load.outputs.iter().any(|value| value.register == "R2"));

    let add = function
        .ops
        .iter()
        .find(|op| op.address == 0x30)
        .expect("integer add should lift");
    assert_eq!(add.class, SassLiftedOpClass::IntegerMath);
    assert_eq!(add.kind, SassLiftedOpKind::IntegerAdd);
    assert!(matches!(
        &add.semantics,
        SassLiftedSemantics::IntegerAdd {
            dst,
            inputs,
            width_bits: None
        } if dst == "R4" && inputs == &vec!["R2".to_string(), "R3".to_string()]
    ));
    assert!(add.inputs.iter().any(|value| value.register == "R2"));
    assert!(add.inputs.iter().any(|value| value.register == "R3"));
    assert!(add.outputs.iter().any(|value| value.register == "R4"));

    let store = function
        .ops
        .iter()
        .find(|op| op.address == 0x40)
        .expect("descriptor store should lift");
    assert_eq!(store.class, SassLiftedOpClass::Memory);
    assert_eq!(store.kind, SassLiftedOpKind::Store);
    assert!(matches!(
        &store.semantics,
        SassLiftedSemantics::Store {
            space: MemorySpace::Descriptor,
            address,
            value
        } if address == "desc[UR8][R0.64]" && value == "R4"
    ));
    assert!(store.inputs.iter().any(|value| value.register == "R4"));
    assert!(store.outputs.is_empty());

    let exit = function
        .ops
        .iter()
        .find(|op| op.address == 0x60)
        .expect("exit should lift");
    assert_eq!(exit.class, SassLiftedOpClass::ControlFlow);
    assert_eq!(exit.kind, SassLiftedOpKind::Exit);
    assert!(matches!(
        &exit.semantics,
        SassLiftedSemantics::Exit { condition: None }
    ));

    let text = lifted.to_text();
    assert!(text.contains("lifted_value_ir"));
    assert!(text.contains("integer-math integer-add"));
    assert!(text.contains("memory load"));
    assert!(text.contains("semantics=load(space=descriptor,dst=R2,address=desc[UR4][R0.64])"));
}

#[test]
fn lift_rows17_slice_keeps_predicates_and_half_fma_visible() {
    let module = parse_nvidia_sass(ROWS17_SLICE).expect("rows17 slice should parse");
    let ir = lift_sass_module(&module);
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
    let module = parse_nvidia_sass(ROWS17_SLICE).expect("rows17 slice should parse");
    let ir = lift_sass_module(&module);
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
    let module = parse_nvidia_sass(CFG_SASS).expect("cfg fixture should parse");
    let ir = lift_sass_module(&module);
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

    let predicate_use = function
        .reaching_uses
        .iter()
        .find(|use_site| use_site.address == 0x10 && use_site.register == "P0")
        .expect("branch predicate use should have reaching definitions");
    assert!(!predicate_use.reaches_entry);
    assert_eq!(predicate_use.reaching_def_addresses.as_slice(), &[0x0]);

    let joined_r2_use = function
        .reaching_uses
        .iter()
        .find(|use_site| use_site.address == 0x30 && use_site.register == "R2")
        .expect("join use of R2 should have reaching definitions");
    assert!(joined_r2_use.reaches_entry);
    assert_eq!(joined_r2_use.reaching_def_addresses.as_slice(), &[0x20]);

    let local_r2_range = function
        .live_ranges
        .iter()
        .find(|range| range.register == "R2" && range.def_address == Some(0x20))
        .expect("R2 definition at 0x20 should have a live range");
    assert_eq!(local_r2_range.use_addresses.as_slice(), &[0x30]);

    let entry_r2_range = function
        .live_ranges
        .iter()
        .find(|range| range.register == "R2" && range.def_address.is_none())
        .expect("entry R2 should be live on the branch path");
    assert_eq!(entry_r2_range.start_address, 0x30);
    assert_eq!(entry_r2_range.end_address, 0x30);
    assert_eq!(entry_r2_range.use_addresses.as_slice(), &[0x30]);

    let entry_r2_value = function
        .ssa_values
        .iter()
        .find(|value| value.register == "R2" && value.def_address.is_none())
        .expect("entry R2 should have an SSA value");
    assert_eq!(entry_r2_value.use_addresses.as_slice(), &[0x30]);

    let local_r2_value = function
        .ssa_values
        .iter()
        .find(|value| value.register == "R2" && value.def_address == Some(0x20))
        .expect("local R2 definition should have an SSA value");
    assert_eq!(local_r2_value.use_addresses.as_slice(), &[0x30]);

    let joined_r2_edges = function
        .def_use_edges
        .iter()
        .filter(|edge| edge.use_address == 0x30 && edge.register == "R2")
        .collect::<Vec<_>>();
    assert_eq!(joined_r2_edges.len(), 2);
    assert!(
        joined_r2_edges
            .iter()
            .any(|edge| edge.value_id == entry_r2_value.value_id && edge.def_address.is_none())
    );
    assert!(joined_r2_edges.iter().any(|edge| {
        edge.value_id == local_r2_value.value_id && edge.def_address == Some(0x20)
    }));

    let joined_op = function
        .value_ops
        .iter()
        .find(|op| op.address == 0x30)
        .expect("join instruction should have a value-op row");
    assert_eq!(joined_op.opcode, "IADD");
    assert_eq!(joined_op.input_registers, ["R2", "R1"]);
    assert_eq!(joined_op.output_registers, ["R3"]);
    assert!(joined_op.input_value_ids.contains(&entry_r2_value.value_id));
    assert!(joined_op.input_value_ids.contains(&local_r2_value.value_id));
    let r3_value = function
        .ssa_values
        .iter()
        .find(|value| value.register == "R3" && value.def_address == Some(0x30))
        .expect("R3 output should have an SSA value");
    assert_eq!(joined_op.output_value_ids.as_slice(), &[r3_value.value_id]);

    let text = analysis.to_text();
    assert!(text.contains("b0 -> b2 [branch condition=P0 target=.L_then]"));
    assert!(text.contains("0x0020: def=[R2] use=[R0,R1]"));
    assert!(text.contains("0x0030: R2 <- [entry,0x0020]"));
    assert!(text.contains("ssa_values"));
    assert!(text.contains("def_use_edges"));
    assert!(text.contains("value_ops"));
    assert!(text.contains("R2@entry 0x0030-0x0030 uses=[0x0030]"));
}

#[test]
fn analysis_recovers_dominators_and_natural_loops() {
    let module = parse_nvidia_sass(LOOP_SASS).expect("loop fixture should parse");
    let ir = lift_sass_module(&module);
    let analysis = analyze_sass_ir(&ir);
    let function = &analysis.functions[0];

    assert_eq!(function.blocks.len(), 4);
    assert_eq!(function.dominators.len(), function.blocks.len());

    let header = function
        .blocks
        .iter()
        .find(|block| block.label.as_deref() == Some(".L_loop"))
        .expect("loop header block should exist");
    let body = function
        .blocks
        .iter()
        .find(|block| block.start_address == 0x40)
        .expect("loop body block should exist");
    let done = function
        .blocks
        .iter()
        .find(|block| block.label.as_deref() == Some(".L_done"))
        .expect("loop exit block should exist");

    let header_dom = function
        .dominators
        .iter()
        .find(|dominator| dominator.block_id == header.id)
        .expect("header dominator row should exist");
    assert!(header_dom.reachable);
    assert_eq!(header_dom.immediate_dominator, Some(0));
    assert_eq!(header_dom.dominators.as_slice(), &[0, header.id]);

    let body_dom = function
        .dominators
        .iter()
        .find(|dominator| dominator.block_id == body.id)
        .expect("body dominator row should exist");
    assert_eq!(body_dom.immediate_dominator, Some(header.id));
    assert_eq!(body_dom.dominators.as_slice(), &[0, header.id, body.id]);

    let done_dom = function
        .dominators
        .iter()
        .find(|dominator| dominator.block_id == done.id)
        .expect("done dominator row should exist");
    assert_eq!(done_dom.immediate_dominator, Some(header.id));

    assert_eq!(function.natural_loops.len(), 1);
    let natural_loop = &function.natural_loops[0];
    assert!(natural_loop.reachable);
    assert_eq!(natural_loop.header_block, header.id);
    assert_eq!(natural_loop.latch_block, body.id);
    assert_eq!(natural_loop.blocks.as_slice(), &[header.id, body.id]);
    assert_eq!(natural_loop.edge_target.as_deref(), Some(".L_loop"));

    let text = analysis.to_text();
    assert!(text.contains("dominators"));
    assert!(text.contains("natural_loops"));
    assert!(text.contains("header=b1 latch=b2 reachable=true blocks=[b1,b2]"));
}

#[test]
fn analysis_keeps_fallthrough_after_predicated_exit() {
    let module =
        parse_nvidia_sass(PREDICATED_EXIT_SASS).expect("predicated exit fixture should parse");
    let ir = lift_sass_module(&module);
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
fn analysis_keeps_previous_definition_after_predicated_write() {
    let module =
        parse_nvidia_sass(PREDICATED_WRITE_SASS).expect("predicated write fixture should parse");
    let ir = lift_sass_module(&module);
    let analysis = analyze_sass_ir(&ir);
    let function = &analysis.functions[0];

    let predicated_write_use = function
        .reaching_uses
        .iter()
        .find(|use_site| use_site.address == 0x20 && use_site.register == "R2")
        .expect("predicated write should use the previous R2 value");
    assert!(!predicated_write_use.reaches_entry);
    assert_eq!(
        predicated_write_use.reaching_def_addresses.as_slice(),
        &[0x0]
    );

    let post_write_use = function
        .reaching_uses
        .iter()
        .find(|use_site| use_site.address == 0x30 && use_site.register == "R2")
        .expect("post-write R2 use should have reaching definitions");
    assert!(!post_write_use.reaches_entry);
    assert_eq!(
        post_write_use.reaching_def_addresses.as_slice(),
        &[0x0, 0x20]
    );

    let original_r2_range = function
        .live_ranges
        .iter()
        .find(|range| range.register == "R2" && range.def_address == Some(0x0))
        .expect("original R2 definition should remain live after predicated write");
    assert_eq!(original_r2_range.use_addresses.as_slice(), &[0x20, 0x30]);

    let predicated_r2_range = function
        .live_ranges
        .iter()
        .find(|range| range.register == "R2" && range.def_address == Some(0x20))
        .expect("predicated R2 definition should get its own live range");
    assert_eq!(predicated_r2_range.use_addresses.as_slice(), &[0x30]);
}

#[test]
fn side_by_side_dump_contains_source_sass_and_ir_sections() {
    let fixture = simple_kernel_fixtures()
        .into_iter()
        .find(|fixture| fixture.kind == SimpleKernelFixtureKind::I32Add)
        .expect("i32 add fixture should exist");
    let module = parse_nvidia_sass(SIMPLE_SASS).expect("fixture SASS should parse");
    let ir = lift_sass_module(&module);
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
    fs::write(input.join("simple.sass"), SIMPLE_SASS).expect("simple test SASS should be written");
    fs::write(input.join("simple.cuobjdump.sass"), CUOBJDUMP_SASS)
        .expect("cuobjdump test SASS should be written");
    fs::write(nested.join("unsupported.sass"), UNSUPPORTED_SASS)
        .expect("unsupported test SASS should be written");

    let report = run_sass_coverage_scan(&SassCoverageOptions {
        root: input,
        output_dir: output,
    })
    .expect("coverage scan should complete");

    assert_eq!(report.files.len(), 3);
    assert_eq!(report.parsed_file_count, 3);
    assert_eq!(report.parse_error_count, 0);
    assert_eq!(report.unsupported_instruction_count, 1);
    assert!(
        report
            .opcode_counts
            .iter()
            .any(|count| count.opcode == "MYSTERY" && count.count == 1)
    );
    let mystery_catalog = report
        .opcode_catalog
        .iter()
        .find(|entry| entry.opcode == "MYSTERY")
        .expect("unsupported opcode should be catalogued");
    assert_eq!(mystery_catalog.support, "unsupported");
    assert_eq!(mystery_catalog.unsupported_count, 1);
    assert!(
        mystery_catalog
            .kinds
            .iter()
            .any(|kind| kind == "unsupported")
    );
    let iadd_catalog = report
        .opcode_catalog
        .iter()
        .find(|entry| entry.opcode == "IADD")
        .expect("IADD should be catalogued");
    assert_eq!(iadd_catalog.support, "mapped");
    assert!(
        iadd_catalog
            .source_formats
            .iter()
            .any(|format| format == "cuobjdump")
    );
    assert!(
        iadd_catalog
            .source_formats
            .iter()
            .any(|format| format == "sass")
    );
    assert!(
        report
            .unsupported_instructions
            .iter()
            .any(|instruction| instruction.opcode == "MYSTERY")
    );
    assert!(report.files.iter().any(
        |file| file.sass_path.file_name().and_then(|name| name.to_str())
            == Some("simple.cuobjdump.sass")
    ));
    assert!(report.summary_path.exists());
    assert!(report.files_path.exists());
    assert!(report.opcode_catalog_path.exists());
    assert!(report.opcode_frequency_path.exists());
    assert!(report.opcode_signature_frequency_path.exists());
    assert!(report.semantic_patterns_path.exists());
    assert!(report.semantic_pattern_frequency_path.exists());
    assert!(report.cfg_blocks_path.exists());
    assert!(report.cfg_edges_path.exists());
    assert!(report.dominators_path.exists());
    assert!(report.natural_loops_path.exists());
    assert!(report.dataflow_path.exists());
    assert!(report.reaching_uses_path.exists());
    assert!(report.ssa_values_path.exists());
    assert!(report.def_use_edges_path.exists());
    assert!(report.value_ops_path.exists());
    assert!(report.lifted_ops_path.exists());
    assert!(report.live_ranges_path.exists());
    assert!(report.memory_accesses_path.exists());
    assert!(report.unsupported_instructions_path.exists());
    assert!(report.cfg_block_count > 0);
    assert!(report.cfg_edge_count > 0);
    assert!(report.dominator_block_count > 0);
    assert!(report.natural_loop_count > 0);
    assert!(report.dataflow_op_count > 0);
    assert!(report.reaching_use_count > 0);
    assert!(report.ssa_value_count > 0);
    assert!(report.def_use_edge_count > 0);
    assert!(report.value_op_count > 0);
    assert!(report.lifted_op_count > 0);
    assert!(report.live_range_count > 0);
    assert!(report.memory_access_count > 0);
    assert!(report.memory_accesses.iter().any(|access| {
        access.kind == "load"
            && access.space == "descriptor"
            && access.address_base.as_deref() == Some("UR4")
    }));
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
            .filter_map(|file| file.lifted_ir_path.as_ref())
            .all(|path| path.exists())
    );
    assert!(
        report
            .lifted_ops
            .iter()
            .any(|op| op.class == "memory" && op.kind == "load")
    );
    assert!(report.lifted_ops.iter().any(|op| {
        op.class == "memory"
            && op.kind == "load"
            && op
                .semantics
                .contains("load(space=descriptor,dst=R2,address=desc[UR4][R0.64])")
    }));
    let lifted_ops_tsv =
        fs::read_to_string(&report.lifted_ops_path).expect("lifted ops TSV should be readable");
    assert!(lifted_ops_tsv.starts_with(
        "sass_path\tfunction\taddress\tblock_id\tpredicate\topcode\tclass\tkind\tsemantics"
    ));
    let opcode_catalog_tsv =
        fs::read_to_string(&report.opcode_catalog_path).expect("opcode catalog TSV should read");
    assert!(opcode_catalog_tsv.starts_with(
        "opcode\tinstruction_count\tsignature_count\tsignatures\tsource_formats\tclasses\tkinds\tsupport\tunsupported_count"
    ));
    assert!(opcode_catalog_tsv.contains("MYSTERY"));
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
