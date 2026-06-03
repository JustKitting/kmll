use super::*;

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
        /*0080*/                   FMUL R23, R23, R32 ;                          /* 0x0 */
        /*0090*/                   HADD2 R5, R6.H0_H0, R5.H0_H0 ;                /* 0x0 */
        /*00a0*/                   HMUL2 R5, R5.H0_H0, 0.5, 0.5 ;                /* 0x0 */
        /*00b0*/               @P0 BRA `(.L_x_1) ;                               /* 0x0 */
.L_x_0:
        /*00c0*/                   BSYNC B0 ;                                    /* 0x0 */
        /*00d0*/                   CALL.REL.NOINC `($helper) ;                   /* 0x0 */
$helper:
        /*00e0*/                   SHFL.DOWN PT, R5, R22, 0x10, 0x1f ;           /* 0x0 */
        /*00f0*/                   RET.REL.NODEC R4 `(matvec_bf16_rows17) ;       /* 0x0 */
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
