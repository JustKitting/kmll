use super::*;
use crate::autotune::{
    AutoOptimizeConfig, InferenceKernelRustCudaGenerator, KernelArtifactStore,
    compile_standalone_kernel_crate, generate_inference_kernel_source,
};
use nn_rust_profiling::{NumericKind, OperationKind};
use std::{collections::BTreeSet, env, fs, path::Path, process, time::SystemTime};

fn reg(raw: &str) -> RegisterRef {
    RegisterRef::parse(raw.to_string())
}

fn scalar(raw: &str) -> ScalarOperand {
    ScalarOperand::parse(raw.to_string())
}

fn aggregate_texts(operands: &[AggregateOperand]) -> Vec<String> {
    operands.iter().map(ToString::to_string).collect()
}

fn is_predicate_register(condition: &Option<PredicateCondition>, expected: &str) -> bool {
    matches!(
        condition,
        Some(PredicateCondition {
            kind: PredicateConditionKind::Register {
                register,
                negated: false
            },
            ..
        }) if register == &reg(expected)
    )
}

fn is_label_target(target: &Option<ControlTarget>, expected: &str) -> bool {
    matches!(
        target,
        Some(ControlTarget {
            kind: ControlTargetKind::Label(label),
            ..
        }) if label.as_str() == expected
    )
}

fn decompile_autotune_test_root() -> std::path::PathBuf {
    nn_rust_inference::runtime::default_artifact_dir()
        .join("test-decompile-autotune")
        .join(format!(
            "{}-{}",
            process::id(),
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        ))
}

fn cleanup_decompile_autotune_test_root(root: &Path) {
    fs::remove_dir_all(root).expect("test decompile autotune artifact directory should clean up");
    if let Some(parent) = root.parent() {
        let _ = fs::remove_dir(parent);
    }
}

#[test]
fn known_tensor_memory_catalog_targets_documented_ldtm_sttm_shapes() {
    let known_opcodes = known_sass_opcodes()
        .iter()
        .map(|known| known.opcode.clone())
        .collect::<BTreeSet<_>>();

    assert!(known_opcodes.contains(&SassOpcodeKind::Ldtm));
    assert!(known_opcodes.contains(&SassOpcodeKind::Sttm));
    assert!(!known_opcodes.contains(&SassOpcodeKind::Ldt));
    assert!(!known_opcodes.contains(&SassOpcodeKind::Stt));
}

#[test]
fn raw_aggregate_operands_capture_label_candidates_once() {
    let operand = SassOperand {
        raw: "R4 `(matvec_bf16_rows17)".to_string(),
        kind: SassOperandKind::Raw,
    };
    let aggregate = AggregateOperand::from_sass_operand(&operand);

    assert!(matches!(
        &aggregate.kind,
        AggregateOperandKind::Raw {
            registers,
            label: Some(label),
        } if registers.as_slice() == [reg("R4")]
            && label.as_str() == "matvec_bf16_rows17"
    ));
    assert_eq!(aggregate.single_register(), Some(&reg("R4")));
    assert!(matches!(
        aggregate.as_scalar_operand().kind,
        ScalarOperandKind::Raw
    ));
}

#[test]
fn raw_single_register_aggregate_becomes_typed_scalar() {
    let operand = SassOperand {
        raw: "UR4".to_string(),
        kind: SassOperandKind::Raw,
    };
    let aggregate = AggregateOperand::from_sass_operand(&operand);

    assert!(matches!(
        aggregate.as_scalar_operand().kind,
        ScalarOperandKind::Register(register) if register == reg("UR4")
    ));
}

#[test]
fn register_refs_canonicalize_modifier_spelling_for_identity() {
    assert_eq!(reg("R13.reuse"), reg("R13"));
    assert_eq!(reg("-RZ"), reg("RZ"));
    assert_ne!(reg("URZ"), reg("RZ"));
    assert_eq!(reg("R13.reuse").to_string(), "R13");
    assert!(reg("-RZ").negated);
    assert!(reg("|R2|").absolute);
}

#[test]
fn hex_branch_targets_with_exponent_digits_stay_typed() {
    const HEX_BRANCH_SASS: &str = r#"
        .target sm_120

        .section .text.hex_branch_fixture,"ax",@progbits
        .global hex_branch_fixture
hex_branch_fixture:
.text.hex_branch_fixture:
        /*0000*/                   BRA 0xe0 ;                                      /* 0x0 */
        /*0010*/                   EXIT ;                                          /* 0x0 */
"#;

    let module = parse_nvidia_sass(HEX_BRANCH_SASS).expect("hex branch SASS should parse");
    let ir = lift_sass_module(&module);
    let branch = &ir.functions[0].ops[0];

    assert!(matches!(
        &branch.source_operands[0].kind,
        AggregateOperandKind::Immediate(ImmediateValue::Integer(value)) if *value == 0xe0
    ));
    assert!(matches!(
        &branch.kind,
        KernelIrOpKind::Branch {
            target: Some(ControlTarget {
                kind: ControlTargetKind::Address(0xe0),
                ..
            }),
            ..
        }
    ));
}

#[test]
fn sass_architecture_parses_sm_spellings_and_displays_canonical_form() {
    assert_eq!(
        SassArchitecture::parse("sm120"),
        Some(SassArchitecture::sm(120))
    );
    assert_eq!(
        SassArchitecture::parse("sm_120"),
        Some(SassArchitecture::sm(120))
    );
    assert_eq!(
        SassArchitecture::parse("sm_90a"),
        Some(SassArchitecture::sm_a(90))
    );
    assert_eq!(
        SassArchitecture::parse("sm90a"),
        Some(SassArchitecture::sm_a(90))
    );
    assert_eq!(SassArchitecture::sm(120).sm_number(), 120);
    assert_eq!(SassArchitecture::sm(120).to_string(), "sm120");
    assert_eq!(SassArchitecture::sm_a(90).sm_number(), 90);
    assert_eq!(
        SassArchitecture::sm_a(90).suffix(),
        Some(SassArchitectureSuffix::A)
    );
    assert_eq!(SassArchitecture::sm_a(90).to_string(), "sm90a");
}

#[test]
fn unsupported_reasons_are_typed() {
    const BAD_ARITY_SASS: &str = r#"
        .target sm_120

        .section .text.bad_arity_fixture,"ax",@progbits
        .global bad_arity_fixture
bad_arity_fixture:
.text.bad_arity_fixture:
        /*0000*/                   FADD R1, R2 ;                                /* 0x0 */
        /*0010*/                   EXIT ;                                        /* 0x0 */
"#;

    let unsupported_module =
        parse_nvidia_sass(UNSUPPORTED_SASS).expect("unsupported SASS should parse");
    let unsupported_ir = lift_sass_module(&unsupported_module);
    let unsupported = unsupported_ir.functions[0]
        .ops
        .iter()
        .find(|op| op.address == 0)
        .expect("unsupported opcode should lift to an IR op");
    assert!(matches!(
        &unsupported.kind,
        KernelIrOpKind::Unsupported {
            opcode,
            reason: SassUnsupportedReason::NoLocalMapping,
        } if opcode == &SassOpcode::new("MYSTERY")
    ));

    let bad_arity_module = parse_nvidia_sass(BAD_ARITY_SASS).expect("bad arity SASS should parse");
    let bad_arity_ir = lift_sass_module(&bad_arity_module);
    let bad_arity = bad_arity_ir.functions[0]
        .ops
        .iter()
        .find(|op| op.address == 0)
        .expect("bad arity opcode should lift to an IR op");
    assert!(matches!(
        &bad_arity.kind,
        KernelIrOpKind::Unsupported {
            opcode,
            reason:
                SassUnsupportedReason::OperandArity {
                    expectation: SassOperandArityExpectation::AtLeast,
                    expected: 3,
                    actual: 2,
                },
        } if opcode == &SassOpcode::new("FADD")
    ));
}

#[test]
fn lift_select_and_integer_to_float_ops_are_typed() {
    const SELECT_CONVERT_SASS: &str = r#"
        .target sm_120

        .section .text.select_convert_fixture,"ax",@progbits
        .global select_convert_fixture
select_convert_fixture:
.text.select_convert_fixture:
        /*0000*/                   SEL R6, R7, R6, P0 ;                         /* 0x0 */
        /*0010*/                   I2FP.F32.U32 R4, R6 ;                        /* 0x0 */
        /*0020*/                   I2F.U32 R2, R4 ;                             /* 0x0 */
        /*0030*/                   EXIT ;                                       /* 0x0 */
"#;

    let module = parse_nvidia_sass(SELECT_CONVERT_SASS).expect("SASS should parse");
    let ir = lift_sass_module(&module);
    assert_eq!(ir.unsupported_instruction_count(), 0);
    let function = &ir.functions[0];

    let select = &function.ops[0];
    assert_eq!(select.source_opcode.kind(), &SassOpcodeKind::Sel);
    assert!(matches!(
        &select.kind,
        KernelIrOpKind::Select {
            dst,
            true_value,
            false_value,
            predicate,
        } if dst == &reg("R6")
            && true_value == &scalar("R7")
            && false_value == &scalar("R6")
            && predicate == &reg("P0")
    ));

    let convert = &function.ops[1];
    assert_eq!(convert.source_opcode.kind(), &SassOpcodeKind::I2fp);
    assert!(matches!(
        &convert.kind,
        KernelIrOpKind::NumericConvert {
            dst,
            src,
            dst_dtype: Some(SassNumericDType::F32),
            src_dtype: Some(SassNumericDType::U32),
        } if dst == &reg("R4") && src == &scalar("R6")
    ));

    let legacy_convert = &function.ops[2];
    assert_eq!(legacy_convert.source_opcode.kind(), &SassOpcodeKind::I2f);
    assert!(matches!(
        &legacy_convert.kind,
        KernelIrOpKind::NumericConvert {
            dst,
            src,
            dst_dtype: Some(SassNumericDType::U32),
            src_dtype: None,
        } if dst == &reg("R2") && src == &scalar("R4")
    ));

    let analysis = analyze_sass_ir(&ir);
    let dataflow = &analysis.functions[0].dataflow;
    assert_eq!(dataflow[0].defines.as_slice(), &[reg("R6")]);
    assert_eq!(
        dataflow[0].uses.iter().cloned().collect::<BTreeSet<_>>(),
        [reg("P0"), reg("R6"), reg("R7")].into_iter().collect()
    );
    assert_eq!(dataflow[1].defines.as_slice(), &[reg("R4")]);
    assert_eq!(dataflow[1].uses.as_slice(), &[reg("R6")]);
    assert_eq!(dataflow[2].defines.as_slice(), &[reg("R2")]);
    assert_eq!(dataflow[2].uses.as_slice(), &[reg("R4")]);

    let lifted = lift_sass_value_ir(&ir, &analysis);
    assert_eq!(lifted.functions[0].ops[0].kind, SassLiftedOpKind::Select);
    assert_eq!(
        lifted.functions[0].ops[1].kind,
        SassLiftedOpKind::NumericConvert
    );
    assert_eq!(
        lifted.functions[0].ops[2].kind,
        SassLiftedOpKind::NumericConvert
    );
    assert!(
        lifted.functions[0].ops[1]
            .semantics
            .to_string()
            .contains("dst-dtype=F32,src-dtype=U32")
    );
    assert!(
        lifted.functions[0].ops[2]
            .semantics
            .to_string()
            .contains("dst-dtype=U32")
    );
}

#[test]
fn lift_matrix_move_op_is_typed() {
    const MOVM_SASS: &str = r#"
        .target sm_120

        .section .text.movm_fixture,"ax",@progbits
        .global movm_fixture
movm_fixture:
.text.movm_fixture:
        /*0000*/                   MOVM.U4TO8.M832 R4, RZ ;                    /* 0x0 */
        /*0010*/                   EXIT ;                                       /* 0x0 */
"#;

    let module = parse_nvidia_sass(MOVM_SASS).expect("MOVM SASS should parse");
    let ir = lift_sass_module(&module);
    assert_eq!(ir.unsupported_instruction_count(), 0);
    let movm = &ir.functions[0].ops[0];

    assert_eq!(movm.source_opcode.kind(), &SassOpcodeKind::Movm);
    assert!(movm.source_modifiers.iter().any(|modifier| {
        matches!(modifier.kind(), SassModifierKind::Raw(raw) if raw == "U4TO8")
    }));
    assert!(movm.source_modifiers.iter().any(|modifier| {
        matches!(modifier.kind(), SassModifierKind::Raw(raw) if raw == "M832")
    }));
    assert!(matches!(
        &movm.kind,
        KernelIrOpKind::Move { dst, src }
            if dst == &reg("R4") && src == &scalar("RZ")
    ));

    let analysis = analyze_sass_ir(&ir);
    assert_eq!(
        analysis.functions[0].dataflow[0].defines.as_slice(),
        &[reg("R4")]
    );
    assert!(analysis.functions[0].dataflow[0].uses.is_empty());
}

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

const TENSOR_CORE_SASS: &str = r#"
        .target sm_120

        .section .text.tensor_core_fixture,"ax",@progbits
        .global tensor_core_fixture
tensor_core_fixture:
.text.tensor_core_fixture:
        /*0000*/                   HMMA R8, R12, R16, R20 ;                      /* 0x0 */
        /*0010*/                   LDT R2, tmem[UR4] ;                           /* 0x0 */
        /*0020*/                   UTMALDG desc[UR8][R0.64], R2 ;                /* 0x0 */
        /*0030*/                   WARPGROUP ;                                   /* 0x0 */
        /*0040*/                   USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xe8 ;      /* 0x0 */
        /*0050*/                   EXIT ;                                        /* 0x0 */
"#;

const TENSOR_CORE_DTYPE_SASS: &str = r#"
        .target sm_120

        .section .text.tensor_core_dtype_fixture,"ax",@progbits
        .global tensor_core_dtype_fixture
tensor_core_dtype_fixture:
.text.tensor_core_dtype_fixture:
        /*0000*/                   HMMA.16816.F32.BF16 R8, R12, R16, R20 ;       /* 0x0 */
        /*0010*/                   HMMA.16816.F32.F16 R8, R12, R16, R20 ;        /* 0x0 */
        /*0020*/                   HMMA.1688.F32.TF32 R8, R12, R16, R20 ;        /* 0x0 */
        /*0030*/                   EXIT ;                                        /* 0x0 */
"#;

const TENSOR_CORE_SM120A_DTYPE_SASS: &str = r#"
        .target sm_120a

        .section .text.tensor_core_sm120a_dtype_fixture,"ax",@progbits
        .global tensor_core_sm120a_dtype_fixture
tensor_core_sm120a_dtype_fixture:
.text.tensor_core_sm120a_dtype_fixture:
        /*0000*/                   OMMA.E2M1 R8, R12, R16, R20 ;                 /* 0x0 */
        /*0010*/                   QMMA.E4M3 R8, R12, R16, R20 ;                 /* 0x0 */
        /*0020*/                   QMMA.E5M2 R8, R12, R16, R20 ;                 /* 0x0 */
        /*0030*/                   EXIT ;                                        /* 0x0 */
"#;

const TENSOR_CORE_SM90A_WARPGROUP_DTYPE_SASS: &str = r#"
        .target sm_90a

        .section .text.tensor_core_sm90a_warpgroup_dtype_fixture,"ax",@progbits
        .global tensor_core_sm90a_warpgroup_dtype_fixture
tensor_core_sm90a_warpgroup_dtype_fixture:
.text.tensor_core_sm90a_warpgroup_dtype_fixture:
        /*0000*/                   QGMMA.E4M3 R8, R12, R16, R20 ;                /* 0x0 */
        /*0010*/                   QGMMA.E5M2 R8, R12, R16, R20 ;                /* 0x0 */
        /*0020*/                   EXIT ;                                        /* 0x0 */
"#;

const SM120_TENSOR_CORE_MMA_SASS: &str = r#"
        .target sm_120

        .section .text.sm120_tensor_core_mma_fixture,"ax",@progbits
        .global sm120_tensor_core_mma_fixture
sm120_tensor_core_mma_fixture:
.text.sm120_tensor_core_mma_fixture:
        /*0000*/                   HMMA.16816.F32.F16 R8, R12, R16, R20 ;        /* 0x0 */
        /*0010*/                   IMMA.16832.S8.S8 R8, R12, R16, R20 ;          /* 0x0 */
        /*0020*/                   DMMA.8x8x4 R8, R12, R16, R20 ;                /* 0x0 */
        /*0030*/                   EXIT ;                                        /* 0x0 */
"#;

const SM120A_TENSOR_CORE_MMA_SASS: &str = r#"
        .target sm_120a

        .section .text.sm120a_tensor_core_mma_fixture,"ax",@progbits
        .global sm120a_tensor_core_mma_fixture
sm120a_tensor_core_mma_fixture:
.text.sm120a_tensor_core_mma_fixture:
        /*0000*/                   QMMA.E4M3 R8, R12, R16, R20 ;                 /* 0x0 */
        /*0010*/                   OMMA.E2M1 R8, R12, R16, R20 ;                 /* 0x0 */
        /*0020*/                   EXIT ;                                        /* 0x0 */
"#;

const SM120_WRONG_ARCH_TENSOR_CORE_MMA_SASS: &str = r#"
        .target sm_120

        .section .text.sm120_wrong_arch_tensor_core_mma_fixture,"ax",@progbits
        .global sm120_wrong_arch_tensor_core_mma_fixture
sm120_wrong_arch_tensor_core_mma_fixture:
.text.sm120_wrong_arch_tensor_core_mma_fixture:
        /*0000*/                   QMMA.E4M3 R8, R12, R16, R20 ;                 /* 0x0 */
        /*0010*/                   OMMA.E2M1 R8, R12, R16, R20 ;                 /* 0x0 */
        /*0020*/                   EXIT ;                                        /* 0x0 */
"#;

const MEMORY_ATOMIC_SASS: &str = r#"
        .target sm_120

        .section .text.memory_atomic_fixture,"ax",@progbits
        .global memory_atomic_fixture
memory_atomic_fixture:
.text.memory_atomic_fixture:
        /*0000*/                   ATOM.E.ADD.U32 R8, [R2], R4 ;                /* 0x0 */
        /*0010*/                   RED.E.MAX.S32 [R6], R8 ;                     /* 0x0 */
        /*0020*/                   ATOMG.E.ADD.STRONG.GPU PT, R0, desc[UR4][R4.64], R11 ; /* 0x0 */
        /*0030*/                   REDG.E.ADD.STRONG.GPU desc[UR4][R4.64], R11 ; /* 0x0 */
        /*0040*/                   EXIT ;                                       /* 0x0 */
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

const SERIAL_BF16_MATVEC_SLICE: &str = r#"
        .target sm_120

        .section .text.matvec_bf16_serial,"ax",@progbits
        .global matvec_bf16_serial
matvec_bf16_serial:
.text.matvec_bf16_serial:
        /*0000*/                   LD.E.U16 R2, desc[UR4][R0.64] ;               /* 0x0 */
        /*0010*/                   IMAD.U32 R2, R2, 0x10000, RZ ;                /* 0x0 */
        /*0020*/                   FMUL R3, R2, R4 ;                             /* 0x0 */
        /*0030*/                   FADD R5, R3, R5 ;                             /* 0x0 */
        /*0040*/                   ST.E desc[UR8][R0.64], R5 ;                   /* 0x0 */
        /*0050*/                   EXIT ;                                        /* 0x0 */
"#;

const SCALAR_GEMM_SLICE: &str = r#"
        .target sm_120

        .section .text.scalar_gemm_fixture,"ax",@progbits
        .global scalar_gemm_fixture
scalar_gemm_fixture:
.text.scalar_gemm_fixture:
        /*0000*/                   LD.E R2, desc[UR8][R0.64] ;                  /* 0x0 */
        /*0010*/                   LD.E.U16 R3, desc[UR10][R4.64] ;             /* 0x0 */
        /*0020*/                   IMAD.U32 R5, R3, 0x10000, RZ ;               /* 0x0 */
        /*0030*/                   STS [R8], R2 ;                               /* 0x0 */
        /*0040*/                   STS [R12], R5 ;                              /* 0x0 */
        /*0050*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;                /* 0x0 */
        /*0060*/                   LDS R14, [R8] ;                              /* 0x0 */
        /*0070*/                   LDS R15, [R12] ;                             /* 0x0 */
        /*0080*/                   FMUL R16, R14, R15 ;                         /* 0x0 */
        /*0090*/                   FADD R17, R16, R17 ;                         /* 0x0 */
        /*00a0*/                   ST.E desc[UR12][R20.64], R17 ;               /* 0x0 */
        /*00b0*/                   EXIT ;                                       /* 0x0 */
"#;

#[test]
fn decompiled_matvec_types_route_to_autotune_generation_and_emit_crate() {
    let module = parse_nvidia_sass(ROWS17_SLICE).expect("rows17 slice should parse");
    let ir = lift_sass_module(&module);
    let function = &ir.functions[0];

    let routed = decompiled_autotune_operation(
        function,
        DecompiledAutotuneShape::MatvecBf16RowMajor {
            rows: 128,
            cols: 256,
        },
    )
    .expect("typed decompiled matvec evidence should route into autotune");

    assert!(routed.evidence.supports_bf16_row_major_matvec());
    assert_eq!(routed.operation.kind, OperationKind::Matvec);
    assert_eq!(routed.operation.inputs[0].dtype, NumericKind::F32);
    assert_eq!(routed.operation.inputs[1].dtype, NumericKind::Bf16);
    assert_eq!(routed.operation.outputs[0].accumulator, NumericKind::F32);

    let config = AutoOptimizeConfig {
        beam_width: 4,
        max_steps: 2,
        require_launchable: false,
        min_score_improvement: 0.0,
    };
    let generated = generate_inference_kernel_source(&routed.operation, config)
        .expect("decompiled operation should autotune and render source");
    assert_eq!(
        generated.optimization.problem.family(),
        "matvec-bf16-row-major"
    );
    assert_eq!(generated.source.symbol, generated.candidate.launch.kernel);
    assert!(generated.source.source.contains("#[kernel]"));

    let root = decompile_autotune_test_root();
    let store = KernelArtifactStore::new(&root);
    let emitted = store
        .emit_standalone_crate(&generated.candidate, &InferenceKernelRustCudaGenerator)
        .expect("decompiled autotune candidate should emit standalone crate");
    assert_eq!(emitted.symbol, generated.source.symbol);
    assert!(emitted.paths.source_path.starts_with(store.root()));
    let source = fs::read_to_string(&emitted.paths.source_path)
        .expect("emitted standalone source should be readable");
    assert!(source.contains(&format!("pub fn {}(", generated.source.symbol)));
    cleanup_decompile_autotune_test_root(&root);
}

#[test]
fn decompiled_scalar_gemm_types_route_to_autotune_generation() {
    let module = parse_nvidia_sass(SCALAR_GEMM_SLICE).expect("scalar GEMM slice should parse");
    let ir = lift_sass_module(&module);
    let routed = decompiled_autotune_operation(
        &ir.functions[0],
        DecompiledAutotuneShape::GemmF32Bf16RowColRow {
            m: 64,
            n: 64,
            k: 128,
        },
    )
    .expect("typed scalar GEMM evidence should route into autotune");

    assert!(routed.evidence.supports_f32_bf16_row_col_row_gemm());
    assert!(routed.evidence.has_f32_descriptor_load);
    assert!(routed.evidence.has_bf16_descriptor_load);
    assert!(routed.evidence.has_bf16_widen);
    assert!(routed.evidence.has_shared_store);
    assert!(routed.evidence.has_shared_load);
    assert!(routed.evidence.has_barrier);
    assert!(routed.evidence.has_descriptor_store);
    assert_eq!(routed.operation.kind, OperationKind::Gemm);
    assert_eq!(routed.operation.inputs[0].dtype, NumericKind::F32);
    assert_eq!(routed.operation.inputs[0].shape, vec![64, 128]);
    assert_eq!(routed.operation.inputs[1].dtype, NumericKind::Bf16);
    assert_eq!(routed.operation.inputs[1].shape, vec![128, 64]);
    assert_eq!(routed.operation.outputs[0].shape, vec![64, 64]);

    let generated = generate_inference_kernel_source(
        &routed.operation,
        AutoOptimizeConfig {
            beam_width: 3,
            max_steps: 1,
            require_launchable: false,
            min_score_improvement: 0.0,
        },
    )
    .expect("decompiled GEMM operation should autotune and render source");
    assert_eq!(
        generated.optimization.problem.family(),
        "gemm-f32-bf16-row-col-row"
    );
    assert!(generated.source.source.contains("#[kernel]"));
}

#[test]
fn serial_decompiled_matvec_evidence_does_not_require_warp_reduce() {
    let module = parse_nvidia_sass(SERIAL_BF16_MATVEC_SLICE).expect("serial slice should parse");
    let ir = lift_sass_module(&module);
    let routed = decompiled_autotune_operation(
        &ir.functions[0],
        DecompiledAutotuneShape::MatvecBf16RowMajor {
            rows: 128,
            cols: 256,
        },
    )
    .expect("serial bf16 matvec evidence should route into autotune");

    assert!(routed.evidence.has_bf16_descriptor_load);
    assert!(routed.evidence.has_bf16_widen);
    assert!(routed.evidence.has_f32_mul_add);
    assert!(!routed.evidence.has_warp_reduce_sum);
    assert!(routed.evidence.supports_bf16_row_major_matvec());
}

#[test]
#[ignore = "runs cargo oxide build for generated standalone crate"]
fn decompiled_matvec_autotune_candidate_recompiles() {
    let module = parse_nvidia_sass(ROWS17_SLICE).expect("rows17 slice should parse");
    let ir = lift_sass_module(&module);
    let routed = decompiled_autotune_operation(
        &ir.functions[0],
        DecompiledAutotuneShape::MatvecBf16RowMajor {
            rows: 128,
            cols: 256,
        },
    )
    .expect("typed decompiled matvec evidence should route into autotune");
    let generated = generate_inference_kernel_source(
        &routed.operation,
        AutoOptimizeConfig {
            beam_width: 4,
            max_steps: 2,
            require_launchable: false,
            min_score_improvement: 0.0,
        },
    )
    .expect("decompiled operation should autotune and render source");

    let root = decompile_autotune_test_root();
    let store = KernelArtifactStore::new(&root);
    let emitted = store
        .emit_standalone_crate(&generated.candidate, &InferenceKernelRustCudaGenerator)
        .expect("decompiled autotune candidate should emit standalone crate");
    let output_dir = root.join("ptx");
    let target_dir = root.join("standalone-target");
    let compiled = compile_standalone_kernel_crate(
        &emitted.paths.crate_dir,
        &output_dir,
        &emitted.package_name,
        Some("sm_120"),
        Some(&target_dir),
    )
    .expect("generated standalone kernel should recompile");

    assert!(compiled.ptx_path.exists());
    assert!(compiled.ptx_path.starts_with(&output_dir));
    cleanup_decompile_autotune_test_root(&root);
}

#[test]
#[ignore = "SM120 hardware e2e: builds naive matvec SASS, decompiles, measured-autotunes, and requires a faster recompiled best candidate"]
fn generated_naive_rust_matvec_sass_routes_to_autotune_and_recompiles_best() {
    let root = decompile_autotune_test_root();
    let report = run_decompile_autotune_matvec(&DecompileAutotuneMatvecOptions {
        artifact_root: root.clone(),
        compile_arch: "sm_120".to_string(),
        rows: 512,
        cols: 1024,
        config: AutoOptimizeConfig {
            beam_width: 6,
            max_steps: 2,
            require_launchable: false,
            min_score_improvement: 0.0,
        },
        measure: Some(DecompileAutotuneMeasureOptions {
            repeat_count: 5,
            warmup_count: 2,
            device_index: 0,
        }),
    })
    .expect("generated naive Rust matvec should decompile, measured-autotune, and recompile");

    assert_eq!(report.naive_symbol, "matvec_bf16_naive");
    assert!(report.source_path.starts_with(&root));
    assert!(report.ptx_path.exists());
    assert!(report.cubin_path.exists());
    assert!(report.sass_path.exists());
    assert!(report.optimized_source_path.exists());
    assert!(report.optimized_ptx_path.exists());
    assert!(report.optimized_cubin_path.exists());
    assert!(report.optimized_sass_path.exists());
    assert!(report.optimized_ir_path.exists());
    assert!(report.optimized_pattern_path.exists());
    assert!(report.optimized_side_by_side_path.exists());
    assert!(report.overview_path.exists());
    assert!(report.overview_graph_path.exists());
    assert!(report.evidence.supports_bf16_row_major_matvec());
    assert!(report.optimized_evidence.supports_bf16_row_major_matvec());
    assert!(report.parsed_instruction_count > 0);
    assert!(report.optimized_parsed_instruction_count > 0);
    assert_eq!(report.unsupported_instruction_count, 0);
    assert_eq!(report.optimized_unsupported_instruction_count, 0);
    assert!(report.explored > 0);
    assert_ne!(report.best_symbol, "matvec_bf16_naive");
    assert!(report.best_action_count > 0);
    assert!(report.best_action_ops.iter().any(|op| op == "split"));
    assert!(report.optimized_evidence.has_warp_reduce_sum);
    assert_eq!(report.source_score_source.as_deref(), Some("measured"));
    assert_eq!(report.best_score_source.as_deref(), Some("measured"));
    let source_score = report
        .source_score
        .expect("measured e2e should report source naive score");
    let best_score = report
        .best_score
        .expect("measured e2e should report best score");
    assert!(
        best_score < source_score,
        "best measured score {best_score} should beat source naive score {source_score}"
    );

    let source = fs::read_to_string(&report.source_path).expect("naive source should be readable");
    assert!(source.contains("pub fn matvec_bf16_naive("));
    let optimized_source =
        fs::read_to_string(&report.optimized_source_path).expect("best source should be readable");
    assert!(optimized_source.contains(&format!("pub fn {}(", report.best_symbol)));
    let optimized_patterns = fs::read_to_string(&report.optimized_pattern_path)
        .expect("best patterns should be readable");
    assert!(optimized_patterns.contains("warp-reduce-sum"));
    cleanup_decompile_autotune_test_root(&root);
}

#[test]
#[ignore = "builds Rust-CUDA GEMM, disassembles SASS, autotunes, and recompiles best candidate"]
fn generated_gemm_sass_routes_to_autotune_and_recompiles_best() {
    let root = decompile_autotune_test_root();
    let report = run_decompile_autotune_gemm(&DecompileAutotuneGemmOptions {
        artifact_root: root.clone(),
        compile_arch: "sm_120".to_string(),
        m: 64,
        n: 64,
        k: 128,
        config: AutoOptimizeConfig {
            beam_width: 3,
            max_steps: 1,
            require_launchable: false,
            min_score_improvement: 0.0,
        },
    })
    .expect("generated GEMM should decompile, autotune, and recompile");

    assert_eq!(report.source_symbol, "gemm_f32_bf16_tile_16x16x8");
    assert!(report.source_path.starts_with(&root));
    assert!(report.ptx_path.exists());
    assert!(report.cubin_path.exists());
    assert!(report.sass_path.exists());
    assert!(report.optimized_source_path.exists());
    assert!(report.optimized_ptx_path.exists());
    assert!(report.optimized_cubin_path.exists());
    assert!(report.optimized_sass_path.exists());
    assert!(report.optimized_ir_path.exists());
    assert!(report.optimized_pattern_path.exists());
    assert!(report.optimized_side_by_side_path.exists());
    assert!(report.overview_path.exists());
    assert!(report.overview_graph_path.exists());
    assert!(report.evidence.supports_f32_bf16_row_col_row_gemm());
    assert!(
        report
            .optimized_evidence
            .supports_f32_bf16_row_col_row_gemm()
    );
    assert!(report.parsed_instruction_count > 0);
    assert!(report.optimized_parsed_instruction_count > 0);
    assert_eq!(report.unsupported_instruction_count, 0);
    assert_eq!(report.optimized_unsupported_instruction_count, 0);
    assert!(report.explored > 0);
    assert_ne!(report.best_symbol, report.source_symbol);
    assert!(report.best_action_count > 0);
    assert!(report.best_action_ops.iter().any(|op| op == "local-tile"));
    assert!(report.best_score.is_some());

    let source = fs::read_to_string(&report.source_path).expect("GEMM source should be readable");
    assert!(source.contains("pub fn gemm_f32_bf16_tile_16x16x8("));
    let optimized_source =
        fs::read_to_string(&report.optimized_source_path).expect("best source should be readable");
    assert!(optimized_source.contains(&format!("pub fn {}(", report.best_symbol)));
    let optimized_patterns = fs::read_to_string(&report.optimized_pattern_path)
        .expect("best patterns should be readable");
    assert!(optimized_patterns.contains("f32-mul-add-pair"));
    cleanup_decompile_autotune_test_root(&root);
}

#[test]
#[ignore = "writes external SASS, autotunes from it, and recompiles best candidate"]
fn external_sass_file_routes_to_autotune_and_recompiles_best() {
    let root = decompile_autotune_test_root();
    let sass_path = root.join("external-scalar-gemm.sass");
    fs::create_dir_all(&root).expect("test root should be creatable");
    fs::write(&sass_path, SCALAR_GEMM_SLICE).expect("external SASS fixture should be writable");

    let report = run_decompile_autotune_sass(&DecompileAutotuneSassOptions {
        sass_path: sass_path.clone(),
        source_path: None,
        output_dir: None,
        artifact_root: root.clone(),
        compile_arch: "sm_120".to_string(),
        function_symbol: Some("scalar_gemm_fixture".to_string()),
        shape: DecompiledAutotuneShape::GemmF32Bf16RowColRow {
            m: 64,
            n: 64,
            k: 128,
        },
        config: AutoOptimizeConfig {
            beam_width: 3,
            max_steps: 1,
            require_launchable: false,
            min_score_improvement: 0.0,
        },
    })
    .expect("external SASS should route into autotune and recompile");

    assert!(report.sass_path.is_absolute());
    assert_eq!(
        report.sass_path.file_name().and_then(|name| name.to_str()),
        Some("external-scalar-gemm.sass")
    );
    assert_eq!(report.function_symbol, "scalar_gemm_fixture");
    assert!(report.output_dir.starts_with(&root));
    assert!(report.ir_path.exists());
    assert!(report.pattern_path.exists());
    assert!(report.side_by_side_path.exists());
    assert!(report.evidence.supports_f32_bf16_row_col_row_gemm());
    assert_eq!(report.unsupported_instruction_count, 0);
    assert!(report.optimized_source_path.exists());
    assert!(report.optimized_ptx_path.exists());
    assert!(report.optimized_cubin_path.exists());
    assert!(report.optimized_sass_path.exists());
    assert!(report.optimized_ir_path.exists());
    assert!(report.optimized_pattern_path.exists());
    assert!(report.overview_path.exists());
    assert!(report.overview_graph_path.exists());
    assert!(
        report
            .optimized_evidence
            .supports_f32_bf16_row_col_row_gemm()
    );
    assert_eq!(report.optimized_unsupported_instruction_count, 0);
    assert_eq!(report.best_action_ops, vec!["local-tile"]);
    cleanup_decompile_autotune_test_root(&root);
}

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

const NESTED_LOOP_SASS: &str = r#"
        .target sm_120

        .section .text.nested_loop_fixture,"ax",@progbits
        .global nested_loop_fixture
nested_loop_fixture:
.text.nested_loop_fixture:
        /*0000*/                   MOV R2, RZ ;                                  /* 0x0 */
.L_outer:
        /*0010*/                   IADD R2, R2, 0x1 ;                            /* 0x0 */
        /*0020*/                   MOV R3, RZ ;                                  /* 0x0 */
.L_inner:
        /*0030*/                   IADD R3, R3, 0x1 ;                            /* 0x0 */
        /*0040*/                   ISETP.LT.U32.AND P0, PT, R3, R1, PT ;         /* 0x0 */
        /*0050*/               @P0 BRA `(.L_inner) ;                             /* 0x0 */
        /*0060*/                   ISETP.LT.U32.AND P1, PT, R2, R0, PT ;         /* 0x0 */
        /*0070*/               @P1 BRA `(.L_outer) ;                             /* 0x0 */
        /*0080*/                   EXIT ;                                        /* 0x0 */
"#;

#[test]
fn parse_nvidia_sass_captures_nvdisasm_function_and_operands() {
    let module = parse_nvidia_sass(SIMPLE_SASS).expect("fixture SASS should parse");

    assert_eq!(module.target.as_deref(), Some("sm_120"));
    assert_eq!(module.functions.len(), 1);
    let function = &module.functions[0];
    assert_eq!(function.name.as_str(), "sass_fixture_i32_add");
    assert_eq!(function.instructions.len(), 8);
    assert!(
        function
            .instructions
            .windows(2)
            .all(|pair| pair[0].source_position < pair[1].source_position)
    );
    let ordered_positions = function
        .instructions
        .iter()
        .map(|instruction| instruction.source_position)
        .collect::<BTreeSet<_>>();
    assert_eq!(ordered_positions.len(), function.instructions.len());
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

    let ir = lift_sass_module(&module);
    assert!(matches!(
        ir.target,
        Some(SassTarget::Architecture(architecture)) if architecture == SassArchitecture::sm(120)
    ));
    assert_eq!(
        ir.functions[0].ops[0].source_position,
        function.instructions[0].source_position
    );
}

#[test]
fn parse_nvidia_sass_captures_cuobjdump_function_header() {
    let module = parse_nvidia_sass(CUOBJDUMP_SASS).expect("cuobjdump SASS should parse");

    assert_eq!(module.target.as_deref(), Some("sm_120"));
    assert_eq!(module.functions.len(), 1);
    let function = &module.functions[0];
    assert_eq!(function.name.as_str(), "cuobjdump_fixture");
    assert_eq!(function.section.as_deref(), Some("cuobjdump_fixture"));
    assert_eq!(function.instructions.len(), 3);
    assert_eq!(function.instructions[0].opcode, "S2R");
    assert_eq!(function.instructions[1].opcode, "IADD");
    assert_eq!(function.instructions[2].opcode, "EXIT");
}

#[test]
fn lift_maps_control_special_register_read() {
    let sass = r#"
	code for sm_120
	.target	sm_120

		Function : cs2r_fixture
	.headerflags	@"EF_CUDA_SM120"
        /*0000*/                   CS2R R6, SRZ ;                                /* 0x0 */
        /*0010*/                   EXIT ;                                        /* 0x0 */
"#;
    let module = parse_nvidia_sass(sass).expect("CS2R fixture should parse");
    let ir = lift_sass_module(&module);
    let op = &ir.functions[0].ops[0];

    assert!(matches!(
        &op.kind,
        KernelIrOpKind::ReadSpecialRegister { dst, special }
            if dst == &reg("R6") && special == &reg("SRZ")
    ));
}

#[test]
fn lift_branch_condition_reads_typed_predicate_negation() {
    const NEGATED_BRANCH_SASS: &str = r#"
        .target sm_120
        .section .text.negated_branch,"ax",@progbits
        .global negated_branch
negated_branch:
.text.negated_branch:
        /*0000*/                   BRA !P0, `(.L_done) ;                          /* 0x0 */
        /*0010*/                   EXIT ;                                          /* 0x0 */
.L_done:
        /*0020*/                   EXIT ;                                          /* 0x0 */
    "#;

    let module = parse_nvidia_sass(NEGATED_BRANCH_SASS).expect("branch SASS should parse");
    let ir = lift_sass_module(&module);
    let branch = ir.functions[0]
        .ops
        .iter()
        .find(|op| op.address == 0x0)
        .expect("branch should lift");

    assert!(matches!(
        &branch.kind,
        KernelIrOpKind::Branch {
            condition: Some(PredicateCondition {
                kind: PredicateConditionKind::Register {
                    register,
                    negated: true,
                },
                ..
            }),
            ..
        } if register == &reg("P0")
    ));
    assert!(matches!(
        &branch.source_operands[0].kind,
        AggregateOperandKind::Register(register) if register.negated
    ));
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
        } if matches!(target.kind, ControlTargetKind::Address(0x10)) && target.raw == "0x10"
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
            if dst == &reg("R0") && special == &reg("SR_TID.X")
    ));
    assert!(matches!(
        kinds[1],
        KernelIrOpKind::Load {
            space: MemorySpace::Descriptor,
            address,
            access,
            ..
        } if access.width_bits.is_none()
            && access.modifiers.as_slice() == [SassMemoryModifier::E]
            && matches!(
                &address.kind,
                MemoryAddressKind::Descriptor {
                    descriptor,
                    address,
                    address_width: Some(64),
                    offset: None,
                } if descriptor.to_string() == "UR4" && address.to_string() == "R0"
            )
    ));
    assert!(matches!(
        kinds[3],
        KernelIrOpKind::IntegerAdd {
            dst,
            inputs,
            width_bits: None
        } if dst == &reg("R4") && inputs.as_slice() == [scalar("R2"), scalar("R3")]
    ));
    assert_eq!(
        ir.functions[0].ops[3].source_opcode.kind(),
        &SassOpcodeKind::Iadd
    );
    assert_eq!(
        ir.functions[0].ops[1].source_modifiers[0].kind(),
        &SassModifierKind::E
    );
    assert!(matches!(
        kinds[4],
        KernelIrOpKind::Store {
            space: MemorySpace::Descriptor,
            address,
            access,
            ..
        } if access.width_bits.is_none()
            && access.modifiers.as_slice() == [SassMemoryModifier::E]
            && matches!(
                &address.kind,
                MemoryAddressKind::Descriptor {
                    descriptor,
                    address,
                    address_width: Some(64),
                    offset: None,
                } if descriptor.to_string() == "UR8" && address.to_string() == "R0"
            )
    ));
    assert!(matches!(
        kinds[5],
        KernelIrOpKind::Permute { dst, inputs } if dst == &reg("R5") && inputs.len() == 3
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
fn memory_address_ir_is_typed_and_orderable() {
    const MEMORY_ADDRESS_SASS: &str = r#"
        .target sm_120

        .section .text.memory_address_fixture,"ax",@progbits
        .global memory_address_fixture
memory_address_fixture:
.text.memory_address_fixture:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                      /* 0x0 */
        /*0010*/                   LD.E.U16 R2, desc[UR4][R0.64+0x4] ;          /* 0x0 */
        /*0020*/                   LDS R3, [R2+0x40] ;                          /* 0x0 */
        /*0030*/                   EXIT ;                                       /* 0x0 */
"#;

    let module = parse_nvidia_sass(MEMORY_ADDRESS_SASS).expect("memory SASS should parse");
    let ir = lift_sass_module(&module);
    let mut ordered_addresses = BTreeSet::new();

    for op in &ir.functions[0].ops {
        match &op.kind {
            KernelIrOpKind::LoadConst { source, .. }
            | KernelIrOpKind::Load {
                address: source, ..
            }
            | KernelIrOpKind::Store {
                address: source, ..
            } => {
                ordered_addresses.insert(source.clone());
            }
            _ => {}
        }
    }

    assert_eq!(ordered_addresses.len(), 3);
    assert!(ordered_addresses.iter().any(|address| matches!(
        &address.kind,
        MemoryAddressKind::Constant { bank, offset }
            if bank.as_integer() == Some(0) && offset.as_integer() == Some(0x37c)
    )));
    assert!(ordered_addresses.iter().any(|address| matches!(
        &address.kind,
        MemoryAddressKind::Descriptor {
            descriptor,
            address: register,
            address_width: Some(64),
            offset: Some(offset),
        } if descriptor == &reg("UR4") && register == &reg("R0") && offset.as_integer() == Some(0x4)
    )));
    assert!(ordered_addresses.iter().any(|address| matches!(
        &address.kind,
        MemoryAddressKind::Indexed {
            base,
            offset: Some(offset),
        } if base == &reg("R2") && offset.as_integer() == Some(0x40)
    )));
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
    assert_eq!(first_load.value_register, reg("R2"));
    assert!(matches!(
        &first_load.memory_address.kind,
        MemoryAddressKind::Descriptor {
            descriptor,
            address,
            address_width: Some(64),
            offset: None,
        } if descriptor == &reg("UR4") && address == &reg("R0")
    ));
    assert_eq!(first_load.address_registers, [reg("UR4"), reg("R0")]);
    assert!(matches!(
        &first_load.address_base,
        Some(MemoryAddressBase::Descriptor(base)) if base == &reg("UR4")
    ));
    assert_eq!(first_load.offset, None);

    let offset_load = function
        .memory_accesses
        .iter()
        .find(|access| access.address == 0x20)
        .expect("offset descriptor load should be recovered");
    assert!(matches!(
        &offset_load.address_base,
        Some(MemoryAddressBase::Descriptor(base)) if base == &reg("UR6")
    ));
    assert!(matches!(
        &offset_load.offset,
        Some(offset) if offset.as_integer() == Some(0x4)
    ));

    let store = function
        .memory_accesses
        .iter()
        .find(|access| access.address == 0x40)
        .expect("descriptor store should be recovered");
    assert_eq!(store.kind, SassMemoryAccessKind::Store);
    assert_eq!(store.space, MemorySpace::Descriptor);
    assert_eq!(store.value_register, reg("R4"));
    assert_eq!(store.address_registers, [reg("UR8"), reg("R0")]);

    let text = analysis.to_text();
    assert!(text.contains("memory_accesses"));
    assert!(text.contains("0x0010: load descriptor value=R2 addr=desc[UR4][R0.64]"));
    assert!(text.contains("base=UR4"));
}

#[test]
fn lift_memory_atom_and_reduction_ops_are_typed() {
    let module = parse_nvidia_sass(MEMORY_ATOMIC_SASS).expect("atomic SASS should parse");
    let ir = lift_sass_module(&module);
    let function = &ir.functions[0];

    assert_eq!(ir.unsupported_instruction_count(), 0);
    assert!(matches!(
        &function.ops[0].kind,
        KernelIrOpKind::MemoryAtomic {
            predicate_dst: None,
            dst,
            address,
            values,
            operation: Some(SassMemoryAtomicOp::Add),
            space: MemorySpace::Global,
            access,
        } if dst == &reg("R8")
            && values.as_slice() == [scalar("R4")]
            && access.width_bits == Some(32)
            && access.modifiers.as_slice() == [
                SassMemoryModifier::E,
                SassMemoryModifier::Raw("ADD".to_string()),
                SassMemoryModifier::Unsigned(32),
            ]
            && matches!(
                &address.kind,
                MemoryAddressKind::Indexed { base, offset: None } if base == &reg("R2")
            )
    ));
    assert_eq!(function.ops[0].source_opcode.kind(), &SassOpcodeKind::Atom);
    assert_eq!(
        function.ops[0].source_modifiers[1].kind(),
        &SassModifierKind::Add
    );
    assert!(matches!(
        &function.ops[1].kind,
        KernelIrOpKind::MemoryReduction {
            address,
            values,
            operation: Some(SassMemoryAtomicOp::Max),
            space: MemorySpace::Global,
            access,
        } if values.as_slice() == [scalar("R8")]
            && access.width_bits == Some(32)
            && matches!(
                &address.kind,
                MemoryAddressKind::Indexed { base, offset: None } if base == &reg("R6")
            )
    ));
    assert_eq!(function.ops[1].source_opcode.kind(), &SassOpcodeKind::Red);
    assert_eq!(
        function.ops[1].source_modifiers[1].kind(),
        &SassModifierKind::Max
    );
    assert!(matches!(
        &function.ops[2].kind,
        KernelIrOpKind::MemoryAtomic {
            predicate_dst: Some(predicate_dst),
            dst,
            address,
            values,
            operation: Some(SassMemoryAtomicOp::Add),
            space: MemorySpace::Descriptor,
            ..
        } if predicate_dst == &reg("PT")
            && dst == &reg("R0")
            && values.as_slice() == [scalar("R11")]
            && matches!(
                &address.kind,
                MemoryAddressKind::Descriptor {
                    descriptor,
                    address,
                    address_width: Some(64),
                    offset: None,
                } if descriptor == &reg("UR4") && address == &reg("R4")
            )
    ));
    assert_eq!(function.ops[2].source_opcode.kind(), &SassOpcodeKind::Atomg);
    assert!(matches!(
        &function.ops[3].kind,
        KernelIrOpKind::MemoryReduction {
            address,
            values,
            operation: Some(SassMemoryAtomicOp::Add),
            space: MemorySpace::Descriptor,
            ..
        } if values.as_slice() == [scalar("R11")]
            && matches!(
                &address.kind,
                MemoryAddressKind::Descriptor {
                    descriptor,
                    address,
                    address_width: Some(64),
                    offset: None,
                } if descriptor == &reg("UR4") && address == &reg("R4")
            )
    ));
    assert_eq!(function.ops[3].source_opcode.kind(), &SassOpcodeKind::Redg);

    let analysis = analyze_sass_ir(&ir);
    let analyzed = &analysis.functions[0];
    let atomic_flow = analyzed
        .dataflow
        .iter()
        .find(|op| op.address == 0)
        .expect("atomic op dataflow should exist");
    assert_eq!(atomic_flow.defines, [reg("R8")]);
    assert_eq!(atomic_flow.uses, [reg("R2"), reg("R4")]);
    let reduction_flow = analyzed
        .dataflow
        .iter()
        .find(|op| op.address == 0x10)
        .expect("reduction op dataflow should exist");
    assert_eq!(reduction_flow.defines, []);
    assert_eq!(reduction_flow.uses, [reg("R6"), reg("R8")]);
    let atomg_flow = analyzed
        .dataflow
        .iter()
        .find(|op| op.address == 0x20)
        .expect("global atomic op dataflow should exist");
    assert_eq!(atomg_flow.defines, [reg("R0")]);
    assert_eq!(atomg_flow.uses, [reg("UR4"), reg("R4"), reg("R11")]);
    let redg_flow = analyzed
        .dataflow
        .iter()
        .find(|op| op.address == 0x30)
        .expect("global reduction op dataflow should exist");
    assert_eq!(redg_flow.defines, []);
    assert_eq!(redg_flow.uses, [reg("UR4"), reg("R4"), reg("R11")]);

    let atomic_access = analyzed
        .memory_accesses
        .iter()
        .find(|access| access.address == 0)
        .expect("atomic memory access should exist");
    assert_eq!(atomic_access.kind, SassMemoryAccessKind::Atomic);
    assert_eq!(atomic_access.value_register, reg("R8"));
    let reduction_access = analyzed
        .memory_accesses
        .iter()
        .find(|access| access.address == 0x10)
        .expect("reduction memory access should exist");
    assert_eq!(reduction_access.kind, SassMemoryAccessKind::Reduction);
    assert_eq!(reduction_access.value_register, reg("R8"));
    let atomg_access = analyzed
        .memory_accesses
        .iter()
        .find(|access| access.address == 0x20)
        .expect("global atomic memory access should exist");
    assert_eq!(atomg_access.kind, SassMemoryAccessKind::Atomic);
    assert_eq!(atomg_access.space, MemorySpace::Descriptor);
    assert_eq!(atomg_access.value_register, reg("R0"));
    let redg_access = analyzed
        .memory_accesses
        .iter()
        .find(|access| access.address == 0x30)
        .expect("global reduction memory access should exist");
    assert_eq!(redg_access.kind, SassMemoryAccessKind::Reduction);
    assert_eq!(redg_access.space, MemorySpace::Descriptor);
    assert_eq!(redg_access.value_register, reg("R11"));

    let lifted = lift_sass_value_ir(&ir, &analysis);
    let atomic = &lifted.functions[0].ops[0];
    assert_eq!(atomic.class, SassLiftedOpClass::Memory);
    assert_eq!(atomic.kind, SassLiftedOpKind::MemoryAtomic);
    assert!(matches!(
        &atomic.semantics,
        SassLiftedSemantics::MemoryAtomic {
            dst,
            operation: Some(SassMemoryAtomicOp::Add),
            ..
        } if dst == &reg("R8")
    ));
    assert!(atomic.semantics.to_string().contains("memory-atomic"));

    let reduction = &lifted.functions[0].ops[1];
    assert_eq!(reduction.class, SassLiftedOpClass::Memory);
    assert_eq!(reduction.kind, SassLiftedOpKind::MemoryReduction);
    assert!(matches!(
        &reduction.semantics,
        SassLiftedSemantics::MemoryReduction {
            operation: Some(SassMemoryAtomicOp::Max),
            ..
        }
    ));
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
            address,
            width_bits,
            modifiers
        } if dst == &reg("R2") && address.raw == "desc[UR4][R0.64]"
            && width_bits.is_none()
            && modifiers.as_slice() == [SassMemoryModifier::E]
    ));
    assert!(load.outputs.iter().any(|value| value.register == reg("R2")));
    assert!(load.source_operands.iter().any(|operand| {
        matches!(
            &operand.kind,
            AggregateOperandKind::Memory(address)
                if matches!(
                    &address.kind,
                    MemoryAddressKind::Descriptor {
                        descriptor,
                        address,
                        address_width: Some(64),
                        offset: None,
                    } if descriptor == &reg("UR4") && address == &reg("R0")
                )
        )
    }));

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
        } if dst == &reg("R4") && inputs.as_slice() == [scalar("R2"), scalar("R3")]
    ));
    assert!(add.inputs.iter().any(|value| value.register == reg("R2")));
    assert!(add.inputs.iter().any(|value| value.register == reg("R3")));
    assert!(add.outputs.iter().any(|value| value.register == reg("R4")));

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
            value,
            width_bits,
            modifiers
        } if address.raw == "desc[UR8][R0.64]" && value == &reg("R4")
            && width_bits.is_none()
            && modifiers.as_slice() == [SassMemoryModifier::E]
    ));
    assert!(store.inputs.iter().any(|value| value.register == reg("R4")));
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
    assert!(text.contains(
        "semantics=load(space=descriptor,dst=R2,address=desc[UR4][R0.64],width=-,modifiers=[E])"
    ));
}

#[test]
fn lift_vector_integer_add_is_typed() {
    let sass = r#"
        .target sm_90
        .section .text.viadd_fixture,"ax",@progbits
        .global viadd_fixture
viadd_fixture:
.text.viadd_fixture:
        /*0000*/                   VIADD R1, R1, 0xfffffe00 ; /* 0x0 */
        /*0010*/                   EXIT ; /* 0x0 */
    "#;

    let module = parse_nvidia_sass(sass).expect("VIADD SASS should parse");
    let ir = lift_sass_module(&module);
    assert!(matches!(
        &ir.functions[0].ops[0].kind,
        KernelIrOpKind::IntegerAdd {
            dst,
            inputs,
            width_bits: None
        } if dst == &reg("R1") && inputs.as_slice() == [scalar("R1"), scalar("0xfffffe00")]
    ));
    assert_eq!(
        ir.functions[0].ops[0].source_opcode.kind(),
        &SassOpcodeKind::Viadd
    );
}

#[test]
fn lift_rows17_slice_keeps_predicates_and_half_fma_visible() {
    let module = parse_nvidia_sass(ROWS17_SLICE).expect("rows17 slice should parse");
    let ir = lift_sass_module(&module);
    let ops = &ir.functions[0].ops;

    let compare_op = ops
        .iter()
        .find(|op| matches!(op.kind, KernelIrOpKind::CompareSet { .. }))
        .expect("rows17 fixture should contain a compare op");
    assert_eq!(
        compare_op.source_modifiers[0].kind(),
        &SassModifierKind::GreaterThan
    );
    assert_eq!(
        compare_op.source_modifiers[1].kind(),
        &SassModifierKind::UnsignedWidth(32)
    );
    assert_eq!(
        compare_op.source_modifiers[2].kind(),
        &SassModifierKind::And
    );

    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::CompareSet {
            comparison: Some(SassComparisonKind::GreaterThan),
            dtype: Some(SassCompareDType::U32),
            ref lhs,
            ref rhs,
            ..
        } if lhs == &scalar("R2") && rhs == &scalar("0x10")
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::Exit {
            condition: Some(ref condition)
        } if matches!(
            &condition.kind,
            PredicateConditionKind::Register {
                register,
                negated: false,
            } if register.to_string() == "P0"
        )
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::Load {
            access: ref memory_access,
            ..
        } if memory_access.width_bits == Some(16)
            && memory_access.modifiers.as_slice()
                == [SassMemoryModifier::E, SassMemoryModifier::Unsigned(16)]
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
        } if matches!(&target.kind, ControlTargetKind::Label(label) if label.as_str() == ".L_x_1")
            && matches!(
                &condition.kind,
                PredicateConditionKind::Register {
                    register,
                    negated: false,
                } if register.to_string() == "P0"
            )
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::Call {
            target: Some(ref target),
            ref operands,
        } if matches!(&target.kind, ControlTargetKind::Label(label) if label.as_str() == "$helper")
            && matches!(&operands[0].kind, AggregateOperandKind::Label(label) if label.as_str() == "$helper")
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::Sync {
            ref kind,
            ref operands,
        } if kind == &SassSyncKind::BarrierSet
            && matches!(&operands[0].kind, AggregateOperandKind::Register(register) if register == &reg("B0"))
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::WarpShuffle {
            mode: Some(SassWarpShuffleMode::Down),
            offset: ref shuffle_offset,
            ..
        } if shuffle_offset == &scalar("0x10")
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        KernelIrOpKind::Return {
            target: Some(ref target),
            ..
        } if matches!(&target.kind, ControlTargetKind::Label(label) if label.as_str() == "matvec_bf16_rows17")
    )));
    assert_eq!(ir.unsupported_instruction_count(), 0);
}

#[test]
fn lift_vote_and_uniform_logic_ops_are_typed() {
    const VOTE_SASS: &str = r#"
        .target sm_75

        .section .text.vote_fixture,"ax",@progbits
        .global vote_fixture
vote_fixture:
.text.vote_fixture:
        /*0000*/                   VOTEU.ANY UR4, UPT, PT ;                    /* 0x0 */
        /*0010*/                   ULOP3.LUT UR4, UR4, 0x2, URZ, 0xc0, !UPT ;  /* 0x0 */
        /*0020*/                   VOTE.ANY R0, PT, P0 ;                       /* 0x0 */
        /*0030*/                   EXIT ;                                      /* 0x0 */
"#;

    let module = parse_nvidia_sass(VOTE_SASS).expect("vote SASS should parse");
    let ir = lift_sass_module(&module);
    let ops = &ir.functions[0].ops;

    assert!(matches!(
        &ops[0].kind,
        KernelIrOpKind::WarpElect { dst, .. } if dst == &reg("UR4")
    ));
    assert_eq!(ops[0].source_opcode.kind(), &SassOpcodeKind::Voteu);
    assert!(matches!(
        &ops[1].kind,
        KernelIrOpKind::LogicLut { dst, .. } if dst == &reg("UR4")
    ));
    assert_eq!(ops[1].source_opcode.kind(), &SassOpcodeKind::Ulop3);
    assert!(matches!(
        &ops[2].kind,
        KernelIrOpKind::WarpElect { dst, .. } if dst == &reg("R0")
    ));
    assert_eq!(ops[2].source_opcode.kind(), &SassOpcodeKind::Vote);
    assert_eq!(ir.unsupported_instruction_count(), 0);
}

#[test]
fn lift_tensor_core_sass_keeps_known_op_families_typed() {
    let module = parse_nvidia_sass(TENSOR_CORE_SASS).expect("tensor SASS should parse");
    let ir = lift_sass_module(&module);
    let function = &ir.functions[0];

    assert!(matches!(
        &function.ops[0].kind,
        KernelIrOpKind::TensorCoreMma {
            opcode,
            element_type: Some(SassTensorElementType::Half),
            signature: None,
            scope: Some(SassTensorScope::Warp),
            operands,
        } if opcode == &SassOpcode::new("HMMA")
            && aggregate_texts(operands).as_slice() == ["R8", "R12", "R16", "R20"]
    ));
    assert!(matches!(
        &function.ops[0].kind,
        KernelIrOpKind::TensorCoreMma { operands, .. }
            if matches!(&operands[0].kind, AggregateOperandKind::Register(register) if register == &reg("R8"))
    ));
    assert!(matches!(
        &function.ops[1].kind,
        KernelIrOpKind::TensorCoreMemory { opcode, operands }
            if opcode == &SassOpcode::new("LDT")
                && aggregate_texts(operands).as_slice() == ["R2", "tmem[UR4]"]
    ));
    assert!(matches!(
        &function.ops[1].kind,
        KernelIrOpKind::TensorCoreMemory { operands, .. }
            if matches!(&operands[1].kind, AggregateOperandKind::Raw { registers, .. }
                if registers.as_slice() == [reg("UR4")])
    ));
    assert!(matches!(
        &function.ops[2].kind,
        KernelIrOpKind::TensorMemoryAccess { opcode, operands }
            if opcode == &SassOpcode::new("UTMALDG")
                && aggregate_texts(operands).as_slice() == ["desc[UR8][R0.64]", "R2"]
    ));
    assert!(matches!(
        &function.ops[2].kind,
        KernelIrOpKind::TensorMemoryAccess { operands, .. }
            if matches!(&operands[0].kind, AggregateOperandKind::Memory(address)
                if matches!(&address.kind, MemoryAddressKind::Descriptor {
                    descriptor,
                    address,
                    address_width: Some(64),
                    offset: None,
                } if descriptor == &reg("UR8") && address == &reg("R0")))
    ));
    assert!(matches!(
        &function.ops[3].kind,
        KernelIrOpKind::WarpGroup { opcode, operands }
            if opcode == &SassOpcode::new("WARPGROUP") && operands.is_empty()
    ));
    assert!(matches!(
        &function.ops[4].kind,
        KernelIrOpKind::WarpGroup { opcode, operands }
            if opcode == &SassOpcode::new("USETMAXREG")
                && aggregate_texts(operands).as_slice() == ["UP0", "0xe8"]
    ));

    let analysis = analyze_sass_ir(&ir);
    let lifted = lift_sass_value_ir(&ir, &analysis);
    let hmma = lifted.functions[0]
        .ops
        .iter()
        .find(|op| op.opcode == SassOpcode::new("HMMA"))
        .expect("HMMA should be lifted");
    assert_eq!(hmma.class, SassLiftedOpClass::TensorCore);
    assert_eq!(hmma.kind, SassLiftedOpKind::TensorCoreMma);
    assert!(matches!(
        &hmma.semantics,
        SassLiftedSemantics::TensorCoreMma { opcode, .. }
            if opcode == &SassOpcode::new("HMMA")
    ));
    assert!(hmma.semantics.to_string().contains("tensor-core-mma"));
}

#[test]
fn lift_bulk_async_support_ops_are_typed() {
    const BULK_ASYNC_SUPPORT_SASS: &str = r#"
        .target sm_120

        .section .text.bulk_async_support,"ax",@progbits
        .global bulk_async_support
bulk_async_support:
.text.bulk_async_support:
        /*0000*/                   LEPC R20, `(.L_x_1) ;                         /* 0x0 */
        /*0010*/                   CALL.ABS.NOINC R2 ;                           /* 0x0 */
.L_x_1:
        /*0020*/                   UTMACMDFLUSH ;                                /* 0x0 */
        /*0030*/                   EXIT ;                                        /* 0x0 */
"#;

    let module = parse_nvidia_sass(BULK_ASYNC_SUPPORT_SASS).expect("SASS should parse");
    let ir = lift_sass_module(&module);
    let function = &ir.functions[0];

    assert_eq!(ir.unsupported_instruction_count(), 0);
    assert!(matches!(
        &function.ops[0].kind,
        KernelIrOpKind::AddressCalc { dst, inputs }
            if dst == &reg("R20") && inputs.len() == 1
    ));
    assert!(matches!(
        &function.ops[1].kind,
        KernelIrOpKind::Call {
            target: None,
            operands,
        } if aggregate_texts(operands).as_slice() == ["R2"]
    ));
    assert!(matches!(
        &function.ops[2].kind,
        KernelIrOpKind::Sync { kind, operands }
            if kind == &SassSyncKind::TensorMemoryCommandFlush && operands.is_empty()
    ));
}

#[test]
fn lift_tensor_core_sass_refines_mma_element_type_from_modifiers() {
    let module = parse_nvidia_sass(TENSOR_CORE_DTYPE_SASS).expect("tensor dtype SASS should parse");
    let ir = lift_sass_module(&module);
    let function = &ir.functions[0];

    assert!(matches!(
        &function.ops[0].kind,
        KernelIrOpKind::TensorCoreMma {
            element_type: Some(SassTensorElementType::Bf16),
            signature: Some(SassTensorMmaSignature {
                shape: Some(SassTensorMmaShape { m: 16, n: 8, k: 16 }),
                output_type: Some(SassTensorElementType::Fp32),
                lhs_type: Some(SassTensorElementType::Bf16),
                rhs_type: Some(SassTensorElementType::Bf16),
                accumulator_type: Some(SassTensorElementType::Fp32),
            }),
            scope: Some(SassTensorScope::Warp),
            ..
        }
    ));
    assert_eq!(
        function.ops[0].source_modifiers[0].kind(),
        &SassModifierKind::TensorShape(SassTensorMmaShape { m: 16, n: 8, k: 16 })
    );
    assert_eq!(
        function.ops[0].source_modifiers[2].kind(),
        &SassModifierKind::Bf16
    );
    assert!(matches!(
        &function.ops[1].kind,
        KernelIrOpKind::TensorCoreMma {
            element_type: Some(SassTensorElementType::F16),
            signature: Some(SassTensorMmaSignature {
                shape: Some(SassTensorMmaShape { m: 16, n: 8, k: 16 }),
                output_type: Some(SassTensorElementType::Fp32),
                lhs_type: Some(SassTensorElementType::F16),
                rhs_type: Some(SassTensorElementType::F16),
                accumulator_type: Some(SassTensorElementType::Fp32),
            }),
            scope: Some(SassTensorScope::Warp),
            ..
        }
    ));
    assert!(matches!(
        &function.ops[2].kind,
        KernelIrOpKind::TensorCoreMma {
            element_type: Some(SassTensorElementType::Tf32),
            signature: Some(SassTensorMmaSignature {
                shape: Some(SassTensorMmaShape { m: 16, n: 8, k: 8 }),
                output_type: Some(SassTensorElementType::Fp32),
                lhs_type: Some(SassTensorElementType::Tf32),
                rhs_type: Some(SassTensorElementType::Tf32),
                accumulator_type: Some(SassTensorElementType::Fp32),
            }),
            scope: Some(SassTensorScope::Warp),
            ..
        }
    ));

    let sm120a_module =
        parse_nvidia_sass(TENSOR_CORE_SM120A_DTYPE_SASS).expect("SM120a dtype SASS should parse");
    let sm120a_ir = lift_sass_module(&sm120a_module);
    let sm120a_function = &sm120a_ir.functions[0];
    assert!(matches!(
        &sm120a_function.ops[0].kind,
        KernelIrOpKind::TensorCoreMma {
            element_type: Some(SassTensorElementType::E2M1),
            signature: Some(SassTensorMmaSignature {
                shape: None,
                output_type: None,
                lhs_type: Some(SassTensorElementType::E2M1),
                rhs_type: Some(SassTensorElementType::E2M1),
                accumulator_type: None,
            }),
            scope: Some(SassTensorScope::Warp),
            ..
        }
    ));
    assert!(matches!(
        &sm120a_function.ops[1].kind,
        KernelIrOpKind::TensorCoreMma {
            element_type: Some(SassTensorElementType::E4M3),
            signature: Some(SassTensorMmaSignature {
                shape: None,
                output_type: None,
                lhs_type: Some(SassTensorElementType::E4M3),
                rhs_type: Some(SassTensorElementType::E4M3),
                accumulator_type: None,
            }),
            scope: Some(SassTensorScope::Warp),
            ..
        }
    ));
    assert!(matches!(
        &sm120a_function.ops[2].kind,
        KernelIrOpKind::TensorCoreMma {
            element_type: Some(SassTensorElementType::E5M2),
            signature: Some(SassTensorMmaSignature {
                shape: None,
                output_type: None,
                lhs_type: Some(SassTensorElementType::E5M2),
                rhs_type: Some(SassTensorElementType::E5M2),
                accumulator_type: None,
            }),
            scope: Some(SassTensorScope::Warp),
            ..
        }
    ));

    let sm90a_module = parse_nvidia_sass(TENSOR_CORE_SM90A_WARPGROUP_DTYPE_SASS)
        .expect("SM90a warpgroup dtype SASS should parse");
    let sm90a_ir = lift_sass_module(&sm90a_module);
    let sm90a_function = &sm90a_ir.functions[0];
    assert!(matches!(
        &sm90a_function.ops[0].kind,
        KernelIrOpKind::TensorCoreMma {
            element_type: Some(SassTensorElementType::E4M3),
            signature: Some(SassTensorMmaSignature {
                shape: None,
                output_type: None,
                lhs_type: Some(SassTensorElementType::E4M3),
                rhs_type: Some(SassTensorElementType::E4M3),
                accumulator_type: None,
            }),
            scope: Some(SassTensorScope::WarpGroup),
            ..
        }
    ));
    assert!(matches!(
        &sm90a_function.ops[1].kind,
        KernelIrOpKind::TensorCoreMma {
            element_type: Some(SassTensorElementType::E5M2),
            signature: Some(SassTensorMmaSignature {
                shape: None,
                output_type: None,
                lhs_type: Some(SassTensorElementType::E5M2),
                rhs_type: Some(SassTensorElementType::E5M2),
                accumulator_type: None,
            }),
            scope: Some(SassTensorScope::WarpGroup),
            ..
        }
    ));

    let analysis = analyze_sass_ir(&ir);
    let lifted = lift_sass_value_ir(&ir, &analysis);
    assert!(
        lifted.functions[0].ops[0]
            .semantics
            .to_string()
            .contains("signature=shape=m16n8k16,output=fp32,lhs=bf16,rhs=bf16,accumulator=fp32")
    );
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
            ref producer,
            ref consumer,
        } if src == &reg("R23") && dst == &reg("R23")
            && matches!(producer, Some(producer)
                if producer.address == 0x0060 && producer.opcode == SassOpcode::new("LD"))
            && matches!(consumer, Some(consumer)
                if consumer.address == 0x0090 && consumer.opcode == SassOpcode::new("FMUL"))
            && pattern.start_address == 0x0060
            && pattern.end_address == 0x0090
            && pattern.source_addresses.as_slice() == [0x0060, 0x0080, 0x0090]
    )));
    assert!(flat.iter().any(|pattern| matches!(
        pattern.kind,
        SassSemanticPatternKind::F32MulAddPair {
            ref mul_dst,
            ref add_dst,
            ..
        } if mul_dst == &reg("R23") && add_dst == &reg("R22")
    )));
    assert!(flat.iter().any(|pattern| matches!(
        pattern.kind,
        SassSemanticPatternKind::WarpReduceSum {
            ref input,
            ref output,
            ref offsets,
            ..
        } if input == &scalar("R22")
            && output == &reg("R7")
            && offsets.as_slice() == [scalar("0x10"), scalar("0x8")]
    )));
    let text = patterns.to_text();
    assert!(text.contains("bf16-widen-bits"));
    assert!(text.contains("warp-reduce-sum"));
}

#[test]
fn address_pair_patterns_match_interleaved_low_high_registers() {
    const INTERLEAVED_LEA_SASS: &str = r#"
        .target sm_120

        .section .text.interleaved_lea,"ax",@progbits
        .global interleaved_lea
interleaved_lea:
.text.interleaved_lea:
        /*0000*/                   LEA R4, P2, R26, UR10, 0x2 ;                 /* 0x0 */
        /*0010*/                   LDCU.64 UR14, c[0x0][0x358] ;                /* 0x0 */
        /*0020*/                   LEA R2, P1, R8, UR11, 0x7 ;                  /* 0x0 */
        /*0030*/                   LEA.HI.X R5, R26, RZ, RZ, 0x2, P2 ;          /* 0x0 */
        /*0040*/                   LEA.HI.X R3, R8, RZ, RZ, 0x7, P1 ;           /* 0x0 */
        /*0050*/                   EXIT ;                                       /* 0x0 */
"#;

    let module = parse_nvidia_sass(INTERLEAVED_LEA_SASS).expect("interleaved LEA should parse");
    let ir = lift_sass_module(&module);
    let patterns = recover_sass_patterns(&ir);
    let address_pairs = patterns.functions[0]
        .patterns
        .iter()
        .filter_map(|pattern| match &pattern.kind {
            SassSemanticPatternKind::AddressPair {
                low_dst, high_dst, ..
            } => Some((low_dst.to_string(), high_dst.to_string())),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(
        address_pairs,
        [
            ("R4".to_string(), "R5".to_string()),
            ("R2".to_string(), "R3".to_string())
        ]
    );
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
            && is_predicate_register(&edge.condition, "P0")
            && is_label_target(&edge.target, ".L_then")
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
    assert_eq!(compare.defines, [reg("P0")]);
    assert_eq!(compare.uses, [reg("R0"), reg("R1")]);
    let add = function
        .dataflow
        .iter()
        .find(|op| op.address == 0x20)
        .expect("add dataflow should exist");
    assert_eq!(add.defines, [reg("R2")]);
    assert_eq!(add.uses, [reg("R0"), reg("R1")]);

    let predicate_use = function
        .reaching_uses
        .iter()
        .find(|use_site| use_site.address == 0x10 && use_site.register == reg("P0"))
        .expect("branch predicate use should have reaching definitions");
    assert!(!predicate_use.reaches_entry);
    assert_eq!(predicate_use.reaching_def_addresses.as_slice(), &[0x0]);

    let joined_r2_use = function
        .reaching_uses
        .iter()
        .find(|use_site| use_site.address == 0x30 && use_site.register == reg("R2"))
        .expect("join use of R2 should have reaching definitions");
    assert!(joined_r2_use.reaches_entry);
    assert_eq!(joined_r2_use.reaching_def_addresses.as_slice(), &[0x20]);

    let local_r2_range = function
        .live_ranges
        .iter()
        .find(|range| range.register == reg("R2") && range.def_address == Some(0x20))
        .expect("R2 definition at 0x20 should have a live range");
    assert_eq!(local_r2_range.use_addresses.as_slice(), &[0x30]);

    let entry_r2_range = function
        .live_ranges
        .iter()
        .find(|range| range.register == reg("R2") && range.def_address.is_none())
        .expect("entry R2 should be live on the branch path");
    assert_eq!(entry_r2_range.start_address, 0x30);
    assert_eq!(entry_r2_range.end_address, 0x30);
    assert_eq!(entry_r2_range.use_addresses.as_slice(), &[0x30]);

    let entry_r2_value = function
        .ssa_values
        .iter()
        .find(|value| value.register == reg("R2") && value.def_address.is_none())
        .expect("entry R2 should have an SSA value");
    assert_eq!(entry_r2_value.origin, SassDataflowSite::Entry);
    assert_eq!(entry_r2_value.origin.address(), None);
    assert_eq!(entry_r2_value.use_addresses.as_slice(), &[0x30]);

    let local_r2_value = function
        .ssa_values
        .iter()
        .find(|value| value.register == reg("R2") && value.def_address == Some(0x20))
        .expect("local R2 definition should have an SSA value");
    assert_eq!(local_r2_value.origin, SassDataflowSite::Instruction(0x20));
    assert_eq!(local_r2_value.origin.address(), Some(0x20));
    assert_eq!(local_r2_value.use_addresses.as_slice(), &[0x30]);

    let joined_r2_edges = function
        .def_use_edges
        .iter()
        .filter(|edge| edge.use_address == 0x30 && edge.register == reg("R2"))
        .collect::<Vec<_>>();
    assert_eq!(joined_r2_edges.len(), 2);
    assert!(
        joined_r2_edges
            .iter()
            .any(|edge| edge.value_id == entry_r2_value.value_id && edge.def_address.is_none())
    );
    assert!(joined_r2_edges.iter().any(|edge| {
        edge.value_id == local_r2_value.value_id
            && edge.def_address == Some(0x20)
            && edge.use_site == SassDataflowSite::Instruction(0x30)
    }));

    let joined_op = function
        .value_ops
        .iter()
        .find(|op| op.address == 0x30)
        .expect("join instruction should have a value-op row");
    assert_eq!(joined_op.opcode, SassOpcode::new("IADD"));
    assert_eq!(joined_op.kind, SassValueOpKind::IntegerAdd);
    assert_eq!(joined_op.input_registers, [reg("R2"), reg("R1")]);
    assert_eq!(joined_op.output_registers, [reg("R3")]);
    assert!(joined_op.input_value_ids.contains(&entry_r2_value.value_id));
    assert!(joined_op.input_value_ids.contains(&local_r2_value.value_id));
    let r3_value = function
        .ssa_values
        .iter()
        .find(|value| value.register == reg("R3") && value.def_address == Some(0x30))
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
        .find(|block| {
            block
                .label
                .as_ref()
                .is_some_and(|label| label.as_str() == ".L_loop")
        })
        .expect("loop header block should exist");
    let body = function
        .blocks
        .iter()
        .find(|block| block.start_address == 0x40)
        .expect("loop body block should exist");
    let done = function
        .blocks
        .iter()
        .find(|block| {
            block
                .label
                .as_ref()
                .is_some_and(|label| label.as_str() == ".L_done")
        })
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
    assert!(is_label_target(&natural_loop.edge_target, ".L_loop"));

    let text = analysis.to_text();
    assert!(text.contains("dominators"));
    assert!(text.contains("natural_loops"));
    assert!(text.contains("header=b1 latch=b2 reachable=true blocks=[b1,b2]"));
}

#[test]
fn analysis_recovers_constructive_region_paths_for_nested_loops() {
    let module = parse_nvidia_sass(NESTED_LOOP_SASS).expect("nested loop fixture should parse");
    let ir = lift_sass_module(&module);
    let analysis = analyze_sass_ir(&ir);
    let function = &analysis.functions[0];

    assert_eq!(function.natural_loops.len(), 2);
    assert!(function.regions.len() >= 3);
    let root = function
        .regions
        .iter()
        .find(|region| region.kind == SassRegionKind::Function)
        .expect("function region should exist");
    assert_eq!(root.depth, 0);
    assert_eq!(root.parent, None);
    assert!(root.opcode_closure.contains(&SassOpcode::new("BRA")));

    let loop_regions = function
        .regions
        .iter()
        .filter(|region| region.kind == SassRegionKind::NaturalLoop)
        .collect::<Vec<_>>();
    assert_eq!(loop_regions.len(), 2);
    let outer = loop_regions
        .iter()
        .find(|region| region.depth == 1)
        .expect("outer loop should be one level under function");
    let inner = loop_regions
        .iter()
        .find(|region| region.depth == 2)
        .expect("inner loop should be nested under the outer loop");
    assert_eq!(inner.parent, Some(outer.id));
    assert!(inner.path.len() > outer.path.len());
    assert!(inner.path.starts_with(&outer.path));
    assert!(outer.path < inner.path);
    let ordered_paths = function
        .regions
        .iter()
        .map(|region| region.path.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(ordered_paths.len(), function.regions.len());
    assert!(inner.opcode_closure.contains(&SassOpcode::new("ISETP")));

    let text = analysis.to_text();
    assert!(text.contains("regions"));
    assert!(text.contains("kind=natural-loop"));
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
            && is_predicate_register(&edge.condition, "P0")
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
        .find(|use_site| use_site.address == 0x20 && use_site.register == reg("R2"))
        .expect("predicated write should use the previous R2 value");
    assert!(!predicated_write_use.reaches_entry);
    assert_eq!(
        predicated_write_use.reaching_def_addresses.as_slice(),
        &[0x0]
    );

    let post_write_use = function
        .reaching_uses
        .iter()
        .find(|use_site| use_site.address == 0x30 && use_site.register == reg("R2"))
        .expect("post-write R2 use should have reaching definitions");
    assert!(!post_write_use.reaches_entry);
    assert_eq!(
        post_write_use.reaching_def_addresses.as_slice(),
        &[0x0, 0x20]
    );

    let original_r2_range = function
        .live_ranges
        .iter()
        .find(|range| range.register == reg("R2") && range.def_address == Some(0x0))
        .expect("original R2 definition should remain live after predicated write");
    assert_eq!(original_r2_range.use_addresses.as_slice(), &[0x20, 0x30]);

    let predicated_r2_range = function
        .live_ranges
        .iter()
        .find(|range| range.register == reg("R2") && range.def_address == Some(0x20))
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
fn all_simple_kernel_fixture_kinds_matches_fixture_definitions() {
    let fixture_kinds = simple_kernel_fixtures()
        .into_iter()
        .map(|fixture| fixture.kind)
        .collect::<Vec<_>>();
    let all_kinds = all_simple_kernel_fixture_kinds();

    assert_eq!(all_kinds.len(), fixture_kinds.len());
    for kind in &all_kinds {
        assert!(fixture_kinds.contains(kind));
    }
}

#[test]
fn all_ptx_decompile_probe_kinds_matches_probe_definitions() {
    let probe_kinds = ptx_decompile_probes()
        .into_iter()
        .map(|probe| probe.kind)
        .collect::<Vec<_>>();
    let all_kinds = all_ptx_decompile_probe_kinds();

    assert_eq!(all_kinds.len(), probe_kinds.len());
    for kind in &all_kinds {
        assert!(probe_kinds.contains(kind));
    }
}

#[test]
fn fixture_coverage_default_runs_all_fixtures_under_managed_artifact_root() {
    let options = DecompileFixtureCoverageOptions::sm120_all_default();

    assert_eq!(
        options.fixture_options.fixtures,
        all_simple_kernel_fixture_kinds()
    );
    assert_eq!(options.fixture_options.compile_arch, "sm_120");
    assert!(
        options
            .coverage_output_dir
            .starts_with(&options.fixture_options.artifact_root)
    );
}

#[test]
fn ptx_probe_default_uses_managed_artifact_root_and_hmma_probe() {
    let options = DecompilePtxProbeOptions::sm120_default();
    let probes = ptx_decompile_probes();
    let hmma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreHmma)
        .expect("HMMA PTX probe should exist");
    let imma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreImma)
        .expect("IMMA PTX probe should exist");
    let dmma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreDmma)
        .expect("DMMA PTX probe should exist");
    let bmma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreBmma)
        .expect("BMMA PTX probe should exist");
    let wgmma_hgmma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreWgmmaHgmma)
        .expect("WGMMA HGMMA PTX probe should exist");
    let wgmma_bgmma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreWgmmaBgmma)
        .expect("WGMMA BGMMA PTX probe should exist");
    let wgmma_igmma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreWgmmaIgmma)
        .expect("WGMMA IGMMA PTX probe should exist");
    let wgmma_qgmma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreWgmmaQgmma)
        .expect("WGMMA QGMMA PTX probe should exist");
    let sm120a_qmma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreSm120aQmma)
        .expect("SM120a QMMA PTX probe should exist");
    let sm120a_omma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreSm120aOmma)
        .expect("SM120a OMMA PTX probe should exist");
    let tcgen05_utchmma_utcimma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreTcgen05UtchmmaUtcimma)
        .expect("tcgen05 UTCHMMA/UTCIMMA PTX probe should exist");
    let tcgen05_utcomma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreTcgen05Utcomma)
        .expect("tcgen05 UTCOMMA PTX probe should exist");
    let tcgen05_utcqmma_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorCoreTcgen05Utcqmma)
        .expect("tcgen05 UTCQMMA PTX probe should exist");
    let bulk_async_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorMemoryBulkAsync)
        .expect("bulk async tensor-memory PTX probe should exist");
    let bulk_reduce_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorMemoryBulkReduce)
        .expect("bulk reduce tensor-memory PTX probe should exist");
    let tma_async_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::TensorMemoryTmaAsync)
        .expect("TMA async tensor-memory PTX probe should exist");
    let warpgroup_register_set_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::WarpGroupRegisterSet)
        .expect("warpgroup register-set PTX probe should exist");
    let scalar_vote_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::ScalarVoteSync)
        .expect("scalar vote-sync PTX probe should exist");
    let scalar_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::ScalarMemoryLogic)
        .expect("scalar memory/logic PTX probe should exist");
    let sm90_scalar_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::ArchitectureSm90Scalar)
        .expect("sm90 scalar PTX probe should exist");
    let atomic_probe = probes
        .iter()
        .find(|probe| probe.kind == PtxDecompileProbeKind::ScalarMemoryAtomic)
        .expect("scalar memory/atomic PTX probe should exist");

    assert_eq!(options.compile_arch, AUTO_COMPILE_ARCH);
    assert_eq!(options.probes, all_ptx_decompile_probe_kinds());
    assert!(
        options
            .artifact_root
            .ends_with("target/cuda-oxide/inference/decompile-probes")
    );
    assert_eq!(hmma_probe.symbol, "tensor_core_hmma_probe");
    assert_eq!(hmma_probe.default_compile_arch, "sm_120");
    assert!(hmma_probe.source.contains("mma.sync.aligned"));
    assert!(hmma_probe.source.contains("st.global.f32"));
    assert_eq!(imma_probe.symbol, "tensor_core_imma_probe");
    assert_eq!(imma_probe.default_compile_arch, "sm_120");
    assert!(
        imma_probe
            .source
            .contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
    );
    assert!(imma_probe.source.contains("st.global.s32"));
    assert_eq!(dmma_probe.symbol, "tensor_core_dmma_probe");
    assert_eq!(dmma_probe.default_compile_arch, "sm_120");
    assert!(
        dmma_probe
            .source
            .contains("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64")
    );
    assert!(dmma_probe.source.contains("st.global.f64"));
    assert_eq!(bmma_probe.symbol, "tensor_core_bmma_probe");
    assert_eq!(bmma_probe.default_compile_arch, "sm_80");
    assert!(
        bmma_probe
            .source
            .contains("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc")
    );
    assert!(bmma_probe.source.contains("st.global.s32"));
    assert_eq!(wgmma_hgmma_probe.symbol, "tensor_core_wgmma_hgmma_probe");
    assert_eq!(wgmma_hgmma_probe.default_compile_arch, "sm_90a");
    assert!(wgmma_hgmma_probe.source.contains(".target sm_90a"));
    assert!(
        wgmma_hgmma_probe
            .source
            .contains("wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16")
    );
    assert!(
        wgmma_hgmma_probe
            .source
            .contains("wgmma.fence.sync.aligned")
    );
    assert!(
        wgmma_hgmma_probe
            .source
            .contains("wgmma.commit_group.sync.aligned")
    );
    assert!(
        wgmma_hgmma_probe
            .source
            .contains("wgmma.wait_group.sync.aligned 0")
    );
    assert_eq!(wgmma_bgmma_probe.symbol, "tensor_core_wgmma_bgmma_probe");
    assert_eq!(wgmma_bgmma_probe.default_compile_arch, "sm_90a");
    assert!(wgmma_bgmma_probe.source.contains(".target sm_90a"));
    assert!(
        wgmma_bgmma_probe
            .source
            .contains("wgmma.mma_async.sync.aligned.m64n8k256.s32.b1.b1.and.popc")
    );
    assert_eq!(wgmma_igmma_probe.symbol, "tensor_core_wgmma_igmma_probe");
    assert_eq!(wgmma_igmma_probe.default_compile_arch, "sm_90a");
    assert!(wgmma_igmma_probe.source.contains(".target sm_90a"));
    assert!(
        wgmma_igmma_probe
            .source
            .contains("wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8")
    );
    assert_eq!(wgmma_qgmma_probe.symbol, "tensor_core_wgmma_qgmma_probe");
    assert_eq!(wgmma_qgmma_probe.default_compile_arch, "sm_90a");
    assert!(wgmma_qgmma_probe.source.contains(".target sm_90a"));
    assert!(
        wgmma_qgmma_probe
            .source
            .contains("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3")
    );
    assert_eq!(sm120a_qmma_probe.symbol, "tensor_core_sm120a_qmma_probe");
    assert_eq!(sm120a_qmma_probe.default_compile_arch, "sm_120a");
    assert!(sm120a_qmma_probe.source.contains(".target sm_120a"));
    assert!(
        sm120a_qmma_probe
            .source
            .contains("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32")
    );
    assert!(sm120a_qmma_probe.source.contains("st.global.f32"));
    assert_eq!(sm120a_omma_probe.symbol, "tensor_core_sm120a_omma_probe");
    assert_eq!(sm120a_omma_probe.default_compile_arch, "sm_120a");
    assert!(sm120a_omma_probe.source.contains(".target sm_120a"));
    assert!(sm120a_omma_probe.source.contains(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"
    ));
    assert!(sm120a_omma_probe.source.contains("st.global.f32"));
    assert_eq!(
        tcgen05_utchmma_utcimma_probe.symbol,
        "tensor_core_tcgen05_utchmma_utcimma_probe"
    );
    assert_eq!(
        tcgen05_utchmma_utcimma_probe.default_compile_arch,
        "sm_100a"
    );
    assert!(
        tcgen05_utchmma_utcimma_probe
            .source
            .contains(".target sm_100a")
    );
    assert!(
        tcgen05_utchmma_utcimma_probe
            .source
            .contains("tcgen05.mma.cta_group::1.kind::f16")
    );
    assert!(
        tcgen05_utchmma_utcimma_probe
            .source
            .contains("tcgen05.mma.cta_group::1.kind::i8")
    );
    assert!(
        tcgen05_utchmma_utcimma_probe
            .source
            .contains("st.global.u32")
    );
    assert_eq!(
        tcgen05_utcomma_probe.symbol,
        "tensor_core_tcgen05_utcomma_probe"
    );
    assert_eq!(tcgen05_utcomma_probe.default_compile_arch, "sm_100a");
    assert!(tcgen05_utcomma_probe.source.contains(".target sm_100a"));
    assert!(
        tcgen05_utcomma_probe
            .source
            .contains("tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32")
    );
    assert!(tcgen05_utcomma_probe.source.contains("st.global.u32"));
    assert_eq!(
        tcgen05_utcqmma_probe.symbol,
        "tensor_core_tcgen05_utcqmma_probe"
    );
    assert_eq!(tcgen05_utcqmma_probe.default_compile_arch, "sm_100a");
    assert!(tcgen05_utcqmma_probe.source.contains(".target sm_100a"));
    assert!(
        tcgen05_utcqmma_probe
            .source
            .contains("tcgen05.mma.cta_group::1.kind::f8f6f4")
    );
    assert!(tcgen05_utcqmma_probe.source.contains("st.global.u32"));
    assert_eq!(bulk_async_probe.symbol, "tensor_memory_bulk_async_probe");
    assert_eq!(bulk_async_probe.default_compile_arch, "sm_120");
    assert!(bulk_async_probe.source.contains(".target sm_120"));
    assert!(
        bulk_async_probe
            .source
            .contains("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes")
    );
    assert!(
        bulk_async_probe
            .source
            .contains("cp.async.bulk.prefetch.L2.global")
    );
    assert!(
        bulk_async_probe
            .source
            .contains("cp.async.bulk.global.shared::cta.bulk_group")
    );
    assert_eq!(bulk_reduce_probe.symbol, "tensor_memory_bulk_reduce_probe");
    assert_eq!(bulk_reduce_probe.default_compile_arch, "sm_120");
    assert!(bulk_reduce_probe.source.contains(".target sm_120"));
    assert!(
        bulk_reduce_probe
            .source
            .contains("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32")
    );
    assert!(bulk_reduce_probe.source.contains(
        "cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32"
    ));
    assert_eq!(tma_async_probe.symbol, "tensor_memory_tma_async_probe");
    assert_eq!(tma_async_probe.default_compile_arch, "sm_120");
    assert!(tma_async_probe.source.contains(".target sm_120"));
    assert!(
        tma_async_probe
            .source
            .contains("cp.async.bulk.tensor.1d.shared::cluster.global")
    );
    assert!(
        tma_async_probe
            .source
            .contains("cp.async.bulk.prefetch.tensor.1d.L2.global")
    );
    assert!(
        tma_async_probe
            .source
            .contains("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group")
    );
    assert!(
        tma_async_probe
            .source
            .contains("cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.bulk_group")
    );
    assert_eq!(
        warpgroup_register_set_probe.symbol,
        "warpgroup_register_set_probe"
    );
    assert_eq!(warpgroup_register_set_probe.default_compile_arch, "sm_90a");
    assert!(
        warpgroup_register_set_probe
            .source
            .contains(".maxntid 384, 1, 1")
    );
    assert!(
        warpgroup_register_set_probe
            .source
            .contains(".minnctapersm 1")
    );
    assert!(
        warpgroup_register_set_probe
            .source
            .contains("setmaxnreg.inc.sync.aligned.u32 232")
    );
    assert!(
        warpgroup_register_set_probe
            .source
            .contains("setmaxnreg.dec.sync.aligned.u32 40")
    );
    assert_eq!(scalar_vote_probe.symbol, "scalar_vote_sync_probe");
    assert_eq!(scalar_vote_probe.default_compile_arch, "sm_75");
    assert!(scalar_vote_probe.source.contains("vote.sync.all.pred"));
    assert!(scalar_vote_probe.source.contains("st.global.u32"));
    assert_eq!(scalar_probe.symbol, "scalar_memory_logic_probe");
    assert_eq!(scalar_probe.default_compile_arch, "sm_75");
    assert!(scalar_probe.source.contains(".target sm_75"));
    assert!(scalar_probe.source.contains("ld.global.nc.u32"));
    assert!(scalar_probe.source.contains("st.local.u32"));
    assert!(scalar_probe.source.contains("ld.local.u32"));
    assert!(scalar_probe.source.contains("lop3.b32"));
    assert!(scalar_probe.source.contains("and.pred"));
    assert!(scalar_probe.source.contains("fma.rn.f32"));
    assert!(scalar_probe.source.contains("setp.gt.f32"));
    assert_eq!(sm90_scalar_probe.symbol, "scalar_memory_logic_probe");
    assert_eq!(sm90_scalar_probe.default_compile_arch, "sm_90");
    assert!(sm90_scalar_probe.source.contains(".target sm_75"));
    assert!(sm90_scalar_probe.source.contains("ld.global.nc.u32"));
    assert!(sm90_scalar_probe.source.contains("fma.rn.f32"));
    assert_eq!(atomic_probe.symbol, "scalar_memory_atomic_probe");
    assert_eq!(atomic_probe.default_compile_arch, "sm_75");
    assert!(atomic_probe.source.contains(".target sm_75"));
    assert!(atomic_probe.source.contains("atom.add.u32"));
    assert!(atomic_probe.source.contains("atom.global.add.u32"));
    assert!(atomic_probe.source.contains("red.global.add.u32"));
    assert!(atomic_probe.source.contains("st.volatile.local.u32"));
    assert!(atomic_probe.source.contains("ld.volatile.local.u32"));
    assert!(atomic_probe.source.contains("or.pred"));
    assert!(atomic_probe.source.contains("xor.pred"));
    assert_eq!(bmma_probe.compile_arch_for(AUTO_COMPILE_ARCH), "sm_80");
    assert_eq!(bmma_probe.compile_arch_for("sm_120"), "sm_120");
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
    assert!(
        report
            .scanned_architectures
            .iter()
            .any(|architecture| architecture == &SassArchitecture::sm(120))
    );
    assert_eq!(report.parse_error_count, 0);
    assert!(report.known_opcode_count > 0);
    assert!(report.locally_mapped_opcode_count > 0);
    assert!(report.known_unobserved_opcode_count > 0);
    assert_eq!(
        report.opcode_probe_target_count,
        report.known_unobserved_opcode_count
    );
    assert_eq!(report.sm120_tensor_core_required_count, 5);
    assert_eq!(report.sm120_tensor_core_supported_count, 0);
    assert_eq!(report.sm120_tensor_core_missing_count, 5);
    assert_eq!(report.known_unmapped_opcode_count, 0);
    assert!(report.observed_unregistered_opcode_count > 0);
    assert!(report.observed_unmapped_opcode_count > 0);
    assert_eq!(report.unsupported_instruction_count, 1);
    assert!(
        report
            .dataflow
            .iter()
            .any(|op| op.function == SassSymbol::new("sass_fixture_i32_add"))
    );
    assert!(
        report
            .cfg_blocks
            .iter()
            .any(|block| block.label == Some(SassSymbol::new(".L_x_0")))
    );
    assert!(
        report
            .opcode_counts
            .iter()
            .any(|count| count.opcode == SassOpcode::new("MYSTERY") && count.count == 1)
    );
    assert!(
        report
            .opcode_signature_counts
            .iter()
            .any(|count| { count.signature.opcode == SassOpcode::new("IADD") && count.count > 0 })
    );
    let mystery_catalog = report
        .opcode_catalog
        .iter()
        .find(|entry| entry.opcode == SassOpcode::new("MYSTERY"))
        .expect("unsupported opcode should be catalogued");
    assert_eq!(mystery_catalog.support, SassOpcodeSupport::Unsupported);
    assert_eq!(
        mystery_catalog.coverage,
        SassOpcodeCoverageState::ObservedUnregisteredUnmapped
    );
    assert!(!mystery_catalog.known);
    assert!(mystery_catalog.observed);
    assert!(!mystery_catalog.locally_mapped);
    assert_eq!(mystery_catalog.unsupported_count, 1);
    assert!(
        mystery_catalog
            .kinds
            .iter()
            .any(|kind| kind == &SassOpcodeCatalogKind::Unsupported)
    );
    let iadd_catalog = report
        .opcode_catalog
        .iter()
        .find(|entry| entry.opcode == SassOpcode::new("IADD"))
        .expect("IADD should be catalogued");
    assert_eq!(iadd_catalog.support, SassOpcodeSupport::Mapped);
    assert!(iadd_catalog.known);
    assert!(iadd_catalog.observed);
    assert!(iadd_catalog.locally_mapped);
    assert_eq!(
        iadd_catalog.coverage,
        SassOpcodeCoverageState::KnownObservedMapped
    );
    assert!(
        iadd_catalog
            .source_formats
            .iter()
            .any(|format| format == &SassCoverageSourceFormat::Cuobjdump)
    );
    assert!(
        iadd_catalog
            .source_formats
            .iter()
            .any(|format| format == &SassCoverageSourceFormat::Sass)
    );
    assert!(
        iadd_catalog
            .signatures
            .iter()
            .any(|signature| signature.opcode == SassOpcode::new("IADD"))
    );
    let hmma_catalog = report
        .opcode_catalog
        .iter()
        .find(|entry| entry.opcode == SassOpcode::new("HMMA"))
        .expect("known tensor-core opcode should be catalogued without local observation");
    assert!(hmma_catalog.known);
    assert!(!hmma_catalog.observed);
    assert!(hmma_catalog.locally_mapped);
    assert_eq!(hmma_catalog.support, SassOpcodeSupport::Mapped);
    assert_eq!(
        hmma_catalog.coverage,
        SassOpcodeCoverageState::KnownUnobservedMapped
    );
    assert!(
        hmma_catalog
            .architectures
            .iter()
            .any(|architecture| architecture == &SassArchitecture::sm(120))
    );
    assert!(
        hmma_catalog
            .classes
            .iter()
            .any(|class| class == &SassOpcodeCatalogClass::TensorCore)
    );
    assert!(hmma_catalog.known_sources.iter().any(|source| {
        source == &SassOpcodeCatalogSource::NvidiaCudaBinaryUtilitiesInstructionReference
    }));
    let hmma_probe = report
        .opcode_probe_targets
        .iter()
        .find(|target| target.opcode == SassOpcode::new("HMMA"))
        .expect("known unobserved HMMA should be a probe target");
    assert_eq!(hmma_probe.priority, 90);
    assert!(hmma_probe.locally_mapped);
    assert_eq!(
        hmma_probe.recommended_action,
        SassOpcodeProbeAction::GenerateSassArtifact
    );
    assert_eq!(
        hmma_probe.reason,
        SassOpcodeProbeReason::TensorCoreMappedUnobserved
    );
    assert!(
        hmma_probe
            .architectures
            .iter()
            .any(|architecture| architecture == &SassArchitecture::sm(120))
    );
    assert!(
        hmma_probe
            .matching_scanned_architectures
            .iter()
            .any(|architecture| architecture == &SassArchitecture::sm(120))
    );
    let atom_probe = report
        .opcode_probe_targets
        .iter()
        .find(|target| target.opcode == SassOpcode::new("ATOM"))
        .expect("known unobserved ATOM should be a probe target");
    assert_eq!(atom_probe.priority, 70);
    assert!(atom_probe.locally_mapped);
    assert_eq!(
        atom_probe.recommended_action,
        SassOpcodeProbeAction::GenerateSassArtifact
    );
    assert_eq!(
        atom_probe.reason,
        SassOpcodeProbeReason::ArchitectureSpecificMappedUnobserved
    );
    assert!(
        atom_probe
            .architectures
            .iter()
            .any(|architecture| architecture == &SassArchitecture::sm(120))
    );
    assert!(atom_probe.known_sources.iter().any(|source| {
        source == &SassOpcodeCatalogSource::NvidiaCudaBinaryUtilitiesInstructionReference
    }));
    assert!(
        atom_probe
            .matching_scanned_architectures
            .iter()
            .any(|architecture| architecture == &SassArchitecture::sm(120))
    );
    let bgmma_probe = report
        .opcode_probe_targets
        .iter()
        .find(|target| target.opcode == SassOpcode::new("BGMMA"))
        .expect("known unobserved sm90 BGMMA should be a probe target");
    assert_eq!(bgmma_probe.priority, 80);
    assert!(bgmma_probe.matching_scanned_architectures.is_empty());
    assert_eq!(
        bgmma_probe.recommended_action,
        SassOpcodeProbeAction::ScanArchitectureArtifact
    );
    assert_eq!(
        bgmma_probe.reason,
        SassOpcodeProbeReason::ArchitectureArtifactNotScanned
    );
    assert!(
        !report
            .opcode_probe_targets
            .iter()
            .any(|target| target.opcode == SassOpcode::new("IADD"))
    );
    assert!(
        report
            .unsupported_instructions
            .iter()
            .any(
                |instruction| instruction.opcode == SassOpcode::new("MYSTERY")
                    && instruction.reason == SassUnsupportedReason::NoLocalMapping
                    && instruction.source_text.contains("MYSTERY")
            )
    );
    assert!(report.files.iter().any(
        |file| file.sass_path.file_name().and_then(|name| name.to_str())
            == Some("simple.cuobjdump.sass")
            && matches!(
                file.target.as_ref().and_then(SassTarget::architecture),
                Some(architecture) if architecture == SassArchitecture::sm(120)
            )
    ));
    assert!(report.summary_path.exists());
    assert!(report.files_path.exists());
    assert!(report.opcode_catalog_path.exists());
    assert!(report.opcode_probe_targets_path.exists());
    assert!(report.sm120_tensor_core_support_path.exists());
    assert!(report.opcode_frequency_path.exists());
    assert!(report.opcode_signature_frequency_path.exists());
    assert!(report.semantic_patterns_path.exists());
    assert!(report.semantic_pattern_frequency_path.exists());
    assert!(report.cfg_blocks_path.exists());
    assert!(report.cfg_edges_path.exists());
    assert!(report.dominators_path.exists());
    assert!(report.natural_loops_path.exists());
    assert!(report.regions_path.exists());
    assert!(report.dataflow_path.exists());
    assert!(report.reaching_uses_path.exists());
    assert!(report.ssa_values_path.exists());
    assert!(report.def_use_edges_path.exists());
    assert!(report.value_ops_path.exists());
    assert!(report.lifted_ops_path.exists());
    assert!(report.live_ranges_path.exists());
    assert!(report.memory_accesses_path.exists());
    assert!(report.unsupported_instructions_path.exists());
    let summary = fs::read_to_string(&report.summary_path).expect("coverage summary should read");
    assert!(summary.contains("scanned_architectures=sm120"));
    assert!(summary.contains("top_opcode_probe_targets"));
    assert!(summary.contains("HMMA\t90\tgenerate-sass-artifact"));
    assert!(report.cfg_block_count > 0);
    assert!(report.cfg_edge_count > 0);
    assert!(report.dominator_block_count > 0);
    assert!(report.natural_loop_count > 0);
    assert!(report.region_count > 0);
    assert!(
        report
            .cfg_blocks
            .iter()
            .any(|block| block.terminator == SassBlockTerminator::Branch)
    );
    assert!(report.cfg_edges.iter().any(|edge| {
        edge.kind == SassCfgEdgeKind::Branch
            && edge
                .target
                .as_ref()
                .and_then(ControlTarget::label_name)
                .is_some()
    }));
    assert!(report.natural_loops.iter().any(|natural_loop| {
        natural_loop
            .edge_target
            .as_ref()
            .and_then(ControlTarget::label_name)
            .is_some()
    }));
    assert!(report.dataflow_op_count > 0);
    assert!(report.reaching_use_count > 0);
    assert!(report.ssa_value_count > 0);
    assert!(report.def_use_edge_count > 0);
    assert!(report.value_op_count > 0);
    assert!(report.lifted_op_count > 0);
    assert!(report.live_range_count > 0);
    assert!(report.memory_access_count > 0);
    assert!(report.dataflow.iter().any(|op| {
        op.address == 0x10 && op.defines == [reg("R2")] && op.uses.contains(&reg("R0"))
    }));
    assert!(
        report
            .reaching_uses
            .iter()
            .any(|use_site| use_site.register == reg("R2"))
    );
    assert!(
        report
            .ssa_values
            .iter()
            .any(|value| value.register == reg("R2"))
    );
    assert!(
        report
            .def_use_edges
            .iter()
            .any(|edge| edge.register == reg("R2"))
    );
    assert!(
        report
            .live_ranges
            .iter()
            .any(|range| range.register == reg("R2"))
    );
    assert!(report.value_ops.iter().any(|op| {
        op.opcode == SassOpcode::new("LD")
            && op.kind == SassValueOpKind::Load
            && op.output_registers == [reg("R2")]
    }));
    assert!(report.memory_accesses.iter().any(|access| {
        access.kind == SassMemoryAccessKind::Load
            && access.space == MemorySpace::Descriptor
            && access.address_base.as_ref() == Some(&MemoryAddressBase::Descriptor(reg("UR4")))
            && matches!(
                &access.memory_address.kind,
                MemoryAddressKind::Descriptor {
                    descriptor,
                    address,
                    address_width: Some(64),
                    offset: None,
                } if descriptor == &reg("UR4") && address == &reg("R0")
            )
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
            .any(|op| op.class == SassLiftedOpClass::Memory && op.kind == SassLiftedOpKind::Load)
    );
    assert!(report.lifted_ops.iter().any(|op| {
        op.class == SassLiftedOpClass::Memory
            && op.kind == SassLiftedOpKind::Load
            && matches!(
                &op.semantics,
                SassLiftedSemantics::Load {
                    space: MemorySpace::Descriptor,
                    dst,
                    address,
                    width_bits: None,
                    modifiers,
                } if dst == &reg("R2")
                    && address.to_string() == "desc[UR4][R0.64]"
                    && modifiers.iter().any(|modifier| modifier.to_string() == "E")
            )
            && op.source_operands.iter().any(|operand| {
                matches!(
                    &operand.kind,
                    AggregateOperandKind::Memory(address)
                        if matches!(
                            &address.kind,
                            MemoryAddressKind::Descriptor {
                                descriptor,
                                address,
                                address_width: Some(64),
                                offset: None,
                            } if descriptor == &reg("UR4") && address == &reg("R0")
                        )
                )
            })
    }));
    let lifted_ops_tsv =
        fs::read_to_string(&report.lifted_ops_path).expect("lifted ops TSV should be readable");
    assert!(lifted_ops_tsv.starts_with(
        "sass_path\tfunction\taddress\tblock_id\tpredicate\topcode\tclass\tkind\tsemantics"
    ));
    assert!(
        lifted_ops_tsv
            .lines()
            .next()
            .is_some_and(|header| header.ends_with("\tsource_text"))
    );
    let dataflow_tsv =
        fs::read_to_string(&report.dataflow_path).expect("dataflow TSV should be readable");
    assert!(dataflow_tsv.starts_with("sass_path\tfunction\taddress\tdefines\tuses\tsource_text"));
    let value_ops_tsv =
        fs::read_to_string(&report.value_ops_path).expect("value ops TSV should be readable");
    assert!(
        value_ops_tsv
            .lines()
            .next()
            .is_some_and(|header| header.ends_with("\tsource_text"))
    );
    let memory_accesses_tsv = fs::read_to_string(&report.memory_accesses_path)
        .expect("memory accesses TSV should be readable");
    assert!(
        memory_accesses_tsv
            .lines()
            .next()
            .is_some_and(|header| header.ends_with("\tsource_text"))
    );
    let unsupported_tsv = fs::read_to_string(&report.unsupported_instructions_path)
        .expect("unsupported instructions TSV should be readable");
    assert!(
        unsupported_tsv.starts_with("sass_path\tfunction\taddress\topcode\treason\tsource_text")
    );
    let opcode_catalog_tsv =
        fs::read_to_string(&report.opcode_catalog_path).expect("opcode catalog TSV should read");
    assert!(opcode_catalog_tsv.starts_with(
        "opcode\tknown\tobserved\tlocally_mapped\tinstruction_count\tsignature_count\tsignatures\tsource_formats\tarchitectures\tobserved_architectures\tknown_sources\tclasses\tkinds\tsupport\tcoverage\tunsupported_count"
    ));
    assert!(opcode_catalog_tsv.contains("MYSTERY"));
    assert!(opcode_catalog_tsv.contains("HMMA"));
    let opcode_probe_targets_tsv = fs::read_to_string(&report.opcode_probe_targets_path)
        .expect("opcode probe target TSV should read");
    assert!(opcode_probe_targets_tsv.starts_with(
        "opcode\tpriority\tlocally_mapped\tarchitectures\tmatching_scanned_architectures\tclasses\tkinds\tknown_sources\trecommended_action\treason"
    ));
    assert!(opcode_probe_targets_tsv.contains("HMMA"));
    assert!(opcode_probe_targets_tsv.contains("tensor-core opcode is mapped"));
    assert!(opcode_probe_targets_tsv.contains("generate-sass-artifact"));
    assert!(report.regions.iter().any(|region| {
        region.kind == SassRegionKind::Function
            && region.opcode_closure.contains(&SassOpcode::new("BRA"))
    }));
    let regions_tsv = fs::read_to_string(&report.regions_path).expect("regions TSV should read");
    assert!(regions_tsv.starts_with(
        "sass_path\tfunction\tregion_id\tparent_region\tchildren\tdepth\tpath\tlocal_rank\tkind"
    ));
    assert!(regions_tsv.contains("function"));
    assert!(regions_tsv.contains("opcode_closure"));
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
    let hmma_support = report
        .sm120_tensor_core_support
        .iter()
        .find(|entry| entry.opcode == SassOpcode::new("HMMA"))
        .expect("HMMA should be part of the SM120 tensor-core support contract");
    assert_eq!(
        hmma_support.required_architecture,
        SassArchitecture::sm(120)
    );
    assert_eq!(
        hmma_support.status,
        Sm120TensorCoreSupportStatus::Unobserved
    );
    let qmma_support = report
        .sm120_tensor_core_support
        .iter()
        .find(|entry| entry.opcode == SassOpcode::new("QMMA"))
        .expect("QMMA should be part of the SM120a tensor-core support contract");
    assert_eq!(
        qmma_support.required_architecture,
        SassArchitecture::sm_a(120)
    );
    assert_eq!(
        qmma_support.status,
        Sm120TensorCoreSupportStatus::Unobserved
    );
    let summary = fs::read_to_string(&report.summary_path).expect("summary should read");
    assert!(summary.contains("sm120_tensor_core_required=5"));
    assert!(summary.contains("sm120_tensor_core_supported=0"));
    assert!(summary.contains("sm120_tensor_core_missing=5"));
    let support_tsv = fs::read_to_string(&report.sm120_tensor_core_support_path)
        .expect("SM120 tensor-core support TSV should read");
    assert!(support_tsv.starts_with(
        "opcode\trequired_architecture\tfamily\trequirement\tobserved\tlocally_mapped"
    ));
    assert!(support_tsv.contains("QMMA\tsm120a"));
}

#[test]
fn coverage_scan_requires_tensor_core_ops_on_exact_sm120_architecture() {
    let supported_root = unique_test_dir("coverage-sm120-tensor-core-supported");
    let supported_input = supported_root.join("input");
    let supported_output = supported_root.join("output");
    fs::create_dir_all(&supported_input).expect("supported input dir should be created");
    fs::write(
        supported_input.join("sm120-mma.sass"),
        SM120_TENSOR_CORE_MMA_SASS,
    )
    .expect("SM120 tensor-core SASS should be written");
    fs::write(
        supported_input.join("sm120a-mma.sass"),
        SM120A_TENSOR_CORE_MMA_SASS,
    )
    .expect("SM120a tensor-core SASS should be written");

    let supported_report = run_sass_coverage_scan(&SassCoverageOptions {
        root: supported_input,
        output_dir: supported_output,
    })
    .expect("supported coverage scan should complete");

    assert_eq!(supported_report.sm120_tensor_core_required_count, 5);
    assert_eq!(supported_report.sm120_tensor_core_supported_count, 5);
    assert_eq!(supported_report.sm120_tensor_core_missing_count, 0);
    assert!(
        supported_report
            .sm120_tensor_core_support
            .iter()
            .all(|entry| {
                entry.status == Sm120TensorCoreSupportStatus::Supported
                    && entry.observed
                    && entry.locally_mapped
            })
    );
    let qmma_support = supported_report
        .sm120_tensor_core_support
        .iter()
        .find(|entry| entry.opcode == SassOpcode::new("QMMA"))
        .expect("QMMA support row should exist");
    assert_eq!(
        qmma_support.required_architecture,
        SassArchitecture::sm_a(120)
    );
    assert!(
        qmma_support
            .observed_architectures
            .contains(&SassArchitecture::sm_a(120))
    );

    let wrong_arch_root = unique_test_dir("coverage-sm120-tensor-core-wrong-arch");
    let wrong_arch_input = wrong_arch_root.join("input");
    let wrong_arch_output = wrong_arch_root.join("output");
    fs::create_dir_all(&wrong_arch_input).expect("wrong-arch input dir should be created");
    fs::write(
        wrong_arch_input.join("sm120-mma.sass"),
        SM120_TENSOR_CORE_MMA_SASS,
    )
    .expect("SM120 tensor-core SASS should be written");
    fs::write(
        wrong_arch_input.join("wrong-arch-mma.sass"),
        SM120_WRONG_ARCH_TENSOR_CORE_MMA_SASS,
    )
    .expect("wrong-arch tensor-core SASS should be written");

    let wrong_arch_report = run_sass_coverage_scan(&SassCoverageOptions {
        root: wrong_arch_input,
        output_dir: wrong_arch_output,
    })
    .expect("wrong-arch coverage scan should complete");

    assert_eq!(wrong_arch_report.sm120_tensor_core_required_count, 5);
    assert_eq!(wrong_arch_report.sm120_tensor_core_supported_count, 3);
    assert_eq!(wrong_arch_report.sm120_tensor_core_missing_count, 2);
    for opcode in ["QMMA", "OMMA"] {
        let entry = wrong_arch_report
            .sm120_tensor_core_support
            .iter()
            .find(|entry| entry.opcode == SassOpcode::new(opcode))
            .expect("SM120a-only support row should exist");
        assert_eq!(entry.required_architecture, SassArchitecture::sm_a(120));
        assert_eq!(
            entry.status,
            Sm120TensorCoreSupportStatus::MissingArchitectureArtifact
        );
        assert!(
            entry
                .observed_architectures
                .contains(&SassArchitecture::sm(120))
        );
        assert!(
            !entry
                .observed_architectures
                .contains(&SassArchitecture::sm_a(120))
        );
    }
}

#[test]
fn coverage_scan_preserves_typed_semantic_pattern_rows() {
    let root = unique_test_dir("coverage_patterns");
    let input = root.join("input");
    let output = root.join("output");
    fs::create_dir_all(&input).expect("test input dir should be created");
    fs::write(input.join("rows17.sass"), ROWS17_SLICE).expect("rows17 SASS should be written");

    let report = run_sass_coverage_scan(&SassCoverageOptions {
        root: input,
        output_dir: output,
    })
    .expect("coverage scan should complete");

    assert!(report.semantic_pattern_count > 0);
    assert!(report.semantic_patterns.iter().any(|pattern| {
        pattern.confidence == SassPatternConfidence::ExactOpcodeSequence
            && matches!(
                &pattern.kind,
                SassSemanticPatternKind::Bf16WidenBits { src, dst, .. }
                    if src == &reg("R23") && dst == &reg("R23")
            )
    }));
    assert!(report.semantic_patterns.iter().any(|pattern| {
        pattern.confidence == SassPatternConfidence::HeuristicDataflow
            && matches!(
                &pattern.kind,
                SassSemanticPatternKind::WarpReduceSum { output, .. }
                    if output == &reg("R7")
            )
    }));
    assert!(report.semantic_pattern_counts.iter().any(|count| {
        count.category == SassSemanticPatternCategory::Bf16WidenBits && count.count == 1
    }));
    assert!(report.semantic_pattern_counts.iter().any(|count| {
        count.category == SassSemanticPatternCategory::WarpReduceSum && count.count == 1
    }));
    assert!(
        report
            .value_ops
            .iter()
            .any(|op| is_predicate_register(&op.predicate, "P0"))
    );
    assert!(
        report
            .lifted_ops
            .iter()
            .any(|op| is_predicate_register(&op.predicate, "P0"))
    );

    let pattern_tsv =
        fs::read_to_string(&report.semantic_patterns_path).expect("patterns TSV should read");
    assert!(pattern_tsv.starts_with("sass_path\tfunction\tstart_address\tend_address\tkind"));
    assert!(pattern_tsv.contains("bf16-widen-bits"));
    assert!(pattern_tsv.contains("exact-opcode-sequence"));
    let pattern_frequency_tsv = fs::read_to_string(&report.semantic_pattern_frequency_path)
        .expect("pattern frequency TSV should read");
    assert!(pattern_frequency_tsv.starts_with("semantic_pattern\tcount"));
    assert!(pattern_frequency_tsv.contains("warp-reduce-sum\t1"));
}

#[test]
fn coverage_scan_preserves_typed_parse_error() {
    const ORPHAN_INSTRUCTION_SASS: &str = r#"
        .target sm_120
        /*0000*/                   MOV R0, R1 ;                                  /* 0x0 */
    "#;

    let root = unique_test_dir("coverage-parse-error");
    let input = root.join("input");
    let output = root.join("output");
    fs::create_dir_all(&input).expect("test input dir should be created");
    fs::write(input.join("orphan.sass"), ORPHAN_INSTRUCTION_SASS)
        .expect("orphan test SASS should be written");

    let report = run_sass_coverage_scan(&SassCoverageOptions {
        root: input,
        output_dir: output,
    })
    .expect("coverage scan should complete with parse errors reported");

    assert_eq!(report.files.len(), 1);
    assert_eq!(report.parsed_file_count, 0);
    assert_eq!(report.parse_error_count, 1);
    let parse_error: &SassParseError = report.files[0]
        .parse_error
        .as_ref()
        .expect("file report should keep the typed parse error");
    assert!(
        parse_error
            .to_string()
            .contains("instruction appears before a function/global header")
    );

    let files_tsv = fs::read_to_string(&report.files_path).expect("files TSV should be readable");
    assert!(files_tsv.contains("parse-error"));
    assert!(files_tsv.contains("instruction appears before a function/global header"));
}

#[test]
fn coverage_comparison_reports_resolved_probe_targets() {
    let root = unique_test_dir("coverage_compare");
    let baseline = root.join("baseline");
    let candidate = root.join("candidate");
    let output = root.join("output");
    fs::create_dir_all(&baseline).expect("baseline dir should be created");
    fs::create_dir_all(&candidate).expect("candidate dir should be created");
    fs::write(baseline.join("unsupported.sass"), UNSUPPORTED_SASS)
        .expect("baseline SASS should be written");
    fs::write(candidate.join("simple.sass"), SIMPLE_SASS)
        .expect("candidate SASS should be written");

    let report = run_sass_coverage_comparison(&SassCoverageComparisonOptions {
        baseline_root: baseline,
        candidate_root: candidate,
        output_dir: output,
    })
    .expect("coverage comparison should complete");

    assert!(report.summary_path.exists());
    assert!(report.opcode_delta_path.exists());
    assert!(report.resolved_probe_targets_path.exists());
    assert!(report.new_probe_targets_path.exists());
    assert!(report.newly_observed_opcode_count > 0);
    assert!(
        report
            .opcode_deltas
            .iter()
            .any(|delta| delta.opcode == SassOpcode::new("IADD")
                && delta.change == SassCoverageOpcodeChange::NewlyObserved)
    );
    let resolved_iadd = report
        .resolved_probe_targets
        .iter()
        .find(|target| target.opcode == SassOpcode::new("IADD"))
        .expect("candidate IADD should resolve a baseline probe target");
    assert_eq!(
        resolved_iadd.baseline_coverage,
        SassOpcodeCoverageState::KnownUnobservedMapped
    );
    assert_eq!(
        resolved_iadd.candidate_coverage,
        SassOpcodeCoverageState::KnownObservedMapped
    );
    assert!(resolved_iadd.candidate_instruction_count > 0);
    let resolved_tsv = fs::read_to_string(&report.resolved_probe_targets_path)
        .expect("resolved probe target TSV should read");
    assert!(resolved_tsv.starts_with(
        "opcode\tbaseline_coverage\tcandidate_coverage\tcandidate_instruction_count\tclasses\tkinds\tarchitectures"
    ));
    assert!(resolved_tsv.contains("IADD"));
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
