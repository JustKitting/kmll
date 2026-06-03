use super::*;
use std::{collections::BTreeSet, env, fs, process, time::SystemTime};

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
        }) if label == expected
    )
}

#[test]
fn register_refs_canonicalize_modifier_spelling_for_identity() {
    assert_eq!(reg("R13.reuse"), reg("R13"));
    assert_eq!(reg("-RZ"), reg("RZ"));
    assert_ne!(reg("URZ"), reg("RZ"));
    assert_eq!(reg("R13.reuse").to_string(), "R13");
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
        /*0040*/                   EXIT ;                                        /* 0x0 */
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
        /*0030*/                   OMMA.E2M1 R8, R12, R16, R20 ;                 /* 0x0 */
        /*0040*/                   QGMMA.E4M3 R8, R12, R16, R20 ;                /* 0x0 */
        /*0050*/                   EXIT ;                                        /* 0x0 */
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
    assert_eq!(function.name, "sass_fixture_i32_add");
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
    assert_eq!(function.name, "cuobjdump_fixture");
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
fn lift_rows17_slice_keeps_predicates_and_half_fma_visible() {
    let module = parse_nvidia_sass(ROWS17_SLICE).expect("rows17 slice should parse");
    let ir = lift_sass_module(&module);
    let ops = &ir.functions[0].ops;

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
        } if matches!(&target.kind, ControlTargetKind::Label(label) if label == ".L_x_1")
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
        } if matches!(&target.kind, ControlTargetKind::Label(label) if label == "$helper")
            && matches!(&operands[0].kind, AggregateOperandKind::Label(label) if label == "$helper")
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
        } if matches!(&target.kind, ControlTargetKind::Label(label) if label == "matvec_bf16_rows17")
    )));
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
            if matches!(&operands[1].kind, AggregateOperandKind::Raw { registers }
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
    assert!(matches!(
        &function.ops[3].kind,
        KernelIrOpKind::TensorCoreMma {
            element_type: Some(SassTensorElementType::Fp4),
            signature: Some(SassTensorMmaSignature {
                shape: None,
                output_type: None,
                lhs_type: Some(SassTensorElementType::Fp4),
                rhs_type: Some(SassTensorElementType::Fp4),
                accumulator_type: None,
            }),
            scope: Some(SassTensorScope::Warp),
            ..
        }
    ));
    assert!(matches!(
        &function.ops[4].kind,
        KernelIrOpKind::TensorCoreMma {
            element_type: Some(SassTensorElementType::Fp8),
            signature: Some(SassTensorMmaSignature {
                shape: None,
                output_type: None,
                lhs_type: Some(SassTensorElementType::Fp8),
                rhs_type: Some(SassTensorElementType::Fp8),
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
            ..
        } if src == &reg("R23") && dst == &reg("R23")
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
    assert_eq!(entry_r2_value.use_addresses.as_slice(), &[0x30]);

    let local_r2_value = function
        .ssa_values
        .iter()
        .find(|value| value.register == reg("R2") && value.def_address == Some(0x20))
        .expect("local R2 definition should have an SSA value");
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
        edge.value_id == local_r2_value.value_id && edge.def_address == Some(0x20)
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

    assert_eq!(options.compile_arch, "sm_120");
    assert_eq!(
        options.probes,
        vec![
            PtxDecompileProbeKind::TensorCoreHmma,
            PtxDecompileProbeKind::TensorCoreImma,
            PtxDecompileProbeKind::TensorCoreDmma
        ]
    );
    assert!(
        options
            .artifact_root
            .ends_with("target/cuda-oxide/inference/decompile-probes")
    );
    assert_eq!(hmma_probe.symbol, "tensor_core_hmma_probe");
    assert!(hmma_probe.source.contains("mma.sync.aligned"));
    assert!(hmma_probe.source.contains("st.global.f32"));
    assert_eq!(imma_probe.symbol, "tensor_core_imma_probe");
    assert!(
        imma_probe
            .source
            .contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
    );
    assert!(imma_probe.source.contains("st.global.s32"));
    assert_eq!(dmma_probe.symbol, "tensor_core_dmma_probe");
    assert!(
        dmma_probe
            .source
            .contains("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64")
    );
    assert!(dmma_probe.source.contains("st.global.f64"));
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
    assert!(report.known_opcode_count > 0);
    assert!(report.locally_mapped_opcode_count > 0);
    assert!(report.known_unobserved_opcode_count > 0);
    assert_eq!(
        report.opcode_probe_target_count,
        report.known_unobserved_opcode_count
    );
    assert_eq!(report.known_unmapped_opcode_count, 0);
    assert!(report.observed_unregistered_opcode_count > 0);
    assert!(report.observed_unmapped_opcode_count > 0);
    assert_eq!(report.unsupported_instruction_count, 1);
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
            .any(|architecture| architecture == "sm120")
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
            .any(|architecture| architecture == "sm120")
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
            .any(|instruction| instruction.opcode == SassOpcode::new("MYSTERY"))
    );
    assert!(report.files.iter().any(
        |file| file.sass_path.file_name().and_then(|name| name.to_str())
            == Some("simple.cuobjdump.sass")
    ));
    assert!(report.summary_path.exists());
    assert!(report.files_path.exists());
    assert!(report.opcode_catalog_path.exists());
    assert!(report.opcode_probe_targets_path.exists());
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
    let opcode_catalog_tsv =
        fs::read_to_string(&report.opcode_catalog_path).expect("opcode catalog TSV should read");
    assert!(opcode_catalog_tsv.starts_with(
        "opcode\tknown\tobserved\tlocally_mapped\tinstruction_count\tsignature_count\tsignatures\tsource_formats\tarchitectures\tknown_sources\tclasses\tkinds\tsupport\tcoverage\tunsupported_count"
    ));
    assert!(opcode_catalog_tsv.contains("MYSTERY"));
    assert!(opcode_catalog_tsv.contains("HMMA"));
    let opcode_probe_targets_tsv = fs::read_to_string(&report.opcode_probe_targets_path)
        .expect("opcode probe target TSV should read");
    assert!(opcode_probe_targets_tsv.starts_with(
        "opcode\tpriority\tlocally_mapped\tarchitectures\tclasses\tkinds\tknown_sources\trecommended_action\treason"
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
