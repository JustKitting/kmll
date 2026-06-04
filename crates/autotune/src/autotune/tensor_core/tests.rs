use crate::decompile::{
    SassModifier, SassModifierKind, SassOpcodeKind, SassTensorElementType, SassTensorMmaShape,
    SassTensorMmaSignature, lift_sass_module, parse_nvidia_sass,
};

use super::*;

#[test]
fn sm120_seed_contains_core_tensor_families() {
    let space = TensorCoreSearchSpace::sm120_seed();
    assert!(
        space
            .iter()
            .any(|spec| spec.family == TensorCoreOpFamily::MmaSync
                && spec.shape == TensorCoreMmaShape::M16N8K8
                && spec.dtypes.lhs == TensorCoreDType::Tf32
                && spec.dtypes.accumulator == TensorCoreAccumulator::F32)
    );
    assert!(
        space
            .iter()
            .any(|spec| spec.family == TensorCoreOpFamily::WgmmaAsync
                && spec.shape == TensorCoreMmaShape::M64N128K16
                && spec.dtypes.lhs == TensorCoreDType::Bf16)
    );
    assert!(
        space
            .iter()
            .any(|spec| spec.family == TensorCoreOpFamily::Tcgen05
                && spec.block_scale == Some(TensorCoreBlockScale::Mxfp4)
                && spec.dtypes.lhs == TensorCoreDType::Fp4)
    );
}

#[test]
fn sass_hmma_signature_lifts_to_typed_tensor_core_spec() {
    let signature = SassTensorMmaSignature {
        shape: Some(SassTensorMmaShape::new(16, 8, 8)),
        output_type: Some(SassTensorElementType::Fp32),
        lhs_type: Some(SassTensorElementType::Tf32),
        rhs_type: Some(SassTensorElementType::Tf32),
        accumulator_type: Some(SassTensorElementType::Fp32),
    };
    let modifiers = [
        SassModifier::parse("1688"),
        SassModifier::parse("ROW"),
        SassModifier::parse("COL"),
        SassModifier::parse("F32"),
        SassModifier::parse("TF32"),
        SassModifier::parse("TF32"),
        SassModifier::parse("F32"),
    ];

    let spec = TensorCoreOpSpec::from_sass(&SassOpcodeKind::Hmma, &signature, None, &modifiers)
        .expect("HMMA should lift to typed tensor-core spec");
    assert_eq!(spec.family, TensorCoreOpFamily::MmaSync);
    assert_eq!(spec.shape, TensorCoreMmaShape::M16N8K8);
    assert_eq!(spec.layouts, TensorCoreOperandLayouts::ROW_COL);
    assert_eq!(
        spec.dtypes,
        TensorCoreDTypes::new(
            TensorCoreDType::F32,
            TensorCoreDType::Tf32,
            TensorCoreDType::Tf32,
            TensorCoreAccumulator::F32,
        )
    );
}

#[test]
fn sass_low_precision_modifiers_are_typed() {
    assert_eq!(SassModifier::parse("FP6").kind(), &SassModifierKind::Fp6);
    assert_eq!(SassModifier::parse("E2M3").kind(), &SassModifierKind::E2M3);
    assert_eq!(SassModifier::parse("E3M2").kind(), &SassModifierKind::E3M2);
    assert_eq!(
        SassModifier::parse("MXFP4").kind(),
        &SassModifierKind::BlockScaleMxfp4
    );
}

#[test]
fn lifted_sass_module_extracts_typed_tensor_core_ops() {
    const SASS: &str = r#"
        .target sm_120

        .section .text.tensor_core_fixture,"ax",@progbits
        .global tensor_core_fixture
    tensor_core_fixture:
    .text.tensor_core_fixture:
        /*0000*/                   HMMA.1688.F32.TF32.TF32.F32 R4, R20, R28, R4 ;       /* 0x0 */
        /*0010*/                   EXIT ;                                               /* 0x0 */
    "#;

    let module = parse_nvidia_sass(SASS).expect("tensor-core SASS should parse");
    let ir = lift_sass_module(&module);
    let ops = tensor_core_ops_from_ir_module(&ir);

    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].function.as_str(), "tensor_core_fixture");
    assert_eq!(ops[0].address, 0);
    assert_eq!(ops[0].spec.family, TensorCoreOpFamily::MmaSync);
    assert_eq!(ops[0].spec.shape, TensorCoreMmaShape::M16N8K8);
    assert_eq!(ops[0].spec.dtypes.lhs, TensorCoreDType::Tf32);
}
