use std::fmt;

use crate::decompile::{
    SassModifier, SassModifierKind, SassOpcodeKind, SassTensorMmaSignature, SassTensorScope,
};

use super::{
    TensorCoreAccumulator, TensorCoreBlockScale, TensorCoreDType, TensorCoreMmaShape,
    TensorCoreOpFamily, TensorCoreOperandLayout, TensorCoreOperandLayouts, TensorCoreSparsity,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorCoreDTypes {
    pub output: TensorCoreDType,
    pub lhs: TensorCoreDType,
    pub rhs: TensorCoreDType,
    pub accumulator: TensorCoreAccumulator,
}

impl TensorCoreDTypes {
    pub fn new(
        output: TensorCoreDType,
        lhs: TensorCoreDType,
        rhs: TensorCoreDType,
        accumulator: TensorCoreAccumulator,
    ) -> Self {
        Self {
            output,
            lhs,
            rhs,
            accumulator,
        }
    }

    pub fn homogeneous(
        output: TensorCoreDType,
        operand: TensorCoreDType,
        accumulator: TensorCoreAccumulator,
    ) -> Self {
        Self::new(output, operand, operand, accumulator)
    }

    pub fn from_sass_signature(signature: &SassTensorMmaSignature) -> Option<Self> {
        let lhs = TensorCoreDType::from_sass(signature.lhs_type.as_ref()?)?;
        let rhs = TensorCoreDType::from_sass(signature.rhs_type.as_ref()?)?;
        let output = signature
            .output_type
            .as_ref()
            .and_then(TensorCoreDType::from_sass)
            .unwrap_or_else(|| {
                signature
                    .accumulator_type
                    .as_ref()
                    .and_then(TensorCoreDType::from_sass)
                    .unwrap_or(lhs)
            });
        let accumulator_dtype = signature
            .accumulator_type
            .as_ref()
            .and_then(TensorCoreDType::from_sass)
            .or(Some(output))?;
        let accumulator = TensorCoreAccumulator::from_dtype(accumulator_dtype)?;
        Some(Self::new(output, lhs, rhs, accumulator))
    }
}

impl fmt::Display for TensorCoreDTypes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}.{}.{}.{}",
            self.output, self.lhs, self.rhs, self.accumulator
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorCoreOpSpec {
    pub family: TensorCoreOpFamily,
    pub shape: TensorCoreMmaShape,
    pub dtypes: TensorCoreDTypes,
    pub layouts: TensorCoreOperandLayouts,
    pub sparsity: Option<TensorCoreSparsity>,
    pub block_scale: Option<TensorCoreBlockScale>,
}

impl TensorCoreOpSpec {
    pub fn new(
        family: TensorCoreOpFamily,
        shape: TensorCoreMmaShape,
        dtypes: TensorCoreDTypes,
        layouts: TensorCoreOperandLayouts,
    ) -> Self {
        Self {
            family,
            shape,
            dtypes,
            layouts,
            sparsity: None,
            block_scale: None,
        }
    }

    pub fn with_sparsity(mut self, sparsity: TensorCoreSparsity) -> Self {
        self.sparsity = Some(sparsity);
        if self.family == TensorCoreOpFamily::MmaSync {
            self.family = TensorCoreOpFamily::MmaSparse;
        }
        self
    }

    pub fn with_block_scale(mut self, block_scale: TensorCoreBlockScale) -> Self {
        self.block_scale = Some(block_scale);
        self
    }

    pub fn from_sass(
        opcode: &SassOpcodeKind,
        signature: &SassTensorMmaSignature,
        scope: Option<&SassTensorScope>,
        modifiers: &[SassModifier],
    ) -> Option<Self> {
        let family = TensorCoreOpFamily::from_sass_opcode(opcode)
            .or_else(|| TensorCoreOpFamily::from_sass_scope(scope?))?;
        let shape = TensorCoreMmaShape::from_sass(signature.shape?);
        let dtypes = TensorCoreDTypes::from_sass_signature(signature)?;
        let layouts =
            layouts_from_modifiers(modifiers).unwrap_or(TensorCoreOperandLayouts::ROW_COL);
        let mut spec = Self::new(family, shape, dtypes, layouts);
        spec.sparsity = sparsity_from_modifiers(modifiers);
        spec.block_scale = block_scale_from_modifiers(modifiers);
        Some(spec)
    }

    pub fn label(&self) -> String {
        let sparse = self
            .sparsity
            .map(|sparsity| format!(".sparse={sparsity}"))
            .unwrap_or_default();
        let scale = self
            .block_scale
            .map(|scale| format!(".scale={scale}"))
            .unwrap_or_default();
        format!(
            "{}.{}.{}.{}{}{}",
            self.family, self.shape, self.layouts, self.dtypes, sparse, scale
        )
    }
}

impl fmt::Display for TensorCoreOpSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

fn layouts_from_modifiers(modifiers: &[SassModifier]) -> Option<TensorCoreOperandLayouts> {
    let mut layouts = modifiers
        .iter()
        .filter_map(|modifier| match modifier.kind() {
            SassModifierKind::Row => Some(TensorCoreOperandLayout::Row),
            SassModifierKind::Col => Some(TensorCoreOperandLayout::Col),
            _ => None,
        });
    let lhs = layouts.next()?;
    let rhs = layouts.next().unwrap_or(TensorCoreOperandLayout::Col);
    Some(TensorCoreOperandLayouts { lhs, rhs })
}

fn sparsity_from_modifiers(modifiers: &[SassModifier]) -> Option<TensorCoreSparsity> {
    modifiers.iter().find_map(|modifier| match modifier.kind() {
        SassModifierKind::Sparse => Some(TensorCoreSparsity::Structured2To4),
        _ => None,
    })
}

fn block_scale_from_modifiers(modifiers: &[SassModifier]) -> Option<TensorCoreBlockScale> {
    modifiers.iter().find_map(|modifier| match modifier.kind() {
        SassModifierKind::BlockScaleMx => Some(TensorCoreBlockScale::Mx),
        SassModifierKind::BlockScaleNvfp4 => Some(TensorCoreBlockScale::Nvfp4),
        SassModifierKind::BlockScaleMxfp4 => Some(TensorCoreBlockScale::Mxfp4),
        SassModifierKind::BlockScaleMxfp6 => Some(TensorCoreBlockScale::Mxfp6),
        SassModifierKind::BlockScaleMxfp8 => Some(TensorCoreBlockScale::Mxfp8),
        _ => None,
    })
}
