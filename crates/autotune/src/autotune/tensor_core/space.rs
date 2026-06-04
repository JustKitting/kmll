use super::{
    TensorCoreAccumulator, TensorCoreBlockScale, TensorCoreDType, TensorCoreDTypes,
    TensorCoreMmaShape, TensorCoreOpFamily, TensorCoreOpSpec, TensorCoreOperandLayouts,
    TensorCoreSparsity,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCoreSearchSpace {
    specs: Vec<TensorCoreOpSpec>,
}

impl TensorCoreSearchSpace {
    pub fn new(specs: impl IntoIterator<Item = TensorCoreOpSpec>) -> Self {
        let mut specs = specs.into_iter().collect::<Vec<_>>();
        specs.sort();
        specs.dedup();
        Self { specs }
    }

    pub fn sm120_seed() -> Self {
        let mut specs = Vec::new();
        let row_col = TensorCoreOperandLayouts::ROW_COL;
        let f32 = TensorCoreDType::F32;
        let f16 = TensorCoreDType::F16;
        let bf16 = TensorCoreDType::Bf16;
        let tf32 = TensorCoreDType::Tf32;
        let fp8 = TensorCoreDType::Fp8;
        let fp4 = TensorCoreDType::Fp4;
        let i8 = TensorCoreDType::I8;

        specs.push(TensorCoreOpSpec::new(
            TensorCoreOpFamily::MmaSync,
            TensorCoreMmaShape::M16N8K8,
            TensorCoreDTypes::homogeneous(f32, tf32, TensorCoreAccumulator::F32),
            row_col,
        ));
        for shape in [TensorCoreMmaShape::M16N8K16, TensorCoreMmaShape::M16N8K32] {
            specs.push(TensorCoreOpSpec::new(
                TensorCoreOpFamily::MmaSync,
                shape,
                TensorCoreDTypes::homogeneous(f32, f16, TensorCoreAccumulator::F32),
                row_col,
            ));
            specs.push(TensorCoreOpSpec::new(
                TensorCoreOpFamily::MmaSync,
                shape,
                TensorCoreDTypes::homogeneous(f32, bf16, TensorCoreAccumulator::F32),
                row_col,
            ));
        }
        specs.push(TensorCoreOpSpec::new(
            TensorCoreOpFamily::MmaSync,
            TensorCoreMmaShape::M16N8K32,
            TensorCoreDTypes::homogeneous(TensorCoreDType::S32, i8, TensorCoreAccumulator::S32),
            row_col,
        ));
        specs.push(TensorCoreOpSpec::new(
            TensorCoreOpFamily::MmaSync,
            TensorCoreMmaShape::M8N8K128,
            TensorCoreDTypes::homogeneous(
                TensorCoreDType::S32,
                TensorCoreDType::Bit,
                TensorCoreAccumulator::S32,
            ),
            row_col,
        ));
        specs.push(
            TensorCoreOpSpec::new(
                TensorCoreOpFamily::MmaSparse,
                TensorCoreMmaShape::M16N8K32,
                TensorCoreDTypes::homogeneous(f32, bf16, TensorCoreAccumulator::F32),
                row_col,
            )
            .with_sparsity(TensorCoreSparsity::Structured2To4),
        );

        for n in [8, 16, 32, 64, 128, 256] {
            let shape = TensorCoreMmaShape::new(64, n, 16);
            specs.push(TensorCoreOpSpec::new(
                TensorCoreOpFamily::WgmmaAsync,
                shape,
                TensorCoreDTypes::homogeneous(f32, f16, TensorCoreAccumulator::F32),
                row_col,
            ));
            specs.push(TensorCoreOpSpec::new(
                TensorCoreOpFamily::WgmmaAsync,
                shape,
                TensorCoreDTypes::homogeneous(f32, bf16, TensorCoreAccumulator::F32),
                row_col,
            ));
        }
        for n in [8, 16, 32, 64, 128, 256] {
            specs.push(TensorCoreOpSpec::new(
                TensorCoreOpFamily::WgmmaAsync,
                TensorCoreMmaShape::new(64, n, 8),
                TensorCoreDTypes::homogeneous(f32, tf32, TensorCoreAccumulator::F32),
                row_col,
            ));
            specs.push(TensorCoreOpSpec::new(
                TensorCoreOpFamily::WgmmaAsync,
                TensorCoreMmaShape::new(64, n, 32),
                TensorCoreDTypes::homogeneous(f32, fp8, TensorCoreAccumulator::F32),
                row_col,
            ));
        }

        for shape in [
            TensorCoreMmaShape::M64N128K32,
            TensorCoreMmaShape::M64N256K32,
            TensorCoreMmaShape::Custom {
                m: 128,
                n: 128,
                k: 64,
            },
        ] {
            specs.push(
                TensorCoreOpSpec::new(
                    TensorCoreOpFamily::Tcgen05,
                    shape,
                    TensorCoreDTypes::homogeneous(f32, fp8, TensorCoreAccumulator::F32),
                    row_col,
                )
                .with_block_scale(TensorCoreBlockScale::Mxfp8),
            );
            specs.push(
                TensorCoreOpSpec::new(
                    TensorCoreOpFamily::Tcgen05,
                    shape,
                    TensorCoreDTypes::homogeneous(f32, fp4, TensorCoreAccumulator::F32),
                    row_col,
                )
                .with_block_scale(TensorCoreBlockScale::Mxfp4),
            );
        }

        Self::new(specs)
    }

    pub fn specs(&self) -> &[TensorCoreOpSpec] {
        &self.specs
    }

    pub fn iter(&self) -> impl Iterator<Item = &TensorCoreOpSpec> {
        self.specs.iter()
    }

    pub fn by_family(&self, family: TensorCoreOpFamily) -> impl Iterator<Item = &TensorCoreOpSpec> {
        self.specs.iter().filter(move |spec| spec.family == family)
    }
}
