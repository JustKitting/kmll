use super::super::{codegen::*, hashing::*, metadata::*, *};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MatvecRustCudaGenerator;

impl KernelSourceGenerator for MatvecRustCudaGenerator {
    fn name(&self) -> &'static str {
        "matvec-rust-cuda-source-generator"
    }

    fn source_for(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<GeneratedKernelSource, KernelGenerationError> {
        if candidate.family != "matvec-bf16-row-major" {
            return Err(KernelGenerationError::UnsupportedCandidate {
                family: candidate.family.clone(),
                generator: self.name(),
            });
        }
        let plan = schedule_matvec_plan(&candidate.schedule).ok_or_else(|| {
            KernelGenerationError::MissingTransform {
                family: candidate.family.clone(),
                transform: "Split",
            }
        })?;
        let symbol = match &candidate.generated.materialization {
            KernelMaterialization::Existing { symbol } => (*symbol).to_string(),
            KernelMaterialization::Generated { symbol } => sanitize_identifier(symbol),
            KernelMaterialization::DeferredGenerated { symbol_hint, .. } => {
                sanitize_identifier(symbol_hint)
            }
        };
        if symbol == "matvec_bf16_naive" {
            return Ok(GeneratedKernelSource {
                symbol: symbol.clone(),
                source: render_bf16_naive_matvec_source(&symbol),
            });
        }
        Ok(GeneratedKernelSource {
            symbol: symbol.clone(),
            source: render_bf16_matvec_source(&symbol, plan),
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GemmRustCudaGenerator;

impl KernelSourceGenerator for GemmRustCudaGenerator {
    fn name(&self) -> &'static str {
        "gemm-rust-cuda-source-generator"
    }

    fn source_for(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<GeneratedKernelSource, KernelGenerationError> {
        if candidate.family != "gemm-f32-bf16-row-col-row" {
            return Err(KernelGenerationError::UnsupportedCandidate {
                family: candidate.family.clone(),
                generator: self.name(),
            });
        }
        let plan = schedule_gemm_plan(&candidate.schedule).ok_or_else(|| {
            KernelGenerationError::MissingTransform {
                family: candidate.family.clone(),
                transform: "TileGemm",
            }
        })?;
        let symbol = match &candidate.generated.materialization {
            KernelMaterialization::Existing { symbol } => (*symbol).to_string(),
            KernelMaterialization::Generated { symbol } => sanitize_identifier(symbol),
            KernelMaterialization::DeferredGenerated { symbol_hint, .. } => {
                sanitize_identifier(symbol_hint)
            }
        };
        Ok(GeneratedKernelSource {
            symbol: symbol.clone(),
            source: render_f32_bf16_gemm_source(&symbol, plan),
        })
    }
}
