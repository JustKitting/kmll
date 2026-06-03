use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InferenceKernelRustCudaGenerator;

impl KernelSourceGenerator for InferenceKernelRustCudaGenerator {
    fn name(&self) -> &'static str {
        "inference-rust-cuda-source-generator"
    }

    fn source_for(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<GeneratedKernelSource, KernelGenerationError> {
        match candidate.family.as_str() {
            "matvec-bf16-row-major" => MatvecRustCudaGenerator.source_for(candidate),
            "gemm-f32-bf16-row-col-row" => GemmRustCudaGenerator.source_for(candidate),
            _ => Err(KernelGenerationError::UnsupportedCandidate {
                family: candidate.family.clone(),
                generator: self.name(),
            }),
        }
    }
}
