use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedKernelSource {
    pub symbol: String,
    pub source: String,
}

pub trait KernelSourceGenerator {
    fn name(&self) -> &'static str;
    fn source_for(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<GeneratedKernelSource, KernelGenerationError>;
}
