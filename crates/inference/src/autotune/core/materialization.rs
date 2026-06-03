use super::super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelMaterialization {
    Existing { symbol: &'static str },
    Generated { symbol: String },
    DeferredGenerated { symbol_hint: String, reason: String },
}

impl KernelMaterialization {
    pub const fn is_launchable(&self) -> bool {
        matches!(self, Self::Existing { .. })
    }

    pub const fn is_materializable(&self) -> bool {
        matches!(self, Self::Existing { .. } | Self::Generated { .. })
    }

    pub const fn is_existing(&self) -> bool {
        matches!(self, Self::Existing { .. })
    }

    pub const fn requires_generated_module(&self) -> bool {
        matches!(
            self,
            Self::Generated { .. } | Self::DeferredGenerated { .. }
        )
    }

    pub const fn profiling_materialization(&self) -> ProfilingCandidateMaterialization {
        match self {
            Self::Existing { .. } => ProfilingCandidateMaterialization::Existing,
            Self::Generated { .. } => ProfilingCandidateMaterialization::Generated,
            Self::DeferredGenerated { .. } => ProfilingCandidateMaterialization::DeferredGenerated,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedKernelMetadata {
    pub generator: &'static str,
    pub artifact_key: KernelMetadataKey,
    pub materialization: KernelMaterialization,
}
