use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelExpansionPolicy {
    pub require_launchable: bool,
    pub max_threads_per_block: Option<u32>,
    pub max_shared_memory_bytes: Option<u32>,
    pub max_accumulator_elements_per_thread: Option<u32>,
    pub max_output_elements_per_thread: Option<u32>,
    pub max_load_elements_per_block: Option<u32>,
}

impl KernelExpansionPolicy {
    pub const DEFAULT_MAX_THREADS_PER_BLOCK: u32 = 1024;
    pub const DEFAULT_MAX_SHARED_MEMORY_BYTES: u32 = 48 * 1024;

    pub const fn new() -> Self {
        Self {
            require_launchable: false,
            max_threads_per_block: Some(Self::DEFAULT_MAX_THREADS_PER_BLOCK),
            max_shared_memory_bytes: Some(Self::DEFAULT_MAX_SHARED_MEMORY_BYTES),
            max_accumulator_elements_per_thread: None,
            max_output_elements_per_thread: None,
            max_load_elements_per_block: None,
        }
    }

    pub const fn for_search_config(require_launchable: bool) -> Self {
        Self {
            require_launchable,
            ..Self::new()
        }
    }

    pub const fn with_require_launchable(mut self, require_launchable: bool) -> Self {
        self.require_launchable = require_launchable;
        self
    }

    pub const fn with_max_threads_per_block(mut self, max: Option<u32>) -> Self {
        self.max_threads_per_block = max;
        self
    }

    pub const fn with_max_shared_memory_bytes(mut self, max: Option<u32>) -> Self {
        self.max_shared_memory_bytes = max;
        self
    }

    pub const fn with_max_accumulator_elements_per_thread(mut self, max: Option<u32>) -> Self {
        self.max_accumulator_elements_per_thread = max;
        self
    }

    pub const fn with_max_output_elements_per_thread(mut self, max: Option<u32>) -> Self {
        self.max_output_elements_per_thread = max;
        self
    }

    pub const fn with_max_load_elements_per_block(mut self, max: Option<u32>) -> Self {
        self.max_load_elements_per_block = max;
        self
    }

    pub fn allows(
        self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<(), KernelCandidateRejectReason> {
        if self.require_launchable && !candidate.is_launchable() {
            return Err(KernelCandidateRejectReason::DeferredGenerated);
        }
        let threads_per_block = launch_threads_per_block(&candidate.launch);
        if let Some(max) = self.max_threads_per_block
            && threads_per_block > max
        {
            return Err(KernelCandidateRejectReason::ThreadsPerBlock {
                actual: threads_per_block,
                max,
            });
        }
        let shared_memory_bytes = candidate
            .resources
            .map(|resources| resources.shared_memory_bytes)
            .unwrap_or(candidate.launch.shared_mem_bytes)
            .max(candidate.launch.shared_mem_bytes);
        if let Some(max) = self.max_shared_memory_bytes
            && shared_memory_bytes > max
        {
            return Err(KernelCandidateRejectReason::SharedMemoryBytes {
                actual: shared_memory_bytes,
                max,
            });
        }
        let Some(resources) = candidate.resources else {
            return Ok(());
        };
        if let Some(max) = self.max_accumulator_elements_per_thread
            && resources.accumulator_elements_per_thread > max
        {
            return Err(KernelCandidateRejectReason::AccumulatorElementsPerThread {
                actual: resources.accumulator_elements_per_thread,
                max,
            });
        }
        if let Some(max) = self.max_output_elements_per_thread
            && resources.output_elements_per_thread > max
        {
            return Err(KernelCandidateRejectReason::OutputElementsPerThread {
                actual: resources.output_elements_per_thread,
                max,
            });
        }
        if let Some(max) = self.max_load_elements_per_block
            && resources.load_elements_per_block > max
        {
            return Err(KernelCandidateRejectReason::LoadElementsPerBlock {
                actual: resources.load_elements_per_block,
                max,
            });
        }
        Ok(())
    }
}

impl Default for KernelExpansionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelCandidateRejectReason {
    DeferredGenerated,
    ThreadsPerBlock { actual: u32, max: u32 },
    SharedMemoryBytes { actual: u32, max: u32 },
    AccumulatorElementsPerThread { actual: u32, max: u32 },
    OutputElementsPerThread { actual: u32, max: u32 },
    LoadElementsPerBlock { actual: u32, max: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelCandidateExpansion {
    pub candidates: Vec<KernelCandidateMetadata>,
    pub accepted: usize,
    pub rejected: usize,
    pub duplicates: usize,
    pub last_reject_reason: Option<KernelCandidateRejectReason>,
}

impl KernelCandidateExpansion {
    pub fn explored(&self) -> usize {
        self.accepted + self.rejected
    }
}

fn launch_threads_per_block(launch: &CudaLaunchSpec) -> u32 {
    let threads = u64::from(launch.block_dim.x)
        .saturating_mul(u64::from(launch.block_dim.y))
        .saturating_mul(u64::from(launch.block_dim.z));
    threads.min(u64::from(u32::MAX)) as u32
}
