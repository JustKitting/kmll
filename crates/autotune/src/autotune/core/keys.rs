pub(in crate::autotune) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
pub(in crate::autotune) const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelMetadataKey(pub(in crate::autotune) u64);

impl KernelMetadataKey {
    pub const fn raw(self) -> u64 {
        self.0
    }

    pub fn hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelImplementationKey(pub(in crate::autotune) u64);

impl KernelImplementationKey {
    pub const fn raw(self) -> u64 {
        self.0
    }

    pub fn hex(self) -> String {
        format!("{:016x}", self.0)
    }
}
