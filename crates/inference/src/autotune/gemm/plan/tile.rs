use crate::autotune::KernelTile3d;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmTileShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl GemmTileShape {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    pub fn with_axis(self, axis: u8, factor: u32) -> Option<Self> {
        match axis {
            0 => Some(Self {
                m: factor,
                n: self.n,
                k: self.k,
            }),
            1 => Some(Self {
                m: self.m,
                n: factor,
                k: self.k,
            }),
            2 => Some(Self {
                m: self.m,
                n: self.n,
                k: factor,
            }),
            _ => None,
        }
    }

    pub const fn axis_factor(self, axis: u8) -> Option<u32> {
        match axis {
            0 => Some(self.m),
            1 => Some(self.n),
            2 => Some(self.k),
            _ => None,
        }
    }

    pub const fn thread_count(self) -> u32 {
        self.m * self.n
    }

    pub const fn is_launchable_shape(self) -> bool {
        self.m > 0 && self.n > 0 && self.k > 0 && self.thread_count() <= 1024
    }

    pub fn block_dim(self) -> (u32, u32, u32) {
        (self.n, self.m, 1)
    }

    pub fn grid_dim(self, m: usize, n: usize) -> (u32, u32, u32) {
        ((n as u32).div_ceil(self.n), (m as u32).div_ceil(self.m), 1)
    }
}

impl From<GemmTileShape> for KernelTile3d {
    fn from(value: GemmTileShape) -> Self {
        Self::new(value.m, value.n, value.k)
    }
}
