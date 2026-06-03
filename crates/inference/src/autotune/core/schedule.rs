#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScheduleTransform {
    Split { axis: u8, factor: u32 },
    Upcast { axis: u8, factor: u32 },
    Unroll { axis: u8, factor: u32 },
    LocalTile { axis: u8, factor: u32 },
    GroupTop { axis: u8, factor: u32 },
    Group { axis: u8, factor: u32 },
    ThreadGroup { axis: u8, factor: u32 },
    TileGemm { m: u32, n: u32, k: u32 },
    StrideOrder { axes: Vec<u8> },
    Swap { axis_a: u8, axis_b: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct KernelSchedule {
    pub transforms: Vec<ScheduleTransform>,
}

impl KernelSchedule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_transform(mut self, transform: ScheduleTransform) -> Self {
        self.transforms.push(transform);
        self
    }

    pub fn depth(&self) -> usize {
        self.transforms.len()
    }
}
