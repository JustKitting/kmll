use std::{error::Error, io};

mod cache;
mod compile;
mod gemm;
mod matvec;
mod validation;

pub use self::{
    cache::cached_measured_score,
    compile::{CompiledStandaloneKernelCrate, compile_standalone_kernel_crate},
    gemm::GemmF32Bf16MeasuredAutotuneScorer,
    matvec::MatvecBf16MeasuredAutotuneScorer,
};

pub type KernelAutotuneMeasureResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, Copy)]
pub struct KernelAutotuneMeasureOptions {
    pub repeat_count: usize,
    pub warmup_count: usize,
}

fn invalid_input(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}
