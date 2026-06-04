use std::{error::Error, io};

mod cache;
mod compile;
mod gemm;
mod gemm_launch;
mod matvec;
mod matvec_launch;
mod top1;
mod validation;

pub use self::{
    cache::cached_measured_score,
    compile::{CompiledStandaloneKernelCrate, compile_standalone_kernel_crate},
    gemm::GemmF32Bf16MeasuredAutotuneScorer,
    matvec::MatvecBf16MeasuredAutotuneScorer,
    top1::Top1Bf16MeasuredAutotuneScorer,
};

pub type KernelAutotuneMeasureResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelAutotuneMeasureOptions {
    pub repeat_count: usize,
    pub warmup_count: usize,
    pub compile_arch: Option<String>,
}

fn invalid_input(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}
