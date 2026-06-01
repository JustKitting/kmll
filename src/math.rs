use crate::backends::Cuda;

#[derive(Debug, Clone, Copy, Default)]
pub struct Rsqrt;

#[derive(Debug, Clone, Copy, Default)]
pub struct Silu;

#[derive(Debug, Clone, Copy, Default)]
pub struct Sin;

#[derive(Debug, Clone, Copy, Default)]
pub struct Cos;

#[derive(Debug, Clone, Copy, Default)]
pub struct Exp;

pub trait UnaryOp<Op, Backend>: Sized {
    fn apply(self) -> Self;
}

#[inline(always)]
pub fn rsqrt<Backend, T>(x: T) -> T
where
    T: UnaryOp<Rsqrt, Backend>,
{
    <T as UnaryOp<Rsqrt, Backend>>::apply(x)
}

#[inline(always)]
pub fn silu<Backend, T>(x: T) -> T
where
    T: UnaryOp<Silu, Backend>,
{
    <T as UnaryOp<Silu, Backend>>::apply(x)
}

#[inline(always)]
pub fn sin<Backend, T>(x: T) -> T
where
    T: UnaryOp<Sin, Backend>,
{
    <T as UnaryOp<Sin, Backend>>::apply(x)
}

#[inline(always)]
pub fn cos<Backend, T>(x: T) -> T
where
    T: UnaryOp<Cos, Backend>,
{
    <T as UnaryOp<Cos, Backend>>::apply(x)
}

#[inline(always)]
pub fn exp<Backend, T>(x: T) -> T
where
    T: UnaryOp<Exp, Backend>,
{
    <T as UnaryOp<Exp, Backend>>::apply(x)
}

impl UnaryOp<Rsqrt, Cuda> for f32 {
    #[inline(always)]
    fn apply(self) -> Self {
        1.0 / self.sqrt()
    }
}

impl UnaryOp<Silu, Cuda> for f32 {
    #[inline(always)]
    fn apply(self) -> Self {
        self / (1.0 + (-self).exp())
    }
}

impl UnaryOp<Sin, Cuda> for f32 {
    #[inline(always)]
    fn apply(self) -> Self {
        self.sin()
    }
}

impl UnaryOp<Cos, Cuda> for f32 {
    #[inline(always)]
    fn apply(self) -> Self {
        self.cos()
    }
}

impl UnaryOp<Exp, Cuda> for f32 {
    #[inline(always)]
    fn apply(self) -> Self {
        self.exp()
    }
}
