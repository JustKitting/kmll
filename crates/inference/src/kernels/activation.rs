use cuda_device::{DisjointSlice, kernel, thread};

#[inline(always)]
fn clamp_f32(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

#[inline(always)]
pub fn swiglu_reference(x: f32) -> f32 {
    let x_clamped = clamp_f32(x, -8.0, 8.0);
    let sigmoid = 0.5 + 0.21875 * x_clamped - 0.03125 * x_clamped * x_clamped * x_clamped;
    let sigmoid_clamped = clamp_f32(sigmoid, 0.0, 1.0);

    x * sigmoid_clamped * x
}

#[kernel]
pub fn relu_kernel(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = if input[i] > 0.0 { input[i] } else { 0.0 };
    }
}

#[kernel]
pub fn swiglu_kernel(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = swiglu_reference(input[i]);
    }
}
