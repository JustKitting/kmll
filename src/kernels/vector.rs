use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn vecadd_kernel(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = a[i] + b[i];
    }
}
