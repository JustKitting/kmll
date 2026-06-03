use super::{KernelAutotuneMeasureResult, invalid_data};
use nn_rust_inference::{
    dtypes::Bf16,
    layout::{Layout2D, MatrixLayout, RowMajor},
};

pub(in crate::autotune::measurement) fn compare_gemm_output<L: Layout2D>(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    c_layout: &MatrixLayout<L>,
    m: usize,
    n: usize,
    k: usize,
) -> KernelAutotuneMeasureResult<()> {
    if actual.len() != expected.len() {
        return Err(invalid_data(format!(
            "GEMM output length mismatch for {label}: actual={} expected={}",
            actual.len(),
            expected.len()
        )));
    }

    let mut max_abs_diff = 0.0_f32;
    for row in 0..m {
        for col in 0..n {
            let offset = c_layout.offset(row, col);
            let diff = (actual[offset] - expected[offset]).abs();
            max_abs_diff = max_abs_diff.max(diff);
        }
    }

    let tolerance = 1.0e-4_f32 * (k.max(1) as f32).sqrt();
    if max_abs_diff > tolerance {
        return Err(invalid_data(format!(
            "GEMM stress {label} {m}x{k} * {k}x{n} failed: max_abs_diff={max_abs_diff:.8}, tolerance={tolerance:.8}"
        )));
    }

    Ok(())
}

pub(in crate::autotune::measurement) fn compare_matvec_output(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    rows: usize,
    cols: usize,
) -> KernelAutotuneMeasureResult<()> {
    if actual.len() != expected.len() {
        return Err(invalid_data(format!(
            "matvec output length mismatch for {label}: actual={} expected={}",
            actual.len(),
            expected.len()
        )));
    }

    let mut max_abs_diff = 0.0_f32;
    for row in 0..rows {
        let diff = (actual[row] - expected[row]).abs();
        max_abs_diff = max_abs_diff.max(diff);
    }

    let tolerance = 1.0e-4_f32 * (cols.max(1) as f32).sqrt();
    if max_abs_diff > tolerance {
        return Err(invalid_data(format!(
            "matvec stress {label} rows={rows} cols={cols} failed: max_abs_diff={max_abs_diff:.8}, tolerance={tolerance:.8}"
        )));
    }

    Ok(())
}

pub(in crate::autotune::measurement) fn fill_matrix<L: Layout2D>(
    values: &mut [f32],
    layout: &MatrixLayout<L>,
    rows: usize,
    cols: usize,
    seed: &mut u64,
) {
    for row in 0..rows {
        for col in 0..cols {
            values[layout.offset(row, col)] = next_stress_value(seed);
        }
    }
}

pub(in crate::autotune::measurement) fn fill_stress_slice(
    values: &mut [f32],
    seed: &mut u64,
    scale: f32,
) {
    for value in values {
        *value = next_stress_value(seed) * scale;
    }
}

pub(in crate::autotune::measurement) fn fill_bf16_matrix<L: Layout2D>(
    values: &mut [Bf16],
    layout: &MatrixLayout<L>,
    rows: usize,
    cols: usize,
    seed: &mut u64,
) {
    for row in 0..rows {
        for col in 0..cols {
            values[layout.offset(row, col)] = Bf16::from_f32(next_stress_value(seed));
        }
    }
}

pub(in crate::autotune::measurement) fn cpu_gemm_bf16_reference<ALayout, BLayout, CLayout>(
    a: &[f32],
    a_layout: &MatrixLayout<ALayout>,
    b: &[Bf16],
    b_layout: &MatrixLayout<BLayout>,
    c_initial: &[f32],
    c_layout: &MatrixLayout<CLayout>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut out = c_initial.to_vec();
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[a_layout.offset(row, kk)] * b[b_layout.offset(kk, col)].to_f32();
            }
            let c_offset = c_layout.offset(row, col);
            out[c_offset] = alpha * acc + beta * c_initial[c_offset];
        }
    }
    out
}

pub(in crate::autotune::measurement) fn cpu_matvec_bf16_reference(
    input: &[f32],
    weight: &[Bf16],
    weight_layout: &MatrixLayout<RowMajor>,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows];
    for row in 0..rows {
        let mut acc = 0.0_f32;
        for col in 0..cols {
            acc += input[col] * weight[weight_layout.offset(row, col)].to_f32();
        }
        out[row] = acc;
    }
    out
}

fn next_stress_value(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bucket = ((*seed >> 32) & 0xffff) as f32 / 65535.0;
    bucket * 2.0 - 1.0
}
