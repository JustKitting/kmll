use super::*;

mod loads;
mod reduce;

use self::loads::{
    render_gemm_a_load_single, render_gemm_a_load_unrolled, render_gemm_b_load_single,
    render_gemm_b_load_unrolled,
};
use self::reduce::render_gemm_reduce_loop;

pub(in crate::autotune) fn render_f32_bf16_gemm_source(
    symbol: &str,
    plan: GemmSchedulePlan,
) -> String {
    let plan = plan.normalized();
    let tile = plan.tile;
    let reduce_unroll = plan.reduce_unroll.max(1);
    let reduce_group = plan.reduce_group_size();
    let has_reduce_group = plan.has_custom_reduce_group();
    let m_per_thread = plan.m_per_thread.max(1);
    let n_per_thread = plan.n_per_thread.max(1);
    let a_load_unroll = plan.a_load_unroll.max(1);
    let b_load_unroll = plan.b_load_unroll.max(1);
    let a_load_threads = plan.a_load_thread_count();
    let b_load_threads = plan.b_load_thread_count();
    let m_contiguous_a_load = plan.a_load_order == GemmATileLoadOrder::MContiguous;
    let k_contiguous_b_load = plan.b_load_order == GemmBTileLoadOrder::KContiguous;
    let mut source = String::new();
    writeln!(
        source,
        "use cuda_device::{{DisjointSlice, SharedArray, kernel, thread}};"
    )
    .expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "#[repr(transparent)]").expect("write to string");
    writeln!(source, "#[derive(Clone, Copy, Default)]").expect("write to string");
    writeln!(source, "pub struct Bf16(u16);").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "impl Bf16 {{").expect("write to string");
    writeln!(source, "    #[inline(always)]").expect("write to string");
    writeln!(source, "    pub fn to_f32(self) -> f32 {{").expect("write to string");
    writeln!(source, "        f32::from_bits((self.0 as u32) << 16)").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source, "}}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "const TILE_M: usize = {};", tile.m).expect("write to string");
    writeln!(source, "const TILE_N: usize = {};", tile.n).expect("write to string");
    writeln!(source, "const TILE_K: usize = {};", tile.k).expect("write to string");
    writeln!(source, "const REDUCE_UNROLL: usize = {reduce_unroll};").expect("write to string");
    if has_reduce_group {
        writeln!(source, "const REDUCE_GROUP: usize = {reduce_group};").expect("write to string");
    }
    writeln!(source, "const THREAD_TILE_M: usize = {m_per_thread};").expect("write to string");
    writeln!(source, "const THREAD_TILE_N: usize = {n_per_thread};").expect("write to string");
    writeln!(source, "const A_LOAD_UNROLL: usize = {a_load_unroll};").expect("write to string");
    writeln!(source, "const B_LOAD_UNROLL: usize = {b_load_unroll};").expect("write to string");
    writeln!(source, "const A_LOAD_THREADS: usize = {a_load_threads};").expect("write to string");
    writeln!(source, "const B_LOAD_THREADS: usize = {b_load_threads};").expect("write to string");
    writeln!(
        source,
        "const THREADS_M: usize = (TILE_M + THREAD_TILE_M - 1) / THREAD_TILE_M;"
    )
    .expect("write to string");
    writeln!(
        source,
        "const THREADS_N: usize = (TILE_N + THREAD_TILE_N - 1) / THREAD_TILE_N;"
    )
    .expect("write to string");
    writeln!(source, "const TILE_A_ELEMS: usize = TILE_M * TILE_K;").expect("write to string");
    writeln!(source, "const TILE_B_ELEMS: usize = TILE_K * TILE_N;").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "#[kernel]").expect("write to string");
    writeln!(source, "pub fn {symbol}(").expect("write to string");
    writeln!(source, "    a: &[f32],").expect("write to string");
    writeln!(source, "    b: &[Bf16],").expect("write to string");
    writeln!(source, "    m: u32,").expect("write to string");
    writeln!(source, "    n: u32,").expect("write to string");
    writeln!(source, "    k: u32,").expect("write to string");
    writeln!(source, "    a_row_stride: u32,").expect("write to string");
    writeln!(source, "    a_col_stride: u32,").expect("write to string");
    writeln!(source, "    b_row_stride: u32,").expect("write to string");
    writeln!(source, "    b_col_stride: u32,").expect("write to string");
    writeln!(source, "    c_row_stride: u32,").expect("write to string");
    writeln!(source, "    c_col_stride: u32,").expect("write to string");
    writeln!(source, "    alpha: f32,").expect("write to string");
    writeln!(source, "    beta: f32,").expect("write to string");
    writeln!(source, "    mut c: DisjointSlice<f32>,").expect("write to string");
    writeln!(source, ") {{").expect("write to string");
    writeln!(
        source,
        "    static mut TILE_A: SharedArray<f32, TILE_A_ELEMS> = SharedArray::UNINIT;"
    )
    .expect("write to string");
    writeln!(
        source,
        "    static mut TILE_B: SharedArray<f32, TILE_B_ELEMS> = SharedArray::UNINIT;"
    )
    .expect("write to string");
    writeln!(source).expect("write to string");
    match plan.thread_order {
        GemmThreadOrder::NThenM => {
            writeln!(source, "    let thread_n = thread::threadIdx_x() as usize;")
                .expect("write to string");
            writeln!(source, "    let thread_m = thread::threadIdx_y() as usize;")
                .expect("write to string");
            writeln!(
                source,
                "    if thread_n >= THREADS_N || thread_m >= THREADS_M {{"
            )
            .expect("write to string");
            writeln!(source, "        return;").expect("write to string");
            writeln!(source, "    }}").expect("write to string");
            writeln!(source, "    let tid = thread_m * THREADS_N + thread_n;")
                .expect("write to string");
        }
        GemmThreadOrder::MThenN => {
            writeln!(source, "    let thread_m = thread::threadIdx_x() as usize;")
                .expect("write to string");
            writeln!(source, "    let thread_n = thread::threadIdx_y() as usize;")
                .expect("write to string");
            writeln!(
                source,
                "    if thread_m >= THREADS_M || thread_n >= THREADS_N {{"
            )
            .expect("write to string");
            writeln!(source, "        return;").expect("write to string");
            writeln!(source, "    }}").expect("write to string");
            writeln!(source, "    let tid = thread_n * THREADS_M + thread_m;")
                .expect("write to string");
        }
    }
    writeln!(source).expect("write to string");
    for row_output in 0..m_per_thread {
        if row_output == 0 {
            writeln!(source, "    let tile_row0 = thread_m * THREAD_TILE_M;")
                .expect("write to string");
        } else {
            writeln!(
                source,
                "    let tile_row{row_output} = thread_m * THREAD_TILE_M + {row_output};"
            )
            .expect("write to string");
        }
        writeln!(
            source,
            "    let row{row_output} = thread::blockIdx_y() as usize * TILE_M + tile_row{row_output};"
        )
        .expect("write to string");
    }
    for output in 0..n_per_thread {
        if output == 0 {
            writeln!(source, "    let tile_col0 = thread_n * THREAD_TILE_N;")
                .expect("write to string");
        } else {
            writeln!(
                source,
                "    let tile_col{output} = thread_n * THREAD_TILE_N + {output};"
            )
            .expect("write to string");
        }
        writeln!(
            source,
            "    let col{output} = thread::blockIdx_x() as usize * TILE_N + tile_col{output};"
        )
        .expect("write to string");
    }
    writeln!(source, "    let m = m as usize;").expect("write to string");
    writeln!(source, "    let n = n as usize;").expect("write to string");
    writeln!(source, "    let k = k as usize;").expect("write to string");
    writeln!(source, "    let a_row_stride = a_row_stride as usize;").expect("write to string");
    writeln!(source, "    let a_col_stride = a_col_stride as usize;").expect("write to string");
    writeln!(source, "    let b_row_stride = b_row_stride as usize;").expect("write to string");
    writeln!(source, "    let b_col_stride = b_col_stride as usize;").expect("write to string");
    writeln!(source, "    let c_row_stride = c_row_stride as usize;").expect("write to string");
    writeln!(source, "    let c_col_stride = c_col_stride as usize;").expect("write to string");
    for row_output in 0..m_per_thread {
        for col_output in 0..n_per_thread {
            let acc = row_output * n_per_thread + col_output;
            writeln!(source, "    let mut acc{acc} = 0.0_f32;").expect("write to string");
        }
    }
    writeln!(source, "    let mut k_base = 0;").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "    while k_base < k {{").expect("write to string");
    writeln!(
        source,
        "        let mut load = if tid < A_LOAD_THREADS {{ tid }} else {{ TILE_A_ELEMS }};"
    )
    .expect("write to string");
    writeln!(source, "        while load < TILE_A_ELEMS {{").expect("write to string");
    if a_load_unroll > 1 {
        render_gemm_a_load_unrolled(
            &mut source,
            a_load_unroll,
            m_contiguous_a_load,
            "A_LOAD_THREADS",
        );
    } else {
        render_gemm_a_load_single(&mut source, m_contiguous_a_load);
    }
    writeln!(source, "        }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(
        source,
        "        load = if tid < B_LOAD_THREADS {{ tid }} else {{ TILE_B_ELEMS }};"
    )
    .expect("write to string");
    writeln!(source, "        while load < TILE_B_ELEMS {{").expect("write to string");
    if b_load_unroll > 1 {
        render_gemm_b_load_unrolled(
            &mut source,
            b_load_unroll,
            k_contiguous_b_load,
            "B_LOAD_THREADS",
        );
    } else {
        render_gemm_b_load_single(&mut source, k_contiguous_b_load);
    }
    writeln!(source, "        }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "        thread::sync_threads();").expect("write to string");
    writeln!(source).expect("write to string");
    render_gemm_reduce_loop(&mut source, plan);
    writeln!(source).expect("write to string");
    writeln!(source, "        thread::sync_threads();").expect("write to string");
    writeln!(source, "        k_base += TILE_K;").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source).expect("write to string");
    for row_output in 0..m_per_thread {
        for col_output in 0..n_per_thread {
            let acc = row_output * n_per_thread + col_output;
            writeln!(
                source,
                "    if row{row_output} < m && col{col_output} < n {{"
            )
            .expect("write to string");
            writeln!(
                source,
                "        let c_offset = row{row_output} * c_row_stride + col{col_output} * c_col_stride;"
            )
            .expect("write to string");
            writeln!(source, "        unsafe {{").expect("write to string");
            writeln!(
                source,
                "            let c_elem = c.get_unchecked_mut(c_offset);"
            )
            .expect("write to string");
            writeln!(source, "            let current = *c_elem;").expect("write to string");
            writeln!(
                source,
                "            *c_elem = alpha * acc{acc} + beta * current;"
            )
            .expect("write to string");
            writeln!(source, "        }}").expect("write to string");
            writeln!(source, "    }}").expect("write to string");
        }
    }
    writeln!(source, "}}").expect("write to string");
    source
}
