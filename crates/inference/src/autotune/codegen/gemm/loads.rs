use std::fmt::Write as _;

fn render_gemm_a_load_body(
    source: &mut String,
    load_name: &str,
    suffix: u32,
    indent: &str,
    m_contiguous_a_load: bool,
) {
    if m_contiguous_a_load {
        writeln!(
            source,
            "{indent}let a_tile_row{suffix} = {load_name} % TILE_M;"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let a_tile_col{suffix} = {load_name} / TILE_M;"
        )
        .expect("write to string");
    } else {
        writeln!(
            source,
            "{indent}let a_tile_row{suffix} = {load_name} / TILE_K;"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let a_tile_col{suffix} = {load_name} % TILE_K;"
        )
        .expect("write to string");
    }
    writeln!(
        source,
        "{indent}let a_smem_index{suffix} = a_tile_row{suffix} * TILE_K + a_tile_col{suffix};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}let a_global_row{suffix} = thread::blockIdx_y() as usize * TILE_M + a_tile_row{suffix};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}let a_global_col{suffix} = k_base + a_tile_col{suffix};"
    )
    .expect("write to string");
    writeln!(source, "{indent}unsafe {{").expect("write to string");
    writeln!(
        source,
        "{indent}    TILE_A[a_smem_index{suffix}] = if a_global_row{suffix} < m && a_global_col{suffix} < k {{"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}        a[a_global_row{suffix} * a_row_stride + a_global_col{suffix} * a_col_stride]"
    )
    .expect("write to string");
    writeln!(source, "{indent}    }} else {{").expect("write to string");
    writeln!(source, "{indent}        0.0").expect("write to string");
    writeln!(source, "{indent}    }};").expect("write to string");
    writeln!(source, "{indent}}}").expect("write to string");
}

fn render_gemm_b_load_body(
    source: &mut String,
    load_name: &str,
    suffix: u32,
    indent: &str,
    k_contiguous_b_load: bool,
) {
    if k_contiguous_b_load {
        writeln!(
            source,
            "{indent}let b_tile_row{suffix} = {load_name} % TILE_K;"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let b_tile_col{suffix} = {load_name} / TILE_K;"
        )
        .expect("write to string");
    } else {
        writeln!(
            source,
            "{indent}let b_tile_row{suffix} = {load_name} / TILE_N;"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let b_tile_col{suffix} = {load_name} % TILE_N;"
        )
        .expect("write to string");
    }
    writeln!(
        source,
        "{indent}let b_smem_index{suffix} = b_tile_row{suffix} * TILE_N + b_tile_col{suffix};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}let b_global_row{suffix} = k_base + b_tile_row{suffix};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}let b_global_col{suffix} = thread::blockIdx_x() as usize * TILE_N + b_tile_col{suffix};"
    )
    .expect("write to string");
    writeln!(source, "{indent}unsafe {{").expect("write to string");
    writeln!(
        source,
        "{indent}    TILE_B[b_smem_index{suffix}] = if b_global_row{suffix} < k && b_global_col{suffix} < n {{"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}        b[b_global_row{suffix} * b_row_stride + b_global_col{suffix} * b_col_stride].to_f32()"
    )
    .expect("write to string");
    writeln!(source, "{indent}    }} else {{").expect("write to string");
    writeln!(source, "{indent}        0.0").expect("write to string");
    writeln!(source, "{indent}    }};").expect("write to string");
    writeln!(source, "{indent}}}").expect("write to string");
}

pub(super) fn render_gemm_a_load_single(source: &mut String, m_contiguous_a_load: bool) {
    if m_contiguous_a_load {
        writeln!(source, "            let tile_row = load % TILE_M;").expect("write to string");
        writeln!(source, "            let tile_col = load / TILE_M;").expect("write to string");
    } else {
        writeln!(source, "            let tile_row = load / TILE_K;").expect("write to string");
        writeln!(source, "            let tile_col = load % TILE_K;").expect("write to string");
    }
    writeln!(
        source,
        "            let a_smem_index = tile_row * TILE_K + tile_col;"
    )
    .expect("write to string");
    writeln!(
        source,
        "            let global_row = thread::blockIdx_y() as usize * TILE_M + tile_row;"
    )
    .expect("write to string");
    writeln!(source, "            let global_col = k_base + tile_col;").expect("write to string");
    writeln!(source, "            unsafe {{").expect("write to string");
    writeln!(
        source,
        "                TILE_A[a_smem_index] = if global_row < m && global_col < k {{"
    )
    .expect("write to string");
    writeln!(
        source,
        "                    a[global_row * a_row_stride + global_col * a_col_stride]"
    )
    .expect("write to string");
    writeln!(source, "                }} else {{").expect("write to string");
    writeln!(source, "                    0.0").expect("write to string");
    writeln!(source, "                }};").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "            load += A_LOAD_THREADS;").expect("write to string");
}

pub(super) fn render_gemm_b_load_single(source: &mut String, k_contiguous_b_load: bool) {
    if k_contiguous_b_load {
        writeln!(source, "            let tile_row = load % TILE_K;").expect("write to string");
        writeln!(source, "            let tile_col = load / TILE_K;").expect("write to string");
    } else {
        writeln!(source, "            let tile_row = load / TILE_N;").expect("write to string");
        writeln!(source, "            let tile_col = load % TILE_N;").expect("write to string");
    }
    writeln!(
        source,
        "            let b_smem_index = tile_row * TILE_N + tile_col;"
    )
    .expect("write to string");
    writeln!(source, "            let global_row = k_base + tile_row;").expect("write to string");
    writeln!(
        source,
        "            let global_col = thread::blockIdx_x() as usize * TILE_N + tile_col;"
    )
    .expect("write to string");
    writeln!(source, "            unsafe {{").expect("write to string");
    writeln!(
        source,
        "                TILE_B[b_smem_index] = if global_row < k && global_col < n {{"
    )
    .expect("write to string");
    writeln!(
        source,
        "                    b[global_row * b_row_stride + global_col * b_col_stride].to_f32()"
    )
    .expect("write to string");
    writeln!(source, "                }} else {{").expect("write to string");
    writeln!(source, "                    0.0").expect("write to string");
    writeln!(source, "                }};").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "            load += B_LOAD_THREADS;").expect("write to string");
}

pub(super) fn render_gemm_a_load_unrolled(
    source: &mut String,
    a_load_unroll: u32,
    m_contiguous_a_load: bool,
    load_thread_const: &str,
) {
    for offset in 0..a_load_unroll {
        let load_name = format!("a_load{offset}");
        if offset == 0 {
            writeln!(source, "            let {load_name} = load;").expect("write to string");
            render_gemm_a_load_body(
                source,
                &load_name,
                offset,
                "            ",
                m_contiguous_a_load,
            );
        } else {
            let load_expr = if offset == 1 {
                format!("load + {load_thread_const}")
            } else {
                format!("load + {load_thread_const} * {offset}")
            };
            writeln!(source, "            let {load_name} = {load_expr};")
                .expect("write to string");
            writeln!(source, "            if {load_name} < TILE_A_ELEMS {{")
                .expect("write to string");
            render_gemm_a_load_body(
                source,
                &load_name,
                offset,
                "                ",
                m_contiguous_a_load,
            );
            writeln!(source, "            }}").expect("write to string");
        }
    }
    writeln!(
        source,
        "            load += {load_thread_const} * A_LOAD_UNROLL;"
    )
    .expect("write to string");
}

pub(super) fn render_gemm_b_load_unrolled(
    source: &mut String,
    b_load_unroll: u32,
    k_contiguous_b_load: bool,
    load_thread_const: &str,
) {
    for offset in 0..b_load_unroll {
        let load_name = format!("b_load{offset}");
        if offset == 0 {
            writeln!(source, "            let {load_name} = load;").expect("write to string");
            render_gemm_b_load_body(
                source,
                &load_name,
                offset,
                "            ",
                k_contiguous_b_load,
            );
        } else {
            let load_expr = if offset == 1 {
                format!("load + {load_thread_const}")
            } else {
                format!("load + {load_thread_const} * {offset}")
            };
            writeln!(source, "            let {load_name} = {load_expr};")
                .expect("write to string");
            writeln!(source, "            if {load_name} < TILE_B_ELEMS {{")
                .expect("write to string");
            render_gemm_b_load_body(
                source,
                &load_name,
                offset,
                "                ",
                k_contiguous_b_load,
            );
            writeln!(source, "            }}").expect("write to string");
        }
    }
    writeln!(
        source,
        "            load += {load_thread_const} * B_LOAD_UNROLL;"
    )
    .expect("write to string");
}
