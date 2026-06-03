use super::*;

pub(super) fn render_matvec_upcast_row_body(
    source: &mut String,
    row_offset: u32,
    reduce_unroll: u32,
    lanes_per_row: u32,
    has_reduce_group: bool,
    indent: &str,
) {
    let inner = format!("{indent}    ");
    writeln!(
        source,
        "{indent}let row{row_offset}_in_block = row_in_block_base + {row_offset};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}let row{row_offset} = (thread::blockIdx_x() * ROWS_PER_BLOCK + row{row_offset}_in_block) as usize;"
    )
    .expect("write to string");
    writeln!(
        source,
        "{indent}if row{row_offset}_in_block < ROWS_PER_BLOCK && row{row_offset} < rows as usize {{"
    )
    .expect("write to string");
    writeln!(
        source,
        "{inner}let row_base = row{row_offset} * row_stride;"
    )
    .expect("write to string");
    writeln!(source, "{inner}let mut acc = 0.0_f32;").expect("write to string");
    if has_reduce_group {
        render_grouped_matvec_reduce_loop(
            source,
            reduce_unroll,
            lanes_per_row,
            &inner,
            &render_single_row_accumulate,
        );
    } else {
        writeln!(source, "{inner}let mut col = lane as usize;").expect("write to string");
        render_flat_matvec_reduce_loop(
            source,
            reduce_unroll,
            lanes_per_row,
            &inner,
            "cols",
            &render_single_row_accumulate,
        );
    }
    writeln!(source, "{inner}let acc = warp_reduce_sum(acc);").expect("write to string");
    writeln!(source, "{inner}if lane == 0 {{").expect("write to string");
    writeln!(source, "{inner}    unsafe {{").expect("write to string");
    writeln!(
        source,
        "{inner}        *out.get_unchecked_mut(row{row_offset}) = acc;"
    )
    .expect("write to string");
    writeln!(source, "{inner}    }}").expect("write to string");
    writeln!(source, "{inner}}}").expect("write to string");
    writeln!(source, "{indent}}}").expect("write to string");
    writeln!(source).expect("write to string");
}

pub(super) fn render_interleaved_matvec_upcast_body(
    source: &mut String,
    rows_per_upcast: u32,
    reduce_unroll: u32,
    lanes_per_row: u32,
    has_reduce_group: bool,
    indent: &str,
) {
    let inner = format!("{indent}    ");
    for row_offset in 0..rows_per_upcast {
        writeln!(
            source,
            "{indent}let row{row_offset}_in_block = row_in_block_base + {row_offset};"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let row{row_offset} = (thread::blockIdx_x() * ROWS_PER_BLOCK + row{row_offset}_in_block) as usize;"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let row{row_offset}_active = row{row_offset}_in_block < ROWS_PER_BLOCK && row{row_offset} < rows as usize;"
        )
        .expect("write to string");
        writeln!(
            source,
            "{indent}let row{row_offset}_base = row{row_offset} * row_stride;"
        )
        .expect("write to string");
        writeln!(source, "{indent}let mut acc{row_offset} = 0.0_f32;").expect("write to string");
    }
    writeln!(source).expect("write to string");
    if has_reduce_group {
        let accumulate = |source: &mut String, col_name: &str, indent: &str| {
            render_interleaved_row_accumulates(source, rows_per_upcast, col_name, indent)
        };
        render_grouped_matvec_reduce_loop(
            source,
            reduce_unroll,
            lanes_per_row,
            indent,
            &accumulate,
        );
    } else {
        writeln!(source, "{indent}let mut col = lane as usize;").expect("write to string");
        let accumulate = |source: &mut String, col_name: &str, indent: &str| {
            render_interleaved_row_accumulates(source, rows_per_upcast, col_name, indent)
        };
        render_flat_matvec_reduce_loop(
            source,
            reduce_unroll,
            lanes_per_row,
            indent,
            "cols",
            &accumulate,
        );
    }
    writeln!(source).expect("write to string");
    for row_offset in 0..rows_per_upcast {
        writeln!(
            source,
            "{indent}let acc{row_offset} = warp_reduce_sum(acc{row_offset});"
        )
        .expect("write to string");
        writeln!(source, "{indent}if lane == 0 && row{row_offset}_active {{")
            .expect("write to string");
        writeln!(source, "{inner}unsafe {{").expect("write to string");
        writeln!(
            source,
            "{inner}    *out.get_unchecked_mut(row{row_offset}) = acc{row_offset};"
        )
        .expect("write to string");
        writeln!(source, "{inner}}}").expect("write to string");
        writeln!(source, "{indent}}}").expect("write to string");
    }
    writeln!(source).expect("write to string");
}

fn render_grouped_matvec_reduce_loop(
    source: &mut String,
    reduce_unroll: u32,
    lanes_per_row: u32,
    inner: &str,
    render_accumulate: &impl Fn(&mut String, &str, &str),
) {
    writeln!(source, "{inner}let mut col_group = 0usize;").expect("write to string");
    writeln!(source, "{inner}while col_group < cols {{").expect("write to string");
    writeln!(
        source,
        "{inner}    let col_group_end = if col_group + REDUCE_GROUP as usize < cols {{ col_group + REDUCE_GROUP as usize }} else {{ cols }};"
    )
    .expect("write to string");
    writeln!(
        source,
        "{inner}    let mut col = col_group + lane as usize;"
    )
    .expect("write to string");
    render_flat_matvec_reduce_loop(
        source,
        reduce_unroll,
        lanes_per_row,
        &format!("{inner}    "),
        "col_group_end",
        render_accumulate,
    );
    writeln!(source, "{inner}    col_group += REDUCE_GROUP as usize;").expect("write to string");
    writeln!(source, "{inner}}}").expect("write to string");
}

fn render_flat_matvec_reduce_loop(
    source: &mut String,
    reduce_unroll: u32,
    lanes_per_row: u32,
    inner: &str,
    limit: &str,
    render_accumulate: &impl Fn(&mut String, &str, &str),
) {
    let last_offset = (reduce_unroll - 1) * lanes_per_row;
    let stride = reduce_unroll * lanes_per_row;
    if reduce_unroll > 1 {
        writeln!(source, "{inner}while col + {last_offset} < {limit} {{").expect("write to string");
        for offset in 0..reduce_unroll {
            let col_expr = if offset == 0 {
                "col".to_string()
            } else {
                format!("col + {}", offset * lanes_per_row)
            };
            writeln!(source, "{inner}    let col{offset} = {col_expr};").expect("write to string");
        }
        for offset in 0..reduce_unroll {
            render_accumulate(source, &format!("col{offset}"), &format!("{inner}    "));
        }
        writeln!(source, "{inner}    col += {stride};").expect("write to string");
        writeln!(source, "{inner}}}").expect("write to string");
    }
    writeln!(source, "{inner}while col < {limit} {{").expect("write to string");
    render_accumulate(source, "col", &format!("{inner}    "));
    writeln!(source, "{inner}    col += LANES_PER_ROW as usize;").expect("write to string");
    writeln!(source, "{inner}}}").expect("write to string");
}

fn render_single_row_accumulate(source: &mut String, col_name: &str, indent: &str) {
    writeln!(
        source,
        "{indent}acc += weight[row_base + {col_name} * col_stride].to_f32() * input[{col_name}];"
    )
    .expect("write to string");
}

fn render_interleaved_row_accumulates(
    source: &mut String,
    rows_per_upcast: u32,
    col_name: &str,
    indent: &str,
) {
    writeln!(source, "{indent}let input_{col_name} = input[{col_name}];").expect("write to string");
    for row_offset in 0..rows_per_upcast {
        writeln!(source, "{indent}if row{row_offset}_active {{").expect("write to string");
        writeln!(
            source,
            "{indent}    acc{row_offset} += weight[row{row_offset}_base + {col_name} * col_stride].to_f32() * input_{col_name};"
        )
        .expect("write to string");
        writeln!(source, "{indent}}}").expect("write to string");
    }
}
