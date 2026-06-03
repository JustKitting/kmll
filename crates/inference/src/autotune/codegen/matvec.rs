use super::*;

pub(in crate::autotune) fn render_bf16_matvec_source(
    symbol: &str,
    plan: MatvecSchedulePlan,
) -> String {
    let plan = plan.normalized();
    let rows_per_block = plan.rows.rows_per_block().max(1);
    let rows_per_upcast = plan.row_upcast.factor().max(1);
    let row_groups_per_block = rows_per_block.div_ceil(rows_per_upcast);
    let lanes_per_row = plan.thread_group.lanes_per_row().max(1);
    let reduce_unroll = plan.reduce_unroll.max(1);
    let reduce_group = plan.reduce_group_size().max(1);
    let has_reduce_group = plan.has_custom_reduce_group();
    let reduce_offsets = warp_subgroup_reduce_offsets(lanes_per_row);
    let mut source = String::new();
    writeln!(
        source,
        "use cuda_device::{{DisjointSlice, kernel, thread, warp}};"
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
    writeln!(source, "const LANES_PER_ROW: u32 = {lanes_per_row};").expect("write to string");
    writeln!(source, "const ROWS_PER_BLOCK: u32 = {rows_per_block};").expect("write to string");
    writeln!(source, "const ROWS_PER_UPCAST: u32 = {rows_per_upcast};").expect("write to string");
    writeln!(
        source,
        "const ROW_GROUPS_PER_BLOCK: u32 = {row_groups_per_block};"
    )
    .expect("write to string");
    writeln!(source, "const REDUCE_UNROLL: u32 = {reduce_unroll};").expect("write to string");
    if has_reduce_group {
        writeln!(source, "const REDUCE_GROUP: u32 = {reduce_group};").expect("write to string");
    }
    writeln!(source).expect("write to string");
    writeln!(source, "#[inline(always)]").expect("write to string");
    writeln!(source, "fn warp_reduce_sum(mut acc: f32) -> f32 {{").expect("write to string");
    for offset in reduce_offsets {
        writeln!(source, "    acc += warp::shuffle_down_f32(acc, {offset});")
            .expect("write to string");
    }
    writeln!(source, "    acc").expect("write to string");
    writeln!(source, "}}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "#[kernel]").expect("write to string");
    writeln!(source, "pub fn {symbol}(").expect("write to string");
    writeln!(source, "    input: &[f32],").expect("write to string");
    writeln!(source, "    weight: &[Bf16],").expect("write to string");
    writeln!(source, "    rows: u32,").expect("write to string");
    writeln!(source, "    cols: u32,").expect("write to string");
    writeln!(source, "    row_stride: u32,").expect("write to string");
    writeln!(source, "    col_stride: u32,").expect("write to string");
    writeln!(source, "    _rows_per_block: u32,").expect("write to string");
    writeln!(source, "    mut out: DisjointSlice<f32>,").expect("write to string");
    writeln!(source, ") {{").expect("write to string");
    writeln!(source, "    let thread_x = thread::threadIdx_x();").expect("write to string");
    writeln!(
        source,
        "    let row_group_in_block = thread_x / LANES_PER_ROW;"
    )
    .expect("write to string");
    writeln!(
        source,
        "    if row_group_in_block >= ROW_GROUPS_PER_BLOCK {{"
    )
    .expect("write to string");
    writeln!(source, "        return;").expect("write to string");
    writeln!(source, "    }}").expect("write to string");
    writeln!(source).expect("write to string");
    writeln!(source, "    let lane = warp::lane_id() % LANES_PER_ROW;").expect("write to string");
    writeln!(source, "    let cols = cols as usize;").expect("write to string");
    writeln!(source, "    let row_stride = row_stride as usize;").expect("write to string");
    writeln!(source, "    let col_stride = col_stride as usize;").expect("write to string");
    writeln!(
        source,
        "    let row_in_block_base = row_group_in_block * ROWS_PER_UPCAST;"
    )
    .expect("write to string");
    writeln!(source).expect("write to string");
    for row_offset in 0..rows_per_upcast {
        render_matvec_upcast_row_body(
            &mut source,
            row_offset,
            reduce_unroll,
            lanes_per_row,
            has_reduce_group,
            "    ",
        );
    }
    writeln!(source, "}}").expect("write to string");
    source
}

pub(in crate::autotune) fn render_matvec_upcast_row_body(
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
        render_grouped_matvec_reduce_loop(source, reduce_unroll, lanes_per_row, &inner);
    } else {
        writeln!(source, "{inner}let mut col = lane as usize;").expect("write to string");
        render_flat_matvec_reduce_loop(source, reduce_unroll, lanes_per_row, &inner, "cols");
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

fn render_grouped_matvec_reduce_loop(
    source: &mut String,
    reduce_unroll: u32,
    lanes_per_row: u32,
    inner: &str,
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
            writeln!(
                source,
                "{inner}    acc += weight[row_base + col{offset} * col_stride].to_f32() * input[col{offset}];"
            )
            .expect("write to string");
        }
        writeln!(source, "{inner}    col += {stride};").expect("write to string");
        writeln!(source, "{inner}}}").expect("write to string");
    }
    writeln!(source, "{inner}while col < {limit} {{").expect("write to string");
    writeln!(
        source,
        "{inner}    acc += weight[row_base + col * col_stride].to_f32() * input[col];"
    )
    .expect("write to string");
    writeln!(source, "{inner}    col += LANES_PER_ROW as usize;").expect("write to string");
    writeln!(source, "{inner}}}").expect("write to string");
}

pub(in crate::autotune) fn warp_subgroup_reduce_offsets(lanes_per_row: u32) -> Vec<u32> {
    let mut offset = lanes_per_row / 2;
    let mut offsets = Vec::new();
    while offset > 0 {
        offsets.push(offset);
        offset /= 2;
    }
    offsets
}
