use std::fmt::Write as _;

use crate::autotune::GemmSchedulePlan;

pub(super) fn render_gemm_reduce_loop(source: &mut String, plan: GemmSchedulePlan) {
    let plan = plan.normalized();
    if plan.has_custom_reduce_group() {
        render_grouped_reduce_loop(source, plan);
    } else {
        render_flat_reduce_loop(source, plan);
    }
}

fn render_flat_reduce_loop(source: &mut String, plan: GemmSchedulePlan) {
    writeln!(source, "        unsafe {{").expect("write to string");
    writeln!(source, "            let mut kk = 0;").expect("write to string");
    writeln!(source, "            while kk + REDUCE_UNROLL <= TILE_K {{").expect("write to string");
    render_unrolled_reduce_body(
        source,
        plan,
        "kk",
        "                ",
        "                    ",
    );
    writeln!(source, "                kk += REDUCE_UNROLL;").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "            while kk < TILE_K {{").expect("write to string");
    render_scalar_reduce_body(
        source,
        plan,
        "kk",
        "                ",
        "                    ",
    );
    writeln!(source, "                kk += 1;").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "        }}").expect("write to string");
}

fn render_grouped_reduce_loop(source: &mut String, plan: GemmSchedulePlan) {
    writeln!(source, "        unsafe {{").expect("write to string");
    writeln!(source, "            let mut kk_group = 0;").expect("write to string");
    writeln!(source, "            while kk_group < TILE_K {{").expect("write to string");
    writeln!(
        source,
        "                let kk_group_end = if kk_group + REDUCE_GROUP < TILE_K {{ kk_group + REDUCE_GROUP }} else {{ TILE_K }};"
    )
    .expect("write to string");
    writeln!(source, "                let mut kk = kk_group;").expect("write to string");
    writeln!(
        source,
        "                while kk + REDUCE_UNROLL <= kk_group_end {{"
    )
    .expect("write to string");
    render_unrolled_reduce_body(
        source,
        plan,
        "kk",
        "                    ",
        "                        ",
    );
    writeln!(source, "                    kk += REDUCE_UNROLL;").expect("write to string");
    writeln!(source, "                }}").expect("write to string");
    writeln!(source, "                while kk < kk_group_end {{").expect("write to string");
    render_scalar_reduce_body(
        source,
        plan,
        "kk",
        "                    ",
        "                        ",
    );
    writeln!(source, "                    kk += 1;").expect("write to string");
    writeln!(source, "                }}").expect("write to string");
    writeln!(source, "                kk_group += REDUCE_GROUP;").expect("write to string");
    writeln!(source, "            }}").expect("write to string");
    writeln!(source, "        }}").expect("write to string");
}

fn render_unrolled_reduce_body(
    source: &mut String,
    plan: GemmSchedulePlan,
    k_base: &str,
    if_indent: &str,
    body_indent: &str,
) {
    let plan = plan.normalized();
    for offset in 0..plan.reduce_unroll.max(1) {
        let k_expr = if offset == 0 {
            k_base.to_string()
        } else {
            format!("{k_base} + {offset}")
        };
        render_reduce_accumulates(source, plan, &k_expr, if_indent, body_indent);
    }
}

fn render_scalar_reduce_body(
    source: &mut String,
    plan: GemmSchedulePlan,
    k_expr: &str,
    if_indent: &str,
    body_indent: &str,
) {
    render_reduce_accumulates(source, plan.normalized(), k_expr, if_indent, body_indent);
}

fn render_reduce_accumulates(
    source: &mut String,
    plan: GemmSchedulePlan,
    k_expr: &str,
    if_indent: &str,
    body_indent: &str,
) {
    let plan = plan.normalized();
    for row_output in 0..plan.m_per_thread.max(1) {
        for col_output in 0..plan.n_per_thread.max(1) {
            let acc = row_output * plan.n_per_thread.max(1) + col_output;
            writeln!(
                source,
                "{if_indent}if tile_row{row_output} < TILE_M && tile_col{col_output} < TILE_N {{"
            )
            .expect("write to string");
            writeln!(
                source,
                "{body_indent}acc{acc} += TILE_A[tile_row{row_output} * TILE_K + {k_expr}] * TILE_B[{} * TILE_N + tile_col{col_output}];",
                b_k_expr(k_expr)
            )
            .expect("write to string");
            writeln!(source, "{if_indent}}}").expect("write to string");
        }
    }
}

fn b_k_expr(k_expr: &str) -> String {
    if k_expr == "kk" {
        "kk".to_string()
    } else {
        format!("({k_expr})")
    }
}
