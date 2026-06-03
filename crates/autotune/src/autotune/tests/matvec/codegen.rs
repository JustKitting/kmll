use super::*;

#[test]
fn matvec_generator_renders_naive_source_on_demand() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_naive_candidate();
    let generated = MatvecRustCudaGenerator
        .source_for(&candidate)
        .expect("matvec generator should render naive source");

    assert_eq!(generated.symbol, "matvec_bf16_naive");
    assert_eq!(candidate.launch.grid_dim.x, 4096);
    assert_eq!(candidate.launch.block_dim.x, 1);
    assert!(generated.source.contains("#[kernel]"));
    assert!(generated.source.contains("pub fn matvec_bf16_naive("));
    assert!(generated.source.contains("if thread::threadIdx_x() != 0"));
    assert!(
        generated
            .source
            .contains("let row = thread::blockIdx_x() as usize;")
    );
    assert!(generated.source.contains("while col < cols"));
    assert!(
        generated
            .source
            .contains("acc += weight[row_base + col * col_stride].to_f32() * input[col];")
    );
    assert!(!generated.source.contains("warp::shuffle_down_f32"));
}

#[test]
fn matvec_generator_renders_rows_per_block_source_on_demand() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_row_split(
        MatvecRowSplit::new(13).expect("rows13 should be a legal generated split"),
    );
    let generated = MatvecRustCudaGenerator
        .source_for(&candidate)
        .expect("matvec generator should render rows-per-block source");

    assert_eq!(generated.symbol, "matvec_bf16_rows13");
    assert!(generated.source.contains("#[kernel]"));
    assert!(generated.source.contains("pub fn matvec_bf16_rows13("));
    assert!(generated.source.contains("pub struct Bf16(u16);"));
    assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 13;"));
    assert!(generated.source.contains("const ROWS_PER_UPCAST: u32 = 1;"));
    assert!(
        generated
            .source
            .contains("const ROW_GROUPS_PER_BLOCK: u32 = 13;")
    );
    assert!(generated.source.contains("const REDUCE_UNROLL: u32 = 4;"));
    assert!(
        generated
            .source
            .contains("let row_group_in_block = thread_x / LANES_PER_ROW;")
    );
    assert!(
        generated
            .source
            .contains("let row0_in_block = row_in_block_base + 0;")
    );
    assert!(generated.source.contains("while col + 96 < cols"));
    assert!(generated.source.contains("warp::shuffle_down_f32(acc, 16)"));
}

#[test]
fn matvec_generator_renders_reduce_unroll_source_on_demand() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_plan(
        MatvecSchedulePlan::new(RowMajorWarpRows::Rows8).with_reduce_unroll(8),
    );
    let generated = MatvecRustCudaGenerator
        .source_for(&candidate)
        .expect("matvec generator should render reduce-unrolled source");

    assert_eq!(generated.symbol, "matvec_bf16_rows8_u8");
    assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
    assert!(generated.source.contains("const REDUCE_UNROLL: u32 = 8;"));
    assert!(generated.source.contains("while col + 224 < cols"));
    assert!(generated.source.contains("let col7 = col + 224;"));
    assert!(
        generated
            .source
            .contains("acc += weight[row_base + col7 * col_stride].to_f32() * input[col7];")
    );
    assert!(generated.source.contains("col += 256;"));
}

#[test]
fn matvec_generator_renders_reduce_group_source_on_demand() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_plan(
        MatvecSchedulePlan::new(RowMajorWarpRows::Rows8).with_reduce_group(64),
    );
    let generated = MatvecRustCudaGenerator
        .source_for(&candidate)
        .expect("matvec generator should render reduce-grouped source");

    assert_eq!(generated.symbol, "matvec_bf16_rows8_cg64");
    assert!(generated.source.contains("const REDUCE_GROUP: u32 = 64;"));
    assert!(generated.source.contains("let mut col_group = 0usize;"));
    assert!(generated.source.contains("while col_group < cols"));
    assert!(
        generated
            .source
            .contains("let col_group_end = if col_group + REDUCE_GROUP as usize < cols")
    );
    assert!(
        generated
            .source
            .contains("let mut col = col_group + lane as usize;")
    );
    assert!(generated.source.contains("while col + 96 < col_group_end"));
    assert!(generated.source.contains("while col < col_group_end"));
    assert!(
        generated
            .source
            .contains("col_group += REDUCE_GROUP as usize;")
    );
}

#[test]
fn matvec_generator_renders_row_upcast_source_on_demand() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_plan(
        MatvecSchedulePlan::new(RowMajorWarpRows::Rows8)
            .with_row_upcast(MatvecRowUpcast::new(2).expect("2 rows should be a supported upcast")),
    );
    let generated = MatvecRustCudaGenerator
        .source_for(&candidate)
        .expect("matvec generator should render row-upcast source");

    assert_eq!(generated.symbol, "matvec_bf16_rows8_up2");
    assert_eq!(candidate.launch.block_dim.x, 128);
    assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
    assert!(generated.source.contains("const ROWS_PER_UPCAST: u32 = 2;"));
    assert!(
        generated
            .source
            .contains("const ROW_GROUPS_PER_BLOCK: u32 = 4;")
    );
    assert!(
        generated
            .source
            .contains("let row1_in_block = row_in_block_base + 1;")
    );
    assert!(
        generated
            .source
            .contains("if row1_in_block < ROWS_PER_BLOCK && row1 < rows as usize")
    );
    assert!(
        generated
            .source
            .contains("*out.get_unchecked_mut(row1) = acc;")
    );
}

#[test]
fn matvec_generator_renders_reduction_first_row_upcast_source_on_demand() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_plan(
        MatvecSchedulePlan::new(RowMajorWarpRows::Rows8)
            .with_row_upcast(MatvecRowUpcast::new(2).expect("2 rows should be a supported upcast"))
            .with_loop_order(MatvecLoopOrder::ReductionThenRow),
    );
    let generated = MatvecRustCudaGenerator
        .source_for(&candidate)
        .expect("matvec generator should render reduction-first row-upcast source");

    assert_eq!(generated.symbol, "matvec_bf16_rows8_up2_rf");
    assert!(generated.source.contains("let row0_active ="));
    assert!(generated.source.contains("let row1_active ="));
    assert!(generated.source.contains("let mut acc0 = 0.0_f32;"));
    assert!(generated.source.contains("let mut acc1 = 0.0_f32;"));
    assert!(generated.source.contains("let mut col = lane as usize;"));
    assert!(generated.source.contains("let input_col0 = input[col0];"));
    assert!(
        generated
            .source
            .contains("acc0 += weight[row0_base + col0 * col_stride].to_f32() * input_col0;")
    );
    assert!(
        generated
            .source
            .contains("acc1 += weight[row1_base + col0 * col_stride].to_f32() * input_col0;")
    );
    assert!(
        generated
            .source
            .contains("let acc1 = warp_reduce_sum(acc1);")
    );
    assert!(generated.source.contains("if lane == 0 && row1_active"));
}

#[test]
fn matvec_generator_renders_thread_group_source_on_demand() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_plan(
        MatvecSchedulePlan::new(RowMajorWarpRows::Rows8)
            .with_reduce_unroll(8)
            .with_thread_group(MatvecThreadGroup::new(16).expect("16 lanes should be supported")),
    );
    let generated = MatvecRustCudaGenerator
        .source_for(&candidate)
        .expect("matvec generator should render thread-grouped source");

    assert_eq!(generated.symbol, "matvec_bf16_rows8_u8_tg16");
    assert_eq!(candidate.launch.block_dim.x, 128);
    assert!(generated.source.contains("const LANES_PER_ROW: u32 = 16;"));
    assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
    assert!(generated.source.contains("const REDUCE_UNROLL: u32 = 8;"));
    assert!(generated.source.contains("warp::shuffle_down_f32(acc, 8)"));
    assert!(!generated.source.contains("warp::shuffle_down_f32(acc, 16)"));
    assert!(
        generated
            .source
            .contains("let lane = warp::lane_id() % LANES_PER_ROW;")
    );
    assert!(generated.source.contains("while col + 112 < cols"));
    assert!(generated.source.contains("let col7 = col + 112;"));
    assert!(generated.source.contains("col += 128;"));
    assert!(generated.source.contains("col += LANES_PER_ROW as usize;"));
}

#[test]
fn matvec_generator_renders_two_lane_thread_group_source_on_demand() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let candidate = problem.generated_candidate_for_plan(
        MatvecSchedulePlan::new(RowMajorWarpRows::Rows8)
            .with_thread_group(MatvecThreadGroup::new(2).expect("2 lanes should be supported")),
    );
    let generated = MatvecRustCudaGenerator
        .source_for(&candidate)
        .expect("matvec generator should render two-lane thread-grouped source");

    assert_eq!(generated.symbol, "matvec_bf16_rows8_tg2");
    assert_eq!(candidate.launch.block_dim.x, 16);
    assert!(generated.source.contains("const LANES_PER_ROW: u32 = 2;"));
    assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
    assert!(generated.source.contains("warp::shuffle_down_f32(acc, 1)"));
    assert!(!generated.source.contains("warp::shuffle_down_f32(acc, 2)"));
    assert!(generated.source.contains("while col + 6 < cols"));
    assert!(generated.source.contains("let col3 = col + 6;"));
    assert!(generated.source.contains("col += 8;"));
}
