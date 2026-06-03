use super::*;

#[test]
fn matvec_search_keeps_only_metadata_for_rows_per_block_variants() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let result = beam_search_metadata(
        &problem,
        BeamSearchConfig {
            beam_width: 2,
            max_depth: 1,
            require_launchable: true,
        },
    );
    let best = result
        .best
        .expect("matvec search should produce a candidate");
    assert_eq!(result.explored, 36);
    assert_eq!(result.rejected, 32);
    assert!(best.is_launchable());
    assert_eq!(best.launch.kernel, "matvec_bf16_kernel");
    assert_eq!(best.launch.grid_dim.x, 512);
    assert_eq!(best.launch.block_dim.x, 256);
    assert_eq!(schedule_rows_per_block(&best.schedule), Some(8));
}

#[test]
fn matvec_search_can_prefer_smaller_row_groups_for_tiny_outputs() {
    let problem = MatvecSearchProblem::bf16_row_major(10, 4096);
    let result = beam_search_metadata(
        &problem,
        BeamSearchConfig {
            beam_width: 1,
            max_depth: 1,
            require_launchable: true,
        },
    );
    let best = result
        .best
        .expect("matvec search should produce a candidate");
    assert_eq!(schedule_rows_per_block(&best.schedule), Some(2));
}

#[test]
fn matvec_search_exposes_generated_row_split_metadata() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let candidates = problem.expand(&seed);
    assert_eq!(candidates.len(), 36);

    let generated = candidates
        .iter()
        .find(|candidate| {
            !candidate.is_launchable() && schedule_rows_per_block(&candidate.schedule) == Some(13)
        })
        .expect("matvec search should expose arbitrary generated rows-per-block metadata");
    assert_eq!(generated.launch.kernel, "matvec_bf16_rows13");
    assert_eq!(generated.launch.grid_dim.x, 316);
    assert_eq!(generated.launch.block_dim.x, 416);
    assert_eq!(
        generated.generated.materialization,
        KernelMaterialization::Generated {
            symbol: "matvec_bf16_rows13".to_string()
        }
    );
    assert_eq!(generated.generated.generator, "row-major-matvec-generator");
}

#[test]
fn matvec_action_space_exposes_existing_and_generated_row_splits() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let spaces = problem.action_spaces(&seed);
    let actions = problem.schedule_actions(&seed);

    assert_eq!(spaces.spaces.len(), 1);
    assert_eq!(spaces.actions(), actions);
    let KernelActionSpace::Split { variants } = &spaces.spaces[0] else {
        panic!("matvec seed should expose split action-space metadata");
    };
    assert_eq!(variants.len(), 36);
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        8,
        KernelActionMaterialization::Existing
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        13,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(variants.contains(&KernelAxisFactorAction::new(
        0,
        32,
        KernelActionMaterialization::DeferredGenerated
    )));
    assert!(!variants.contains(&KernelAxisFactorAction::new(
        0,
        32,
        KernelActionMaterialization::Existing
    )));
    let complete_space = problem.search_space();
    assert_eq!(complete_space.spaces.len(), 4);
    assert!(matches!(
        complete_space.spaces[0],
        KernelActionSpace::Split { .. }
    ));
    assert!(matches!(
        complete_space.spaces[1],
        KernelActionSpace::Upcast { .. }
    ));
    assert!(matches!(
        complete_space.spaces[2],
        KernelActionSpace::Unroll { .. }
    ));
    assert!(matches!(
        complete_space.spaces[3],
        KernelActionSpace::ThreadGroup { .. }
    ));
    assert_eq!(actions.len(), 36);
    assert!(actions.contains(&KernelScheduleAction::split(
        0,
        8,
        KernelActionMaterialization::Existing
    )));
    assert!(actions.contains(&KernelScheduleAction::split(
        0,
        13,
        KernelActionMaterialization::DeferredGenerated
    )));

    let generated = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 13, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("row split action should produce candidate metadata");
    assert_eq!(generated.launch.kernel, "matvec_bf16_rows13");
    assert_eq!(schedule_rows_per_block(&generated.schedule), Some(13));
    assert_eq!(
        generated.action_trace,
        vec![KernelScheduleAction::split(
            0,
            13,
            KernelActionMaterialization::DeferredGenerated
        )]
    );
    assert_eq!(
        generated.generated.materialization,
        KernelMaterialization::Generated {
            symbol: "matvec_bf16_rows13".to_string()
        }
    );
}

#[test]
fn matvec_generated_row_split_exposes_reduce_unroll_actions() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let seed = problem.seed();
    let rows8 = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("row split action should produce generated candidate metadata");
    let spaces = problem.action_spaces(&rows8);
    let actions = problem.schedule_actions(&rows8);

    assert_eq!(spaces.spaces.len(), 3);
    assert_eq!(spaces.actions(), actions);
    let KernelActionSpace::Upcast { axis, factors } = &spaces.spaces[0] else {
        panic!("generated matvec split should expose upcast action-space metadata");
    };
    assert_eq!(*axis, 0);
    assert_eq!(
        factors.as_slice(),
        MatvecRowUpcast::SEARCH_FACTORS.as_slice()
    );
    assert!(actions.contains(&KernelScheduleAction::upcast(0, 2)));
    assert!(actions.contains(&KernelScheduleAction::upcast(0, 4)));

    let KernelActionSpace::Unroll { axis, factors } = &spaces.spaces[1] else {
        panic!("generated matvec split should expose unroll action-space metadata");
    };
    assert_eq!(*axis, 1);
    assert_eq!(factors.len(), 31);
    assert_eq!(factors.first().copied(), Some(1));
    assert_eq!(factors.last().copied(), Some(32));
    assert!(!factors.contains(&MatvecSchedulePlan::DEFAULT_REDUCE_UNROLL));
    assert!(factors.contains(&7));
    assert!(actions.contains(&KernelScheduleAction::unroll(1, 1)));
    assert!(actions.contains(&KernelScheduleAction::unroll(1, 2)));
    assert!(actions.contains(&KernelScheduleAction::unroll(1, 7)));
    assert!(actions.contains(&KernelScheduleAction::unroll(1, 32)));
    let KernelActionSpace::ThreadGroup { axis, factors } = &spaces.spaces[2] else {
        panic!("generated matvec split should expose thread-group action-space metadata");
    };
    assert_eq!(*axis, 1);
    assert_eq!(
        factors.as_slice(),
        MatvecThreadGroup::SEARCH_LANES_PER_ROW.as_slice()
    );
    assert!(actions.contains(&KernelScheduleAction::thread_group(1, 16)));

    let unrolled = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::unroll(1, 7))
        .expect("reduce unroll action should produce generated candidate metadata");
    assert_eq!(unrolled.launch.kernel, "matvec_bf16_rows8_u7");
    assert_eq!(schedule_matvec_reduce_unroll(&unrolled.schedule), Some(7));
    assert_eq!(
        unrolled.action_trace,
        vec![
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::unroll(1, 7),
        ]
    );
    assert_ne!(rows8.artifact_key(), unrolled.artifact_key());

    let upcast = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::upcast(0, 2))
        .expect("row upcast action should produce generated candidate metadata");
    assert_eq!(upcast.launch.kernel, "matvec_bf16_rows8_up2");
    assert_eq!(upcast.launch.block_dim.x, 128);
    assert_eq!(
        schedule_matvec_row_upcast(&upcast.schedule).map(MatvecRowUpcast::factor),
        Some(2)
    );
    assert_eq!(
        upcast.action_trace,
        vec![
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::upcast(0, 2),
        ]
    );
    assert_ne!(rows8.artifact_key(), upcast.artifact_key());

    let grouped = problem
        .apply_schedule_action(&rows8, &KernelScheduleAction::thread_group(1, 16))
        .expect("thread-group action should produce generated candidate metadata");
    assert_eq!(grouped.launch.kernel, "matvec_bf16_rows8_tg16");
    assert_eq!(grouped.launch.block_dim.x, 128);
    assert_eq!(
        schedule_matvec_thread_group(&grouped.schedule).map(MatvecThreadGroup::lanes_per_row),
        Some(16)
    );
    assert_eq!(
        grouped.action_trace,
        vec![
            KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
            KernelScheduleAction::thread_group(1, 16),
        ]
    );
    assert_ne!(rows8.artifact_key(), grouped.artifact_key());
}

#[test]
fn matvec_unroll_action_space_is_bounded_by_problem_shape() {
    let problem = MatvecSearchProblem::bf16_row_major(16, 3);
    let seed = problem.seed();
    let rows8 = problem
        .apply_schedule_action(
            &seed,
            &KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
        )
        .expect("row split action should produce generated candidate metadata");
    let spaces = problem.action_spaces(&rows8);
    let KernelActionSpace::Unroll { factors, .. } = spaces
        .spaces
        .iter()
        .find(|space| matches!(space, KernelActionSpace::Unroll { .. }))
        .expect("generated matvec split should expose unroll metadata")
    else {
        panic!("generated matvec split should expose unroll metadata");
    };

    assert_eq!(factors.as_slice(), &[1, 2, 3]);
    assert!(
        problem
            .apply_schedule_action(&rows8, &KernelScheduleAction::unroll(1, 3))
            .is_some()
    );
    assert!(
        problem
            .apply_schedule_action(&rows8, &KernelScheduleAction::unroll(1, 5))
            .is_none()
    );
}

#[test]
fn matvec_search_can_rank_reduce_unroll_variants_when_allowed() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let result = beam_search_metadata_with_scorer(
        &problem,
        BeamSearchConfig {
            beam_width: 40,
            max_depth: 2,
            require_launchable: false,
        },
        |candidate| {
            let plan = schedule_matvec_plan(&candidate.schedule)?;
            if plan.rows.rows_per_block() == 13 && plan.reduce_unroll == 7 {
                SearchScore::measured(0.0)
            } else {
                SearchScore::measured(
                    100.0 + f64::from(plan.rows.rows_per_block()) + f64::from(plan.reduce_unroll),
                )
            }
        },
    );
    let best = result
        .best
        .expect("matvec search should produce an unrolled generated candidate");
    let plan = schedule_matvec_plan(&best.schedule).expect("best candidate should have plan");

    assert_eq!(plan.rows.rows_per_block(), 13);
    assert_eq!(plan.reduce_unroll, 7);
    assert_eq!(best.launch.kernel, "matvec_bf16_rows13_u7");
    assert_eq!(
        best.score.map(|score| score.source),
        Some(SearchScoreSource::Measured)
    );
}

#[test]
fn action_trace_replay_reconstructs_generated_matvec_candidate() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let actions = vec![
        KernelScheduleAction::split(0, 8, KernelActionMaterialization::DeferredGenerated),
        KernelScheduleAction::unroll(1, 8),
    ];

    let candidate = replay_schedule_actions(&problem, &actions)
        .expect("valid matvec action trace should replay into candidate metadata");

    assert_eq!(candidate.family, "matvec-bf16-row-major");
    assert_eq!(candidate.action_trace, actions);
    assert_eq!(candidate.launch.kernel, "matvec_bf16_rows8_u8");
    assert_eq!(schedule_rows_per_block(&candidate.schedule), Some(8));
    assert_eq!(schedule_matvec_reduce_unroll(&candidate.schedule), Some(8));
    assert!(!candidate.is_launchable());

    let generated = MatvecRustCudaGenerator
        .source_for(&candidate)
        .expect("replayed generated candidate should render source on demand");
    assert_eq!(generated.symbol, "matvec_bf16_rows8_u8");
    assert!(generated.source.contains("const ROWS_PER_BLOCK: u32 = 8;"));
    assert!(generated.source.contains("const REDUCE_UNROLL: u32 = 8;"));
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
fn matvec_search_accepts_external_measured_scores_for_generated_candidate() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let result = beam_search_metadata_with_scorer(
        &problem,
        BeamSearchConfig {
            beam_width: 1,
            max_depth: 1,
            require_launchable: false,
        },
        |candidate| {
            let rows_per_block = schedule_rows_per_block(&candidate.schedule)?;
            let generated_bonus = if candidate.is_launchable() {
                100.0
            } else {
                0.0
            };
            SearchScore::measured(generated_bonus + (32.0 - f64::from(rows_per_block)))
        },
    );
    let best = result
        .best
        .expect("matvec search should keep externally best generated candidate");

    assert!(!best.is_launchable());
    assert_eq!(best.launch.kernel, "matvec_bf16_rows32");
    assert_eq!(schedule_rows_per_block(&best.schedule), Some(32));
    assert_eq!(
        best.score.map(|score| score.source),
        Some(SearchScoreSource::Measured)
    );
}

#[test]
fn metadata_key_changes_when_schedule_changes() {
    let problem = MatvecSearchProblem::bf16_row_major(4096, 4096);
    let rows4 = problem.candidate_for_rows(RowMajorWarpRows::Rows4);
    let rows8 = problem.candidate_for_rows(RowMajorWarpRows::Rows8);
    assert_ne!(rows4.artifact_key(), rows8.artifact_key());
    assert_ne!(
        rows4.generated.artifact_key.hex(),
        rows8.generated.artifact_key.hex()
    );
}
