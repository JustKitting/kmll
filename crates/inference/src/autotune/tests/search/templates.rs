use super::*;

#[test]
fn schedule_action_template_captures_tinygrad_like_beam_factors() {
    let template = KernelScheduleActionTemplate::TINYGRAD_LIKE;

    assert_eq!(template.upcast_factors, &[2, 3, 4, 5, 7]);
    assert_eq!(template.unroll_factors, &[4, 7]);
    assert_eq!(template.local_tile_factors, &[2, 3, 4, 8, 13, 16, 29, 32]);
    assert_eq!(
        template.group_top_factors,
        &[13, 16, 28, 29, 32, 49, 64, 256]
    );
    assert_eq!(
        template.thread_group_factors,
        &[2, 3, 4, 5, 8, 12, 16, 24, 32, 64]
    );
}

#[test]
fn inference_template_keeps_arbitrary_legal_split_and_unroll_factors() {
    let template = KernelScheduleActionTemplate::INFERENCE_DEFAULT;

    assert_eq!(
        template.bounded_split_factors(4096, 32, None),
        vec![8, 13, 16, 24, 32]
    );
    assert_eq!(template.bounded_split_factors(10, 32, None), vec![8, 10]);
    assert_eq!(
        template.exhaustive_unroll_factors(4096, 32, Some(4)),
        (1..=32).filter(|factor| *factor != 4).collect::<Vec<_>>()
    );
    assert_eq!(
        template.legal_upcast_factors(|factor| factor <= 4 && 16 % factor == 0),
        vec![2, 4]
    );
    assert_eq!(
        template.legal_thread_group_factors(|factor| factor >= 32 && factor < 128),
        vec![32, 64]
    );
}
