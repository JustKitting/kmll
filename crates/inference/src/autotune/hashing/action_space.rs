use super::primitives::*;
use super::*;

pub(in crate::autotune) fn hash_optional_profiling_action_space_set(
    state: u64,
    action_space: Option<&ProfilingActionSpaceSet>,
) -> u64 {
    if let Some(action_space) = action_space {
        hash_profiling_action_space_set(state, action_space)
    } else {
        hash_str(state, "no-action-space")
    }
}

pub(in crate::autotune) fn hash_profiling_action_space_set(
    mut state: u64,
    action_space: &ProfilingActionSpaceSet,
) -> u64 {
    state = hash_str(state, "profiling-action-space-set");
    state = hash_u64(state, action_space.spaces.len() as u64);
    for space in &action_space.spaces {
        state = hash_profiling_action_space(state, space);
    }
    state
}

pub(in crate::autotune) fn hash_profiling_action_space(
    mut state: u64,
    action_space: &ProfilingActionSpace,
) -> u64 {
    match action_space {
        ProfilingActionSpace::Split { variants } => {
            state = hash_str(state, "split");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.axis as u64);
                state = hash_u64(state, variant.factor as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        ProfilingActionSpace::Upcast { axis, factors } => {
            state = hash_str(state, "upcast");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::Unroll { axis, factors } => {
            state = hash_str(state, "unroll");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::LocalTile { axis, factors } => {
            state = hash_str(state, "local-tile");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::ThreadGroup { axis, factors } => {
            state = hash_str(state, "thread-group");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        ProfilingActionSpace::TileGemm { variants } => {
            state = hash_str(state, "tile-gemm");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.m as u64);
                state = hash_u64(state, variant.n as u64);
                state = hash_u64(state, variant.k as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        ProfilingActionSpace::StrideOrder { orders } => {
            state = hash_str(state, "stride-order");
            state = hash_u64(state, orders.len() as u64);
            for order in orders {
                state = hash_u64(state, order.len() as u64);
                for axis in order {
                    state = hash_u64(state, *axis as u64);
                }
            }
            state
        }
        ProfilingActionSpace::Swap { pairs } => {
            state = hash_str(state, "swap");
            state = hash_u64(state, pairs.len() as u64);
            for (axis_a, axis_b) in pairs {
                state = hash_u64(state, *axis_a as u64);
                state = hash_u64(state, *axis_b as u64);
            }
            state
        }
    }
}

pub(in crate::autotune) fn hash_action_space_set(
    mut state: u64,
    action_space: &KernelActionSpaceSet,
) -> u64 {
    state = hash_str(state, "action-space-set");
    state = hash_u64(state, action_space.spaces.len() as u64);
    for space in &action_space.spaces {
        state = hash_action_space(state, space);
    }
    state
}

pub(in crate::autotune) fn hash_action_space(
    mut state: u64,
    action_space: &KernelActionSpace,
) -> u64 {
    match action_space {
        KernelActionSpace::Split { variants } => {
            state = hash_str(state, "split");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.axis as u64);
                state = hash_u64(state, variant.factor as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        KernelActionSpace::Upcast { axis, factors } => {
            state = hash_str(state, "upcast");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        KernelActionSpace::Unroll { axis, factors } => {
            state = hash_str(state, "unroll");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        KernelActionSpace::LocalTile { axis, factors } => {
            state = hash_str(state, "local-tile");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        KernelActionSpace::ThreadGroup { axis, factors } => {
            state = hash_str(state, "thread-group");
            state = hash_u64(state, *axis as u64);
            state = hash_u64(state, factors.len() as u64);
            for factor in factors {
                state = hash_u64(state, *factor as u64);
            }
            state
        }
        KernelActionSpace::TileGemm { variants } => {
            state = hash_str(state, "tile-gemm");
            state = hash_u64(state, variants.len() as u64);
            for variant in variants {
                state = hash_u64(state, variant.tile.m as u64);
                state = hash_u64(state, variant.tile.n as u64);
                state = hash_u64(state, variant.tile.k as u64);
                state = hash_str(state, variant.materialization.label());
            }
            state
        }
        KernelActionSpace::StrideOrder { orders } => {
            state = hash_str(state, "stride-order");
            state = hash_u64(state, orders.len() as u64);
            for order in orders {
                state = hash_u64(state, order.len() as u64);
                for axis in order {
                    state = hash_u64(state, *axis as u64);
                }
            }
            state
        }
        KernelActionSpace::Swap { pairs } => {
            state = hash_str(state, "swap");
            state = hash_u64(state, pairs.len() as u64);
            for (axis_a, axis_b) in pairs {
                state = hash_u64(state, *axis_a as u64);
                state = hash_u64(state, *axis_b as u64);
            }
            state
        }
    }
}
