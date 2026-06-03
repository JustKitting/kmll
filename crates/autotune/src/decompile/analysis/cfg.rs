use std::collections::BTreeSet;

use super::{
    super::{KernelIrFunction, KernelIrOp, KernelIrOpKind},
    types::{
        SassBasicBlock, SassBlockTerminator, SassCfgEdge, SassCfgEdgeKind, SassDominatorBlock,
        SassNaturalLoop,
    },
};

pub(super) fn analyze_control_structure(
    blocks: &[SassBasicBlock],
    edges: &[SassCfgEdge],
) -> (Vec<SassDominatorBlock>, Vec<SassNaturalLoop>) {
    if blocks.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let successors = successors_by_block(blocks.len(), edges);
    let reachable = reachable_blocks(&successors);
    let predecessors = predecessors_by_block(blocks.len(), edges);
    let dominator_sets = compute_dominator_sets(blocks.len(), &predecessors, &reachable);
    let dominators = build_dominator_blocks(&dominator_sets, &reachable);
    let natural_loops = recover_natural_loops(edges, &predecessors, &dominator_sets, &reachable);
    (dominators, natural_loops)
}

fn compute_dominator_sets(
    block_count: usize,
    predecessors: &[Vec<usize>],
    reachable: &BTreeSet<usize>,
) -> Vec<BTreeSet<usize>> {
    let all_reachable = reachable.iter().copied().collect::<BTreeSet<_>>();
    let mut dominators = (0..block_count)
        .map(|block_id| {
            if reachable.contains(&block_id) {
                all_reachable.clone()
            } else {
                BTreeSet::from([block_id])
            }
        })
        .collect::<Vec<_>>();
    if block_count > 0 {
        dominators[0] = BTreeSet::from([0]);
    }

    loop {
        let mut changed = false;
        for block_id in 1..block_count {
            if !reachable.contains(&block_id) {
                continue;
            }
            let reachable_predecessors = predecessors[block_id]
                .iter()
                .copied()
                .filter(|predecessor| reachable.contains(predecessor))
                .collect::<Vec<_>>();
            let mut next = if reachable_predecessors.is_empty() {
                BTreeSet::new()
            } else {
                reachable_predecessors
                    .iter()
                    .map(|pred| dominators[*pred].clone())
                    .reduce(|lhs, rhs| lhs.intersection(&rhs).copied().collect())
                    .unwrap_or_default()
            };
            next.insert(block_id);
            if next != dominators[block_id] {
                dominators[block_id] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    dominators
}

fn build_dominator_blocks(
    dominator_sets: &[BTreeSet<usize>],
    reachable: &BTreeSet<usize>,
) -> Vec<SassDominatorBlock> {
    (0..dominator_sets.len())
        .map(|block_id| {
            let immediate_dominator = immediate_dominator(block_id, dominator_sets);
            let dominated_blocks = (0..dominator_sets.len())
                .filter(|candidate| dominator_sets[*candidate].contains(&block_id))
                .collect::<Vec<_>>();
            SassDominatorBlock {
                block_id,
                reachable: reachable.contains(&block_id),
                immediate_dominator,
                dominators: dominator_sets[block_id].iter().copied().collect(),
                dominated_blocks,
            }
        })
        .collect()
}

fn immediate_dominator(block_id: usize, dominator_sets: &[BTreeSet<usize>]) -> Option<usize> {
    if block_id == 0 {
        return None;
    }
    dominator_sets[block_id]
        .iter()
        .copied()
        .filter(|dominator| *dominator != block_id)
        .max_by_key(|dominator| dominator_sets[*dominator].len())
}

fn recover_natural_loops(
    edges: &[SassCfgEdge],
    predecessors: &[Vec<usize>],
    dominator_sets: &[BTreeSet<usize>],
    reachable: &BTreeSet<usize>,
) -> Vec<SassNaturalLoop> {
    let mut loops = Vec::new();
    for edge in edges {
        let Some(header) = edge.to_block else {
            continue;
        };
        if !dominates(dominator_sets, header, edge.from_block) {
            continue;
        }
        let mut blocks = BTreeSet::from([header, edge.from_block]);
        let mut stack = vec![edge.from_block];
        while let Some(block) = stack.pop() {
            for predecessor in &predecessors[block] {
                if *predecessor != header && !dominates(dominator_sets, header, *predecessor) {
                    continue;
                }
                if blocks.insert(*predecessor) {
                    stack.push(*predecessor);
                }
            }
        }
        loops.push(SassNaturalLoop {
            header_block: header,
            latch_block: edge.from_block,
            reachable: reachable.contains(&header) && reachable.contains(&edge.from_block),
            blocks: blocks.into_iter().collect(),
            edge_condition: edge.condition.clone(),
            edge_target: edge.target.clone(),
        });
    }
    loops.sort_by(|lhs, rhs| {
        lhs.header_block
            .cmp(&rhs.header_block)
            .then_with(|| lhs.latch_block.cmp(&rhs.latch_block))
    });
    loops
}

fn dominates(dominator_sets: &[BTreeSet<usize>], dominator: usize, block: usize) -> bool {
    dominator_sets
        .get(block)
        .is_some_and(|dominators| dominators.contains(&dominator))
}

pub(super) fn build_blocks(function: &KernelIrFunction) -> Vec<SassBasicBlock> {
    if function.ops.is_empty() {
        return Vec::new();
    }

    let mut starts = BTreeSet::new();
    starts.insert(0usize);
    for (index, op) in function.ops.iter().enumerate() {
        if op.label.is_some() {
            starts.insert(index);
        }
        if is_block_boundary_terminator(op) && index + 1 < function.ops.len() {
            starts.insert(index + 1);
        }
        if let Some(target) = branch_target(op) {
            if let Some(target_index) = label_index(function, target) {
                starts.insert(target_index);
            }
        }
    }

    let starts = starts.into_iter().collect::<Vec<_>>();
    let mut blocks = Vec::new();
    for (id, start) in starts.iter().copied().enumerate() {
        let end = starts
            .get(id + 1)
            .copied()
            .unwrap_or(function.ops.len())
            .saturating_sub(1);
        let start_op = &function.ops[start];
        let end_op = &function.ops[end];
        blocks.push(SassBasicBlock {
            id,
            label: start_op.label.clone(),
            start_address: start_op.address,
            end_address: end_op.address,
            start_op_index: start,
            end_op_index: end,
            instruction_count: end.saturating_sub(start) + 1,
            terminator: terminator_for(end_op),
        });
    }
    blocks
}

pub(super) fn build_edges(
    function: &KernelIrFunction,
    blocks: &[SassBasicBlock],
) -> Vec<SassCfgEdge> {
    let mut edges = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        let Some(last_op) = function.ops.get(block.end_op_index) else {
            continue;
        };
        let next_block = blocks.get(index + 1).map(|block| block.id);
        match &last_op.kind {
            KernelIrOpKind::Branch { target, condition } => {
                let target_block = target
                    .as_deref()
                    .and_then(|target| label_index(function, target))
                    .and_then(|target_index| block_id_for_op_index(blocks, target_index));
                edges.push(SassCfgEdge {
                    from_block: block.id,
                    to_block: target_block,
                    kind: SassCfgEdgeKind::Branch,
                    condition: condition.clone(),
                    target: target.clone(),
                });
                if condition.is_some() {
                    if let Some(next_block) = next_block {
                        edges.push(SassCfgEdge {
                            from_block: block.id,
                            to_block: Some(next_block),
                            kind: SassCfgEdgeKind::Fallthrough,
                            condition: None,
                            target: None,
                        });
                    }
                }
            }
            KernelIrOpKind::Call { target, .. } => {
                edges.push(SassCfgEdge {
                    from_block: block.id,
                    to_block: None,
                    kind: SassCfgEdgeKind::Call,
                    condition: None,
                    target: target.clone(),
                });
                if let Some(next_block) = next_block {
                    edges.push(SassCfgEdge {
                        from_block: block.id,
                        to_block: Some(next_block),
                        kind: SassCfgEdgeKind::Fallthrough,
                        condition: None,
                        target: None,
                    });
                }
            }
            KernelIrOpKind::Return { target, .. } => {
                edges.push(SassCfgEdge {
                    from_block: block.id,
                    to_block: None,
                    kind: SassCfgEdgeKind::Return,
                    condition: last_op.predicate.clone(),
                    target: target.clone(),
                });
                if last_op.predicate.is_some() {
                    if let Some(next_block) = next_block {
                        edges.push(SassCfgEdge {
                            from_block: block.id,
                            to_block: Some(next_block),
                            kind: SassCfgEdgeKind::Fallthrough,
                            condition: None,
                            target: None,
                        });
                    }
                }
            }
            KernelIrOpKind::Exit { condition } => {
                edges.push(SassCfgEdge {
                    from_block: block.id,
                    to_block: None,
                    kind: SassCfgEdgeKind::Exit,
                    condition: condition.clone(),
                    target: None,
                });
                if condition.is_some() {
                    if let Some(next_block) = next_block {
                        edges.push(SassCfgEdge {
                            from_block: block.id,
                            to_block: Some(next_block),
                            kind: SassCfgEdgeKind::Fallthrough,
                            condition: None,
                            target: None,
                        });
                    }
                }
            }
            _ => {
                if let Some(next_block) = next_block {
                    edges.push(SassCfgEdge {
                        from_block: block.id,
                        to_block: Some(next_block),
                        kind: SassCfgEdgeKind::Fallthrough,
                        condition: None,
                        target: None,
                    });
                }
            }
        }
    }
    edges
}
pub(super) fn predecessors_by_block(block_count: usize, edges: &[SassCfgEdge]) -> Vec<Vec<usize>> {
    let mut predecessors = vec![Vec::new(); block_count];
    for edge in edges {
        if let Some(to_block) = edge.to_block {
            if to_block < block_count && !predecessors[to_block].contains(&edge.from_block) {
                predecessors[to_block].push(edge.from_block);
            }
        }
    }
    predecessors
}

pub(super) fn successors_by_block(block_count: usize, edges: &[SassCfgEdge]) -> Vec<Vec<usize>> {
    let mut successors = vec![Vec::new(); block_count];
    for edge in edges {
        let Some(to_block) = edge.to_block else {
            continue;
        };
        if edge.from_block < block_count
            && to_block < block_count
            && !successors[edge.from_block].contains(&to_block)
        {
            successors[edge.from_block].push(to_block);
        }
    }
    successors
}

fn reachable_blocks(successors: &[Vec<usize>]) -> BTreeSet<usize> {
    if successors.is_empty() {
        return BTreeSet::new();
    }
    let mut reachable = BTreeSet::from([0usize]);
    let mut stack = vec![0usize];
    while let Some(block) = stack.pop() {
        for successor in &successors[block] {
            if reachable.insert(*successor) {
                stack.push(*successor);
            }
        }
    }
    reachable
}
fn is_block_boundary_terminator(op: &KernelIrOp) -> bool {
    matches!(
        op.kind,
        KernelIrOpKind::Branch { .. }
            | KernelIrOpKind::Call { .. }
            | KernelIrOpKind::Return { .. }
            | KernelIrOpKind::Exit { .. }
    )
}

fn terminator_for(op: &KernelIrOp) -> SassBlockTerminator {
    match op.kind {
        KernelIrOpKind::Branch { .. } => SassBlockTerminator::Branch,
        KernelIrOpKind::Call { .. } => SassBlockTerminator::Call,
        KernelIrOpKind::Return { .. } => SassBlockTerminator::Return,
        KernelIrOpKind::Exit { .. } => SassBlockTerminator::Exit,
        _ => SassBlockTerminator::Fallthrough,
    }
}

fn branch_target(op: &KernelIrOp) -> Option<&str> {
    match &op.kind {
        KernelIrOpKind::Branch {
            target: Some(target),
            ..
        } => Some(target),
        _ => None,
    }
}

fn label_index(function: &KernelIrFunction, label: &str) -> Option<usize> {
    function
        .ops
        .iter()
        .position(|op| op.label.as_deref() == Some(label))
}

pub(super) fn block_id_for_op_index(blocks: &[SassBasicBlock], op_index: usize) -> Option<usize> {
    blocks
        .iter()
        .find(|block| block.start_op_index <= op_index && op_index <= block.end_op_index)
        .map(|block| block.id)
}
