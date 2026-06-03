use std::collections::{BTreeMap, BTreeSet};

use super::{
    super::{ControlTarget, KernelIrFunction, KernelIrOpKind, PredicateCondition},
    types::{
        SassBasicBlock, SassCfgEdge, SassCfgEdgeKind, SassNaturalLoop, SassRegion, SassRegionKind,
        SassRegionPath,
    },
};

#[derive(Debug, Clone)]
struct RegionSeed {
    kind: SassRegionKind,
    parent: Option<usize>,
    local_rank: usize,
    header_block: Option<usize>,
    latch_block: Option<usize>,
    branch_block: Option<usize>,
    entry_blocks: Vec<usize>,
    blocks: BTreeSet<usize>,
    condition: Option<PredicateCondition>,
    target: Option<ControlTarget>,
}

impl RegionSeed {
    fn new(kind: SassRegionKind, blocks: impl IntoIterator<Item = usize>) -> Self {
        Self {
            kind,
            parent: None,
            local_rank: 0,
            header_block: None,
            latch_block: None,
            branch_block: None,
            entry_blocks: Vec::new(),
            blocks: blocks.into_iter().collect(),
            condition: None,
            target: None,
        }
    }
}

pub(super) fn recover_regions(
    function: &KernelIrFunction,
    blocks: &[SassBasicBlock],
    edges: &[SassCfgEdge],
    natural_loops: &[SassNaturalLoop],
) -> Vec<SassRegion> {
    if blocks.is_empty() {
        return Vec::new();
    }

    let mut seeds = Vec::new();
    seeds.push(RegionSeed {
        entry_blocks: vec![0],
        ..RegionSeed::new(
            SassRegionKind::Function,
            blocks.iter().map(|block| block.id),
        )
    });

    append_loop_regions(&mut seeds, natural_loops);
    assign_containment_parents(&mut seeds);
    append_branch_regions(&mut seeds, blocks, edges);
    assign_local_ranks(&mut seeds);
    let children = children_by_parent(&seeds);
    build_regions(function, blocks, seeds, children)
}

fn append_loop_regions(seeds: &mut Vec<RegionSeed>, natural_loops: &[SassNaturalLoop]) {
    let mut loops = natural_loops
        .iter()
        .filter(|natural_loop| natural_loop.reachable)
        .collect::<Vec<_>>();
    loops.sort_by(|lhs, rhs| {
        rhs.blocks
            .len()
            .cmp(&lhs.blocks.len())
            .then_with(|| lhs.header_block.cmp(&rhs.header_block))
            .then_with(|| lhs.latch_block.cmp(&rhs.latch_block))
    });
    for natural_loop in loops {
        let mut seed = RegionSeed::new(SassRegionKind::NaturalLoop, natural_loop.blocks.clone());
        seed.header_block = Some(natural_loop.header_block);
        seed.latch_block = Some(natural_loop.latch_block);
        seed.entry_blocks = vec![natural_loop.header_block];
        seed.condition = natural_loop.edge_condition.clone();
        seed.target = natural_loop.edge_target.clone();
        seeds.push(seed);
    }
}

fn assign_containment_parents(seeds: &mut [RegionSeed]) {
    for index in 1..seeds.len() {
        seeds[index].parent = containing_parent(index, seeds);
    }
}

fn append_branch_regions(
    seeds: &mut Vec<RegionSeed>,
    blocks: &[SassBasicBlock],
    edges: &[SassCfgEdge],
) {
    let block_count = blocks.len();
    let successors = successors_by_block(block_count, edges);
    let postdominators = compute_postdominators(block_count, &successors);
    for block in blocks {
        let outgoing = edges
            .iter()
            .filter(|edge| edge.from_block == block.id)
            .collect::<Vec<_>>();
        if outgoing
            .iter()
            .filter(|edge| edge.to_block.is_some())
            .count()
            < 2
        {
            continue;
        }
        if !outgoing
            .iter()
            .any(|edge| edge.kind == SassCfgEdgeKind::Branch)
        {
            continue;
        }
        let postdominator = immediate_postdominator(block.id, &postdominators);
        let arm_blocks = outgoing
            .iter()
            .filter_map(|edge| {
                let entry = edge.to_block?;
                Some(branch_arm_blocks(entry, postdominator, &successors))
            })
            .collect::<Vec<_>>();
        let mut region_blocks = BTreeSet::from([block.id]);
        for arm in &arm_blocks {
            region_blocks.extend(arm.iter().copied());
        }
        if region_blocks.len() <= 1 {
            continue;
        }

        let parent = smallest_existing_container(&region_blocks, seeds);
        let mut branch_seed = RegionSeed::new(SassRegionKind::Branch, region_blocks);
        branch_seed.parent = parent;
        branch_seed.branch_block = Some(block.id);
        branch_seed.entry_blocks = outgoing.iter().filter_map(|edge| edge.to_block).collect();
        branch_seed.condition = outgoing.iter().find_map(|edge| edge.condition.clone());
        branch_seed.target = outgoing.iter().find_map(|edge| edge.target.clone());
        let branch_id = seeds.len();
        seeds.push(branch_seed);

        for edge in outgoing {
            let Some(entry) = edge.to_block else {
                continue;
            };
            let blocks = branch_arm_blocks(entry, postdominator, &successors);
            if blocks.is_empty() {
                continue;
            }
            let mut arm_seed = RegionSeed::new(SassRegionKind::BranchArm, blocks);
            arm_seed.parent = Some(branch_id);
            arm_seed.branch_block = Some(block.id);
            arm_seed.entry_blocks = vec![entry];
            arm_seed.condition = edge.condition.clone();
            arm_seed.target = edge.target.clone();
            seeds.push(arm_seed);
        }
    }
}

fn containing_parent(index: usize, seeds: &[RegionSeed]) -> Option<usize> {
    let candidate_blocks = &seeds[index].blocks;
    (0..index)
        .filter(|candidate| contains_all(&seeds[*candidate].blocks, candidate_blocks))
        .min_by_key(|candidate| seeds[*candidate].blocks.len())
}

fn smallest_existing_container(blocks: &BTreeSet<usize>, seeds: &[RegionSeed]) -> Option<usize> {
    seeds
        .iter()
        .enumerate()
        .filter(|(_, seed)| contains_all(&seed.blocks, blocks))
        .min_by_key(|(_, seed)| seed.blocks.len())
        .map(|(index, _)| index)
}

fn contains_all(lhs: &BTreeSet<usize>, rhs: &BTreeSet<usize>) -> bool {
    rhs.iter().all(|block| lhs.contains(block))
}

fn assign_local_ranks(seeds: &mut [RegionSeed]) {
    let mut by_parent = BTreeMap::<Option<usize>, Vec<usize>>::new();
    for (id, seed) in seeds.iter().enumerate() {
        by_parent.entry(seed.parent).or_default().push(id);
    }
    for siblings in by_parent.values_mut() {
        siblings.sort_by(|lhs, rhs| {
            region_sort_key(&seeds[*lhs])
                .cmp(&region_sort_key(&seeds[*rhs]))
                .then_with(|| lhs.cmp(rhs))
        });
        for (rank, id) in siblings.iter().copied().enumerate() {
            seeds[id].local_rank = rank;
        }
    }
}

fn region_sort_key(seed: &RegionSeed) -> (usize, usize, usize) {
    (
        seed.blocks.iter().next().copied().unwrap_or(usize::MAX),
        region_kind_sort_key(seed.kind),
        seed.blocks.len(),
    )
}

fn region_kind_sort_key(kind: SassRegionKind) -> usize {
    match kind {
        SassRegionKind::Function => 0,
        SassRegionKind::Branch => 1,
        SassRegionKind::BranchArm => 2,
        SassRegionKind::NaturalLoop => 3,
    }
}

fn children_by_parent(seeds: &[RegionSeed]) -> Vec<Vec<usize>> {
    let mut children = vec![Vec::new(); seeds.len()];
    for (id, seed) in seeds.iter().enumerate() {
        if let Some(parent) = seed.parent {
            if parent < children.len() {
                children[parent].push(id);
            }
        }
    }
    for child_ids in &mut children {
        child_ids.sort_by(|lhs, rhs| {
            seeds[*lhs]
                .local_rank
                .cmp(&seeds[*rhs].local_rank)
                .then_with(|| {
                    seeds[*lhs]
                        .blocks
                        .iter()
                        .next()
                        .cmp(&seeds[*rhs].blocks.iter().next())
                })
                .then_with(|| lhs.cmp(rhs))
        });
    }
    children
}

fn build_regions(
    function: &KernelIrFunction,
    blocks: &[SassBasicBlock],
    seeds: Vec<RegionSeed>,
    children: Vec<Vec<usize>>,
) -> Vec<SassRegion> {
    let mut regions = seeds
        .into_iter()
        .enumerate()
        .map(|(id, seed)| {
            let (op_addresses, opcode_closure) = region_ops(function, blocks, &seed.blocks);
            SassRegion {
                id,
                parent: seed.parent,
                children: children[id].clone(),
                depth: 0,
                path: SassRegionPath::default(),
                local_rank: seed.local_rank,
                kind: seed.kind,
                header_block: seed.header_block,
                latch_block: seed.latch_block,
                branch_block: seed.branch_block,
                entry_blocks: seed.entry_blocks,
                blocks: seed.blocks.into_iter().collect(),
                op_addresses,
                opcode_closure,
                condition: seed.condition,
                target: seed.target,
            }
        })
        .collect::<Vec<_>>();

    if !regions.is_empty() {
        assign_paths(0, SassRegionPath::default(), 0, &mut regions);
    }
    regions
}

fn assign_paths(id: usize, mut prefix: SassRegionPath, depth: usize, regions: &mut [SassRegion]) {
    prefix.push(regions[id].local_rank);
    regions[id].depth = depth;
    regions[id].path = prefix.clone();
    let children = regions[id].children.clone();
    for child in children {
        assign_paths(child, prefix.clone(), depth + 1, regions);
    }
}

fn region_ops(
    function: &KernelIrFunction,
    blocks: &[SassBasicBlock],
    region_blocks: &BTreeSet<usize>,
) -> (Vec<u64>, Vec<String>) {
    let mut addresses = Vec::new();
    let mut opcodes = BTreeSet::<String>::new();
    for block in blocks {
        if !region_blocks.contains(&block.id) {
            continue;
        }
        for op_index in block.start_op_index..=block.end_op_index {
            let Some(op) = function.ops.get(op_index) else {
                continue;
            };
            addresses.push(op.address);
            opcodes.insert(opcode_for_op(op));
        }
    }
    (addresses, opcodes.into_iter().collect())
}

fn opcode_for_op(op: &super::super::KernelIrOp) -> String {
    match &op.kind {
        KernelIrOpKind::Unsupported { opcode, .. } => opcode.clone(),
        _ => op.source_opcode.clone(),
    }
}

fn branch_arm_blocks(
    entry: usize,
    stop_block: Option<usize>,
    successors: &[Vec<usize>],
) -> BTreeSet<usize> {
    if Some(entry) == stop_block {
        return BTreeSet::new();
    }
    let mut blocks = BTreeSet::new();
    let mut stack = vec![entry];
    while let Some(block) = stack.pop() {
        if Some(block) == stop_block || !blocks.insert(block) {
            continue;
        }
        let Some(next) = successors.get(block) else {
            continue;
        };
        for successor in next {
            if Some(*successor) != stop_block {
                stack.push(*successor);
            }
        }
    }
    blocks
}

fn successors_by_block(block_count: usize, edges: &[SassCfgEdge]) -> Vec<Vec<usize>> {
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

fn compute_postdominators(block_count: usize, successors: &[Vec<usize>]) -> Vec<BTreeSet<usize>> {
    let all_blocks = (0..block_count).collect::<BTreeSet<_>>();
    let mut postdominators = (0..block_count)
        .map(|block| {
            if successors.get(block).is_none_or(Vec::is_empty) {
                BTreeSet::from([block])
            } else {
                all_blocks.clone()
            }
        })
        .collect::<Vec<_>>();

    loop {
        let mut changed = false;
        for block in 0..block_count {
            if successors[block].is_empty() {
                continue;
            }
            let mut next = successors[block]
                .iter()
                .map(|successor| postdominators[*successor].clone())
                .reduce(|lhs, rhs| lhs.intersection(&rhs).copied().collect())
                .unwrap_or_default();
            next.insert(block);
            if next != postdominators[block] {
                postdominators[block] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    postdominators
}

fn immediate_postdominator(block: usize, postdominators: &[BTreeSet<usize>]) -> Option<usize> {
    postdominators
        .get(block)?
        .iter()
        .copied()
        .filter(|candidate| *candidate != block)
        .max_by_key(|candidate| postdominators[*candidate].len())
}
