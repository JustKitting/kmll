use std::collections::{BTreeMap, BTreeSet};

use super::{
    super::{KernelIrFunction, KernelIrOp, KernelIrOpKind, RegisterRef, SassOpcode},
    cfg::{block_id_for_op_index, predecessors_by_block},
    registers::push_register_refs,
    types::{
        SassBasicBlock, SassCfgEdge, SassDataflowOp, SassDefUseEdge, SassLiveRange,
        SassReachingUse, SassSsaValue, SassValueOp, SassValueOpKind,
    },
};

pub(super) fn build_value_ops(
    function: &KernelIrFunction,
    blocks: &[SassBasicBlock],
    dataflow: &[SassDataflowOp],
    ssa_values: &[SassSsaValue],
    def_use_edges: &[SassDefUseEdge],
) -> Vec<SassValueOp> {
    let mut values_by_definition = BTreeMap::<(RegisterRef, Option<u64>), Vec<usize>>::new();
    for value in ssa_values {
        values_by_definition
            .entry((value.register.clone(), value.def_address))
            .or_default()
            .push(value.value_id);
    }

    let mut input_values_by_address = BTreeMap::<u64, BTreeSet<usize>>::new();
    for edge in def_use_edges {
        input_values_by_address
            .entry(edge.use_address)
            .or_default()
            .insert(edge.value_id);
    }

    function
        .ops
        .iter()
        .enumerate()
        .map(|(op_index, op)| {
            let dataflow = &dataflow[op_index];
            let output_value_ids = dataflow
                .defines
                .iter()
                .filter_map(|register| {
                    values_by_definition.get(&(register.clone(), Some(op.address)))
                })
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let input_value_ids = input_values_by_address
                .get(&op.address)
                .map(|values| values.iter().copied().collect())
                .unwrap_or_default();
            SassValueOp {
                address: op.address,
                block_id: block_id_for_op_index(blocks, op_index),
                predicate: op.predicate.clone(),
                opcode: SassOpcode::from_ir_op(op),
                kind: SassValueOpKind::from_ir_kind(&op.kind),
                input_registers: dataflow.uses.clone(),
                output_registers: dataflow.defines.clone(),
                input_value_ids,
                output_value_ids,
                source: op.source.clone(),
            }
        })
        .collect()
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct ReachingDef {
    register: RegisterRef,
    address: Option<u64>,
    source: Option<String>,
}

pub(super) fn analyze_reaching_defs(
    blocks: &[SassBasicBlock],
    edges: &[SassCfgEdge],
    dataflow: &[SassDataflowOp],
) -> (
    Vec<SassReachingUse>,
    Vec<SassLiveRange>,
    Vec<SassSsaValue>,
    Vec<SassDefUseEdge>,
) {
    if blocks.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let mut definitions = Vec::new();
    let mut entry_def_by_register = BTreeMap::<RegisterRef, usize>::new();
    let mut def_by_op_register = BTreeMap::<(usize, RegisterRef), usize>::new();
    let mut defs_by_register = BTreeMap::<RegisterRef, BTreeSet<usize>>::new();
    let mut registers = BTreeSet::<RegisterRef>::new();
    for op in dataflow {
        registers.extend(op.defines.iter().cloned());
        registers.extend(op.uses.iter().cloned());
    }
    for register in registers {
        let def_id = definitions.len();
        definitions.push(ReachingDef {
            register: register.clone(),
            address: None,
            source: None,
        });
        entry_def_by_register.insert(register.clone(), def_id);
        defs_by_register.entry(register).or_default().insert(def_id);
    }
    for (op_index, op) in dataflow.iter().enumerate() {
        for register in &op.defines {
            let def_id = definitions.len();
            definitions.push(ReachingDef {
                register: register.clone(),
                address: Some(op.address),
                source: Some(op.source.clone()),
            });
            def_by_op_register.insert((op_index, register.clone()), def_id);
            defs_by_register
                .entry(register.clone())
                .or_default()
                .insert(def_id);
        }
    }

    let predecessors = predecessors_by_block(blocks.len(), edges);
    let entry_defs = definitions
        .iter()
        .enumerate()
        .filter_map(|(def_id, definition)| definition.address.is_none().then_some(def_id))
        .collect::<BTreeSet<_>>();
    let mut in_sets = vec![BTreeSet::<usize>::new(); blocks.len()];
    let mut out_sets = vec![BTreeSet::<usize>::new(); blocks.len()];
    loop {
        let mut changed = false;
        for block in blocks {
            let mut next_in = BTreeSet::new();
            if block.id == 0 {
                next_in.extend(entry_defs.iter().copied());
            }
            for pred in &predecessors[block.id] {
                next_in.extend(out_sets[*pred].iter().copied());
            }
            let next_out = transfer_block(
                block,
                dataflow,
                &def_by_op_register,
                &defs_by_register,
                next_in.clone(),
            );
            if next_in != in_sets[block.id] || next_out != out_sets[block.id] {
                in_sets[block.id] = next_in;
                out_sets[block.id] = next_out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut reaching_uses = Vec::new();
    let mut def_use_edges = Vec::new();
    let mut ssa_value_uses = (0..definitions.len())
        .map(|def_id| (def_id, BTreeSet::<u64>::new()))
        .collect::<BTreeMap<_, _>>();
    let mut live_range_uses = BTreeMap::<(RegisterRef, Option<u64>), BTreeSet<u64>>::new();
    for definition in &definitions {
        if let Some(address) = definition.address {
            live_range_uses
                .entry((definition.register.clone(), Some(address)))
                .or_default();
        }
    }

    for block in blocks {
        let mut state = in_sets[block.id].clone();
        for op_index in block.start_op_index..=block.end_op_index {
            let op = &dataflow[op_index];
            for register in &op.uses {
                let reaching = reaching_defs_for_register(&state, &definitions, register);
                let value_def_ids = if reaching.is_empty() {
                    entry_def_by_register
                        .get(register)
                        .copied()
                        .into_iter()
                        .collect::<Vec<_>>()
                } else {
                    reaching.clone()
                };
                let reaches_entry = reaching.is_empty();
                let reaches_entry = reaches_entry
                    || reaching
                        .iter()
                        .any(|def_id| definitions[*def_id].address.is_none());
                let reaching_def_addresses = reaching
                    .iter()
                    .filter_map(|def_id| definitions[*def_id].address)
                    .collect::<Vec<_>>();
                if reaches_entry {
                    live_range_uses
                        .entry((register.clone(), None))
                        .or_default()
                        .insert(op.address);
                }
                for address in &reaching_def_addresses {
                    live_range_uses
                        .entry((register.clone(), Some(*address)))
                        .or_default()
                        .insert(op.address);
                }
                for def_id in &value_def_ids {
                    ssa_value_uses
                        .entry(*def_id)
                        .or_default()
                        .insert(op.address);
                    def_use_edges.push(SassDefUseEdge {
                        value_id: *def_id,
                        register: register.clone(),
                        def_address: definitions[*def_id].address,
                        use_address: op.address,
                        use_source: op.source.clone(),
                    });
                }
                reaching_uses.push(SassReachingUse {
                    address: op.address,
                    register: register.clone(),
                    reaching_def_addresses,
                    reaches_entry,
                });
            }
            apply_defs(
                op_index,
                op,
                &mut state,
                &def_by_op_register,
                &defs_by_register,
            );
        }
    }

    let mut live_ranges = live_range_uses
        .into_iter()
        .map(|((register, def_address), uses)| {
            let start_address = def_address
                .or_else(|| uses.iter().next().copied())
                .unwrap_or(0);
            let end_address = uses.iter().next_back().copied().unwrap_or(start_address);
            SassLiveRange {
                register,
                def_address,
                start_address,
                end_address,
                use_addresses: uses.into_iter().collect(),
            }
        })
        .collect::<Vec<_>>();
    live_ranges.sort_by(|lhs, rhs| {
        lhs.register
            .cmp(&rhs.register)
            .then_with(|| lhs.start_address.cmp(&rhs.start_address))
            .then_with(|| lhs.def_address.cmp(&rhs.def_address))
    });
    reaching_uses.sort_by(|lhs, rhs| {
        lhs.address
            .cmp(&rhs.address)
            .then_with(|| lhs.register.cmp(&rhs.register))
    });
    let mut ssa_values = definitions
        .iter()
        .enumerate()
        .map(|(value_id, definition)| SassSsaValue {
            value_id,
            register: definition.register.clone(),
            def_address: definition.address,
            source: definition.source.clone(),
            use_addresses: ssa_value_uses
                .remove(&value_id)
                .unwrap_or_default()
                .into_iter()
                .collect(),
        })
        .collect::<Vec<_>>();
    ssa_values.sort_by_key(|value| value.value_id);
    def_use_edges.sort_by(|lhs, rhs| {
        lhs.use_address
            .cmp(&rhs.use_address)
            .then_with(|| lhs.register.cmp(&rhs.register))
            .then_with(|| lhs.value_id.cmp(&rhs.value_id))
    });
    (reaching_uses, live_ranges, ssa_values, def_use_edges)
}
fn transfer_block(
    block: &SassBasicBlock,
    dataflow: &[SassDataflowOp],
    def_by_op_register: &BTreeMap<(usize, RegisterRef), usize>,
    defs_by_register: &BTreeMap<RegisterRef, BTreeSet<usize>>,
    mut state: BTreeSet<usize>,
) -> BTreeSet<usize> {
    for op_index in block.start_op_index..=block.end_op_index {
        apply_defs(
            op_index,
            &dataflow[op_index],
            &mut state,
            def_by_op_register,
            defs_by_register,
        );
    }
    state
}

fn apply_defs(
    op_index: usize,
    op: &SassDataflowOp,
    state: &mut BTreeSet<usize>,
    def_by_op_register: &BTreeMap<(usize, RegisterRef), usize>,
    defs_by_register: &BTreeMap<RegisterRef, BTreeSet<usize>>,
) {
    let conditional_write = op.predicate.is_some();
    for register in &op.defines {
        if !conditional_write {
            if let Some(kill_set) = defs_by_register.get(register) {
                for def_id in kill_set {
                    state.remove(def_id);
                }
            }
        }
        if let Some(def_id) = def_by_op_register.get(&(op_index, register.clone())) {
            state.insert(*def_id);
        }
    }
}

fn reaching_defs_for_register(
    state: &BTreeSet<usize>,
    definitions: &[ReachingDef],
    register: &RegisterRef,
) -> Vec<usize> {
    state
        .iter()
        .copied()
        .filter(|def_id| &definitions[*def_id].register == register)
        .collect()
}

pub(super) fn analyze_dataflow(op: &KernelIrOp) -> SassDataflowOp {
    let mut defines = Vec::new();
    let mut uses = Vec::new();
    if let Some(predicate) = &op.predicate {
        push_register_refs(predicate.registers(), &mut uses);
    }
    match &op.kind {
        KernelIrOpKind::ReadSpecialRegister { dst, special } => {
            push_register_refs([dst.clone()], &mut defines);
            push_register_refs([special.clone()], &mut uses);
        }
        KernelIrOpKind::Move { dst, src } => {
            push_register_refs([dst.clone()], &mut defines);
            push_register_refs(src.registers(), &mut uses);
        }
        KernelIrOpKind::LoadConst { dst, source } => {
            push_register_refs([dst.clone()], &mut defines);
            push_register_refs(source.registers(), &mut uses);
        }
        KernelIrOpKind::Load { dst, address, .. } => {
            push_register_refs([dst.clone()], &mut defines);
            push_register_refs(address.registers(), &mut uses);
        }
        KernelIrOpKind::Store { address, value, .. } => {
            push_register_refs(address.registers(), &mut uses);
            push_register_refs([value.clone()], &mut uses);
        }
        KernelIrOpKind::IntegerAdd { dst, inputs, .. }
        | KernelIrOpKind::PackedHalfAdd { dst, inputs, .. }
        | KernelIrOpKind::PackedHalfMul { dst, inputs, .. }
        | KernelIrOpKind::Shift { dst, inputs }
        | KernelIrOpKind::LogicLut { dst, inputs }
        | KernelIrOpKind::Permute { dst, inputs }
        | KernelIrOpKind::AddressCalc { dst, inputs } => {
            push_register_refs([dst.clone()], &mut defines);
            for input in inputs {
                push_register_refs(input.registers(), &mut uses);
            }
        }
        KernelIrOpKind::FloatAdd { dst, lhs, rhs } | KernelIrOpKind::FloatMul { dst, lhs, rhs } => {
            push_register_refs([dst.clone()], &mut defines);
            push_register_refs(lhs.registers(), &mut uses);
            push_register_refs(rhs.registers(), &mut uses);
        }
        KernelIrOpKind::FusedMultiplyAdd { dst, a, b, c, .. }
        | KernelIrOpKind::IntegerMad { dst, a, b, c, .. } => {
            push_register_refs([dst.clone()], &mut defines);
            push_register_refs(a.registers(), &mut uses);
            push_register_refs(b.registers(), &mut uses);
            push_register_refs(c.registers(), &mut uses);
        }
        KernelIrOpKind::CompareSet { dst, lhs, rhs, .. } => {
            push_register_refs([dst.clone()], &mut defines);
            push_register_refs(lhs.registers(), &mut uses);
            push_register_refs(rhs.registers(), &mut uses);
        }
        KernelIrOpKind::Branch { condition, .. } | KernelIrOpKind::Exit { condition } => {
            if let Some(condition) = condition {
                push_register_refs(condition.registers(), &mut uses);
            }
        }
        KernelIrOpKind::Call { operands, .. }
        | KernelIrOpKind::Return { operands, .. }
        | KernelIrOpKind::TensorCoreMma { operands, .. }
        | KernelIrOpKind::TensorCoreMemory { operands, .. }
        | KernelIrOpKind::TensorMemoryAccess { operands, .. }
        | KernelIrOpKind::WarpGroup { operands, .. }
        | KernelIrOpKind::Sync { operands, .. } => {
            for operand in operands {
                push_register_refs(operand.registers(), &mut uses);
            }
        }
        KernelIrOpKind::WarpShuffle {
            predicate,
            dst,
            src,
            offset,
            mask,
            ..
        } => {
            push_register_refs([dst.clone()], &mut defines);
            push_register_refs([predicate.clone()], &mut uses);
            push_register_refs(src.registers(), &mut uses);
            push_register_refs(offset.registers(), &mut uses);
            push_register_refs(mask.registers(), &mut uses);
        }
        KernelIrOpKind::NoOp | KernelIrOpKind::Unsupported { .. } => {}
    }
    SassDataflowOp {
        address: op.address,
        predicate: op.predicate.clone(),
        defines,
        uses,
        source: op.source.clone(),
    }
}
