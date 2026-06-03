use std::{collections::BTreeSet, fmt::Write as _};

use super::{KernelIrFunction, KernelIrModule, KernelIrOp, KernelIrOpKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassAnalysisModule {
    pub target: Option<String>,
    pub functions: Vec<SassAnalysisFunction>,
}

impl SassAnalysisModule {
    pub fn block_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.blocks.len())
            .sum()
    }

    pub fn edge_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.edges.len())
            .sum()
    }

    pub fn dataflow_op_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.dataflow.len())
            .sum()
    }

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        if let Some(target) = &self.target {
            writeln!(out, "target {target}").expect("write to string");
        }
        for function in &self.functions {
            writeln!(out, "fn {} {{", function.name).expect("write to string");
            writeln!(out, "  blocks").expect("write to string");
            for block in &function.blocks {
                writeln!(
                    out,
                    "    b{} {:#06x}-{:#06x} label={} instructions={} terminator={}",
                    block.id,
                    block.start_address,
                    block.end_address,
                    block.label.as_deref().unwrap_or("-"),
                    block.instruction_count,
                    block.terminator
                )
                .expect("write to string");
            }
            writeln!(out, "  edges").expect("write to string");
            for edge in &function.edges {
                writeln!(
                    out,
                    "    b{} -> {} [{} condition={} target={}]",
                    edge.from_block,
                    edge.to_block
                        .map(|block| format!("b{block}"))
                        .unwrap_or_else(|| "external".to_string()),
                    edge.kind,
                    edge.condition.as_deref().unwrap_or("-"),
                    edge.target.as_deref().unwrap_or("-")
                )
                .expect("write to string");
            }
            writeln!(out, "  dataflow").expect("write to string");
            for dataflow in &function.dataflow {
                writeln!(
                    out,
                    "    {:#06x}: def=[{}] use=[{}] <- {}",
                    dataflow.address,
                    dataflow.defines.join(","),
                    dataflow.uses.join(","),
                    dataflow.source
                )
                .expect("write to string");
            }
            writeln!(out, "}}").expect("write to string");
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassAnalysisFunction {
    pub name: String,
    pub blocks: Vec<SassBasicBlock>,
    pub edges: Vec<SassCfgEdge>,
    pub dataflow: Vec<SassDataflowOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassBasicBlock {
    pub id: usize,
    pub label: Option<String>,
    pub start_address: u64,
    pub end_address: u64,
    pub start_op_index: usize,
    pub end_op_index: usize,
    pub instruction_count: usize,
    pub terminator: SassBlockTerminator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassBlockTerminator {
    Fallthrough,
    Branch,
    Call,
    Return,
    Exit,
}

impl std::fmt::Display for SassBlockTerminator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fallthrough => f.write_str("fallthrough"),
            Self::Branch => f.write_str("branch"),
            Self::Call => f.write_str("call"),
            Self::Return => f.write_str("return"),
            Self::Exit => f.write_str("exit"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCfgEdge {
    pub from_block: usize,
    pub to_block: Option<usize>,
    pub kind: SassCfgEdgeKind,
    pub condition: Option<String>,
    pub target: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassCfgEdgeKind {
    Fallthrough,
    Branch,
    Call,
    Return,
    Exit,
}

impl std::fmt::Display for SassCfgEdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fallthrough => f.write_str("fallthrough"),
            Self::Branch => f.write_str("branch"),
            Self::Call => f.write_str("call"),
            Self::Return => f.write_str("return"),
            Self::Exit => f.write_str("exit"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassDataflowOp {
    pub address: u64,
    pub defines: Vec<String>,
    pub uses: Vec<String>,
    pub source: String,
}

pub fn analyze_sass_ir(module: &KernelIrModule) -> SassAnalysisModule {
    SassAnalysisModule {
        target: module.target.clone(),
        functions: module.functions.iter().map(analyze_function).collect(),
    }
}

fn analyze_function(function: &KernelIrFunction) -> SassAnalysisFunction {
    let blocks = build_blocks(function);
    let edges = build_edges(function, &blocks);
    let dataflow = function.ops.iter().map(analyze_dataflow).collect();
    SassAnalysisFunction {
        name: function.name.clone(),
        blocks,
        edges,
        dataflow,
    }
}

fn build_blocks(function: &KernelIrFunction) -> Vec<SassBasicBlock> {
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

fn build_edges(function: &KernelIrFunction, blocks: &[SassBasicBlock]) -> Vec<SassCfgEdge> {
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

fn analyze_dataflow(op: &KernelIrOp) -> SassDataflowOp {
    let mut defines = Vec::new();
    let mut uses = Vec::new();
    if let Some(predicate) = &op.predicate {
        push_registers(predicate, &mut uses);
    }
    match &op.kind {
        KernelIrOpKind::ReadSpecialRegister { dst, special } => {
            push_registers(dst, &mut defines);
            push_registers(special, &mut uses);
        }
        KernelIrOpKind::Move { dst, src } => {
            push_registers(dst, &mut defines);
            push_registers(src, &mut uses);
        }
        KernelIrOpKind::LoadConst { dst, source } => {
            push_registers(dst, &mut defines);
            push_registers(source, &mut uses);
        }
        KernelIrOpKind::Load { dst, address, .. } => {
            push_registers(dst, &mut defines);
            push_registers(address, &mut uses);
        }
        KernelIrOpKind::Store { address, value, .. } => {
            push_registers(address, &mut uses);
            push_registers(value, &mut uses);
        }
        KernelIrOpKind::IntegerAdd { dst, inputs, .. }
        | KernelIrOpKind::PackedHalfAdd { dst, inputs, .. }
        | KernelIrOpKind::PackedHalfMul { dst, inputs, .. }
        | KernelIrOpKind::Shift { dst, inputs }
        | KernelIrOpKind::LogicLut { dst, inputs }
        | KernelIrOpKind::Permute { dst, inputs }
        | KernelIrOpKind::AddressCalc { dst, inputs } => {
            push_registers(dst, &mut defines);
            for input in inputs {
                push_registers(input, &mut uses);
            }
        }
        KernelIrOpKind::FloatAdd { dst, lhs, rhs } | KernelIrOpKind::FloatMul { dst, lhs, rhs } => {
            push_registers(dst, &mut defines);
            push_registers(lhs, &mut uses);
            push_registers(rhs, &mut uses);
        }
        KernelIrOpKind::FusedMultiplyAdd { dst, a, b, c, .. }
        | KernelIrOpKind::IntegerMad { dst, a, b, c, .. } => {
            push_registers(dst, &mut defines);
            push_registers(a, &mut uses);
            push_registers(b, &mut uses);
            push_registers(c, &mut uses);
        }
        KernelIrOpKind::CompareSet { dst, lhs, rhs, .. } => {
            push_registers(dst, &mut defines);
            push_registers(lhs, &mut uses);
            push_registers(rhs, &mut uses);
        }
        KernelIrOpKind::Branch { condition, .. } | KernelIrOpKind::Exit { condition } => {
            if let Some(condition) = condition {
                push_registers(condition, &mut uses);
            }
        }
        KernelIrOpKind::Call { operands, .. }
        | KernelIrOpKind::Return { operands, .. }
        | KernelIrOpKind::Sync { operands, .. } => {
            for operand in operands {
                push_registers(operand, &mut uses);
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
            push_registers(dst, &mut defines);
            push_registers(predicate, &mut uses);
            push_registers(src, &mut uses);
            push_registers(offset, &mut uses);
            push_registers(mask, &mut uses);
        }
        KernelIrOpKind::NoOp | KernelIrOpKind::Unsupported { .. } => {}
    }
    SassDataflowOp {
        address: op.address,
        defines,
        uses,
        source: op.source.clone(),
    }
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

fn block_id_for_op_index(blocks: &[SassBasicBlock], op_index: usize) -> Option<usize> {
    blocks
        .iter()
        .find(|block| block.start_op_index <= op_index && op_index <= block.end_op_index)
        .map(|block| block.id)
}

fn push_registers(text: &str, out: &mut Vec<String>) {
    for register in extract_registers(text) {
        if !out.contains(&register) {
            out.push(register);
        }
    }
}

fn extract_registers(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut index = 0usize;
    let mut registers = Vec::new();
    while index < bytes.len() {
        let Some((register, consumed)) = parse_register_at(text, index) else {
            index += 1;
            continue;
        };
        if !is_pseudo_register(&register) && !registers.contains(&register) {
            registers.push(register);
        }
        index += consumed;
    }
    registers
}

fn parse_register_at(text: &str, index: usize) -> Option<(String, usize)> {
    if !is_token_boundary(text, index) {
        return None;
    }
    let tail = &text[index..];
    for literal in ["SR_", "URZ", "UPT", "UR", "UP", "RZ", "PT", "R", "P", "B"] {
        if let Some(register) = parse_register_prefix(tail, literal) {
            return Some(register);
        }
    }
    None
}

fn parse_register_prefix(tail: &str, prefix: &str) -> Option<(String, usize)> {
    let rest = tail.strip_prefix(prefix)?;
    match prefix {
        "SR_" => {
            let len = rest
                .char_indices()
                .take_while(|(_, ch)| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '.')
                .map(|(index, ch)| index + ch.len_utf8())
                .last()
                .unwrap_or(0);
            (len > 0).then(|| (tail[..prefix.len() + len].to_string(), prefix.len() + len))
        }
        "URZ" | "UPT" | "RZ" | "PT" => Some((prefix.to_string(), prefix.len())),
        "UR" | "UP" | "R" | "P" | "B" => {
            let len = rest
                .char_indices()
                .take_while(|(_, ch)| ch.is_ascii_digit())
                .map(|(index, ch)| index + ch.len_utf8())
                .last()
                .unwrap_or(0);
            (len > 0).then(|| (tail[..prefix.len() + len].to_string(), prefix.len() + len))
        }
        _ => None,
    }
}

fn is_token_boundary(text: &str, index: usize) -> bool {
    if index == 0 {
        return true;
    }
    let before = text[..index]
        .chars()
        .next_back()
        .expect("index > 0 should have previous char");
    !before.is_ascii_alphanumeric() && before != '_'
}

fn is_pseudo_register(register: &str) -> bool {
    matches!(register, "RZ" | "URZ" | "PT" | "UPT")
}
