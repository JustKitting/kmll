use std::fmt::Write as _;

use super::{
    super::RegisterRef,
    types::{SassAnalysisModule, format_register_definition},
};

impl SassAnalysisModule {
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
                    block.label.as_ref().map_or("-", |label| label.as_str()),
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
                    format_optional(&edge.condition),
                    format_optional(&edge.target)
                )
                .expect("write to string");
            }
            writeln!(out, "  dominators").expect("write to string");
            for dominator in &function.dominators {
                writeln!(
                    out,
                    "    b{} reachable={} idom={} dom=[{}] dominated=[{}]",
                    dominator.block_id,
                    dominator.reachable,
                    format_block_id(dominator.immediate_dominator),
                    format_block_ids(&dominator.dominators),
                    format_block_ids(&dominator.dominated_blocks)
                )
                .expect("write to string");
            }
            writeln!(out, "  natural_loops").expect("write to string");
            for natural_loop in &function.natural_loops {
                writeln!(
                    out,
                    "    header=b{} latch=b{} reachable={} blocks=[{}] condition={} target={}",
                    natural_loop.header_block,
                    natural_loop.latch_block,
                    natural_loop.reachable,
                    format_block_ids(&natural_loop.blocks),
                    format_optional(&natural_loop.edge_condition),
                    format_optional(&natural_loop.edge_target)
                )
                .expect("write to string");
            }
            writeln!(out, "  regions").expect("write to string");
            for region in &function.regions {
                writeln!(
                    out,
                    "    r{} parent={} depth={} path=[{}] rank={} kind={} blocks=[{}] children=[{}] ops=[{}] opcodes=[{}] header={} latch={} branch={} entries=[{}] condition={} target={}",
                    region.id,
                    format_region_id(region.parent),
                    region.depth,
                    region.path,
                    region.local_rank,
                    region.kind,
                    format_block_ids(&region.blocks),
                    format_region_ids(&region.children),
                    format_addresses(&region.op_addresses),
                    format_display_list(&region.opcode_closure),
                    format_block_id(region.header_block),
                    format_block_id(region.latch_block),
                    format_block_id(region.branch_block),
                    format_block_ids(&region.entry_blocks),
                    format_optional(&region.condition),
                    format_optional(&region.target)
                )
                .expect("write to string");
            }
            writeln!(out, "  dataflow").expect("write to string");
            for dataflow in &function.dataflow {
                writeln!(
                    out,
                    "    {:#06x}: def=[{}] use=[{}] <- {}",
                    dataflow.address,
                    format_registers(&dataflow.defines),
                    format_registers(&dataflow.uses),
                    dataflow.source_text
                )
                .expect("write to string");
            }
            writeln!(out, "  reaching_uses").expect("write to string");
            for reaching in &function.reaching_uses {
                writeln!(
                    out,
                    "    {:#06x}: {} <- [{}]",
                    reaching.address,
                    reaching.register,
                    reaching.sources_text()
                )
                .expect("write to string");
            }
            writeln!(out, "  ssa_values").expect("write to string");
            for value in &function.ssa_values {
                writeln!(
                    out,
                    "    v{} {} uses=[{}] <- {}",
                    value.value_id,
                    value.name(),
                    format_addresses(&value.use_addresses),
                    value.origin
                )
                .expect("write to string");
            }
            writeln!(out, "  def_use_edges").expect("write to string");
            for edge in &function.def_use_edges {
                writeln!(
                    out,
                    "    {}: {} <- v{} {}",
                    edge.use_site,
                    edge.register,
                    edge.value_id,
                    format_register_definition(&edge.register, edge.def_address)
                )
                .expect("write to string");
            }
            writeln!(out, "  value_ops").expect("write to string");
            for op in &function.value_ops {
                writeln!(
                    out,
                    "    {:#06x}: block={} opcode={} in=[{}] out=[{}] predicate={} <- {}",
                    op.address,
                    format_block_id(op.block_id),
                    op.opcode,
                    format_value_ids(&op.input_value_ids),
                    format_value_ids(&op.output_value_ids),
                    format_optional_display(op.predicate.as_ref()),
                    op.source_text
                )
                .expect("write to string");
            }
            writeln!(out, "  live_ranges").expect("write to string");
            for range in &function.live_ranges {
                writeln!(
                    out,
                    "    {}@{} {:#06x}-{:#06x} uses=[{}]",
                    range.register,
                    range.def_text(),
                    range.start_address,
                    range.end_address,
                    format_addresses(&range.use_addresses)
                )
                .expect("write to string");
            }
            writeln!(out, "  memory_accesses").expect("write to string");
            for access in &function.memory_accesses {
                writeln!(
                    out,
                    "    {:#06x}: {} {} value={} addr={} regs=[{}] base={} offset={} width={} predicate={} <- {}",
                    access.address,
                    access.kind,
                    access.space,
                    access.value_register,
                    access.memory_address,
                    format_registers(&access.address_registers),
                    access
                        .address_base
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "-".to_string()),
                    access
                        .offset
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "-".to_string()),
                    access
                        .width_bits
                        .map(|bits| bits.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    format_optional_display(access.predicate.as_ref()),
                    access.source_text
                )
                .expect("write to string");
            }
            writeln!(out, "}}").expect("write to string");
        }
        out
    }
}

fn format_addresses(addresses: &[u64]) -> String {
    addresses
        .iter()
        .map(|address| format!("{address:#06x}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_optional<T: ToString>(value: &Option<T>) -> String {
    value
        .as_ref()
        .map(ToString::to_string)
        .unwrap_or_else(|| "-".to_string())
}

fn format_registers(registers: &[RegisterRef]) -> String {
    registers
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn format_display_list<T: ToString>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn format_optional_display<T: std::fmt::Display>(value: Option<&T>) -> String {
    value
        .map(ToString::to_string)
        .unwrap_or_else(|| "-".to_string())
}

fn format_block_id(block_id: Option<usize>) -> String {
    block_id
        .map(|block_id| format!("b{block_id}"))
        .unwrap_or_else(|| "-".to_string())
}

fn format_region_id(region_id: Option<usize>) -> String {
    region_id
        .map(|region_id| format!("r{region_id}"))
        .unwrap_or_else(|| "-".to_string())
}

fn format_block_ids(block_ids: &[usize]) -> String {
    block_ids
        .iter()
        .map(|block_id| format!("b{block_id}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_region_ids(region_ids: &[usize]) -> String {
    region_ids
        .iter()
        .map(|region_id| format!("r{region_id}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_value_ids(value_ids: &[usize]) -> String {
    value_ids
        .iter()
        .map(|value_id| format!("v{value_id}"))
        .collect::<Vec<_>>()
        .join(",")
}
