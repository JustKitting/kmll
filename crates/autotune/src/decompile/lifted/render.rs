use std::fmt::Write as _;

use super::types::{SassLiftedModule, SassLiftedValueRef};

impl SassLiftedModule {
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        if let Some(target) = &self.target {
            writeln!(out, "target {target}").expect("write to string");
        }
        for function in &self.functions {
            writeln!(out, "fn {} {{", function.name).expect("write to string");
            writeln!(out, "  lifted_value_ir").expect("write to string");
            for op in &function.ops {
                writeln!(
                    out,
                    "    {:#06x}: block={} {} {} in=[{}] out=[{}] predicate={} operands=[{}] <- {}",
                    op.address,
                    format_block_id(op.block_id),
                    op.class,
                    op.kind,
                    format_value_refs(&op.inputs),
                    format_value_refs(&op.outputs),
                    op.predicate.as_deref().unwrap_or("-"),
                    op.source_operands.join(","),
                    op.source
                )
                .expect("write to string");
            }
            writeln!(out, "}}").expect("write to string");
        }
        out
    }
}
fn format_value_refs(values: &[SassLiftedValueRef]) -> String {
    values
        .iter()
        .map(SassLiftedValueRef::name)
        .collect::<Vec<_>>()
        .join(",")
}

fn format_block_id(block_id: Option<usize>) -> String {
    block_id
        .map(|block| format!("b{block}"))
        .unwrap_or_else(|| "-".to_string())
}
