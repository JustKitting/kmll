use std::fmt::Write as _;

use super::types::SassPatternModule;

impl SassPatternModule {
    pub fn pattern_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.patterns.len())
            .sum()
    }

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        if let Some(target) = &self.target {
            writeln!(out, "target {target}").expect("write to string");
        }
        for function in &self.functions {
            writeln!(out, "fn {} {{", function.name).expect("write to string");
            for pattern in &function.patterns {
                writeln!(
                    out,
                    "  {:#06x}-{:#06x}: {} {:?} [{}] <- {}",
                    pattern.start_address,
                    pattern.end_address,
                    pattern.kind_name(),
                    pattern.kind,
                    pattern.confidence,
                    format_addresses(&pattern.source_addresses)
                )
                .expect("write to string");
            }
            writeln!(out, "}}").expect("write to string");
        }
        out
    }
}

fn format_addresses(addresses: &[u64]) -> String {
    let body = addresses
        .iter()
        .map(|address| format!("{address:#06x}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}
