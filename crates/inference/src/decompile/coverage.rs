use std::{
    collections::BTreeMap,
    error::Error,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use crate::runtime;

use super::{
    KernelIrModule, KernelIrOpKind, SassAnalysisModule, SassPatternModule, analyze_sass_ir,
    parse_nvdisasm_sass, recover_sass_patterns, render_sass_file_side_by_side,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageOptions {
    pub root: PathBuf,
    pub output_dir: PathBuf,
}

impl SassCoverageOptions {
    pub fn default_artifact_scan() -> Self {
        let artifact_root = runtime::default_artifact_dir();
        Self {
            root: artifact_root.clone(),
            output_dir: artifact_root.join("decompile-coverage"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageReport {
    pub root: PathBuf,
    pub output_dir: PathBuf,
    pub summary_path: PathBuf,
    pub files_path: PathBuf,
    pub opcode_frequency_path: PathBuf,
    pub opcode_signature_frequency_path: PathBuf,
    pub semantic_patterns_path: PathBuf,
    pub semantic_pattern_frequency_path: PathBuf,
    pub cfg_blocks_path: PathBuf,
    pub cfg_edges_path: PathBuf,
    pub dataflow_path: PathBuf,
    pub unsupported_instructions_path: PathBuf,
    pub files: Vec<SassCoverageFileReport>,
    pub opcode_counts: Vec<SassOpcodeCount>,
    pub opcode_signature_counts: Vec<SassOpcodeCount>,
    pub semantic_pattern_counts: Vec<SassOpcodeCount>,
    pub semantic_patterns: Vec<SassCoverageSemanticPattern>,
    pub cfg_blocks: Vec<SassCoverageBasicBlock>,
    pub cfg_edges: Vec<SassCoverageCfgEdge>,
    pub dataflow: Vec<SassCoverageDataflowOp>,
    pub unsupported_instructions: Vec<SassUnsupportedInstruction>,
    pub parsed_file_count: usize,
    pub parse_error_count: usize,
    pub parsed_instruction_count: usize,
    pub cfg_block_count: usize,
    pub cfg_edge_count: usize,
    pub dataflow_op_count: usize,
    pub semantic_pattern_count: usize,
    pub unsupported_instruction_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageFileReport {
    pub sass_path: PathBuf,
    pub ir_path: Option<PathBuf>,
    pub analysis_path: Option<PathBuf>,
    pub pattern_path: Option<PathBuf>,
    pub side_by_side_path: Option<PathBuf>,
    pub parsed_instruction_count: usize,
    pub cfg_block_count: usize,
    pub cfg_edge_count: usize,
    pub semantic_pattern_count: usize,
    pub unsupported_instruction_count: usize,
    pub parse_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassOpcodeCount {
    pub opcode: String,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassUnsupportedInstruction {
    pub sass_path: PathBuf,
    pub function: String,
    pub address: u64,
    pub opcode: String,
    pub reason: String,
    pub raw: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageSemanticPattern {
    pub sass_path: PathBuf,
    pub function: String,
    pub start_address: u64,
    pub end_address: u64,
    pub kind: String,
    pub confidence: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageBasicBlock {
    pub sass_path: PathBuf,
    pub function: String,
    pub id: usize,
    pub label: Option<String>,
    pub start_address: u64,
    pub end_address: u64,
    pub instruction_count: usize,
    pub terminator: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageCfgEdge {
    pub sass_path: PathBuf,
    pub function: String,
    pub from_block: usize,
    pub to_block: Option<usize>,
    pub kind: String,
    pub condition: Option<String>,
    pub target: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageDataflowOp {
    pub sass_path: PathBuf,
    pub function: String,
    pub address: u64,
    pub defines: Vec<String>,
    pub uses: Vec<String>,
    pub source: String,
}

pub fn run_sass_coverage_scan(
    options: &SassCoverageOptions,
) -> Result<SassCoverageReport, Box<dyn Error>> {
    let mut sass_paths = Vec::new();
    collect_sass_paths(&options.root, &mut sass_paths)?;
    sass_paths.sort();

    fs::create_dir_all(&options.output_dir)?;
    let mut files = Vec::new();
    let mut opcode_counts = BTreeMap::<String, usize>::new();
    let mut opcode_signature_counts = BTreeMap::<String, usize>::new();
    let mut semantic_pattern_counts = BTreeMap::<String, usize>::new();
    let mut semantic_patterns = Vec::new();
    let mut cfg_blocks = Vec::new();
    let mut cfg_edges = Vec::new();
    let mut dataflow = Vec::new();
    let mut unsupported_instructions = Vec::new();
    let files_output_root = options.output_dir.join("files");

    for sass_path in sass_paths {
        let sass = fs::read_to_string(&sass_path)?;
        let relative = relative_sass_path(&options.root, &sass_path);
        match parse_nvdisasm_sass(&sass) {
            Ok(parsed) => {
                for function in &parsed.functions {
                    for instruction in &function.instructions {
                        *opcode_counts.entry(instruction.opcode.clone()).or_default() += 1;
                        *opcode_signature_counts
                            .entry(opcode_signature(
                                &instruction.opcode,
                                &instruction.modifiers,
                            ))
                            .or_default() += 1;
                    }
                }

                let lowered = super::lower_sass_module(&parsed);
                let analysis = analyze_sass_ir(&lowered);
                let patterns = recover_sass_patterns(&lowered);
                for function in &patterns.functions {
                    for pattern in &function.patterns {
                        *semantic_pattern_counts
                            .entry(pattern.kind_name().to_string())
                            .or_default() += 1;
                    }
                }
                append_analysis(
                    &sass_path,
                    &analysis,
                    &mut cfg_blocks,
                    &mut cfg_edges,
                    &mut dataflow,
                );
                append_semantic_patterns(&sass_path, &patterns, &mut semantic_patterns);
                append_unsupported(&sass_path, &lowered, &mut unsupported_instructions);

                let ir_path = files_output_root
                    .join(&relative)
                    .with_extension("lifted.ir.txt");
                if let Some(parent) = ir_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&ir_path, lowered.to_text().as_bytes())?;

                let analysis_path = files_output_root
                    .join(&relative)
                    .with_extension("analysis.txt");
                fs::write(&analysis_path, analysis.to_text().as_bytes())?;

                let pattern_path = files_output_root
                    .join(&relative)
                    .with_extension("patterns.txt");
                fs::write(&pattern_path, patterns.to_text().as_bytes())?;

                let side_by_side_path = files_output_root
                    .join(&relative)
                    .with_extension("source-sass-ir.txt");
                let side_by_side = render_sass_file_side_by_side(&sass_path, None, &sass, &lowered);
                fs::write(&side_by_side_path, side_by_side.as_bytes())?;

                files.push(SassCoverageFileReport {
                    sass_path,
                    ir_path: Some(ir_path),
                    analysis_path: Some(analysis_path),
                    pattern_path: Some(pattern_path),
                    side_by_side_path: Some(side_by_side_path),
                    parsed_instruction_count: parsed.instruction_count(),
                    cfg_block_count: analysis.block_count(),
                    cfg_edge_count: analysis.edge_count(),
                    semantic_pattern_count: patterns.pattern_count(),
                    unsupported_instruction_count: lowered.unsupported_instruction_count(),
                    parse_error: None,
                });
            }
            Err(error) => {
                files.push(SassCoverageFileReport {
                    sass_path,
                    ir_path: None,
                    analysis_path: None,
                    pattern_path: None,
                    side_by_side_path: None,
                    parsed_instruction_count: 0,
                    cfg_block_count: 0,
                    cfg_edge_count: 0,
                    semantic_pattern_count: 0,
                    unsupported_instruction_count: 0,
                    parse_error: Some(error.to_string()),
                });
            }
        }
    }

    let opcode_counts = sorted_counts(opcode_counts);
    let opcode_signature_counts = sorted_counts(opcode_signature_counts);
    let semantic_pattern_counts = sorted_counts(semantic_pattern_counts);
    let parsed_file_count = files
        .iter()
        .filter(|file| file.parse_error.is_none())
        .count();
    let parse_error_count = files.len().saturating_sub(parsed_file_count);
    let parsed_instruction_count = files.iter().map(|file| file.parsed_instruction_count).sum();
    let cfg_block_count = cfg_blocks.len();
    let cfg_edge_count = cfg_edges.len();
    let dataflow_op_count = dataflow.len();
    let semantic_pattern_count = semantic_patterns.len();
    let unsupported_instruction_count = unsupported_instructions.len();

    let summary_path = options.output_dir.join("summary.txt");
    let files_path = options.output_dir.join("files.tsv");
    let opcode_frequency_path = options.output_dir.join("opcode-frequency.tsv");
    let opcode_signature_frequency_path = options.output_dir.join("opcode-signature-frequency.tsv");
    let semantic_patterns_path = options.output_dir.join("semantic-patterns.tsv");
    let semantic_pattern_frequency_path = options.output_dir.join("semantic-pattern-frequency.tsv");
    let cfg_blocks_path = options.output_dir.join("cfg-blocks.tsv");
    let cfg_edges_path = options.output_dir.join("cfg-edges.tsv");
    let dataflow_path = options.output_dir.join("dataflow.tsv");
    let unsupported_instructions_path = options.output_dir.join("unsupported-instructions.tsv");

    let report = SassCoverageReport {
        root: options.root.clone(),
        output_dir: options.output_dir.clone(),
        summary_path,
        files_path,
        opcode_frequency_path,
        opcode_signature_frequency_path,
        semantic_patterns_path,
        semantic_pattern_frequency_path,
        cfg_blocks_path,
        cfg_edges_path,
        dataflow_path,
        unsupported_instructions_path,
        files,
        opcode_counts,
        opcode_signature_counts,
        semantic_pattern_counts,
        semantic_patterns,
        cfg_blocks,
        cfg_edges,
        dataflow,
        unsupported_instructions,
        parsed_file_count,
        parse_error_count,
        parsed_instruction_count,
        cfg_block_count,
        cfg_edge_count,
        dataflow_op_count,
        semantic_pattern_count,
        unsupported_instruction_count,
    };

    write_coverage_reports(&report)?;
    Ok(report)
}

fn collect_sass_paths(root: &Path, out: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    if root.is_file() {
        if is_nvdisasm_sass(root) {
            out.push(root.to_path_buf());
        }
        return Ok(());
    }

    let mut entries = fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_sass_paths(&path, out)?;
        } else if is_nvdisasm_sass(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_nvdisasm_sass(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".nvdisasm.sass"))
}

fn relative_sass_path(root: &Path, sass_path: &Path) -> PathBuf {
    if root.is_file() {
        return sass_path
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("sass.nvdisasm.sass"));
    }
    sass_path
        .strip_prefix(root)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| {
            sass_path
                .file_name()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("sass.nvdisasm.sass"))
        })
}

fn opcode_signature(opcode: &str, modifiers: &[String]) -> String {
    if modifiers.is_empty() {
        opcode.to_string()
    } else {
        format!("{}.{}", opcode, modifiers.join("."))
    }
}

fn append_unsupported(
    sass_path: &Path,
    lowered: &KernelIrModule,
    unsupported_instructions: &mut Vec<SassUnsupportedInstruction>,
) {
    for function in &lowered.functions {
        for op in &function.ops {
            let KernelIrOpKind::Unsupported { opcode, reason } = &op.kind else {
                continue;
            };
            unsupported_instructions.push(SassUnsupportedInstruction {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                address: op.address,
                opcode: opcode.clone(),
                reason: reason.clone(),
                raw: op.source.clone(),
            });
        }
    }
}

fn append_semantic_patterns(
    sass_path: &Path,
    patterns: &SassPatternModule,
    semantic_patterns: &mut Vec<SassCoverageSemanticPattern>,
) {
    for function in &patterns.functions {
        for pattern in &function.patterns {
            semantic_patterns.push(SassCoverageSemanticPattern {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                start_address: pattern.start_address,
                end_address: pattern.end_address,
                kind: pattern.kind_name().to_string(),
                confidence: pattern.confidence.to_string(),
                detail: format!("{:?}", pattern.kind),
            });
        }
    }
}

fn append_analysis(
    sass_path: &Path,
    analysis: &SassAnalysisModule,
    cfg_blocks: &mut Vec<SassCoverageBasicBlock>,
    cfg_edges: &mut Vec<SassCoverageCfgEdge>,
    dataflow: &mut Vec<SassCoverageDataflowOp>,
) {
    for function in &analysis.functions {
        for block in &function.blocks {
            cfg_blocks.push(SassCoverageBasicBlock {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                id: block.id,
                label: block.label.clone(),
                start_address: block.start_address,
                end_address: block.end_address,
                instruction_count: block.instruction_count,
                terminator: block.terminator.to_string(),
            });
        }
        for edge in &function.edges {
            cfg_edges.push(SassCoverageCfgEdge {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                from_block: edge.from_block,
                to_block: edge.to_block,
                kind: edge.kind.to_string(),
                condition: edge.condition.clone(),
                target: edge.target.clone(),
            });
        }
        for op in &function.dataflow {
            dataflow.push(SassCoverageDataflowOp {
                sass_path: sass_path.to_path_buf(),
                function: function.name.clone(),
                address: op.address,
                defines: op.defines.clone(),
                uses: op.uses.clone(),
                source: op.source.clone(),
            });
        }
    }
}

fn sorted_counts(counts: BTreeMap<String, usize>) -> Vec<SassOpcodeCount> {
    let mut counts = counts
        .into_iter()
        .map(|(opcode, count)| SassOpcodeCount { opcode, count })
        .collect::<Vec<_>>();
    counts.sort_by(|lhs, rhs| {
        rhs.count
            .cmp(&lhs.count)
            .then_with(|| lhs.opcode.cmp(&rhs.opcode))
    });
    counts
}

fn write_coverage_reports(report: &SassCoverageReport) -> Result<(), Box<dyn Error>> {
    fs::write(
        &report.summary_path,
        render_coverage_summary(report).as_bytes(),
    )?;
    fs::write(&report.files_path, render_files_tsv(report).as_bytes())?;
    fs::write(
        &report.opcode_frequency_path,
        render_counts_tsv("opcode", &report.opcode_counts).as_bytes(),
    )?;
    fs::write(
        &report.opcode_signature_frequency_path,
        render_counts_tsv("opcode_signature", &report.opcode_signature_counts).as_bytes(),
    )?;
    fs::write(
        &report.semantic_patterns_path,
        render_semantic_patterns_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.semantic_pattern_frequency_path,
        render_counts_tsv("semantic_pattern", &report.semantic_pattern_counts).as_bytes(),
    )?;
    fs::write(
        &report.cfg_blocks_path,
        render_cfg_blocks_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.cfg_edges_path,
        render_cfg_edges_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.dataflow_path,
        render_dataflow_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.unsupported_instructions_path,
        render_unsupported_tsv(report).as_bytes(),
    )?;
    Ok(())
}

fn render_coverage_summary(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(out, "root={}", report.root.display()).expect("write to string");
    writeln!(out, "output_dir={}", report.output_dir.display()).expect("write to string");
    writeln!(out, "files_seen={}", report.files.len()).expect("write to string");
    writeln!(out, "files_parsed={}", report.parsed_file_count).expect("write to string");
    writeln!(out, "parse_errors={}", report.parse_error_count).expect("write to string");
    writeln!(
        out,
        "parsed_instructions={}",
        report.parsed_instruction_count
    )
    .expect("write to string");
    writeln!(out, "cfg_blocks={}", report.cfg_block_count).expect("write to string");
    writeln!(out, "cfg_edges={}", report.cfg_edge_count).expect("write to string");
    writeln!(out, "dataflow_ops={}", report.dataflow_op_count).expect("write to string");
    writeln!(out, "semantic_patterns={}", report.semantic_pattern_count).expect("write to string");
    writeln!(
        out,
        "unsupported_instructions={}",
        report.unsupported_instruction_count
    )
    .expect("write to string");
    writeln!(out, "unique_opcodes={}", report.opcode_counts.len()).expect("write to string");
    writeln!(
        out,
        "unique_opcode_signatures={}",
        report.opcode_signature_counts.len()
    )
    .expect("write to string");
    writeln!(out).expect("write to string");
    writeln!(out, "top_opcodes").expect("write to string");
    for count in report.opcode_counts.iter().take(32) {
        writeln!(out, "{}\t{}", count.opcode, count.count).expect("write to string");
    }
    if !report.semantic_pattern_counts.is_empty() {
        writeln!(out).expect("write to string");
        writeln!(out, "top_semantic_patterns").expect("write to string");
        for count in report.semantic_pattern_counts.iter().take(32) {
            writeln!(out, "{}\t{}", count.opcode, count.count).expect("write to string");
        }
    }
    if !report.unsupported_instructions.is_empty() {
        writeln!(out).expect("write to string");
        writeln!(out, "unsupported_by_opcode").expect("write to string");
        let mut by_opcode = BTreeMap::<String, usize>::new();
        for instruction in &report.unsupported_instructions {
            *by_opcode.entry(instruction.opcode.clone()).or_default() += 1;
        }
        for count in sorted_counts(by_opcode) {
            writeln!(out, "{}\t{}", count.opcode, count.count).expect("write to string");
        }
    }
    out
}

fn render_files_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "status\tsass_path\tparsed_instructions\tcfg_blocks\tcfg_edges\tsemantic_patterns\tunsupported_instructions\tir_path\tanalysis_path\tpatterns_path\tside_by_side_path\terror"
    )
    .expect("write to string");
    for file in &report.files {
        let status = if file.parse_error.is_some() {
            "parse-error"
        } else {
            "parsed"
        };
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            status,
            tsv(&file.sass_path.display().to_string()),
            file.parsed_instruction_count,
            file.cfg_block_count,
            file.cfg_edge_count,
            file.semantic_pattern_count,
            file.unsupported_instruction_count,
            tsv(&optional_path(&file.ir_path)),
            tsv(&optional_path(&file.analysis_path)),
            tsv(&optional_path(&file.pattern_path)),
            tsv(&optional_path(&file.side_by_side_path)),
            tsv(file.parse_error.as_deref().unwrap_or(""))
        )
        .expect("write to string");
    }
    out
}

fn render_semantic_patterns_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tstart_address\tend_address\tkind\tconfidence\tdetail"
    )
    .expect("write to string");
    for pattern in &report.semantic_patterns {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{:#06x}\t{}\t{}\t{}",
            tsv(&pattern.sass_path.display().to_string()),
            tsv(&pattern.function),
            pattern.start_address,
            pattern.end_address,
            tsv(&pattern.kind),
            tsv(&pattern.confidence),
            tsv(&pattern.detail),
        )
        .expect("write to string");
    }
    out
}

fn render_cfg_blocks_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tblock_id\tlabel\tstart_address\tend_address\tinstruction_count\tterminator"
    )
    .expect("write to string");
    for block in &report.cfg_blocks {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{:#06x}\t{:#06x}\t{}\t{}",
            tsv(&block.sass_path.display().to_string()),
            tsv(&block.function),
            block.id,
            tsv(block.label.as_deref().unwrap_or("")),
            block.start_address,
            block.end_address,
            block.instruction_count,
            tsv(&block.terminator),
        )
        .expect("write to string");
    }
    out
}

fn render_cfg_edges_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "sass_path\tfunction\tfrom_block\tto_block\tkind\tcondition\ttarget"
    )
    .expect("write to string");
    for edge in &report.cfg_edges {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&edge.sass_path.display().to_string()),
            tsv(&edge.function),
            edge.from_block,
            edge.to_block
                .map(|block| block.to_string())
                .unwrap_or_default(),
            tsv(&edge.kind),
            tsv(edge.condition.as_deref().unwrap_or("")),
            tsv(edge.target.as_deref().unwrap_or("")),
        )
        .expect("write to string");
    }
    out
}

fn render_dataflow_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(out, "sass_path\tfunction\taddress\tdefines\tuses\traw").expect("write to string");
    for op in &report.dataflow {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}",
            tsv(&op.sass_path.display().to_string()),
            tsv(&op.function),
            op.address,
            tsv(&op.defines.join(",")),
            tsv(&op.uses.join(",")),
            tsv(&op.source),
        )
        .expect("write to string");
    }
    out
}

fn render_counts_tsv(header: &str, counts: &[SassOpcodeCount]) -> String {
    let mut out = String::new();
    writeln!(out, "{header}\tcount").expect("write to string");
    for count in counts {
        writeln!(out, "{}\t{}", tsv(&count.opcode), count.count).expect("write to string");
    }
    out
}

fn render_unsupported_tsv(report: &SassCoverageReport) -> String {
    let mut out = String::new();
    writeln!(out, "sass_path\tfunction\taddress\topcode\treason\traw").expect("write to string");
    for instruction in &report.unsupported_instructions {
        writeln!(
            out,
            "{}\t{}\t{:#06x}\t{}\t{}\t{}",
            tsv(&instruction.sass_path.display().to_string()),
            tsv(&instruction.function),
            instruction.address,
            tsv(&instruction.opcode),
            tsv(&instruction.reason),
            tsv(&instruction.raw)
        )
        .expect("write to string");
    }
    out
}

fn optional_path(path: &Option<PathBuf>) -> String {
    path.as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_default()
}

fn tsv(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
}
