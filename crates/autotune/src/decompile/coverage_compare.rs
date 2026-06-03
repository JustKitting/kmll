use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt::{self, Write as _},
    fs,
    path::PathBuf,
};

use super::{
    SassCoverageOptions, SassCoverageReport, SassOpcode, SassOpcodeCatalogClass,
    SassOpcodeCatalogEntry, SassOpcodeCatalogKind, SassOpcodeCoverageState, run_sass_coverage_scan,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageComparisonOptions {
    pub baseline_root: PathBuf,
    pub candidate_root: PathBuf,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageComparisonReport {
    pub baseline_report: SassCoverageReport,
    pub candidate_report: SassCoverageReport,
    pub output_dir: PathBuf,
    pub summary_path: PathBuf,
    pub opcode_delta_path: PathBuf,
    pub resolved_probe_targets_path: PathBuf,
    pub new_probe_targets_path: PathBuf,
    pub opcode_deltas: Vec<SassCoverageOpcodeDelta>,
    pub resolved_probe_targets: Vec<SassCoverageProbeTargetDelta>,
    pub new_probe_targets: Vec<SassCoverageProbeTargetDelta>,
    pub newly_observed_opcode_count: usize,
    pub no_longer_observed_opcode_count: usize,
    pub coverage_changed_opcode_count: usize,
    pub count_changed_opcode_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageOpcodeDelta {
    pub opcode: SassOpcode,
    pub change: SassCoverageOpcodeChange,
    pub baseline_known: bool,
    pub candidate_known: bool,
    pub baseline_observed: bool,
    pub candidate_observed: bool,
    pub baseline_locally_mapped: bool,
    pub candidate_locally_mapped: bool,
    pub baseline_instruction_count: usize,
    pub candidate_instruction_count: usize,
    pub baseline_coverage: SassOpcodeCoverageState,
    pub candidate_coverage: SassOpcodeCoverageState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassCoverageProbeTargetDelta {
    pub opcode: SassOpcode,
    pub baseline_coverage: SassOpcodeCoverageState,
    pub candidate_coverage: SassOpcodeCoverageState,
    pub candidate_instruction_count: usize,
    pub classes: Vec<SassOpcodeCatalogClass>,
    pub kinds: Vec<SassOpcodeCatalogKind>,
    pub architectures: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SassCoverageOpcodeChange {
    NewlyObserved,
    NoLongerObserved,
    CoverageChanged,
    CountChanged,
}

impl fmt::Display for SassCoverageOpcodeChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NewlyObserved => f.write_str("newly-observed"),
            Self::NoLongerObserved => f.write_str("no-longer-observed"),
            Self::CoverageChanged => f.write_str("coverage-changed"),
            Self::CountChanged => f.write_str("count-changed"),
        }
    }
}

pub fn run_sass_coverage_comparison(
    options: &SassCoverageComparisonOptions,
) -> Result<SassCoverageComparisonReport, Box<dyn Error>> {
    fs::create_dir_all(&options.output_dir)?;
    let baseline_report = run_sass_coverage_scan(&SassCoverageOptions {
        root: options.baseline_root.clone(),
        output_dir: options.output_dir.join("baseline"),
    })?;
    let candidate_report = run_sass_coverage_scan(&SassCoverageOptions {
        root: options.candidate_root.clone(),
        output_dir: options.output_dir.join("candidate"),
    })?;
    let summary_path = options.output_dir.join("summary.txt");
    let opcode_delta_path = options.output_dir.join("opcode-delta.tsv");
    let resolved_probe_targets_path = options.output_dir.join("resolved-probe-targets.tsv");
    let new_probe_targets_path = options.output_dir.join("new-probe-targets.tsv");
    let opcode_deltas = opcode_deltas(&baseline_report, &candidate_report);
    let resolved_probe_targets = resolved_probe_targets(&candidate_report, &opcode_deltas);
    let new_probe_targets = new_probe_targets(&candidate_report, &opcode_deltas);
    let newly_observed_opcode_count = opcode_deltas
        .iter()
        .filter(|delta| delta.change == SassCoverageOpcodeChange::NewlyObserved)
        .count();
    let no_longer_observed_opcode_count = opcode_deltas
        .iter()
        .filter(|delta| delta.change == SassCoverageOpcodeChange::NoLongerObserved)
        .count();
    let coverage_changed_opcode_count = opcode_deltas
        .iter()
        .filter(|delta| delta.change == SassCoverageOpcodeChange::CoverageChanged)
        .count();
    let count_changed_opcode_count = opcode_deltas
        .iter()
        .filter(|delta| delta.change == SassCoverageOpcodeChange::CountChanged)
        .count();

    let report = SassCoverageComparisonReport {
        baseline_report,
        candidate_report,
        output_dir: options.output_dir.clone(),
        summary_path,
        opcode_delta_path,
        resolved_probe_targets_path,
        new_probe_targets_path,
        opcode_deltas,
        resolved_probe_targets,
        new_probe_targets,
        newly_observed_opcode_count,
        no_longer_observed_opcode_count,
        coverage_changed_opcode_count,
        count_changed_opcode_count,
    };
    write_comparison_reports(&report)?;
    Ok(report)
}

fn opcode_deltas(
    baseline: &SassCoverageReport,
    candidate: &SassCoverageReport,
) -> Vec<SassCoverageOpcodeDelta> {
    let baseline_catalog = catalog_by_opcode(baseline);
    let candidate_catalog = catalog_by_opcode(candidate);
    let mut opcodes = BTreeSet::<SassOpcode>::new();
    opcodes.extend(baseline_catalog.keys().cloned());
    opcodes.extend(candidate_catalog.keys().cloned());

    opcodes
        .into_iter()
        .filter_map(|opcode| {
            let baseline = baseline_catalog.get(&opcode);
            let candidate = candidate_catalog.get(&opcode);
            let change = opcode_change(baseline, candidate)?;
            Some(SassCoverageOpcodeDelta {
                opcode,
                change,
                baseline_known: catalog_known(baseline),
                candidate_known: catalog_known(candidate),
                baseline_observed: catalog_observed(baseline),
                candidate_observed: catalog_observed(candidate),
                baseline_locally_mapped: catalog_locally_mapped(baseline),
                candidate_locally_mapped: catalog_locally_mapped(candidate),
                baseline_instruction_count: catalog_instruction_count(baseline),
                candidate_instruction_count: catalog_instruction_count(candidate),
                baseline_coverage: catalog_coverage(baseline),
                candidate_coverage: catalog_coverage(candidate),
            })
        })
        .collect()
}

fn resolved_probe_targets(
    candidate: &SassCoverageReport,
    deltas: &[SassCoverageOpcodeDelta],
) -> Vec<SassCoverageProbeTargetDelta> {
    let candidate_catalog = catalog_by_opcode(candidate);
    deltas
        .iter()
        .filter(|delta| {
            delta.baseline_known && !delta.baseline_observed && delta.candidate_observed
        })
        .map(|delta| probe_target_delta(delta, candidate_catalog.get(&delta.opcode)))
        .collect()
}

fn new_probe_targets(
    candidate: &SassCoverageReport,
    deltas: &[SassCoverageOpcodeDelta],
) -> Vec<SassCoverageProbeTargetDelta> {
    let candidate_catalog = catalog_by_opcode(candidate);
    deltas
        .iter()
        .filter(|delta| {
            delta.candidate_known && !delta.candidate_observed && delta.baseline_observed
        })
        .map(|delta| probe_target_delta(delta, candidate_catalog.get(&delta.opcode)))
        .collect()
}

fn probe_target_delta(
    delta: &SassCoverageOpcodeDelta,
    candidate: Option<&&SassOpcodeCatalogEntry>,
) -> SassCoverageProbeTargetDelta {
    SassCoverageProbeTargetDelta {
        opcode: delta.opcode.clone(),
        baseline_coverage: delta.baseline_coverage.clone(),
        candidate_coverage: delta.candidate_coverage.clone(),
        candidate_instruction_count: delta.candidate_instruction_count,
        classes: candidate
            .map(|entry| entry.classes.clone())
            .unwrap_or_default(),
        kinds: candidate
            .map(|entry| entry.kinds.clone())
            .unwrap_or_default(),
        architectures: candidate
            .map(|entry| entry.architectures.clone())
            .unwrap_or_default(),
    }
}

fn catalog_by_opcode(report: &SassCoverageReport) -> BTreeMap<SassOpcode, &SassOpcodeCatalogEntry> {
    report
        .opcode_catalog
        .iter()
        .map(|entry| (entry.opcode.clone(), entry))
        .collect()
}

fn opcode_change(
    baseline: Option<&&SassOpcodeCatalogEntry>,
    candidate: Option<&&SassOpcodeCatalogEntry>,
) -> Option<SassCoverageOpcodeChange> {
    let baseline_observed = catalog_observed(baseline);
    let candidate_observed = catalog_observed(candidate);
    if !baseline_observed && candidate_observed {
        return Some(SassCoverageOpcodeChange::NewlyObserved);
    }
    if baseline_observed && !candidate_observed {
        return Some(SassCoverageOpcodeChange::NoLongerObserved);
    }
    if catalog_coverage(baseline) != catalog_coverage(candidate) {
        return Some(SassCoverageOpcodeChange::CoverageChanged);
    }
    if catalog_instruction_count(baseline) != catalog_instruction_count(candidate) {
        return Some(SassCoverageOpcodeChange::CountChanged);
    }
    None
}

fn catalog_known(entry: Option<&&SassOpcodeCatalogEntry>) -> bool {
    entry.is_some_and(|entry| entry.known)
}

fn catalog_observed(entry: Option<&&SassOpcodeCatalogEntry>) -> bool {
    entry.is_some_and(|entry| entry.observed)
}

fn catalog_locally_mapped(entry: Option<&&SassOpcodeCatalogEntry>) -> bool {
    entry.is_some_and(|entry| entry.locally_mapped)
}

fn catalog_instruction_count(entry: Option<&&SassOpcodeCatalogEntry>) -> usize {
    entry.map(|entry| entry.instruction_count).unwrap_or(0)
}

fn catalog_coverage(entry: Option<&&SassOpcodeCatalogEntry>) -> SassOpcodeCoverageState {
    entry
        .map(|entry| entry.coverage)
        .unwrap_or(SassOpcodeCoverageState::Absent)
}

fn write_comparison_reports(report: &SassCoverageComparisonReport) -> Result<(), Box<dyn Error>> {
    fs::write(
        &report.summary_path,
        render_comparison_summary(report).as_bytes(),
    )?;
    fs::write(
        &report.opcode_delta_path,
        render_opcode_delta_tsv(report).as_bytes(),
    )?;
    fs::write(
        &report.resolved_probe_targets_path,
        render_probe_target_delta_tsv(&report.resolved_probe_targets).as_bytes(),
    )?;
    fs::write(
        &report.new_probe_targets_path,
        render_probe_target_delta_tsv(&report.new_probe_targets).as_bytes(),
    )?;
    Ok(())
}

fn render_comparison_summary(report: &SassCoverageComparisonReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "baseline_root={}",
        report.baseline_report.root.display()
    )
    .expect("write to string");
    writeln!(
        out,
        "candidate_root={}",
        report.candidate_report.root.display()
    )
    .expect("write to string");
    writeln!(
        out,
        "baseline_files_seen={}",
        report.baseline_report.files.len()
    )
    .expect("write to string");
    writeln!(
        out,
        "candidate_files_seen={}",
        report.candidate_report.files.len()
    )
    .expect("write to string");
    writeln!(
        out,
        "baseline_probe_targets={}",
        report.baseline_report.opcode_probe_target_count
    )
    .expect("write to string");
    writeln!(
        out,
        "candidate_probe_targets={}",
        report.candidate_report.opcode_probe_target_count
    )
    .expect("write to string");
    writeln!(
        out,
        "newly_observed_opcodes={}",
        report.newly_observed_opcode_count
    )
    .expect("write to string");
    writeln!(
        out,
        "no_longer_observed_opcodes={}",
        report.no_longer_observed_opcode_count
    )
    .expect("write to string");
    writeln!(
        out,
        "coverage_changed_opcodes={}",
        report.coverage_changed_opcode_count
    )
    .expect("write to string");
    writeln!(
        out,
        "count_changed_opcodes={}",
        report.count_changed_opcode_count
    )
    .expect("write to string");
    writeln!(
        out,
        "resolved_probe_targets={}",
        report.resolved_probe_targets.len()
    )
    .expect("write to string");
    writeln!(out, "new_probe_targets={}", report.new_probe_targets.len()).expect("write to string");
    writeln!(out, "opcode_delta={}", report.opcode_delta_path.display()).expect("write to string");
    writeln!(
        out,
        "resolved_probe_targets_path={}",
        report.resolved_probe_targets_path.display()
    )
    .expect("write to string");
    writeln!(
        out,
        "new_probe_targets_path={}",
        report.new_probe_targets_path.display()
    )
    .expect("write to string");
    out
}

fn render_opcode_delta_tsv(report: &SassCoverageComparisonReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "opcode\tchange\tbaseline_known\tcandidate_known\tbaseline_observed\tcandidate_observed\tbaseline_locally_mapped\tcandidate_locally_mapped\tbaseline_instruction_count\tcandidate_instruction_count\tbaseline_coverage\tcandidate_coverage"
    )
    .expect("write to string");
    for delta in &report.opcode_deltas {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&delta.opcode.to_string()),
            tsv(&delta.change.to_string()),
            delta.baseline_known,
            delta.candidate_known,
            delta.baseline_observed,
            delta.candidate_observed,
            delta.baseline_locally_mapped,
            delta.candidate_locally_mapped,
            delta.baseline_instruction_count,
            delta.candidate_instruction_count,
            tsv(&delta.baseline_coverage.to_string()),
            tsv(&delta.candidate_coverage.to_string()),
        )
        .expect("write to string");
    }
    out
}

fn render_probe_target_delta_tsv(targets: &[SassCoverageProbeTargetDelta]) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "opcode\tbaseline_coverage\tcandidate_coverage\tcandidate_instruction_count\tclasses\tkinds\tarchitectures"
    )
    .expect("write to string");
    for target in targets {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv(&target.opcode.to_string()),
            tsv(&target.baseline_coverage.to_string()),
            tsv(&target.candidate_coverage.to_string()),
            target.candidate_instruction_count,
            tsv(&display_list(&target.classes)),
            tsv(&display_list(&target.kinds)),
            tsv(&target.architectures.join(",")),
        )
        .expect("write to string");
    }
    out
}

fn display_list<T: fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn tsv(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}
