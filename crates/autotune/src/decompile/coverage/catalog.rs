use std::collections::BTreeMap;

use super::super::{
    KernelIrModule, KernelIrOpKind, KnownSassOpcode, SassArchitecture, SassLiftedModule,
    SassLiftedOpClass, SassOpcode, SassOpcodeCatalogClass, SassOpcodeKind,
    SassSemanticPatternCategory, known_sass_opcodes,
};

use super::types::{
    OpcodeCatalogBuilder, SassOpcodeCatalogEntry, SassOpcodeCount, SassOpcodeProbeAction,
    SassOpcodeProbeReason, SassOpcodeProbeTarget, SassOpcodeSignature, SassOpcodeSignatureCount,
    SassSemanticPatternCount, Sm120TensorCoreFamily, Sm120TensorCoreSupportEntry,
    Sm120TensorCoreSupportStatus,
};

pub(super) fn seed_known_opcode_catalog(
    opcode_catalog: &mut BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
) {
    for known in known_sass_opcodes() {
        append_known_opcode(opcode_catalog, known);
    }
}

fn append_known_opcode(
    opcode_catalog: &mut BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
    known: &KnownSassOpcode,
) {
    let entry = opcode_catalog
        .entry(SassOpcode::from_kind(known.opcode.clone()))
        .or_default();
    entry.known = true;
    entry.locally_mapped |= known.locally_mapped;
    entry.classes.insert(known.class);
    entry.kinds.insert(known.kind);
    entry.known_sources.insert(known.source);
    for architecture in known.architectures {
        entry.architectures.insert(*architecture);
    }
}

pub(super) fn append_opcode_catalog_lifted_ops(
    lifted: &SassLiftedModule,
    opcode_catalog: &mut BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
) {
    for function in &lifted.functions {
        for op in &function.ops {
            let entry = opcode_catalog.entry(op.opcode.clone()).or_default();
            if op.class != SassLiftedOpClass::Unsupported {
                entry.locally_mapped = true;
            }
            entry.classes.insert(op.class.into());
            entry.kinds.insert(op.kind.into());
        }
    }
}

pub(super) fn append_opcode_catalog_unsupported(
    project_ir: &KernelIrModule,
    opcode_catalog: &mut BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
) {
    for function in &project_ir.functions {
        for op in &function.ops {
            let KernelIrOpKind::Unsupported { opcode, .. } = &op.kind else {
                continue;
            };
            opcode_catalog
                .entry(opcode.clone())
                .or_default()
                .unsupported_count += 1;
        }
    }
}

pub(super) fn opcode_catalog_entries(
    opcode_catalog: BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
) -> Vec<SassOpcodeCatalogEntry> {
    opcode_catalog
        .into_iter()
        .map(|(opcode, entry)| entry.into_entry(opcode))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Sm120TensorCoreRequirement {
    opcode: SassOpcodeKind,
    required_architecture: SassArchitecture,
    family: Sm120TensorCoreFamily,
    requirement: &'static str,
}

const SM120_TENSOR_CORE_REQUIREMENTS: &[Sm120TensorCoreRequirement] = &[
    Sm120TensorCoreRequirement {
        opcode: SassOpcodeKind::Hmma,
        required_architecture: SassArchitecture::sm(120),
        family: Sm120TensorCoreFamily::WarpMma,
        requirement: "half/bfloat/tf32 warp-scope tensor-core MMA",
    },
    Sm120TensorCoreRequirement {
        opcode: SassOpcodeKind::Imma,
        required_architecture: SassArchitecture::sm(120),
        family: Sm120TensorCoreFamily::WarpMma,
        requirement: "integer warp-scope tensor-core MMA",
    },
    Sm120TensorCoreRequirement {
        opcode: SassOpcodeKind::Dmma,
        required_architecture: SassArchitecture::sm(120),
        family: Sm120TensorCoreFamily::WarpMma,
        requirement: "fp64 warp-scope tensor-core MMA",
    },
    Sm120TensorCoreRequirement {
        opcode: SassOpcodeKind::Qmma,
        required_architecture: SassArchitecture::sm_a(120),
        family: Sm120TensorCoreFamily::WarpMmaSm120a,
        requirement: "sm120a f8/f6/f4 warp-scope tensor-core MMA",
    },
    Sm120TensorCoreRequirement {
        opcode: SassOpcodeKind::Omma,
        required_architecture: SassArchitecture::sm_a(120),
        family: Sm120TensorCoreFamily::WarpMmaSm120a,
        requirement: "sm120a block-scaled fp4 warp-scope tensor-core MMA",
    },
];

pub(super) fn sm120_tensor_core_support(
    opcode_catalog: &[SassOpcodeCatalogEntry],
) -> Vec<Sm120TensorCoreSupportEntry> {
    SM120_TENSOR_CORE_REQUIREMENTS
        .iter()
        .map(|requirement| {
            let opcode = SassOpcode::from_kind(requirement.opcode.clone());
            let catalog_entry = opcode_catalog.iter().find(|entry| entry.opcode == opcode);
            let observed_architectures = catalog_entry
                .map(|entry| entry.observed_architectures.clone())
                .unwrap_or_default();
            let observed = observed_architectures.contains(&requirement.required_architecture);
            let locally_mapped = catalog_entry.is_some_and(|entry| entry.locally_mapped);
            let instruction_count = catalog_entry
                .map(|entry| entry.instruction_count)
                .unwrap_or_default();
            let status = if observed && locally_mapped {
                Sm120TensorCoreSupportStatus::Supported
            } else if observed {
                Sm120TensorCoreSupportStatus::MissingLifterMapping
            } else if instruction_count > 0 {
                Sm120TensorCoreSupportStatus::MissingArchitectureArtifact
            } else {
                Sm120TensorCoreSupportStatus::Unobserved
            };

            Sm120TensorCoreSupportEntry {
                opcode,
                required_architecture: requirement.required_architecture,
                family: requirement.family,
                requirement: requirement.requirement,
                observed,
                locally_mapped,
                instruction_count,
                observed_architectures,
                status,
            }
        })
        .collect()
}

pub(super) fn opcode_probe_targets(
    opcode_catalog: &BTreeMap<SassOpcode, OpcodeCatalogBuilder>,
    scanned_architectures: &[SassArchitecture],
) -> Vec<SassOpcodeProbeTarget> {
    let mut targets = opcode_catalog
        .iter()
        .filter(|(_, entry)| entry.known && entry.instruction_count == 0)
        .map(|(opcode, entry)| {
            let matching_scanned_architectures =
                matching_scanned_architectures(entry, scanned_architectures);
            let priority = opcode_probe_priority(entry, &matching_scanned_architectures);
            let recommended_action = opcode_probe_action(entry, &matching_scanned_architectures);
            let reason = opcode_probe_reason(entry, &matching_scanned_architectures);
            SassOpcodeProbeTarget {
                opcode: opcode.clone(),
                priority,
                architectures: entry.architectures.iter().cloned().collect(),
                matching_scanned_architectures,
                classes: entry.classes.iter().copied().collect(),
                kinds: entry.kinds.iter().copied().collect(),
                known_sources: entry.known_sources.iter().copied().collect(),
                locally_mapped: entry.locally_mapped,
                recommended_action,
                reason,
            }
        })
        .collect::<Vec<_>>();
    targets.sort_by(|left, right| {
        right
            .priority
            .cmp(&left.priority)
            .then_with(|| left.opcode.cmp(&right.opcode))
    });
    targets
}

fn matching_scanned_architectures(
    entry: &OpcodeCatalogBuilder,
    scanned_architectures: &[SassArchitecture],
) -> Vec<SassArchitecture> {
    if entry.architectures.is_empty() {
        return Vec::new();
    }
    scanned_architectures
        .iter()
        .filter(|architecture| entry.architectures.contains(architecture))
        .copied()
        .collect()
}

fn opcode_probe_action(
    entry: &OpcodeCatalogBuilder,
    matching_scanned_architectures: &[SassArchitecture],
) -> SassOpcodeProbeAction {
    if !entry.locally_mapped {
        SassOpcodeProbeAction::AddLifterMapping
    } else if !entry.architectures.is_empty() && matching_scanned_architectures.is_empty() {
        SassOpcodeProbeAction::ScanArchitectureArtifact
    } else {
        SassOpcodeProbeAction::GenerateSassArtifact
    }
}

fn opcode_probe_priority(
    entry: &OpcodeCatalogBuilder,
    matching_scanned_architectures: &[SassArchitecture],
) -> u8 {
    let architecture_not_scanned =
        !entry.architectures.is_empty() && matching_scanned_architectures.is_empty();
    if !entry.locally_mapped {
        100
    } else if opcode_has_class(entry, SassOpcodeCatalogClass::TensorCore)
        || opcode_has_class(entry, SassOpcodeCatalogClass::TensorMemory)
        || opcode_has_class(entry, SassOpcodeCatalogClass::WarpGroup)
    {
        if architecture_not_scanned { 80 } else { 90 }
    } else if !entry.architectures.is_empty() {
        if architecture_not_scanned { 60 } else { 70 }
    } else {
        50
    }
}

fn opcode_probe_reason(
    entry: &OpcodeCatalogBuilder,
    matching_scanned_architectures: &[SassArchitecture],
) -> SassOpcodeProbeReason {
    if !entry.locally_mapped {
        return SassOpcodeProbeReason::MissingLocalLifterMapping;
    }
    if !entry.architectures.is_empty() && matching_scanned_architectures.is_empty() {
        return SassOpcodeProbeReason::ArchitectureArtifactNotScanned;
    }
    if opcode_has_class(entry, SassOpcodeCatalogClass::TensorCore) {
        return SassOpcodeProbeReason::TensorCoreMappedUnobserved;
    }
    if opcode_has_class(entry, SassOpcodeCatalogClass::TensorMemory) {
        return SassOpcodeProbeReason::TensorMemoryMappedUnobserved;
    }
    if opcode_has_class(entry, SassOpcodeCatalogClass::WarpGroup) {
        return SassOpcodeProbeReason::WarpGroupMappedUnobserved;
    }
    if !entry.architectures.is_empty() {
        return SassOpcodeProbeReason::ArchitectureSpecificMappedUnobserved;
    }
    SassOpcodeProbeReason::ScalarMappedUnobserved
}

fn opcode_has_class(entry: &OpcodeCatalogBuilder, class: SassOpcodeCatalogClass) -> bool {
    entry.classes.contains(&class)
}

pub(super) fn sorted_opcode_counts(counts: BTreeMap<SassOpcode, usize>) -> Vec<SassOpcodeCount> {
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

pub(super) fn sorted_opcode_signature_counts(
    counts: BTreeMap<SassOpcodeSignature, usize>,
) -> Vec<SassOpcodeSignatureCount> {
    let mut counts = counts
        .into_iter()
        .map(|(signature, count)| SassOpcodeSignatureCount { signature, count })
        .collect::<Vec<_>>();
    counts.sort_by(|lhs, rhs| {
        rhs.count
            .cmp(&lhs.count)
            .then_with(|| lhs.signature.cmp(&rhs.signature))
    });
    counts
}

pub(super) fn sorted_semantic_pattern_counts(
    counts: BTreeMap<SassSemanticPatternCategory, usize>,
) -> Vec<SassSemanticPatternCount> {
    let mut counts = counts
        .into_iter()
        .map(|(category, count)| SassSemanticPatternCount { category, count })
        .collect::<Vec<_>>();
    counts.sort_by(|lhs, rhs| {
        rhs.count
            .cmp(&lhs.count)
            .then_with(|| lhs.category.cmp(&rhs.category))
    });
    counts
}
