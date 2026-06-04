mod analysis;
mod architecture;
mod autotune_bridge;
mod coverage;
mod coverage_compare;
mod driver;
mod driver_support;
mod fixtures;
mod ir;
mod known_opcodes;
mod lifted;
mod patterns;
mod ptx_probes;
mod sass;

pub use self::{
    analysis::{
        SassAnalysisFunction, SassAnalysisModule, SassBasicBlock, SassBlockTerminator, SassCfgEdge,
        SassCfgEdgeKind, SassDataflowOp, SassDataflowSite, SassDefUseEdge, SassDominatorBlock,
        SassLiveRange, SassMemoryAccess, SassMemoryAccessKind, SassNaturalLoop, SassReachingUse,
        SassRegion, SassRegionKind, SassRegionPath, SassSsaValue, SassValueOp, SassValueOpKind,
        analyze_sass_ir,
    },
    architecture::{SassArchitecture, SassTarget},
    autotune_bridge::{
        DecompiledAutotuneError, DecompiledAutotuneEvidence, DecompiledAutotuneOperation,
        DecompiledAutotuneShape, decompiled_autotune_operation,
        decompiled_autotune_operation_with_module,
    },
    coverage::{
        SassCoverageBasicBlock, SassCoverageCfgEdge, SassCoverageDataflowOp,
        SassCoverageDefUseEdge, SassCoverageDominatorBlock, SassCoverageFileReport,
        SassCoverageLiveRange, SassCoverageMemoryAccess, SassCoverageNaturalLoop,
        SassCoverageOptions, SassCoverageReachingUse, SassCoverageRegion, SassCoverageReport,
        SassCoverageSemanticPattern, SassCoverageSourceFormat, SassCoverageSsaValue,
        SassCoverageValueOp, SassOpcodeCatalogEntry, SassOpcodeCount, SassOpcodeCoverageState,
        SassOpcodeProbeAction, SassOpcodeProbeReason, SassOpcodeProbeTarget, SassOpcodeSignature,
        SassOpcodeSignatureCount, SassOpcodeSupport, SassSemanticPatternCount,
        SassUnsupportedInstruction, run_sass_coverage_scan,
    },
    coverage_compare::{
        SassCoverageComparisonOptions, SassCoverageComparisonReport, SassCoverageOpcodeChange,
        SassCoverageOpcodeDelta, SassCoverageProbeTargetDelta, run_sass_coverage_comparison,
    },
    driver::*,
    driver_support::{render_sass_file_side_by_side, render_side_by_side},
    fixtures::{
        SimpleKernelFixture, SimpleKernelFixtureKind, all_simple_kernel_fixture_kinds,
        simple_kernel_fixtures,
    },
    ir::{
        AggregateOperand, AggregateOperandKind, ControlTarget, ControlTargetKind, ImmediateValue,
        KernelIrFunction, KernelIrModule, KernelIrOp, KernelIrOpKind, MemoryAccessInfo,
        MemoryAddress, MemoryAddressBase, MemoryAddressImmediate, MemoryAddressImmediateKind,
        MemoryAddressKind, MemorySpace, PredicateCondition, PredicateConditionKind, RegisterRef,
        RegisterRefKind, SassCompareDType, SassComparisonKind, SassMappingConfidence,
        SassMemoryModifier, SassModifier, SassModifierKind, SassOpcode, SassOpcodeKind,
        SassOperandArityExpectation, SassSymbol, SassSyncKind, SassTensorElementType,
        SassTensorMmaShape, SassTensorMmaSignature, SassTensorScope, SassUnsupportedReason,
        SassWarpShuffleMode, ScalarOperand, ScalarOperandKind, lift_sass_module,
    },
    known_opcodes::{
        KnownSassOpcode, SassOpcodeCatalogClass, SassOpcodeCatalogKind, SassOpcodeCatalogSource,
        known_sass_opcodes,
    },
    lifted::{
        SassLiftedFunction, SassLiftedModule, SassLiftedOp, SassLiftedOpClass, SassLiftedOpDetail,
        SassLiftedOpKind, SassLiftedSemantics, SassLiftedValueRef, lift_sass_value_ir,
    },
    patterns::{
        SassPatternConfidence, SassPatternFunction, SassPatternLinkedOp, SassPatternModule,
        SassSemanticPattern, SassSemanticPatternCategory, SassSemanticPatternKind,
        recover_sass_patterns,
    },
    ptx_probes::{
        PtxDecompileProbe, PtxDecompileProbeKind, all_ptx_decompile_probe_kinds,
        ptx_decompile_probes,
    },
    sass::{
        RegisterClass, SassFunction, SassInstruction, SassModule, SassOperand, SassOperandKind,
        SassParseError, SassPredicate, SassRegister, SassSourcePosition, parse_nvidia_sass,
    },
};

#[cfg(test)]
mod tests;
