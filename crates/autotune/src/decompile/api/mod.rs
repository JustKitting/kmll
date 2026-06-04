mod analysis;
mod architecture;
mod autotune_bridge;
mod coverage;
mod driver;
mod fixtures;
mod ir;
mod lifted;
mod opcodes;
mod patterns;
mod probes;
mod render;
mod sass;

pub use self::{
    analysis::{
        SassAnalysisFunction, SassAnalysisModule, SassBasicBlock, SassBlockTerminator, SassCfgEdge,
        SassCfgEdgeKind, SassDataflowOp, SassDataflowSite, SassDefUseEdge, SassDominatorBlock,
        SassLiveRange, SassMemoryAccess, SassMemoryAccessKind, SassNaturalLoop, SassReachingUse,
        SassRegion, SassRegionKind, SassRegionPath, SassSsaValue, SassValueOp, SassValueOpKind,
        analyze_sass_ir,
    },
    architecture::{SassArchitecture, SassArchitectureSuffix, SassTarget},
    autotune_bridge::{
        DecompiledAutotuneError, DecompiledAutotuneEvidence, DecompiledAutotuneOperation,
        DecompiledAutotuneShape, decompiled_autotune_operation,
        decompiled_autotune_operation_with_module,
    },
    coverage::{
        SassCoverageBasicBlock, SassCoverageCfgEdge, SassCoverageComparisonOptions,
        SassCoverageComparisonReport, SassCoverageDataflowOp, SassCoverageDefUseEdge,
        SassCoverageDominatorBlock, SassCoverageFileReport, SassCoverageLiveRange,
        SassCoverageMemoryAccess, SassCoverageNaturalLoop, SassCoverageOpcodeChange,
        SassCoverageOpcodeDelta, SassCoverageOptions, SassCoverageProbeTargetDelta,
        SassCoverageReachingUse, SassCoverageRegion, SassCoverageReport,
        SassCoverageSemanticPattern, SassCoverageSourceFormat, SassCoverageSsaValue,
        SassCoverageValueOp, SassOpcodeCatalogEntry, SassOpcodeCount, SassOpcodeCoverageState,
        SassOpcodeProbeAction, SassOpcodeProbeReason, SassOpcodeProbeTarget, SassOpcodeSignature,
        SassOpcodeSignatureCount, SassOpcodeSupport, SassSemanticPatternCount,
        SassUnsupportedInstruction, run_sass_coverage_comparison, run_sass_coverage_scan,
    },
    driver::{
        DecompileAutotuneGemmOptions, DecompileAutotuneGemmReport, DecompileAutotuneMatvecOptions,
        DecompileAutotuneMatvecReport, DecompileAutotuneSassOptions, DecompileAutotuneSassReport,
        DecompileFixtureCoverageOptions, DecompileFixtureCoverageReport, DecompileFixtureOptions,
        DecompileFixtureReport, DecompilePtxProbeOptions, DecompilePtxProbeReport,
        SassFileDecompileOptions, SassFileDecompileReport, run_decompile_autotune_gemm,
        run_decompile_autotune_matvec, run_decompile_autotune_sass, run_decompile_fixture_coverage,
        run_decompile_fixtures, run_decompile_ptx_probes, run_sass_file_decompile,
    },
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
        SassMemoryAtomicOp, SassMemoryModifier, SassModifier, SassModifierKind, SassNumericDType,
        SassOpcode, SassOpcodeKind, SassOperandArityExpectation, SassSymbol, SassSyncKind,
        SassTensorElementType, SassTensorMmaShape, SassTensorMmaSignature, SassTensorScope,
        SassUnsupportedReason, SassWarpShuffleMode, ScalarOperand, ScalarOperandKind,
        lift_sass_module,
    },
    lifted::{
        SassLiftedFunction, SassLiftedModule, SassLiftedOp, SassLiftedOpClass, SassLiftedOpDetail,
        SassLiftedOpKind, SassLiftedSemantics, SassLiftedValueRef, lift_sass_value_ir,
    },
    opcodes::{
        KnownSassOpcode, SassOpcodeCatalogClass, SassOpcodeCatalogKind, SassOpcodeCatalogSource,
        known_sass_opcodes,
    },
    patterns::{
        SassPatternConfidence, SassPatternFunction, SassPatternLinkedOp, SassPatternModule,
        SassSemanticPattern, SassSemanticPatternCategory, SassSemanticPatternKind,
        recover_sass_patterns,
    },
    probes::{
        AUTO_COMPILE_ARCH, PtxDecompileProbe, PtxDecompileProbeKind, all_ptx_decompile_probe_kinds,
        ptx_decompile_probes,
    },
    render::{render_sass_file_side_by_side, render_side_by_side},
    sass::{
        RegisterClass, SassFunction, SassInstruction, SassModule, SassOperand, SassOperandKind,
        SassParseError, SassPredicate, SassRegister, SassSourcePosition, parse_nvidia_sass,
    },
};
