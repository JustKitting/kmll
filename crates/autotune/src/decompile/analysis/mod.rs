mod cfg;
mod dataflow;
mod memory;
mod regions;
mod registers;
mod render;
mod types;

pub use self::types::{
    SassAnalysisFunction, SassAnalysisModule, SassBasicBlock, SassBlockTerminator, SassCfgEdge,
    SassCfgEdgeKind, SassDataflowOp, SassDefUseEdge, SassDominatorBlock, SassLiveRange,
    SassMemoryAccess, SassMemoryAccessKind, SassNaturalLoop, SassReachingUse, SassRegion,
    SassRegionKind, SassRegionPath, SassSsaValue, SassValueOp, SassValueOpKind,
};

use super::{KernelIrFunction, KernelIrModule};

use self::{
    cfg::{analyze_control_structure, build_blocks, build_edges},
    dataflow::{analyze_dataflow, analyze_reaching_defs, build_value_ops},
    memory::analyze_memory_accesses,
    regions::recover_regions,
};

pub fn analyze_sass_ir(module: &KernelIrModule) -> SassAnalysisModule {
    SassAnalysisModule {
        target: module.target.clone(),
        functions: module.functions.iter().map(analyze_function).collect(),
    }
}

fn analyze_function(function: &KernelIrFunction) -> SassAnalysisFunction {
    let blocks = build_blocks(function);
    let edges = build_edges(function, &blocks);
    let (dominators, natural_loops) = analyze_control_structure(&blocks, &edges);
    let regions = recover_regions(function, &blocks, &edges, &natural_loops);
    let dataflow = function
        .ops
        .iter()
        .map(analyze_dataflow)
        .collect::<Vec<_>>();
    let (reaching_uses, live_ranges, ssa_values, def_use_edges) =
        analyze_reaching_defs(&blocks, &edges, &dataflow);
    let value_ops = build_value_ops(function, &blocks, &dataflow, &ssa_values, &def_use_edges);
    let memory_accesses = analyze_memory_accesses(function);
    SassAnalysisFunction {
        name: function.name.clone(),
        blocks,
        edges,
        dominators,
        natural_loops,
        regions,
        dataflow,
        reaching_uses,
        ssa_values,
        def_use_edges,
        value_ops,
        live_ranges,
        memory_accesses,
    }
}
