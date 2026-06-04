use std::{error::Error, fs, path::Path};

use super::super::{
    analyze_sass_ir, driver_support::render_sass_file_side_by_side, lift_sass_module,
    lift_sass_value_ir, parse_nvidia_sass, recover_sass_patterns,
};
use super::types::{SassFileDecompileOptions, SassFileDecompileReport};

pub fn run_sass_file_decompile(
    options: &SassFileDecompileOptions,
) -> Result<SassFileDecompileReport, Box<dyn Error>> {
    let sass = fs::read_to_string(&options.sass_path)?;
    let source = match &options.source_path {
        Some(path) => Some((path.clone(), fs::read_to_string(path)?)),
        None => None,
    };
    let parsed = parse_nvidia_sass(&sass)?;
    let project_ir = lift_sass_module(&parsed);
    let analysis = analyze_sass_ir(&project_ir);
    let lifted = lift_sass_value_ir(&project_ir, &analysis);
    let patterns = recover_sass_patterns(&project_ir);
    let output_dir = options.output_dir.clone().unwrap_or_else(|| {
        options
            .sass_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    });
    fs::create_dir_all(&output_dir)?;
    let stem = options
        .sass_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("sass");
    let ir_path = output_dir.join(format!("{stem}.lifted.ir.txt"));
    fs::write(&ir_path, project_ir.to_text().as_bytes())?;
    let lifted_ir_path = output_dir.join(format!("{stem}.lifted-value-ir.txt"));
    fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;
    let analysis_path = output_dir.join(format!("{stem}.analysis.txt"));
    fs::write(&analysis_path, analysis.to_text().as_bytes())?;
    let pattern_path = output_dir.join(format!("{stem}.patterns.txt"));
    fs::write(&pattern_path, patterns.to_text().as_bytes())?;
    let side_by_side_path = output_dir.join(format!("{stem}.source-sass-ir.txt"));
    let side_by_side = render_sass_file_side_by_side(
        options.sass_path.as_path(),
        source
            .as_ref()
            .map(|(path, text)| (path.as_path(), text.as_str())),
        &sass,
        &project_ir,
    );
    fs::write(&side_by_side_path, side_by_side.as_bytes())?;

    Ok(SassFileDecompileReport {
        sass_path: options.sass_path.clone(),
        source_path: options.source_path.clone(),
        ir_path,
        lifted_ir_path,
        analysis_path,
        pattern_path,
        side_by_side_path,
        parsed_instruction_count: parsed.instruction_count(),
        cfg_block_count: analysis.block_count(),
        cfg_edge_count: analysis.edge_count(),
        dominator_block_count: analysis.dominator_block_count(),
        natural_loop_count: analysis.natural_loop_count(),
        region_count: analysis.region_count(),
        reaching_use_count: analysis.reaching_use_count(),
        ssa_value_count: analysis.ssa_value_count(),
        def_use_edge_count: analysis.def_use_edge_count(),
        value_op_count: analysis.value_op_count(),
        lifted_op_count: lifted.op_count(),
        live_range_count: analysis.live_range_count(),
        memory_access_count: analysis.memory_access_count(),
        semantic_pattern_count: patterns.pattern_count(),
        unsupported_instruction_count: project_ir.unsupported_instruction_count(),
    })
}
