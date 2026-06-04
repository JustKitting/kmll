use std::{error::Error, fs, path::PathBuf};

use super::super::{
    KernelIrModule, SassAnalysisModule, SassLiftedModule, SassModule, SassPatternModule,
    analyze_sass_ir,
    driver_support::{render_sass_file_side_by_side, run_capture, run_checked},
    lift_sass_module, lift_sass_value_ir, parse_nvidia_sass, recover_sass_patterns,
};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DisassembledSassArtifact {
    pub cubin_path: PathBuf,
    pub sass_path: PathBuf,
    pub sass: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DecompiledSassArtifacts {
    pub ir: KernelIrModule,
    pub analysis: SassAnalysisModule,
    pub lifted: SassLiftedModule,
    pub patterns: SassPatternModule,
    pub ir_path: PathBuf,
    pub lifted_ir_path: PathBuf,
    pub analysis_path: PathBuf,
    pub pattern_path: PathBuf,
    pub side_by_side_path: PathBuf,
}

impl DecompiledSassArtifacts {
    pub fn from_parsed(
        output_dir: PathBuf,
        sass_path: PathBuf,
        source: Option<(PathBuf, String)>,
        sass: String,
        parsed: SassModule,
    ) -> Result<Self, Box<dyn Error>> {
        let ir = lift_sass_module(&parsed);
        let analysis = analyze_sass_ir(&ir);
        let lifted = lift_sass_value_ir(&ir, &analysis);
        let patterns = recover_sass_patterns(&ir);
        fs::create_dir_all(&output_dir)?;
        let ir_path = output_dir.join("lifted.ir.txt");
        fs::write(&ir_path, ir.to_text().as_bytes())?;
        let lifted_ir_path = output_dir.join("lifted-value-ir.txt");
        fs::write(&lifted_ir_path, lifted.to_text().as_bytes())?;
        let analysis_path = output_dir.join("analysis.txt");
        fs::write(&analysis_path, analysis.to_text().as_bytes())?;
        let pattern_path = output_dir.join("patterns.txt");
        fs::write(&pattern_path, patterns.to_text().as_bytes())?;
        let side_by_side = render_sass_file_side_by_side(
            &sass_path,
            source
                .as_ref()
                .map(|(path, text)| (path.as_path(), text.as_str())),
            &sass,
            &ir,
        );
        let side_by_side_path = output_dir.join("source-sass-ir.txt");
        fs::write(&side_by_side_path, side_by_side.as_bytes())?;

        Ok(Self {
            ir,
            analysis,
            lifted,
            patterns,
            ir_path,
            lifted_ir_path,
            analysis_path,
            pattern_path,
            side_by_side_path,
        })
    }

    pub fn parse_and_write(
        output_dir: PathBuf,
        sass_path: PathBuf,
        source: Option<(PathBuf, String)>,
        sass: String,
    ) -> Result<(SassModule, Self), Box<dyn Error>> {
        let parsed = parse_nvidia_sass(&sass)?;
        let artifacts = Self::from_parsed(output_dir, sass_path, source, sass, parsed.clone())?;
        Ok((parsed, artifacts))
    }
}

pub(super) fn disassemble_ptx_to_sass(
    ptx_path: PathBuf,
    output_dir: PathBuf,
    symbol: &str,
    compile_arch: &str,
) -> Result<DisassembledSassArtifact, Box<dyn Error>> {
    fs::create_dir_all(&output_dir)?;
    let cubin_path = output_dir.join(format!("{symbol}.{compile_arch}.cubin"));
    run_checked(
        "ptxas",
        &[
            format!("-arch={compile_arch}"),
            "-o".to_string(),
            cubin_path.display().to_string(),
            ptx_path.display().to_string(),
        ],
        &output_dir,
    )?;

    let sass_path = output_dir.join(format!("{symbol}.{compile_arch}.nvdisasm.sass"));
    let sass = run_capture("nvdisasm", &[cubin_path.display().to_string()], &output_dir)?;
    fs::write(&sass_path, sass.as_bytes())?;
    Ok(DisassembledSassArtifact {
        cubin_path,
        sass_path,
        sass,
    })
}
