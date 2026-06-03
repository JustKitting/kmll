use std::path::PathBuf;

use nn_rust_autotune::{
    KernelActionMaterialization, KernelArtifactStore, KernelScheduleAction,
    MatvecRustCudaGenerator, MatvecSearchProblem, compile_standalone_kernel_crate,
    replay_schedule_actions,
};
use nn_rust_inference::runtime;

use crate::{AppResult, invalid_input, parse_required_flag_value, parse_required_usize};

const MATVEC_INSTRUCTIONS_USAGE: &str =
    "kernel-matvec-instructions ROWS COLS ACTION... [--artifact-root PATH] [--compile-arch sm_120]";

pub(crate) fn run_kernel_matvec_instructions(args: &[String]) -> AppResult<()> {
    let mut index = 0;
    let rows = parse_required_usize(args, &mut index, "rows", "kernel-matvec-instructions")?;
    let cols = parse_required_usize(args, &mut index, "cols", "kernel-matvec-instructions")?;
    if rows == 0 || cols == 0 {
        return Err(invalid_input(
            "kernel-matvec-instructions dimensions must be nonzero",
        ));
    }

    let mut actions = Vec::new();
    let mut artifact_root = runtime::default_artifact_dir().join("kernel-instructions");
    let mut compile_arch = None;
    while index < args.len() {
        match args[index].as_str() {
            "--artifact-root" => {
                artifact_root = PathBuf::from(parse_required_flag_value(
                    args,
                    &mut index,
                    "--artifact-root",
                )?);
            }
            "--compile-arch" => {
                compile_arch = Some(
                    parse_required_flag_value(args, &mut index, "--compile-arch")?.to_string(),
                );
            }
            flag if flag.starts_with("--") => {
                return Err(invalid_input(format!(
                    "kernel-matvec-instructions unknown argument {flag:?}; usage: {MATVEC_INSTRUCTIONS_USAGE}"
                )));
            }
            action => {
                actions.push(parse_matvec_action(action)?);
                index += 1;
            }
        }
    }
    if actions.is_empty() {
        return Err(invalid_input(format!(
            "kernel-matvec-instructions requires at least one ACTION; usage: {MATVEC_INSTRUCTIONS_USAGE}"
        )));
    }

    let problem = MatvecSearchProblem::bf16_row_major(rows, cols);
    let candidate = replay_schedule_actions(&problem, &actions)?;
    let store = KernelArtifactStore::new(artifact_root);
    let emitted_metadata = store.emit_metadata(&candidate)?;
    let emitted_crate = store.emit_standalone_crate(&candidate, &MatvecRustCudaGenerator)?;
    let output_dir = store.paths_for(&candidate).directory;
    let standalone_target_root = store.standalone_target_root();
    let compiled = compile_standalone_kernel_crate(
        &emitted_crate.paths.crate_dir,
        &output_dir,
        &emitted_crate.package_name,
        compile_arch.as_deref(),
        Some(&standalone_target_root),
    )?;

    println!(
        "kernel_matvec_instructions rows={rows} cols={cols} artifact_key={} symbol={} manifest_path={} source_path={} ptx_path={}",
        candidate.artifact_key().hex(),
        emitted_crate.symbol,
        emitted_metadata.paths.manifest_path.display(),
        emitted_crate.paths.source_path.display(),
        compiled.ptx_path.display(),
    );
    println!(
        "compiled_crate crate_dir={} output_dir={} target_dir={} stdout_bytes={} stderr_bytes={}",
        compiled.crate_dir.display(),
        compiled.output_dir.display(),
        standalone_target_root.display(),
        compiled.stdout_bytes,
        compiled.stderr_bytes,
    );
    Ok(())
}

fn parse_matvec_action(value: &str) -> AppResult<KernelScheduleAction> {
    let parts = value.split(':').collect::<Vec<_>>();
    if !(parts.len() == 3 || parts.len() == 4) {
        return Err(invalid_input(format!(
            "invalid matvec action {value:?}; expected op:axis:factor or op:axis:factor:materialization"
        )));
    }
    let op = parts[0];
    let axis = parse_action_axis(parts[1], value)?;
    let factor = parse_action_factor(parts[2], value)?;
    let materialization = if parts.len() == 4 {
        parse_action_materialization(parts[3], value)?
    } else {
        KernelActionMaterialization::DeferredGenerated
    };

    match op {
        "split" => Ok(KernelScheduleAction::split(axis, factor, materialization)),
        "upcast" => parse_deferred_only_action(value, materialization)
            .map(|()| KernelScheduleAction::upcast(axis, factor)),
        "unroll" => parse_deferred_only_action(value, materialization)
            .map(|()| KernelScheduleAction::unroll(axis, factor)),
        "local-tile" | "local_tile" => parse_deferred_only_action(value, materialization)
            .map(|()| KernelScheduleAction::local_tile(axis, factor)),
        "group-top" | "group_top" => parse_deferred_only_action(value, materialization)
            .map(|()| KernelScheduleAction::group_top(axis, factor)),
        "group" => parse_deferred_only_action(value, materialization)
            .map(|()| KernelScheduleAction::group(axis, factor)),
        "thread-group" | "thread_group" => parse_deferred_only_action(value, materialization)
            .map(|()| KernelScheduleAction::thread_group(axis, factor)),
        _ => Err(invalid_input(format!(
            "invalid matvec action {value:?}; expected split, upcast, unroll, local-tile, group-top, group, or thread-group"
        ))),
    }
}

fn parse_action_axis(value: &str, action: &str) -> AppResult<u8> {
    value.parse::<u8>().map_err(|error| {
        invalid_input(format!(
            "matvec action {action:?} has invalid axis {value:?}: {error}"
        ))
    })
}

fn parse_action_factor(value: &str, action: &str) -> AppResult<u32> {
    value.parse::<u32>().map_err(|error| {
        invalid_input(format!(
            "matvec action {action:?} has invalid factor {value:?}: {error}"
        ))
    })
}

fn parse_action_materialization(
    value: &str,
    action: &str,
) -> AppResult<KernelActionMaterialization> {
    match value {
        "existing" => Ok(KernelActionMaterialization::Existing),
        "deferred-generated" | "generated" => Ok(KernelActionMaterialization::DeferredGenerated),
        _ => Err(invalid_input(format!(
            "matvec action {action:?} has invalid materialization {value:?}; expected existing or deferred-generated"
        ))),
    }
}

fn parse_deferred_only_action(
    action: &str,
    materialization: KernelActionMaterialization,
) -> AppResult<()> {
    if materialization == KernelActionMaterialization::DeferredGenerated {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "matvec action {action:?} only supports deferred-generated materialization"
        )))
    }
}
