use std::path::PathBuf;

use nn_rust_inference::autotune::{
    AutoOptimizeConfig, BeamSearchConfig, KernelArtifactStore, KernelExpansionPolicy,
};

use crate::{AppResult, invalid_input, parse_required_flag_value};

pub(super) const GEMM_USAGE: &str = "kernel-autotune-gemm M N K [--allow-generated] [--measure] [--measure-repeat N] [--measure-warmup N] [--emit] [--emit-crate] [--compile] [--compile-arch sm_120] [--beam-width N] [--max-depth N] [--max-threads-per-block N|none] [--max-shared-memory-bytes N|none] [--max-accumulator-elements-per-thread N|none] [--max-output-elements-per-thread N|none] [--max-load-elements-per-block N|none] [--artifact-root PATH]";

pub(super) const MATVEC_USAGE: &str = "kernel-autotune-matvec ROWS COLS [--allow-generated] [--measure] [--measure-repeat N] [--measure-warmup N] [--emit] [--emit-crate] [--compile] [--compile-arch sm_120] [--beam-width N] [--max-depth N] [--max-threads-per-block N|none] [--max-shared-memory-bytes N|none] [--max-accumulator-elements-per-thread N|none] [--max-output-elements-per-thread N|none] [--max-load-elements-per-block N|none] [--artifact-root PATH]";

#[derive(Debug, Clone)]
pub(super) struct AutotuneCliOptions {
    pub(super) beam_width: usize,
    pub(super) max_depth: usize,
    pub(super) allow_generated: bool,
    pub(super) emit: bool,
    pub(super) emit_crate: bool,
    pub(super) compile: bool,
    pub(super) compile_arch: Option<String>,
    pub(super) measure: bool,
    pub(super) measure_repeat_count: usize,
    pub(super) measure_warmup_count: usize,
    artifact_root: Option<PathBuf>,
    policy_overrides: KernelExpansionPolicyOverrides,
}

impl AutotuneCliOptions {
    pub(super) fn parse(
        args: &[String],
        index: &mut usize,
        command: &str,
        usage: &str,
    ) -> AppResult<Self> {
        let mut options = Self::default();
        while *index < args.len() {
            if parse_kernel_expansion_policy_flag(args, index, &mut options.policy_overrides)? {
                continue;
            }
            match args[*index].as_str() {
                "--allow-generated" => {
                    options.allow_generated = true;
                    *index += 1;
                }
                "--emit" => {
                    options.emit = true;
                    *index += 1;
                }
                "--emit-crate" => {
                    options.emit_crate = true;
                    *index += 1;
                }
                "--compile" => {
                    options.compile = true;
                    options.emit_crate = true;
                    *index += 1;
                }
                "--measure" => {
                    options.measure = true;
                    *index += 1;
                }
                "--measure-repeat" => {
                    let value = parse_required_flag_value(args, index, "--measure-repeat")?;
                    options.measure_repeat_count =
                        parse_positive_usize(value, command, "--measure-repeat")?;
                }
                "--measure-warmup" => {
                    let value = parse_required_flag_value(args, index, "--measure-warmup")?;
                    options.measure_warmup_count =
                        parse_nonnegative_usize(value, command, "--measure-warmup")?;
                }
                "--beam-width" => {
                    let value = parse_required_flag_value(args, index, "--beam-width")?;
                    options.beam_width = parse_positive_usize(value, command, "--beam-width")?;
                }
                "--max-depth" => {
                    let value = parse_required_flag_value(args, index, "--max-depth")?;
                    options.max_depth = parse_positive_usize(value, command, "--max-depth")?;
                }
                "--artifact-root" => {
                    let value = parse_required_flag_value(args, index, "--artifact-root")?;
                    options.artifact_root = Some(PathBuf::from(value));
                }
                "--compile-arch" => {
                    let value = parse_required_flag_value(args, index, "--compile-arch")?;
                    options.compile_arch = Some(value.to_string());
                }
                other => {
                    return Err(invalid_input(format!(
                        "{command} unknown argument {other:?}; usage: {usage}"
                    )));
                }
            }
        }
        options.validate(command)?;
        Ok(options)
    }

    pub(super) fn config(&self) -> AutoOptimizeConfig {
        AutoOptimizeConfig::from_beam_search_config(BeamSearchConfig {
            beam_width: self.beam_width,
            max_depth: self.max_depth,
            require_launchable: !self.allow_generated,
        })
    }

    pub(super) fn policy(&self) -> KernelExpansionPolicy {
        self.policy_overrides
            .apply(KernelExpansionPolicy::for_search_config(
                !self.allow_generated,
            ))
    }

    pub(super) fn store(&self) -> KernelArtifactStore {
        self.artifact_root
            .as_ref()
            .map(|root| KernelArtifactStore::new(root.clone()))
            .unwrap_or_else(KernelArtifactStore::managed)
    }

    pub(super) fn score_namespace(&self) -> String {
        if self.measure {
            format!(
                "measured-cuda-event-r{}-w{}",
                self.measure_repeat_count, self.measure_warmup_count
            )
        } else {
            "heuristic".to_string()
        }
    }

    fn validate(&self, command: &str) -> AppResult<()> {
        if self.beam_width == 0 {
            return Err(invalid_input(format!(
                "{command} --beam-width must be nonzero"
            )));
        }
        if self.measure && self.measure_repeat_count == 0 {
            return Err(invalid_input(format!(
                "{command} --measure-repeat must be nonzero"
            )));
        }
        Ok(())
    }
}

impl Default for AutotuneCliOptions {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_depth: 1,
            allow_generated: false,
            emit: false,
            emit_crate: false,
            compile: false,
            compile_arch: None,
            measure: false,
            measure_repeat_count: 5,
            measure_warmup_count: 2,
            artifact_root: None,
            policy_overrides: KernelExpansionPolicyOverrides::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct KernelExpansionPolicyOverrides {
    max_threads_per_block: Option<Option<u32>>,
    max_shared_memory_bytes: Option<Option<u32>>,
    max_accumulator_elements_per_thread: Option<Option<u32>>,
    max_output_elements_per_thread: Option<Option<u32>>,
    max_load_elements_per_block: Option<Option<u32>>,
}

impl KernelExpansionPolicyOverrides {
    fn apply(self, mut policy: KernelExpansionPolicy) -> KernelExpansionPolicy {
        if let Some(value) = self.max_threads_per_block {
            policy = policy.with_max_threads_per_block(value);
        }
        if let Some(value) = self.max_shared_memory_bytes {
            policy = policy.with_max_shared_memory_bytes(value);
        }
        if let Some(value) = self.max_accumulator_elements_per_thread {
            policy = policy.with_max_accumulator_elements_per_thread(value);
        }
        if let Some(value) = self.max_output_elements_per_thread {
            policy = policy.with_max_output_elements_per_thread(value);
        }
        if let Some(value) = self.max_load_elements_per_block {
            policy = policy.with_max_load_elements_per_block(value);
        }
        policy
    }
}

fn parse_kernel_expansion_policy_flag(
    args: &[String],
    index: &mut usize,
    overrides: &mut KernelExpansionPolicyOverrides,
) -> AppResult<bool> {
    let Some(flag) = args.get(*index).cloned() else {
        return Ok(false);
    };
    match flag.as_str() {
        "--max-threads-per-block" => {
            let value = parse_required_flag_value(args, index, &flag)?;
            overrides.max_threads_per_block = Some(parse_optional_u32_cap(value, &flag)?);
            Ok(true)
        }
        "--max-shared-memory-bytes" => {
            let value = parse_required_flag_value(args, index, &flag)?;
            overrides.max_shared_memory_bytes = Some(parse_optional_u32_cap(value, &flag)?);
            Ok(true)
        }
        "--max-accumulator-elements-per-thread" | "--max-accumulators-per-thread" => {
            let value = parse_required_flag_value(args, index, &flag)?;
            overrides.max_accumulator_elements_per_thread =
                Some(parse_optional_u32_cap(value, &flag)?);
            Ok(true)
        }
        "--max-output-elements-per-thread" => {
            let value = parse_required_flag_value(args, index, &flag)?;
            overrides.max_output_elements_per_thread = Some(parse_optional_u32_cap(value, &flag)?);
            Ok(true)
        }
        "--max-load-elements-per-block" => {
            let value = parse_required_flag_value(args, index, &flag)?;
            overrides.max_load_elements_per_block = Some(parse_optional_u32_cap(value, &flag)?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

pub(super) fn format_optional_u32_cap(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn parse_optional_u32_cap(value: &str, flag: &str) -> AppResult<Option<u32>> {
    if ["none", "unlimited", "off"]
        .iter()
        .any(|keyword| value.eq_ignore_ascii_case(keyword))
    {
        return Ok(None);
    }
    let parsed = value.parse::<u32>().map_err(|error| {
        invalid_input(format!(
            "{flag} must be a positive integer or `none`, got {value:?}: {error}"
        ))
    })?;
    if parsed == 0 {
        return Err(invalid_input(format!(
            "{flag} must be nonzero or `none`, got {value:?}"
        )));
    }
    Ok(Some(parsed))
}

fn parse_positive_usize(value: &str, command: &str, flag: &str) -> AppResult<usize> {
    let parsed = parse_nonnegative_usize(value, command, flag)?;
    if parsed == 0 {
        return Err(invalid_input(format!("{command} {flag} must be nonzero")));
    }
    Ok(parsed)
}

fn parse_nonnegative_usize(value: &str, command: &str, flag: &str) -> AppResult<usize> {
    value.parse::<usize>().map_err(|error| {
        invalid_input(format!(
            "{command} {flag} must be a nonnegative integer, got {value:?}: {error}"
        ))
    })
}
