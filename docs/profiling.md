# Profiling

`crates/profiling` is the project-local source of truth for profiling math and
timing primitives. Model code and CLI commands should use this crate instead of
open-coded rate, median, or generation-timing calculations.

Current scope:

- `ProfileTimer` wraps wall-clock phase timing.
- `GenerationTimings` owns prefill/decode duration fields and derived
  token-rate calculations.
- `tokens_per_second` and `rate_per_second` return `None` for zero, negative, or
  non-finite elapsed time.
- `SampleStats` and `median_f64` reject empty or non-finite samples.

The current Qwen token/text/chat CLI timing uses wall-clock timers around runtime
initialization and each generation call. Those numbers are valid for end-to-end
CLI phases, but they do not prove CUDA kernel-level occupancy, memory bandwidth,
or launch overhead. The next profiling layer should add CUDA event timing around
device phases and a Qwen profile report that records prompt length, generated
token count, prefill/decode phase timings, output tokens, and quality metrics.
