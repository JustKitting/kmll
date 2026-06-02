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
- Typed operation specs describe the operation algebra used for profiling:
  numeric type, accumulator type, tensor shapes/layouts, operation kind, route,
  and CUDA launch dimensions.
- `ProfileTraceBuilder` records nested host spans and computes parent
  `self_seconds`, so parent work is credited explicitly instead of dropped from
  the report.
- `QueueOperationProfile` records the Tokio CUDA worker boundary:
  send wait, queue-to-worker latency, worker execution, completion wait, total
  time, and the typed operation spec for the job.

The current Qwen token/text/chat CLI timing uses wall-clock timers around runtime
initialization and each generation call. Those numbers are valid for end-to-end
CLI phases, but they do not prove CUDA kernel-level occupancy, memory bandwidth,
or launch overhead.

The intended inference profiling shape is functional rather than ad hoc:
Qwen-level phases should compose typed operations, and those operations should
carry the timing hooks and launch specs for the route they take. Host call-stack
spans account for CPU/control-flow time with parent self-time. CUDA launch
records bridge host enqueue work to per-stream device event timing once the
inference launch wrappers are instrumented.

Current live check:

```text
cargo run -- smoke-workers 3 8
```

This exercises the Tokio CUDA worker queue and prints one `QueueOperationProfile`
per smoke operation. In the first run after a build, queue-to-worker latency also
captures worker-side CUDA context/module startup before the queued job is
received, which is useful signal and should stay separately credited.
