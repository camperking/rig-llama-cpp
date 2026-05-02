# Changelog

All notable changes to `rig-llama-cpp` are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The crate is pre-1.0, so the [SemVer](https://semver.org/) policy below applies.

## Versioning policy

While the crate is on `0.x`:

- A bump to `0.Y` (e.g. `0.1` → `0.2`) signals a **breaking change** in the public
  API or in the embedded `llama-cpp-2` / `llama-cpp-sys-2` versions.
- A bump to `0.x.Z` (e.g. `0.1.0` → `0.1.1`) is reserved for additive or
  non-breaking changes.

The public surface is fully owned by this crate — `KvCacheType`, the
parameter structs, and `LoadError` are all defined here, not re-exported
from `llama-cpp-2`. A new upstream `ggml_type` is therefore an additive
`0.1.x` change here (we add a corresponding shim variant), not a breaking
release.

## [0.1.1] — 2026-05-02

### Changed

- **README polish.** Centered the title, added a tagline, surfaced
  crates.io / docs.rs / license / CI shields.io badges, expanded the
  intro paragraph, and named MIT explicitly in the License section.
  No code changes.

## [0.1.0] — 2026-05-01

Initial public release.

### Highlights

- **Rig integration.** Implements `rig::client::CompletionClient` /
  `rig::completion::CompletionModel` and the matching embedding traits, so
  any GGUF model is a drop-in for cloud Rig providers.
- **Local GGUF inference.** Any architecture supported by upstream
  `llama-cpp-2` (`0.1.146`).
- **Streaming and one-shot** completion, **tool calling** on OpenAI-template
  models, **structured output** via grammar-constrained sampling, and
  **reasoning / thinking deltas** surfaced separately from the main response
  stream.
- **Vision (multimodal) inference** via the `mtmd` feature for models that
  ship an `mmproj` projector.
- **Automatic GPU/CPU layer fitting** — llama.cpp probes available device
  memory and picks `n_gpu_layers` for you. Tunable per-device margins via
  `FitParams`.
- **KV-cache prefix reuse + state checkpoints** so multi-turn conversations
  skip re-decoding the shared prefix, including a checkpoint-based fallback
  for hybrid / recurrent architectures whose memory rejects partial trims.
- **Configurable KV-cache quantization** (`F16` default, `Q8_0` / `Q4_0`
  available) for VRAM savings at long contexts.
- **Pluggable backends** as opt-in Cargo features: `vulkan`, `cuda`, `metal`,
  `rocm`, plus `openmp` (CPU threading) and `mtmd` (multimodal). Default
  build is CPU-only and works on any host.
- **Builder-pattern construction** via `Client::builder(model_path)`, with
  the legacy positional `Client::from_gguf` constructors retained for
  backward compatibility.
- **Bounded inference command channel** with backpressure, plus an
  `Arc<AtomicBool>` cancel signal that lets `Drop` (and future per-request
  cancel hooks) tear down a long generation within a single decode step.
- **Typed errors** (`LoadError`, `#[non_exhaustive]`) on every load-stage
  entry point — no `anyhow` in the public API.
- **`log` crate facade** for library-level diagnostics; configure verbosity
  via `RUST_LOG=rig_llama_cpp=debug`. The `RIG_LLAMA_CPP_LOGS=1` env var
  toggles llama.cpp's own C-side log stream.

### Known caveats

- mtmd log suppression is temporarily disabled — upstream `llama-cpp-2`
  `0.1.146` does not yet expose `mtmd::void_mtmd_logs`, so loading an
  `mmproj` projector with the `mtmd` feature on may print to stderr.
  Tracked as a follow-up; will be re-enabled when the upstream API lands.
