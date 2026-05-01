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

### Caveat — re-exported `llama-cpp-2` types

This crate re-exports `llama_cpp_2::context::params::KvCacheType` (and may add
more upstream types over time). A breaking change in `llama-cpp-2` therefore
forces a breaking release of `rig-llama-cpp`, even when our own surface area
is unchanged. We aim to absorb upstream churn behind shim types where it is
cheap, but consumers should pin minor versions (e.g. `rig-llama-cpp = "0.1"`)
rather than caret-resolving across `0.x` boundaries.

## [Unreleased]

### Added

- `with_*` setters (chainable, `#[must_use]`) on `SamplingParams`,
  `FitParams`, `KvCacheParams`, and `CheckpointParams`. External callers
  should now build these via `Default::default().with_x(...)` instead of
  struct-literal syntax.
- `Client::builder(model_path)` returns a `ClientBuilder` with `.n_ctx`,
  `.sampling`, `.fit`, `.kv_cache`, `.checkpoints`, and `.mmproj` (mtmd-only)
  setters. Defaults to `n_ctx = 4096` plus `Default::default()` for every
  parameter struct. Future optional knobs can be added without breaking
  existing call sites. The legacy positional `from_gguf` /
  `from_gguf_with_mmproj` constructors still work and now delegate to the
  same internal spawn helper as the builder.
- `LoadError` (`thiserror`, `#[non_exhaustive]`): typed error returned by
  `Client::from_gguf`, `Client::from_gguf_with_mmproj`, `Client::reload`,
  `Client::builder().build()`, and `EmbeddingClient::from_gguf` instead of
  `anyhow::Error` / `Result<(), String>`. Variants: `BackendInit`,
  `ConfigureDevices`, `Fit`, `ModelLoad`, `MmprojInit` (mtmd-only),
  `InvalidPath`, `WorkerInitDisconnected`, `WorkerNotRunning`.

### Changed

- **BREAKING (build-time):** the `vulkan` Cargo feature is no longer a
  default. Pick a backend explicitly with `--features vulkan` (or `cuda`,
  `metal`, `rocm`); with no backend feature you get a CPU-only build that
  works on any host. Downstream callers that already pass
  `default-features = false` (e.g. the Chatty parent crate) are
  unaffected.
- **BREAKING (source):** `SamplingParams`, `FitParams`, `KvCacheParams`,
  `CheckpointParams`, `RawResponse`, and `StreamChunk` are now
  `#[non_exhaustive]`. External crates can no longer construct them via
  struct-literal syntax; use `Default::default()` plus the new `with_*`
  setters on the parameter structs. Future fields can now be added in a
  minor release.
- Re-exported `ClientBuilder` from the crate root so its rustdoc renders
  on docs.rs.
- Cleaned up stale rustdoc intra-doc links (`Client::from_gguf_with_fit`,
  bare `CompletionModel`) so `cargo doc` is warning-free both with and
  without the `mtmd` feature.
- `anyhow` is no longer a direct dependency; it is dev-only (used by the
  examples). Doctests use `Box<dyn std::error::Error>`.

- Repointed `llama-cpp-2` and `llama-cpp-sys-2` to the upstream
  `utilityai/llama-cpp-rs` crates.io releases (`0.1.146`); the previously
  vendored `camperking/llama-cpp-rs` fork is no longer required for the
  fitting-parameters API.
- `loader::fit_and_load_model` now constructs `LlamaContextParams` via the
  safe upstream wrapper instead of the raw `llama_context_default_params`
  FFI call.
- mtmd log suppression is temporarily disabled — upstream 0.1.146 does not
  yet expose `mtmd::void_mtmd_logs`. Backend logs are still silenced unless
  `RIG_LLAMA_CPP_LOGS=1`.

### Added

- `Cargo.toml` package metadata (`description`, `license`, `repository`,
  `homepage`, `documentation`, `readme`, `keywords`, `categories`) so the
  crate is publishable to crates.io.
- `package.metadata.docs.rs` enables the `mtmd` feature so docs.rs renders
  the multimodal API surface.
- `rust-version = "1.88"` to declare the minimum supported Rust version.

## [0.1.0] — unreleased

Initial public release. See [README.md](README.md) for the feature list.
