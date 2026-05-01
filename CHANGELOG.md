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

### Changed

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
