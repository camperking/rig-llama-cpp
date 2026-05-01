# TODO — Production Readiness for crates.io

Punch list for getting `rig-llama-cpp` ready to publish on crates.io. Roughly ordered by "what crates.io and downstream users will trip on first."

## Hard blockers for crates.io

- [x] **Replace git deps in `Cargo.toml`.** Switched to `llama-cpp-2`/`llama-cpp-sys-2` 0.1.146 from crates.io (upstream `utilityai/llama-cpp-rs` now carries the fit-params change we needed). One follow-up: upstream does not yet expose `mtmd::void_mtmd_logs`, so mtmd log suppression is temporarily disabled when the `mtmd` feature is on (tracked in `lib.rs`).
- [x] **Add package metadata.** `description`, `license = "MIT"`, `repository`, `homepage`, `documentation`, `readme`, `keywords`, `categories` set in `Cargo.toml`. Verified with `cargo publish --dry-run`.
- [x] **Add docs.rs metadata.** `[package.metadata.docs.rs] features = ["mtmd"]` plus `--cfg docsrs` rustdoc args so gated APIs render.
- [x] **Declare MSRV.** `rust-version = "1.88"` (let-chains were stabilized in 1.88; edition 2024 already requires ≥1.85).
- [x] **Create `CHANGELOG.md`** with a 0.x semver policy that calls out the re-exported `KvCacheType` caveat.

## API hygiene

- [x] **Remove `anyhow` from public signatures.** Added `LoadError` (thiserror, `#[non_exhaustive]`) covering `BackendInit`, `ConfigureDevices`, `Fit`, `ModelLoad`, `MmprojInit` (mtmd-only), `InvalidPath`, `WorkerInitDisconnected`, `WorkerNotRunning`. `Client::{from_gguf, from_gguf_with_mmproj, reload}` and `EmbeddingClient::from_gguf` now return `Result<_, LoadError>`. `anyhow` moved to `dev-dependencies`; doctests use `Box<dyn std::error::Error>`.
- [x] **Add a builder for `Client::from_gguf`.** `Client::builder(model_path)` returns a `ClientBuilder` with `.n_ctx`, `.sampling`, `.fit`, `.kv_cache`, `.checkpoints`, and `.mmproj` (mtmd-only) setters; `.build()` returns `Result<Client, LoadError>`. Defaults: `n_ctx = 4096`, all other params via `Default`. Legacy `from_gguf` / `from_gguf_with_mmproj` retained as thin wrappers around the same shared `spawn` helper.
- [ ] **Reconsider default `vulkan` feature.** Hostile for `cargo add rig-llama-cpp` on a machine without Vulkan. Pragmatic defaults: no GPU feature (CPU/openmp) or a docs note + example. Add a build-time `compile_error!` when zero or two-plus mutually-exclusive backends are enabled.
- [ ] **`#[non_exhaustive]` on public structs.** `SamplingParams`, `FitParams`, `KvCacheParams`, `CheckpointParams`, `RawResponse`, `StreamChunk` are `pub` with `pub` fields. `#[non_exhaustive]` lets you add fields without a major bump; pair with `with_*` builders or document the `..Default::default()` pattern.
- [ ] **Reconsider re-exporting `llama_cpp_2::context::params::KvCacheType`.** Ties our semver to theirs — upstream breakage forces a breaking release. Consider a thin shim enum.

## Runtime quality

- [ ] **Replace `eprintln!` with `log` or `tracing`.** 49 call sites. Library output should go through a logger downstream code can capture/route. `RIG_LLAMA_CPP_LOGS` env shim can stay as a thin compat layer.
- [ ] **Eliminate `unwrap()` on `persistent.as_ref()/as_mut()`** in `worker.rs:327/336/353/375` and `image.rs:85`. Invariant-protected today (`ensure_persistent_ctx` was just called) but brittle. Make `ensure_persistent_ctx` return `&mut PersistentCtx` instead of mutating an `Option`, so the type system carries the invariant.
- [ ] **Document `expect("backend just set")` in `lib.rs:143`.** OnceLock flow guarantees it; add a `// SAFETY:`-style comment.
- [ ] **Fix clippy warnings.** `cargo clippy --all-targets` reports 6 lib + 1 test:
  - [ ] Delete or `#[cfg(test)]`/`#[allow(dead_code)]` `normalized_tool_parts` and `detect_image_media_type`.
  - [ ] Fix 5× `too_many_arguments` in `worker.rs` (`run_inference`, `run_inference_inner`, `run_text_inference`, `prepare_prompt_decode`) by grouping into a `RunCtx<'_>` struct.
- [ ] **Switch to bounded channel for inference requests** (`client.rs:56`). `unbounded_channel` lets a misbehaving caller OOM the worker thread. `mpsc::channel` with backpressure is the production posture.
- [ ] **Document `Drop` blocking behavior.** Single `Drop` joins the worker thread synchronously; if the worker is mid-decode this can block for seconds. Either document or send `Shutdown` and detach.

## Testing & CI

- [ ] **Add CI.** No `.github/workflows/` in the repo. At minimum: `cargo check` on each backend feature combo (matrix: `default`, `cuda` no-default, `metal` no-default on mac runner, `openmp`, `mtmd`), `cargo clippy -- -D warnings`, `cargo fmt --check`, `cargo doc --no-deps`. Model-bearing tests stay `#[ignore]` and run only with `MODEL_PATH` set.
- [ ] **Split unit tests from model-bearing tests.** `src/test.rs` is 928 lines and largely `#[ignore]`. Move pure parsing tests into a non-ignored unit-test module so `cargo test` exercises real coverage without a GGUF.
- [ ] **Document or replace `run_tests.sh`.** Either document in README or replace with `cargo test --features mtmd -- --include-ignored` + env vars.

## Docs

- [ ] **Audit doc comments for stale names.** `types.rs:164` references the old `from_gguf_with_fit` name. Sweep for similar.
- [ ] **Expand backend feature matrix in README.** Current matrix says "Vulkan default" but doesn't tell users what `vulkan` actually requires (libvulkan + drivers + working device). Add a 5-line "platform notes" section.
- [ ] **Verify MSRV in CI.** Add `package.rust-version` and have CI compile against that exact version, or doctests will rot silently.

## Recommended order

Tackle in this order to reach a publishable 0.1.0 with stable error/log surfaces:

1. Git deps → registry deps
2. Package metadata
3. Typed errors (replace `anyhow` in public API)
4. Default backend feature
5. Replace `eprintln!` with logger
6. Clippy clean
7. CI

Polish (builders, `#[non_exhaustive]`, channel backpressure, doc cleanup) can land in 0.1.x.
