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
- [x] **Reconsider default `vulkan` feature.** Dropped `default = ["vulkan"]` so `cargo add rig-llama-cpp` produces a CPU-only build that works on every host. Backend features (`vulkan`, `cuda`, `metal`, `rocm`) are now opt-in, with a hardware/runtime-requirement matrix in the README and `lib.rs` rustdoc. Skipped the `compile_error!` mutual-exclusion check — none of the backends are actually mutually exclusive in upstream llama-cpp-2 (they're additive), and platform mismatches like `metal` on Linux already fail loudly during the `llama-cpp-sys-2` build, so duplicating that here would just rot.
- [x] **`#[non_exhaustive]` on public structs.** All six (`SamplingParams`, `FitParams`, `KvCacheParams`, `CheckpointParams`, `RawResponse`, `StreamChunk`) are now non-exhaustive. The four parameter structs gained `with_*` setters (chainable, `#[must_use]`) so external callers can build them via `Default::default().with_x(...)`. The two response structs are read-only for external code, so they keep their existing `pub` fields without builders.
- [ ] **Reconsider re-exporting `llama_cpp_2::context::params::KvCacheType`.** Ties our semver to theirs — upstream breakage forces a breaking release. Consider a thin shim enum.

## Runtime quality

- [x] **Replace `eprintln!` with `log` or `tracing`.** Migrated to the `log` facade. ~25 library-side `eprintln!` sites became `log::{info,debug,warn}!` calls; level-gating is now the consumer's responsibility (`RUST_LOG=rig_llama_cpp=debug`). The `RIG_LLAMA_CPP_LOGS` env var is now scoped to llama.cpp's own C-side logging only (the `void_logs()` flag and the `fit_params` log_level), since those bypass Rust's logger. Doc-comment / test-only `println!` left untouched.
- [x] **Eliminate `unwrap()` on `persistent.as_ref()/as_mut()`.** `ensure_persistent_ctx` now returns `Result<&mut PersistentCtx<'m>, String>`, so callers chain operations on the live reference instead of poking through the `Option`. All five `persistent.{as_ref,as_mut}().unwrap()` sites in `worker.rs` and `image.rs` are gone. Drive-by: the two `RwLock::{read,write}().unwrap()` calls on `sampling_params` in `client.rs` switched to `unwrap_or_else(|p| p.into_inner())` (recovering past poison is safe — the field is `Copy` floats), and the `expect("run_image_inference called without mtmd context")` in `image.rs` is now a returned `BUG:` error instead of a panic.
- [x] **Document `expect("backend just set")` in `lib.rs`.** Replaced with a multi-line `INVARIANT:` comment that walks through why holding `INIT_LOCK` makes the `BACKEND.get()` after `set()` infallible (and why the loser of a `set()` race still observes `Some`).
- [x] **Fix clippy warnings.** `cargo clippy --no-deps --all-targets [--features mtmd] -- -D warnings` is now clean for both feature sets.
  - [x] `normalized_tool_parts` (request.rs) and `detect_image_media_type` (test.rs) gated behind `#[cfg(feature = "mtmd")]` along with their imports — both are mtmd-only call paths.
  - [x] Worker-internal call chain refactored to thread a borrowed `RunCtx<'a, 'm>` (backend, model, n_ctx, kv_cache, checkpoint_params, mtmd_ctx) instead of seven positional parameters. `run_inference` collapsed into a single dispatch fn. `inference_worker`'s six bringup args folded into a `WorkerInit<'a>` struct. The two cfg-arm `run_inference_inner` shims are gone.
  - [x] `Client::reload`'s positional 0.1.x signature carries a targeted `#[allow(clippy::too_many_arguments)]` with a comment that a `ReloadOptions` / `reload_builder` shape is the planned 0.2 follow-up. (Not split into a separate TODO item; `Client::builder` already exists for construction so the parallel `reload_builder` is a small future PR.)
- [x] **Switch to bounded channel for inference requests.** Replaced `mpsc::unbounded_channel` with `mpsc::channel(8)`. `Model::completion` / `Model::stream` use `send().await` (natural backpressure for async callers); `Client::reload` uses `blocking_send` (sync API, documented to be called from `spawn_blocking`); `Client::drop` uses `try_send(Shutdown)` as a best-effort wake-up — the new cancel-after-command check in the worker handles the queue-full case without needing to retry.
- [x] **Document `Drop` blocking behavior + add cancel signal.** Drop now flips an `Arc<AtomicBool>` that the worker polls per-token in the sampler loop and per-chunk in prompt prefill / mtmd suffix decode. Pessimal Drop wait is now one decode step (single token), not the rest of an in-flight generation. The `# Lifecycle` section on `Client` and the `impl Drop` doc-comment spell out the memory-ordering rationale (avoid 2× model size in RAM/VRAM during reload), the cancel mechanics, and how `Model` clones outliving the `Client` interact (their sends fail with `SendError` once the worker drops its receiver — they don't prevent shutdown).

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
