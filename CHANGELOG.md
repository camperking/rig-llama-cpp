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

## [0.1.4] — 2026-05-06

### Fixed

- **Avoided a `GGML_ASSERT(!stacks.empty())` abort in grammar-constrained
  sampling.** Upstream
  [llama-cpp-rs#1007](https://github.com/utilityai/llama-cpp-rs/issues/1007)
  reports that `LlamaSampler::sample(ctx, idx)` aborts on the first
  sample call whenever the chain contains `LlamaSampler::grammar(...)`,
  even with a trivial `root ::= "a"` grammar (`llama-grammar.cpp:940`).
  Both of our grammar consumers go through that API: tool-call grammar
  from `ChatTemplateResult.grammar` and the GBNF that llama.cpp
  synthesizes for `output_schema` / json_schema requests. We could not
  reproduce the abort locally on Qwen3.5-2B Q4_K_M with the default
  Vulkan backend, but the upstream API combination is identical and the
  failure mode is `abort()` — not catchable from Rust — so we work
  around it preemptively. When grammar is present we now sample via the
  manual `LlamaTokenDataArray` + `apply_sampler` path that the issue
  reporter confirmed is crash-free, and call `sampler.accept(token)`
  explicitly (the manual path doesn't auto-accept the way `sample()`
  does). The non-grammar hot path is unchanged. Removable once upstream
  llama.cpp resolves the assert and `llama-cpp-2` ships a release that
  resyncs to it — see the comment on `sample_one` in `src/sampling.rs`.

## [0.1.3] — 2026-05-03

### Fixed

- **Streaming structured output silently swallowed every chunk.** When
  a `json_schema` was set on the request and the stream path was used
  (`agent.stream_chat(...)`), the OAI-compatible chat-template streaming
  parser (`llama_rs_chat_parse_state_update_oaicompat`) buffered every
  partial piece and then errored out on the final flush
  (`FfiError(-3)`), leaving consumers with zero text chunks. End users
  of crates like `chatty` saw `EOF while parsing a value at line 1
  column 0` because the accumulated buffer was empty. We now bypass
  that parser entirely whenever a `json_schema` is set: pieces still
  accumulate into the inference buffer, and after the loop completes
  we emit a single corrective chunk containing the result of
  `extract_structured_json`. That strips any leading role markers a
  template may leak (`<|im_start|>assistant\n\n…`) and any trailing
  junk before the JSON is sent downstream. Reproduces against both
  Qwen-3 and Gemma-4 — new e2e tests
  `qwen_structured_output_streaming` /
  `gemma_structured_output_streaming` (gated behind the existing
  `--ignored` flag like the other model-bearing tests) keep this
  honest.

## [0.1.2] — 2026-05-03

### Fixed

- **Empty-piece tokens no longer abort generation.** When `llama.cpp`'s
  `llama_token_to_piece` returns size 0 (control / unused / unknown-
  attribute tokens like Qwen3's `<|object_ref_*|>` pair, or a
  grammar-constrained sample landing on `<|fim_pad|>`), `llama-cpp-2`
  surfaces it as `TokenToStringError::UnknownTokenType`. Previously the
  sampling loop turned this into a hard error
  (`Token to piece failed: Unknown Token Type`), aborting the whole
  generation on the first such token. Canonical `llama.cpp` treats empty
  pieces as "no text emitted, keep generating" — the token is still
  consistent with the KV cache because we add it to the batch on the
  next iteration. We now do the same: empty pieces are emitted as empty
  strings and generation continues. Real errors
  (`InsufficientBufferSpace`, `FromUtf8Error`, …) still propagate. New
  unit tests in `sampling::tests` cover the three branches.

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
