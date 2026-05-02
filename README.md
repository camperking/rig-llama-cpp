<div align="center">

# rig-llama-cpp

**Run GGUF models locally inside your [Rig](https://github.com/0xPlaygrounds/rig) agents.**

[![crates.io](https://img.shields.io/crates/v/rig-llama-cpp.svg)](https://crates.io/crates/rig-llama-cpp)
[![docs.rs](https://img.shields.io/docsrs/rig-llama-cpp)](https://docs.rs/rig-llama-cpp)
[![license](https://img.shields.io/crates/l/rig-llama-cpp.svg)](#license)
[![ci](https://img.shields.io/github/actions/workflow/status/camperking/rig-llama-cpp/ci.yml?branch=master&label=CI)](https://github.com/camperking/rig-llama-cpp/actions/workflows/ci.yml)

</div>

A [Rig](https://github.com/0xPlaygrounds/rig) completion provider that runs GGUF models locally via [llama.cpp](https://github.com/ggml-org/llama.cpp) and their Rust bindings [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs).
Drop it in wherever you'd use a cloud provider — same `CompletionModel` trait, same agent API, but inference happens on your hardware with no API keys, no rate limits, and no data leaving the machine.

## Usage

```rust
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig_llama_cpp::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The minimal form — every other knob has a sensible default. Chain
    // .n_ctx, .sampling, .fit, .kv_cache, .checkpoints, or (with the
    // `mtmd` feature) .mmproj to override.
    let client = Client::builder("path/to/model.gguf")
        .n_ctx(8192)
        .build()?;

    let agent = client
        .agent("local")
        .preamble("You are a helpful assistant.")
        .max_tokens(512)
        .build();

    let response = agent.prompt("Hello!").await?;
    println!("{response}");
    Ok(())
}
```

The legacy positional `Client::from_gguf(...)` constructor is still
available for callers pinned to the 0.1.x API.

## Features

- Local inference with any GGUF model
- Completion and streaming support
- Tool calling (for models with OpenAI-compatible chat templates)
- Reasoning / thinking output
- Vision (multimodal) inference for models with an `mmproj` projector — opt in via the `mtmd` feature
- Automatic GPU/CPU layer fitting — llama.cpp probes available device memory and picks `n_gpu_layers` for you, no manual tuning required
- Backend selection via Cargo feature flags
- Configurable sampling parameters (top-p, top-k, min-p, temperature, penalties)

## Feature Flags

No default GPU backend — pick the one that matches your hardware:

| Feature  | Use for                                                          |
| -------- | ---------------------------------------------------------------- |
| _(none)_ | CPU-only inference                                               |
| `vulkan` | Cross-vendor GPU on Linux/Windows                                |
| `cuda`   | NVIDIA GPUs                                                      |
| `metal`  | Apple Silicon / macOS                                            |
| `rocm`   | AMD GPUs on Linux                                                |
| `openmp` | OpenMP CPU threading; combine with any GPU backend               |
| `mtmd`   | Multimodal (vision) inference; enables `ClientBuilder::mmproj`   |

```sh
cargo build --features vulkan
cargo build --features "cuda,mtmd"
```

Toolchain and runtime requirements per backend are documented upstream
in [llama.cpp's build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
A successful build does not guarantee a successful run — if backend
init fails at runtime, [`LoadError::BackendInit`] is returned rather
than panicking, so the application can fall back gracefully.

## Examples

```sh
MODEL_PATH=./model.gguf cargo run --example completion
MODEL_PATH=./model.gguf cargo run --example streaming
MODEL_PATH=./model.gguf cargo run --example stream_chat
MODEL_PATH=./model.gguf cargo run --example structured_output
MODEL_PATH=./model.gguf cargo run --example kv_cache
MODEL_PATH=./embedding-model.gguf cargo run --example embeddings

# Vision (requires mtmd feature + mmproj file)
MODEL_PATH=./vision-model.gguf MMPROJ_PATH=./mmproj.gguf IMAGE_PATH=./image.jpg \
    cargo run --features mtmd --example vision

# Hot-swap the loaded model on the same worker thread
RIG_MODEL_A=./model_a.gguf RIG_MODEL_B=./model_b.gguf cargo run --example reload
```

`N_GPU_LAYERS=20` can be used to offload 20 layers to the GPU.

By default, llama.cpp backend logs are suppressed so streaming and test output stay readable.
Set `RIG_LLAMA_CPP_LOGS=1` to re-enable raw backend logs when debugging model startup or decode issues.

## Testing

```sh
# Fast unit tests + doctests — no model required, run on every CI build.
cargo test --lib
cargo test --doc
```

The full integration suite (`tests/e2e/`) covers streaming completions,
vision, tool roundtrips, structured output, KV-cache quantization,
embedding, and sequential model reload. All tests are `#[ignore]`d and
auto-download their fixtures via `hf-hub` into the standard HuggingFace
cache (`~/.cache/huggingface/hub`) on first run — plan for ~20 GB.
Backend compilation is already covered upstream by `llama-cpp-rs`, and
the model fixtures are too large for hosted runners, so the e2e suite
does not run in CI.

```sh
cargo test --test e2e --features mtmd -- --ignored --nocapture
```

## Contributing

Issues and pull requests are welcome at
[github.com/camperking/rig-llama-cpp](https://github.com/camperking/rig-llama-cpp).

Before opening a PR, please run the same checks CI does
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

```sh
cargo fmt --all --check
cargo clippy --no-deps --all-targets -- -D warnings
cargo clippy --no-deps --all-targets --features mtmd -- -D warnings
cargo test --lib
cargo test --doc
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features mtmd
```

If your change touches inference behaviour, validate it locally with
`cargo test --test e2e --features mtmd -- --ignored --nocapture` — the
fixtures auto-download on first run (~20 GB; see the
[Testing](#testing) section).

For changes that affect the public API or the embedded `llama-cpp-2`
version, add an entry to [`CHANGELOG.md`](CHANGELOG.md) under
`[Unreleased]`. The crate's pre-1.0 SemVer policy is documented at the
top of that file.

## License

Licensed under the [MIT License](LICENSE). See [Cargo.toml](Cargo.toml) for dependency details.
