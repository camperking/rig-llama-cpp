# rig-llama-cpp

A [Rig](https://github.com/0xPlaygrounds/rig) completion provider that runs GGUF models locally via [llama.cpp](https://github.com/ggml-org/llama.cpp) and their Rust bindings [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs).

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

There is **no default GPU backend** — `cargo add rig-llama-cpp` gives you a
CPU-only build. Pick exactly the backend that matches your hardware:

| Feature  | When to pick it                                                      | Build-time requirements                       |
| -------- | -------------------------------------------------------------------- | --------------------------------------------- |
| _(none)_ | CPU-only inference                                                   | C/C++ toolchain                               |
| `vulkan` | Cross-vendor GPU on Linux/Windows; default for AMD without ROCm      | Vulkan SDK or `libvulkan` + working ICD       |
| `cuda`   | NVIDIA GPUs                                                          | CUDA toolkit, matching driver                 |
| `metal`  | Apple Silicon / macOS                                                | Xcode command-line tools                      |
| `rocm`   | AMD GPUs on Linux                                                    | ROCm toolchain                                |
| `openmp` | OpenMP CPU threading; orthogonal — combine with any GPU backend      | OpenMP runtime (libgomp / libomp)             |
| `mtmd`   | Multimodal (vision) inference; enables `ClientBuilder::mmproj` etc.  | (none beyond the chosen backend)              |

Examples:

```sh
# CPU-only
cargo build

# Vulkan
cargo build --features vulkan

# CUDA + multimodal
cargo build --features "cuda,mtmd"

# CPU + OpenMP threading
cargo build --features openmp
```

Backend support also depends on the corresponding `llama-cpp-2` feature and the
host machine actually having the listed runtime libraries available.

### Platform notes

A successful build does not guarantee a successful run — the host still
needs the right driver and an actually-supported device. If backend init
fails at runtime, [`LoadError::BackendInit`] is returned rather than
panicking, so the application can fall back gracefully.

- **`vulkan`** — needs `libvulkan` (e.g. `libvulkan1` on Debian/Ubuntu) plus a
  working ICD: `mesa-vulkan-drivers` for AMD/Intel/llvmpipe, the proprietary
  driver for NVIDIA. `vulkaninfo` should report a non-CPU device for real
  performance; `lavapipe` (CPU-rendered Vulkan) builds and runs but is slow.
- **`cuda`** — needs a CUDA toolkit version that matches the installed
  NVIDIA driver. Mismatch produces a runtime error from `cudaErrorDriver`.
- **`metal`** — macOS only. Cross-compiling to Linux/Windows with this
  feature on will fail at the `llama-cpp-sys-2` build step.
- **`rocm`** — needs the ROCm runtime (`/opt/rocm`) and a supported AMD GPU
  (gfx9 / RDNA / CDNA). Older / consumer-only devices may be ignored even
  if ROCm itself installs cleanly.

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

The full integration suite needs real GGUF models and runs every test marked
`#[ignore]`: streaming completions, vision, tool roundtrips, structured
output, KV-cache quantization, embedding, and sequential model reload.
`./run_tests.sh` downloads the fixtures (Qwen 3.5-2B, Gemma-4 E4B, and the
nomic-embed-text-v2 embedding model) into the working directory via
`hf download` and runs each suite end-to-end. Plan for ~20 GB of model
downloads on the first run. The script does not run in CI — backend
compilation is already covered upstream by `llama-cpp-rs`, and the model
fixtures are too large for hosted runners.

## Use latest llama.cpp

To use the latest version of llama.cpp, clone the repo and point to the path in `Cargo.toml`. Make sure to update the submodules as well.

```sh
git clone --recursive https://github.com/utilityai/llama-cpp-rs
git submodule update --init --recursive
```

Then update the dependency in `Cargo.toml`:
`llama-cpp-2 = { path = "../llama-cpp-rs/llama-cpp-2" }`

## License

See [Cargo.toml](Cargo.toml) for dependency details.
