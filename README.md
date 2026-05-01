# rig-llama-cpp

A [Rig](https://github.com/0xPlaygrounds/rig) completion provider that runs GGUF models locally via [llama.cpp](https://github.com/ggml-org/llama.cpp) and their Rust bindings [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs).

## Features

- Local inference with any GGUF model
- Completion and streaming support
- Tool calling (for models with OpenAI-compatible chat templates)
- Reasoning / thinking output
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
