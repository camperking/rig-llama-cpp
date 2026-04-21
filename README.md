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

This crate forwards backend feature flags to `llama-cpp-2`.

- `vulkan` (default)
- `cuda`
- `metal`
- `rocm`
- `openmp`

Examples:

```sh
# Default build (Vulkan)
cargo build

# CUDA build
cargo build --no-default-features --features cuda

# ROCm build
cargo build --no-default-features --features rocm
```

Backend support depends on the corresponding `llama-cpp-2` feature and any required
native toolchain or system libraries being available on the host machine.

## Usage

```rust
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig_llama_cpp::{ Client, FitParams, KvCacheParams, SamplingParams };

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = Client::from_gguf(
        "path/to/model.gguf",
        8192, // n_ctx
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams::default(), // F16 K + F16 V — try KvCacheType::Q8_0 to halve KV VRAM
    )?;

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

## Examples

```sh
MODEL_PATH=./model.gguf cargo run --example completion
MODEL_PATH=./model.gguf cargo run --example streaming
MODEL_PATH=./model.gguf cargo run --example stream_chat
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
