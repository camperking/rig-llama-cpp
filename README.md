# rig-llama-cpp

A [Rig](https://github.com/0xPlaygrounds/rig) completion provider that runs GGUF models locally via [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Features

- Local inference with any GGUF model
- Completion and streaming support
- Tool calling (for models with OpenAI-compatible chat templates)
- Reasoning / thinking output
- Vulkan GPU acceleration
- Configurable sampling parameters (top-p, top-k, min-p, temperature, penalties)

## Usage

```rust
use rig::client::CompletionClient;
use rig::completion::Prompt;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = rig_llama_cpp::Client::from_gguf(
        "path/to/model.gguf",
        99,   // n_gpu_layers (u32::MAX for all)
        8192, // n_ctx
        0.95, // top_p
        20,   // top_k
        0.0,  // min_p
        1.5,  // presence_penalty
        1.0,  // repetition_penalty
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

## License

See [Cargo.toml](Cargo.toml) for dependency details.
