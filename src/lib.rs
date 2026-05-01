//! # rig-llama-cpp
//!
//! A [Rig](https://docs.rs/rig-core) provider that runs GGUF models locally
//! via [llama.cpp](https://github.com/ggml-org/llama.cpp), with optional Vulkan GPU acceleration.
//!
//! This crate implements Rig's [`rig::completion::CompletionModel`] and [`rig::embeddings::EmbeddingModel`] traits
//! so that any GGUF model can be used as a drop-in replacement for cloud-based providers. It supports:
//!
//! - **Completion and streaming** — both one-shot and token-by-token responses.
//! - **Tool calling** — models with OpenAI-compatible chat templates can invoke tools.
//! - **Reasoning / thinking** — extended thinking output is forwarded when the model supports it.
//! - **Configurable sampling** — top-p, top-k, min-p, temperature, presence and repetition penalties.
//! - **Embeddings** — generate text embeddings using GGUF embedding models.
//!
//! # Feature flags
//!
//! There is **no default GPU backend** — pick exactly the one that matches
//! your hardware. With no feature enabled the build is CPU-only.
//!
//! GPU backends (forwarded to `llama-cpp-2`):
//!
//! - `vulkan` — cross-vendor GPU (recommended on Linux/Windows when CUDA/ROCm aren't set up).
//! - `cuda` — NVIDIA GPUs with the CUDA toolkit installed.
//! - `metal` — Apple Silicon / macOS.
//! - `rocm` — AMD GPUs on Linux with the ROCm toolchain.
//!
//! Other:
//!
//! - `openmp` — OpenMP CPU threading; orthogonal to the GPU backends and may be combined with any of them.
//! - `mtmd` — multimodal (vision) inference; required for `Client::from_gguf_with_mmproj` and `ClientBuilder::mmproj`.
//!
//! Examples:
//!
//! ```text
//! cargo build --features vulkan
//! cargo build --features cuda
//! cargo build --features "vulkan,mtmd"
//! ```
//!
//! Backend support depends on the corresponding `llama-cpp-2` feature and any required
//! native toolchain or system libraries being available on the host machine.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use rig::client::CompletionClient;
//! use rig::completion::Prompt;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let client = rig_llama_cpp::Client::builder("path/to/model.gguf")
//!     .n_ctx(8192)
//!     .build()?;
//!
//! let agent = client
//!     .agent("local")
//!     .preamble("You are a helpful assistant.")
//!     .max_tokens(512)
//!     .build();
//!
//! let response = agent.prompt("Hello!").await?;
//! println!("{response}");
//! # Ok(())
//! # }
//! ```

mod types;
mod client;
mod error;
mod request;
mod slot;
mod worker;
mod parsing;
mod prompt;
mod sampling;
mod loader;
mod checkpoint;
#[cfg(feature = "mtmd")]
mod image;
mod embedding;

#[cfg(test)]
mod test;

pub use types::{
    CheckpointParams, FitParams, KvCacheParams, RawResponse, SamplingParams, StreamChunk,
};
pub use llama_cpp_2::context::params::KvCacheType;
pub use client::{Client, ClientBuilder, Model};
pub use embedding::{EmbeddingClient, EmbeddingModelHandle};
pub use error::LoadError;

fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

/// Whether to forward llama.cpp's *C-side* logging to stderr.
///
/// This only controls log lines that originate inside the `llama-cpp-2` /
/// `llama-cpp-sys-2` C++ code (via `printf`-style writes that bypass Rust's
/// `log` facade). Library-level diagnostics from `rig-llama-cpp` itself go
/// through the [`log`] crate and are controlled by the consumer's logger
/// configuration (e.g. `RUST_LOG=rig_llama_cpp=debug`), not this env var.
fn llama_logs_enabled() -> bool {
    env_flag_enabled("RIG_LLAMA_CPP_LOGS")
}

/// Process-wide [`LlamaBackend`] initialised on first use and shared by every
/// worker (chat + embedding). The underlying llama.cpp backend is a global
/// singleton — calling `LlamaBackend::init()` twice in the same process
/// returns `BackendAlreadyInitialized`. Routing all callers through this
/// helper means a chat client and an embedding client can coexist without
/// racing on the C-side init flag.
///
/// Returns `Ok(&'static LlamaBackend)` once the backend is up; subsequent
/// calls are cheap (single `OnceLock::get`). On platforms where init can
/// fail (e.g. no Vulkan device) the error is sticky for the lifetime of
/// the process — there's no recovering anyway.
pub(crate) fn shared_backend()
-> Result<&'static llama_cpp_2::llama_backend::LlamaBackend, String> {
    use llama_cpp_2::llama_backend::LlamaBackend;
    use std::sync::{Mutex, OnceLock};

    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    static INIT_LOCK: Mutex<()> = Mutex::new(());

    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    // Serialise concurrent first-time initialisations. The C-side init flag
    // is process-global so multiple threads racing on `LlamaBackend::init`
    // will produce `BackendAlreadyInitialized` for the loser even though
    // they all want the same handle.
    let _guard = INIT_LOCK.lock().map_err(|e| e.to_string())?;
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }

    let mut backend =
        LlamaBackend::init().map_err(|e| format!("Backend init failed: {e}"))?;
    if !llama_logs_enabled() {
        backend.void_logs();
        // NOTE: upstream llama-cpp-2 0.1.146 does not yet expose a way to
        // silence mtmd's own log stream — when the `mtmd` feature is on,
        // mmproj init may print to stderr. Track upstream for an mtmd
        // log-silencing API and re-enable suppression here.
    }
    let _ = BACKEND.set(backend);
    // INVARIANT: we hold `INIT_LOCK` for the duration of this function and
    // just called `BACKEND.set(backend)`. Any concurrent caller that
    // reached the second `BACKEND.get().is_some()` check above already
    // returned, so reaching this line means we are the unique writer and
    // `BACKEND` is now `Some`. Even if `set()` raced (`Err`-returning),
    // the "loser" still observes the state filled by the winner — `get()`
    // is guaranteed to return `Some`.
    Ok(BACKEND.get().expect("BACKEND set above under INIT_LOCK"))
}
