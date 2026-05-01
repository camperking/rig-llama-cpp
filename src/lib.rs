//! # rig-llama-cpp
//!
//! A [Rig](https://docs.rs/rig-core) provider that runs GGUF models locally
//! via [llama.cpp](https://github.com/ggml-org/llama.cpp), with optional Vulkan GPU acceleration.
//!
//! This crate implements Rig's [`CompletionModel`] and [`rig::embeddings::EmbeddingModel`] traits
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
//! This crate forwards backend feature flags to `llama-cpp-2`.
//!
//! - `vulkan` (default)
//! - `cuda`
//! - `metal`
//! - `rocm`
//! - `openmp`
//!
//! Examples:
//!
//! ```text
//! cargo build
//! cargo build --no-default-features --features cuda
//! cargo build --no-default-features --features rocm
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
//! use rig_llama_cpp::{FitParams, KvCacheParams, SamplingParams};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), anyhow::Error> {
//! let client = rig_llama_cpp::Client::from_gguf(
//!     "path/to/model.gguf",
//!     8192, // n_ctx
//!     SamplingParams::default(),
//!     FitParams::default(),
//!     KvCacheParams::default(),
//! )?;
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
mod request;
mod worker;
mod parsing;
mod embedding;

#[cfg(test)]
mod test;

pub use types::{
    CheckpointParams, FitParams, KvCacheParams, RawResponse, SamplingParams, StreamChunk,
};
pub use llama_cpp_2::context::params::KvCacheType;
pub use client::{Client, Model};
pub use embedding::{EmbeddingClient, EmbeddingModelHandle};

fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

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
        #[cfg(feature = "mtmd")]
        llama_cpp_2::mtmd::void_mtmd_logs();
    }
    let _ = BACKEND.set(backend);
    Ok(BACKEND.get().expect("backend just set"))
}
