//! Public error types for `rig-llama-cpp`.
//!
//! Public-API constructors return [`LoadError`] (instead of `anyhow::Error` or
//! `String`) so that downstream code can match on the failure mode without
//! resorting to substring checks on a stringly-typed message.

use thiserror::Error;

/// Failure modes returned when constructing or reloading a [`crate::Client`]
/// or [`crate::EmbeddingClient`].
///
/// Variants are intentionally coarse-grained: each one maps to a distinct
/// stage of model bring-up, and the embedded `String` carries the underlying
/// llama.cpp / FFI message verbatim for diagnostic purposes.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LoadError {
    /// The process-wide [`llama_cpp_2::llama_backend::LlamaBackend`] could not
    /// be initialised. This failure is sticky for the lifetime of the
    /// process — typically it means there is no usable GPU device for the
    /// selected backend (e.g. no Vulkan ICD loadable, no CUDA driver, etc.).
    #[error("llama.cpp backend initialisation failed: {0}")]
    BackendInit(String),

    /// Configuring the requested set of GPU devices failed (e.g. an invalid
    /// device list was passed to `LlamaModelParams::with_devices`).
    #[error("failed to configure backend devices: {0}")]
    ConfigureDevices(String),

    /// Automatic parameter fitting failed — llama.cpp could not find a
    /// layer-offload allocation that fits within the supplied memory
    /// margins for the given context size.
    #[error("automatic parameter fitting failed: {0}")]
    Fit(String),

    /// Loading the GGUF model file failed (file not found, invalid format,
    /// out-of-memory during allocation, etc.).
    #[error("model load failed: {0}")]
    ModelLoad(String),

    /// Initialising the multimodal projector (mmproj) for a vision model
    /// failed. Only produced when the `mtmd` feature is enabled and a
    /// `mmproj_path` was supplied.
    #[cfg(feature = "mtmd")]
    #[error("multimodal projector init failed: {0}")]
    MmprojInit(String),

    /// The model path contained an interior NUL byte and could not be
    /// converted to a `CString` for the FFI call.
    #[error("invalid model path: {0}")]
    InvalidPath(String),

    /// The inference worker thread exited before it could report the result
    /// of initialisation. This usually means the worker panicked; check the
    /// process's stderr for a backtrace.
    #[error("inference worker thread exited during initialisation")]
    WorkerInitDisconnected,

    /// A reload was requested, but the inference worker thread is no longer
    /// accepting commands (it has either been shut down or panicked on a
    /// previous request).
    #[error("inference worker thread is no longer running")]
    WorkerNotRunning,
}
