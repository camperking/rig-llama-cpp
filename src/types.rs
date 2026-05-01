use std::collections::HashMap;

use rig::completion::{CompletionError, GetTokenUsage, Usage};
use rig::message::AssistantContent;
use rig::one_or_many::OneOrMany;
use rig::streaming::{RawStreamingChoice, RawStreamingToolCall};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

/// Raw completion response returned by the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawResponse {
    /// The full generated text.
    pub text: String,
}

/// A single chunk emitted during streaming inference.
///
/// The final chunk in a stream includes token usage counts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamChunk {
    /// The text fragment for this chunk.
    pub text: String,
    /// Number of prompt tokens (only set on the final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Number of completion tokens (only set on the final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Number of prompt tokens that were served from the persistent KV-cache prefix
    /// (only set on the final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
}

impl GetTokenUsage for StreamChunk {
    fn token_usage(&self) -> Option<Usage> {
        let (input, output) = self.prompt_tokens.zip(self.completion_tokens)?;
        Some(Usage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            cached_input_tokens: self.cached_input_tokens.unwrap_or(0),
            cache_creation_input_tokens: 0,
        })
    }
}

pub(crate) type StreamSender =
    mpsc::UnboundedSender<Result<RawStreamingChoice<StreamChunk>, CompletionError>>;

pub(crate) enum ResponseChannel {
    Completion(oneshot::Sender<Result<InferenceResult, String>>),
    Streaming(StreamSender),
}

pub(crate) enum InferenceCommand {
    Request(InferenceRequest),
    Reload(ReloadRequest),
    Shutdown,
}

pub(crate) struct ReloadRequest {
    pub model_path: String,
    pub mmproj_path: Option<String>,
    pub n_ctx: u32,
    pub fit_params: FitParams,
    pub kv_cache_params: KvCacheParams,
    pub checkpoint_params: CheckpointParams,
    pub result_tx: std::sync::mpsc::Sender<Result<(), String>>,
}

pub(crate) struct InferenceRequest {
    pub params: InferenceParams,
    pub response_channel: ResponseChannel,
}

pub(crate) struct InferenceParams {
    pub prepared_request: PreparedRequest,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
}

pub(crate) struct InferenceResult {
    pub text: String,
    pub choice: OneOrMany<AssistantContent>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Tokens of the prompt that were already present in the persistent KV cache
    /// (i.e. the longest common prefix shared with the previous request).
    pub cached_input_tokens: u64,
}

pub(crate) struct PreparedRequest {
    pub messages_json: String,
    pub tools_json: Option<String>,
    pub tool_choice: Option<String>,
    pub json_schema: Option<String>,
    pub enable_thinking: bool,
    #[cfg(feature = "mtmd")]
    pub images: Vec<PreparedImage>,
}

/// One image extracted from the chat history with its FNV-1a hash precomputed.
/// The hash is propagated into the underlying `MtmdBitmap` via `set_id` so
/// that `MtmdInputChunk::id()` round-trips it for the prefix-cache diff.
#[cfg(feature = "mtmd")]
#[derive(Clone, Debug)]
pub(crate) struct PreparedImage {
    pub bytes: Vec<u8>,
    pub hash: u64,
}

pub(crate) struct PromptBuildResult {
    pub prompt: String,
    pub template_result: Option<llama_cpp_2::model::ChatTemplateResult>,
}

/// Sampling parameters that control token generation.
///
/// Use `Default::default()` for reasonable starting values, then override
/// individual fields as needed.
///
/// ```
/// let params = rig_llama_cpp::SamplingParams {
///     top_k: 40,
///     presence_penalty: 1.5,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    /// Nucleus sampling threshold (default: `0.95`).
    pub top_p: f32,
    /// Top-k sampling parameter (default: `40`).
    pub top_k: i32,
    /// Minimum probability threshold (default: `0.0`).
    pub min_p: f32,
    /// Penalty for token presence (default: `0.0`).
    pub presence_penalty: f32,
    /// Penalty for token repetition (default: `1.0`).
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            top_p: 0.95,
            top_k: 40,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
        }
    }
}

/// Configuration for automatic GPU/CPU layer fitting.
///
/// When used with [`Client::from_gguf_with_fit`], llama.cpp will automatically
/// determine the optimal number of layers to offload to GPU based on available VRAM,
/// instead of requiring a manual `n_gpu_layers` value.
#[derive(Clone, Debug)]
pub struct FitParams {
    /// Memory margin per device in bytes. If `None`, defaults to 1 GiB per device.
    pub margins: Option<Vec<usize>>,
    /// Minimum context size to preserve during fitting (default: `4096`).
    pub n_ctx_min: u32,
}

impl Default for FitParams {
    fn default() -> Self {
        Self {
            margins: None,
            n_ctx_min: 4096,
        }
    }
}

/// Tunable parameters for the in-memory state-checkpoint cache used to
/// preserve KV/recurrent state across chat turns for hybrid models.
///
/// Hybrid architectures (Qwen 3.5, Jamba, etc.) interleave Mamba-style
/// recurrent layers with transformer layers. The recurrent state can't be
/// rolled back to an arbitrary earlier position, so a partial KV trim
/// fails whenever the next prompt diverges deep into the conversation.
/// To work around this, we periodically snapshot the partial seq state
/// (recurrent + SWA, via `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`) during
/// prompt prefill and restore the closest snapshot when the next prompt
/// arrives. Mirrors the mechanism used by upstream `llama-server`.
///
/// For non-hybrid models (Qwen 2.5, Llama 3, Gemma, ...) checkpoints are
/// created but never used because the cheaper partial-trim path
/// succeeds.
#[derive(Clone, Copy, Debug)]
pub struct CheckpointParams {
    /// Maximum number of checkpoints retained per persistent context.
    /// `0` disables checkpointing entirely. Each checkpoint is a few MB
    /// for typical hybrid models.
    pub max_checkpoints: u32,
    /// Approximate spacing between checkpoints during prompt prefill, in
    /// tokens. The last `4..=4 + n_ubatch` tokens always get a
    /// checkpoint regardless. `<= 0` means "only checkpoint near the end
    /// of the prompt".
    pub every_n_tokens: i32,
    /// Don't checkpoint the very start of a prompt — saves space for
    /// no benefit because we'd have to re-decode that prefix anyway if
    /// it's the entire reuse window.
    pub min_tokens: u32,
    /// Don't take two checkpoints closer than this many tokens apart.
    pub min_gap: u32,
}

impl Default for CheckpointParams {
    fn default() -> Self {
        Self {
            // llama-server uses 32; cap lower because each checkpoint is
            // a few MB and we'd rather not balloon RSS.
            max_checkpoints: 8,
            every_n_tokens: 8192,
            min_tokens: 64,
            min_gap: 64,
        }
    }
}

/// KV cache quantization configuration.
///
/// Controls the data type used for the attention K and V caches. llama.cpp defaults
/// both to `F16` (`GGML_TYPE_F16`), which is what `KvCacheParams::default()` preserves.
/// Quantizing the KV cache (e.g. `Q8_0` → ~½ size, `Q4_0` → ~¼ size) trades a small
/// amount of accuracy for a large reduction in VRAM usage, which is often the dominant
/// cost at long `n_ctx`.
///
/// ```
/// use rig_llama_cpp::{KvCacheParams, KvCacheType};
///
/// let kv = KvCacheParams {
///     type_k: KvCacheType::Q8_0,
///     type_v: KvCacheType::Q8_0,
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct KvCacheParams {
    /// Data type for the K cache (default: `KvCacheType::F16`).
    pub type_k: llama_cpp_2::context::params::KvCacheType,
    /// Data type for the V cache (default: `KvCacheType::F16`).
    pub type_v: llama_cpp_2::context::params::KvCacheType,
}

impl Default for KvCacheParams {
    fn default() -> Self {
        Self {
            type_k: llama_cpp_2::context::params::KvCacheType::F16,
            type_v: llama_cpp_2::context::params::KvCacheType::F16,
        }
    }
}

/// Result of building a sampler chain: the chain itself plus whether grammar is active.
///
/// When grammar is present, `llama_sampler_sample()` already calls `accept()` internally
/// and we must NOT call it again (double-accept corrupts grammar state). When grammar is
/// absent, we call `accept()` explicitly after `sample()` to preserve the legacy
/// double-accept behavior that the base samplers were tuned around.
pub(crate) struct SamplerChain {
    pub sampler: llama_cpp_2::sampling::LlamaSampler,
    pub has_grammar: bool,
}

pub(crate) struct StreamDeltaState {
    pub tool_calls: HashMap<u64, RawStreamingToolCall>,
}
