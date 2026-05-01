use std::collections::HashMap;

use rig::completion::{CompletionError, GetTokenUsage, Usage};
use rig::message::AssistantContent;
use rig::one_or_many::OneOrMany;
use rig::streaming::{RawStreamingChoice, RawStreamingToolCall};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

/// Raw completion response returned by the model.
///
/// Marked `#[non_exhaustive]` because new fields may be added in future
/// minor releases.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RawResponse {
    /// The full generated text.
    pub text: String,
}

/// A single chunk emitted during streaming inference.
///
/// The final chunk in a stream includes token usage counts. Marked
/// `#[non_exhaustive]` because new fields may be added in future minor
/// releases.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
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
    pub result_tx: std::sync::mpsc::Sender<Result<(), crate::error::LoadError>>,
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
/// Marked `#[non_exhaustive]` so future sampling knobs can be added without
/// a breaking release. Start from [`SamplingParams::default`] and chain
/// `with_*` setters:
///
/// ```
/// let params = rig_llama_cpp::SamplingParams::default()
///     .with_top_k(40)
///     .with_presence_penalty(1.5);
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
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

impl SamplingParams {
    /// Set the nucleus sampling threshold.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set the top-k sampling parameter.
    #[must_use]
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the minimum probability threshold.
    #[must_use]
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// Set the presence penalty.
    #[must_use]
    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = presence_penalty;
        self
    }

    /// Set the repetition penalty.
    #[must_use]
    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }
}

/// Configuration for automatic GPU/CPU layer fitting.
///
/// Passed to [`crate::Client::builder`] (or [`crate::Client::from_gguf`]) so
/// llama.cpp can probe available device memory and pick the optimal number
/// of layers to offload to GPU automatically, instead of requiring a manual
/// `n_gpu_layers` value.
///
/// Marked `#[non_exhaustive]`; build via `Default::default()` and chain the
/// `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
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

impl FitParams {
    /// Override the per-device memory margin in bytes.
    #[must_use]
    pub fn with_margins(mut self, margins: Option<Vec<usize>>) -> Self {
        self.margins = margins;
        self
    }

    /// Override the minimum context size to preserve during fitting.
    #[must_use]
    pub fn with_n_ctx_min(mut self, n_ctx_min: u32) -> Self {
        self.n_ctx_min = n_ctx_min;
        self
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
///
/// Marked `#[non_exhaustive]`; build via `Default::default()` and chain the
/// `with_*` setters.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
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

impl CheckpointParams {
    /// Override the maximum number of checkpoints retained per context.
    #[must_use]
    pub fn with_max_checkpoints(mut self, max_checkpoints: u32) -> Self {
        self.max_checkpoints = max_checkpoints;
        self
    }

    /// Override the approximate spacing between checkpoints (in tokens).
    #[must_use]
    pub fn with_every_n_tokens(mut self, every_n_tokens: i32) -> Self {
        self.every_n_tokens = every_n_tokens;
        self
    }

    /// Override the minimum prompt length before checkpoints are taken.
    #[must_use]
    pub fn with_min_tokens(mut self, min_tokens: u32) -> Self {
        self.min_tokens = min_tokens;
        self
    }

    /// Override the minimum spacing between two consecutive checkpoints.
    #[must_use]
    pub fn with_min_gap(mut self, min_gap: u32) -> Self {
        self.min_gap = min_gap;
        self
    }
}

/// Data type used for an entry in the attention KV cache.
///
/// Mirrors the subset of `ggml_type` values that `llama.cpp` accepts as KV
/// cache element types. The `F16` default preserves full attention quality;
/// quantizing (e.g. `Q8_0` ≈ ½ size, `Q4_0` ≈ ¼ size) trades a small amount
/// of accuracy for a large VRAM reduction at long `n_ctx`.
///
/// This is a local shim around `llama_cpp_2::context::params::KvCacheType`
/// so a future `llama-cpp-2` update doesn't force a breaking release of
/// `rig-llama-cpp`. Marked `#[non_exhaustive]`: when llama.cpp adds a new
/// `ggml_type`, we add a corresponding variant in a minor (`0.1.x`) release.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
#[non_exhaustive]
pub enum KvCacheType {
    /// IEEE 754 single precision.
    F32,
    /// IEEE 754 half precision (llama.cpp's default for both K and V).
    F16,
    /// Brain floating-point 16, common on newer NVIDIA / AMD GPUs.
    BF16,
    /// IEEE 754 double precision.
    F64,
    /// 4-bit block quantization, type 0.
    Q4_0,
    /// 4-bit block quantization, type 1.
    Q4_1,
    /// 5-bit block quantization, type 0.
    Q5_0,
    /// 5-bit block quantization, type 1.
    Q5_1,
    /// 8-bit block quantization, type 0.
    Q8_0,
    /// 8-bit block quantization, type 1.
    Q8_1,
    /// 2-bit K-quant.
    Q2_K,
    /// 3-bit K-quant.
    Q3_K,
    /// 4-bit K-quant.
    Q4_K,
    /// 5-bit K-quant.
    Q5_K,
    /// 6-bit K-quant.
    Q6_K,
    /// 8-bit K-quant.
    Q8_K,
    /// Importance-weighted 2-bit, extra-extra-small.
    IQ2_XXS,
    /// Importance-weighted 2-bit, extra-small.
    IQ2_XS,
    /// Importance-weighted 2-bit, small.
    IQ2_S,
    /// Importance-weighted 3-bit, extra-extra-small.
    IQ3_XXS,
    /// Importance-weighted 3-bit, small.
    IQ3_S,
    /// Importance-weighted 1-bit, small.
    IQ1_S,
    /// Importance-weighted 1-bit, medium.
    IQ1_M,
    /// Importance-weighted 4-bit, extra-small.
    IQ4_XS,
    /// Importance-weighted 4-bit, non-linear.
    IQ4_NL,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Ternary 1-bit, type 0.
    TQ1_0,
    /// Ternary 2-bit, type 0.
    TQ2_0,
    /// Microscaling FP4.
    MXFP4,
}

impl From<KvCacheType> for llama_cpp_2::context::params::KvCacheType {
    fn from(value: KvCacheType) -> Self {
        use llama_cpp_2::context::params::KvCacheType as Upstream;
        match value {
            KvCacheType::F32 => Upstream::F32,
            KvCacheType::F16 => Upstream::F16,
            KvCacheType::BF16 => Upstream::BF16,
            KvCacheType::F64 => Upstream::F64,
            KvCacheType::Q4_0 => Upstream::Q4_0,
            KvCacheType::Q4_1 => Upstream::Q4_1,
            KvCacheType::Q5_0 => Upstream::Q5_0,
            KvCacheType::Q5_1 => Upstream::Q5_1,
            KvCacheType::Q8_0 => Upstream::Q8_0,
            KvCacheType::Q8_1 => Upstream::Q8_1,
            KvCacheType::Q2_K => Upstream::Q2_K,
            KvCacheType::Q3_K => Upstream::Q3_K,
            KvCacheType::Q4_K => Upstream::Q4_K,
            KvCacheType::Q5_K => Upstream::Q5_K,
            KvCacheType::Q6_K => Upstream::Q6_K,
            KvCacheType::Q8_K => Upstream::Q8_K,
            KvCacheType::IQ2_XXS => Upstream::IQ2_XXS,
            KvCacheType::IQ2_XS => Upstream::IQ2_XS,
            KvCacheType::IQ2_S => Upstream::IQ2_S,
            KvCacheType::IQ3_XXS => Upstream::IQ3_XXS,
            KvCacheType::IQ3_S => Upstream::IQ3_S,
            KvCacheType::IQ1_S => Upstream::IQ1_S,
            KvCacheType::IQ1_M => Upstream::IQ1_M,
            KvCacheType::IQ4_XS => Upstream::IQ4_XS,
            KvCacheType::IQ4_NL => Upstream::IQ4_NL,
            KvCacheType::I8 => Upstream::I8,
            KvCacheType::I16 => Upstream::I16,
            KvCacheType::I32 => Upstream::I32,
            KvCacheType::I64 => Upstream::I64,
            KvCacheType::TQ1_0 => Upstream::TQ1_0,
            KvCacheType::TQ2_0 => Upstream::TQ2_0,
            KvCacheType::MXFP4 => Upstream::MXFP4,
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
/// Marked `#[non_exhaustive]`; build via `Default::default()` and chain the
/// `with_*` setters:
///
/// ```
/// use rig_llama_cpp::{KvCacheParams, KvCacheType};
///
/// let kv = KvCacheParams::default()
///     .with_type_k(KvCacheType::Q8_0)
///     .with_type_v(KvCacheType::Q8_0);
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct KvCacheParams {
    /// Data type for the K cache (default: [`KvCacheType::F16`]).
    pub type_k: KvCacheType,
    /// Data type for the V cache (default: [`KvCacheType::F16`]).
    pub type_v: KvCacheType,
}

impl Default for KvCacheParams {
    fn default() -> Self {
        Self {
            type_k: KvCacheType::F16,
            type_v: KvCacheType::F16,
        }
    }
}

impl KvCacheParams {
    /// Override the K cache data type.
    #[must_use]
    pub fn with_type_k(mut self, type_k: KvCacheType) -> Self {
        self.type_k = type_k;
        self
    }

    /// Override the V cache data type.
    #[must_use]
    pub fn with_type_v(mut self, type_v: KvCacheType) -> Self {
        self.type_v = type_v;
        self
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
