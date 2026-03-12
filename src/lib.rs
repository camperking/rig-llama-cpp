//! # rig-llama-cpp
//!
//! A [Rig](https://docs.rs/rig-core) completion provider that runs GGUF models locally
//! via [llama.cpp](https://github.com/ggml-org/llama.cpp), with optional Vulkan GPU acceleration.
//!
//! This crate implements Rig's [`CompletionModel`] trait so that any GGUF model can be used
//! as a drop-in replacement for cloud-based providers. It supports:
//!
//! - **Completion and streaming** — both one-shot and token-by-token responses.
//! - **Tool calling** — models with OpenAI-compatible chat templates can invoke tools.
//! - **Reasoning / thinking** — extended thinking output is forwarded when the model supports it.
//! - **Configurable sampling** — top-p, top-k, min-p, temperature, presence and repetition penalties.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use rig::client::CompletionClient;
//! use rig::completion::Prompt;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), anyhow::Error> {
//! let client = rig_llama_cpp::Client::from_gguf(
//!     "path/to/model.gguf",
//!     99,   // n_gpu_layers
//!     8192, // n_ctx
//!     0.95, // top_p
//!     20,   // top_k
//!     0.0,  // min_p
//!     1.5,  // presence_penalty
//!     1.0,  // repetition_penalty
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

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::thread;

use rig::client::CompletionClient;
use rig::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, GetTokenUsage, Usage,
};
use rig::message::{
    AssistantContent, Message, Reasoning, ToolCall, ToolChoice, ToolFunction, UserContent,
};
use rig::one_or_many::OneOrMany;
use rig::streaming::{
    RawStreamingChoice, RawStreamingToolCall, StreamingCompletionResponse, ToolCallDeltaContent,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

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
}

impl GetTokenUsage for StreamChunk {
    fn token_usage(&self) -> Option<Usage> {
        let (input, output) = self.prompt_tokens.zip(self.completion_tokens)?;
        Some(Usage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            cached_input_tokens: 0,
        })
    }
}

// === Internal types ===

type StreamSender = mpsc::UnboundedSender<Result<RawStreamingChoice<StreamChunk>, CompletionError>>;

enum ResponseChannel {
    Completion(oneshot::Sender<Result<InferenceResult, String>>),
    Streaming(StreamSender),
}

struct InferenceRequest {
    params: InferenceParams,
    response_channel: ResponseChannel,
}

struct InferenceParams {
    prepared_request: PreparedRequest,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_p: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
}

struct InferenceResult {
    text: String,
    choice: OneOrMany<AssistantContent>,
    prompt_tokens: u64,
    completion_tokens: u64,
}

struct PreparedRequest {
    messages_json: String,
    tools_json: Option<String>,
    tool_choice: Option<String>,
    json_schema: Option<String>,
    enable_thinking: bool,
}

struct PromptBuildResult {
    prompt: String,
    template_result: Option<llama_cpp_2::model::ChatTemplateResult>,
}

/// The llama.cpp completion client.
///
/// `Client` loads a GGUF model on a dedicated inference thread and exposes it
/// through Rig's [`CompletionClient`] trait. Create one with [`Client::from_gguf`].
pub struct Client {
    request_tx: mpsc::UnboundedSender<InferenceRequest>,
    sampling_params: SamplingParams,
}

#[derive(Clone, Copy)]
struct SamplingParams {
    top_p: f32,
    top_k: i32,
    min_p: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
}

impl Client {
    /// Load a GGUF model and start the inference worker thread.
    ///
    /// # Arguments
    ///
    /// * `model_path` — Path to a `.gguf` model file.
    /// * `n_gpu_layers` — Number of layers to offload to the GPU (`u32::MAX` for all).
    /// * `n_ctx` — Context window size in tokens.
    /// * `top_p` — Nucleus sampling threshold.
    /// * `top_k` — Top-k sampling parameter.
    /// * `min_p` — Minimum probability threshold.
    /// * `presence_penalty` — Penalty for token presence.
    /// * `repetition_penalty` — Penalty for token repetition.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to initialize or the model cannot be loaded.
    pub fn from_gguf(
        model_path: impl Into<String>,
        n_gpu_layers: u32,
        n_ctx: u32,
        top_p: f32,
        top_k: i32,
        min_p: f32,
        presence_penalty: f32,
        repetition_penalty: f32,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.into();
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<InferenceRequest>();
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let sampling_params = SamplingParams {
            top_p,
            top_k,
            min_p,
            presence_penalty,
            repetition_penalty,
        };

        thread::spawn(move || {
            inference_worker(&model_path, n_gpu_layers, n_ctx, init_tx, &mut request_rx);
        });

        init_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("Inference thread panicked during initialization"))?
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            request_tx,
            sampling_params,
        })
    }
}

impl CompletionClient for Client {
    type CompletionModel = Model;
}

/// A handle to a loaded model that implements Rig's [`CompletionModel`] trait.
///
/// Obtained via [`CompletionClient::agent`] on a [`Client`].
#[derive(Clone)]
pub struct Model {
    request_tx: mpsc::UnboundedSender<InferenceRequest>,
    sampling_params: SamplingParams,
    #[allow(dead_code)]
    model_id: String,
}

impl CompletionModel for Model {
    type Response = RawResponse;
    type StreamingResponse = StreamChunk;
    type Client = Client;

    fn make(client: &Client, model: impl Into<String>) -> Self {
        Self {
            request_tx: client.request_tx.clone(),
            sampling_params: client.sampling_params,
            model_id: model.into(),
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let prepared_request = prepare_request(&request).map_err(CompletionError::ProviderError)?;
        let max_tokens = request.max_tokens.unwrap_or(512) as u32;
        let temperature = request.temperature.unwrap_or(0.7) as f32;

        let (response_tx, response_rx) = oneshot::channel();

        self.request_tx
            .send(InferenceRequest {
                params: InferenceParams {
                    prepared_request,
                    max_tokens,
                    temperature,
                    top_p: self.sampling_params.top_p,
                    top_k: self.sampling_params.top_k,
                    min_p: self.sampling_params.min_p,
                    presence_penalty: self.sampling_params.presence_penalty,
                    repetition_penalty: self.sampling_params.repetition_penalty,
                },
                response_channel: ResponseChannel::Completion(response_tx),
            })
            .map_err(|_| CompletionError::ProviderError("Inference thread shut down".into()))?;

        let result = response_rx
            .await
            .map_err(|_| CompletionError::ProviderError("Response channel closed".into()))?
            .map_err(CompletionError::ProviderError)?;

        Ok(CompletionResponse {
            choice: result.choice,
            usage: Usage {
                input_tokens: result.prompt_tokens,
                output_tokens: result.completion_tokens,
                total_tokens: result.prompt_tokens + result.completion_tokens,
                cached_input_tokens: 0,
            },
            raw_response: RawResponse { text: result.text },
            message_id: None,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let prepared_request = prepare_request(&request).map_err(CompletionError::ProviderError)?;
        let max_tokens = request.max_tokens.unwrap_or(512) as u32;
        let temperature = request.temperature.unwrap_or(0.7) as f32;

        let (stream_tx, stream_rx) = mpsc::unbounded_channel();

        self.request_tx
            .send(InferenceRequest {
                params: InferenceParams {
                    prepared_request,
                    max_tokens,
                    temperature,
                    top_p: self.sampling_params.top_p,
                    top_k: self.sampling_params.top_k,
                    min_p: self.sampling_params.min_p,
                    presence_penalty: self.sampling_params.presence_penalty,
                    repetition_penalty: self.sampling_params.repetition_penalty,
                },
                response_channel: ResponseChannel::Streaming(stream_tx),
            })
            .map_err(|_| CompletionError::ProviderError("Inference thread shut down".into()))?;

        Ok(StreamingCompletionResponse::stream(Box::pin(
            UnboundedReceiverStream::new(stream_rx),
        )))
    }
}

// === Message extraction ===

fn prepare_request(request: &CompletionRequest) -> Result<PreparedRequest, String> {
    let mut messages = Vec::new();

    let mut system = request.preamble.clone().unwrap_or_default();
    if let Some(Message::User { content }) = request.normalized_documents() {
        let doc_text: String = content
            .iter()
            .filter_map(|c| match c {
                UserContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !doc_text.is_empty() {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&doc_text);
        }
    }

    if !system.is_empty() {
        messages.push(json!({
            "role": "system",
            "content": system,
        }));
    }

    for msg in request.chat_history.iter() {
        append_message_json(&mut messages, msg);
    }

    let tools_json = if request.tools.is_empty() {
        None
    } else {
        Some(
            serde_json::to_string(
                &request
                    .tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters,
                            }
                        })
                    })
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| format!("Tool serialization failed: {e}"))?,
        )
    };

    let tool_choice = match request.tool_choice.as_ref() {
        None => None,
        Some(ToolChoice::Auto) => Some("auto".to_string()),
        Some(ToolChoice::None) => Some("none".to_string()),
        Some(ToolChoice::Required) => Some("required".to_string()),
        Some(ToolChoice::Specific { .. }) => {
            return Err("Specific tool choice is not supported by local llama adapter".into());
        }
    };

    let json_schema = request
        .output_schema
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|e| format!("Schema serialization failed: {e}"))?;

    Ok(PreparedRequest {
        messages_json: serde_json::to_string(&messages)
            .map_err(|e| format!("Message serialization failed: {e}"))?,
        tools_json,
        tool_choice,
        json_schema,
        enable_thinking: request
            .additional_params
            .as_ref()
            .map(has_thinking_request)
            .unwrap_or(false),
    })
}

fn append_message_json(messages: &mut Vec<Value>, msg: &Message) {
    match msg {
        Message::User { content } => {
            let text = content
                .iter()
                .filter_map(user_content_text)
                .collect::<Vec<_>>()
                .join("\n");

            if !text.is_empty() {
                messages.push(json!({
                    "role": "user",
                    "content": text,
                }));
            }

            for tool_result in content.iter().filter_map(|c| match c {
                UserContent::ToolResult(tool_result) => Some(tool_result),
                _ => None,
            }) {
                let content = tool_result
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        rig::message::ToolResultContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_result.call_id.as_deref().unwrap_or(&tool_result.id),
                    "content": content,
                }));
            }
        }
        Message::Assistant { content, .. } => {
            let text = content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            let tool_calls = content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call),
                    _ => None,
                })
                .map(tool_call_json)
                .collect::<Vec<_>>();

            if !text.is_empty() || !tool_calls.is_empty() {
                messages.push(json!({
                    "role": "assistant",
                    "content": if text.is_empty() { Value::Null } else { Value::String(text) },
                    "tool_calls": if tool_calls.is_empty() { Value::Null } else { Value::Array(tool_calls) },
                }));
            }
        }
    }
}

fn user_content_text(content: &UserContent) -> Option<String> {
    match content {
        UserContent::Text(text) => Some(text.text.clone()),
        UserContent::Document(document) => Some(document_text(document)),
        _ => None,
    }
}

fn document_text(document: &rig::message::Document) -> String {
    match &document.data {
        rig::message::DocumentSourceKind::String(text)
        | rig::message::DocumentSourceKind::Url(text)
        | rig::message::DocumentSourceKind::Base64(text) => text.clone(),
        rig::message::DocumentSourceKind::Raw(bytes) => String::from_utf8_lossy(bytes).into_owned(),
        rig::message::DocumentSourceKind::Unknown => String::new(),
        _ => String::new(),
    }
}

fn tool_call_json(tool_call: &ToolCall) -> Value {
    json!({
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments.to_string(),
        }
    })
}

fn has_thinking_request(params: &Value) -> bool {
    // check actual value of reasoning/thinking param if present
    if let Some(reasoning) = params.get("reasoning").or_else(|| params.get("thinking")) {
        if let Some(enabled) = reasoning.as_bool() {
            return enabled;
        }
    }

    return false;
}

// === Inference worker (runs on dedicated thread) ===

fn inference_worker(
    model_path: &str,
    n_gpu_layers: u32,
    n_ctx: u32,
    init_tx: std::sync::mpsc::Sender<Result<(), String>>,
    rx: &mut mpsc::UnboundedReceiver<InferenceRequest>,
) {
    use llama_cpp_2::list_llama_ggml_backend_devices;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::model::LlamaModel as LlamaCppModel;
    use llama_cpp_2::model::params::LlamaModelParams;

    let backend = match LlamaBackend::init() {
        Ok(b) => b,
        Err(e) => {
            let _ = init_tx.send(Err(format!("Backend init failed: {e}")));
            return;
        }
    };

    let mut model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

    if backend.supports_gpu_offload() {
        let vulkan_devices: Vec<usize> = list_llama_ggml_backend_devices()
            .into_iter()
            .filter(|device| device.backend.eq_ignore_ascii_case("vulkan"))
            .map(|device| device.index)
            .collect();

        if !vulkan_devices.is_empty() {
            model_params = match model_params.with_devices(&vulkan_devices) {
                Ok(params) => {
                    eprintln!("Using Vulkan backend devices: {vulkan_devices:?}");
                    params
                }
                Err(e) => {
                    let _ = init_tx.send(Err(format!("Failed to configure Vulkan devices: {e}")));
                    return;
                }
            };
        }
    }

    eprintln!("Loading model from {model_path}...");

    let model = match LlamaCppModel::load_from_file(&backend, model_path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            let _ = init_tx.send(Err(format!("Model load failed: {e}")));
            return;
        }
    };
    eprintln!("Model loaded.");

    // Signal successful initialization
    let _ = init_tx.send(Ok(()));

    // Process inference requests
    while let Some(req) = rx.blocking_recv() {
        let InferenceRequest {
            params,
            response_channel,
        } = req;
        match response_channel {
            ResponseChannel::Completion(tx) => {
                let result = run_inference(&backend, &model, n_ctx, &params, None);
                let _ = tx.send(result);
            }
            ResponseChannel::Streaming(stream_tx) => {
                let result = run_inference(&backend, &model, n_ctx, &params, Some(&stream_tx));
                match result {
                    Ok(result) => {
                        let _ =
                            stream_tx.send(Ok(RawStreamingChoice::FinalResponse(StreamChunk {
                                text: result.text,
                                prompt_tokens: Some(result.prompt_tokens),
                                completion_tokens: Some(result.completion_tokens),
                            })));
                    }
                    Err(e) => {
                        let _ = stream_tx.send(Err(CompletionError::ProviderError(e)));
                    }
                }
            }
        }
    }
}

fn run_inference(
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    model: &llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
) -> Result<InferenceResult, String> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::AddBos;
    use llama_cpp_2::sampling::LlamaSampler;

    let prompt_build = build_prompt(model, &req.prepared_request)?;
    let prompt = prompt_build.prompt.as_str();

    let ctx_params =
        LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx).map(Some).unwrap_or(None));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("Context creation failed: {e}"))?;

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| format!("Tokenization failed: {e}"))?;
    let prompt_tokens = tokens.len() as u64;

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let last_index = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        batch
            .add(token, i, &[0], i == last_index)
            .map_err(|e| format!("Batch add failed: {e}"))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| format!("Prompt decode failed: {e}"))?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::top_k(req.top_k),
        LlamaSampler::top_p(req.top_p, 1),
        LlamaSampler::min_p(req.min_p, 1),
        LlamaSampler::temp(req.temperature),
        LlamaSampler::penalties(-1, req.repetition_penalty, 0.0, req.presence_penalty),
        LlamaSampler::dist(42),
    ]);

    let mut output = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = batch.n_tokens();
    let mut completion_tokens = 0u64;

    // Initialize streaming parser if streaming and we have a template result
    let mut stream_parser = if stream_tx.is_some() {
        prompt_build
            .template_result
            .as_ref()
            .and_then(|tr| tr.streaming_state_oaicompat().ok())
    } else {
        None
    };
    let mut delta_state = StreamDeltaState::new();

    for _ in 0..req.max_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, false, None)
            .map_err(|e| format!("Token to piece failed: {e}"))?;
        output.push_str(&piece);
        completion_tokens += 1;

        if let Some(tx) = stream_tx {
            if let Some(parser) = stream_parser.as_mut() {
                match parser.update(&piece, true) {
                    Ok(deltas) => {
                        for delta_json in deltas {
                            for choice in delta_state.parse_delta(&delta_json) {
                                let _ = tx.send(Ok(choice));
                            }
                        }
                    }
                    Err(_) => {
                        let _ = tx.send(Ok(RawStreamingChoice::Message(piece)));
                    }
                }
            } else {
                let _ = tx.send(Ok(RawStreamingChoice::Message(piece)));
            }
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| format!("Batch add failed: {e}"))?;
        ctx.decode(&mut batch)
            .map_err(|e| format!("Decode failed: {e}"))?;
        n_cur += 1;
    }

    // Flush remaining deltas from the streaming parser
    if let Some(tx) = stream_tx {
        if let Some(parser) = stream_parser.as_mut() {
            if let Ok(deltas) = parser.update("", false) {
                for delta_json in deltas {
                    for choice in delta_state.parse_delta(&delta_json) {
                        let _ = tx.send(Ok(choice));
                    }
                }
            }
        }
        // Emit complete tool calls so they get accumulated into assistant_items
        for choice in delta_state.flush_tool_calls() {
            let _ = tx.send(Ok(choice));
        }
    }

    let choice = if stream_tx.is_some() {
        // For streaming, choice was already sent through the stream;
        // return a minimal placeholder for InferenceResult
        OneOrMany::one(AssistantContent::text(output.clone()))
    } else {
        parse_completion_output(&output, prompt_build.template_result.as_ref())?
    };

    Ok(InferenceResult {
        text: output,
        choice,
        prompt_tokens,
        completion_tokens,
    })
}

// === Streaming delta parser ===

struct StreamDeltaState {
    tool_calls: HashMap<u64, RawStreamingToolCall>,
}

impl StreamDeltaState {
    fn new() -> Self {
        Self {
            tool_calls: HashMap::new(),
        }
    }

    fn parse_delta(&mut self, delta_json: &str) -> Vec<RawStreamingChoice<StreamChunk>> {
        let mut choices = Vec::new();
        let Ok(value) = serde_json::from_str::<Value>(delta_json) else {
            return choices;
        };
        let Some(obj) = value.as_object() else {
            return choices;
        };

        if let Some(content) = obj.get("content").and_then(Value::as_str) {
            if !content.is_empty() {
                choices.push(RawStreamingChoice::Message(content.to_string()));
            }
        }

        if let Some(reasoning) = obj.get("reasoning_content").and_then(Value::as_str) {
            if !reasoning.is_empty() {
                choices.push(RawStreamingChoice::ReasoningDelta {
                    id: None,
                    reasoning: reasoning.to_string(),
                });
            }
        }

        if let Some(tool_calls) = obj.get("tool_calls").and_then(Value::as_array) {
            for tc in tool_calls {
                let index = tc.get("index").and_then(Value::as_u64).unwrap_or(0);

                // Get or create the accumulated tool call entry.
                // RawStreamingToolCall::empty() generates a unique internal_call_id via nanoid.
                let existing = self
                    .tool_calls
                    .entry(index)
                    .or_insert_with(RawStreamingToolCall::empty);

                // First delta carries the provider-supplied id
                if let Some(id) = tc.get("id").and_then(Value::as_str) {
                    if !id.is_empty() {
                        existing.id = id.to_string();
                    }
                }

                if let Some(function) = tc.get("function").and_then(Value::as_object) {
                    if let Some(name) = function.get("name").and_then(Value::as_str) {
                        if !name.is_empty() {
                            existing.name = name.to_string();

                            choices.push(RawStreamingChoice::ToolCallDelta {
                                id: existing.id.clone(),
                                internal_call_id: existing.internal_call_id.clone(),
                                content: ToolCallDeltaContent::Name(name.to_string()),
                            });
                        }
                    }
                    if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                        if !arguments.is_empty() {
                            // Accumulate arguments like the OpenAI implementation
                            let current_args = match &existing.arguments {
                                Value::Null => String::new(),
                                Value::String(s) => s.clone(),
                                v => v.to_string(),
                            };
                            let combined = format!("{current_args}{arguments}");
                            if combined.trim_start().starts_with('{')
                                && combined.trim_end().ends_with('}')
                            {
                                match serde_json::from_str(&combined) {
                                    Ok(parsed) => existing.arguments = parsed,
                                    Err(_) => existing.arguments = Value::String(combined),
                                }
                            } else {
                                existing.arguments = Value::String(combined);
                            }

                            choices.push(RawStreamingChoice::ToolCallDelta {
                                id: existing.id.clone(),
                                internal_call_id: existing.internal_call_id.clone(),
                                content: ToolCallDeltaContent::Delta(arguments.to_string()),
                            });
                        }
                    }
                }
            }
        }

        choices
    }

    /// Flush all accumulated tool calls as complete RawStreamingChoice::ToolCall events.
    fn flush_tool_calls(&mut self) -> Vec<RawStreamingChoice<StreamChunk>> {
        self.tool_calls
            .drain()
            .filter(|(_, tc)| !tc.name.is_empty())
            .map(|(_, tool_call)| RawStreamingChoice::ToolCall(tool_call))
            .collect()
    }
}

fn build_prompt(
    model: &llama_cpp_2::model::LlamaModel,
    request: &PreparedRequest,
) -> Result<PromptBuildResult, String> {
    use llama_cpp_2::model::LlamaChatMessage;
    use llama_cpp_2::openai::OpenAIChatTemplateParams;

    let chat_template_kwargs = json!({ "enable_thinking": request.enable_thinking }).to_string();

    if let Ok(tmpl) = model.chat_template(None) {
        let params = OpenAIChatTemplateParams {
            messages_json: &request.messages_json,
            tools_json: request.tools_json.as_deref(),
            tool_choice: request.tool_choice.as_deref(),
            json_schema: request.json_schema.as_deref(),
            grammar: None,
            reasoning_format: Some("auto"),
            chat_template_kwargs: Some(&chat_template_kwargs),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: request.enable_thinking,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: request.tools_json.is_some(),
        };

        if let Ok(result) = model.apply_chat_template_oaicompat(&tmpl, &params) {
            return Ok(PromptBuildResult {
                prompt: result.prompt.clone(),
                template_result: Some(result),
            });
        }
    }

    let parsed_messages: Vec<(String, String)> =
        serde_json::from_str::<Vec<Value>>(&request.messages_json)
            .map_err(|e| format!("Message deserialization failed: {e}"))?
            .into_iter()
            .filter_map(|msg| {
                Some((
                    msg.get("role")?.as_str()?.to_string(),
                    message_content_as_text(&msg).to_string(),
                ))
            })
            .collect();

    let chat_msgs: Vec<LlamaChatMessage> = parsed_messages
        .iter()
        .map(|(role, content)| LlamaChatMessage::new(role.clone(), content.clone()))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("Chat message creation failed: {e}"))?;

    // Try model's built-in chat template first
    if let Ok(tmpl) = model.chat_template(None) {
        if let Ok(prompt) = model.apply_chat_template(&tmpl, &chat_msgs, true) {
            return Ok(PromptBuildResult {
                prompt,
                template_result: None,
            });
        }
    }

    // Fallback to ChatML format
    let mut prompt = String::new();
    for (role, content) in &parsed_messages {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(PromptBuildResult {
        prompt,
        template_result: None,
    })
}

fn message_content_as_text(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn parse_completion_output(
    raw_text: &str,
    template_result: Option<&llama_cpp_2::model::ChatTemplateResult>,
) -> Result<OneOrMany<AssistantContent>, String> {
    if let Some(template_result) = template_result {
        match template_result.parse_response_oaicompat(raw_text, false) {
            Ok(parsed_json) => {
                if let Ok(choice) = parse_oaicompat_message(&parsed_json, raw_text) {
                    return Ok(choice);
                }
            }
            Err(err) => {
                eprintln!("Failed to parse llama response as OpenAI-compatible content: {err}");
            }
        }
    }

    Ok(OneOrMany::one(AssistantContent::text(raw_text.to_string())))
}

fn parse_oaicompat_message(
    parsed_json: &str,
    raw_text: &str,
) -> Result<OneOrMany<AssistantContent>, String> {
    let value: Value = serde_json::from_str(parsed_json)
        .map_err(|e| format!("Parsed response JSON deserialization failed: {e}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Parsed response is not a JSON object".to_string())?;

    let mut content = Vec::new();

    if let Some(reasoning) = object
        .get("reasoning_content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        content.push(AssistantContent::Reasoning(Reasoning::new(reasoning)));
    }

    let text = extract_text_content(object.get("content"));
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        content.push(AssistantContent::text(text));
    }

    if let Some(tool_calls) = object.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            content.push(AssistantContent::ToolCall(parse_tool_call(tool_call)?));
        }
    }

    if content.is_empty() {
        content.push(AssistantContent::text(raw_text.to_string()));
    }

    OneOrMany::many(content).map_err(|_| "Parsed response produced no content".to_string())
}

fn extract_text_content(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(Value::as_str)
                        .or_else(|| part.get("refusal").and_then(Value::as_str))
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(text)
        }
        _ => None,
    }
}

fn parse_tool_call(value: &Value) -> Result<ToolCall, String> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| "Tool call is missing id".to_string())?
        .to_string();
    let function = value
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| "Tool call is missing function".to_string())?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "Tool call function is missing name".to_string())?
        .to_string();
    let arguments = match function.get("arguments") {
        Some(Value::String(arguments)) => {
            serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.clone()))
        }
        Some(other) => other.clone(),
        None => Value::Null,
    };

    Ok(ToolCall::new(id, ToolFunction::new(name, arguments)))
}
