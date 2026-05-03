//! Shared helpers for the e2e integration tests.
//!
//! Models are downloaded via `hf-hub` into the standard HuggingFace cache
//! (`~/.cache/huggingface/hub`) on first use; subsequent runs are
//! cache-hits. Tests stay `#[ignore]` so `cargo test --lib` and the
//! default `cargo test` invocation never trigger downloads.

#![allow(dead_code)] // each test file uses a subset; suppress per-target dead_code warnings

use std::fmt;
use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, ensure};
use hf_hub::api::sync::Api;
use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, GetTokenUsage, ToolDefinition};
use rig::message::{AssistantContent, Message, ToolChoice, ToolResultContent, UserContent};
use rig::streaming::{StreamedAssistantContent, StreamingChat};
use rig_llama_cpp::{CheckpointParams, Client, FitParams, KvCacheParams, Model, SamplingParams};
use serde_json::json;
use tokio_stream::StreamExt;

// ── Model registry ────────────────────────────────────────────────────

pub struct ModelSpec {
    pub repo: &'static str,
    pub file: &'static str,
}

pub const QWEN: ModelSpec = ModelSpec {
    repo: "unsloth/Qwen3.5-2B-GGUF",
    file: "Qwen3.5-2B-Q4_K_M.gguf",
};

pub const QWEN_MMPROJ: ModelSpec = ModelSpec {
    repo: "unsloth/Qwen3.5-2B-GGUF",
    file: "mmproj-BF16.gguf",
};

pub const GEMMA: ModelSpec = ModelSpec {
    repo: "unsloth/gemma-4-E4B-it-GGUF",
    file: "gemma-4-E4B-it-Q4_K_M.gguf",
};

pub const GEMMA_MMPROJ: ModelSpec = ModelSpec {
    repo: "unsloth/gemma-4-E4B-it-GGUF",
    file: "mmproj-BF16.gguf",
};

pub const NOMIC_EMBED: ModelSpec = ModelSpec {
    repo: "nomic-ai/nomic-embed-text-v2-moe-GGUF",
    file: "nomic-embed-text-v2-moe.Q4_K_M.gguf",
};

/// Image fixture used by the vision tests; checked into the repo.
pub fn test_image_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test.jpg")
}

/// Resolve a `ModelSpec` to a local path, downloading via `hf-hub` if not
/// already cached. The HF API client is initialised once per process.
pub fn ensure_model(spec: &ModelSpec) -> anyhow::Result<PathBuf> {
    static API: OnceLock<Api> = OnceLock::new();
    let api = API.get_or_init(|| Api::new().expect("hf-hub Api init"));
    let path = api
        .model(spec.repo.to_string())
        .get(spec.file)
        .with_context(|| format!("downloading {}/{}", spec.repo, spec.file))?;
    Ok(path)
}

// ── Env-var overrides (kept for the long e2e test; defaults are sane) ─

pub fn env_parse_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

pub fn env_parse_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

pub fn env_parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

// ── Long-form e2e infra: corpus prompts + run summary ────────────────

#[derive(Debug, Default)]
pub struct RunSummary {
    pub total_output_tokens: u64,
    pub completion_turns: usize,
    pub streaming_turns: usize,
    pub streamed_text_chunks: usize,
    pub conversation_messages: usize,
    pub tool_call_observed: bool,
    pub tool_roundtrip_completed: bool,
}

impl fmt::Display for RunSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RunSummary {{ total_output_tokens: {}, completion_turns: {}, streaming_turns: {}, streamed_text_chunks: {}, conversation_messages: {}, tool_call_observed: {}, tool_roundtrip_completed: {} }}",
            self.total_output_tokens,
            self.completion_turns,
            self.streaming_turns,
            self.streamed_text_chunks,
            self.conversation_messages,
            self.tool_call_observed,
            self.tool_roundtrip_completed,
        )
    }
}

pub fn corpus_preamble() -> String {
    "You are generating a deterministic corpus for local inference validation. Respond only with numbered lines in the form 'NNNN: sentence'. Each sentence must be between 14 and 20 words, describe a distinct LLM testing scenario, and avoid markdown or extra commentary.".to_string()
}

pub fn corpus_prompt(start: usize, end: usize) -> String {
    format!(
        "Continue the corpus with lines {start:04} through {end:04}. Keep the numbering contiguous, output one line per item, and stop exactly after line {end:04}."
    )
}

pub fn seed_history() -> Vec<Message> {
    [
        (
            "We are preparing a validation transcript for a local GGUF model.",
            "I will keep the transcript concise and preserve continuity across turns.",
        ),
        (
            "The transcript must later expand into long-form output for token accounting.",
            "Understood. I will be ready to continue into a large deterministic corpus.",
        ),
        (
            "Keep the earlier turns short so the context budget is available for generation.",
            "I will keep setup turns brief and reserve context for longer completions.",
        ),
        (
            "We also need coverage for streaming and regular completion paths.",
            "Both modes can be exercised while maintaining the same conversation history.",
        ),
        (
            "Function calling should be probed separately if the model template supports it.",
            "I can attempt a tool call and then continue after a synthetic tool result.",
        ),
        (
            "The final validation target is at least ten thousand output tokens.",
            "That target can be reached across several long continuation turns.",
        ),
        (
            "Make the long-form output easy to inspect when the run is captured.",
            "Numbered lines provide a simple way to audit continuity and truncation.",
        ),
        (
            "We need a conversation with at least twenty-four messages overall.",
            "The seeded transcript plus generation turns will satisfy that requirement.",
        ),
        (
            "Avoid markdown wrappers once the numbered corpus starts.",
            "I will output plain text lines only.",
        ),
        (
            "The conversation is ready; switch to corpus mode on the next turn.",
            "Ready to continue the corpus when prompted.",
        ),
    ]
    .into_iter()
    .flat_map(|(user, assistant)| [Message::user(user), Message::assistant(assistant)])
    .collect()
}

pub fn assistant_text(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            AssistantContent::Reasoning(reasoning) => Some(reasoning.display_text()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Generic loaders ──────────────────────────────────────────────────

pub fn load_default(path: &std::path::Path) -> anyhow::Result<(Client, Model)> {
    ensure!(path.is_file(), "model file not found at {}", path.display());
    let client = Client::from_gguf(
        path.to_string_lossy().into_owned(),
        env_parse_u32("N_CTX", 8192),
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams::default(),
        CheckpointParams::default(),
    )?;
    let model = client.completion_model("local");
    Ok((client, model))
}

// ── Long-form streaming/completion turn runners ──────────────────────

pub async fn run_completion_turn(
    model: &Model,
    history: &mut Vec<Message>,
    prompt: String,
    preamble: &str,
    max_tokens: u64,
    temperature: f64,
    summary: &mut RunSummary,
) -> anyhow::Result<()> {
    let response = model
        .completion_request(prompt.clone())
        .preamble(preamble.to_owned())
        .messages(history.clone())
        .max_tokens(max_tokens)
        .temperature(temperature)
        .send()
        .await?;

    ensure!(
        !response.raw_response.text.trim().is_empty(),
        "completion turn returned empty text"
    );
    ensure!(
        response.usage.output_tokens > 0,
        "completion turn returned zero output tokens"
    );

    history.push(Message::user(prompt));
    history.push(response.choice.clone().into());

    summary.total_output_tokens += response.usage.output_tokens;
    summary.completion_turns += 1;
    summary.conversation_messages = history.len();

    Ok(())
}

pub async fn run_streaming_turn(
    model: &Model,
    history: &mut Vec<Message>,
    prompt: String,
    preamble: &str,
    max_tokens: u64,
    temperature: f64,
    summary: &mut RunSummary,
) -> anyhow::Result<()> {
    let mut stream = model
        .completion_request(prompt.clone())
        .preamble(preamble.to_owned())
        .messages(history.clone())
        .max_tokens(max_tokens)
        .temperature(temperature)
        .stream()
        .await?;

    let mut saw_text_chunk = false;

    while let Some(item) = stream.next().await {
        match item? {
            StreamedAssistantContent::Text(text) => {
                if !text.text.is_empty() {
                    saw_text_chunk = true;
                    summary.streamed_text_chunks += 1;
                }
            }
            StreamedAssistantContent::Reasoning(_) => {}
            StreamedAssistantContent::ReasoningDelta { .. } => {}
            StreamedAssistantContent::ToolCall { .. } => {}
            StreamedAssistantContent::ToolCallDelta { .. } => {}
            StreamedAssistantContent::Final(_) => {}
        }
    }

    let final_chunk = stream
        .response
        .clone()
        .context("stream did not surface a final response chunk")?;
    let usage = final_chunk
        .token_usage()
        .context("stream final response did not include token usage")?;
    let aggregated_text = assistant_text(&stream.choice);

    ensure!(saw_text_chunk, "streaming turn emitted no text chunks");
    ensure!(
        !aggregated_text.trim().is_empty(),
        "streaming turn aggregated no assistant text"
    );
    ensure!(
        usage.output_tokens > 0,
        "streaming turn returned zero output tokens"
    );

    history.push(Message::user(prompt));
    history.push(stream.choice.clone().into());

    summary.total_output_tokens += usage.output_tokens;
    summary.streaming_turns += 1;
    summary.conversation_messages = history.len();

    Ok(())
}

pub async fn attempt_tool_call(model: &Model, summary: &mut RunSummary) -> anyhow::Result<()> {
    let tool = ToolDefinition {
        name: "get_time".to_string(),
        description: "Return the current UTC time as plain text.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        }),
    };

    let prompt = "What time is it right now? You must call get_time before giving a final answer.";

    let response = model
        .completion_request(prompt)
        .preamble("You are validating function calling. When a tool is required, emit the tool call first.".to_string())
        .tool(tool)
        .tool_choice(ToolChoice::Required)
        .max_tokens(256)
        .temperature(0.0)
        .send()
        .await?;

    let maybe_tool_call = response.choice.iter().find_map(|content| match content {
        AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
        _ => None,
    });

    let Some(tool_call) = maybe_tool_call else {
        eprintln!(
            "Tool calling was attempted but the model returned no tool call: {}",
            response.raw_response.text.trim()
        );
        return Ok(());
    };

    summary.tool_call_observed = true;

    let tool_result = Message::from(UserContent::tool_result_with_call_id(
        "tool-result-utc",
        tool_call
            .call_id
            .clone()
            .unwrap_or_else(|| tool_call.id.clone()),
        OneOrMany::one(ToolResultContent::text(
            "Current time: 2026-03-13 00:00:00 UTC",
        )),
    ));

    let follow_up = model
        .completion_request("Use the tool result to answer in one short sentence.")
        .preamble(
            "Finish the function-calling validation by using the provided tool result.".to_string(),
        )
        .messages(vec![
            Message::user(prompt),
            Message::from(tool_call),
            tool_result,
        ])
        .max_tokens(96)
        .temperature(0.0)
        .send()
        .await?;

    ensure!(
        !follow_up.raw_response.text.trim().is_empty(),
        "tool-call follow-up returned empty text"
    );

    summary.tool_roundtrip_completed = true;

    Ok(())
}

// ── Per-model focused helpers (thinking + tool roundtrip) ────────────

pub async fn completion_with_thinking(
    model: &Model,
    prompt: &str,
    preamble: &str,
) -> anyhow::Result<(bool, bool, String)> {
    let response = model
        .completion_request(prompt)
        .preamble(preamble.to_string())
        .max_tokens(2048)
        .temperature(0.3)
        .additional_params(json!({ "thinking": true }))
        .send()
        .await?;

    let has_reasoning = response
        .choice
        .iter()
        .any(|c| matches!(c, AssistantContent::Reasoning(_)));
    let has_text = response
        .choice
        .iter()
        .any(|c| matches!(c, AssistantContent::Text(_)));
    Ok((has_reasoning, has_text, response.raw_response.text))
}

pub async fn tool_roundtrip(model: &Model) -> anyhow::Result<(String, String)> {
    let tool = ToolDefinition {
        name: "get_time".to_string(),
        description: "Return the current UTC time as plain text.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        }),
    };

    let prompt = "What time is it? Call get_time to find out.";
    let response = model
        .completion_request(prompt)
        .preamble("You have access to tools. Use them when needed.".to_string())
        .tool(tool)
        .max_tokens(256)
        .temperature(0.0)
        .additional_params(json!({ "thinking": true }))
        .send()
        .await?;

    let tool_call = response
        .choice
        .iter()
        .find_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some(tc.clone()),
            _ => None,
        })
        .context("model did not produce a tool call")?;

    let tool_name = tool_call.function.name.clone();

    let tool_result = Message::from(UserContent::tool_result_with_call_id(
        "tool-result-utc",
        tool_call
            .call_id
            .clone()
            .unwrap_or_else(|| tool_call.id.clone()),
        OneOrMany::one(ToolResultContent::text(
            "Current time: 2026-04-12 15:30:00 UTC",
        )),
    ));

    let follow_up = model
        .completion_request("Use the tool result to answer briefly.")
        .preamble("Answer using the tool result provided.".to_string())
        .messages(vec![
            Message::user(prompt),
            Message::from(tool_call),
            tool_result,
        ])
        .max_tokens(128)
        .temperature(0.0)
        .additional_params(json!({ "thinking": true }))
        .send()
        .await?;

    let text = assistant_text(&follow_up.choice);
    Ok((tool_name, text))
}

/// Diagnostic record from one structured-output streaming run.
///
/// Tests assert on `parsed_ok`, but `raw` and `chunk_count` are
/// printed regardless to make failures actionable: an empty `raw`
/// (typical Gemma symptom) indicates the model never emitted any
/// content tokens, while a non-empty `raw` that fails parsing is a
/// formatting issue we can extract from.
#[derive(Debug)]
pub struct StreamedStructuredOutcome {
    pub raw: String,
    pub chunk_count: usize,
    pub parsed_ok: bool,
    pub parse_error: Option<String>,
}

/// Run a structured-output prompt over the **streaming** path, mirroring
/// the way `chatty` invokes us: schema set on the `AgentBuilder`,
/// `.stream_chat()` consumed chunk-by-chunk, accumulated text parsed as
/// JSON at the end. Returns the accumulated text, the chunk count, and
/// whether the result deserialized into the requested type.
///
/// `schema` is the runtime-built `schemars::Schema` (so we exercise the
/// same code path chatty does, where the schema is constructed from a
/// runtime spec rather than `derive(JsonSchema)`).
pub async fn run_streamed_structured<T: serde::de::DeserializeOwned>(
    client: &Client,
    schema: schemars::Schema,
    preamble: &str,
    prompt: &str,
) -> anyhow::Result<StreamedStructuredOutcome> {
    let agent = client
        .agent("local")
        .preamble(preamble)
        .max_tokens(256)
        .temperature(0.2)
        .output_schema_raw(schema)
        .build();

    use rig::agent::MultiTurnStreamItem;

    let mut stream = agent.stream_chat(prompt, Vec::<Message>::new()).await;

    let mut raw = String::new();
    let mut chunk_count = 0usize;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) =
            item?
        {
            raw.push_str(&text.text);
            chunk_count += 1;
        }
    }

    let (parsed_ok, parse_error) = match serde_json::from_str::<T>(raw.trim()) {
        Ok(_) => (true, None),
        Err(e) => (false, Some(e.to_string())),
    };

    Ok(StreamedStructuredOutcome {
        raw,
        chunk_count,
        parsed_ok,
        parse_error,
    })
}

/// Long-form streaming + completion + tool-call validation. Shared by the
/// per-model `e2e_inference_*` tests.
pub async fn run_long_e2e(model_path: &std::path::Path) -> anyhow::Result<()> {
    let n_ctx = env_parse_u32("N_CTX", 32_768);
    let max_tokens_per_turn = env_parse_u64("RIG_MAX_TOKENS_PER_TURN", 3_072);
    let target_output_tokens = env_parse_u64("RIG_TARGET_OUTPUT_TOKENS", 10_000);
    let lines_per_turn = env_parse_usize("RIG_LINES_PER_TURN", 160);
    let max_generation_turns = env_parse_usize("RIG_MAX_GENERATION_TURNS", 6);

    let client = Client::from_gguf(
        model_path.to_string_lossy().into_owned(),
        n_ctx,
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams::default(),
        CheckpointParams::default(),
    )?;
    let model = client.completion_model("local");

    let smoke = model
        .completion_request("Reply with exactly: model ready")
        .max_tokens(32)
        .temperature(0.0)
        .send()
        .await?;
    ensure!(
        !smoke.raw_response.text.trim().is_empty(),
        "smoke completion returned empty text"
    );

    let mut history = seed_history();
    let preamble = corpus_preamble();
    let mut summary = RunSummary {
        conversation_messages: history.len(),
        ..RunSummary::default()
    };

    let mut next_start = 1usize;

    for turn in 0..max_generation_turns {
        if summary.total_output_tokens >= target_output_tokens && history.len() >= 24 {
            break;
        }

        let end = next_start + lines_per_turn - 1;
        let prompt = corpus_prompt(next_start, end);

        if turn % 2 == 0 {
            run_completion_turn(
                &model,
                &mut history,
                prompt,
                &preamble,
                max_tokens_per_turn,
                0.2,
                &mut summary,
            )
            .await?;
        } else {
            run_streaming_turn(
                &model,
                &mut history,
                prompt,
                &preamble,
                max_tokens_per_turn,
                0.2,
                &mut summary,
            )
            .await?;
        }

        next_start = end + 1;
    }

    ensure!(
        history.len() >= 24,
        "conversation too short: {} messages",
        history.len()
    );
    ensure!(
        summary.completion_turns > 0,
        "completion path was not exercised"
    );
    ensure!(
        summary.streaming_turns > 0,
        "streaming path was not exercised"
    );
    ensure!(
        summary.total_output_tokens >= target_output_tokens,
        "generated {} output tokens, below target {}",
        summary.total_output_tokens,
        target_output_tokens
    );

    attempt_tool_call(&model, &mut summary).await?;

    if !summary.tool_call_observed {
        eprintln!(
            "[WARN] Tool call was NOT observed. \
             Set RIG_REQUIRE_TOOL_CALL=1 to make this a hard failure."
        );
    }
    if std::env::var("RIG_REQUIRE_TOOL_CALL").as_deref() == Ok("1") {
        ensure!(summary.tool_call_observed, "tool call not observed");
        ensure!(
            summary.tool_roundtrip_completed,
            "tool roundtrip not completed"
        );
    }

    println!("{summary}");

    Ok(())
}

// ── Vision (mtmd-only) helpers ───────────────────────────────────────

#[cfg(feature = "mtmd")]
pub fn detect_image_media_type(path: &std::path::Path) -> rig::message::ImageMediaType {
    use rig::message::ImageMediaType;
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg") => ImageMediaType::JPEG,
        Some("png") => ImageMediaType::PNG,
        Some("gif") => ImageMediaType::GIF,
        Some("webp") => ImageMediaType::WEBP,
        _ => ImageMediaType::JPEG,
    }
}

#[cfg(feature = "mtmd")]
pub async fn run_vision(
    model_path: &std::path::Path,
    mmproj_path: &std::path::Path,
) -> anyhow::Result<()> {
    use rig::message::{DocumentSourceKind, Image};

    ensure!(
        model_path.is_file(),
        "vision model not found at {}",
        model_path.display()
    );
    ensure!(
        mmproj_path.is_file(),
        "mmproj file not found at {}",
        mmproj_path.display()
    );

    let image_path = test_image_path();
    ensure!(
        image_path.is_file(),
        "image fixture not found at {}",
        image_path.display()
    );

    let media_type = detect_image_media_type(&image_path);
    let n_ctx = env_parse_u32("N_CTX", 8192);
    let image_bytes = std::fs::read(&image_path)
        .with_context(|| format!("failed to read image at {}", image_path.display()))?;

    let client = Client::from_gguf_with_mmproj(
        model_path.to_string_lossy().into_owned(),
        mmproj_path.to_string_lossy().into_owned(),
        n_ctx,
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams::default(),
        CheckpointParams::default(),
    )?;
    let model = client.completion_model("local");

    let response = model
        .completion_request("Describe this image briefly.")
        .messages(vec![Message::from(OneOrMany::many(vec![
            UserContent::Image(Image {
                media_type: Some(media_type),
                data: DocumentSourceKind::Raw(image_bytes),
                detail: None,
                additional_params: None,
            }),
            UserContent::text("What do you see in this image?"),
        ])?)])
        .max_tokens(256)
        .temperature(0.3)
        .send()
        .await?;

    ensure!(
        !response.raw_response.text.trim().is_empty(),
        "vision completion returned empty text"
    );
    ensure!(
        response.usage.output_tokens > 0,
        "vision completion returned zero output tokens"
    );

    println!(
        "Vision test passed: output_tokens={}, text_preview={}",
        response.usage.output_tokens,
        &response.raw_response.text[..response.raw_response.text.len().min(100)]
    );

    Ok(())
}
