use std::fmt;
use std::path::PathBuf;

use anyhow::{Context, ensure};
use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, GetTokenUsage, ToolDefinition};
use rig::message::{
    AssistantContent, ImageMediaType, Message, ToolChoice, ToolResultContent, UserContent,
};
use rig::streaming::StreamedAssistantContent;
use serde_json::json;
use tokio_stream::StreamExt;

use rig::embeddings::EmbeddingModel as _;

use crate::{Client, EmbeddingClient, FitParams, KvCacheParams, KvCacheType, Model, SamplingParams};
use rig::completion::TypedPrompt;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Default)]
struct RunSummary {
    total_output_tokens: u64,
    completion_turns: usize,
    streaming_turns: usize,
    streamed_text_chunks: usize,
    conversation_messages: usize,
    tool_call_observed: bool,
    tool_roundtrip_completed: bool,
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

fn detect_image_media_type(path: &std::path::Path) -> ImageMediaType {
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

fn required_model_path(name: &str) -> anyhow::Result<PathBuf> {
    let path = std::env::var(name)
        .with_context(|| format!("missing required environment variable {name}"))?;
    Ok(PathBuf::from(path))
}

fn env_parse_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn env_parse_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn corpus_preamble() -> String {
    "You are generating a deterministic corpus for local inference validation. Respond only with numbered lines in the form 'NNNN: sentence'. Each sentence must be between 14 and 20 words, describe a distinct LLM testing scenario, and avoid markdown or extra commentary.".to_string()
}

fn corpus_prompt(start: usize, end: usize) -> String {
    format!(
        "Continue the corpus with lines {start:04} through {end:04}. Keep the numbering contiguous, output one line per item, and stop exactly after line {end:04}."
    )
}

fn seed_history() -> Vec<Message> {
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

fn assistant_text(choice: &rig::OneOrMany<AssistantContent>) -> String {
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

async fn run_completion_turn(
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

async fn run_streaming_turn(
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

async fn attempt_tool_call(model: &Model, summary: &mut RunSummary) -> anyhow::Result<()> {
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

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads a local GGUF model (set MODEL_PATH) and generates a long validation transcript"]
async fn e2e_inference_streaming_completion() -> anyhow::Result<()> {
    let model_path = required_model_path("MODEL_PATH")?;
    ensure!(
        model_path.is_file(),
        "model file not found at {}",
        model_path.display()
    );

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

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads a local GGUF model (set MODEL_PATH) with Q8_0 KV cache quantization"]
async fn e2e_kv_cache_q8_0() -> anyhow::Result<()> {
    let model_path = required_model_path("MODEL_PATH")?;
    ensure!(
        model_path.is_file(),
        "model file not found at {}",
        model_path.display()
    );

    let n_ctx = env_parse_u32("N_CTX", 8192);

    let client = Client::from_gguf(
        model_path.to_string_lossy().into_owned(),
        n_ctx,
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams {
            type_k: KvCacheType::Q8_0,
            type_v: KvCacheType::Q8_0,
        },
    )?;
    let model = client.completion_model("local");

    let response = model
        .completion_request("Reply with exactly: kv cache ok")
        .max_tokens(32)
        .temperature(0.0)
        .send()
        .await?;
    ensure!(
        !response.raw_response.text.trim().is_empty(),
        "Q8_0 KV cache completion returned empty text"
    );

    println!("Q8_0 KV cache response: {}", response.raw_response.text);

    Ok(())
}

#[test]
#[ignore = "loads two real GGUF models sequentially to validate backend reinitialization"]
fn sequential_real_model_reload() -> anyhow::Result<()> {
    let first = required_model_path("RIG_MODEL_A")?;
    let second = required_model_path("RIG_MODEL_B")?;
    ensure!(
        first.is_file(),
        "first model file not found at {}",
        first.display()
    );
    ensure!(
        second.is_file(),
        "second model file not found at {}",
        second.display()
    );

    let n_ctx = env_parse_u32("N_CTX", 8192);

    {
        let _client = Client::from_gguf(
            first.to_string_lossy().into_owned(),
            n_ctx,
            SamplingParams::default(),
            FitParams::default(),
            KvCacheParams::default(),
        )?;
    }

    let _client = Client::from_gguf(
        second.to_string_lossy().into_owned(),
        n_ctx,
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams::default(),
    )?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a GGUF embedding model (set EMBEDDING_MODEL_PATH)"]
async fn embedding_basic() -> anyhow::Result<()> {
    let model_path = required_model_path("EMBEDDING_MODEL_PATH")?;
    ensure!(
        model_path.is_file(),
        "embedding model file not found at {}",
        model_path.display()
    );

    let n_gpu_layers = env_parse_u32("N_GPU_LAYERS", u32::MAX);
    let n_ctx = env_parse_u32("N_CTX", 8192);

    let client = EmbeddingClient::from_gguf(
        model_path.to_string_lossy().into_owned(),
        n_gpu_layers,
        n_ctx,
    )?;
    let model = client.embedding_model("local");

    // Single text embedding
    let emb = model.embed_text("Hello, world!").await?;
    ensure!(
        emb.vec.len() == model.ndims(),
        "embedding dimension mismatch: got {}, expected {}",
        emb.vec.len(),
        model.ndims()
    );
    ensure!(
        emb.vec.iter().any(|v| *v != 0.0),
        "embedding should not be all zeros"
    );

    // Multiple texts
    let embeddings = model
        .embed_texts(vec![
            "The cat sat on the mat.".to_string(),
            "Dogs are loyal animals.".to_string(),
            "The weather is sunny today.".to_string(),
        ])
        .await?;
    ensure!(
        embeddings.len() == 3,
        "expected 3 embeddings, got {}",
        embeddings.len()
    );
    for (i, emb) in embeddings.iter().enumerate() {
        ensure!(
            emb.vec.len() == model.ndims(),
            "embedding {i} dimension mismatch: got {}, expected {}",
            emb.vec.len(),
            model.ndims()
        );
    }

    println!(
        "Embedding test passed: ndims={}, single_ok=true, batch_count={}",
        model.ndims(),
        embeddings.len()
    );

    Ok(())
}

#[cfg(feature = "mtmd")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a vision GGUF model, mmproj, and image file"]
async fn vision_basic() -> anyhow::Result<()> {
    use rig::message::{DocumentSourceKind, Image};

    let model_path = required_model_path("MODEL_PATH")?;
    let mmproj_path = required_model_path("MMPROJ_PATH")?;
    let image_path = required_model_path("IMAGE_PATH")?;
    let media_type = detect_image_media_type(&image_path);

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
    ensure!(
        image_path.is_file(),
        "image file not found at {}",
        image_path.display()
    );

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

// ---------------------------------------------------------------------------
// Focused integration tests: thinking and tool calling per model
// ---------------------------------------------------------------------------

/// Helper: load a model from a path relative to the crate root.
fn load_model(gguf: &str) -> anyhow::Result<(Client, Model)> {
    let path = PathBuf::from(gguf);
    ensure!(path.is_file(), "model file not found: {gguf}");
    let client = Client::from_gguf(
        gguf.to_string(),
        env_parse_u32("N_CTX", 8192),
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams::default(),
    )?;
    let model = client.completion_model("local");
    Ok((client, model))
}

/// Helper: run a completion with thinking enabled and return (has_reasoning, has_text, raw).
async fn completion_with_thinking(
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

/// Helper: run a tool-call roundtrip and return (tool_name, follow_up_text).
async fn tool_roundtrip(model: &Model) -> anyhow::Result<(String, String)> {
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

// --- Qwen3.5 tests ---

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Qwen3.5-2B-Q4_K_M.gguf in cwd"]
async fn qwen_thinking() -> anyhow::Result<()> {
    let (_client, model) = load_model("./Qwen3.5-2B-Q4_K_M.gguf")?;
    let (has_reasoning, has_text, raw) = completion_with_thinking(
        &model,
        "Explain why the sky is blue in one sentence.",
        "You are a helpful assistant.",
    )
    .await?;

    println!("qwen_thinking: reasoning={has_reasoning}, text={has_text}, raw_len={}", raw.len());
    ensure!(has_reasoning, "Qwen should produce reasoning content with thinking enabled");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Qwen3.5-2B-Q4_K_M.gguf in cwd"]
async fn qwen_tool_roundtrip() -> anyhow::Result<()> {
    let (_client, model) = load_model("./Qwen3.5-2B-Q4_K_M.gguf")?;
    let (tool_name, follow_up) = tool_roundtrip(&model).await?;

    println!("qwen_tool_roundtrip: called={tool_name}, follow_up_len={}", follow_up.len());
    ensure!(
        tool_name == "get_time",
        "Qwen called wrong tool: {tool_name}"
    );
    ensure!(
        !follow_up.trim().is_empty(),
        "Qwen follow-up after tool result was empty"
    );
    Ok(())
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct ExtractedPerson {
    name: String,
    age: u32,
    occupation: String,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Qwen3.5-2B-Q4_K_M.gguf in cwd"]
async fn qwen_structured_output() -> anyhow::Result<()> {
    let (client, _model) = load_model("./Qwen3.5-2B-Q4_K_M.gguf")?;
    let agent = client
        .agent("local")
        .preamble("Extract the single person described in the user's text as structured data.")
        .max_tokens(256)
        .temperature(0.2)
        .build();

    let person: ExtractedPerson = agent
        .prompt_typed("Ada is a 36-year-old software engineer living in Berlin.")
        .await?;

    println!(
        "qwen_structured_output: name={}, age={}, occupation={}",
        person.name, person.age, person.occupation
    );
    ensure!(!person.name.is_empty(), "Qwen structured output: name was empty");
    ensure!(person.age > 0, "Qwen structured output: age was zero");
    ensure!(
        !person.occupation.is_empty(),
        "Qwen structured output: occupation was empty"
    );
    Ok(())
}

// --- Gemma-4 tests ---

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires gemma-4-E4B-it-Q4_K_M.gguf in cwd"]
async fn gemma_thinking() -> anyhow::Result<()> {
    let (_client, model) = load_model("./gemma-4-E4B-it-Q4_K_M.gguf")?;
    let (has_reasoning, has_text, raw) = completion_with_thinking(
        &model,
        "Explain why the sky is blue in one sentence.",
        "You are a helpful assistant.",
    )
    .await?;

    println!("gemma_thinking: reasoning={has_reasoning}, text={has_text}, raw_len={}", raw.len());
    ensure!(has_reasoning, "Gemma-4 should produce reasoning content with thinking enabled");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires gemma-4-E4B-it-Q4_K_M.gguf in cwd"]
async fn gemma_structured_output() -> anyhow::Result<()> {
    let (client, _model) = load_model("./gemma-4-E4B-it-Q4_K_M.gguf")?;
    let agent = client
        .agent("local")
        .preamble("Extract the single person described in the user's text as structured data.")
        .max_tokens(256)
        .temperature(0.2)
        .build();

    let person: ExtractedPerson = agent
        .prompt_typed("Ada is a 36-year-old software engineer living in Berlin.")
        .await?;

    println!(
        "gemma_structured_output: name={}, age={}, occupation={}",
        person.name, person.age, person.occupation
    );
    ensure!(!person.name.is_empty(), "Gemma structured output: name was empty");
    ensure!(person.age > 0, "Gemma structured output: age was zero");
    ensure!(
        !person.occupation.is_empty(),
        "Gemma structured output: occupation was empty"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires gemma-4-E4B-it-Q4_K_M.gguf in cwd"]
async fn gemma_tool_roundtrip() -> anyhow::Result<()> {
    let (_client, model) = load_model("./gemma-4-E4B-it-Q4_K_M.gguf")?;
    let (tool_name, follow_up) = tool_roundtrip(&model).await?;

    println!("gemma_tool_roundtrip: called={tool_name}, follow_up_len={}", follow_up.len());
    ensure!(
        tool_name == "get_time",
        "Gemma-4 called wrong tool: {tool_name}"
    );
    ensure!(
        !follow_up.trim().is_empty(),
        "Gemma-4 follow-up after tool result was empty"
    );
    Ok(())
}
