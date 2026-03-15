use std::fmt;
use std::path::PathBuf;

use anyhow::{Context, ensure};
use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, GetTokenUsage, ToolDefinition};
use rig::message::{AssistantContent, Message, ToolChoice, ToolResultContent, UserContent};
use rig::streaming::StreamedAssistantContent;
use serde_json::json;
use tokio_stream::StreamExt;

use rig::embeddings::EmbeddingModel as _;

use crate::{Client, EmbeddingClient, Model, SamplingParams};

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

fn default_model_path() -> PathBuf {
    std::env::var("MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Qwen3.5-2B-Q4_K_M.gguf")
        })
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
#[ignore = "loads the local Qwen GGUF and generates a long validation transcript"]
async fn qwen35_e2e_inference_streaming_completion() -> anyhow::Result<()> {
    let model_path = default_model_path();
    ensure!(
        model_path.is_file(),
        "model file not found at {}",
        model_path.display()
    );

    let n_gpu_layers = env_parse_u32("N_GPU_LAYERS", u32::MAX);
    let n_ctx = env_parse_u32("N_CTX", 32_768);
    let max_tokens_per_turn = env_parse_u64("RIG_MAX_TOKENS_PER_TURN", 3_072);
    let target_output_tokens = env_parse_u64("RIG_TARGET_OUTPUT_TOKENS", 10_000);
    let lines_per_turn = env_parse_usize("RIG_LINES_PER_TURN", 160);
    let max_generation_turns = env_parse_usize("RIG_MAX_GENERATION_TURNS", 6);

    let client = Client::from_gguf(
        model_path.to_string_lossy().into_owned(),
        n_gpu_layers,
        n_ctx,
        SamplingParams::default(),
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

    println!("{summary}");

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

    let n_gpu_layers = env_parse_u32("N_GPU_LAYERS", u32::MAX);
    let n_ctx = env_parse_u32("N_CTX", 8192);

    {
        let _client = Client::from_gguf(
            first.to_string_lossy().into_owned(),
            n_gpu_layers,
            n_ctx,
            SamplingParams::default(),
        )?;
    }

    let _client = Client::from_gguf(
        second.to_string_lossy().into_owned(),
        n_gpu_layers,
        n_ctx,
        SamplingParams::default(),
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
