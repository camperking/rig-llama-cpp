//! Qwen 3.5-2B integration tests.

use anyhow::ensure;
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, TypedPrompt};
use rig_llama_cpp::{
    CheckpointParams, Client, FitParams, KvCacheParams, KvCacheType, SamplingParams,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serial_test::serial;

use super::common::{
    QWEN, completion_with_thinking, ensure_model, env_parse_u32, load_default, run_long_e2e,
    run_streamed_structured, tool_roundtrip,
};

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct ExtractedPerson {
    name: String,
    age: u32,
    occupation: String,
}

#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B and runs a long validation transcript"]
async fn e2e_inference_qwen() -> anyhow::Result<()> {
    let path = ensure_model(&QWEN)?;
    run_long_e2e(&path).await
}

#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B and runs an inference with Q8_0 KV cache"]
async fn kv_cache_q8_0_qwen() -> anyhow::Result<()> {
    let path = ensure_model(&QWEN)?;
    let n_ctx = env_parse_u32("N_CTX", 8192);
    let client = Client::from_gguf(
        path.to_string_lossy().into_owned(),
        n_ctx,
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams::default()
            .with_type_k(KvCacheType::Q8_0)
            .with_type_v(KvCacheType::Q8_0),
        CheckpointParams::default(),
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

#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B and validates reasoning output"]
async fn qwen_thinking() -> anyhow::Result<()> {
    let path = ensure_model(&QWEN)?;
    let (_client, model) = load_default(&path)?;
    let (has_reasoning, has_text, raw) = completion_with_thinking(
        &model,
        "Explain why the sky is blue in one sentence.",
        "You are a helpful assistant.",
    )
    .await?;

    println!(
        "qwen_thinking: reasoning={has_reasoning}, text={has_text}, raw_len={}",
        raw.len()
    );
    ensure!(
        has_reasoning,
        "Qwen should produce reasoning content with thinking enabled"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B and validates a tool-call roundtrip"]
async fn qwen_tool_roundtrip() -> anyhow::Result<()> {
    let path = ensure_model(&QWEN)?;
    let (_client, model) = load_default(&path)?;
    let (tool_name, follow_up) = tool_roundtrip(&model).await?;

    println!(
        "qwen_tool_roundtrip: called={tool_name}, follow_up_len={}",
        follow_up.len()
    );
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

#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B and validates structured-output extraction"]
async fn qwen_structured_output() -> anyhow::Result<()> {
    let path = ensure_model(&QWEN)?;
    let (client, _model) = load_default(&path)?;
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
    ensure!(
        !person.name.is_empty(),
        "Qwen structured output: name was empty"
    );
    ensure!(person.age > 0, "Qwen structured output: age was zero");
    ensure!(
        !person.occupation.is_empty(),
        "Qwen structured output: occupation was empty"
    );
    Ok(())
}

#[cfg(feature = "mtmd")]
#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B + mmproj and runs a vision completion"]
async fn vision_basic_qwen() -> anyhow::Result<()> {
    let model_path = ensure_model(&QWEN)?;
    let mmproj_path = ensure_model(&super::common::QWEN_MMPROJ)?;
    super::common::run_vision(&model_path, &mmproj_path).await
}

/// Streaming structured-output over a runtime-built schema. Mirrors the
/// path `chatty` takes for workflow agents (schema set on
/// `AgentBuilder`, response consumed via `stream_chat`, accumulated
/// text parsed as JSON afterwards). Regression guard: previously
/// passed for Qwen but broke on Gemma; we keep both tests so divergence
/// surfaces here next time.
#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B and validates streaming structured output"]
async fn qwen_structured_output_streaming() -> anyhow::Result<()> {
    let path = ensure_model(&QWEN)?;
    let (client, _model) = load_default(&path)?;

    // Runtime-built schema (matches what chatty produces from its
    // `Vec<SchemaField>`), not a `derive(JsonSchema)` type.
    let schema_value = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "number" },
            "occupation": { "type": "string" }
        },
        "required": ["name", "age", "occupation"],
        "additionalProperties": false,
    });
    let schema = schemars::Schema::try_from(schema_value)?;

    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct Person {
        name: String,
        age: u32,
        occupation: String,
    }

    let outcome = run_streamed_structured::<Person>(
        &client,
        schema,
        "Extract the single person described in the user's text as structured JSON. Respond with the JSON object only.",
        "Ada is a 36-year-old software engineer living in Berlin.",
    )
    .await?;

    println!(
        "qwen_structured_output_streaming: chunks={}, raw_len={}, parsed_ok={}, raw={:?}",
        outcome.chunk_count,
        outcome.raw.len(),
        outcome.parsed_ok,
        outcome.raw,
    );
    ensure!(
        outcome.chunk_count > 0,
        "Qwen streaming structured output: no text chunks emitted"
    );
    ensure!(
        outcome.parsed_ok,
        "Qwen streaming structured output failed to parse: {:?} — raw was {:?}",
        outcome.parse_error,
        outcome.raw
    );
    Ok(())
}
