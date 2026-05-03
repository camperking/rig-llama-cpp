//! Gemma-4 E4B integration tests.

use anyhow::ensure;
use rig::client::CompletionClient;
use rig::completion::TypedPrompt;
use schemars::JsonSchema;
use serde::Deserialize;
use serial_test::serial;

use super::common::{
    GEMMA, completion_with_thinking, ensure_model, load_default, run_long_e2e,
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
#[ignore = "downloads Gemma-4 E4B and runs a long validation transcript"]
async fn e2e_inference_gemma() -> anyhow::Result<()> {
    let path = ensure_model(&GEMMA)?;
    run_long_e2e(&path).await
}

#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Gemma-4 E4B and validates reasoning output"]
async fn gemma_thinking() -> anyhow::Result<()> {
    let path = ensure_model(&GEMMA)?;
    let (_client, model) = load_default(&path)?;
    let (has_reasoning, has_text, raw) = completion_with_thinking(
        &model,
        "Explain why the sky is blue in one sentence.",
        "You are a helpful assistant.",
    )
    .await?;

    println!(
        "gemma_thinking: reasoning={has_reasoning}, text={has_text}, raw_len={}",
        raw.len()
    );
    ensure!(
        has_reasoning,
        "Gemma-4 should produce reasoning content with thinking enabled"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Gemma-4 E4B and validates a tool-call roundtrip"]
async fn gemma_tool_roundtrip() -> anyhow::Result<()> {
    let path = ensure_model(&GEMMA)?;
    let (_client, model) = load_default(&path)?;
    let (tool_name, follow_up) = tool_roundtrip(&model).await?;

    println!(
        "gemma_tool_roundtrip: called={tool_name}, follow_up_len={}",
        follow_up.len()
    );
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

#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Gemma-4 E4B and validates structured-output extraction"]
async fn gemma_structured_output() -> anyhow::Result<()> {
    let path = ensure_model(&GEMMA)?;
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
        "gemma_structured_output: name={}, age={}, occupation={}",
        person.name, person.age, person.occupation
    );
    ensure!(
        !person.name.is_empty(),
        "Gemma structured output: name was empty"
    );
    ensure!(person.age > 0, "Gemma structured output: age was zero");
    ensure!(
        !person.occupation.is_empty(),
        "Gemma structured output: occupation was empty"
    );
    Ok(())
}

#[cfg(feature = "mtmd")]
#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Gemma-4 E4B + mmproj and runs a vision completion"]
async fn vision_basic_gemma() -> anyhow::Result<()> {
    let model_path = ensure_model(&GEMMA)?;
    let mmproj_path = ensure_model(&super::common::GEMMA_MMPROJ)?;
    super::common::run_vision(&model_path, &mmproj_path).await
}

/// Streaming structured-output over a runtime-built schema — the same
/// path Gemma was failing on inside `chatty`'s workflow runtime
/// ("structured response was not valid JSON: EOF while parsing a value
/// at line 1 column 0", indicating an empty accumulated string). This
/// regression test reproduces the issue against the real model so we
/// can iterate a fix in `rig-llama-cpp`.
#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Gemma-4 E4B and validates streaming structured output"]
async fn gemma_structured_output_streaming() -> anyhow::Result<()> {
    let path = ensure_model(&GEMMA)?;
    let (client, _model) = load_default(&path)?;

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
        "gemma_structured_output_streaming: chunks={}, raw_len={}, parsed_ok={}, raw={:?}",
        outcome.chunk_count,
        outcome.raw.len(),
        outcome.parsed_ok,
        outcome.raw,
    );
    ensure!(
        outcome.chunk_count > 0,
        "Gemma streaming structured output: no text chunks emitted (likely the bug)"
    );
    ensure!(
        outcome.parsed_ok,
        "Gemma streaming structured output failed to parse: {:?} — raw was {:?}",
        outcome.parse_error,
        outcome.raw
    );
    Ok(())
}
