//! Qwen 3.5-2B integration tests.

use anyhow::ensure;
use rig_agent::AgentBuilder;
use rig_core::completion::CompletionRequest;
use rig_llama_cpp::{
    CheckpointParams, Client, FitParams, KvCacheParams, KvCacheType, SamplingParams,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serial_test::serial;

use super::common::{
    QWEN, completion_with_thinking, ensure_model, env_parse_u32, load_default, raw_text,
    run_long_e2e, run_streamed_structured, tool_roundtrip,
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
        .call(
            CompletionRequest::new("Reply with exactly: kv cache ok")
                .max_tokens(32)
                .temperature(0.0),
        )
        .await?;
    let text = raw_text(&response);
    ensure!(
        !text.trim().is_empty(),
        "Q8_0 KV cache completion returned empty text"
    );

    println!("Q8_0 KV cache response: {text}");
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
        // Seed 42, the sampler default, samples a run that skips thinking.
        1,
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

/// The regression `0.4.0` introduced and `0.5.1` fixed.
///
/// One call plus a prose follow-up does not catch it: the prompt had lost the
/// assistant's own `tool_calls`, and a model handed a tool result it never
/// asked for will still say *something*. What it stops doing is calling a
/// second tool — small models give up there, which is exactly how this
/// surfaced. So drive two sequential calls and assert the second happens.
#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B and validates a two-step tool sequence"]
async fn qwen_calls_a_second_tool_after_a_result() -> anyhow::Result<()> {
    use anyhow::Context;
    use rig_core::completion::ToolDefinition;
    use rig_core::message::{AssistantContent, Message, ToolName, ToolResultContent};
    use serde_json::json;

    let path = ensure_model(&QWEN)?;
    let (_client, model) = load_default(&path)?;

    let get_time = ToolDefinition {
        name: ToolName::new("get_time")?,
        description: "Return the current UTC time as plain text.".to_string(),
        parameters: json!({"type": "object", "properties": {}, "additionalProperties": false}),
    };
    let save_time = ToolDefinition {
        name: ToolName::new("save_time")?,
        description: "Persist a time string. Call this after get_time.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": false,
        }),
    };

    let preamble = "You have access to tools. Use them when needed, one at a time.";
    let task = "Find the current time with get_time, then store it with save_time.";

    let first = model
        .call(
            CompletionRequest::new(task)
                .preamble(preamble)
                .tool(get_time.clone())
                .tool(save_time.clone())
                .max_tokens(256)
                .temperature(0.0),
        )
        .await?;

    let call = first
        .choice
        .iter()
        .find_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some(tc.clone()),
            _ => None,
        })
        .context("model did not produce a first tool call")?;
    ensure!(
        call.function.name.as_str() == "get_time",
        "expected get_time first, got {}",
        call.function.name.as_str()
    );

    let result = Message::from(call.result(vec![ToolResultContent::text(
        "Current time: 2026-04-12 15:30:00 UTC",
    )]));

    let second = model
        .call(
            CompletionRequest::new("Continue.")
                .preamble(preamble)
                .tool(get_time)
                .tool(save_time)
                .messages(vec![Message::user(task), Message::from(call), result])
                .max_tokens(256)
                .temperature(0.0),
        )
        .await?;

    let follow_up = second
        .choice
        .iter()
        .find_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some(tc.clone()),
            _ => None,
        })
        .context(
            "model made no second tool call — the assistant's first call is probably \
             missing from the rendered prompt",
        )?;
    println!(
        "qwen_calls_a_second_tool_after_a_result: second call={} args={}",
        follow_up.function.name.as_str(),
        serde_json::Value::Object(follow_up.function.arguments.clone())
    );
    ensure!(
        follow_up.function.name.as_str() == "save_time",
        "expected save_time second, got {}",
        follow_up.function.name.as_str()
    );
    Ok(())
}

/// Qwen's `<parameter=…>` form carries no types — every value is bare text —
/// so an object-valued argument has to be parsed back into an object using the
/// tool's declared schema. Without that the tool rejects the call with
/// "invalid type: string ..., expected a map" and the agent gives up.
#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B and validates an object-valued tool argument"]
async fn qwen_passes_an_object_valued_argument() -> anyhow::Result<()> {
    use anyhow::Context;
    use rig_core::completion::ToolDefinition;
    use rig_core::message::{AssistantContent, ToolName};
    use serde_json::{Value, json};

    let path = ensure_model(&QWEN)?;
    let (_client, model) = load_default(&path)?;

    let update_entity = ToolDefinition {
        name: ToolName::new("update_entity")?,
        description: "Update fields on an entity.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The entity ID."},
                "fields": {
                    "type": "object",
                    "description": "Field name to new value, e.g. {\"Vendor\": \"Acme\"}.",
                },
            },
            "required": ["id", "fields"],
            "additionalProperties": false,
        }),
    };

    let response = model
        .call(
            CompletionRequest::new("Set the Vendor field of asset ABB1-NTP1 to Meinberg.")
                .preamble("You have access to tools. Use them when needed.")
                .tool(update_entity)
                .max_tokens(256)
                .temperature(0.0),
        )
        .await?;

    let call = response
        .choice
        .iter()
        .find_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some(tc.clone()),
            _ => None,
        })
        .context("model did not produce a tool call")?;
    println!(
        "qwen_passes_an_object_valued_argument: args={}",
        Value::Object(call.function.arguments.clone())
    );

    let fields = call
        .function
        .arguments
        .get("fields")
        .context("call carried no `fields` argument")?;
    ensure!(
        matches!(fields, Value::Object(_)),
        "`fields` reached the tool as {fields:?}, not an object — the tool's \
         deserializer would reject this with \"expected a map\""
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B and validates structured-output extraction"]
async fn qwen_structured_output() -> anyhow::Result<()> {
    let path = ensure_model(&QWEN)?;
    let (client, _model) = load_default(&path)?;
    let agent = AgentBuilder::new(client.completion_model("local"))
        .preamble("Extract the single person described in the user's text as structured data.")
        .max_tokens(256)
        .temperature(0.2)
        .build();

    let person = agent
        .prompt_typed::<ExtractedPerson>("Ada is a 36-year-old software engineer living in Berlin.")
        .await?
        .output;

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
/// `AgentBuilder`, response consumed as a stream, accumulated
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
