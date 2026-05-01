use rig::agent::MultiTurnStreamItem;
use rig::client::CompletionClient;
use rig::completion::ToolDefinition;
use rig::message::Message;
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingChat};
use rig::tool::{Tool, ToolDyn};
use rig_llama_cpp::{CheckpointParams, Client, FitParams, KvCacheParams, SamplingParams};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::StreamExt;

#[derive(Debug, Deserialize, Serialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct WriteFileError(String);

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WriteFile;

impl Tool for WriteFile {
    const NAME: &'static str = "write_file";
    type Error = WriteFileError;
    type Args = WriteFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Write the given content to the file at the given path.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" },
                },
                "required": ["path", "content"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(format!(
            "Successfully wrote {} bytes to '{}'",
            args.content.len(),
            args.path
        ))
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model_path =
        std::env::var("MODEL_PATH").expect("Set MODEL_PATH env var to your GGUF model file path");

    let client = Client::from_gguf(
        &model_path,
        8192,
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams::default(),
        CheckpointParams::default(),
    )?;

    let tools: Vec<Box<dyn ToolDyn>> = vec![Box::new(WriteFile)];

    let agent = client
        .agent("local")
        .preamble(
            "You are the Chatty agent builder. You help the user author custom agents by \
             writing Markdown files with TOML frontmatter to the current working directory. \
             On confirmation, call write_file with the filename and full contents.",
        )
        .tools(tools)
        .max_tokens(4096)
        .default_max_turns(6)
        .temperature(0.3)
        .additional_params(json!({ "thinking": true }))
        .build();

    let mut stream = agent
        .stream_chat(
            "Please write the string FOO to bar.md, then confirm in one sentence.",
            Vec::<Message>::new(),
        )
        .await;

    let mut iteration_assistant: Vec<String> = Vec::new();
    let mut tool_call_count = 0u32;
    let mut tool_result_count = 0u32;

    println!("--- streaming start ---");
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(t))) => {
                iteration_assistant.push(format!("TEXT: {}", t.text));
                print!("{}", t.text);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { reasoning, .. },
            )) => {
                iteration_assistant.push(format!("THINK: {}", reasoning));
                print!("[think]{}[/think]", reasoning);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call,
                ..
            })) => {
                tool_call_count += 1;
                println!(
                    "\n[TOOL_CALL #{}] name={} id={} args={}",
                    tool_call_count,
                    tool_call.function.name,
                    tool_call.id,
                    tool_call.function.arguments
                );
            }
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result,
                ..
            })) => {
                tool_result_count += 1;
                println!(
                    "[TOOL_RESULT #{}] id={} call_id={:?}",
                    tool_result_count, tool_result.id, tool_result.call_id
                );
            }
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                println!("\n--- final response ---");
                println!("usage: {:?}", res.usage());
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("\n[ERROR] {e}");
                break;
            }
        }
    }

    println!("\n--- summary ---");
    println!("tool_calls   : {tool_call_count}");
    println!("tool_results : {tool_result_count}");
    if tool_call_count > 1 {
        println!("BUG REPRODUCED: model issued the same tool call {tool_call_count} times");
    }

    Ok(())
}
