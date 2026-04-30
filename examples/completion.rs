use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::tool::ToolDyn;
use rig_llama_cpp::{CheckpointParams, Client, FitParams, KvCacheParams, SamplingParams};
use serde_json::json;

#[path = "./helper/time.rs"]
mod time;

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

    let tools: Vec<Box<dyn ToolDyn>> = vec![Box::new(time::GetCurrentTime)];

    let agent = client
        .agent("local")
        .preamble("You are a helpful ai assistant with access to tools that can provide information about the users request. Use the tools to provide accurate and helpful responses to the user.")
        .tools(tools)
        .max_tokens(2048)
        .temperature(1.0)
        .additional_params(json!({ "thinking": true }))
        .build();

    let response = agent.prompt("What time is it?").await?;
    println!("{response}");

    Ok(())
}
