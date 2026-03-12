use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::tool::ToolDyn;
use serde_json::json;

#[path = "./helper/time.rs"]
mod time;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model_path = std::env::var("MODEL_PATH")
        .expect("Set MODEL_PATH env var to your GGUF model file path");

    let n_gpu_layers = std::env::var("N_GPU_LAYERS")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(u32::MAX);

    let client = rig_llama_cpp::Client::from_gguf(
        &model_path,
        n_gpu_layers,
        262144,
        0.95,
        20,
        0.0,
        1.5,
        1.0,
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

