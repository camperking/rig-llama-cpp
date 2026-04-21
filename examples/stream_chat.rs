use rig::client::CompletionClient;
use rig::message::Message;
use rig::streaming::StreamingChat;
use rig::tool::ToolDyn;
use rig_llama_cpp::{Client, FitParams, KvCacheParams, SamplingParams};
use serde_json::json;

#[path = "./helper/time.rs"]
mod time;

#[path = "./helper/stream_out.rs"]
mod stream_out;

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

    let messages = vec![
        Message::user("Tell me a joke!"),
        Message::assistant("Why did the chicken cross the road?\n\nTo get to the other side!"),
    ];

    // Prompt the agent and print the response
    let mut stream = agent.stream_chat("Entertain me!", messages).await;

    let res = stream_out::stream_out(&mut stream).await.unwrap();

    println!("Response: {res:?}");

    Ok(())
}
