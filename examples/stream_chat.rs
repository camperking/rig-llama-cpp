use rig_agent::AgentBuilder;
use rig_core::message::Message;
use rig_llama_cpp::{CheckpointParams, Client, FitParams, KvCacheParams, SamplingParams};
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
        CheckpointParams::default(),
    )?;

    let agent = AgentBuilder::new(client.completion_model("local"))
        .preamble("You are a helpful ai assistant with access to tools that can provide information about the users request. Use the tools to provide accurate and helpful responses to the user.")
        .tool(time::GetCurrentTime)
        .max_tokens(2048)
        .temperature(1.0)
        .additional_params(json!({ "thinking": true }))
        .build();

    let messages = vec![
        Message::user("Tell me a joke!"),
        Message::assistant("Why did the chicken cross the road?\n\nTo get to the other side!"),
    ];

    // Prompt the agent and print the response
    let mut stream = agent.prompt("Entertain me!").history(messages).stream();

    let res = stream_out::stream_out(&mut stream).await.unwrap();

    println!("Response: {res:?}");

    Ok(())
}
