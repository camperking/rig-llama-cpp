use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig_llama_cpp::{Client, FitParams, KvCacheParams, SamplingParams};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model_a = std::env::var("RIG_MODEL_A")
        .expect("Set RIG_MODEL_A env var to your first GGUF model file path");
    let model_b = std::env::var("RIG_MODEL_B")
        .expect("Set RIG_MODEL_B env var to your second GGUF model file path");

    let client = Client::from_gguf(
        &model_a,
        8192,
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams::default(),
    )?;

    let response_a = client
        .agent("local")
        .preamble("You are a helpful assistant. Answer in one short sentence.")
        .max_tokens(128)
        .build()
        .prompt("Say hello and mention you are model A.")
        .await?;
    println!("Model A: {response_a}\n");

    // Swap the model in-place on the existing worker thread. This avoids
    // re-initializing the llama.cpp backend singleton.
    client
        .reload(
            model_b.clone(),
            None,
            8192,
            SamplingParams::default(),
            FitParams::default(),
            KvCacheParams::default(),
        )
        .map_err(anyhow::Error::msg)?;

    let response_b = client
        .agent("local")
        .preamble("You are a helpful assistant. Answer in one short sentence.")
        .max_tokens(128)
        .build()
        .prompt("Say hello and mention you are model B.")
        .await?;
    println!("Model B: {response_b}");

    Ok(())
}
