use rig_agent::AgentBuilder;
use rig_llama_cpp::{CheckpointParams, Client, FitParams, KvCacheParams, SamplingParams};

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
        CheckpointParams::default(),
    )?;

    let response_a = AgentBuilder::new(client.completion_model("local"))
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
            CheckpointParams::default(),
        )
        .map_err(anyhow::Error::msg)?;

    // Take a fresh model handle: a handle snapshots the client's sampling
    // parameters when it is created.
    let response_b = AgentBuilder::new(client.completion_model("local"))
        .preamble("You are a helpful assistant. Answer in one short sentence.")
        .max_tokens(128)
        .build()
        .prompt("Say hello and mention you are model B.")
        .await?;
    println!("Model B: {response_b}");

    Ok(())
}
