use rig::client::CompletionClient;
use rig::completion::TypedPrompt;
use rig_llama_cpp::{Client, FitParams, KvCacheParams, SamplingParams};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Person {
    name: String,
    age: u32,
    occupation: String,
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
    )?;

    let agent = client
        .agent("local")
        .preamble("Extract a single person described in the user's text as structured data.")
        .max_tokens(256)
        .temperature(0.2)
        .build();

    // `prompt_typed::<T>` derives a JSON schema from `T`, constrains generation
    // via llama.cpp grammar, and deserializes the result into `T`.
    let person: Person = agent
        .prompt_typed("Ada is a 36-year-old software engineer living in Berlin.")
        .await?;

    println!("Extracted: {person:#?}");
    Ok(())
}
