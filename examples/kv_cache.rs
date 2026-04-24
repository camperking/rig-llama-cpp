use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig_llama_cpp::{Client, FitParams, KvCacheParams, KvCacheType, SamplingParams};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model_path =
        std::env::var("MODEL_PATH").expect("Set MODEL_PATH env var to your GGUF model file path");

    // Quantize both K and V caches to Q8_0 to roughly halve KV-cache VRAM usage
    // at long `n_ctx`, at a small accuracy cost. Try `Q4_0` for ~1/4 VRAM.
    let kv_cache = KvCacheParams {
        type_k: KvCacheType::Q8_0,
        type_v: KvCacheType::Q8_0,
    };

    let client = Client::from_gguf(
        &model_path,
        32_768,
        SamplingParams::default(),
        FitParams::default(),
        kv_cache,
    )?;

    let response = client
        .agent("local")
        .preamble("You are a helpful assistant.")
        .max_tokens(256)
        .build()
        .prompt("In one sentence, what is KV-cache quantization?")
        .await?;

    println!("{response}");
    Ok(())
}
