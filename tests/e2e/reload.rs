//! Sequential model reload — validates llama.cpp backend re-init across loads.

use anyhow::ensure;
use rig_llama_cpp::{CheckpointParams, Client, FitParams, KvCacheParams, SamplingParams};
use serial_test::serial;

use super::common::{GEMMA, QWEN, ensure_model, env_parse_u32};

#[test]
#[serial(model)]
#[ignore = "downloads Qwen 3.5-2B and Gemma-4 E4B to validate sequential model reload"]
fn sequential_real_model_reload() -> anyhow::Result<()> {
    let first = ensure_model(&QWEN)?;
    let second = ensure_model(&GEMMA)?;
    ensure!(
        first.is_file(),
        "first model file not found at {}",
        first.display()
    );
    ensure!(
        second.is_file(),
        "second model file not found at {}",
        second.display()
    );

    let n_ctx = env_parse_u32("N_CTX", 8192);

    {
        let _client = Client::from_gguf(
            first.to_string_lossy().into_owned(),
            n_ctx,
            SamplingParams::default(),
            FitParams::default(),
            KvCacheParams::default(),
            CheckpointParams::default(),
        )?;
    }

    let _client = Client::from_gguf(
        second.to_string_lossy().into_owned(),
        n_ctx,
        SamplingParams::default(),
        FitParams::default(),
        KvCacheParams::default(),
        CheckpointParams::default(),
    )?;

    Ok(())
}
