//! Embedding integration test.

use anyhow::ensure;
use rig::embeddings::EmbeddingModel as _;
use rig_llama_cpp::EmbeddingClient;
use serial_test::serial;

use super::common::{NOMIC_EMBED, ensure_model, env_parse_u32};

#[tokio::test(flavor = "multi_thread")]
#[serial(model)]
#[ignore = "downloads nomic-embed-text-v2-moe and validates embedding output"]
async fn embedding_basic() -> anyhow::Result<()> {
    let path = ensure_model(&NOMIC_EMBED)?;
    let n_gpu_layers = env_parse_u32("N_GPU_LAYERS", u32::MAX);
    let n_ctx = env_parse_u32("N_CTX", 8192);

    let client =
        EmbeddingClient::from_gguf(path.to_string_lossy().into_owned(), n_gpu_layers, n_ctx)?;
    let model = client.embedding_model("local");

    // Single text embedding
    let emb = model.embed_text("Hello, world!").await?;
    ensure!(
        emb.vec.len() == model.ndims(),
        "embedding dimension mismatch: got {}, expected {}",
        emb.vec.len(),
        model.ndims()
    );
    ensure!(
        emb.vec.iter().any(|v| *v != 0.0),
        "embedding should not be all zeros"
    );

    // Multiple texts
    let embeddings = model
        .embed_texts(vec![
            "The cat sat on the mat.".to_string(),
            "Dogs are loyal animals.".to_string(),
            "The weather is sunny today.".to_string(),
        ])
        .await?;
    ensure!(
        embeddings.len() == 3,
        "expected 3 embeddings, got {}",
        embeddings.len()
    );
    for (i, emb) in embeddings.iter().enumerate() {
        ensure!(
            emb.vec.len() == model.ndims(),
            "embedding {i} dimension mismatch: got {}, expected {}",
            emb.vec.len(),
            model.ndims()
        );
    }

    println!(
        "Embedding test passed: ndims={}, single_ok=true, batch_count={}",
        model.ndims(),
        embeddings.len()
    );

    Ok(())
}
