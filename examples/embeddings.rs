use rig::embeddings::EmbeddingModel;
use rig_llama_cpp::EmbeddingClient;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model_path = std::env::var("MODEL_PATH")
        .expect("Set MODEL_PATH env var to your GGUF embedding model file path");

    let n_gpu_layers = std::env::var("N_GPU_LAYERS")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(u32::MAX);

    let client = EmbeddingClient::from_gguf(&model_path, n_gpu_layers, 8192)?;
    let model = client.embedding_model("local");

    println!("Embedding dimensions: {}", model.ndims());

    // Embed a single text
    let embedding = model.embed_text("The sky is blue.").await?;
    println!(
        "Single embedding: {} dims, first 5: {:?}",
        embedding.vec.len(),
        &embedding.vec[..5]
    );

    // Embed multiple texts in one call
    let texts = vec![
        "The cat sat on the mat.".to_string(),
        "Dogs are loyal animals.".to_string(),
        "Machine learning models process data.".to_string(),
    ];
    let embeddings = model.embed_texts(texts).await?;

    for emb in &embeddings {
        println!(
            "  \"{}\": [{:.4}, {:.4}, {:.4}, ...]",
            &emb.document[..emb.document.len().min(40)],
            emb.vec[0],
            emb.vec[1],
            emb.vec[2],
        );
    }

    // Compute cosine similarity between pairs
    let sim_01 = cosine_similarity(&embeddings[0].vec, &embeddings[1].vec);
    let sim_02 = cosine_similarity(&embeddings[0].vec, &embeddings[2].vec);
    let sim_12 = cosine_similarity(&embeddings[1].vec, &embeddings[2].vec);

    println!("\nCosine similarities:");
    println!(
        "  '{}' vs '{}': {sim_01:.4}",
        embeddings[0].document, embeddings[1].document
    );
    println!(
        "  '{}' vs '{}': {sim_02:.4}",
        embeddings[0].document, embeddings[2].document
    );
    println!(
        "  '{}' vs '{}': {sim_12:.4}",
        embeddings[1].document, embeddings[2].document
    );

    Ok(())
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}
