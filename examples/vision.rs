#[cfg(not(feature = "mtmd"))]
fn main() {
    eprintln!(
        "This example requires the `mtmd` feature: cargo run --features mtmd --example vision"
    );
    std::process::exit(1);
}

#[cfg(feature = "mtmd")]
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    use rig::OneOrMany;
    use rig::client::CompletionClient;
    use rig::completion::CompletionModel;
    use rig::message::{DocumentSourceKind, Image, ImageMediaType, Message, UserContent};
    use rig_llama_cpp::{Client, SamplingParams};

    let model_path =
        std::env::var("MODEL_PATH").expect("Set MODEL_PATH env var to your vision GGUF model");
    let mmproj_path =
        std::env::var("MMPROJ_PATH").expect("Set MMPROJ_PATH env var to your mmproj GGUF file");
    let image_path =
        std::env::var("IMAGE_PATH").expect("Set IMAGE_PATH env var to an image file path");

    let n_gpu_layers = std::env::var("N_GPU_LAYERS")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(u32::MAX);

    let image_bytes = std::fs::read(&image_path)?;

    let client = Client::from_gguf_with_mmproj(
        &model_path,
        &mmproj_path,
        n_gpu_layers,
        8192,
        SamplingParams::default(),
    )?;

    let model = client.completion_model("local");

    let response = model
        .completion_request("Describe this image.")
        .preamble("You are a helpful assistant that can describe images.".to_string())
        .messages(vec![Message::from(OneOrMany::many(vec![
            UserContent::Image(Image {
                media_type: Some(ImageMediaType::PNG),
                data: DocumentSourceKind::Raw(image_bytes),
                detail: None,
                additional_params: None,
            }),
            UserContent::text("What do you see in this image? Describe it in detail."),
        ])?)])
        .max_tokens(512)
        .send()
        .await?;

    println!("{}", response.raw_response.text);

    Ok(())
}
