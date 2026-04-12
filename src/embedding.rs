use std::num::NonZeroU32;
use std::thread;

use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel as _};
use tokio::sync::{mpsc, oneshot};

enum EmbeddingCommand {
    Request(EmbeddingRequest),
    Shutdown,
}

struct EmbeddingRequest {
    texts: Vec<String>,
    response_tx: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
}

/// The llama.cpp embedding client.
///
/// `EmbeddingClient` loads a GGUF embedding model on a dedicated worker thread
/// and exposes it through Rig's [`rig::embeddings::EmbeddingModel`] trait.
/// Create one with [`EmbeddingClient::from_gguf`].
///
/// ```rust,no_run
/// use rig::embeddings::EmbeddingModel;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), anyhow::Error> {
/// let client = rig_llama_cpp::EmbeddingClient::from_gguf(
///     "path/to/embedding-model.gguf",
///     99,   // n_gpu_layers
///     8192, // n_ctx
/// )?;
/// let model = client.embedding_model("local");
/// let embedding = model.embed_text("Hello, world!").await?;
/// println!("dims: {}", embedding.vec.len());
/// # Ok(())
/// # }
/// ```
pub struct EmbeddingClient {
    request_tx: mpsc::UnboundedSender<EmbeddingCommand>,
    ndims: usize,
    worker_handle: Option<thread::JoinHandle<()>>,
}

impl EmbeddingClient {
    /// Load a GGUF embedding model and start the embedding worker thread.
    ///
    /// # Arguments
    ///
    /// * `model_path` — Path to a `.gguf` embedding model file.
    /// * `n_gpu_layers` — Number of layers to offload to the GPU (`u32::MAX` for all).
    /// * `n_ctx` — Context window size in tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to initialize or the model cannot be loaded.
    pub fn from_gguf(
        model_path: impl Into<String>,
        n_gpu_layers: u32,
        n_ctx: u32,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.into();
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<EmbeddingCommand>();
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<usize, String>>();

        let worker_handle = thread::spawn(move || {
            embedding_worker(&model_path, n_gpu_layers, n_ctx, init_tx, &mut request_rx);
        });

        let ndims = init_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("Embedding thread panicked during initialization"))?
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            request_tx,
            ndims,
            worker_handle: Some(worker_handle),
        })
    }

    /// Create an embedding model handle from this client.
    pub fn embedding_model(&self, model: impl Into<String>) -> EmbeddingModelHandle {
        EmbeddingModelHandle::make(self, model, None)
    }

    /// Create an embedding model handle with explicit dimensions.
    pub fn embedding_model_with_ndims(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> EmbeddingModelHandle {
        EmbeddingModelHandle::make(self, model, Some(ndims))
    }
}

impl Drop for EmbeddingClient {
    fn drop(&mut self) {
        let _ = self.request_tx.send(EmbeddingCommand::Shutdown);

        if let Some(worker_handle) = self.worker_handle.take() {
            let _ = worker_handle.join();
        }
    }
}

/// A handle to a loaded embedding model that implements Rig's [`rig::embeddings::EmbeddingModel`] trait.
///
/// Obtained via [`EmbeddingClient::embedding_model`].
#[derive(Clone)]
pub struct EmbeddingModelHandle {
    request_tx: mpsc::UnboundedSender<EmbeddingCommand>,
    ndims: usize,
    #[allow(dead_code)]
    model_id: String,
}

impl rig::embeddings::EmbeddingModel for EmbeddingModelHandle {
    const MAX_DOCUMENTS: usize = 256;
    type Client = EmbeddingClient;

    fn make(client: &EmbeddingClient, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self {
            request_tx: client.request_tx.clone(),
            ndims: dims.unwrap_or(client.ndims),
            model_id: model.into(),
        }
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let texts: Vec<String> = texts.into_iter().collect();
        let documents = texts.clone();

        let (tx, rx) = oneshot::channel();
        self.request_tx
            .send(EmbeddingCommand::Request(EmbeddingRequest {
                texts,
                response_tx: tx,
            }))
            .map_err(|_| EmbeddingError::ProviderError("Embedding worker shut down".into()))?;

        let raw_embeddings = rx
            .await
            .map_err(|_| EmbeddingError::ProviderError("Response channel closed".into()))?
            .map_err(EmbeddingError::ProviderError)?;

        Ok(documents
            .into_iter()
            .zip(raw_embeddings)
            .map(|(doc, vec)| Embedding {
                document: doc,
                vec: vec.into_iter().map(|v| v as f64).collect(),
            })
            .collect())
    }
}

// === Embedding worker (runs on dedicated thread) ===

fn embedding_worker(
    model_path: &str,
    n_gpu_layers: u32,
    n_ctx: u32,
    init_tx: std::sync::mpsc::Sender<Result<usize, String>>,
    rx: &mut mpsc::UnboundedReceiver<EmbeddingCommand>,
) {
    use llama_cpp_2::list_llama_ggml_backend_devices;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::model::LlamaModel as LlamaCppModel;
    use llama_cpp_2::model::params::LlamaModelParams;

    let mut backend = {
        let mut attempts = 0;
        loop {
            match LlamaBackend::init() {
                Ok(b) => break b,
                Err(e) => {
                    attempts += 1;
                    if attempts >= 10 {
                        let _ = init_tx.send(Err(format!("Backend init failed: {e}")));
                        return;
                    }
                    // Previous backend is being torn down — wait briefly
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    };
    let logs_enabled = crate::llama_logs_enabled();

    if !logs_enabled {
        backend.void_logs();
        #[cfg(feature = "mtmd")]
        llama_cpp_2::mtmd::void_mtmd_logs();
    }

    let mut model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

    if backend.supports_gpu_offload() {
        let vulkan_devices: Vec<usize> = list_llama_ggml_backend_devices()
            .into_iter()
            .filter(|device| device.backend.eq_ignore_ascii_case("vulkan"))
            .map(|device| device.index)
            .collect();

        if !vulkan_devices.is_empty() {
            model_params = match model_params.with_devices(&vulkan_devices) {
                Ok(params) => {
                    if logs_enabled {
                        eprintln!("Using Vulkan backend devices: {vulkan_devices:?}");
                    }
                    params
                }
                Err(e) => {
                    let _ = init_tx.send(Err(format!("Failed to configure Vulkan devices: {e}")));
                    return;
                }
            };
        }
    }

    if logs_enabled {
        eprintln!("Loading embedding model from {model_path}...");
    }

    let model = match LlamaCppModel::load_from_file(&backend, model_path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            let _ = init_tx.send(Err(format!("Model load failed: {e}")));
            return;
        }
    };

    let ndims = model.n_embd() as usize;
    if logs_enabled {
        eprintln!("Embedding model loaded (ndims={ndims}).");
    }

    let _ = init_tx.send(Ok(ndims));

    while let Some(command) = rx.blocking_recv() {
        let req = match command {
            EmbeddingCommand::Request(req) => req,
            EmbeddingCommand::Shutdown => break,
        };

        let result = run_embedding(&backend, &model, n_ctx, &req.texts);
        let _ = req.response_tx.send(result);
    }
}

fn run_embedding(
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    model: &llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, String> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::AddBos;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx).map(Some).unwrap_or(None))
        .with_embeddings(true);

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("Embedding context creation failed: {e}"))?;

    let batch_limit = ctx.n_batch().max(1) as usize;

    // Tokenize all texts
    let tokenized: Vec<Vec<_>> = texts
        .iter()
        .map(|text| model.str_to_token(text, AddBos::Always))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("Tokenization failed: {e}"))?;

    let mut results = Vec::with_capacity(texts.len());
    let mut text_idx = 0;

    while text_idx < texts.len() {
        let mut batch = LlamaBatch::new(batch_limit, texts.len().min(batch_limit) as i32);
        let mut total_tokens = 0;
        let mut batch_seq_ids = Vec::new();
        let batch_start = text_idx;

        // Pack as many texts as fit in one batch
        while text_idx < texts.len() {
            let tokens = &tokenized[text_idx];
            if total_tokens + tokens.len() > batch_limit && !batch_seq_ids.is_empty() {
                break;
            }
            let seq_id = (text_idx - batch_start) as i32;
            for (pos, &token) in tokens.iter().enumerate() {
                batch
                    .add(token, pos as i32, &[seq_id], true)
                    .map_err(|e| format!("Batch add failed: {e}"))?;
            }
            batch_seq_ids.push(seq_id);
            total_tokens += tokens.len();
            text_idx += 1;
        }

        ctx.encode(&mut batch)
            .map_err(|e| format!("Embedding encode failed: {e}"))?;

        for &seq_id in &batch_seq_ids {
            let emb = ctx
                .embeddings_seq_ith(seq_id)
                .map_err(|e| format!("Failed to get embedding for seq {seq_id}: {e}"))?;
            results.push(emb.to_vec());
        }

        ctx.clear_kv_cache();
    }

    Ok(results)
}
