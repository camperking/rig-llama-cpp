use std::num::NonZeroU32;
use std::thread;

use rig_core::driver::{Exchange, Local, Opened, Opening, Step, Transport};
use rig_core::embeddings::{Embedding, EmbeddingResponse};
use rig_core::error::ProviderError;
use rig_core::operation::Embedding as EmbeddingOp;
use rig_core::wire::Capabilities;
use tokio::sync::{mpsc, oneshot};

use crate::error::LoadError;

/// The most texts one embedding request takes.
const MAX_DOCUMENTS: usize = 256;

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
/// and exposes it as a Rig embedding [`Model`](rig_core::Model) (see
/// [`EmbeddingModelHandle`]). Create one with [`EmbeddingClient::from_gguf`].
///
/// ```rust,no_run
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    /// Returns a [`LoadError`] if the backend fails to initialise or the
    /// model cannot be loaded.
    pub fn from_gguf(
        model_path: impl Into<String>,
        n_gpu_layers: u32,
        n_ctx: u32,
    ) -> Result<Self, LoadError> {
        let model_path = model_path.into();
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<EmbeddingCommand>();
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<usize, LoadError>>();

        let worker_handle = thread::spawn(move || {
            embedding_worker(&model_path, n_gpu_layers, n_ctx, init_tx, &mut request_rx);
        });

        let ndims = init_rx
            .recv()
            .map_err(|_| LoadError::WorkerInitDisconnected)??;

        Ok(Self {
            request_tx,
            ndims,
            worker_handle: Some(worker_handle),
        })
    }

    /// Create an embedding model handle from this client.
    pub fn embedding_model(&self, model: impl Into<String>) -> EmbeddingModelHandle {
        self.embedding_model_with_ndims(model, self.ndims)
    }

    /// Create an embedding model handle with explicit dimensions.
    pub fn embedding_model_with_ndims(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> EmbeddingModelHandle {
        let wire = Local::new(crate::PROVIDER)
            .with_id(model)
            .with_capabilities(Capabilities::embedding(MAX_DOCUMENTS, ndims));
        rig_core::Model::new(
            wire,
            EmbeddingRuntime {
                request_tx: self.request_tx.clone(),
            },
        )
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

/// A loaded embedding model: Rig's local embedding wire over the embedding
/// worker.
///
/// Obtained via [`EmbeddingClient::embedding_model`]. Call it directly
/// (`model.embed_text(..)`, `model.call(texts)`), or erase it with
/// `.erase()` / `.into()` wherever Rig takes a `DynModel<Embedding>`, such as
/// `EmbeddingsBuilder`.
pub type EmbeddingModelHandle = rig_core::Model<Local<EmbeddingOp>, EmbeddingRuntime>;

/// The transport behind [`EmbeddingModelHandle`]: it hands each batch of
/// texts to the embedding worker thread.
#[derive(Clone)]
pub struct EmbeddingRuntime {
    request_tx: mpsc::UnboundedSender<EmbeddingCommand>,
}

impl Transport<Local<EmbeddingOp>> for EmbeddingRuntime {
    fn send(&self, texts: Vec<String>, _exchange: Exchange) -> Opening<Step<EmbeddingOp>> {
        let request_tx = self.request_tx.clone();
        Opening::new(async move {
            let (tx, rx) = oneshot::channel();
            request_tx
                .send(EmbeddingCommand::Request(EmbeddingRequest {
                    texts,
                    response_tx: tx,
                }))
                .map_err(|_| ProviderError::Provider("Embedding worker shut down".into()))?;

            let vectors = rx
                .await
                .map_err(|_| ProviderError::Provider("Response channel closed".into()))?
                .map_err(ProviderError::Provider)?;

            // Rig's embedding fold pairs each vector with its input text.
            let response = EmbeddingResponse::new(
                vectors
                    .into_iter()
                    .map(|vec| Embedding {
                        document: String::new(),
                        vec: vec.into_iter().map(f64::from).collect(),
                    })
                    .collect(),
            );
            Ok(Opened::new(tokio_stream::iter([Ok(Step::End(response))])))
        })
    }
}

// === Embedding worker (runs on dedicated thread) ===

fn embedding_worker(
    model_path: &str,
    n_gpu_layers: u32,
    n_ctx: u32,
    init_tx: std::sync::mpsc::Sender<Result<usize, LoadError>>,
    rx: &mut mpsc::UnboundedReceiver<EmbeddingCommand>,
) {
    use llama_cpp_2::list_llama_ggml_backend_devices;
    use llama_cpp_2::model::LlamaModel as LlamaCppModel;
    use llama_cpp_2::model::params::LlamaModelParams;

    let backend = match crate::shared_backend() {
        Ok(b) => b,
        Err(e) => {
            let _ = init_tx.send(Err(LoadError::BackendInit(e)));
            return;
        }
    };
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
                    log::info!("Using Vulkan backend devices: {vulkan_devices:?}");
                    params
                }
                Err(e) => {
                    let _ = init_tx.send(Err(LoadError::ConfigureDevices(e.to_string())));
                    return;
                }
            };
        }
    }

    log::info!("Loading embedding model from {model_path}...");

    let model = match LlamaCppModel::load_from_file(backend, model_path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            let _ = init_tx.send(Err(LoadError::ModelLoad(e.to_string())));
            return;
        }
    };

    let ndims = model.n_embd() as usize;
    log::info!("Embedding model loaded (ndims={ndims}).");

    let _ = init_tx.send(Ok(ndims));

    while let Some(command) = rx.blocking_recv() {
        let req = match command {
            EmbeddingCommand::Request(req) => req,
            EmbeddingCommand::Shutdown => break,
        };

        let result = run_embedding(backend, &model, n_ctx, &req.texts);
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
    // Encoder-only embedding models require `n_ubatch >= n_tokens` for
    // every single sequence in the batch (llama.cpp asserts on this).
    // The default `n_ubatch` is 512, which is smaller than a typical
    // chunk's token count, so we widen both the batch and micro-batch to
    // match `n_ctx`. This lets any chunk that fits the context window also
    // fit in one micro-batch step.
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx).map(Some).unwrap_or(None))
        .with_n_batch(n_ctx)
        .with_n_ubatch(n_ctx)
        .with_n_seq_max(1)
        .with_embeddings(true);

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("Embedding context creation failed: {e}"))?;

    let batch_limit = ctx.n_batch().max(1) as usize;

    // Encode each text in its own single-sequence batch. Packing multiple
    // sequences into one batch trips a `ggml_can_mul_mat` assert on
    // mixture-of-experts architectures (e.g. nomic-embed-text-v2-moe) where
    // expert routing produces incompatible tensor shapes across concurrent
    // sequences. One-at-a-time encoding is correct for every architecture at
    // the cost of some throughput.
    let mut results = Vec::with_capacity(texts.len());
    for text in texts {
        let tokens = model.vocab().tokenize(text.as_bytes(), true, true);

        // Tokenise in chunks that fit `batch_limit` (very long inputs).
        for chunk in tokens.chunks(batch_limit) {
            let mut batch = LlamaBatch::new(batch_limit, 1);
            for (pos, &token) in chunk.iter().enumerate() {
                batch
                    .add(token, pos as i32, &[0], true)
                    .map_err(|e| format!("Batch add failed: {e}"))?;
            }
            ctx.encode(&mut batch)
                .map_err(|e| format!("Embedding encode failed: {e}"))?;
        }

        let emb = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| format!("Failed to get embedding: {e}"))?;
        results.push(emb.to_vec());

        ctx.clear_kv_cache();
    }

    Ok(results)
}
