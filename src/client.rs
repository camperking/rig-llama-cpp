use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use rig::client::CompletionClient;
use rig::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Usage,
};
use rig::streaming::StreamingCompletionResponse;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::error::LoadError;
use crate::request::prepare_request;
use crate::types::{
    CheckpointParams, FitParams, InferenceCommand, InferenceParams, InferenceRequest,
    KvCacheParams, RawResponse, ReloadRequest, ResponseChannel, SamplingParams, StreamChunk,
};
use crate::worker::{WorkerInit, inference_worker};

/// Default context window used by [`ClientBuilder`] when `n_ctx` is not set.
const DEFAULT_N_CTX: u32 = 4096;

/// Capacity of the inference command channel. Bounded to apply backpressure
/// to misbehaving callers (a flood of requests can't grow the worker's queue
/// without limit). Eight is generous for a single-worker llama.cpp client —
/// generation is the bottleneck, not enqueueing — and leaves headroom for
/// `Reload` / `Shutdown` to slip in alongside in-flight `Request`s.
const COMMAND_CHANNEL_CAPACITY: usize = 8;

/// Builder for [`Client`].
///
/// Construct one with [`Client::builder`], then chain optional setters and
/// finish with [`ClientBuilder::build`]. Every field except `model_path`
/// has a sensible default, so the minimal usage is:
///
/// ```rust,no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let client = rig_llama_cpp::Client::builder("path/to/model.gguf").build()?;
/// # let _ = client;
/// # Ok(())
/// # }
/// ```
///
/// The builder shape is forward-compatible: new optional knobs can be added
/// without breaking existing call sites.
#[must_use]
pub struct ClientBuilder {
    model_path: String,
    #[cfg(feature = "mtmd")]
    mmproj_path: Option<String>,
    n_ctx: u32,
    sampling: SamplingParams,
    fit: FitParams,
    kv_cache: KvCacheParams,
    checkpoint: CheckpointParams,
}

impl ClientBuilder {
    fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            #[cfg(feature = "mtmd")]
            mmproj_path: None,
            n_ctx: DEFAULT_N_CTX,
            sampling: SamplingParams::default(),
            fit: FitParams::default(),
            kv_cache: KvCacheParams::default(),
            checkpoint: CheckpointParams::default(),
        }
    }

    /// Desired context window size in tokens. Defaults to `4096`.
    pub fn n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    /// Token sampling parameters.
    pub fn sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = sampling;
        self
    }

    /// Automatic-fit parameters (per-device memory margins, minimum context).
    pub fn fit(mut self, fit: FitParams) -> Self {
        self.fit = fit;
        self
    }

    /// KV cache data-type configuration. Defaults to F16 / F16.
    pub fn kv_cache(mut self, kv_cache: KvCacheParams) -> Self {
        self.kv_cache = kv_cache;
        self
    }

    /// In-memory state-checkpoint cache tunables (used by hybrid/recurrent
    /// models to preserve KV state across turns).
    pub fn checkpoints(mut self, checkpoint: CheckpointParams) -> Self {
        self.checkpoint = checkpoint;
        self
    }

    /// Path to a multimodal projector (`mmproj`) GGUF file. Setting this
    /// switches the resulting [`Client`] into vision mode. Only available
    /// when the `mtmd` feature is enabled.
    #[cfg(feature = "mtmd")]
    pub fn mmproj(mut self, mmproj_path: impl Into<String>) -> Self {
        self.mmproj_path = Some(mmproj_path.into());
        self
    }

    /// Spawn the inference worker thread, load the model, and return a
    /// ready-to-use [`Client`].
    ///
    /// # Errors
    ///
    /// Returns a [`LoadError`] if the backend fails to initialise, automatic
    /// fitting fails, the GGUF file cannot be loaded, or — when `mmproj` was
    /// set — the multimodal projector cannot be initialised.
    pub fn build(self) -> Result<Client, LoadError> {
        #[cfg(feature = "mtmd")]
        let mmproj_path = self.mmproj_path;
        #[cfg(not(feature = "mtmd"))]
        let mmproj_path: Option<String> = None;

        Client::spawn(
            self.model_path,
            mmproj_path,
            self.n_ctx,
            self.sampling,
            self.fit,
            self.kv_cache,
            self.checkpoint,
        )
    }
}

/// The llama.cpp completion client.
///
/// `Client` loads a GGUF model on a dedicated inference thread and exposes it
/// through Rig's [`CompletionClient`] trait. Construct one with
/// [`Client::builder`], or — for backward-compatible positional construction —
/// [`Client::from_gguf`].
///
/// # Lifecycle
///
/// The worker thread owns the `LlamaModel`, `LlamaContext`, and (when the
/// `mtmd` feature is on) the multimodal projector. It only releases that
/// memory when it exits, which happens in two cases:
///
/// - On [`Client::reload`], the worker drops the old model and loads the new
///   one in place — the `Client` itself is **not** dropped, and the worker
///   thread is reused. Caller blocks on the reload result.
/// - On [`Client::drop`], the worker thread is signalled and joined. See
///   [`impl Drop for Client`](Client#impl-Drop-for-Client) for the exact
///   semantics — including how a long in-flight generation is cancelled so
///   the dropping thread doesn't have to wait for it to finish naturally.
pub struct Client {
    request_tx: mpsc::Sender<InferenceCommand>,
    /// Shared shutdown flag. Set by [`Client::drop`] so the worker's prompt
    /// prefill and sampling loops short-circuit at their next polling point.
    /// Cloned into the worker via [`WorkerInit::cancel`].
    cancel: Arc<AtomicBool>,
    sampling_params: std::sync::RwLock<SamplingParams>,
    worker_handle: Option<thread::JoinHandle<()>>,
}

impl Client {
    /// Start a [`ClientBuilder`] for a GGUF model at `model_path`.
    pub fn builder(model_path: impl Into<String>) -> ClientBuilder {
        ClientBuilder::new(model_path)
    }

    /// Load a GGUF model with automatic GPU/CPU layer fitting and start the inference worker thread.
    ///
    /// llama.cpp will probe available device memory and determine the optimal layer
    /// distribution automatically.
    ///
    /// Prefer [`Client::builder`] for new code — this constructor is kept for
    /// backward compatibility with the positional 0.1.x API and forwards
    /// directly to the builder.
    ///
    /// # Arguments
    ///
    /// * `model_path` — Path to a `.gguf` model file.
    /// * `n_ctx` — Desired context window size in tokens.
    /// * `sampling_params` — Sampling parameters for token generation.
    /// * `fit_params` — Configuration for the fitting algorithm.
    /// * `kv_cache_params` — KV cache data-type configuration (defaults to F16/F16).
    /// * `checkpoint_params` — Tunables for the in-memory state-checkpoint cache
    ///   used to preserve KV/recurrent state across chat turns for hybrid models.
    ///
    /// # Errors
    ///
    /// Returns a [`LoadError`] if the backend fails to initialise, automatic
    /// fitting fails, or the model cannot be loaded.
    pub fn from_gguf(
        model_path: impl Into<String>,
        n_ctx: u32,
        sampling_params: SamplingParams,
        fit_params: FitParams,
        kv_cache_params: KvCacheParams,
        checkpoint_params: CheckpointParams,
    ) -> Result<Self, LoadError> {
        Self::spawn(
            model_path.into(),
            None,
            n_ctx,
            sampling_params,
            fit_params,
            kv_cache_params,
            checkpoint_params,
        )
    }

    /// Load a GGUF vision model with a multimodal projector and automatic GPU/CPU layer fitting.
    ///
    /// This constructor enables multimodal (vision) inference. The `mmproj_path` should point
    /// to a GGUF multimodal projector file (mmproj) that corresponds to the vision model.
    ///
    /// Prefer [`Client::builder`] with [`ClientBuilder::mmproj`] for new code.
    ///
    /// # Arguments
    ///
    /// * `model_path` — Path to a `.gguf` vision model file.
    /// * `mmproj_path` — Path to the corresponding multimodal projector `.gguf` file.
    /// * `n_ctx` — Desired context window size in tokens.
    /// * `sampling_params` — Sampling parameters for token generation.
    /// * `fit_params` — Configuration for the fitting algorithm.
    /// * `kv_cache_params` — KV cache data-type configuration (defaults to F16/F16).
    ///
    /// # Errors
    ///
    /// Returns a [`LoadError`] if the backend fails to initialise, the model
    /// cannot be loaded, or the multimodal projector cannot be initialised.
    #[cfg(feature = "mtmd")]
    pub fn from_gguf_with_mmproj(
        model_path: impl Into<String>,
        mmproj_path: impl Into<String>,
        n_ctx: u32,
        sampling_params: SamplingParams,
        fit_params: FitParams,
        kv_cache_params: KvCacheParams,
        checkpoint_params: CheckpointParams,
    ) -> Result<Self, LoadError> {
        Self::spawn(
            model_path.into(),
            Some(mmproj_path.into()),
            n_ctx,
            sampling_params,
            fit_params,
            kv_cache_params,
            checkpoint_params,
        )
    }

    /// Shared spawn path used by the builder and by the positional constructors.
    /// `mmproj_path` is only consulted when the `mtmd` feature is enabled; with
    /// the feature off, callers always pass `None` and the worker thread
    /// silently ignores any value.
    fn spawn(
        model_path: String,
        mmproj_path: Option<String>,
        n_ctx: u32,
        sampling_params: SamplingParams,
        fit_params: FitParams,
        kv_cache_params: KvCacheParams,
        checkpoint_params: CheckpointParams,
    ) -> Result<Self, LoadError> {
        let (request_tx, mut request_rx) =
            mpsc::channel::<InferenceCommand>(COMMAND_CHANNEL_CAPACITY);
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), LoadError>>();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);

        let worker_handle = thread::spawn(move || {
            let init = WorkerInit {
                model_path: &model_path,
                mmproj_path: mmproj_path.as_deref(),
                n_ctx,
                fit_params: &fit_params,
                kv_cache_params: &kv_cache_params,
                checkpoint_params,
                cancel: worker_cancel,
            };
            inference_worker(init, init_tx, &mut request_rx);
        });

        init_rx
            .recv()
            .map_err(|_| LoadError::WorkerInitDisconnected)??;

        Ok(Self {
            request_tx,
            cancel,
            sampling_params: std::sync::RwLock::new(sampling_params),
            worker_handle: Some(worker_handle),
        })
    }

    /// Reload the worker thread with a new model without destroying the backend.
    ///
    /// This swaps the model in-place on the existing inference thread, avoiding the
    /// `LlamaBackend` singleton re-initialization race that occurs when dropping and
    /// recreating a `Client`.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::WorkerNotRunning`] if the inference worker is no
    /// longer accepting commands, or any of the load-stage variants if the
    /// new model fails to come up.
    // The positional signature is part of the 0.1.x public API. A future minor
    // release can introduce a `ReloadOptions`/`reload_builder` shape; until
    // then, the eight params (self + 7 fields) intentionally stay positional.
    #[allow(clippy::too_many_arguments)]
    pub fn reload(
        &self,
        model_path: String,
        mmproj_path: Option<String>,
        n_ctx: u32,
        sampling: SamplingParams,
        fit_params: FitParams,
        kv_cache_params: KvCacheParams,
        checkpoint_params: CheckpointParams,
    ) -> Result<(), LoadError> {
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        // `blocking_send` is the right call here: `reload` is a sync API and
        // is documented to be invoked from a `spawn_blocking` task (or any
        // non-async thread) when used from a tokio context. Backpressure on a
        // full command queue is fine — reload is itself a blocking operation.
        self.request_tx
            .blocking_send(InferenceCommand::Reload(ReloadRequest {
                model_path,
                mmproj_path,
                n_ctx,
                fit_params,
                kv_cache_params,
                checkpoint_params,
                result_tx,
            }))
            .map_err(|_| LoadError::WorkerNotRunning)?;
        let result = result_rx
            .recv()
            .map_err(|_| LoadError::WorkerInitDisconnected)?;
        if result.is_ok() {
            // SamplingParams is `Copy` (just numeric scalars) — a poisoned
            // lock can't represent torn or invalid data, so recover the
            // guard rather than propagate a panic.
            let mut guard = self
                .sampling_params
                .write()
                .unwrap_or_else(|p| p.into_inner());
            *guard = sampling;
        }
        result
    }
}

impl Drop for Client {
    /// Tear down the worker thread synchronously.
    ///
    /// `Drop` blocks until the worker thread has fully exited and the
    /// `LlamaModel` / `LlamaContext` (and `LlamaBackend` device handles, plus
    /// the multimodal projector when the `mtmd` feature is on) are released.
    /// This is intentional: the caller almost always wants to allocate a
    /// replacement `Client` immediately after dropping this one, and a
    /// non-blocking drop would briefly hold 2× the model's RAM/VRAM and risk
    /// OOM. [`Client::reload`] reuses the same worker and avoids this whole
    /// path; prefer it over drop-and-recreate when you can.
    ///
    /// To keep the wait short even when a long generation is mid-flight,
    /// `Drop` flips the shared cancel flag before signalling shutdown. The
    /// worker polls the flag at every prompt-prefill chunk boundary and
    /// every sampled token, so an in-flight `Request` returns within a
    /// single decode step. The pessimal wait is therefore one decode step,
    /// not the rest of the generation.
    ///
    /// `try_send(Shutdown)` is best-effort: if the bounded command queue is
    /// full at this instant, the `Shutdown` command isn't enqueued — but the
    /// in-flight request still bails on the cancel flag, and the worker's
    /// per-iteration cancel check at the top of its command loop also exits
    /// the thread before pulling more queued commands.
    ///
    /// `Model` clones outliving the `Client` keep the channel sender count
    /// above zero; their `send` calls naturally fail with `SendError` once
    /// the receiver is dropped on worker exit, so they don't prevent
    /// shutdown.
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        let _ = self.request_tx.try_send(InferenceCommand::Shutdown);

        if let Some(worker_handle) = self.worker_handle.take() {
            let _ = worker_handle.join();
        }
    }
}

impl CompletionClient for Client {
    type CompletionModel = Model;
}

/// A handle to a loaded model that implements Rig's [`CompletionModel`] trait.
///
/// Obtained via [`CompletionClient::agent`] on a [`Client`].
#[derive(Clone)]
pub struct Model {
    request_tx: mpsc::Sender<InferenceCommand>,
    sampling_params: SamplingParams,
    #[allow(dead_code)]
    model_id: String,
}

impl CompletionModel for Model {
    type Response = RawResponse;
    type StreamingResponse = StreamChunk;
    type Client = Client;

    fn make(client: &Client, model: impl Into<String>) -> Self {
        // See the matching `unwrap_or_else` in `reload`: SamplingParams is
        // `Copy`, so a poisoned lock still holds valid data.
        let sampling_params = *client
            .sampling_params
            .read()
            .unwrap_or_else(|p| p.into_inner());
        Self {
            request_tx: client.request_tx.clone(),
            sampling_params,
            model_id: model.into(),
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let prepared_request = prepare_request(&request).map_err(CompletionError::ProviderError)?;
        let max_tokens = request.max_tokens.unwrap_or(512) as u32;
        let temperature = request.temperature.unwrap_or(0.7) as f32;

        let (response_tx, response_rx) = oneshot::channel();

        self.request_tx
            .send(InferenceCommand::Request(InferenceRequest {
                params: InferenceParams {
                    prepared_request,
                    max_tokens,
                    temperature,
                    top_p: self.sampling_params.top_p,
                    top_k: self.sampling_params.top_k,
                    min_p: self.sampling_params.min_p,
                    presence_penalty: self.sampling_params.presence_penalty,
                    repetition_penalty: self.sampling_params.repetition_penalty,
                },
                response_channel: ResponseChannel::Completion(response_tx),
            }))
            .await
            .map_err(|_| CompletionError::ProviderError("Inference thread shut down".into()))?;

        let result = response_rx
            .await
            .map_err(|_| CompletionError::ProviderError("Response channel closed".into()))?
            .map_err(CompletionError::ProviderError)?;

        Ok(CompletionResponse {
            choice: result.choice,
            usage: Usage {
                input_tokens: result.prompt_tokens,
                output_tokens: result.completion_tokens,
                total_tokens: result.prompt_tokens + result.completion_tokens,
                cached_input_tokens: result.cached_input_tokens,
                cache_creation_input_tokens: 0,
            },
            raw_response: RawResponse { text: result.text },
            message_id: None,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let prepared_request = prepare_request(&request).map_err(CompletionError::ProviderError)?;
        let max_tokens = request.max_tokens.unwrap_or(512) as u32;
        let temperature = request.temperature.unwrap_or(0.7) as f32;

        let (stream_tx, stream_rx) = mpsc::unbounded_channel();

        self.request_tx
            .send(InferenceCommand::Request(InferenceRequest {
                params: InferenceParams {
                    prepared_request,
                    max_tokens,
                    temperature,
                    top_p: self.sampling_params.top_p,
                    top_k: self.sampling_params.top_k,
                    min_p: self.sampling_params.min_p,
                    presence_penalty: self.sampling_params.presence_penalty,
                    repetition_penalty: self.sampling_params.repetition_penalty,
                },
                response_channel: ResponseChannel::Streaming(stream_tx),
            }))
            .await
            .map_err(|_| CompletionError::ProviderError("Inference thread shut down".into()))?;

        Ok(StreamingCompletionResponse::stream(Box::pin(
            UnboundedReceiverStream::new(stream_rx),
        )))
    }
}
