use std::thread;

use rig::client::CompletionClient;
use rig::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Usage,
};
use rig::streaming::StreamingCompletionResponse;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::request::prepare_request;
use crate::types::{
    FitParams, InferenceCommand, InferenceParams, InferenceRequest, RawResponse, ReloadRequest,
    ResponseChannel, SamplingParams, StreamChunk,
};
use crate::worker::inference_worker;

/// The llama.cpp completion client.
///
/// `Client` loads a GGUF model on a dedicated inference thread and exposes it
/// through Rig's [`CompletionClient`] trait. Create one with [`Client::from_gguf`].
pub struct Client {
    request_tx: mpsc::UnboundedSender<InferenceCommand>,
    sampling_params: std::sync::RwLock<SamplingParams>,
    worker_handle: Option<thread::JoinHandle<()>>,
}

impl Client {
    /// Load a GGUF model with automatic GPU/CPU layer fitting and start the inference worker thread.
    ///
    /// llama.cpp will probe available device memory and determine the optimal layer
    /// distribution automatically.
    ///
    /// # Arguments
    ///
    /// * `model_path` — Path to a `.gguf` model file.
    /// * `n_ctx` — Desired context window size in tokens.
    /// * `sampling_params` — Sampling parameters for token generation.
    /// * `fit_params` — Configuration for the fitting algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to initialize or the model cannot be loaded.
    pub fn from_gguf(
        model_path: impl Into<String>,
        n_ctx: u32,
        sampling_params: SamplingParams,
        fit_params: FitParams,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.into();
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<InferenceCommand>();
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();

        let worker_handle = thread::spawn(move || {
            inference_worker(
                &model_path,
                None,
                n_ctx,
                &fit_params,
                init_tx,
                &mut request_rx,
            );
        });

        init_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("Inference thread panicked during initialization"))?
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            request_tx,
            sampling_params: std::sync::RwLock::new(sampling_params),
            worker_handle: Some(worker_handle),
        })
    }

    /// Load a GGUF vision model with a multimodal projector and automatic GPU/CPU layer fitting.
    ///
    /// This constructor enables multimodal (vision) inference. The `mmproj_path` should point
    /// to a GGUF multimodal projector file (mmproj) that corresponds to the vision model.
    ///
    /// # Arguments
    ///
    /// * `model_path` — Path to a `.gguf` vision model file.
    /// * `mmproj_path` — Path to the corresponding multimodal projector `.gguf` file.
    /// * `n_ctx` — Desired context window size in tokens.
    /// * `sampling_params` — Sampling parameters for token generation.
    /// * `fit_params` — Configuration for the fitting algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to initialize, the model cannot be loaded,
    /// or the multimodal projector cannot be initialized.
    #[cfg(feature = "mtmd")]
    pub fn from_gguf_with_mmproj(
        model_path: impl Into<String>,
        mmproj_path: impl Into<String>,
        n_ctx: u32,
        sampling_params: SamplingParams,
        fit_params: FitParams,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.into();
        let mmproj_path = mmproj_path.into();
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<InferenceCommand>();
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();

        let worker_handle = thread::spawn(move || {
            inference_worker(
                &model_path,
                Some(&mmproj_path),
                n_ctx,
                &fit_params,
                init_tx,
                &mut request_rx,
            );
        });

        init_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("Inference thread panicked during initialization"))?
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            request_tx,
            sampling_params: std::sync::RwLock::new(sampling_params),
            worker_handle: Some(worker_handle),
        })
    }

    /// Reload the worker thread with a new model without destroying the backend.
    ///
    /// This swaps the model in-place on the existing inference thread, avoiding the
    /// `LlamaBackend` singleton re-initialization race that occurs when dropping and
    /// recreating a `Client`.
    pub fn reload(
        &self,
        model_path: String,
        mmproj_path: Option<String>,
        n_ctx: u32,
        sampling: SamplingParams,
        fit_params: FitParams,
    ) -> Result<(), String> {
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        self.request_tx
            .send(InferenceCommand::Reload(ReloadRequest {
                model_path,
                mmproj_path,
                n_ctx,
                fit_params,
                result_tx,
            }))
            .map_err(|_| "Worker thread not running".to_string())?;
        let result = result_rx
            .recv()
            .map_err(|_| "Worker thread exited during reload".to_string())?;
        if result.is_ok() {
            *self.sampling_params.write().unwrap() = sampling;
        }
        result
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.request_tx.send(InferenceCommand::Shutdown);

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
    request_tx: mpsc::UnboundedSender<InferenceCommand>,
    sampling_params: SamplingParams,
    #[allow(dead_code)]
    model_id: String,
}

impl CompletionModel for Model {
    type Response = RawResponse;
    type StreamingResponse = StreamChunk;
    type Client = Client;

    fn make(client: &Client, model: impl Into<String>) -> Self {
        Self {
            request_tx: client.request_tx.clone(),
            sampling_params: *client.sampling_params.read().unwrap(),
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
                cached_input_tokens: 0,
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
            .map_err(|_| CompletionError::ProviderError("Inference thread shut down".into()))?;

        Ok(StreamingCompletionResponse::stream(Box::pin(
            UnboundedReceiverStream::new(stream_rx),
        )))
    }
}
