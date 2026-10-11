//! The Rig side of a loaded model: the completion wire that describes it to
//! Rig and decodes its replies, and the runtime that carries each request to
//! the inference worker.
//!
//! Rig drives a [`Model`] as a [`CompletionWire`] bound to a [`Runtime`]
//! transport. The wire's payload is the prepared [`CompletionRequest`]
//! itself; the runtime hands it to the worker thread, whose [`Frame`]s the
//! wire's decoder writes into Rig's completion turn.

use rig_core::completion::options::{Mapping, OptionFields, OptionMap};
use rig_core::completion::{Accepts, CompletionRequest, LaterSystem, Media, ReplayTarget, Usage};
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::message::{Api, DocumentSourceKind};
use rig_core::operation::{Block, CallFragment, Completion, Finish};
use rig_core::wire::document::{Reassemble, Serves};
use rig_core::wire::{Decoder, Descriptor, Flow, Mode, Out, Wire, WireEvent};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::request::prepare_request;
use crate::types::{
    Done, Frame, FrameKind, InferenceCommand, InferenceParams, InferenceRequest, OutputPart,
    RawResponse, ResponseChannel, SamplingParams,
};

/// The wire format this crate's turns record as their origin.
const API: Api = Api::from_static("llama.cpp.chat-template");

/// `max_tokens` when the request names none.
const DEFAULT_MAX_TOKENS: u32 = 512;

/// `temperature` when the request names none.
const DEFAULT_TEMPERATURE: f32 = 0.7;

/// Seed of the final sampler when the request names none.
const DEFAULT_SEED: u32 = 42;

/// A loaded model as Rig drives it: [`CompletionWire`] over [`Runtime`].
///
/// Obtained via [`Client::completion_model`](crate::Client::completion_model).
/// Call it directly (`model.call(request)`, `model.stream(request)`), or hand
/// it to anything that takes `impl Into<DynModel<Completion>>`, such as
/// `rig_agent::AgentBuilder::new`. Clones share the loaded model.
pub type Model = rig_core::Model<CompletionWire, Runtime>;

/// The completion wire of a local llama.cpp model.
///
/// Its payload is the request as Rig prepared it, and its replies are the
/// inference worker's [`Frame`]s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionWire {
    model: String,
}

impl CompletionWire {
    pub(crate) fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl ReplayTarget for CompletionWire {
    fn api(&self) -> Api {
        API
    }

    fn provider(&self) -> &str {
        crate::PROVIDER
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Images reach the model only through a multimodal projector, so a build
    /// without `mtmd` reads text and tools alone. Assistant images are never
    /// rendered into the prompt.
    fn accepts(&self, _model: &str) -> Accepts {
        if cfg!(feature = "mtmd") {
            Accepts {
                assistant_images: false,
                ..Accepts::ALL
            }
        } else {
            Accepts::TEXT
        }
    }

    /// Images must carry their bytes: there is no network access to fetch a
    /// URL or a provider file id, so those become placeholders instead.
    fn encodes(&self, _model: &str, media: Media<'_>) -> bool {
        match media {
            Media::Image(image, _) => matches!(
                image.data,
                DocumentSourceKind::Raw(_) | DocumentSourceKind::Base64(_)
            ),
            _ => true,
        }
    }

    fn map_options(&self, _request: &CompletionRequest, fields: OptionFields<'_>) -> OptionMap {
        let OptionFields {
            reasoning,
            cache,
            service_tier,
            verbosity,
            parallel_tool_calls,
            top_p,
            seed,
            stop,
        } = fields;
        OptionMap {
            reasoning: Mapping::of(reasoning, |_| {
                Mapping::Omit("passed to the chat template as `enable_thinking`")
            }),
            cache: Mapping::of(
                cache,
                refuse("the persistent KV cache is configured on the client"),
            ),
            service_tier: Mapping::of(service_tier, refuse("a local model has no tiers")),
            verbosity: Mapping::of(verbosity, refuse("no verbosity control")),
            parallel_tool_calls: Mapping::of(
                parallel_tool_calls,
                refuse("tool calls are parsed from free-form output"),
            ),
            top_p: Mapping::of(top_p, |_| Mapping::Omit("applied by the local sampler")),
            seed: Mapping::of(seed, |_| Mapping::Omit("seeds the local sampler")),
            stop: Mapping::of_stop(stop, |_| Mapping::unsupported("no stop sequences")),
        }
    }

    /// Generation ends at an end-of-generation token or the token budget;
    /// neither is reported as a finish reason.
    fn states_finish_reason(&self) -> bool {
        false
    }

    /// Chat templates commonly accept a system message only at the start.
    fn later_system(&self, _model: &str) -> LaterSystem {
        LaterSystem::Leading
    }
}

/// The answer for an option this wire cannot honour.
fn refuse<T>(reason: &'static str) -> impl FnOnce(T) -> Mapping {
    move |_| Mapping::unsupported(reason)
}

impl Wire for CompletionWire {
    type Op = Completion;
    type Payload = CompletionRequest;
    type Frame = Frame;
    type Decoder<'id> = FrameDecoder<'id>;
    type Reassembler = RawDocument;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(crate::PROVIDER)
            .model(Some(self.model.as_str()))
            .replay(self)
    }

    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<Self::Payload, EncodeError> {
        Ok(request)
    }

    fn decoder<'id>(&self) -> FrameDecoder<'id> {
        FrameDecoder::default()
    }
}

/// The decoder of [`CompletionWire`]: writes each worker [`Frame`] into the
/// reply.
#[derive(Default)]
pub struct FrameDecoder<'id> {
    brand: std::marker::PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id> Decoder<'id, Completion, Frame> for FrameDecoder<'id> {
    type Event = Frame;

    fn classify(&self, frame: Frame) -> WireEvent<Frame> {
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        frame: Frame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match frame.0 {
            // Streamed text arrives one token piece at a time; consecutive
            // pieces extend one text block.
            FrameKind::Part(OutputPart::Text(text)) => {
                if !text.is_empty() {
                    out.run(Block::Text, &text)?;
                }
            }
            FrameKind::Part(OutputPart::Reasoning(text)) => {
                out.end_run()?;
                let index = out.fresh_index();
                out.whole(
                    index,
                    Block::Reasoning { redacted: false },
                    serde_json::Value::Null,
                    &text,
                )?;
            }
            FrameKind::Part(OutputPart::ToolCall { name, arguments }) => {
                out.end_run()?;
                // The model issues no call ids, so Rig mints one per call.
                let index = out.fresh_index();
                let arguments = match arguments {
                    serde_json::Value::String(text) => text,
                    other => other.to_string(),
                };
                out.fragment(
                    Some(index),
                    CallFragment {
                        id: None,
                        name: Some(&name),
                        arguments: Some(&arguments),
                    },
                )?;
                out.finish(index)?;
            }
            FrameKind::End(done) => {
                out.end_run()?;
                return Ok(out.end(Finish {
                    usage: usage(&done),
                    ..Finish::default()
                }));
            }
        }
        Ok(Flow::More)
    }
}

fn usage(done: &Done) -> Usage {
    Usage::new()
        .input_tokens(done.prompt_tokens)
        .output_tokens(done.completion_tokens)
        .total_tokens(done.prompt_tokens + done.completion_tokens)
        .cached_input_tokens(done.cached_input_tokens)
}

/// Rebuilds a reply's provider document — a serialized [`RawResponse`] —
/// from the worker's end frame.
#[derive(Debug, Default)]
pub struct RawDocument {
    document: Option<serde_json::Value>,
}

impl Serves<Completion> for RawDocument {}

impl Reassemble<Frame> for RawDocument {
    fn absorb(&mut self, frame: &Frame) {
        if let FrameKind::End(done) = &frame.0 {
            self.document = serde_json::to_value(RawResponse {
                text: done.text.clone(),
            })
            .ok();
        }
    }

    fn finish(self) -> serde_json::Value {
        self.document.unwrap_or(serde_json::Value::Null)
    }
}

/// The transport of a [`Model`]: carries each request to the inference
/// worker thread and delivers its reply as [`Frame`]s.
#[derive(Clone)]
pub struct Runtime {
    request_tx: mpsc::Sender<InferenceCommand>,
    sampling: SamplingParams,
}

impl Runtime {
    pub(crate) fn new(
        request_tx: mpsc::Sender<InferenceCommand>,
        sampling: SamplingParams,
    ) -> Self {
        Self {
            request_tx,
            sampling,
        }
    }

    /// The worker's parameters for `request`, or why it cannot be rendered.
    fn params(&self, request: &CompletionRequest) -> Result<InferenceParams, String> {
        let prepared_request = prepare_request(request)?;
        let options = &request.options;
        Ok(InferenceParams {
            prepared_request,
            max_tokens: request.max_tokens.map_or(DEFAULT_MAX_TOKENS, |max| {
                u32::try_from(max).unwrap_or(u32::MAX)
            }),
            temperature: request
                .temperature
                .map_or(DEFAULT_TEMPERATURE, |temperature| temperature as f32),
            top_p: options
                .top_p
                .map_or(self.sampling.top_p, |top_p| top_p as f32),
            top_k: self.sampling.top_k,
            min_p: self.sampling.min_p,
            presence_penalty: self.sampling.presence_penalty,
            repetition_penalty: self.sampling.repetition_penalty,
            // llama.cpp seeds take 32 bits; a wider seed keeps its low half.
            seed: options.seed.map_or(DEFAULT_SEED, |seed| seed as u32),
        })
    }

    async fn submit(
        &self,
        params: InferenceParams,
        response_channel: ResponseChannel,
    ) -> Result<(), ProviderError> {
        self.request_tx
            .send(InferenceCommand::Request(InferenceRequest {
                params,
                response_channel,
            }))
            .await
            .map_err(|_| ProviderError::Provider("Inference thread shut down".into()))
    }

    async fn open(
        self,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Opened<Frame>, ProviderError> {
        let params = self.params(&request).map_err(ProviderError::request)?;
        match mode {
            Mode::Unary => {
                let (response_tx, response_rx) = oneshot::channel();
                self.submit(params, ResponseChannel::Completion(response_tx))
                    .await?;
                let result = response_rx
                    .await
                    .map_err(|_| ProviderError::Provider("Response channel closed".into()))?
                    .map_err(ProviderError::Provider)?;
                let frames: Vec<Result<Frame, ProviderError>> = result
                    .parts
                    .into_iter()
                    .map(Frame::part)
                    .chain([Frame::end(result.done)])
                    .map(Ok)
                    .collect();
                Ok(Opened::new(tokio_stream::iter(frames)))
            }
            // Dropping the stream closes the channel, which stops generation
            // at the next sampled token.
            Mode::Streaming => {
                let (stream_tx, stream_rx) = mpsc::unbounded_channel();
                self.submit(params, ResponseChannel::Streaming(stream_tx))
                    .await?;
                Ok(Opened::new(UnboundedReceiverStream::new(stream_rx)))
            }
        }
    }
}

impl Transport<CompletionWire> for Runtime {
    fn send(&self, request: CompletionRequest, exchange: Exchange) -> Opening<Frame> {
        Opening::new(self.clone().open(request, exchange.mode))
    }
}

#[cfg(test)]
mod tests {
    use rig_core::completion::{CompletionRequest, Effort};
    use rig_core::driver::{Exchange, Opened, Opening, Transport};
    use rig_core::message::AssistantContent;
    use rig_core::streaming::{Item, StreamEvent};
    use serde_json::json;
    use tokio_stream::StreamExt;

    use super::CompletionWire;
    use crate::types::{Done, Frame, OutputPart, RawResponse};

    /// A runtime that answers every request with the same frames.
    #[derive(Clone)]
    struct Scripted(Vec<Frame>);

    impl Transport<CompletionWire> for Scripted {
        fn send(&self, _request: CompletionRequest, _exchange: Exchange) -> Opening<Frame> {
            let frames: Vec<_> = self.0.iter().cloned().map(Ok).collect();
            Opening::ready(Opened::new(tokio_stream::iter(frames)))
        }
    }

    fn model(parts: Vec<OutputPart>) -> rig_core::Model<CompletionWire, Scripted> {
        let mut frames: Vec<Frame> = parts.into_iter().map(Frame::part).collect();
        frames.push(Frame::end(Done {
            text: "the raw output".to_string(),
            prompt_tokens: 12,
            completion_tokens: 5,
            cached_input_tokens: 4,
        }));
        rig_core::Model::new(CompletionWire::new("test"), Scripted(frames))
    }

    fn turn() -> Vec<OutputPart> {
        vec![
            OutputPart::Reasoning("weigh it".to_string()),
            OutputPart::Text("Hello ".to_string()),
            OutputPart::Text("world".to_string()),
            OutputPart::ToolCall {
                name: "get_time".to_string(),
                arguments: json!({"zone": "UTC"}),
            },
        ]
    }

    #[tokio::test]
    async fn a_turn_assembles_its_blocks_usage_and_raw_text() {
        let response = model(turn()).call("hi").await.unwrap();

        let [
            AssistantContent::Reasoning(reasoning),
            AssistantContent::Text(text),
            AssistantContent::ToolCall(call),
        ] = response.choice.as_slice()
        else {
            panic!("unexpected blocks: {:?}", response.choice);
        };
        assert_eq!(reasoning.text, "weigh it");
        assert_eq!(text.text, "Hello world");
        assert_eq!(call.function.name.as_str(), "get_time");
        assert_eq!(call.function.arguments["zone"], "UTC");

        assert_eq!(response.usage.input_tokens, Some(12));
        assert_eq!(response.usage.output_tokens, Some(5));
        assert_eq!(response.usage.total_tokens, Some(17));
        assert_eq!(response.usage.cached_input_tokens, Some(4));

        let raw: RawResponse = serde_json::from_value(response.raw).unwrap();
        assert_eq!(raw.text, "the raw output");
    }

    #[tokio::test]
    async fn every_tool_call_gets_its_own_id() {
        let call = OutputPart::ToolCall {
            name: "get_time".to_string(),
            arguments: json!({}),
        };
        let response = model(vec![call.clone(), call]).call("hi").await.unwrap();
        let ids: Vec<_> = response
            .choice
            .iter()
            .filter_map(|block| match block {
                AssistantContent::ToolCall(call) => Some(call.id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], ids[1]);
    }

    #[tokio::test]
    async fn arguments_given_as_json_text_are_parsed() {
        let call = OutputPart::ToolCall {
            name: "search".to_string(),
            arguments: json!("{\"q\": 1}"),
        };
        let response = model(vec![call]).call("hi").await.unwrap();
        let AssistantContent::ToolCall(call) = &response.choice[0] else {
            panic!("expected a tool call, got {:?}", response.choice);
        };
        assert_eq!(call.function.arguments["q"], 1);
    }

    #[tokio::test]
    async fn a_stream_delivers_text_deltas_and_folds_into_the_same_response() {
        let mut stream = model(turn()).stream("hi").unwrap();
        let mut text = String::new();
        while let Some(item) = stream.next().await {
            if let Item::Event(StreamEvent::Text { text: delta, .. }) = item.unwrap() {
                text.push_str(&delta);
            }
        }
        assert_eq!(text, "Hello world");

        let response = stream.finish().await.unwrap();
        assert_eq!(response.choice.len(), 3);
        assert_eq!(response.usage.output_tokens, Some(5));
    }

    #[tokio::test]
    async fn sampler_options_are_accepted() {
        let request = CompletionRequest::new("hi")
            .reasoning(Effort::Low)
            .top_p(0.5)
            .seed(7);
        assert!(model(turn()).call(request).await.is_ok());
    }

    #[tokio::test]
    async fn an_option_the_runtime_cannot_honour_is_refused() {
        let request = CompletionRequest::new("hi").stop(["END"]);
        let error = model(turn()).call(request).await.unwrap_err();
        assert!(
            matches!(error, rig_core::ProviderError::UnsupportedOption(_)),
            "got {error:?}"
        );
    }
}
