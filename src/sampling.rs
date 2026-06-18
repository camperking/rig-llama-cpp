use std::sync::atomic::{AtomicBool, Ordering};

use rig_core::message::{AssistantContent, ToolCall, ToolFunction};
use rig_core::one_or_many::OneOrMany;
use rig_core::streaming::{RawStreamingChoice, RawStreamingToolCall};

use crate::parsing::{extract_structured_json, parse_completion_output, parse_tool_calls};
use crate::slot::SlotEntry;
use crate::types::{InferenceParams, InferenceResult, PromptBuildResult, StreamSender};
use crate::worker::CANCEL_ERR;

/// Convert a `token_to_piece` outcome into a piece-or-empty result.
///
/// `llama.cpp`'s `llama_token_to_piece` returns size 0 when the token has
/// no printable representation — typically a control / unused / unknown-
/// attribute token (e.g. `<|im_start|>`, `<|fim_pad|>`). The `llama-cpp-2`
/// wrapper surfaces that as `TokenToStringError::UnknownTokenType`.
///
/// Canonical `llama.cpp` treats empty pieces as "no text to emit, keep
/// generating" — the sampled token is still consistent with the KV cache
/// because the caller appends it to the batch on the next iteration.
/// Real errors (`InsufficientBufferSpace`, `FromUtf8Error`, …) still
/// propagate so genuine bugs aren't silently swallowed.
pub(crate) fn token_piece_or_empty(
    result: Result<String, llama_cpp_2::TokenToStringError>,
) -> Result<String, String> {
    match result {
        Ok(piece) => Ok(piece),
        Err(llama_cpp_2::TokenToStringError::UnknownTokenType) => Ok(String::new()),
        Err(other) => Err(format!("Token to piece failed: {other}")),
    }
}

/// Sample one token.
///
/// `LlamaSampler::sample()` applies the chain and accepts the result
/// internally. When the chain contains the llguidance JSON-schema sampler we
/// must NOT accept again (double-consuming would desync the matcher); when
/// only the base samplers are present we preserve the legacy double-accept
/// that the penalty samplers were calibrated against.
fn sample_one(
    ctx: &llama_cpp_2::context::LlamaContext,
    sampler: &mut llama_cpp_2::sampling::LlamaSampler,
    idx: i32,
    has_schema_sampler: bool,
) -> llama_cpp_2::token::LlamaToken {
    let token = sampler.sample(ctx, idx);
    if !has_schema_sampler {
        sampler.accept(token);
    }
    token
}

/// Build a sampler chain, prepending an llguidance JSON-schema constraint when
/// the request carries an `output_schema`.
///
/// `llama-cpp-2` 0.1.147 removed the chat-template-derived grammar that used
/// to constrain tool-call output (`ChatTemplateResult.grammar`). Tool calls
/// are now parsed from the model's free-form output (see `src/parsing.rs`),
/// and structured-output (`json_schema`) constraints are applied via the
/// llguidance sampler, which accepts a JSON schema directly.
fn build_sampler_chain(
    model: &llama_cpp_2::model::LlamaModel,
    req: &InferenceParams,
) -> (llama_cpp_2::sampling::LlamaSampler, bool) {
    use llama_cpp_2::sampling::LlamaSampler;

    let base_samplers = vec![
        LlamaSampler::top_k(req.top_k),
        LlamaSampler::top_p(req.top_p, 1),
        LlamaSampler::min_p(req.min_p, 1),
        LlamaSampler::temp(req.temperature),
        LlamaSampler::penalties(-1, req.repetition_penalty, 0.0, req.presence_penalty),
        LlamaSampler::dist(42),
    ];

    // JSON-schema constrained generation via llguidance. The schema string is
    // the serialized JSON Schema; llguidance's "json" tag parses it directly.
    let schema_sampler = req.prepared_request.json_schema.as_deref().and_then(|schema| {
        match LlamaSampler::llguidance(model, "json", schema) {
            Ok(s) => {
                log::debug!("llguidance json-schema sampler created");
                Some(s)
            }
            Err(e) => {
                log::warn!(
                    "llguidance sampler creation failed, falling back to unconstrained sampling: {e}"
                );
                None
            }
        }
    });

    let has_schema = schema_sampler.is_some();
    let mut samplers = Vec::with_capacity(base_samplers.len() + 1);
    if let Some(s) = schema_sampler {
        samplers.push(s);
    }
    samplers.extend(base_samplers);
    (LlamaSampler::chain_simple(samplers), has_schema)
}

#[cfg(feature = "mtmd")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_tokens_from_pos(
    model: &llama_cpp_2::model::LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    batch: &mut llama_cpp_2::llama_batch::LlamaBatch,
    _prompt_build: &PromptBuildResult,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    prompt_tokens: u64,
    cached_input_tokens: u64,
    n_past: i32,
    last_entries: &mut Vec<SlotEntry>,
    cancel: &AtomicBool,
) -> Result<InferenceResult, String> {
    let (output, choice, completion_tokens) = sample_loop(
        model,
        ctx,
        batch,
        req,
        stream_tx,
        n_past,
        last_entries,
        cancel,
    )?;
    Ok(InferenceResult {
        text: output,
        choice,
        prompt_tokens,
        completion_tokens,
        cached_input_tokens,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_tokens(
    model: &llama_cpp_2::model::LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    batch: &mut llama_cpp_2::llama_batch::LlamaBatch,
    _prompt_build: &PromptBuildResult,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    prompt_tokens: u64,
    cached_input_tokens: u64,
    last_entries: &mut Vec<SlotEntry>,
    cancel: &AtomicBool,
) -> Result<InferenceResult, String> {
    // Text path: generation resumes right after the prompt tokens.
    let n_past = prompt_tokens as i32;
    let (output, choice, completion_tokens) = sample_loop(
        model,
        ctx,
        batch,
        req,
        stream_tx,
        n_past,
        last_entries,
        cancel,
    )?;
    Ok(InferenceResult {
        text: output,
        choice,
        prompt_tokens,
        completion_tokens,
        cached_input_tokens,
    })
}

/// Core sampling loop shared by the text and mtmd paths. The mtmd path enters
/// at `n_past` (the position after image chunks); the text path enters at
/// `prompt_tokens`.
///
/// Returns `(output_text, AssistantContent choice, completion_token_count)`.
#[allow(clippy::too_many_arguments)]
fn sample_loop(
    model: &llama_cpp_2::model::LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    batch: &mut llama_cpp_2::llama_batch::LlamaBatch,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    n_past: i32,
    last_entries: &mut Vec<SlotEntry>,
    cancel: &AtomicBool,
) -> Result<(String, OneOrMany<AssistantContent>, u64), String> {
    let (mut sampler, has_schema) = build_sampler_chain(model, req);

    let has_tools = req.prepared_request.tools_json.is_some();
    let has_schema_request = req.prepared_request.json_schema.is_some();
    // Tool-calling turns buffer the whole generation: the model may emit
    // `<tool_call>` XML that we must parse whole before emitting anything, so
    // consumers don't see the raw tool-call text as message deltas.
    let buffer_output = has_tools;

    let mut output = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut completion_tokens = 0u64;

    for n_cur in (n_past..).take(req.max_tokens as usize) {
        if cancel.load(Ordering::Relaxed) {
            return Err(CANCEL_ERR.to_string());
        }
        if let Some(tx) = stream_tx
            && tx.is_closed()
        {
            break;
        }

        // First token samples from -1 (last logits); subsequent ones from the
        // last batch position.
        let sample_idx = if completion_tokens == 0 {
            -1
        } else {
            batch.n_tokens() - 1
        };
        let token = sample_one(ctx, &mut sampler, sample_idx, has_schema);

        if model.is_eog_token(token) {
            break;
        }

        let piece = token_piece_or_empty(model.token_to_piece(token, &mut decoder, false, None))?;
        output.push_str(&piece);
        completion_tokens += 1;

        // Stream incremental text only when not buffering a tool-call turn.
        // For structured output the pieces are JSON tokens and stream fine.
        if let Some(tx) = stream_tx
            && !buffer_output
        {
            let _ = tx.send(Ok(RawStreamingChoice::Message(piece)));
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| format!("Batch add failed: {e}"))?;
        ctx.decode(batch)
            .map_err(|e| format!("Decode failed: {e}"))?;
        last_entries.push(SlotEntry::Text(token));
    }

    log::debug!("raw output:\n{output}");

    if let Some(tx) = stream_tx {
        flush_stream(tx, &output, has_tools, has_schema_request);
    }

    let choice = build_choice(&output, stream_tx.is_some(), has_tools, has_schema_request)?;

    Ok((output, choice, completion_tokens))
}

/// Emit the tail of a stream after generation completes.
///
/// - **json_schema**: emit a single corrective chunk containing the cleaned
///   JSON (strips role tokens / markdown fences the template may leak).
/// - **tools**: parse the buffered output for tool calls and emit complete
///   `ToolCall` events. Any leading prose before the first `<tool_call>` is
///   also surfaced as a message chunk.
/// - **plain text**: nothing to flush (text was streamed incrementally).
fn flush_stream(tx: &StreamSender, output: &str, has_tools: bool, has_schema: bool) {
    if has_schema {
        if let Some(json) = extract_structured_json(output) {
            let _ = tx.send(Ok(RawStreamingChoice::Message(json)));
        }
        return;
    }

    if has_tools {
        if let Some(tool_calls) = parse_tool_calls(output) {
            // Surface any prose preceding the first tool call as text.
            if let Some(prefix_end) = output.find("<tool_call>")
                && !output[..prefix_end].trim().is_empty()
            {
                let _ = tx.send(Ok(RawStreamingChoice::Message(
                    output[..prefix_end].trim().to_string(),
                )));
            }
            for (i, (name, arguments)) in tool_calls.into_iter().enumerate() {
                let id = format!("tool-call-{i}");
                let _ = tx.send(Ok(RawStreamingChoice::ToolCall(RawStreamingToolCall::new(
                    id.clone(),
                    name,
                    arguments,
                ))));
            }
        } else {
            // No tool calls parsed; emit the buffered text as one message.
            let _ = tx.send(Ok(RawStreamingChoice::Message(output.to_string())));
        }
    }
}

/// Build the final `OneOrMany<AssistantContent>` choice from the raw output.
fn build_choice(
    output: &str,
    is_stream: bool,
    has_tools: bool,
    has_schema: bool,
) -> Result<OneOrMany<AssistantContent>, String> {
    if is_stream {
        if has_schema && let Some(json) = extract_structured_json(output) {
            return Ok(OneOrMany::one(AssistantContent::text(json)));
        }
        if has_tools && let Some(tool_calls) = parse_tool_calls(output) {
            let mut content: Vec<AssistantContent> = Vec::new();
            for (i, (name, arguments)) in tool_calls.into_iter().enumerate() {
                content.push(AssistantContent::ToolCall(ToolCall::new(
                    format!("tool-call-{i}"),
                    ToolFunction::new(name, arguments),
                )));
            }
            if let Ok(result) = OneOrMany::many(content) {
                return Ok(result);
            }
        }
        return Ok(OneOrMany::one(AssistantContent::text(output.to_string())));
    }

    parse_completion_output(output, has_schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_piece_or_empty_passes_ok_through() {
        let result = token_piece_or_empty(Ok("hello".to_string()));
        assert_eq!(result.as_deref(), Ok("hello"));
    }

    #[test]
    fn token_piece_or_empty_swallows_unknown_token_type() {
        let result = token_piece_or_empty(Err(llama_cpp_2::TokenToStringError::UnknownTokenType));
        assert_eq!(result.as_deref(), Ok(""));
    }

    #[test]
    fn token_piece_or_empty_propagates_real_errors() {
        let result = token_piece_or_empty(Err(
            llama_cpp_2::TokenToStringError::InsufficientBufferSpace(-32),
        ));
        let err = result.expect_err("expected error to propagate");
        assert!(err.starts_with("Token to piece failed:"), "got: {err}");
    }
}
