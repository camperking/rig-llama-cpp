use std::collections::HashSet;

use rig::message::AssistantContent;
use rig::one_or_many::OneOrMany;
use rig::streaming::RawStreamingChoice;

use crate::parsing::parse_completion_output;
use crate::slot::SlotEntry;
use crate::types::{
    InferenceParams, InferenceResult, PromptBuildResult, SamplerChain, StreamDeltaState,
    StreamSender,
};

fn build_preserved_token_set(
    model: &llama_cpp_2::model::LlamaModel,
    template_result: Option<&llama_cpp_2::model::ChatTemplateResult>,
) -> HashSet<llama_cpp_2::token::LlamaToken> {
    use llama_cpp_2::model::AddBos;

    let mut set = HashSet::new();
    let Some(tr) = template_result else {
        return set;
    };
    for token_str in &tr.preserved_tokens {
        if let Ok(ids) = model.str_to_token(token_str, AddBos::Never) {
            if ids.len() == 1 {
                set.insert(ids[0]);
            } else if crate::llama_logs_enabled() {
                eprintln!(
                    "[rig-llama-cpp] preserved token {token_str:?} tokenized to {} ids (expected 1), skipping",
                    ids.len()
                );
            }
        } else if crate::llama_logs_enabled() {
            eprintln!("[rig-llama-cpp] preserved token {token_str:?} not found in vocabulary");
        }
    }
    if crate::llama_logs_enabled() && !set.is_empty() {
        eprintln!(
            "[rig-llama-cpp] preserved tokens: {:?}",
            tr.preserved_tokens
        );
    }
    set
}

/// Collect additional stop sequences from the template result.
fn get_additional_stops(
    template_result: Option<&llama_cpp_2::model::ChatTemplateResult>,
) -> Vec<String> {
    template_result
        .map(|tr| tr.additional_stops.clone())
        .unwrap_or_default()
}

/// Escape regex metacharacters in a string (for Word-type grammar triggers).
fn regex_escape(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        if r"\.^$*+?()[]{}|".contains(c) {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}

/// Build a sampler chain with optional grammar constraints from the chat template.
///
/// When `ChatTemplateResult` provides a grammar (e.g. for tool-call output), the grammar
/// sampler is prepended to the chain so invalid tokens are zeroed before other samplers rank
/// the remaining candidates.
fn build_sampler_chain(
    model: &llama_cpp_2::model::LlamaModel,
    template_result: Option<&llama_cpp_2::model::ChatTemplateResult>,
    req: &InferenceParams,
) -> SamplerChain {
    use llama_cpp_2::model::GrammarTriggerType;
    use llama_cpp_2::sampling::LlamaSampler;

    let base_samplers = vec![
        LlamaSampler::top_k(req.top_k),
        LlamaSampler::top_p(req.top_p, 1),
        LlamaSampler::min_p(req.min_p, 1),
        LlamaSampler::temp(req.temperature),
        LlamaSampler::penalties(-1, req.repetition_penalty, 0.0, req.presence_penalty),
        LlamaSampler::dist(42),
    ];

    // Attempt to create a grammar sampler from the template result.
    let grammar_sampler = template_result
        .and_then(|tr| tr.grammar.as_ref().map(|g| (g, tr)))
        .and_then(|(grammar_str, tr)| {
            let result = if tr.grammar_lazy {
                // Convert triggers into patterns and tokens for lazy grammar.
                let mut trigger_patterns = Vec::new();
                let mut trigger_tokens = Vec::new();

                for trigger in &tr.grammar_triggers {
                    match trigger.trigger_type {
                        GrammarTriggerType::Token => {
                            if let Some(tok) = trigger.token {
                                trigger_tokens.push(tok);
                            }
                        }
                        GrammarTriggerType::Word => {
                            trigger_patterns.push(regex_escape(&trigger.value));
                        }
                        GrammarTriggerType::Pattern => {
                            trigger_patterns.push(trigger.value.clone());
                        }
                        GrammarTriggerType::PatternFull => {
                            let mut pat = trigger.value.clone();
                            if !pat.starts_with('^') {
                                pat.insert(0, '^');
                            }
                            if !pat.ends_with('$') {
                                pat.push('$');
                            }
                            trigger_patterns.push(pat);
                        }
                    }
                }

                if trigger_patterns.is_empty() && trigger_tokens.is_empty() {
                    // No triggers means lazy grammar would never activate; fall back to eager.
                    if crate::llama_logs_enabled() {
                        eprintln!(
                            "[rig-llama-cpp] grammar_lazy is true but no triggers found, \
                             falling back to eager grammar"
                        );
                    }
                    LlamaSampler::grammar(model, grammar_str, "root")
                } else {
                    LlamaSampler::grammar_lazy_patterns(
                        model,
                        grammar_str,
                        "root",
                        &trigger_patterns,
                        &trigger_tokens,
                    )
                }
            } else {
                LlamaSampler::grammar(model, grammar_str, "root")
            };

            match result {
                Ok(sampler) => {
                    if crate::llama_logs_enabled() {
                        eprintln!(
                            "[rig-llama-cpp] grammar sampler created (lazy={})",
                            tr.grammar_lazy
                        );
                    }
                    Some(sampler)
                }
                Err(e) => {
                    if crate::llama_logs_enabled() {
                        eprintln!(
                            "[rig-llama-cpp] grammar sampler creation failed, \
                             falling back to unconstrained sampling: {e}"
                        );
                    }
                    None
                }
            }
        });

    let has_grammar = grammar_sampler.is_some();
    let mut samplers = Vec::with_capacity(7);
    if let Some(gs) = grammar_sampler {
        samplers.push(gs);
    }
    samplers.extend(base_samplers);
    SamplerChain {
        sampler: llama_cpp_2::sampling::LlamaSampler::chain_simple(samplers),
        has_grammar,
    }
}

#[cfg(feature = "mtmd")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_tokens_from_pos(
    model: &llama_cpp_2::model::LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    batch: &mut llama_cpp_2::llama_batch::LlamaBatch,
    prompt_build: &PromptBuildResult,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    prompt_tokens: u64,
    cached_input_tokens: u64,
    n_past: i32,
    last_entries: &mut Vec<SlotEntry>,
) -> Result<InferenceResult, String> {
    let SamplerChain {
        mut sampler,
        has_grammar,
    } = build_sampler_chain(model, prompt_build.template_result.as_ref(), req);

    let preserved_tokens = build_preserved_token_set(model, prompt_build.template_result.as_ref());
    let additional_stops = get_additional_stops(prompt_build.template_result.as_ref());

    let mut output = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = n_past;
    let mut completion_tokens = 0u64;

    let mut stream_parser = if stream_tx.is_some() {
        prompt_build
            .template_result
            .as_ref()
            .and_then(|tr| tr.streaming_state_oaicompat().ok())
    } else {
        None
    };
    let mut delta_state = StreamDeltaState::new();

    for _ in 0..req.max_tokens {
        if let Some(tx) = stream_tx
            && tx.is_closed()
        {
            break;
        }

        // For the first token after eval_chunks, sample from index -1 (last logits)
        let sample_idx = if completion_tokens == 0 {
            -1
        } else {
            batch.n_tokens() - 1
        };
        let token = sampler.sample(ctx, sample_idx);
        // sample() internally calls accept(). When there is no grammar sampler,
        // accept again to preserve legacy double-accept that base samplers were
        // calibrated with. Skip when grammar is present to avoid corrupting its
        // parser state.
        if !has_grammar {
            sampler.accept(token);
        }

        if model.is_eog_token(token) {
            break;
        }

        let decode_special = preserved_tokens.contains(&token);
        let piece = model
            .token_to_piece(token, &mut decoder, decode_special, None)
            .map_err(|e| format!("Token to piece failed: {e}"))?;
        output.push_str(&piece);
        completion_tokens += 1;

        // Check for additional stop sequences
        if let Some(stop) = additional_stops.iter().find(|s| output.ends_with(s.as_str())) {
            let stop_len = stop.len();
            output.truncate(output.len() - stop_len);
            break;
        }

        if let Some(tx) = stream_tx {
            if let Some(parser) = stream_parser.as_mut() {
                match parser.update(&piece, true) {
                    Ok(deltas) => {
                        for delta_json in deltas {
                            for choice in delta_state.parse_delta(&delta_json) {
                                let _ = tx.send(Ok(choice));
                            }
                        }
                    }
                    Err(_) => {
                        let _ = tx.send(Ok(RawStreamingChoice::Message(piece.clone())));
                    }
                }
            } else {
                let _ = tx.send(Ok(RawStreamingChoice::Message(piece.clone())));
            }
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| format!("Batch add failed: {e}"))?;
        ctx.decode(batch)
            .map_err(|e| format!("Decode failed: {e}"))?;
        last_entries.push(SlotEntry::Text(token));
        n_cur += 1;
    }

    if crate::llama_logs_enabled() {
        eprintln!("[rig-llama-cpp] raw output:\n{output}");
    }

    // Flush remaining deltas from the streaming parser
    if let Some(tx) = stream_tx {
        if let Some(parser) = stream_parser.as_mut()
            && let Ok(deltas) = parser.update("", false)
        {
            for delta_json in deltas {
                for choice in delta_state.parse_delta(&delta_json) {
                    let _ = tx.send(Ok(choice));
                }
            }
        }
        for choice in delta_state.flush_tool_calls(&output, prompt_build.template_result.as_ref()) {
            let _ = tx.send(Ok(choice));
        }
    }

    let choice = if stream_tx.is_some() {
        OneOrMany::one(AssistantContent::text(output.clone()))
    } else {
        parse_completion_output(
            &output,
            prompt_build.template_result.as_ref(),
            req.prepared_request.json_schema.is_some(),
        )?
    };

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
    prompt_build: &PromptBuildResult,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    prompt_tokens: u64,
    cached_input_tokens: u64,
    last_entries: &mut Vec<SlotEntry>,
) -> Result<InferenceResult, String> {
    let SamplerChain {
        mut sampler,
        has_grammar,
    } = build_sampler_chain(model, prompt_build.template_result.as_ref(), req);

    let preserved_tokens = build_preserved_token_set(model, prompt_build.template_result.as_ref());
    let additional_stops = get_additional_stops(prompt_build.template_result.as_ref());

    let mut output = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = prompt_tokens as i32;
    let mut completion_tokens = 0u64;

    // Initialize streaming parser if streaming and we have a template result
    let mut stream_parser = if stream_tx.is_some() {
        prompt_build
            .template_result
            .as_ref()
            .and_then(|tr| tr.streaming_state_oaicompat().ok())
    } else {
        None
    };
    let mut delta_state = StreamDeltaState::new();

    for _ in 0..req.max_tokens {
        // Stop early if the consumer disconnected (e.g. user cancelled).
        if let Some(tx) = stream_tx
            && tx.is_closed()
        {
            break;
        }

        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        if !has_grammar {
            sampler.accept(token);
        }

        if model.is_eog_token(token) {
            break;
        }

        let decode_special = preserved_tokens.contains(&token);
        let piece = model
            .token_to_piece(token, &mut decoder, decode_special, None)
            .map_err(|e| format!("Token to piece failed: {e}"))?;
        output.push_str(&piece);
        completion_tokens += 1;

        // Check for additional stop sequences
        if let Some(stop) = additional_stops.iter().find(|s| output.ends_with(s.as_str())) {
            let stop_len = stop.len();
            output.truncate(output.len() - stop_len);
            break;
        }

        if let Some(tx) = stream_tx {
            if let Some(parser) = stream_parser.as_mut() {
                match parser.update(&piece, true) {
                    Ok(deltas) => {
                        for delta_json in deltas {
                            for choice in delta_state.parse_delta(&delta_json) {
                                let _ = tx.send(Ok(choice));
                            }
                        }
                    }
                    Err(_) => {
                        let _ = tx.send(Ok(RawStreamingChoice::Message(piece)));
                    }
                }
            } else {
                let _ = tx.send(Ok(RawStreamingChoice::Message(piece)));
            }
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| format!("Batch add failed: {e}"))?;
        ctx.decode(batch)
            .map_err(|e| format!("Decode failed: {e}"))?;
        // Track tokens that are now committed to the KV cache so the next
        // request can detect the longest common prefix correctly.
        last_entries.push(SlotEntry::Text(token));
        n_cur += 1;
    }

    if crate::llama_logs_enabled() {
        eprintln!("[rig-llama-cpp] raw output:\n{output}");
    }

    // Flush remaining deltas from the streaming parser
    if let Some(tx) = stream_tx {
        if let Some(parser) = stream_parser.as_mut()
            && let Ok(deltas) = parser.update("", false)
        {
            for delta_json in deltas {
                for choice in delta_state.parse_delta(&delta_json) {
                    let _ = tx.send(Ok(choice));
                }
            }
        }
        // Emit complete tool calls so they get accumulated into assistant_items
        for choice in delta_state.flush_tool_calls(&output, prompt_build.template_result.as_ref()) {
            let _ = tx.send(Ok(choice));
        }
    }

    let choice = if stream_tx.is_some() {
        // For streaming, choice was already sent through the stream;
        // return a minimal placeholder for InferenceResult
        OneOrMany::one(AssistantContent::text(output.clone()))
    } else {
        parse_completion_output(
            &output,
            prompt_build.template_result.as_ref(),
            req.prepared_request.json_schema.is_some(),
        )?
    };

    Ok(InferenceResult {
        text: output,
        choice,
        prompt_tokens,
        completion_tokens,
        cached_input_tokens,
    })
}
