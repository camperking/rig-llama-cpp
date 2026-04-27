use std::collections::HashSet;
use std::num::NonZeroU32;

use rig::completion::CompletionError;
use rig::message::AssistantContent;
use rig::one_or_many::OneOrMany;
use rig::streaming::RawStreamingChoice;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::parsing::parse_completion_output;
use crate::types::{
    FitParams, InferenceCommand, InferenceParams, InferenceResult, KvCacheParams, PreparedRequest,
    PromptBuildResult, ResponseChannel, SamplerChain, StreamChunk, StreamDeltaState, StreamSender,
};

struct WorkerModel {
    model: llama_cpp_2::model::LlamaModel,
    #[cfg(feature = "mtmd")]
    mtmd_ctx: Option<llama_cpp_2::mtmd::MtmdContext>,
    n_ctx: u32,
    kv_cache: KvCacheParams,
}

/// Load a model with automatic parameter fitting to available device memory.
fn fit_and_load_model(
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    model_path: &str,
    mmproj_path: Option<&str>,
    n_ctx: u32,
    fit: &FitParams,
    kv_cache: &KvCacheParams,
    logs_enabled: bool,
) -> Result<WorkerModel, String> {
    use llama_cpp_2::list_llama_ggml_backend_devices;
    use llama_cpp_2::model::LlamaModel as LlamaCppModel;
    use llama_cpp_2::model::params::LlamaModelParams;
    use std::pin::pin;

    // Do NOT call with_n_gpu_layers — fit requires n_gpu_layers at default (-1)
    let mut model_params = LlamaModelParams::default();

    if backend.supports_gpu_offload() {
        let vulkan_devices: Vec<usize> = list_llama_ggml_backend_devices()
            .into_iter()
            .filter(|device| device.backend.eq_ignore_ascii_case("vulkan"))
            .map(|device| device.index)
            .collect();

        if !vulkan_devices.is_empty() {
            model_params = model_params
                .with_devices(&vulkan_devices)
                .map_err(|e| format!("Failed to configure Vulkan devices: {e}"))?;
            if logs_enabled {
                eprintln!("Using Vulkan backend devices: {vulkan_devices:?}");
            }
        }
    }

    let mut pinned_params = pin!(model_params);

    // Prepare raw context params for the fit call
    let mut cparams = unsafe { llama_cpp_sys_2::llama_context_default_params() };
    cparams.n_ctx = n_ctx;

    // Prepare margins
    let max_devices = unsafe { llama_cpp_sys_2::llama_max_devices() };
    let mut margins = fit
        .margins
        .clone()
        .unwrap_or_else(|| vec![1 << 30; max_devices]);
    margins.resize(max_devices, 1 << 30);

    let model_cstr =
        std::ffi::CString::new(model_path).map_err(|e| format!("Invalid model path: {e}"))?;

    let log_level = if logs_enabled {
        llama_cpp_sys_2::GGML_LOG_LEVEL_INFO
    } else {
        llama_cpp_sys_2::GGML_LOG_LEVEL_NONE
    };

    if logs_enabled {
        eprintln!("Fitting model parameters for {model_path}...");
    }

    let fit_result = pinned_params
        .as_mut()
        .fit_params(
            &model_cstr,
            &mut cparams,
            &mut margins,
            fit.n_ctx_min,
            log_level as u32,
        )
        .map_err(|e| format!("Parameter fitting failed: {e}"))?;

    let actual_n_ctx = fit_result.n_ctx;

    if logs_enabled {
        eprintln!(
            "Fit complete: n_gpu_layers={}, n_ctx={}",
            pinned_params.n_gpu_layers(),
            actual_n_ctx
        );
        eprintln!("Loading model from {model_path}...");
    }

    let model = LlamaCppModel::load_from_file(backend, model_path, &pinned_params)
        .map_err(|e| format!("Model load failed: {e}"))?;

    if logs_enabled {
        eprintln!("Model loaded.");
    }

    #[cfg(feature = "mtmd")]
    let mtmd_ctx = if let Some(mmproj) = mmproj_path {
        let mtmd_params = llama_cpp_2::mtmd::MtmdContextParams::default();
        let ctx = llama_cpp_2::mtmd::MtmdContext::init_from_file(mmproj, &model, &mtmd_params)
            .map_err(|e| format!("Multimodal projector init failed: {e}"))?;
        if logs_enabled {
            eprintln!("Multimodal projector loaded from {mmproj}.");
        }
        Some(ctx)
    } else {
        None
    };

    #[cfg(not(feature = "mtmd"))]
    let _ = mmproj_path;

    Ok(WorkerModel {
        model,
        #[cfg(feature = "mtmd")]
        mtmd_ctx,
        n_ctx: actual_n_ctx,
        kv_cache: *kv_cache,
    })
}

pub(crate) fn inference_worker(
    model_path: &str,
    mmproj_path: Option<&str>,
    n_ctx: u32,
    fit_params: &FitParams,
    kv_cache_params: &KvCacheParams,
    init_tx: std::sync::mpsc::Sender<Result<(), String>>,
    rx: &mut mpsc::UnboundedReceiver<InferenceCommand>,
) {
    let backend = match crate::shared_backend() {
        Ok(b) => b,
        Err(e) => {
            let _ = init_tx.send(Err(e));
            return;
        }
    };
    let logs_enabled = crate::llama_logs_enabled();

    let mut wm = match fit_and_load_model(
        backend,
        model_path,
        mmproj_path,
        n_ctx,
        fit_params,
        kv_cache_params,
        logs_enabled,
    ) {
        Ok(wm) => wm,
        Err(e) => {
            let _ = init_tx.send(Err(e));
            return;
        }
    };

    // Signal successful initialization
    let _ = init_tx.send(Ok(()));

    // Process inference requests
    while let Some(command) = rx.blocking_recv() {
        match command {
            InferenceCommand::Request(req) => {
                let crate::types::InferenceRequest {
                    params,
                    response_channel,
                } = req;

                #[cfg(feature = "mtmd")]
                let mtmd_ref = wm.mtmd_ctx.as_ref();
                #[cfg(not(feature = "mtmd"))]
                let mtmd_ref: Option<&()> = None;

                match response_channel {
                    ResponseChannel::Completion(tx) => {
                        let result = run_inference(
                            backend,
                            &wm.model,
                            wm.n_ctx,
                            &wm.kv_cache,
                            &params,
                            None,
                            mtmd_ref,
                        );
                        let _ = tx.send(result);
                    }
                    ResponseChannel::Streaming(stream_tx) => {
                        let result = run_inference(
                            backend,
                            &wm.model,
                            wm.n_ctx,
                            &wm.kv_cache,
                            &params,
                            Some(&stream_tx),
                            mtmd_ref,
                        );
                        match result {
                            Ok(result) => {
                                let _ = stream_tx.send(Ok(RawStreamingChoice::FinalResponse(
                                    StreamChunk {
                                        text: result.text,
                                        prompt_tokens: Some(result.prompt_tokens),
                                        completion_tokens: Some(result.completion_tokens),
                                    },
                                )));
                            }
                            Err(e) => {
                                let _ = stream_tx.send(Err(CompletionError::ProviderError(e)));
                            }
                        }
                    }
                }
            }
            InferenceCommand::Reload(reload) => {
                drop(wm);

                let result = fit_and_load_model(
                    backend,
                    &reload.model_path,
                    reload.mmproj_path.as_deref(),
                    reload.n_ctx,
                    &reload.fit_params,
                    &reload.kv_cache_params,
                    logs_enabled,
                );

                match result {
                    Ok(new_wm) => {
                        wm = new_wm;
                        let _ = reload.result_tx.send(Ok(()));
                    }
                    Err(e) => {
                        let _ = reload.result_tx.send(Err(e));
                        return;
                    }
                }
            }
            InferenceCommand::Shutdown => break,
        }
    }
}

#[cfg(feature = "mtmd")]
fn run_inference(
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    model: &llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    mtmd_ctx: Option<&llama_cpp_2::mtmd::MtmdContext>,
) -> Result<InferenceResult, String> {
    run_inference_inner(backend, model, n_ctx, kv_cache, req, stream_tx, mtmd_ctx)
}

#[cfg(not(feature = "mtmd"))]
fn run_inference(
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    model: &llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    _mtmd_ctx: Option<&()>,
) -> Result<InferenceResult, String> {
    run_inference_inner(backend, model, n_ctx, kv_cache, req, stream_tx)
}

#[cfg(not(feature = "mtmd"))]
fn run_inference_inner(
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    model: &llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
) -> Result<InferenceResult, String> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::AddBos;

    let prompt_build = build_prompt(model, &req.prepared_request)?;
    let prompt = prompt_build.prompt.as_str();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx).map(Some).unwrap_or(None))
        .with_type_k(kv_cache.type_k)
        .with_type_v(kv_cache.type_v);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("Context creation failed: {e}"))?;

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| format!("Tokenization failed: {e}"))?;
    let prompt_tokens = tokens.len() as u64;

    let prompt_len = tokens.len();
    let prompt_batch_limit = ctx.n_batch().max(1) as usize;
    let mut batch = LlamaBatch::new(prompt_batch_limit, 1);

    for (chunk_index, chunk) in tokens.chunks(prompt_batch_limit).enumerate() {
        batch.clear();

        for (offset, token) in chunk.iter().copied().enumerate() {
            let position = (chunk_index * prompt_batch_limit + offset) as i32;
            let is_last_prompt_token = chunk_index * prompt_batch_limit + offset + 1 == prompt_len;

            batch
                .add(token, position, &[0], is_last_prompt_token)
                .map_err(|e| format!("Batch add failed: {e}"))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| format!("Prompt decode failed: {e}"))?;
    }

    sample_tokens(
        model,
        &mut ctx,
        &mut batch,
        &prompt_build,
        req,
        stream_tx,
        prompt_tokens,
    )
}

#[cfg(feature = "mtmd")]
fn run_inference_inner(
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    model: &llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    mtmd_ctx: Option<&llama_cpp_2::mtmd::MtmdContext>,
) -> Result<InferenceResult, String> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::AddBos;

    let prompt_build = build_prompt(model, &req.prepared_request)?;
    let prompt = prompt_build.prompt.as_str();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx).map(Some).unwrap_or(None))
        .with_type_k(kv_cache.type_k)
        .with_type_v(kv_cache.type_v);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("Context creation failed: {e}"))?;

    let has_images = !req.prepared_request.images.is_empty();

    if has_images && mtmd_ctx.is_some() {
        let mtmd = mtmd_ctx.unwrap();

        // Create bitmaps from raw image bytes
        let bitmaps: Vec<llama_cpp_2::mtmd::MtmdBitmap> = req
            .prepared_request
            .images
            .iter()
            .map(|bytes| {
                llama_cpp_2::mtmd::MtmdBitmap::from_buffer(mtmd, bytes)
                    .map_err(|e| format!("Failed to create bitmap from image data: {e}"))
            })
            .collect::<Result<_, _>>()?;

        let bitmap_refs: Vec<&llama_cpp_2::mtmd::MtmdBitmap> = bitmaps.iter().collect();

        let text_input = llama_cpp_2::mtmd::MtmdInputText {
            text: prompt.to_string(),
            add_special: true,
            parse_special: true,
        };

        let chunks = mtmd
            .tokenize(text_input, &bitmap_refs)
            .map_err(|e| format!("Multimodal tokenization failed: {e}"))?;

        let prompt_tokens = chunks.total_tokens() as u64;
        let n_batch = ctx.n_batch() as i32;

        let n_past = chunks
            .eval_chunks(mtmd, &ctx, 0, 0, n_batch, true)
            .map_err(|e| format!("Multimodal eval_chunks failed: {e}"))?;

        // Set up batch for sampling — we need a batch positioned at n_past
        let prompt_batch_limit = ctx.n_batch().max(1) as usize;
        let mut batch = LlamaBatch::new(prompt_batch_limit, 1);

        // After eval_chunks, the context is ready for sampling.
        // We need to set n_cur to n_past for the sampling loop.
        // The batch from eval_chunks already set logits_last=true, so we can sample directly.
        sample_tokens_from_pos(
            model,
            &mut ctx,
            &mut batch,
            &prompt_build,
            req,
            stream_tx,
            prompt_tokens,
            n_past as i32,
        )
    } else {
        // Standard text-only path
        let tokens = model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| format!("Tokenization failed: {e}"))?;
        let prompt_tokens = tokens.len() as u64;

        let prompt_len = tokens.len();
        let prompt_batch_limit = ctx.n_batch().max(1) as usize;
        let mut batch = LlamaBatch::new(prompt_batch_limit, 1);

        for (chunk_index, chunk) in tokens.chunks(prompt_batch_limit).enumerate() {
            batch.clear();

            for (offset, token) in chunk.iter().copied().enumerate() {
                let position = (chunk_index * prompt_batch_limit + offset) as i32;
                let is_last_prompt_token =
                    chunk_index * prompt_batch_limit + offset + 1 == prompt_len;

                batch
                    .add(token, position, &[0], is_last_prompt_token)
                    .map_err(|e| format!("Batch add failed: {e}"))?;
            }

            ctx.decode(&mut batch)
                .map_err(|e| format!("Prompt decode failed: {e}"))?;
        }

        sample_tokens(
            model,
            &mut ctx,
            &mut batch,
            &prompt_build,
            req,
            stream_tx,
            prompt_tokens,
        )
    }
}

/// Build a set of token IDs that should be decoded with `special = true`.
///
/// Models like Gemma-4 use control tokens (`<|channel>`, `<channel|>`, etc.) for
/// structured output (thinking, tool calls). These tokens are invisible when decoded
/// with `special = false`, making the PEG parser unable to extract reasoning content.
/// The template result's `preserved_tokens` lists the token strings that must remain
/// visible — matching the approach used by llama.cpp's server.
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
fn sample_tokens_from_pos(
    model: &llama_cpp_2::model::LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    batch: &mut llama_cpp_2::llama_batch::LlamaBatch,
    prompt_build: &PromptBuildResult,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    prompt_tokens: u64,
    n_past: i32,
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
    })
}

fn sample_tokens(
    model: &llama_cpp_2::model::LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    batch: &mut llama_cpp_2::llama_batch::LlamaBatch,
    prompt_build: &PromptBuildResult,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    prompt_tokens: u64,
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
    })
}

fn build_prompt(
    model: &llama_cpp_2::model::LlamaModel,
    request: &PreparedRequest,
) -> Result<PromptBuildResult, String> {
    use llama_cpp_2::model::LlamaChatMessage;
    use llama_cpp_2::openai::OpenAIChatTemplateParams;

    let chat_template_kwargs = json!({ "enable_thinking": request.enable_thinking }).to_string();

    if let Ok(tmpl) = model.chat_template(None) {
        let params = OpenAIChatTemplateParams {
            messages_json: &request.messages_json,
            tools_json: request.tools_json.as_deref(),
            tool_choice: request.tool_choice.as_deref(),
            json_schema: request.json_schema.as_deref(),
            grammar: None,
            reasoning_format: if request.enable_thinking {
                Some("auto")
            } else {
                Some("none")
            },
            chat_template_kwargs: Some(&chat_template_kwargs),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: request.enable_thinking,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: request.tools_json.is_some(),
        };

        match model.apply_chat_template_oaicompat(&tmpl, &params) {
            Ok(result) => {
                if crate::llama_logs_enabled() {
                    eprintln!("[rig-llama-cpp] messages_json: {}", request.messages_json);
                    eprintln!("[rig-llama-cpp] enable_thinking: {}", request.enable_thinking);
                    eprintln!("[rig-llama-cpp] chat_format: {}", result.chat_format);
                    eprintln!("[rig-llama-cpp] has_parser: {}", result.parser.is_some());
                    eprintln!(
                        "[rig-llama-cpp] prompt contains <|think|>: {}",
                        result.prompt.contains("<|think|>")
                    );
                    eprintln!("[rig-llama-cpp] rendered prompt:\n{}", result.prompt);
                }
                return Ok(PromptBuildResult {
                    prompt: result.prompt.clone(),
                    template_result: Some(result),
                });
            }
            Err(e) => {
                if crate::llama_logs_enabled() {
                    eprintln!(
                        "[rig-llama-cpp] apply_chat_template_oaicompat failed: {e}, falling back"
                    );
                }
                #[cfg(feature = "mtmd")]
                if !request.images.is_empty() {
                    return Err(format!(
                        "Chat template failed for multimodal request: {e}. \
                         The model's chat template may not support the current configuration."
                    ));
                }
            }
        }
    }

    let parsed_messages: Vec<(String, String)> =
        serde_json::from_str::<Vec<Value>>(&request.messages_json)
            .map_err(|e| format!("Message deserialization failed: {e}"))?
            .into_iter()
            .filter_map(|msg| {
                Some((
                    msg.get("role")?.as_str()?.to_string(),
                    message_content_as_text(&msg).to_string(),
                ))
            })
            .collect();

    let chat_msgs: Vec<LlamaChatMessage> = parsed_messages
        .iter()
        .map(|(role, content)| LlamaChatMessage::new(role.clone(), content.clone()))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("Chat message creation failed: {e}"))?;

    // Try model's built-in chat template first
    if let Ok(tmpl) = model.chat_template(None)
        && let Ok(prompt) = model.apply_chat_template(&tmpl, &chat_msgs, true)
    {
        return Ok(PromptBuildResult {
            prompt,
            template_result: None,
        });
    }

    // Fallback to ChatML format
    let mut prompt = String::new();
    for (role, content) in &parsed_messages {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(PromptBuildResult {
        prompt,
        template_result: None,
    })
}

fn message_content_as_text(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
