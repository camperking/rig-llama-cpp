use rig::completion::CompletionError;
use rig::streaming::RawStreamingChoice;
use tokio::sync::mpsc;

use crate::checkpoint::{
    PersistentCtx, ensure_persistent_ctx, maybe_create_checkpoint, restore_or_clear,
};
#[cfg(feature = "mtmd")]
use crate::image::run_image_inference;
use crate::loader::{WorkerModel, fit_and_load_model};
use crate::prompt::build_prompt;
use crate::sampling::sample_tokens;
use crate::slot::{SlotEntry, get_common_prefix};
use crate::types::{
    CheckpointParams, FitParams, InferenceCommand, InferenceParams, InferenceResult, KvCacheParams,
    ResponseChannel, StreamChunk, StreamSender,
};
enum LoopOutcome {
    Reload(crate::types::ReloadRequest),
    Shutdown,
}

/// Inner request loop that owns the persistent context.
///
/// Returns when the channel closes, a reload is requested, or shutdown is requested.
/// On Reload the caller drops `wm` and reloads the model — the persistent context
/// (which borrows `&wm.model`) is dropped automatically when this function returns.
fn handle_until_reload<'m>(
    backend: &'m llama_cpp_2::llama_backend::LlamaBackend,
    wm: &'m WorkerModel,
    checkpoint_params: CheckpointParams,
    rx: &mut mpsc::UnboundedReceiver<InferenceCommand>,
) -> LoopOutcome {
    let mut persistent: Option<PersistentCtx<'m>> = None;

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
                            checkpoint_params,
                            &mut persistent,
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
                            checkpoint_params,
                            &mut persistent,
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
                                        cached_input_tokens: Some(result.cached_input_tokens),
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
            InferenceCommand::Reload(reload) => return LoopOutcome::Reload(reload),
            InferenceCommand::Shutdown => return LoopOutcome::Shutdown,
        }
    }
    LoopOutcome::Shutdown
}

pub(crate) fn inference_worker(
    model_path: &str,
    mmproj_path: Option<&str>,
    n_ctx: u32,
    fit_params: &FitParams,
    kv_cache_params: &KvCacheParams,
    checkpoint_params: CheckpointParams,
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

    let mut checkpoint_params = checkpoint_params;

    while let LoopOutcome::Reload(reload) = handle_until_reload(backend, &wm, checkpoint_params, rx)
    {
        // The persistent context (held inside handle_until_reload) has
        // already been dropped by the time we get here, so it is safe
        // to drop and replace `wm`.
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
                checkpoint_params = reload.checkpoint_params;
                let _ = reload.result_tx.send(Ok(()));
            }
            Err(e) => {
                let _ = reload.result_tx.send(Err(e));
                return;
            }
        }
    }
}

#[cfg(feature = "mtmd")]
fn run_inference<'m>(
    backend: &'m llama_cpp_2::llama_backend::LlamaBackend,
    model: &'m llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    checkpoint_params: CheckpointParams,
    persistent: &mut Option<PersistentCtx<'m>>,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    mtmd_ctx: Option<&llama_cpp_2::mtmd::MtmdContext>,
) -> Result<InferenceResult, String> {
    run_inference_inner(
        backend,
        model,
        n_ctx,
        kv_cache,
        checkpoint_params,
        persistent,
        req,
        stream_tx,
        mtmd_ctx,
    )
}

#[cfg(not(feature = "mtmd"))]
fn run_inference<'m>(
    backend: &'m llama_cpp_2::llama_backend::LlamaBackend,
    model: &'m llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    checkpoint_params: CheckpointParams,
    persistent: &mut Option<PersistentCtx<'m>>,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    _mtmd_ctx: Option<&()>,
) -> Result<InferenceResult, String> {
    run_inference_inner(
        backend,
        model,
        n_ctx,
        kv_cache,
        checkpoint_params,
        persistent,
        req,
        stream_tx,
    )
}

#[cfg(not(feature = "mtmd"))]
fn run_inference_inner<'m>(
    backend: &'m llama_cpp_2::llama_backend::LlamaBackend,
    model: &'m llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    checkpoint_params: CheckpointParams,
    persistent: &mut Option<PersistentCtx<'m>>,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
) -> Result<InferenceResult, String> {
    run_text_inference(
        backend,
        model,
        n_ctx,
        kv_cache,
        checkpoint_params,
        persistent,
        req,
        stream_tx,
    )
}

#[cfg(feature = "mtmd")]
fn run_inference_inner<'m>(
    backend: &'m llama_cpp_2::llama_backend::LlamaBackend,
    model: &'m llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    checkpoint_params: CheckpointParams,
    persistent: &mut Option<PersistentCtx<'m>>,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    mtmd_ctx: Option<&llama_cpp_2::mtmd::MtmdContext>,
) -> Result<InferenceResult, String> {
    let has_images = !req.prepared_request.images.is_empty();

    if has_images && mtmd_ctx.is_some() {
        run_image_inference(
            backend,
            model,
            n_ctx,
            kv_cache,
            persistent,
            req,
            stream_tx,
            mtmd_ctx,
        )
    } else {
        run_text_inference(
            backend,
            model,
            n_ctx,
            kv_cache,
            checkpoint_params,
            persistent,
            req,
            stream_tx,
        )
    }
}

/// Text-only inference with persistent-context + prefix-cache reuse.
///
/// On each call we tokenize the new prompt, find the longest common prefix with
/// the tokens currently committed in the KV cache, trim everything after that
/// prefix, and decode only the suffix. If prefix-cache reuse fails (which can
/// happen e.g. on memory implementations that don't support arbitrary partial
/// trims), we invalidate the persistent slot and retry once with a fresh
/// context — so the user's request still succeeds at the cost of a full decode.
fn run_text_inference<'m>(
    backend: &'m llama_cpp_2::llama_backend::LlamaBackend,
    model: &'m llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    checkpoint_params: CheckpointParams,
    persistent: &mut Option<PersistentCtx<'m>>,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
) -> Result<InferenceResult, String> {
    use llama_cpp_2::model::AddBos;

    let prompt_build = build_prompt(model, &req.prepared_request)?;
    let prompt = prompt_build.prompt.as_str();

    let new_tokens = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| format!("Tokenization failed: {e}"))?;
    let prompt_len = new_tokens.len();

    if prompt_len == 0 {
        return Err("Empty prompt after tokenization".to_string());
    }
    if prompt_len > n_ctx as usize {
        return Err(format!(
            "Prompt {prompt_len} tokens exceeds n_ctx {n_ctx}"
        ));
    }

    ensure_persistent_ctx(backend, model, n_ctx, kv_cache, persistent)?;

    // Build the candidate as all-Text entries for the diff. Image entries
    // from a previous mtmd turn (if any) compare unequal to text tokens,
    // which is exactly what we want — divergence at the first image position.
    let new_entries: Vec<SlotEntry> = new_tokens.iter().map(|t| SlotEntry::Text(*t)).collect();
    let cached = {
        let p = persistent.as_ref().unwrap();
        get_common_prefix(&p.last_entries, &new_entries)
    };

    // Phase 1: prompt decode (with prefix-cache reuse). This phase is safe to
    // retry on failure because no output has been streamed yet. The helper
    // gracefully handles trim-unsupported memories (recurrent/hybrid) by
    // restoring the closest checkpoint or fully clearing the cache.
    let (mut batch, effective_cached) = match prepare_prompt_decode(
        persistent.as_mut().unwrap(),
        &new_tokens,
        cached,
        prompt_len,
        checkpoint_params,
    ) {
        Ok(out) => out,
        Err(e) if cached > 0 => {
            // Some other phase-1 failure mode. Drop persistent, rebuild fresh,
            // and retry from scratch. Safe because no output has streamed yet.
            eprintln!(
                "[rig-llama-cpp] prefix-cache decode failed (cached={cached}, prompt_len={prompt_len}): {e}. \
                 Falling back to fresh-context decode."
            );
            *persistent = None;
            ensure_persistent_ctx(backend, model, n_ctx, kv_cache, persistent)?;
            match prepare_prompt_decode(
                persistent.as_mut().unwrap(),
                &new_tokens,
                0,
                prompt_len,
                checkpoint_params,
            ) {
                Ok(out) => out,
                Err(e) => {
                    *persistent = None;
                    return Err(e);
                }
            }
        }
        Err(e) => {
            *persistent = None;
            return Err(e);
        }
    };

    // Phase 2: commit the prompt to last_entries and sample. From this point on
    // we may have streamed tokens to the consumer, so any failure invalidates
    // the persistent slot but cannot be retried.
    let p = persistent.as_mut().unwrap();
    p.last_entries = new_entries;
    let prompt_tokens = prompt_len as u64;
    let cached_tokens = effective_cached as u64;

    let result = sample_tokens(
        model,
        &mut p.ctx,
        &mut batch,
        &prompt_build,
        req,
        stream_tx,
        prompt_tokens,
        cached_tokens,
        &mut p.last_entries,
    );

    if result.is_err() {
        *persistent = None;
    }
    result
}

/// Decode the prompt suffix into the persistent context's KV cache and return
/// a batch ready for sampling, plus the count of tokens that were actually
/// served from the cache (which may be less than the LCP if a rollback wasn't
/// possible). This is "phase 1" — safe to retry on failure because no output
/// has been streamed to the consumer yet.
///
/// For models whose memory rejects partial trims (recurrent/hybrid), we
/// attempt to restore from the closest in-memory state checkpoint before
/// falling back to a full clear.
fn prepare_prompt_decode<'b>(
    p: &mut PersistentCtx<'_>,
    new_tokens: &[llama_cpp_2::token::LlamaToken],
    cached: usize,
    prompt_len: usize,
    checkpoint_params: CheckpointParams,
) -> Result<(llama_cpp_2::llama_batch::LlamaBatch<'b>, usize), String> {
    use llama_cpp_2::llama_batch::LlamaBatch;

    if crate::llama_logs_enabled() {
        eprintln!(
            "[rig-llama-cpp] prefix-cache: prompt_len={prompt_len} last_entries.len={} cached={cached} trim_unsupported={} checkpoints={}",
            p.last_entries.len(),
            p.trim_unsupported,
            p.checkpoint_count(),
        );
    }

    let mut effective_cached = cached;

    if cached < p.last_entries.len() {
        // Need to roll back the cache to position `cached`.
        if p.trim_unsupported {
            // Already known: trim refused before. Try checkpoint restore.
            effective_cached = restore_or_clear(p, cached);
        } else {
            let removed = p
                .ctx
                .clear_kv_cache_seq(Some(0), Some(cached as u32), None)
                .map_err(|e| format!("KV cache trim failed: {e:?}"))?;
            if removed {
                // Trim worked. Drop checkpoints whose pos_max >= cached because
                // the state they captured is now invalid (positions ahead of
                // the trim boundary).
                p.retain_checkpoints_below(cached);
            } else {
                // First time this model rejects a partial trim. Mark it and
                // try the checkpoint path.
                eprintln!(
                    "[rig-llama-cpp] partial KV-cache trim not supported by this model \
                     (likely recurrent/hybrid). Routing rollbacks through checkpoint restore."
                );
                p.trim_unsupported = true;
                effective_cached = restore_or_clear(p, cached);
            }
        }
    } else {
        // No rollback needed (extension only or full match). Drop checkpoints
        // whose pos_max would land past where we're now operating.
        p.retain_checkpoints_below(cached.max(1));
    }

    let prompt_batch_limit = p.ctx.n_batch().max(1) as usize;
    let mut batch = LlamaBatch::new(prompt_batch_limit, 1);

    if effective_cached < prompt_len {
        // Decode the new suffix.
        let suffix = &new_tokens[effective_cached..];
        for (chunk_index, chunk) in suffix.chunks(prompt_batch_limit).enumerate() {
            batch.clear();
            for (offset, token) in chunk.iter().copied().enumerate() {
                let abs = effective_cached + chunk_index * prompt_batch_limit + offset;
                let is_last_prompt_token = abs + 1 == prompt_len;
                batch
                    .add(token, abs as i32, &[0], is_last_prompt_token)
                    .map_err(|e| format!("Batch add failed: {e}"))?;
            }
            if batch.n_tokens() == 0 {
                return Err(format!(
                    "BUG: empty prompt batch at chunk {chunk_index} (suffix.len={}, prompt_batch_limit={})",
                    suffix.len(),
                    prompt_batch_limit,
                ));
            }
            p.ctx
                .decode(&mut batch)
                .map_err(|e| format!("Prompt decode failed: {e}"))?;

            let n_tokens_decoded =
                effective_cached + chunk_index * prompt_batch_limit + chunk.len();
            maybe_create_checkpoint(p, checkpoint_params, n_tokens_decoded, prompt_len);
        }
    } else {
        // Whole prompt already cached. Roll back the last position by one and
        // re-decode it so the sampler has a fresh `logits=true` slot to read.
        // Only reachable when trim is supported (otherwise effective_cached
        // would have been reset to 0 above).
        let removed = p
            .ctx
            .clear_kv_cache_seq(Some(0), Some((prompt_len - 1) as u32), None)
            .map_err(|e| format!("KV cache trim failed: {e:?}"))?;
        if !removed {
            return Err(format!(
                "KV cache trim (rollback) returned false at pos {}",
                prompt_len - 1
            ));
        }
        batch.clear();
        batch
            .add(
                new_tokens[prompt_len - 1],
                (prompt_len - 1) as i32,
                &[0],
                true,
            )
            .map_err(|e| format!("Batch add failed: {e}"))?;
        p.ctx
            .decode(&mut batch)
            .map_err(|e| format!("Prompt decode failed: {e}"))?;
    }

    Ok((batch, effective_cached))
}
