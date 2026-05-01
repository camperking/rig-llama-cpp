use std::collections::VecDeque;
use std::num::NonZeroU32;

use rig::completion::CompletionError;
use rig::streaming::RawStreamingChoice;
use tokio::sync::mpsc;

use crate::prompt::build_prompt;
#[cfg(feature = "mtmd")]
use crate::sampling::sample_tokens_from_pos;
use crate::sampling::sample_tokens;
use crate::slot::{SlotEntry, get_common_prefix};
use crate::types::{
    CheckpointParams, FitParams, InferenceCommand, InferenceParams, InferenceResult, KvCacheParams,
    PromptBuildResult, ResponseChannel, StreamChunk, StreamSender,
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
            log_level,
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

/// Persistent inference state carried across requests for prefix-cache reuse.
///
/// We hold one `LlamaContext` for the worker's lifetime instead of recreating it
/// per request, plus the entries currently decoded into KV-cache slot 0. On each
/// new prompt we compute the longest common prefix against `last_entries` and
/// only decode the suffix — the matching prefix already lives in the cache.
///
/// `last_entries` carries text tokens and image positions in a single flat
/// vector (see [`SlotEntry`]). For text-only conversations every entry is
/// `Text`; for image conversations the entries interleave text tokens with
/// image groups identified by FNV-1a hash, so an image stays in the matched
/// prefix when the next turn re-attaches the same image.
///
/// The lifetime `'m` borrows from the active `WorkerModel::model`. The struct is
/// only ever constructed and dropped inside `handle_until_reload`, so it never
/// outlives the model reference it borrows from.
/// Snapshot of the partial seq state (recurrent + SWA KV) at a specific
/// point in the conversation. Used by hybrid models (Qwen 3.5, Jamba, etc.)
/// to recover prefix-cache reuse when a partial trim isn't possible.
///
/// Mirrors llama-server's `server_prompt_checkpoint`.
struct Checkpoint {
    /// Min position covered by the saved partial state at save time. We
    /// don't currently consult this (the binding doesn't expose
    /// `seq_pos_min`), but kept for diagnostic logging.
    #[allow(dead_code)]
    pos_min: i32,
    /// Max position covered by the saved partial state — equal to
    /// `n_tokens - 1`. Used to verify the snapshot is consistent with the
    /// requested rollback target.
    pos_max: i32,
    /// Number of prompt tokens that had been decoded into the cache when
    /// the snapshot was taken. The next request can resume decoding at
    /// position `n_tokens` if its LCP length is at least this large.
    n_tokens: usize,
    /// Serialized partial state. Size matches whatever
    /// `state_seq_get_size_ext(0, PARTIAL_ONLY)` returned at save time.
    data: Vec<u8>,
}

struct PersistentCtx<'m> {
    ctx: llama_cpp_2::context::LlamaContext<'m>,
    last_entries: Vec<SlotEntry>,
    /// Set to true once we've observed that this model's memory implementation
    /// rejects partial cache trims (`clear_kv_cache_seq` returning `Ok(false)`).
    /// Recurrent/hybrid models like Mamba/RWKV/Jamba can't roll back the
    /// recurrent state to an arbitrary position. When this flag is set we
    /// route rollback requests through the checkpoint-restore path (or full
    /// clear when no usable checkpoint exists). Extension-mode reuse
    /// (`cached == last_entries.len()`) works regardless.
    trim_unsupported: bool,
    /// In-memory partial-state snapshots, oldest first. Bounded by
    /// `CheckpointParams::max_checkpoints`.
    checkpoints: VecDeque<Checkpoint>,
}

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

fn ensure_persistent_ctx<'m>(
    backend: &'m llama_cpp_2::llama_backend::LlamaBackend,
    model: &'m llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    persistent: &mut Option<PersistentCtx<'m>>,
) -> Result<(), String> {
    use llama_cpp_2::context::params::LlamaContextParams;

    if persistent.is_some() {
        return Ok(());
    }
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_type_k(kv_cache.type_k)
        .with_type_v(kv_cache.type_v);
    let ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("Context creation failed: {e}"))?;
    *persistent = Some(PersistentCtx {
        ctx,
        last_entries: Vec::new(),
        trim_unsupported: false,
        checkpoints: VecDeque::new(),
    });
    Ok(())
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
            p.checkpoints.len(),
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
                p.checkpoints.retain(|c| (c.pos_max as usize) < cached);
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
        p.checkpoints.retain(|c| (c.pos_max as usize) < cached.max(1));
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

/// Attempt to restore the closest in-memory checkpoint that covers a prefix
/// of length `<= cached`. Returns the number of tokens now committed to the
/// cache (`n_tokens` of the restored checkpoint, or `0` if no checkpoint
/// was usable and we had to fully clear).
///
/// On success, the regular KV cache is also trimmed to the checkpoint's
/// position so the cache state is consistent for subsequent decoding.
fn restore_or_clear(p: &mut PersistentCtx<'_>, cached: usize) -> usize {
    // Newest-first search for a checkpoint whose covered range fits within
    // the prefix we want to keep. A checkpoint with `n_tokens > cached` would
    // overshoot the LCP and put the cache into a state the new prompt
    // doesn't actually share.
    let candidate_idx = p
        .checkpoints
        .iter()
        .rposition(|c| c.n_tokens <= cached && (c.pos_max as usize) < cached);

    if let Some(idx) = candidate_idx {
        let n_tokens = p.checkpoints[idx].n_tokens;
        let restored = unsafe {
            p.ctx.state_seq_set_data_ext(
                &p.checkpoints[idx].data,
                0,
                llama_cpp_2::context::session::LlamaStateSeqFlags::PARTIAL_ONLY,
            )
        };

        if restored {
            // After restoring partial state, the recurrent/SWA state is at
            // the checkpoint's position. Trim any stale regular-KV positions
            // ahead of it. For hybrid models this trim now succeeds because
            // the recurrent tail is no longer in the trim range.
            let _ = p
                .ctx
                .clear_kv_cache_seq(Some(0), Some(n_tokens as u32), None);

            // Truncate the tracked entries to match.
            p.last_entries.truncate(n_tokens);

            // Drop checkpoints AFTER this one (they captured later state we
            // just rolled past).
            p.checkpoints.truncate(idx + 1);

            eprintln!(
                "[rig-llama-cpp] restored checkpoint at n_tokens={n_tokens} (cached LCP was {cached})"
            );
            return n_tokens;
        }

        eprintln!("[rig-llama-cpp] state_seq_set_data_ext failed; clearing cache.");
    }

    // No usable checkpoint; full clear.
    p.ctx.clear_kv_cache();
    p.last_entries.clear();
    p.checkpoints.clear();
    0
}

/// Decide whether to snapshot the partial seq state at this point in the
/// prompt prefill, and do it if so. Mirrors the cadence used by
/// `llama-server`: always near the end of the prompt, plus optionally
/// every `every_n_tokens` during the bulk of the prompt.
fn maybe_create_checkpoint(
    p: &mut PersistentCtx<'_>,
    params: CheckpointParams,
    n_tokens_decoded: usize,
    prompt_len: usize,
) {
    if params.max_checkpoints == 0 {
        return;
    }
    if n_tokens_decoded < params.min_tokens as usize {
        return;
    }

    let n_ubatch = p.ctx.n_ubatch().max(1) as usize;
    // Same offsets as llama-server: 4 + n_ubatch and 4 tokens before the end
    // of the prompt.
    let near_end = n_tokens_decoded + 4 + n_ubatch == prompt_len
        || n_tokens_decoded + 4 == prompt_len;

    let last_n_tokens = p.checkpoints.back().map(|c| c.n_tokens).unwrap_or(0);
    let cadence_ok = params.every_n_tokens > 0
        && n_tokens_decoded.saturating_sub(last_n_tokens) >= params.every_n_tokens as usize;

    if !(near_end || cadence_ok) {
        return;
    }
    if p
        .checkpoints
        .back()
        .is_some_and(|c| n_tokens_decoded.saturating_sub(c.n_tokens) < params.min_gap as usize)
    {
        return;
    }

    let size = p
        .ctx
        .state_seq_get_size_ext(0, llama_cpp_2::context::session::LlamaStateSeqFlags::PARTIAL_ONLY);
    if size == 0 {
        return;
    }

    let mut data = vec![0u8; size];
    let written = unsafe {
        p.ctx.state_seq_get_data_ext(
            data.as_mut_ptr(),
            0,
            llama_cpp_2::context::session::LlamaStateSeqFlags::PARTIAL_ONLY,
        )
    };
    if written == 0 {
        return;
    }
    data.truncate(written);

    while p.checkpoints.len() >= params.max_checkpoints as usize {
        p.checkpoints.pop_front();
    }

    let pos_max = (n_tokens_decoded as i32).saturating_sub(1);
    p.checkpoints.push_back(Checkpoint {
        pos_min: 0,
        pos_max,
        n_tokens: n_tokens_decoded,
        data,
    });

    if crate::llama_logs_enabled() {
        eprintln!(
            "[rig-llama-cpp] checkpoint created at n_tokens={n_tokens_decoded} (size={} KiB, total={})",
            written / 1024,
            p.checkpoints.len(),
        );
    }
}

/// Multimodal (image) inference with the same persistent context the text
/// path uses, plus image-aware prefix-cache reuse.
///
/// We stamp each `MtmdBitmap` with a hex-encoded FNV-1a hash of its bytes via
/// `set_id`. mtmd propagates this id into the resulting `MtmdInputChunk`s,
/// which lets us round-trip image identity through the prefix diff: an image
/// chunk in the slot matches an image chunk in the new prompt iff their ids
/// and token counts agree (atomic per-group match — see `slot::SlotEntry`).
///
/// Reuse strategy:
/// - When the matched prefix covers all chunks (no suffix), or when the
///   diverging suffix is text-only, we trim the KV tail and decode only the
///   new text tokens — the image's KV state stays put.
/// - When the suffix contains an image chunk, or when a prefix rollback is
///   needed on a model whose memory rejects partial trims (recurrent /
///   hybrid), we full-clear the cache and run `eval_chunks` from scratch.
///   Image-suffix reuse via `mtmd_helper_eval_chunk_single` is not
///   implemented yet (the safe binding doesn't expose per-chunk eval).
#[cfg(feature = "mtmd")]
fn run_image_inference<'m>(
    backend: &'m llama_cpp_2::llama_backend::LlamaBackend,
    model: &'m llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    persistent: &mut Option<PersistentCtx<'m>>,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
    mtmd_ctx: Option<&llama_cpp_2::mtmd::MtmdContext>,
) -> Result<InferenceResult, String> {
    use llama_cpp_2::llama_batch::LlamaBatch;

    let prompt_build = build_prompt(model, &req.prepared_request)?;
    let prompt = prompt_build.prompt.as_str();

    let mtmd = mtmd_ctx.expect("run_image_inference called without mtmd context");

    // Build bitmaps and stamp each with the FNV id so chunk ids round-trip.
    let bitmaps: Vec<llama_cpp_2::mtmd::MtmdBitmap> = req
        .prepared_request
        .images
        .iter()
        .map(|img| -> Result<_, String> {
            let bm = llama_cpp_2::mtmd::MtmdBitmap::from_buffer(mtmd, &img.bytes)
                .map_err(|e| format!("Failed to create bitmap from image data: {e}"))?;
            bm.set_id(&format!("{:016x}", img.hash))
                .map_err(|e| format!("Failed to set bitmap id: {e}"))?;
            Ok(bm)
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
    let new_entries = build_mtmd_candidate(&chunks)?;
    let prompt_len = new_entries.len();

    if prompt_len == 0 {
        return Err("Empty prompt after multimodal tokenization".to_string());
    }
    if prompt_len > n_ctx as usize {
        return Err(format!(
            "Multimodal prompt {prompt_len} entries exceeds n_ctx {n_ctx}"
        ));
    }

    ensure_persistent_ctx(backend, model, n_ctx, kv_cache, persistent)?;
    let p = persistent.as_mut().unwrap();

    let cached_lcp = get_common_prefix(&p.last_entries, &new_entries);
    let suffix_has_image = new_entries[cached_lcp..]
        .iter()
        .any(|e| matches!(e, SlotEntry::Image { .. }));
    let prefix_has_image = new_entries[..cached_lcp]
        .iter()
        .any(|e| matches!(e, SlotEntry::Image { .. }));

    if crate::llama_logs_enabled() {
        eprintln!(
            "[rig-llama-cpp] mtmd prefix-cache: prompt_len={prompt_len} last_entries.len={} \
             cached_lcp={cached_lcp} suffix_has_image={suffix_has_image} \
             prefix_has_image={prefix_has_image} trim_unsupported={}",
            p.last_entries.len(),
            p.trim_unsupported,
        );
    }

    // A rollback is needed iff the slot has more entries than the matched
    // prefix. On a hybrid/recurrent model this is only safe when the rollback
    // range is empty or the partial-trim path actually works for it.
    let need_rollback = cached_lcp < p.last_entries.len();
    let must_full_reeval =
        suffix_has_image || (need_rollback && p.trim_unsupported && prefix_has_image);

    let n_batch = p.ctx.n_batch() as i32;

    let (effective_cached, n_past) = if must_full_reeval || cached_lcp == 0 {
        // Full clear + full re-eval through mtmd. Drops any partial image KV.
        p.ctx.clear_kv_cache();
        p.last_entries.clear();
        p.checkpoints.clear();
        let n_past = chunks
            .eval_chunks(mtmd, &p.ctx, 0, 0, n_batch, true)
            .map_err(|e| format!("Multimodal eval_chunks failed: {e}"))?;
        (0usize, n_past)
    } else {
        // Reuse the matched prefix; the suffix is text-only by the guards above.
        if need_rollback {
            // Trim the KV tail. If the model rejects partial trims, fall back
            // to a full re-eval rather than corrupting state.
            match p
                .ctx
                .clear_kv_cache_seq(Some(0), Some(cached_lcp as u32), None)
            {
                Ok(true) => {
                    p.checkpoints
                        .retain(|c| (c.pos_max as usize) < cached_lcp);
                    p.last_entries.truncate(cached_lcp);
                }
                Ok(false) => {
                    eprintln!(
                        "[rig-llama-cpp] mtmd: partial KV trim refused; full re-eval."
                    );
                    p.trim_unsupported = true;
                    p.ctx.clear_kv_cache();
                    p.last_entries.clear();
                    p.checkpoints.clear();
                    let n_past = chunks
                        .eval_chunks(mtmd, &p.ctx, 0, 0, n_batch, true)
                        .map_err(|e| format!("Multimodal eval_chunks failed: {e}"))?;
                    return finish_image_sample(
                        model,
                        p,
                        new_entries,
                        prompt_tokens,
                        0,
                        n_past as i32,
                        &prompt_build,
                        req,
                        stream_tx,
                    );
                }
                Err(e) => return Err(format!("KV cache trim failed: {e:?}")),
            }
        } else {
            p.checkpoints
                .retain(|c| (c.pos_max as usize) < cached_lcp.max(1));
        }

        // If the entire prompt was already cached, roll back the last position
        // by one and re-decode it so the sampler has fresh logits.
        let (start, suffix_tokens): (usize, Vec<llama_cpp_2::token::LlamaToken>) =
            if cached_lcp >= prompt_len {
                let last_idx = prompt_len - 1;
                let token = match new_entries[last_idx] {
                    SlotEntry::Text(t) => t,
                    SlotEntry::Image { .. } => {
                        // Last slot is an image — can't recompute with a text
                        // batch. Fall through to a full re-eval.
                        p.ctx.clear_kv_cache();
                        p.last_entries.clear();
                        p.checkpoints.clear();
                        let n_past = chunks
                            .eval_chunks(mtmd, &p.ctx, 0, 0, n_batch, true)
                            .map_err(|e| format!("Multimodal eval_chunks failed: {e}"))?;
                        return finish_image_sample(
                            model,
                            p,
                            new_entries,
                            prompt_tokens,
                            0,
                            n_past as i32,
                            &prompt_build,
                            req,
                            stream_tx,
                        );
                    }
                };
                let removed = p
                    .ctx
                    .clear_kv_cache_seq(Some(0), Some(last_idx as u32), None)
                    .map_err(|e| format!("KV cache trim failed: {e:?}"))?;
                if !removed {
                    return Err(format!(
                        "KV cache trim (rollback) returned false at pos {last_idx}"
                    ));
                }
                p.last_entries.truncate(last_idx);
                (last_idx, vec![token])
            } else {
                let mut tokens = Vec::with_capacity(prompt_len - cached_lcp);
                for entry in &new_entries[cached_lcp..] {
                    match entry {
                        SlotEntry::Text(t) => tokens.push(*t),
                        SlotEntry::Image { .. } => {
                            unreachable!(
                                "suffix_has_image guard should have routed image suffix to full re-eval"
                            )
                        }
                    }
                }
                (cached_lcp, tokens)
            };

        let prompt_batch_limit = p.ctx.n_batch().max(1) as usize;
        let mut batch = LlamaBatch::new(prompt_batch_limit, 1);
        let total = suffix_tokens.len();
        for (chunk_index, chunk) in suffix_tokens.chunks(prompt_batch_limit).enumerate() {
            batch.clear();
            for (offset, token) in chunk.iter().copied().enumerate() {
                let abs = start + chunk_index * prompt_batch_limit + offset;
                let is_last = abs + 1 == prompt_len;
                batch
                    .add(token, abs as i32, &[0], is_last)
                    .map_err(|e| format!("Batch add failed: {e}"))?;
            }
            if batch.n_tokens() == 0 {
                return Err(format!(
                    "BUG: empty mtmd-suffix batch at chunk {chunk_index} (suffix.len={}, prompt_batch_limit={})",
                    total, prompt_batch_limit,
                ));
            }
            p.ctx
                .decode(&mut batch)
                .map_err(|e| format!("Mtmd-suffix prompt decode failed: {e}"))?;
        }

        (start, prompt_len as i32)
    };

    finish_image_sample(
        model,
        p,
        new_entries,
        prompt_tokens,
        effective_cached as u64,
        n_past,
        &prompt_build,
        req,
        stream_tx,
    )
}

/// Common tail for the mtmd path: commit `new_entries` to the slot, then
/// hand off to `sample_tokens_from_pos` so the persistent slot picks up the
/// generated tokens.
#[cfg(feature = "mtmd")]
fn finish_image_sample(
    model: &llama_cpp_2::model::LlamaModel,
    p: &mut PersistentCtx<'_>,
    new_entries: Vec<SlotEntry>,
    prompt_tokens: u64,
    cached_input_tokens: u64,
    n_past: i32,
    prompt_build: &PromptBuildResult,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
) -> Result<InferenceResult, String> {
    use llama_cpp_2::llama_batch::LlamaBatch;

    p.last_entries = new_entries;
    let prompt_batch_limit = p.ctx.n_batch().max(1) as usize;
    let mut batch = LlamaBatch::new(prompt_batch_limit, 1);

    let result = sample_tokens_from_pos(
        model,
        &mut p.ctx,
        &mut batch,
        prompt_build,
        req,
        stream_tx,
        prompt_tokens,
        cached_input_tokens,
        n_past,
        &mut p.last_entries,
    );
    if result.is_err() {
        // Slot is in an unknown state on sampling failure; the next request
        // will rebuild from scratch.
        p.last_entries.clear();
        p.ctx.clear_kv_cache();
        p.checkpoints.clear();
    }
    result
}

/// Walk an [`MtmdInputChunks`] collection and produce a flat slot-entry vector
/// that mirrors what the chunks contribute to the KV cache.
///
/// Text chunks contribute one [`SlotEntry::Text`] per token. Image and audio
/// chunks contribute `n_tokens` [`SlotEntry::Image`] entries that all share
/// the same `(hash, group_id)`, so the prefix matcher treats each image
/// atomically.
#[cfg(feature = "mtmd")]
fn build_mtmd_candidate(
    chunks: &llama_cpp_2::mtmd::MtmdInputChunks,
) -> Result<Vec<SlotEntry>, String> {
    use llama_cpp_2::mtmd::MtmdInputChunkType;

    let mut out = Vec::with_capacity(chunks.total_tokens());
    let mut group_id: u32 = 0;

    for i in 0..chunks.len() {
        let chunk = chunks
            .get(i)
            .ok_or_else(|| format!("Failed to access mtmd chunk at index {i}"))?;
        match chunk.chunk_type() {
            MtmdInputChunkType::Text => {
                let toks = chunk.text_tokens().ok_or("Text chunk without tokens")?;
                for &t in toks {
                    out.push(SlotEntry::Text(t));
                }
            }
            MtmdInputChunkType::Image | MtmdInputChunkType::Audio => {
                let id = chunk
                    .id()
                    .ok_or("Image/audio chunk missing id (set_id not propagated?)")?;
                let hash = u64::from_str_radix(id.trim(), 16)
                    .map_err(|e| format!("Image chunk id {id:?} is not a 16-hex FNV: {e}"))?;
                let n = chunk.n_tokens();
                for _ in 0..n {
                    out.push(SlotEntry::Image { hash, group_id });
                }
                group_id = group_id.wrapping_add(1);
            }
        }
    }
    Ok(out)
}
