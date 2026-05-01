use std::sync::atomic::{AtomicBool, Ordering};

use crate::checkpoint::{PersistentCtx, ensure_persistent_ctx};
use crate::prompt::build_prompt;
use crate::sampling::sample_tokens_from_pos;
use crate::slot::{SlotEntry, get_common_prefix};
use crate::types::{InferenceParams, InferenceResult, PromptBuildResult, StreamSender};
use crate::worker::{CANCEL_ERR, RunCtx};

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
pub(crate) fn run_image_inference<'m>(
    ctx: &RunCtx<'_, 'm>,
    persistent: &mut Option<PersistentCtx<'m>>,
    req: &InferenceParams,
    stream_tx: Option<&StreamSender>,
) -> Result<InferenceResult, String> {
    use llama_cpp_2::llama_batch::LlamaBatch;

    let prompt_build = build_prompt(ctx.model, &req.prepared_request)?;
    let prompt = prompt_build.prompt.as_str();

    // The worker dispatches here only when `ctx.mtmd_ctx.is_some()`, but
    // return a typed error rather than panicking if a future refactor breaks
    // that contract — the worker thread is the only one that can recover.
    let mtmd = ctx
        .mtmd_ctx
        .ok_or_else(|| "BUG: run_image_inference called without mtmd context".to_string())?;

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
    if prompt_len > ctx.n_ctx as usize {
        return Err(format!(
            "Multimodal prompt {prompt_len} entries exceeds n_ctx {}",
            ctx.n_ctx
        ));
    }

    let p = ensure_persistent_ctx(ctx.backend, ctx.model, ctx.n_ctx, ctx.kv_cache, persistent)?;

    let cached_lcp = get_common_prefix(&p.last_entries, &new_entries);
    let suffix_has_image = new_entries[cached_lcp..]
        .iter()
        .any(|e| matches!(e, SlotEntry::Image { .. }));
    let prefix_has_image = new_entries[..cached_lcp]
        .iter()
        .any(|e| matches!(e, SlotEntry::Image { .. }));

    log::debug!(
        "mtmd prefix-cache: prompt_len={prompt_len} last_entries.len={} \
         cached_lcp={cached_lcp} suffix_has_image={suffix_has_image} \
         prefix_has_image={prefix_has_image} trim_unsupported={}",
        p.last_entries.len(),
        p.trim_unsupported,
    );

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
        p.clear_checkpoints();
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
                    p.retain_checkpoints_below(cached_lcp);
                    p.last_entries.truncate(cached_lcp);
                }
                Ok(false) => {
                    log::info!("mtmd: partial KV trim refused; full re-eval.");
                    p.trim_unsupported = true;
                    p.ctx.clear_kv_cache();
                    p.last_entries.clear();
                    p.clear_checkpoints();
                    let n_past = chunks
                        .eval_chunks(mtmd, &p.ctx, 0, 0, n_batch, true)
                        .map_err(|e| format!("Multimodal eval_chunks failed: {e}"))?;
                    return finish_image_sample(
                        ctx.model,
                        p,
                        new_entries,
                        prompt_tokens,
                        0,
                        n_past as i32,
                        &prompt_build,
                        req,
                        stream_tx,
                        ctx.cancel,
                    );
                }
                Err(e) => return Err(format!("KV cache trim failed: {e:?}")),
            }
        } else {
            p.retain_checkpoints_below(cached_lcp.max(1));
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
                        p.clear_checkpoints();
                        let n_past = chunks
                            .eval_chunks(mtmd, &p.ctx, 0, 0, n_batch, true)
                            .map_err(|e| format!("Multimodal eval_chunks failed: {e}"))?;
                        return finish_image_sample(
                            ctx.model,
                            p,
                            new_entries,
                            prompt_tokens,
                            0,
                            n_past as i32,
                            &prompt_build,
                            req,
                            stream_tx,
                            ctx.cancel,
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
            // Bail at chunk boundaries on shutdown; mtmd suffix decode can be
            // long for high-resolution image prompts.
            if ctx.cancel.load(Ordering::Relaxed) {
                return Err(CANCEL_ERR.to_string());
            }
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
        ctx.model,
        p,
        new_entries,
        prompt_tokens,
        effective_cached as u64,
        n_past,
        &prompt_build,
        req,
        stream_tx,
        ctx.cancel,
    )
}

/// Common tail for the mtmd path: commit `new_entries` to the slot, then
/// hand off to `sample_tokens_from_pos` so the persistent slot picks up the
/// generated tokens.
#[allow(clippy::too_many_arguments)]
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
    cancel: &AtomicBool,
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
        cancel,
    );
    if result.is_err() {
        // Slot is in an unknown state on sampling failure; the next request
        // will rebuild from scratch.
        p.last_entries.clear();
        p.ctx.clear_kv_cache();
        p.clear_checkpoints();
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
