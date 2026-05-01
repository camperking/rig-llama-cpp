use std::collections::VecDeque;
use std::num::NonZeroU32;

use crate::slot::SlotEntry;
use crate::types::{CheckpointParams, KvCacheParams};

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
pub(crate) struct PersistentCtx<'m> {
    pub(crate) ctx: llama_cpp_2::context::LlamaContext<'m>,
    pub(crate) last_entries: Vec<SlotEntry>,
    /// Set to true once we've observed that this model's memory implementation
    /// rejects partial cache trims (`clear_kv_cache_seq` returning `Ok(false)`).
    /// Recurrent/hybrid models like Mamba/RWKV/Jamba can't roll back the
    /// recurrent state to an arbitrary position. When this flag is set we
    /// route rollback requests through the checkpoint-restore path (or full
    /// clear when no usable checkpoint exists). Extension-mode reuse
    /// (`cached == last_entries.len()`) works regardless.
    pub(crate) trim_unsupported: bool,
    /// In-memory partial-state snapshots, oldest first. Bounded by
    /// `CheckpointParams::max_checkpoints`.
    checkpoints: VecDeque<Checkpoint>,
}

impl PersistentCtx<'_> {
    /// Drop checkpoints whose covered range overlaps or exceeds `upper`.
    pub(crate) fn retain_checkpoints_below(&mut self, upper: usize) {
        self.checkpoints.retain(|c| (c.pos_max as usize) < upper);
    }

    /// Forget every cached checkpoint.
    #[cfg(feature = "mtmd")]
    pub(crate) fn clear_checkpoints(&mut self) {
        self.checkpoints.clear();
    }

    /// Number of checkpoints currently held — for diagnostic logging.
    pub(crate) fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }
}

/// Lazily construct (or reuse) the worker's persistent context and return a
/// mutable reference to it. The returned reference borrows `persistent` for
/// its lifetime, so callers can chain operations on the live context without
/// re-checking the `Option` — when this function returns `Ok`, the slot is
/// guaranteed to be `Some(_)`.
pub(crate) fn ensure_persistent_ctx<'a, 'm>(
    backend: &'m llama_cpp_2::llama_backend::LlamaBackend,
    model: &'m llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
    kv_cache: &KvCacheParams,
    persistent: &'a mut Option<PersistentCtx<'m>>,
) -> Result<&'a mut PersistentCtx<'m>, String> {
    use llama_cpp_2::context::params::LlamaContextParams;

    // Initialise the slot first, then return the mutable borrow. Done in two
    // steps because the stable borrow checker (pre-Polonius) refuses the
    // shorter `if let Some(p) = persistent.as_mut() { return Ok(p) }`-then-
    // `persistent.insert(...)` form: the early-return borrow is treated as
    // outliving the second branch.
    if persistent.is_none() {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_type_k(kv_cache.type_k.into())
            .with_type_v(kv_cache.type_v.into());
        let ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| format!("Context creation failed: {e}"))?;
        *persistent = Some(PersistentCtx {
            ctx,
            last_entries: Vec::new(),
            trim_unsupported: false,
            checkpoints: VecDeque::new(),
        });
    }
    Ok(persistent
        .as_mut()
        .expect("persistent context was just initialised above"))
}

/// Roll the persistent context back to the most recent checkpoint that fits
/// within `cached`, or fully clear it if none exists. Used by recurrent /
/// hybrid models where partial trims aren't supported but a saved partial
/// state can recover prefix reuse.
///
/// Searches newest-first for a checkpoint with `n_tokens <= cached` and
/// `pos_max < cached` (so the snapshot covers a strict prefix of the new
/// prompt). If found and its data is restorable, the slot's tracked entries
/// are truncated to that length and the regular KV cache is trimmed past it.
/// Otherwise we full-clear so the caller decodes from scratch.
///
/// Returns the number of tokens now committed to the cache (`n_tokens` of the
/// restored checkpoint, or 0 on full clear).
pub(crate) fn restore_or_clear(p: &mut PersistentCtx<'_>, cached: usize) -> usize {
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

            log::debug!("restored checkpoint at n_tokens={n_tokens} (cached LCP was {cached})");
            return n_tokens;
        }

        log::warn!("state_seq_set_data_ext failed; clearing cache.");
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
pub(crate) fn maybe_create_checkpoint(
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
    let near_end =
        n_tokens_decoded + 4 + n_ubatch == prompt_len || n_tokens_decoded + 4 == prompt_len;

    let last_n_tokens = p.checkpoints.back().map(|c| c.n_tokens).unwrap_or(0);
    let cadence_ok = params.every_n_tokens > 0
        && n_tokens_decoded.saturating_sub(last_n_tokens) >= params.every_n_tokens as usize;

    if !(near_end || cadence_ok) {
        return;
    }
    if p.checkpoints
        .back()
        .is_some_and(|c| n_tokens_decoded.saturating_sub(c.n_tokens) < params.min_gap as usize)
    {
        return;
    }

    let size = p.ctx.state_seq_get_size_ext(
        0,
        llama_cpp_2::context::session::LlamaStateSeqFlags::PARTIAL_ONLY,
    );
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

    log::debug!(
        "checkpoint created at n_tokens={n_tokens_decoded} (size={} KiB, total={})",
        written / 1024,
        p.checkpoints.len(),
    );
}
