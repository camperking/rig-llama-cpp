#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Model definitions ────────────────────────────────────────────────
QWEN_REPO="unsloth/Qwen3.5-2B-GGUF"
QWEN_MODEL="Qwen3.5-2B-Q4_K_M.gguf"
QWEN_MMPROJ_REMOTE="mmproj-BF16.gguf"
QWEN_MMPROJ_LOCAL="mmproj-BF16_Qwen3.5-2B.gguf"

GEMMA_REPO="unsloth/gemma-4-E4B-it-GGUF"
GEMMA_MODEL="gemma-4-E4B-it-Q4_K_M.gguf"
GEMMA_MMPROJ_REMOTE="mmproj-BF16.gguf"
GEMMA_MMPROJ_LOCAL="mmproj-BF16_gemma-4-E4B.gguf"

EMBED_REPO="nomic-ai/nomic-embed-text-v2-moe-GGUF"
EMBED_MODEL="nomic-embed-text-v2-moe.Q4_K_M.gguf"

IMAGE="test.jpg"

# ── Helpers ───────────────────────────────────────────────────────────
download_file() {
    local repo="$1" remote_name="$2" local_name="$3"
    if [[ -f "$local_name" ]]; then
        echo "  ✓ $local_name already present"
        return
    fi
    echo "  ↓ downloading $remote_name from $repo …"
    local tmpdir
    tmpdir=$(mktemp -d)
    hf download "$repo" "$remote_name" --local-dir "$tmpdir"
    mv "$tmpdir/$remote_name" "$local_name"
    rm -rf "$tmpdir"
    echo "  ✓ saved as $local_name"
}

section() { printf '\n══════ %s ══════\n' "$1"; }

# ── Download models ───────────────────────────────────────────────────
section "Downloading models"

echo "Qwen 3.5-2B:"
download_file "$QWEN_REPO" "$QWEN_MODEL"        "$QWEN_MODEL"
download_file "$QWEN_REPO" "$QWEN_MMPROJ_REMOTE" "$QWEN_MMPROJ_LOCAL"

echo "Gemma-4 E4B:"
download_file "$GEMMA_REPO" "$GEMMA_MODEL"        "$GEMMA_MODEL"
download_file "$GEMMA_REPO" "$GEMMA_MMPROJ_REMOTE" "$GEMMA_MMPROJ_LOCAL"

echo "Nomic Embed v2 MoE:"
download_file "$EMBED_REPO" "$EMBED_MODEL" "$EMBED_MODEL"

# ── Unit tests (no model needed) ─────────────────────────────────────
section "Unit tests"
cargo test

# ── E2E inference: Qwen ──────────────────────────────────────────────
section "E2E inference — Qwen 3.5-2B"
MODEL_PATH="./$QWEN_MODEL" \
    cargo test e2e_inference -- --ignored --nocapture

# ── E2E inference: Gemma-4 ───────────────────────────────────────────
section "E2E inference — Gemma-4 E4B"
MODEL_PATH="./$GEMMA_MODEL" \
    cargo test e2e_inference -- --ignored --nocapture

# ── Vision: Qwen ─────────────────────────────────────────────────────
section "Vision — Qwen 3.5-2B"
MODEL_PATH="./$QWEN_MODEL" \
MMPROJ_PATH="./$QWEN_MMPROJ_LOCAL" \
IMAGE_PATH="./$IMAGE" \
    cargo test --features mtmd vision_basic -- --ignored --nocapture

# ── Vision: Gemma-4 ──────────────────────────────────────────────────
section "Vision — Gemma-4 E4B"
MODEL_PATH="./$GEMMA_MODEL" \
MMPROJ_PATH="./$GEMMA_MMPROJ_LOCAL" \
IMAGE_PATH="./$IMAGE" \
    cargo test --features mtmd vision_basic -- --ignored --nocapture

# ── Per-model: Qwen thinking + tool roundtrip ────────────────────────
section "Qwen 3.5-2B — thinking + tool roundtrip"
MODEL_PATH="./$QWEN_MODEL" \
    cargo test qwen_thinking -- --ignored --nocapture
MODEL_PATH="./$QWEN_MODEL" \
    cargo test qwen_tool_roundtrip -- --ignored --nocapture

# ── Per-model: Gemma thinking + tool roundtrip ───────────────────────
section "Gemma-4 E4B — thinking + tool roundtrip"
MODEL_PATH="./$GEMMA_MODEL" \
    cargo test gemma_thinking -- --ignored --nocapture
MODEL_PATH="./$GEMMA_MODEL" \
    cargo test gemma_tool_roundtrip -- --ignored --nocapture

# ── KV cache quantization ────────────────────────────────────────────
section "KV cache Q8_0 — Qwen 3.5-2B"
MODEL_PATH="./$QWEN_MODEL" \
    cargo test e2e_kv_cache_q8_0 -- --ignored --nocapture

# ── Embedding ────────────────────────────────────────────────────────
section "Embedding — nomic-embed-text-v2-moe"
EMBEDDING_MODEL_PATH="./$EMBED_MODEL" \
    cargo test embedding_basic -- --ignored --nocapture

# ── Model reload ─────────────────────────────────────────────────────
section "Sequential model reload"
RIG_MODEL_A="./$QWEN_MODEL" \
RIG_MODEL_B="./$GEMMA_MODEL" \
    cargo test sequential -- --ignored --nocapture

section "All tests passed"
