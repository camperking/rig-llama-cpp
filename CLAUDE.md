# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`rig-llama-cpp` is a Rust library that integrates local GGUF model inference (via llama.cpp) with the [Rig](https://github.com/0xPlaygrounds/rig) LLM framework. It implements Rig's `CompletionModel` trait so local models can be used interchangeably with cloud-based models.

## Build Commands

```bash
# Default build (Vulkan backend)
cargo build

# Feature-specific builds (mutually exclusive GPU backends)
cargo build --no-default-features --features cuda
cargo build --no-default-features --features rocm
cargo build --no-default-features --features metal
cargo build --no-default-features --features openmp  # CPU only
```

## Running Tests

Integration tests require a GGUF model file and are marked `#[ignore]` by default:

```bash
# Run unit tests (no model needed)
cargo test

# Run the main integration test (requires Qwen3.5-2B-Q4_K_M.gguf in cwd)
MODEL_PATH=./Qwen3.5-2B-Q4_K_M.gguf cargo test -- --ignored --nocapture

# Run sequential reload test
RIG_MODEL_A=./model_a.gguf RIG_MODEL_B=./model_b.gguf cargo test sequential -- --ignored
```

## Running Examples

```bash
MODEL_PATH=./model.gguf cargo run --example completion
MODEL_PATH=./model.gguf cargo run --example streaming
MODEL_PATH=./model.gguf cargo run --example stream_chat

# With GPU offloading
N_GPU_LAYERS=20 MODEL_PATH=./model.gguf cargo run --example completion
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Path to GGUF model file |
| `N_GPU_LAYERS` | Layers to offload to GPU (default: all) |
| `N_CTX` | Context window size |
| `RIG_LLAMA_CPP_LOGS` | Enable llama.cpp backend logs (`1` or `true`) |
| `RIG_MAX_TOKENS_PER_TURN` | Max tokens per generation turn |
| `RIG_TARGET_OUTPUT_TOKENS` | Target total output tokens |
| `RIG_LINES_PER_TURN` | Lines to generate per turn |
| `RIG_MAX_GENERATION_TURNS` | Max generation turns |

## Architecture

The key challenge this library solves: llama.cpp is synchronous and not thread-safe for concurrent use, but Rig's API is async. The solution is a **dedicated inference worker thread** that owns the model, communicating with async callers via channels.

```
Async caller (Tokio)
  └─> InferenceCommand via UnboundedSender<InferenceCommand>
        └─> inference_worker() on std::thread
              └─> llama-cpp-2 (synchronous, owns model)
                    └─> results via oneshot (completion) or mpsc (streaming)
```

### Key Types (`src/lib.rs`)

- **`Client`** — Entry point. Loads a GGUF model and spawns the inference worker thread. Holds the channel sender.
- **`Model`** — Clone-able handle to the loaded model. Implements Rig's `CompletionModel` + `CompletionClient` traits.
- **`SamplingParams`** — Token sampling configuration (temperature, top-k, top-p, min-p, repetition/presence penalties).
- **`StreamDeltaState`** — Accumulates streaming tool call fragments and reasoning deltas across tokens before emitting complete events.

### Request Flow

1. `Model::completion()` / `Model::stream()` is called
2. `prepare_request()` formats messages using llama.cpp's chat template (falls back to ChatML)
3. `InferenceCommand` sent to worker thread
4. Worker runs `run_inference()`: tokenizes prompt in batches, then samples tokens one at a time
5. Streaming responses go through `StreamDeltaState` to parse incremental tool calls
6. Final response parsed by `parse_completion_output()` into Rig's `AssistantContent`

### Chat Template Handling

llama.cpp's built-in chat template is applied when available. If the model has no template, ChatML format is used as fallback. Tool schemas are serialized as JSON and injected into the system prompt section.

### `.gguf` Files

GGUF model files are gitignored. The integration test expects `Qwen3.5-2B-Q4_K_M.gguf` in the repo root.
