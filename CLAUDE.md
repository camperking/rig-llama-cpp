# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`rig-llama-cpp` is a Rust library that integrates local GGUF model inference (via llama.cpp) with the [Rig](https://github.com/0xPlaygrounds/rig) LLM framework. A loaded model is a `rig_core::Model` (a completion wire bound to an in-process transport), so local models can be used interchangeably with cloud-based models. Agents come from the separate `rig-agent` crate (a dev-dependency here, used by examples and e2e tests).

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
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Path to GGUF model file |
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

### Key Types

- **`Client`** (`src/client.rs`) — Entry point. Loads a GGUF model and spawns the inference worker thread. `completion_model(id)` returns a `Model`.
- **`Model`** (`src/wire.rs`) — Alias for `rig_core::Model<CompletionWire, Runtime>`. `CompletionWire` is Rig's `Wire` + `ReplayTarget` (describes the model, maps generation options, decodes replies); `Runtime` is the `Transport` that sends each request to the worker.
- **`Frame` / `OutputPart`** (`src/types.rs`) — What the worker sends back: text pieces, whole reasoning blocks, whole tool calls, then an end frame with token counts and the raw text. `FrameDecoder` writes them into Rig's completion turn; `RawDocument` turns the end frame into `CompletionResponse::raw` (a serialized `RawResponse`).
- **`SamplingParams`** — Token sampling configuration (temperature, top-k, top-p, min-p, repetition/presence penalties).
- **`EmbeddingModelHandle`** (`src/embedding.rs`) — Alias for `rig_core::Model<Local<Embedding>, EmbeddingRuntime>`.

### Request Flow

1. `model.call(request)` / `model.stream(request)`; Rig's `Completion::prepare` validates the request, puts documents into the history, adapts it to what `CompletionWire` accepts and checks the generation options
2. `Runtime` runs `prepare_request()` (OpenAI-shaped messages + tools JSON) and sends an `InferenceCommand` to the worker thread
3. Worker runs `run_inference()`: renders the chat template, tokenizes, reuses the KV-cache prefix, then samples tokens one at a time
4. Streaming turns send each text piece as a frame (tool-call and structured-output turns are buffered and flushed at the end); unary turns are parsed by `parse_completion_output()` into `OutputPart`s
5. `FrameDecoder` assembles the frames into Rig's `CompletionResponse`

### Chat Template Handling

llama.cpp's built-in chat template is applied when available. If the model has no template, ChatML format is used as fallback. Tool schemas are serialized as JSON and injected into the system prompt section.

### `.gguf` Files

GGUF model files are gitignored. The integration test expects `Qwen3.5-2B-Q4_K_M.gguf` in the repo root.
