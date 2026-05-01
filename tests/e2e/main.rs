//! Single integration binary covering all model-bearing tests.
//!
//! All tests in this binary are `#[ignore]` and download GGUF fixtures
//! via `hf-hub` on first run (cached at `~/.cache/huggingface/hub`).
//! Plan for ~20 GB of downloads on a cold cache.
//!
//! Run with:
//!
//! ```sh
//! cargo test --test e2e --features mtmd -- --ignored --nocapture
//! ```

mod common;
mod embedding;
mod gemma;
mod qwen;
mod reload;
