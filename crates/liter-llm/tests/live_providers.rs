//! Integration tests that hit real LLM provider APIs.
//!
//! Gated on environment variables — tests skip gracefully when the
//! corresponding provider key is not set.  Safe to run in CI when
//! secrets are configured, zero-cost when they are not.
//!
//! # Environment variables
//!
//! | Variable | Provider |
//! |----------|----------|
//! | `OPENAI_API_KEY` | OpenAI |
//! | `ANTHROPIC_API_KEY` | Anthropic |
//! | `GEMINI_API_KEY` | Google AI (Gemini) |

use liter_llm::{ChatCompletionRequest, ClientConfigBuilder, DefaultClient, EmbeddingInput, EmbeddingRequest};

#[path = "live_providers/anthropic.rs"]
mod anthropic;
#[path = "live_providers/cross_provider.rs"]
mod cross_provider;
#[path = "live_providers/google_ai.rs"]
mod google_ai;
#[path = "live_providers/openai.rs"]
mod openai;

// ── Skip macro ──────────────────────────────────────────────────────────────

/// Skip a test if the named env var is not set or empty.
macro_rules! require_env {
    ($var:expr) => {
        match std::env::var($var) {
            Ok(val) if !val.is_empty() => val,
            _ => {
                eprintln!("SKIP: {} not set, skipping live provider test", $var);
                return;
            }
        }
    };
}
pub(crate) use require_env;

// ── Client factories ────────────────────────────────────────────────────────

pub fn openai_client(api_key: &str) -> DefaultClient {
    let config = ClientConfigBuilder::new(api_key).max_retries(2).build();
    DefaultClient::new(config, Some("openai/gpt-4o-mini")).unwrap()
}

pub fn anthropic_client(api_key: &str) -> DefaultClient {
    let config = ClientConfigBuilder::new(api_key).max_retries(2).build();
    DefaultClient::new(config, Some("anthropic/claude-haiku-4-5-20251001")).unwrap()
}

pub fn google_ai_client(api_key: &str) -> DefaultClient {
    let config = ClientConfigBuilder::new(api_key).max_retries(2).build();
    DefaultClient::new(config, Some("gemini/gemini-2.5-flash-lite")).unwrap()
}

// ── Shared request builders ─────────────────────────────────────────────────

pub fn simple_chat_request(model: &str) -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
    }))
    .expect("failed to build chat request from JSON")
}

pub fn simple_embed_request(model: &str) -> EmbeddingRequest {
    EmbeddingRequest {
        model: model.into(),
        input: EmbeddingInput::Single("hello world".into()),
        encoding_format: None,
        dimensions: None,
        user: None,
    }
}

// ── Assertion helpers ───────────────────────────────────────────────────────

pub fn assert_chat_response_valid(resp: &liter_llm::ChatCompletionResponse, label: &str) {
    assert!(!resp.choices.is_empty(), "{label}: choices should not be empty");
    let choice = &resp.choices[0];
    assert!(
        choice.message.content.as_ref().is_some_and(|c| !c.is_empty()),
        "{label}: first choice content should be non-empty"
    );
    assert!(
        choice.finish_reason.is_some(),
        "{label}: finish_reason should be present"
    );
    assert!(!resp.model.is_empty(), "{label}: model field should be non-empty");
}
