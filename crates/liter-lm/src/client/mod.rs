pub mod config;

use std::future::Future;
use std::pin::Pin;

use futures_core::Stream;
use secrecy::ExposeSecret;

use crate::error::{LiterLmError, Result};
use crate::http;
use crate::provider::{self, OpenAiCompatibleProvider, OpenAiProvider, Provider};
use crate::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
    ModelsListResponse,
};

pub use config::{ClientConfig, ClientConfigBuilder};

/// A boxed future returning `Result<T>`.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// A boxed stream of `Result<T>`.
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = Result<T>> + Send + 'a>>;

/// Core LLM client trait.
pub trait LlmClient: Send + Sync {
    /// Send a chat completion request.
    fn chat(&self, req: ChatCompletionRequest) -> BoxFuture<'_, ChatCompletionResponse>;

    /// Send a streaming chat completion request.
    fn chat_stream(&self, req: ChatCompletionRequest) -> BoxFuture<'_, BoxStream<'_, ChatCompletionChunk>>;

    /// Send an embedding request.
    fn embed(&self, req: EmbeddingRequest) -> BoxFuture<'_, EmbeddingResponse>;

    /// List available models.
    fn list_models(&self) -> BoxFuture<'_, ModelsListResponse>;
}

/// Default client implementation backed by `reqwest`.
///
/// The provider is resolved **once** at construction time.  For most
/// use-cases a single client talks to a single provider, so detecting the
/// provider per-request is unnecessary overhead and creates subtle bugs (e.g.
/// the old `list_models` hardcoded `"gpt-4"` as the detection key).
///
/// If you need to talk to multiple providers, create one `DefaultClient` per
/// provider.
pub struct DefaultClient {
    config: ClientConfig,
    http: reqwest::Client,
    /// Provider resolved at construction; used for all requests.
    provider: Box<dyn Provider>,
}

impl DefaultClient {
    /// Build a client.
    ///
    /// `model_hint` guides provider auto-detection when no explicit
    /// `base_url` override is present in the config.  For example, passing
    /// `Some("groq/llama3-70b")` selects the Groq provider.  Pass `None` to
    /// default to OpenAI.
    ///
    /// # Errors
    ///
    /// Returns [`LiterLmError::InvalidHeader`] if any entry in
    /// `config.extra_headers` is not a valid HTTP header name or value.
    /// Returns a wrapped [`reqwest::Error`] if the underlying HTTP client
    /// cannot be constructed.
    pub fn new(config: ClientConfig, model_hint: Option<&str>) -> Result<Self> {
        let provider = build_provider(&config, model_hint);

        // Validate and register extra headers on the shared reqwest client so
        // they are sent on every request without per-call overhead.
        let mut header_map = reqwest::header::HeaderMap::new();
        for (k, v) in &config.extra_headers {
            let name =
                reqwest::header::HeaderName::from_bytes(k.as_bytes()).map_err(|e| LiterLmError::InvalidHeader {
                    name: k.clone(),
                    reason: e.to_string(),
                })?;
            let val = reqwest::header::HeaderValue::from_str(v).map_err(|e| LiterLmError::InvalidHeader {
                name: k.clone(),
                reason: e.to_string(),
            })?;
            header_map.insert(name, val);
        }

        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .default_headers(header_map)
            .build()
            .map_err(LiterLmError::from)?;

        Ok(Self { config, http, provider })
    }
}

/// Resolve the provider to use for all requests on this client.
///
/// Priority:
/// 1. Explicit `base_url` in config → custom OpenAI-compatible provider.
/// 2. `model_hint` → auto-detect by model name prefix.
/// 3. Default → OpenAI.
fn build_provider(config: &ClientConfig, model_hint: Option<&str>) -> Box<dyn Provider> {
    if let Some(ref base_url) = config.base_url {
        return Box::new(OpenAiCompatibleProvider {
            name: "custom".into(),
            base_url: base_url.clone(),
            env_var: String::new(),
            model_prefixes: vec![],
        });
    }

    if let Some(model) = model_hint
        && let Some(p) = provider::detect_provider(model)
    {
        return p;
    }

    Box::new(OpenAiProvider)
}

impl LlmClient for DefaultClient {
    fn chat(&self, req: ChatCompletionRequest) -> BoxFuture<'_, ChatCompletionResponse> {
        Box::pin(async move {
            let url = format!("{}{}", self.provider.base_url(), self.provider.chat_completions_path());
            let (header_name, header_value) = self.provider.auth_header(self.config.api_key.expose_secret());

            // Strip the provider routing prefix from the model name before
            // putting it in the request body (e.g. "groq/llama3-70b" → "llama3-70b").
            let bare_model = self.provider.strip_model_prefix(&req.model).to_owned();

            let mut body = serde_json::to_value(&req)?;
            if let Some(obj) = body.as_object_mut() {
                obj.insert("model".into(), serde_json::Value::String(bare_model));
                // Ensure stream is false for non-streaming requests.
                obj.insert("stream".into(), serde_json::Value::Bool(false));
            }
            self.provider.transform_request(&mut body)?;

            http::request::post_json(
                &self.http,
                &url,
                &header_name,
                &header_value,
                body,
                self.config.max_retries,
            )
            .await
        })
    }

    fn chat_stream(&self, req: ChatCompletionRequest) -> BoxFuture<'_, BoxStream<'_, ChatCompletionChunk>> {
        Box::pin(async move {
            let url = format!("{}{}", self.provider.base_url(), self.provider.chat_completions_path());
            let (header_name, header_value) = self.provider.auth_header(self.config.api_key.expose_secret());

            let bare_model = self.provider.strip_model_prefix(&req.model).to_owned();

            let mut body = serde_json::to_value(&req)?;
            if let Some(obj) = body.as_object_mut() {
                obj.insert("model".into(), serde_json::Value::String(bare_model));
                // Force stream = true.
                obj.insert("stream".into(), serde_json::Value::Bool(true));
            }
            self.provider.transform_request(&mut body)?;

            let stream = http::streaming::post_stream(
                &self.http,
                &url,
                &header_name,
                &header_value,
                body,
                self.config.max_retries,
            )
            .await?;
            Ok(stream)
        })
    }

    fn embed(&self, req: EmbeddingRequest) -> BoxFuture<'_, EmbeddingResponse> {
        Box::pin(async move {
            let url = format!("{}{}", self.provider.base_url(), self.provider.embeddings_path());
            let (header_name, header_value) = self.provider.auth_header(self.config.api_key.expose_secret());

            let bare_model = self.provider.strip_model_prefix(&req.model).to_owned();

            let mut body = serde_json::to_value(&req)?;
            if let Some(obj) = body.as_object_mut() {
                obj.insert("model".into(), serde_json::Value::String(bare_model));
            }

            http::request::post_json(
                &self.http,
                &url,
                &header_name,
                &header_value,
                body,
                self.config.max_retries,
            )
            .await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, ModelsListResponse> {
        Box::pin(async move {
            // Use the stored provider — no more hardcoded "gpt-4" fallback.
            let url = format!("{}{}", self.provider.base_url(), self.provider.models_path());
            let (header_name, header_value) = self.provider.auth_header(self.config.api_key.expose_secret());

            http::request::get_json(&self.http, &url, &header_name, &header_value, self.config.max_retries).await
        })
    }
}
