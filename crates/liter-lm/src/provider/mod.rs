use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

use crate::error::{LiterLmError, Result};

// Embed the generated providers registry at compile time.
// Path: crates/liter-lm/src/provider/mod.rs → ../../../../schemas/providers.json
const PROVIDERS_JSON: &str = include_str!("../../../../schemas/providers.json");

/// Lazy-initialised registry parsed from the embedded JSON.
/// Stores a `Result` so that parse failures surface at call time rather than
/// panicking the process (fix for the `.expect()` on LazyLock).
static REGISTRY: LazyLock<std::result::Result<ProviderRegistry, String>> =
    LazyLock::new(|| serde_json::from_str(PROVIDERS_JSON).map_err(|e| e.to_string()));

/// Access the registry, returning an error if the embedded JSON was invalid.
fn registry() -> Result<&'static ProviderRegistry> {
    REGISTRY.as_ref().map_err(|e| LiterLmError::ServerError {
        message: format!("embedded schemas/providers.json is invalid: {e}"),
    })
}

// ── Registry types (deserialised from providers.json) ────────────────────────

#[derive(Debug, Deserialize)]
struct ProviderRegistry {
    providers: Vec<ProviderConfig>,
    #[serde(default)]
    complex_providers: Vec<String>,
}

/// Static configuration for a single provider entry in providers.json.
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub display_name: Option<String>,
    pub base_url: Option<String>,
    pub auth: Option<AuthConfig>,
    pub endpoints: Option<Vec<String>>,
    pub model_prefixes: Option<Vec<String>>,
    pub param_mappings: Option<HashMap<String, String>>,
}

/// Auth configuration block.
#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    #[serde(rename = "type")]
    pub auth_type: String,
    pub env_var: Option<String>,
}

// ── Provider trait ───────────────────────────────────────────────────────────

/// A provider defines how to reach an LLM API endpoint.
pub trait Provider: Send + Sync {
    /// Provider name (e.g., "openai").
    fn name(&self) -> &str;

    /// Base URL (e.g., "https://api.openai.com/v1").
    fn base_url(&self) -> &str;

    /// Build the authorization header as (header-name, header-value).
    ///
    /// Returns a static header name and a borrowed-or-owned value to avoid
    /// allocating the header name string on every request.
    fn auth_header<'a>(&'a self, api_key: &'a str) -> (Cow<'static, str>, Cow<'a, str>);

    /// Whether this provider matches a given model string.
    fn matches_model(&self, model: &str) -> bool;

    /// Strip any provider-routing prefix from a model name before sending it
    /// in the request body.
    ///
    /// E.g. `"groq/llama3-70b"` → `"llama3-70b"`.
    /// Returns the model name unchanged when no prefix is present.
    fn strip_model_prefix<'m>(&self, model: &'m str) -> &'m str {
        // Try "name/" prefix without allocating.
        if let Some(rest) = model.strip_prefix(self.name())
            && let Some(stripped) = rest.strip_prefix('/')
        {
            return stripped;
        }
        model
    }

    /// Path for chat completions endpoint.
    fn chat_completions_path(&self) -> &str {
        "/chat/completions"
    }

    /// Path for embeddings endpoint.
    fn embeddings_path(&self) -> &str {
        "/embeddings"
    }

    /// Path for list models endpoint.
    fn models_path(&self) -> &str {
        "/models"
    }

    /// Whether streaming is supported.
    fn supports_streaming(&self) -> bool {
        true
    }

    /// Transform the request body before sending, if needed.
    fn transform_request(&self, body: &mut serde_json::Value) -> Result<()> {
        let _ = body;
        Ok(())
    }
}

// ── Built-in providers ───────────────────────────────────────────────────────

/// Built-in OpenAI provider.
pub struct OpenAiProvider;

impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn base_url(&self) -> &str {
        "https://api.openai.com/v1"
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> (Cow<'static, str>, Cow<'a, str>) {
        (Cow::Borrowed("Authorization"), Cow::Owned(format!("Bearer {api_key}")))
    }

    fn matches_model(&self, model: &str) -> bool {
        model.starts_with("gpt-")
            || model.starts_with("o1-")
            || model.starts_with("o3-")
            || model.starts_with("o4-")
            || model.starts_with("dall-e-")
            || model.starts_with("whisper-")
            || model.starts_with("tts-")
            || model.starts_with("text-embedding-")
            || model.starts_with("chatgpt-")
    }

    fn strip_model_prefix<'m>(&self, model: &'m str) -> &'m str {
        // OpenAI models have no routing prefix.
        model
    }
}

/// A generic OpenAI-compatible provider (configurable base_url + bearer auth).
pub struct OpenAiCompatibleProvider {
    pub name: String,
    pub base_url: String,
    pub env_var: String,
    pub model_prefixes: Vec<String>,
}

impl Provider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> (Cow<'static, str>, Cow<'a, str>) {
        (Cow::Borrowed("Authorization"), Cow::Owned(format!("Bearer {api_key}")))
    }

    fn matches_model(&self, model: &str) -> bool {
        self.model_prefixes
            .iter()
            .any(|prefix| model.starts_with(prefix.as_str()))
    }
}

/// A data-driven provider backed by a [`ProviderConfig`] entry from providers.json.
///
/// Used for simple providers that are fully described by their JSON config.
/// Complex providers (AWS Bedrock, Vertex AI, etc.) use dedicated implementations.
///
/// # Construction
///
/// Construct only via [`ConfigDrivenProvider::new`], which is intentionally
/// `pub(crate)` — callers outside this crate must go through [`detect_provider`].
///
/// # `base_url` contract
///
/// [`Provider::base_url`] returns an empty string when the provider config has
/// no `base_url` entry.  This is safe because [`detect_provider`] guards the
/// `base_url.is_some()` condition before constructing a `ConfigDrivenProvider`,
/// so a correctly-routed request will never produce an empty URL.  A manually
/// constructed instance (hypothetically) would produce a clearly-broken URL
/// (`/chat/completions`) that fails immediately at the HTTP layer.
pub struct ConfigDrivenProvider {
    config: ProviderConfig,
    // Resolved base_url — `None` when not configured; request will fail at
    // send time with a clear error rather than silently sending to an empty URL.
    resolved_base_url: Option<String>,
}

impl ConfigDrivenProvider {
    #[must_use]
    pub(crate) fn new(config: ProviderConfig) -> Self {
        let resolved_base_url = config.base_url.clone();
        Self {
            config,
            resolved_base_url,
        }
    }
}

impl Provider for ConfigDrivenProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn base_url(&self) -> &str {
        // Return an empty string when unconfigured; `transform_request` or the
        // HTTP layer will surface a useful error before any network call goes out.
        self.resolved_base_url.as_deref().unwrap_or("")
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> (Cow<'static, str>, Cow<'a, str>) {
        let auth_type = self
            .config
            .auth
            .as_ref()
            .map(|a| a.auth_type.as_str())
            .unwrap_or("bearer");

        match auth_type {
            "none" => {
                // No auth header required; return empty values that the HTTP
                // layer will ignore when the key is also empty.
                (Cow::Borrowed(""), Cow::Borrowed(""))
            }
            "api-key" | "header" | "x-api-key" => (Cow::Borrowed("x-api-key"), Cow::Borrowed(api_key)),
            // "bearer" and anything else defaults to Bearer token.
            _ => (Cow::Borrowed("Authorization"), Cow::Owned(format!("Bearer {api_key}"))),
        }
    }

    fn matches_model(&self, model: &str) -> bool {
        if let Some(prefixes) = &self.config.model_prefixes {
            prefixes.iter().any(|p| model.starts_with(p.as_str()))
        } else {
            false
        }
    }
}

// ── Provider detection ───────────────────────────────────────────────────────

/// Detect which provider to use based on model name.
///
/// Strategy:
/// 1. OpenAI hardcoded patterns (gpt-*, o1-*, text-embedding-*, …).
/// 2. `"provider/"` prefix — look up the prefix in the registry.
/// 3. Walk all registry entries and check their `model_prefixes`.
///
/// Returns `None` when no built-in provider matches.  The caller should fall
/// back to a config-specified `base_url` or default to [`OpenAiProvider`].
///
/// Complex providers (those listed in `complex_providers` in providers.json)
/// are excluded from config-driven routing because they require custom
/// auth/request logic beyond simple bearer tokens.
pub fn detect_provider(model: &str) -> Option<Box<dyn Provider>> {
    // 1. OpenAI hardcoded patterns.
    let openai = OpenAiProvider;
    if openai.matches_model(model) {
        return Some(Box::new(openai));
    }

    // Grab the registry; if it failed to parse we cannot route.
    let reg = match REGISTRY.as_ref() {
        Ok(r) => r,
        Err(_) => return None,
    };

    // 2. Slash-prefix routing (e.g. "groq/llama3-70b").
    if let Some((prefix, _)) = model.split_once('/')
        && let Some(cfg) = reg.providers.iter().find(|p| p.name == prefix)
        && cfg.base_url.is_some()
        && !reg.complex_providers.contains(&cfg.name)
    {
        // Only use the registry entry if it has a usable base_url and is not
        // a complex provider requiring dedicated auth logic.
        return Some(Box::new(ConfigDrivenProvider::new(cfg.clone())));
    }

    // 3. Walk registry model_prefixes for unprefixed model names.
    for cfg in &reg.providers {
        if reg.complex_providers.contains(&cfg.name) {
            continue;
        }
        if let Some(prefixes) = &cfg.model_prefixes {
            let matches = prefixes
                .iter()
                .any(|p| model.starts_with(p.as_str()) && !p.ends_with('/'));
            if matches && cfg.base_url.is_some() {
                return Some(Box::new(ConfigDrivenProvider::new(cfg.clone())));
            }
        }
    }

    None
}

/// Return all provider configs from the registry.
///
/// Useful for tooling, documentation generation, or runtime enumeration.
pub fn all_providers() -> Result<&'static [ProviderConfig]> {
    Ok(&registry()?.providers)
}

/// Return the list of complex provider names.
///
/// Complex providers require custom auth/routing logic beyond simple bearer
/// tokens (e.g. AWS Bedrock SigV4, Vertex AI OAuth2).
pub fn complex_provider_names() -> Result<&'static [String]> {
    Ok(&registry()?.complex_providers)
}
