use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

use crate::error::Result;

// Embed the generated providers registry at compile time.
// Path: crates/liter-lm/src/provider/mod.rs → ../../../../schemas/providers.json
const PROVIDERS_JSON: &str = include_str!("../../../../schemas/providers.json");

/// Lazy-initialised registry parsed from the embedded JSON.
static REGISTRY: LazyLock<ProviderRegistry> =
    LazyLock::new(|| serde_json::from_str(PROVIDERS_JSON).expect("embedded schemas/providers.json is valid JSON"));

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
    fn auth_header(&self, api_key: &str) -> (String, String);

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

    fn auth_header(&self, api_key: &str) -> (String, String) {
        ("Authorization".into(), format!("Bearer {api_key}"))
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

    fn auth_header(&self, api_key: &str) -> (String, String) {
        ("Authorization".into(), format!("Bearer {api_key}"))
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
pub struct ConfigDrivenProvider {
    config: ProviderConfig,
    // Resolved base_url — falls back to empty string when none is configured.
    resolved_base_url: String,
}

impl ConfigDrivenProvider {
    fn new(config: ProviderConfig) -> Self {
        let resolved_base_url = config.base_url.clone().unwrap_or_default();
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
        &self.resolved_base_url
    }

    fn auth_header(&self, api_key: &str) -> (String, String) {
        // All providers in providers.json currently use bearer auth or no auth.
        // When auth_type is "none" or missing, still send the key if provided.
        ("Authorization".into(), format!("Bearer {api_key}"))
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
pub fn detect_provider(model: &str) -> Option<Box<dyn Provider>> {
    // 1. OpenAI hardcoded patterns.
    let openai = OpenAiProvider;
    if openai.matches_model(model) {
        return Some(Box::new(openai));
    }

    // 2. Slash-prefix routing (e.g. "groq/llama3-70b").
    if let Some((prefix, _)) = model.split_once('/')
        && let Some(cfg) = registry_find(prefix)
        && cfg.base_url.is_some()
    {
        // Only use the registry entry if it has a usable base_url.
        return Some(Box::new(ConfigDrivenProvider::new(cfg.clone())));
    }

    // 3. Walk registry model_prefixes for unprefixed model names.
    for cfg in &REGISTRY.providers {
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

/// Look up a provider config by exact name in the registry.
fn registry_find(name: &str) -> Option<&'static ProviderConfig> {
    REGISTRY.providers.iter().find(|p| p.name == name)
}

/// Return all provider configs from the registry.
///
/// Useful for tooling, documentation generation, or runtime enumeration.
pub fn all_providers() -> &'static [ProviderConfig] {
    &REGISTRY.providers
}

/// Return the list of complex provider names.
///
/// Complex providers require custom auth/routing logic beyond simple bearer
/// tokens (e.g. AWS Bedrock SigV4, Vertex AI OAuth2).
pub fn complex_provider_names() -> &'static [String] {
    &REGISTRY.complex_providers
}
