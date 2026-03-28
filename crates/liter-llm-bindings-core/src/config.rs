//! Shared config parsing and client construction for binding crates.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use liter_llm::tower::LlmHook;
use liter_llm::tower::{BudgetConfig, CacheBackend, CacheConfig, Enforcement, RateLimitConfig};
use liter_llm::{AuthHeaderFormat, ClientConfigBuilder, CustomProviderConfig, ManagedClient};

/// Parse a `CacheConfig` from a JSON value.
///
/// Expected shape: `{ "max_entries": 256, "ttl_seconds": 300 }`
pub fn parse_cache_config(val: &serde_json::Value) -> Result<CacheConfig, String> {
    let max_entries = val.get("max_entries").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
    let ttl_seconds = val.get("ttl_seconds").and_then(|v| v.as_u64()).unwrap_or(300);

    let backend = match val.get("backend").and_then(|v| v.as_str()).unwrap_or("memory") {
        "memory" => CacheBackend::Memory,
        scheme => {
            let backend_config: HashMap<String, String> = val
                .get("backend_config")
                .and_then(|v| v.as_object())
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect()
                })
                .unwrap_or_default();
            CacheBackend::OpenDal {
                scheme: scheme.to_string(),
                config: backend_config,
            }
        }
    };

    Ok(CacheConfig {
        max_entries,
        ttl: Duration::from_secs(ttl_seconds),
        backend,
    })
}

/// Parse a `BudgetConfig` from a JSON value.
///
/// Expected shape: `{ "global_limit": 10.0, "model_limits": { "gpt-4": 5.0 }, "enforcement": "hard" }`
pub fn parse_budget_config(val: &serde_json::Value) -> Result<BudgetConfig, String> {
    let global_limit = val.get("global_limit").and_then(|v| v.as_f64());
    let model_limits: HashMap<String, f64> = val
        .get("model_limits")
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                .collect()
        })
        .unwrap_or_default();
    let enforcement = match val.get("enforcement").and_then(|v| v.as_str()).unwrap_or("hard") {
        "soft" => Enforcement::Soft,
        _ => Enforcement::Hard,
    };
    Ok(BudgetConfig {
        global_limit,
        model_limits,
        enforcement,
    })
}

/// Parse an auth header format string.
///
/// - `"none"` → `AuthHeaderFormat::None`
/// - `"api-key:X-Custom"` → `AuthHeaderFormat::ApiKey("X-Custom")`
/// - `"bearer"` or anything else → `AuthHeaderFormat::Bearer`
pub fn parse_auth_header(s: &str) -> AuthHeaderFormat {
    match s.to_lowercase().as_str() {
        "none" => AuthHeaderFormat::None,
        s if s.starts_with("api-key:") => AuthHeaderFormat::ApiKey(s.trim_start_matches("api-key:").to_string()),
        _ => AuthHeaderFormat::Bearer,
    }
}

/// Parse a `CustomProviderConfig` from a JSON value.
pub fn parse_provider_config(val: &serde_json::Value) -> Result<CustomProviderConfig, String> {
    let name = val
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing 'name' field".to_string())?
        .to_string();
    let base_url = val
        .get("base_url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing 'base_url' field".to_string())?
        .to_string();
    let auth_header = val
        .get("auth_header")
        .and_then(|v| v.as_str())
        .map(parse_auth_header)
        .unwrap_or(AuthHeaderFormat::Bearer);
    let model_prefixes = val
        .get("model_prefixes")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    Ok(CustomProviderConfig {
        name,
        base_url,
        auth_header,
        model_prefixes,
    })
}

/// Parse a `RateLimitConfig` from a JSON value.
///
/// Expected shape: `{ "rpm": 60, "tpm": 100000, "window_seconds": 60 }`
pub fn parse_rate_limit_config(val: &serde_json::Value) -> Result<RateLimitConfig, String> {
    let rpm = val.get("rpm").and_then(|v| v.as_u64()).map(|v| v as u32);
    let tpm = val.get("tpm").and_then(|v| v.as_u64());
    let window = val
        .get("window_seconds")
        .and_then(|v| v.as_u64())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(60));
    Ok(RateLimitConfig { rpm, tpm, window })
}

/// Common client options for building a `ManagedClient`.
#[derive(Default)]
pub struct ClientOptions {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model_hint: Option<String>,
    pub timeout_secs: Option<u64>,
    pub max_retries: Option<u32>,
    pub cache_config: Option<CacheConfig>,
    pub budget_config: Option<BudgetConfig>,
    pub hooks: Vec<Arc<dyn LlmHook>>,
    pub cooldown_secs: Option<u64>,
    pub rate_limit_config: Option<RateLimitConfig>,
    pub health_check_secs: Option<u64>,
    pub enable_cost_tracking: bool,
    pub enable_tracing: bool,
}

/// Build a `ManagedClient` from common configuration parameters.
///
/// This consolidates the client construction pattern duplicated across all bindings.
pub fn build_managed_client(opts: ClientOptions) -> Result<ManagedClient, liter_llm::LiterLlmError> {
    let mut builder = ClientConfigBuilder::new(&opts.api_key);

    if let Some(url) = &opts.base_url {
        builder = builder.base_url(url);
    }
    if let Some(t) = opts.timeout_secs {
        builder = builder.timeout(Duration::from_secs(t));
    }
    if let Some(r) = opts.max_retries {
        builder = builder.max_retries(r);
    }
    if let Some(cache) = opts.cache_config {
        builder = builder.cache(cache);
    }
    if let Some(budget) = opts.budget_config {
        builder = builder.budget(budget);
    }
    if !opts.hooks.is_empty() {
        builder = builder.hooks(opts.hooks);
    }
    if let Some(secs) = opts.cooldown_secs {
        builder = builder.cooldown(Duration::from_secs(secs));
    }
    if let Some(rl) = opts.rate_limit_config {
        builder = builder.rate_limit(rl);
    }
    if let Some(secs) = opts.health_check_secs {
        builder = builder.health_check(Duration::from_secs(secs));
    }
    if opts.enable_cost_tracking {
        builder = builder.cost_tracking(true);
    }
    if opts.enable_tracing {
        builder = builder.tracing(true);
    }

    let config = builder.build();
    ManagedClient::new(config, opts.model_hint.as_deref())
}
