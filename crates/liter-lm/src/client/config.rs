use std::collections::HashMap;
use std::time::Duration;

use secrecy::SecretString;

use crate::error::{LiterLmError, Result};

/// Configuration for an LLM client.
///
/// `api_key` is stored as a [`SecretString`] so it is zeroed on drop and never
/// printed accidentally.  Access it via [`secrecy::ExposeSecret`].
#[derive(Clone)]
pub struct ClientConfig {
    /// API key for authentication (stored as a secret).
    pub api_key: SecretString,
    /// Override base URL.  When set, all requests go here regardless of model
    /// name, and provider auto-detection is skipped.
    pub base_url: Option<String>,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum number of retries on 429 / 5xx responses.
    pub max_retries: u32,
    /// Extra headers sent on every request.
    pub extra_headers: HashMap<String, String>,
}

impl ClientConfig {
    /// Create a config with the given API key and sensible defaults.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: None,
            timeout: Duration::from_secs(60),
            max_retries: 3,
            extra_headers: HashMap::new(),
        }
    }
}

/// Note: intentionally does *not* implement `Debug` so the secret key is never
/// accidentally logged via `{:?}`.
impl std::fmt::Debug for ClientConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientConfig")
            .field("api_key", &"[redacted]")
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("extra_headers", &self.extra_headers)
            .finish()
    }
}

/// Builder for [`ClientConfig`].
pub struct ClientConfigBuilder {
    config: ClientConfig,
}

impl ClientConfigBuilder {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            config: ClientConfig::new(api_key),
        }
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.config.base_url = Some(url.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Add a custom header sent on every request.
    ///
    /// Returns an error if either `key` or `value` is not a valid HTTP header
    /// name / value.
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let key = key.into();
        let value = value.into();

        // Validate header name.
        reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| LiterLmError::InvalidHeader {
            name: key.clone(),
            reason: e.to_string(),
        })?;

        // Validate header value.
        reqwest::header::HeaderValue::from_str(&value).map_err(|e| LiterLmError::InvalidHeader {
            name: key.clone(),
            reason: e.to_string(),
        })?;

        self.config.extra_headers.insert(key, value);
        Ok(self)
    }

    pub fn build(self) -> ClientConfig {
        self.config
    }
}
