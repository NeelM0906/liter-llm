//! OpenDAL-backed cache store for the response cache.
//!
//! Implements [`CacheStore`] using an [`opendal::Operator`] for persistence.
//! Supports any OpenDAL backend (S3, Redis, GCS, local filesystem, etc.).

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use opendal::Operator;
use serde::{Deserialize, Serialize};

use super::cache::{CacheStore, CachedResponse};

/// A cached entry stored via OpenDAL, including metadata for TTL and
/// collision detection.
#[derive(Serialize, Deserialize)]
struct StoredEntry {
    request_body: String,
    response: CachedResponse,
    /// Unix timestamp (seconds) when this entry expires.
    expires_at: u64,
}

/// Cache store backed by an [`opendal::Operator`].
///
/// Entries are stored as JSON files under `{prefix}/{key}`. TTL is embedded
/// in the stored entry and checked on read. Backend failures are non-fatal:
/// they log a warning and behave as a cache miss / no-op.
pub struct OpenDalCacheStore {
    operator: Operator,
    prefix: String,
    ttl: Duration,
}

impl OpenDalCacheStore {
    /// Create a new OpenDAL cache store.
    ///
    /// `operator` must be a fully configured OpenDAL operator.
    /// `prefix` is prepended to all cache keys (e.g. `"llm-cache/"`).
    /// `ttl` controls how long entries are valid.
    pub fn new(operator: Operator, prefix: impl Into<String>, ttl: Duration) -> Self {
        Self {
            operator,
            prefix: prefix.into(),
            ttl,
        }
    }

    /// Build an OpenDAL operator from a scheme name and config map.
    ///
    /// # Errors
    /// Returns an error if the scheme is unknown or the config is invalid.
    pub fn from_config(
        scheme: &str,
        config: HashMap<String, String>,
        prefix: impl Into<String>,
        ttl: Duration,
    ) -> Result<Self, String> {
        let parsed_scheme =
            opendal::Scheme::from_str(scheme).map_err(|e| format!("unknown OpenDAL scheme '{scheme}': {e}"))?;
        let operator = Operator::via_iter(parsed_scheme, config)
            .map_err(|e| format!("failed to build OpenDAL operator for '{scheme}': {e}"))?;
        Ok(Self::new(operator, prefix, ttl))
    }

    fn key_path(&self, key: u64) -> String {
        format!("{}{key}", self.prefix)
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl CacheStore for OpenDalCacheStore {
    fn get(&self, key: u64, request_body: &str) -> Pin<Box<dyn Future<Output = Option<CachedResponse>> + Send + '_>> {
        let path = self.key_path(key);
        let request_body = request_body.to_owned();
        Box::pin(async move {
            let bytes = match self.operator.read(&path).await {
                Ok(b) => b,
                Err(_) => return None,
            };
            let entry: StoredEntry = match serde_json::from_slice(bytes.to_bytes().as_ref()) {
                Ok(e) => e,
                Err(_) => return None,
            };
            // Check TTL
            if Self::now_secs() > entry.expires_at {
                // Lazily delete expired entry
                let _ = self.operator.delete(&path).await;
                return None;
            }
            // Verify request body matches (collision guard)
            if entry.request_body != request_body {
                return None;
            }
            Some(entry.response)
        })
    }

    fn put(
        &self,
        key: u64,
        request_body: String,
        response: CachedResponse,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let path = self.key_path(key);
        let entry = StoredEntry {
            request_body,
            response,
            expires_at: Self::now_secs() + self.ttl.as_secs(),
        };
        Box::pin(async move {
            let bytes = match serde_json::to_vec(&entry) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("OpenDAL cache: failed to serialize entry: {e}");
                    return;
                }
            };
            if let Err(e) = self.operator.write(&path, bytes).await {
                tracing::warn!("OpenDAL cache: failed to write {path}: {e}");
            }
        })
    }

    fn remove(&self, key: u64) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let path = self.key_path(key);
        Box::pin(async move {
            if let Err(e) = self.operator.delete(&path).await {
                tracing::warn!("OpenDAL cache: failed to delete {path}: {e}");
            }
        })
    }
}
