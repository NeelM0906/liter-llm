use serde::de::DeserializeOwned;

use crate::error::{LiterLmError, Result};
use crate::http::retry;

// ---------------------------------------------------------------------------
// Shared retry loop helpers
// ---------------------------------------------------------------------------

/// Extract an optional `Retry-After` delay from a response.
fn retry_after_from_response(resp: &reqwest::Response) -> Option<std::time::Duration> {
    let value = resp.headers().get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    retry::parse_retry_after(value)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Send a POST request with a JSON body and deserialize the JSON response.
///
/// Retries on 429 / 5xx according to `max_retries`.  When the server sends a
/// `Retry-After` header on a 429 response the supplied delay is respected
/// instead of the default exponential backoff.
pub async fn post_json<T: DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    auth_header_name: &str,
    auth_header_value: &str,
    body: serde_json::Value,
    max_retries: u32,
) -> Result<T> {
    let mut attempt = 0u32;

    loop {
        let resp = client
            .post(url)
            .header(auth_header_name, auth_header_value)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status().as_u16();

        if resp.status().is_success() {
            // Use reqwest's built-in JSON deserializer — avoids buffering
            // the entire body as a String before parsing.
            return resp.json::<T>().await.map_err(LiterLmError::from);
        }

        // Check Retry-After before consuming the body.
        let server_retry_after = retry_after_from_response(&resp);

        if let Some(delay) = retry::should_retry(status, attempt, max_retries, server_retry_after) {
            attempt += 1;
            tokio::time::sleep(delay).await;
            continue;
        }

        // Non-retryable error — read the body for a useful error message.
        let text = resp
            .text()
            .await
            .unwrap_or_else(|e| format!("(failed to read body: {e})"));
        return Err(LiterLmError::from_status(status, &text));
    }
}

/// Send a GET request and deserialize the JSON response.
///
/// Retries on 429 / 5xx according to `max_retries`, honouring any
/// `Retry-After` header from the server.
pub async fn get_json<T: DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    auth_header_name: &str,
    auth_header_value: &str,
    max_retries: u32,
) -> Result<T> {
    let mut attempt = 0u32;

    loop {
        let resp = client
            .get(url)
            .header(auth_header_name, auth_header_value)
            .send()
            .await?;

        let status = resp.status().as_u16();

        if resp.status().is_success() {
            return resp.json::<T>().await.map_err(LiterLmError::from);
        }

        let server_retry_after = retry_after_from_response(&resp);

        if let Some(delay) = retry::should_retry(status, attempt, max_retries, server_retry_after) {
            attempt += 1;
            tokio::time::sleep(delay).await;
            continue;
        }

        let text = resp
            .text()
            .await
            .unwrap_or_else(|e| format!("(failed to read body: {e})"));
        return Err(LiterLmError::from_status(status, &text));
    }
}
