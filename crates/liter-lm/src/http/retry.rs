use std::time::Duration;

/// Determine whether to retry based on status code and attempt number.
///
/// Returns `Some(delay)` if the request should be retried, `None` otherwise.
///
/// When `retry_after` is provided (parsed from the `Retry-After` response
/// header) it takes precedence over exponential backoff for 429 responses.
pub fn should_retry(status: u16, attempt: u32, max_retries: u32, retry_after: Option<Duration>) -> Option<Duration> {
    if attempt >= max_retries {
        return None;
    }

    // Only retry on rate limit (429) and server errors (500, 502, 503).
    if !matches!(status, 429 | 500 | 502 | 503) {
        return None;
    }

    // For 429, prefer the server-supplied Retry-After value when present.
    if status == 429
        && let Some(server_delay) = retry_after
    {
        // Cap the server-supplied delay to 60 seconds to avoid stalling forever.
        return Some(server_delay.min(Duration::from_secs(60)));
    }

    // Exponential backoff: 1s, 2s, 4s, 8s … capped at 30 s.
    // Use checked_shl to avoid overflow on large attempt counts.
    let base_delay = Duration::from_secs(1u64.checked_shl(attempt).unwrap_or(u64::MAX));
    Some(base_delay.min(Duration::from_secs(30)))
}

/// Parse the value of a `Retry-After` header into a `Duration`.
///
/// The header may be:
/// - A non-negative integer (number of seconds to wait), or
/// - An HTTP-date (not yet supported; returns `None`).
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let secs: u64 = value.trim().parse().ok()?;
    Some(Duration::from_secs(secs))
}
