use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use pin_project_lite::pin_project;

use crate::error::{LiterLmError, Result};
use crate::http::retry;
use crate::types::ChatCompletionChunk;

/// Maximum number of bytes buffered before declaring a streaming error.
const MAX_BUFFER_BYTES: usize = 1024 * 1024; // 1 MiB

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Send a streaming POST request and return an SSE stream of
/// `ChatCompletionChunk`s.
///
/// Before opening the stream, retries on 429 / 503 up to `max_retries` times
/// honouring any `Retry-After` header.  Once the stream is open, individual
/// chunk errors are yielded as `Err` items rather than causing a retry.
pub async fn post_stream(
    client: &reqwest::Client,
    url: &str,
    auth_header_name: &str,
    auth_header_value: &str,
    body: serde_json::Value,
    max_retries: u32,
) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
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
            let byte_stream = resp.bytes_stream();
            let stream = SseParser::new(byte_stream);
            return Ok(Box::pin(stream));
        }

        // Parse Retry-After before consuming the body.
        let server_retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(retry::parse_retry_after);

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

// ---------------------------------------------------------------------------
// SSE parser
// ---------------------------------------------------------------------------

pin_project! {
    /// Wraps a `bytes::Bytes` stream and yields parsed `ChatCompletionChunk`s.
    struct SseParser<S> {
        #[pin]
        inner: S,
        buffer: String,
        // Set to true once the inner stream is exhausted.
        done: bool,
    }
}

impl<S> SseParser<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
            done: false,
        }
    }
}

impl<S> Stream for SseParser<S>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send,
{
    type Item = Result<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            // --- Process any complete lines already in the buffer ---
            if let Some(newline_pos) = this.buffer.find('\n') {
                // Extract the line in-place using drain to avoid a clone.
                // We include the '\n' in the drain so the buffer advances past
                // it; we then trim the extracted slice.
                let mut line_bytes: String = this.buffer.drain(..=newline_pos).collect();

                // Remove trailing '\r' (CRLF) and leading/trailing whitespace.
                if line_bytes.ends_with('\n') {
                    line_bytes.pop();
                }
                if line_bytes.ends_with('\r') {
                    line_bytes.pop();
                }
                let line = line_bytes.trim();

                // Skip empty lines and SSE comments.
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(raw) = line.strip_prefix("data:") {
                    // Strip exactly one optional leading space (RFC 8895 §3.3).
                    let data = raw.strip_prefix(' ').unwrap_or(raw).trim();
                    if data == "[DONE]" {
                        return Poll::Ready(None);
                    }
                    return Poll::Ready(Some(serde_json::from_str::<ChatCompletionChunk>(data).map_err(|e| {
                        LiterLmError::Streaming {
                            message: format!("failed to parse SSE data: {e}"),
                        }
                    })));
                }

                // Ignore other SSE fields (event:, id:, retry:).
                continue;
            }

            // --- Buffer is empty or has only a partial line; fetch more bytes ---

            if *this.done {
                // Drain any remaining buffered text without a newline.
                // Real SSE streams always end cleanly, so just signal end.
                return Poll::Ready(None);
            }

            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    // Guard against unbounded growth.
                    if this.buffer.len() + bytes.len() > MAX_BUFFER_BYTES {
                        return Poll::Ready(Some(Err(LiterLmError::Streaming {
                            message: format!("SSE buffer exceeded {MAX_BUFFER_BYTES} bytes; stream aborted"),
                        })));
                    }
                    match std::str::from_utf8(&bytes) {
                        Ok(s) => this.buffer.push_str(s),
                        Err(e) => {
                            return Poll::Ready(Some(Err(LiterLmError::Streaming {
                                message: format!("invalid UTF-8 in SSE stream: {e}"),
                            })));
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(LiterLmError::from(e))));
                }
                Poll::Ready(None) => {
                    *this.done = true;
                    // Loop once more to flush any remaining buffered line.
                    continue;
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Parse a single SSE `data:` line into a `ChatCompletionChunk`.
///
/// Returns `None` for the terminal `[DONE]` sentinel.
pub fn parse_sse_line(line: &str) -> Option<Result<ChatCompletionChunk>> {
    // Strip "data:" then optionally one leading space (RFC 8895 §3.3).
    let raw = line.strip_prefix("data:")?;
    let data = raw.strip_prefix(' ').unwrap_or(raw).trim();
    if data == "[DONE]" {
        return None;
    }
    Some(serde_json::from_str(data).map_err(|e| LiterLmError::Streaming {
        message: format!("failed to parse SSE data: {e}"),
    }))
}
