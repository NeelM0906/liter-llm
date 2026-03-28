//! Shared error formatting for binding crates.

use liter_llm::LiterLlmError;

/// Return a short, stable label for each error variant.
///
/// Used to prefix error messages with `[Label]` so callers can programmatically
/// inspect error types even when the binding only exposes string messages.
pub fn error_kind_label(e: &LiterLlmError) -> &'static str {
    match e {
        LiterLlmError::Authentication { .. } => "Authentication",
        LiterLlmError::RateLimited { .. } => "RateLimited",
        LiterLlmError::BadRequest { .. } => "BadRequest",
        LiterLlmError::ContextWindowExceeded { .. } => "ContextWindowExceeded",
        LiterLlmError::ContentPolicy { .. } => "ContentPolicy",
        LiterLlmError::NotFound { .. } => "NotFound",
        LiterLlmError::ServerError { .. } => "ServerError",
        LiterLlmError::ServiceUnavailable { .. } => "ServiceUnavailable",
        LiterLlmError::Timeout => "Timeout",
        LiterLlmError::Network(_) => "Network",
        LiterLlmError::Streaming { .. } => "Streaming",
        LiterLlmError::EndpointNotSupported { .. } => "EndpointNotSupported",
        LiterLlmError::InvalidHeader { .. } => "InvalidHeader",
        LiterLlmError::Serialization(_) => "Serialization",
        LiterLlmError::BudgetExceeded { .. } => "BudgetExceeded",
        LiterLlmError::HookRejected { .. } => "HookRejected",
        _ => "Unknown",
    }
}

/// Format an error with a `[Label] message` prefix.
pub fn format_error(e: &LiterLlmError) -> String {
    format!("[{}] {}", error_kind_label(e), e)
}
