//! JSON serialization/deserialization helpers for string-based FFI bindings.
//!
//! Used by bindings that pass JSON strings across the FFI boundary
//! (Elixir NIF, C FFI, PHP).

/// Serialize a value to a JSON string.
pub fn to_json<T: serde::Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string(value).map_err(|e| format!("JSON serialization failed: {e}"))
}

/// Deserialize a JSON string into a typed value.
///
/// `label` is used in error messages to identify what was being parsed.
pub fn from_json<T: serde::de::DeserializeOwned>(json: &str, label: &str) -> Result<T, String> {
    serde_json::from_str(json).map_err(|e| format!("invalid {label} JSON: {e}"))
}
