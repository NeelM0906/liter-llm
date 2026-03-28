//! Bidirectional snake_case ↔ camelCase conversion for JSON keys.
//!
//! Used by Node.js, WASM, and any other binding that needs to bridge
//! between Rust's snake_case types and JavaScript's camelCase convention.

/// Convert a `snake_case` identifier to `camelCase`.
///
/// - Leading underscores are preserved: `__foo` → `__foo`
/// - Consecutive underscores collapse: `foo__bar` → `fooBar`
/// - Trailing underscores are preserved: `__init__` → `__init__`
pub fn snake_to_camel(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    // Preserve leading underscores verbatim.
    while chars.peek() == Some(&'_') {
        result.push('_');
        chars.next();
    }

    let mut pending_underscores: usize = 0;
    for ch in chars {
        if ch == '_' {
            pending_underscores += 1;
        } else if pending_underscores > 0 {
            result.extend(ch.to_uppercase());
            pending_underscores = 0;
        } else {
            result.push(ch);
        }
    }

    // Preserve trailing underscores.
    for _ in 0..pending_underscores {
        result.push('_');
    }
    result
}

/// Convert a `camelCase` identifier to `snake_case`.
///
/// - Leading underscores are preserved: `_foo` → `_foo`
/// - Uppercase letters are lowered and prefixed with `_`: `fooBar` → `foo_bar`
/// - Consecutive uppercase (acronyms) handled: `parseJSON` → `parse_json`
pub fn camel_to_snake(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 4);
    let mut chars = s.chars().peekable();

    // Preserve leading underscores.
    while chars.peek() == Some(&'_') {
        result.push('_');
        chars.next();
    }

    let mut prev_upper = false;
    for ch in chars {
        if ch.is_uppercase() {
            if !result.is_empty() && !result.ends_with('_') && !prev_upper {
                result.push('_');
            }
            result.extend(ch.to_lowercase());
            prev_upper = true;
        } else {
            // Handle acronym boundaries: "parseJSON" → before a lowercase char
            // after consecutive uppercase, insert underscore before the last upper.
            if prev_upper && !result.is_empty() && ch.is_lowercase() {
                let last = result.pop().unwrap();
                if !result.is_empty() && !result.ends_with('_') {
                    result.push('_');
                }
                result.push(last);
            }
            result.push(ch);
            prev_upper = false;
        }
    }
    result
}

/// Recursively convert all object keys in a JSON value from `snake_case` to `camelCase`.
///
/// Leaves string values unchanged (including `tool_calls[].function.arguments`
/// which is a JSON-encoded string, not a nested object).
pub fn to_camel_case_keys(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let converted = map
                .into_iter()
                .map(|(k, v)| (snake_to_camel(&k), to_camel_case_keys(v)))
                .collect();
            serde_json::Value::Object(converted)
        }
        serde_json::Value::Array(arr) => serde_json::Value::Array(arr.into_iter().map(to_camel_case_keys).collect()),
        other => other,
    }
}

/// Recursively convert all object keys in a JSON value from `camelCase` to `snake_case`.
///
/// This allows JS callers to pass either `{ maxTokens: 100 }` or `{ max_tokens: 100 }`.
pub fn to_snake_case_keys(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let converted = map
                .into_iter()
                .map(|(k, v)| (camel_to_snake(&k), to_snake_case_keys(v)))
                .collect();
            serde_json::Value::Object(converted)
        }
        serde_json::Value::Array(arr) => serde_json::Value::Array(arr.into_iter().map(to_snake_case_keys).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── snake_to_camel ──────────────────────────────────────────────────

    #[test]
    fn snake_to_camel_basic() {
        assert_eq!(snake_to_camel("foo_bar"), "fooBar");
        assert_eq!(snake_to_camel("foo_bar_baz"), "fooBarBaz");
    }

    #[test]
    fn snake_to_camel_no_underscores() {
        assert_eq!(snake_to_camel("foobar"), "foobar");
    }

    #[test]
    fn snake_to_camel_leading_underscore_preserved() {
        assert_eq!(snake_to_camel("_foo"), "_foo");
        assert_eq!(snake_to_camel("__foo"), "__foo");
        assert_eq!(snake_to_camel("__init__"), "__init__");
    }

    #[test]
    fn snake_to_camel_consecutive_underscores_collapse() {
        assert_eq!(snake_to_camel("foo__bar"), "fooBar");
    }

    #[test]
    fn snake_to_camel_empty() {
        assert_eq!(snake_to_camel(""), "");
    }

    #[test]
    fn snake_to_camel_openai_fields() {
        assert_eq!(snake_to_camel("prompt_tokens"), "promptTokens");
        assert_eq!(snake_to_camel("completion_tokens"), "completionTokens");
        assert_eq!(snake_to_camel("finish_reason"), "finishReason");
        assert_eq!(snake_to_camel("tool_calls"), "toolCalls");
        assert_eq!(snake_to_camel("tool_call_id"), "toolCallId");
        assert_eq!(snake_to_camel("system_fingerprint"), "systemFingerprint");
        assert_eq!(snake_to_camel("max_tokens"), "maxTokens");
        assert_eq!(snake_to_camel("top_p"), "topP");
    }

    // ─── camel_to_snake ──────────────────────────────────────────────────

    #[test]
    fn camel_to_snake_basic() {
        assert_eq!(camel_to_snake("fooBar"), "foo_bar");
        assert_eq!(camel_to_snake("fooBarBaz"), "foo_bar_baz");
    }

    #[test]
    fn camel_to_snake_no_uppercase() {
        assert_eq!(camel_to_snake("foobar"), "foobar");
    }

    #[test]
    fn camel_to_snake_leading_underscore_preserved() {
        assert_eq!(camel_to_snake("_foo"), "_foo");
    }

    #[test]
    fn camel_to_snake_empty() {
        assert_eq!(camel_to_snake(""), "");
    }

    #[test]
    fn camel_to_snake_openai_fields() {
        assert_eq!(camel_to_snake("promptTokens"), "prompt_tokens");
        assert_eq!(camel_to_snake("completionTokens"), "completion_tokens");
        assert_eq!(camel_to_snake("finishReason"), "finish_reason");
        assert_eq!(camel_to_snake("toolCalls"), "tool_calls");
        assert_eq!(camel_to_snake("toolCallId"), "tool_call_id");
        assert_eq!(camel_to_snake("maxTokens"), "max_tokens");
        assert_eq!(camel_to_snake("topP"), "top_p");
    }

    #[test]
    fn camel_to_snake_acronym() {
        assert_eq!(camel_to_snake("parseJSON"), "parse_json");
    }

    // ─── round-trip ──────────────────────────────────────────────────────

    #[test]
    fn round_trip_snake_through_camel() {
        let cases = ["prompt_tokens", "finish_reason", "tool_call_id", "max_tokens", "top_p"];
        for s in cases {
            assert_eq!(camel_to_snake(&snake_to_camel(s)), s, "round-trip failed for {s}");
        }
    }

    // ─── key conversion ──────────────────────────────────────────────────

    #[test]
    fn to_camel_case_keys_nested() {
        let input: serde_json::Value = serde_json::json!({
            "prompt_tokens": 10,
            "choices": [{
                "finish_reason": "stop",
                "message": { "tool_calls": [] }
            }]
        });
        let expected: serde_json::Value = serde_json::json!({
            "promptTokens": 10,
            "choices": [{
                "finishReason": "stop",
                "message": { "toolCalls": [] }
            }]
        });
        assert_eq!(to_camel_case_keys(input), expected);
    }

    #[test]
    fn to_snake_case_keys_nested() {
        let input: serde_json::Value = serde_json::json!({
            "promptTokens": 10,
            "choices": [{
                "finishReason": "stop",
                "message": { "toolCalls": [] }
            }]
        });
        let expected: serde_json::Value = serde_json::json!({
            "prompt_tokens": 10,
            "choices": [{
                "finish_reason": "stop",
                "message": { "tool_calls": [] }
            }]
        });
        assert_eq!(to_snake_case_keys(input), expected);
    }
}
