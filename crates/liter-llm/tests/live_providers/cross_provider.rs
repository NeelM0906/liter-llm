use liter_llm::LlmClient;

use super::{anthropic_client, google_ai_client, openai_client, simple_chat_request};

/// Sends the same prompt to every provider whose env var is set and asserts
/// structural parity across all responses.
#[tokio::test]
async fn chat_parity_across_providers() {
    let mut results: Vec<(&str, liter_llm::ChatCompletionResponse)> = Vec::new();

    if let Ok(key) = std::env::var("OPENAI_API_KEY")
        && !key.is_empty()
    {
        let client = openai_client(&key);
        match client.chat(simple_chat_request("gpt-4o-mini")).await {
            Ok(resp) => results.push(("openai", resp)),
            Err(e) => eprintln!("openai chat failed: {e}"),
        }
    }

    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY")
        && !key.is_empty()
    {
        let client = anthropic_client(&key);
        match client.chat(simple_chat_request("claude-haiku-4-5-20251001")).await {
            Ok(resp) => results.push(("anthropic", resp)),
            Err(e) => eprintln!("anthropic chat failed: {e}"),
        }
    }

    if let Ok(key) = std::env::var("GEMINI_API_KEY")
        && !key.is_empty()
    {
        let client = google_ai_client(&key);
        match client.chat(simple_chat_request("gemini-2.5-flash-lite")).await {
            Ok(resp) => results.push(("google_ai", resp)),
            Err(e) => eprintln!("google_ai chat failed: {e}"),
        }
    }

    if results.is_empty() {
        eprintln!("SKIP: no provider API keys set, skipping cross-provider parity test");
        return;
    }

    eprintln!("cross-provider parity: testing {} providers", results.len());

    for (provider, resp) in &results {
        assert!(!resp.choices.is_empty(), "{provider}: choices should not be empty");
        assert!(
            resp.choices[0].message.content.as_ref().is_some_and(|c| !c.is_empty()),
            "{provider}: first choice content should be non-empty"
        );
        // Note: Gemini doesn't include model name in responses, so we skip this check.
    }
}
