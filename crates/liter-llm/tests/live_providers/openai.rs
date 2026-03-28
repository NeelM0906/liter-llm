use futures_util::StreamExt;
use liter_llm::{ClientConfigBuilder, DefaultClient, LiterLlmError, LlmClient};

use super::{assert_chat_response_valid, openai_client, require_env, simple_chat_request, simple_embed_request};

#[tokio::test]
async fn chat_basic() {
    let key = require_env!("OPENAI_API_KEY");
    let client = openai_client(&key);

    let resp = client.chat(simple_chat_request("gpt-4o-mini")).await.unwrap();

    assert_chat_response_valid(&resp, "openai/chat_basic");
    let usage = resp.usage.as_ref().expect("usage should be present");
    assert!(usage.prompt_tokens > 0, "prompt_tokens should be > 0");
    assert!(usage.total_tokens > 0, "total_tokens should be > 0");
}

#[tokio::test]
async fn chat_stream() {
    let key = require_env!("OPENAI_API_KEY");
    let client = openai_client(&key);

    let mut stream = client.chat_stream(simple_chat_request("gpt-4o-mini")).await.unwrap();

    let mut content = String::new();
    let mut chunk_count = 0u32;
    let mut saw_finish = false;

    while let Some(result) = stream.next().await {
        let chunk = result.unwrap();
        chunk_count += 1;
        if let Some(choice) = chunk.choices.first() {
            if let Some(text) = &choice.delta.content {
                content.push_str(text);
            }
            if choice.finish_reason.is_some() {
                saw_finish = true;
            }
        }
        if chunk_count > 200 {
            break;
        }
    }

    assert!(chunk_count >= 1, "should receive at least 1 chunk");
    assert!(!content.is_empty(), "concatenated content should be non-empty");
    assert!(saw_finish, "should see a finish_reason in the stream");
}

#[tokio::test]
async fn embed() {
    let key = require_env!("OPENAI_API_KEY");
    let client = openai_client(&key);

    let resp = client
        .embed(simple_embed_request("text-embedding-3-small"))
        .await
        .unwrap();

    assert!(!resp.data.is_empty(), "embedding data should not be empty");
    assert!(
        !resp.data[0].embedding.is_empty(),
        "embedding vector should not be empty"
    );
    assert!(!resp.model.is_empty(), "model field should be non-empty");
}

#[tokio::test]
async fn list_models() {
    let key = require_env!("OPENAI_API_KEY");
    let client = openai_client(&key);

    let resp = client.list_models().await.unwrap();

    assert!(!resp.data.is_empty(), "models list should not be empty");
    assert!(!resp.data[0].id.is_empty(), "first model id should be non-empty");
}

#[tokio::test]
async fn error_invalid_model() {
    let key = require_env!("OPENAI_API_KEY");
    let client = openai_client(&key);

    let err = client
        .chat(simple_chat_request("nonexistent-model-xyz-999"))
        .await
        .unwrap_err();

    assert!(
        matches!(err, LiterLlmError::NotFound { .. } | LiterLlmError::BadRequest { .. }),
        "expected NotFound or BadRequest, got: {err:?}"
    );
}

#[tokio::test]
async fn error_invalid_key() {
    let _key = require_env!("OPENAI_API_KEY");

    let config = ClientConfigBuilder::new("sk-invalid-key-for-testing").build();
    let client = DefaultClient::new(config, Some("openai/gpt-4o-mini")).unwrap();

    let err = client.chat(simple_chat_request("gpt-4o-mini")).await.unwrap_err();

    assert!(
        matches!(err, LiterLlmError::Authentication { .. }),
        "expected Authentication error, got: {err:?}"
    );
}
