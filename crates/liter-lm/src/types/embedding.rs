use serde::{Deserialize, Serialize};

use super::common::Usage;

// ─── Object discriminators ────────────────────────────────────────────────────

/// The `object` field of an [`EmbeddingResponse`]. Always `"list"`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingListObject {
    #[default]
    #[serde(rename = "list")]
    List,
}

/// The `object` field of an [`EmbeddingObject`]. Always `"embedding"`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingObjectType {
    #[default]
    #[serde(rename = "embedding")]
    Embedding,
}

// ─── Request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

// ─── Response ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: EmbeddingListObject,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub object: EmbeddingObjectType,
    pub embedding: Vec<f64>,
    pub index: u32,
}
