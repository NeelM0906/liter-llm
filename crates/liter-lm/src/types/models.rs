use serde::{Deserialize, Serialize};

// ─── Object discriminators ────────────────────────────────────────────────────

/// The `object` field of a [`ModelsListResponse`]. Always `"list"`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelsListObject {
    #[default]
    #[serde(rename = "list")]
    List,
}

/// The `object` field of a [`ModelObject`]. Always `"model"`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelObjectType {
    #[default]
    #[serde(rename = "model")]
    Model,
}

// ─── Responses ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsListResponse {
    pub object: ModelsListObject,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: ModelObjectType,
    pub created: u64,
    pub owned_by: String,
}
