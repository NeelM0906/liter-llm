//! Types for web/document search requests and responses.

use serde::{Deserialize, Serialize};

/// A search request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    /// The model/provider to use (e.g. `"brave/web-search"`, `"tavily/search"`).
    pub model: String,
    /// The search query.
    pub query: String,
    /// Maximum number of results to return.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u32>,
    /// Domain filter — restrict results to specific domains.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_domain_filter: Option<Vec<String>>,
    /// Country code for localized results (ISO 3166-1 alpha-2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
}

/// A search response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// The search results.
    pub results: Vec<SearchResult>,
    /// The model used.
    pub model: String,
}

/// An individual search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Title of the result.
    pub title: String,
    /// URL of the result.
    pub url: String,
    /// Text snippet / excerpt.
    pub snippet: String,
    /// Publication or last-updated date, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
}
