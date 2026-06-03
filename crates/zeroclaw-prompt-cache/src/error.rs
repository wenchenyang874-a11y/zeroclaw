use thiserror::Error;

#[derive(Debug, Error)]
pub enum PromptCacheError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("schema version mismatch: db={db}, expected={expected}")]
    SchemaVersionMismatch { db: String, expected: String },

    #[error("embedding model mismatch: db={db}, runtime={runtime}")]
    EmbeddingModelMismatch { db: String, runtime: String },

    #[error("vector dim mismatch: rule_id={rule_id} stored={stored} runtime={runtime}")]
    VectorDimMismatch { rule_id: i64, stored: usize, runtime: usize },

    #[error("invalid vector blob: rule_id={rule_id} (expected {expected_bytes} bytes, got {got_bytes})")]
    InvalidVectorBlob { rule_id: i64, expected_bytes: usize, got_bytes: usize },

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("embedder: {0}")]
    Embedder(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
