//! Schema 嵌入 + 启动期校验。
//!
//! migration SQL 直接 `include_str!` 进二进制，runtime 不依赖外部文件。

use rusqlite::{Connection, params};

use crate::error::PromptCacheError;

const SCHEMA_VERSION: &str = "2";
const INITIAL_MIGRATION: &str = include_str!("../migrations/001_initial.sql");
const LEARNING_MIGRATION: &str = include_str!("../migrations/002_learning.sql");

/// 在已打开的 connection 上幂等应用 schema 演进。
/// 001 先跑（无 IF NOT EXISTS 的部分由 SQLite 自身处理）；
/// 002 用 ALTER TABLE ADD COLUMN，列已存在会报错，所以这里读 meta 决定要不要跑。
pub fn apply(conn: &Connection) -> Result<(), PromptCacheError> {
    conn.execute_batch(INITIAL_MIGRATION)?;

    let current: String = conn
        .query_row(
            "SELECT value FROM prompt_cache_meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap_or_else(|_| "1".to_string());

    if current.as_str() < "2" {
        conn.execute_batch(LEARNING_MIGRATION)?;
    }
    Ok(())
}

/// 启动期校验 schema_version 与 embedding_model。
///
/// - schema_version 不匹配：直接报错（数据结构不兼容）。
/// - embedding_model 不匹配：报错。模型换了之后所有 rule_embeddings 失效，必须重算。
///
/// 留意 `runtime_model_id == ""` 表示调用侧暂不校验模型——目前只在 build_db 阶段成立。
pub fn validate(conn: &Connection, runtime_model_id: &str) -> Result<(), PromptCacheError> {
    let db_schema: String = conn
        .query_row(
            "SELECT value FROM prompt_cache_meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap_or_default();

    if db_schema != SCHEMA_VERSION {
        return Err(PromptCacheError::SchemaVersionMismatch {
            db: db_schema,
            expected: SCHEMA_VERSION.to_string(),
        });
    }

    if !runtime_model_id.is_empty() {
        let db_model: String = conn
            .query_row(
                "SELECT value FROM prompt_cache_meta WHERE key = 'embedding_model'",
                [],
                |r| r.get(0),
            )
            .unwrap_or_default();
        if !db_model.is_empty() && db_model != runtime_model_id {
            return Err(PromptCacheError::EmbeddingModelMismatch {
                db: db_model,
                runtime: runtime_model_id.to_string(),
            });
        }
    }
    Ok(())
}

/// 写 embedding_model meta（build_db 阶段调用）。
pub fn set_embedding_model(conn: &Connection, model_id: &str) -> Result<(), PromptCacheError> {
    conn.execute(
        "INSERT OR REPLACE INTO prompt_cache_meta(key, value) VALUES('embedding_model', ?1)",
        params![model_id],
    )?;
    Ok(())
}
