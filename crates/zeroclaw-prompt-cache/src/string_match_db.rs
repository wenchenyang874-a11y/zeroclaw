//! Fast-path 规则的 sqlite 持久层。
//!
//! 独立的 db 文件（默认 l1_string_match.db），与 prompt-cache 的 l2_vector.db 分离。
//! 设计要点：
//! 1. namespace 隔离：system + 用户自定义，按 priority 排序
//! 2. 首次启动 seed 工厂规则到 system namespace
//! 3. DB 不可达时调用方回退到源码规则集

use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;

use crate::error::PromptCacheError;

/// 一条从 DB 读出的 string_match 规则。
#[derive(Debug, Clone)]
pub struct DbRule {
    pub id: i64,
    pub namespace: String,
    pub name: String,
    pub pattern: String,
    pub tool: String,
    pub args_json: String,
    pub description: Option<String>,
    pub priority: i64,
    pub updated_at: i64,
}

/// 写入新规则用的入参。
#[derive(Debug, Clone)]
pub struct NewRule<'a> {
    pub namespace: &'a str,
    pub name: &'a str,
    pub pattern: &'a str,
    pub tool: &'a str,
    pub args_json: &'a str,
    pub description: Option<&'a str>,
    pub priority: i64,
}

/// 打开（或创建）db 文件并跑 schema migration。
pub fn open(db_path: &Path) -> Result<Connection, PromptCacheError> {
    if let Some(parent) = db_path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(db_path)?;
    init_schema(&conn)?;
    Ok(conn)
}

fn init_schema(conn: &Connection) -> Result<(), PromptCacheError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS string_match_rules (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace    TEXT NOT NULL DEFAULT 'default',
            name         TEXT NOT NULL,
            pattern      TEXT NOT NULL,
            tool         TEXT NOT NULL,
            args_json    TEXT NOT NULL,
            description  TEXT,
            priority     INTEGER NOT NULL DEFAULT 100,
            enabled      INTEGER NOT NULL DEFAULT 1,
            extra_json   TEXT,
            created_at   INTEGER NOT NULL,
            updated_at   INTEGER NOT NULL,
            UNIQUE(namespace, name)
        );
        CREATE INDEX IF NOT EXISTS idx_string_match_lookup
            ON string_match_rules(namespace, enabled, priority);",
    )?;
    Ok(())
}

/// 检查指定 namespace 是否有任何规则。
pub fn namespace_has_any(conn: &Connection, namespace: &str) -> Result<bool, PromptCacheError> {
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM string_match_rules WHERE namespace = ?",
            params![namespace],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0);
    Ok(n > 0)
}

/// 插入一条规则；UNIQUE(namespace, name) 冲突会失败。
pub fn insert_rule(conn: &Connection, r: &NewRule<'_>) -> Result<i64, PromptCacheError> {
    let now = chrono::Utc::now().timestamp();
    conn.execute(
        "INSERT INTO string_match_rules
            (namespace, name, pattern, tool, args_json, description, priority,
             enabled, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
        params![
            r.namespace,
            r.name,
            r.pattern,
            r.tool,
            r.args_json,
            r.description,
            r.priority,
            now,
            now,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// 列出所有规则（含 enabled 字段），CLI 用。
pub fn list_all_rules_with_enabled(
    conn: &Connection,
    namespace_filter: Option<&str>,
) -> Result<Vec<(DbRule, bool)>, PromptCacheError> {
    let row_to_rule = |row: &rusqlite::Row<'_>| {
        Ok((
            DbRule {
                id: row.get(0)?,
                namespace: row.get(1)?,
                name: row.get(2)?,
                pattern: row.get(3)?,
                tool: row.get(4)?,
                args_json: row.get(5)?,
                description: row.get(6)?,
                priority: row.get(7)?,
                updated_at: row.get(8)?,
            },
            row.get::<_, i64>(9)? != 0,
        ))
    };
    let mut out = Vec::new();
    if let Some(ns) = namespace_filter {
        let mut stmt = conn.prepare(
            "SELECT id, namespace, name, pattern, tool, args_json, description, priority, updated_at, enabled
             FROM string_match_rules WHERE namespace = ?
             ORDER BY namespace, priority ASC, id ASC",
        )?;
        let rows = stmt.query_map(params![ns], row_to_rule)?;
        for r in rows {
            out.push(r?);
        }
    } else {
        let mut stmt = conn.prepare(
            "SELECT id, namespace, name, pattern, tool, args_json, description, priority, updated_at, enabled
             FROM string_match_rules
             ORDER BY namespace, priority ASC, id ASC",
        )?;
        let rows = stmt.query_map([], row_to_rule)?;
        for r in rows {
            out.push(r?);
        }
    }
    Ok(out)
}

/// 按 (namespace, name) 删除一条规则。
pub fn delete_rule(conn: &Connection, namespace: &str, name: &str) -> Result<bool, PromptCacheError> {
    let n = conn.execute(
        "DELETE FROM string_match_rules WHERE namespace = ? AND name = ?",
        params![namespace, name],
    )?;
    Ok(n > 0)
}

/// 启用/禁用一条规则。
pub fn set_rule_enabled(
    conn: &Connection,
    namespace: &str,
    name: &str,
    enabled: bool,
) -> Result<bool, PromptCacheError> {
    let now = chrono::Utc::now().timestamp();
    let n = conn.execute(
        "UPDATE string_match_rules SET enabled = ?, updated_at = ?
         WHERE namespace = ? AND name = ?",
        params![if enabled { 1 } else { 0 }, now, namespace, name],
    )?;
    Ok(n > 0)
}

/// 加载 active namespace + 'system' + 'auto' 的所有 enabled 规则，按 priority 升序。
/// 同名规则用户 namespace 优先于 system；'auto' 是 PromotionWorker 晋升的运行时学习
/// 产物，对所有 namespace 可见（共享自学习成果）。
pub fn load_active_rules(conn: &Connection, namespace: &str) -> Result<Vec<DbRule>, PromptCacheError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, name, pattern, tool, args_json, description,
                priority, updated_at
         FROM string_match_rules
         WHERE enabled = 1
           AND (namespace = ? OR namespace = 'system' OR namespace = 'auto')
         ORDER BY priority ASC,
                  CASE namespace WHEN 'system' THEN 2 WHEN 'auto' THEN 1 ELSE 0 END ASC,
                  id ASC",
    )?;
    let rows = stmt.query_map(params![namespace], |row| {
        Ok(DbRule {
            id: row.get(0)?,
            namespace: row.get(1)?,
            name: row.get(2)?,
            pattern: row.get(3)?,
            tool: row.get(4)?,
            args_json: row.get(5)?,
            description: row.get(6)?,
            priority: row.get(7)?,
            updated_at: row.get(8)?,
        })
    })?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|r| seen.insert(r.name.clone()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn
    }

    #[test]
    fn schema_migration_is_idempotent() {
        let conn = fresh_conn();
        init_schema(&conn).unwrap();
    }

    #[test]
    fn empty_namespace_check() {
        let conn = fresh_conn();
        assert!(!namespace_has_any(&conn, "default").unwrap());
        assert!(!namespace_has_any(&conn, "system").unwrap());
    }

    #[test]
    fn insert_then_load_returns_rule() {
        let conn = fresh_conn();
        insert_rule(
            &conn,
            &NewRule {
                namespace: "system",
                name: "memory_check",
                pattern: r"(?i)\bmemory\b",
                tool: "shell",
                args_json: r#"{"command":"cat /proc/meminfo"}"#,
                description: Some("show memory"),
                priority: 100,
            },
        )
        .unwrap();
        assert!(namespace_has_any(&conn, "system").unwrap());
        let rules = load_active_rules(&conn, "default").unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].name, "memory_check");
        assert_eq!(rules[0].namespace, "system");
    }

    #[test]
    fn customer_namespace_overrides_system_by_priority() {
        let conn = fresh_conn();
        insert_rule(
            &conn,
            &NewRule {
                namespace: "system",
                name: "memory_check",
                pattern: r"\bmem\b",
                tool: "shell",
                args_json: r#"{"command":"free"}"#,
                description: Some("system version"),
                priority: 100,
            },
        )
        .unwrap();
        insert_rule(
            &conn,
            &NewRule {
                namespace: "customer_acme",
                name: "memory_check",
                pattern: r"\bmem\b",
                tool: "shell",
                args_json: r#"{"command":"cat /proc/meminfo"}"#,
                description: Some("customer override"),
                priority: 10,
            },
        )
        .unwrap();
        let rules = load_active_rules(&conn, "customer_acme").unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].namespace, "customer_acme");
        assert_eq!(rules[0].description.as_deref(), Some("customer override"));
    }

    #[test]
    fn other_namespaces_are_invisible() {
        let conn = fresh_conn();
        insert_rule(
            &conn,
            &NewRule {
                namespace: "customer_a",
                name: "thing",
                pattern: r"x",
                tool: "shell",
                args_json: "{}",
                description: None,
                priority: 100,
            },
        )
        .unwrap();
        let rules = load_active_rules(&conn, "customer_b").unwrap();
        assert!(rules.is_empty());
    }
}
