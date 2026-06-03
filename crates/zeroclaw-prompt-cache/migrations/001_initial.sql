-- 三级提示词匹配 schema (Q2 A 扁平方案)
-- 对齐 doc/610llama/架构设计/三级匹配落地方案选型与权衡.md §4
-- - rules: ToolCall 字段直接展开 (tool_name / tool_args / tool_version)
-- - kind: Intent | ParameterizedIntent (Q3：v1 只 L1/L2 处理 Intent，ParameterizedIntent 强制走 L3)
-- - learnable: 工具级安全开关 (Q7 b 默认 false)
-- - rule_embeddings.vec: BLOB (in-process Rust 直接操作 byte，不走文本中转)
-- - pending_rules: 自学习候选 (§6)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS rules (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern       TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'Intent',          -- Intent | ParameterizedIntent
    tool_name     TEXT NOT NULL,
    tool_args     TEXT NOT NULL DEFAULT '{}',              -- JSON object
    tool_version  TEXT NOT NULL,                           -- sha256 of tool spec at registration (Q8)
    category      TEXT,
    priority      INTEGER NOT NULL DEFAULT 0,
    learnable     INTEGER NOT NULL DEFAULT 0,              -- 0/1, 工具级安全开关 (Q7)
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rules_tool_name ON rules(tool_name);
CREATE INDEX IF NOT EXISTS idx_rules_kind      ON rules(kind);
CREATE INDEX IF NOT EXISTS idx_rules_category  ON rules(category);

-- L1.2 FTS5 trigram 倒排索引
CREATE VIRTUAL TABLE IF NOT EXISTS rules_fts USING fts5(
    pattern,
    content='rules',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS rules_ai AFTER INSERT ON rules BEGIN
    INSERT INTO rules_fts(rowid, pattern) VALUES (new.id, new.pattern);
END;
CREATE TRIGGER IF NOT EXISTS rules_ad AFTER DELETE ON rules BEGIN
    INSERT INTO rules_fts(rules_fts, rowid, pattern) VALUES ('delete', old.id, old.pattern);
END;
CREATE TRIGGER IF NOT EXISTS rules_au AFTER UPDATE ON rules BEGIN
    INSERT INTO rules_fts(rules_fts, rowid, pattern) VALUES ('delete', old.id, old.pattern);
    INSERT INTO rules_fts(rowid, pattern) VALUES (new.id, new.pattern);
END;

-- L2 向量。BLOB 存连续 float32 (little-endian, dim 个 4-byte 元素)
-- 已 L2 归一化，cosine 退化为点积
CREATE TABLE IF NOT EXISTS rule_embeddings (
    rule_id  INTEGER PRIMARY KEY REFERENCES rules(id) ON DELETE CASCADE,
    dim      INTEGER NOT NULL,
    model    TEXT NOT NULL,                               -- 例 'bge-small-zh-v1.5-q4_k_m'
    vec      BLOB NOT NULL
);

-- 自学习候选 (§6.1)。L3 命中并执行成功 + 用户非反悔后写入
-- UNIQUE(prompt, tool_name, tool_args, tool_version) 保证 record_learning 的 upsert 触发 ON CONFLICT
CREATE TABLE IF NOT EXISTS pending_rules (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt        TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    tool_args     TEXT NOT NULL,
    tool_version  TEXT NOT NULL,
    embedding     BLOB,                                   -- nullable, 后台异步补 (Q5 异步)
    occurrences   INTEGER NOT NULL DEFAULT 1,             -- 同 (prompt, tool_call) 出现次数 (Q6 b 自动晋升用)
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    UNIQUE(prompt, tool_name, tool_args, tool_version)
);

CREATE INDEX IF NOT EXISTS idx_pending_tool ON pending_rules(tool_name);

-- schema 版本号，启动期校验
CREATE TABLE IF NOT EXISTS prompt_cache_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO prompt_cache_meta(key, value) VALUES
    ('schema_version', '1'),
    ('embedding_model', '');                              -- build_db 时填
