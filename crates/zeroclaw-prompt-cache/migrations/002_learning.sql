-- v4 自学习闭环扩展
--   - rules.match_kind: L1 子级（exact / fts / regex）
--   - pending_rules: 加 regex_proposal / validation_state / last_user_turn_at（反悔窗口）
--   - negative_samples: 验证 regex 不误伤的负样本集
--
-- 现有 schema 由 001_initial.sql 建立。这一 migration 增量打。

-- ── rules: 加 match_kind ─────────────────────────────────────────
-- exact:  WHERE pattern = ? (L1.1)
-- fts:    rules_fts MATCH ?  (L1.2)
-- regex:  regex::Regex::is_match(prompt) (L1.3)
-- 现有规则默认 exact——SQL 行为不变。
ALTER TABLE rules ADD COLUMN match_kind TEXT NOT NULL DEFAULT 'exact';
CREATE INDEX IF NOT EXISTS idx_rules_match_kind ON rules(match_kind);

-- ── pending_rules: 自学习候选状态机 ─────────────────────────────
-- validation_state:
--   'pending'    刚 record_learning 写入；等 occurrences 累积或反悔窗口结束
--   'generating' worker 正在调 LLM 生成 regex
--   'validating' regex 拿到，正在跑 validate
--   'rejected'   验证失败（编译错/原 prompt 不命中/与现有规则冲突/误伤负样本）
--   'promoted'   已写入 rules 表（pending_rules 这条可以留作记录或归档）
--   'withdrawn'  下一轮用户反悔（不对/取消/算了）撤销，永远不晋升
ALTER TABLE pending_rules ADD COLUMN regex_proposal TEXT;
ALTER TABLE pending_rules ADD COLUMN validation_state TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE pending_rules ADD COLUMN validation_log TEXT;
-- 反悔窗口：record 时刻 + 下一轮用户消息到达时刻
ALTER TABLE pending_rules ADD COLUMN recorded_at INTEGER NOT NULL DEFAULT 0;
ALTER TABLE pending_rules ADD COLUMN confirmed_at INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_pending_state ON pending_rules(validation_state);

-- ── negative_samples: regex 验证的禁区 ───────────────────────────
-- 候选 regex 必须不能命中任何 negative_samples.prompt
-- 来源：手工灌入 + test_set/cn/long_set/queries 里 idx=0 的那些"该 MISS"的查询
CREATE TABLE IF NOT EXISTS negative_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt      TEXT NOT NULL UNIQUE,
    note        TEXT,                       -- 可选：为什么这条该 MISS
    created_at  INTEGER NOT NULL
);

-- schema version bump
INSERT OR REPLACE INTO prompt_cache_meta(key, value) VALUES ('schema_version', '2');
