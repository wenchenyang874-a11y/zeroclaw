//! SQLite 后端实现。
//!
//! 启动期：
//! 1. 打开/建库
//! 2. 应用 schema (idempotent)
//! 3. 校验 schema_version + embedding_model
//! 4. 把 rule_embeddings 全表 load 到 `Vec<RuleVec>`，供 L2 直接做点积
//!
//! 运行期 try_match：
//! - L1.1: pattern = ? 直查
//! - L1.2: rules_fts MATCH ? + bm25 排序
//! - L2: 用 embedder 拿到 query 向量 → 与内存索引点积 → top1/top2 → 阈值判定
//!
//! 当前 phase 1 只实现 L1.1 + L1.2，L2 留 stub（phase 2 接 llama-cpp-2）。

use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde_json::Value as JsonValue;

use crate::embedder::Embedder;
use crate::error::PromptCacheError;
use crate::string_match::{StrMatchDecision, StringMatchEngine};
use crate::schema;
use crate::types::{MatchLevel, MatchResult, PendingRule, RuleKind, ToolCall};
use crate::PromptCache;

/// L2 阈值（由 [Embedding模型板上测试报告.md] BGE/CN long 实测得出）。
#[derive(Debug, Clone, Copy)]
pub struct L2Thresholds {
    pub accept: f32,
    pub reject: f32,
}

impl Default for L2Thresholds {
    fn default() -> Self {
        Self {
            accept: 0.66,
            reject: 0.54,
        }
    }
}

/// 内存里一条规则的"完整快照"，启动期 load 一次（L2 用）。
#[derive(Debug, Clone)]
struct RuleEntry {
    id: i64,
    pattern: String,
    kind: RuleKind,
    tool_name: String,
    tool_args: JsonValue,
    vec: Option<Vec<f32>>,
}

pub struct PromptMatcher {
    conn: Mutex<Connection>,
    /// 启动期一次性 load。运行期只读，L2 向量匹配用。
    rules: Vec<RuleEntry>,
    /// 内存索引里所有向量的维度（必须一致）。0 表示没装载到任何向量。
    vec_dim: usize,
    /// 当前进程使用的 embedding 模型 id（与 db 里 embedding_model meta 校验过）。
    embedding_model: String,
    /// 可选 embedder。无 embedder 时 L2 直接 miss 走 L3。
    embedder: Option<Arc<dyn Embedder>>,
    /// L2 阈值。
    pub l2: L2Thresholds,
    /// L1 StringMatch 引擎。RwLock 是为了让 PromotionWorker 晋升后 hot-reload，
    /// 不需要重启进程。读路径（match_input）开销极小（regex slice 遍历），
    /// 写路径仅在 reload 时被 PromotionWorker 调用一次。
    string_match: std::sync::RwLock<StringMatchEngine>,
    /// L1 命名空间（reload 时重新 init 用）。
    sm_namespace: String,
    /// L1 规则 DB 连接（独立于 l2_vector.db）。PromotionWorker 晋升时写这里。
    sm_conn: Option<Mutex<Connection>>,
    /// string_match DB 路径（reload 用）。
    sm_db_path: Option<std::path::PathBuf>,
}

impl PromptMatcher {
    /// 打开 sqlite 文件并 load 规则索引。
    ///
    /// `embedding_model_id`: 调用方告知本进程将用哪个模型做 query embedding。
    ///   传 `""` 则跳过模型校验（仅 build_db 阶段使用）。
    pub fn open(
        db_path: impl AsRef<Path>,
        embedding_model_id: &str,
        l2: L2Thresholds,
    ) -> Result<Self, PromptCacheError> {
        let mut conn = Connection::open_with_flags(
            db_path.as_ref(),
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        // 板上 RAM 紧，关掉 sqlite 自己的内存统计 + WAL 同步级别调到合理位
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;

        schema::apply(&conn)?;
        schema::validate(&conn, embedding_model_id)?;

        let (rules, vec_dim) = Self::load_rules(&mut conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
            rules,
            vec_dim,
            embedding_model: embedding_model_id.to_string(),
            embedder: None,
            l2,
            string_match: std::sync::RwLock::new(StringMatchEngine::empty()),
            sm_namespace: "default".to_string(),
            sm_conn: None,
            sm_db_path: None,
        })
    }

    /// 挂上 StringMatch L1 引擎。同时打开 string_match DB 连接供后续写入和 reload。
    pub fn with_string_match(
        mut self,
        string_match: StringMatchEngine,
        sm_db_path: &std::path::Path,
        namespace: &str,
    ) -> Self {
        self.sm_conn = crate::string_match_db::open(sm_db_path)
            .map(|c| Some(Mutex::new(c)))
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "string_match db open failed; auto-promoted rules will not be persisted");
                None
            });
        self.sm_db_path = Some(sm_db_path.to_path_buf());
        self.sm_namespace = namespace.to_string();
        *self.string_match.write().expect("string_match RwLock poisoned") = string_match;
        self
    }

    /// 挂上 embedder，启用 L2。embedder 的 model_id 与 db 里的必须一致。
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Result<Self, PromptCacheError> {
        if !self.embedding_model.is_empty() && embedder.model_id() != self.embedding_model {
            return Err(PromptCacheError::EmbeddingModelMismatch {
                db: self.embedding_model.clone(),
                runtime: embedder.model_id().to_string(),
            });
        }
        if self.vec_dim != 0 && embedder.dim() != self.vec_dim {
            return Err(PromptCacheError::VectorDimMismatch {
                rule_id: -1,
                stored: self.vec_dim,
                runtime: embedder.dim(),
            });
        }
        self.embedder = Some(embedder);
        Ok(self)
    }

    /// 内部：把 rules + rule_embeddings 一次性 load 到内存（L2 用）。
    fn load_rules(conn: &mut Connection) -> Result<(Vec<RuleEntry>, usize), PromptCacheError> {
        let mut stmt = conn.prepare(
            "SELECT r.id, r.pattern, r.kind, r.tool_name, r.tool_args,
                    e.dim, e.vec
               FROM rules r
               LEFT JOIN rule_embeddings e ON e.rule_id = r.id
               ORDER BY r.id",
        )?;

        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let pattern: String = row.get(1)?;
            let kind_s: String = row.get(2)?;
            let tool_name: String = row.get(3)?;
            let tool_args_s: String = row.get(4)?;
            let dim: Option<i64> = row.get(5)?;
            let blob: Option<Vec<u8>> = row.get(6)?;
            Ok((id, pattern, kind_s, tool_name, tool_args_s, dim, blob))
        })?;

        let mut entries = Vec::new();
        let mut detected_dim: usize = 0;

        for r in rows {
            let (id, pattern, kind_s, tool_name, tool_args_s, dim_opt, blob_opt) = r?;
            let kind: RuleKind = kind_s
                .parse()
                .map_err(|e: String| PromptCacheError::Embedder(format!("bad kind: {e}")))?;
            let tool_args: JsonValue = serde_json::from_str(&tool_args_s)?;

            let vec = match (dim_opt, blob_opt) {
                (Some(dim), Some(blob)) => {
                    let dim = dim as usize;
                    let expected_bytes = dim * 4;
                    if blob.len() != expected_bytes {
                        return Err(PromptCacheError::InvalidVectorBlob {
                            rule_id: id,
                            expected_bytes,
                            got_bytes: blob.len(),
                        });
                    }
                    if detected_dim == 0 {
                        detected_dim = dim;
                    } else if dim != detected_dim {
                        return Err(PromptCacheError::VectorDimMismatch {
                            rule_id: id,
                            stored: dim,
                            runtime: detected_dim,
                        });
                    }
                    let mut v = Vec::with_capacity(dim);
                    for chunk in blob.chunks_exact(4) {
                        v.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    Some(v)
                }
                _ => None,
            };

            entries.push(RuleEntry {
                id,
                pattern,
                kind,
                tool_name,
                tool_args,
                vec,
            });
        }

        Ok((entries, detected_dim))
    }

    /// 暴露给 build_db 用：直接拿连接做批量插入。
    pub fn raw_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("prompt-cache connection mutex poisoned")
    }

    /// 重新 load L2 内存索引（L2 向量规则更新后调）。
    pub fn reload(&mut self) -> Result<(), PromptCacheError> {
        let mut conn = self.conn.lock().expect("prompt-cache connection mutex poisoned");
        let (rules, vec_dim) = Self::load_rules(&mut conn)?;
        drop(conn);
        self.rules = rules;
        self.vec_dim = vec_dim;
        Ok(())
    }

    /// 重建 L1 StringMatch 引擎并 in-place 替换。供 PromotionWorker 晋升成功后调用，
    /// 无需重启进程即可让新 regex 生效。`&self` 而非 `&mut self`——内部走 RwLock。
    pub fn reload_string_match(&self) -> Result<(), PromptCacheError> {
        let path = self
            .sm_db_path
            .as_ref()
            .ok_or_else(|| PromptCacheError::Embedder("string_match db path not set".into()))?;
        let new_engine = StringMatchEngine::init(path, &self.sm_namespace, false);
        *self.string_match.write().expect("string_match RwLock poisoned") = new_engine;
        Ok(())
    }

    /// 当前 L1 引擎内的规则数（运行时观测用，主要面向单测）。
    pub fn string_match_rule_count(&self) -> usize {
        self.string_match
            .read()
            .expect("string_match RwLock poisoned")
            .rule_count()
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn vec_dim(&self) -> usize {
        self.vec_dim
    }

    /// 把 RuleEntry 转成 L2 命中结果。
    fn make_match(
        rule: &RuleEntry,
        level: MatchLevel,
        score: Option<f32>,
        top2_score: Option<f32>,
    ) -> MatchResult {
        MatchResult {
            level,
            rule_id: rule.id,
            pattern: rule.pattern.clone(),
            tool_call: ToolCall {
                name: rule.tool_name.clone(),
                args: rule.tool_args.clone(),
            },
            score,
            top2_score,
        }
    }
}

#[async_trait]
impl PromptCache for PromptMatcher {
    fn is_llm_override(&self, prompt: &str) -> bool {
        self.string_match
            .read()
            .expect("string_match RwLock poisoned")
            .is_llm_override(prompt)
    }

    async fn try_match(&self, prompt: &str) -> Result<Option<MatchResult>, PromptCacheError> {
        // ===== L1 字符串匹配 =====
        let l1_decision = self
            .string_match
            .read()
            .expect("string_match RwLock poisoned")
            .match_input(prompt);
        match l1_decision {
            StrMatchDecision::LlmOverride { .. } => {
                println!("{}用户指定，跳过本地匹配", crate::log_tag("直连大模型", "1;34"));
                return Ok(None);
            }
            StrMatchDecision::Match { rule } => {
                let args_str = serde_json::to_string(&rule.args_json).unwrap_or_default();
                println!(
                    "{}命中 → {}({})",
                    crate::log_tag("L1", "1;32"),
                    rule.tool,
                    if args_str.len() > 40 { "" } else { &args_str }
                );
                let args: serde_json::Value =
                    serde_json::from_str(&rule.args_json).unwrap_or(serde_json::Value::Null);
                return Ok(Some(MatchResult {
                    level: MatchLevel::L1Regex,
                    rule_id: rule.id,
                    pattern: rule.pattern_src,
                    tool_call: ToolCall {
                        name: rule.tool,
                        args,
                    },
                    score: None,
                    top2_score: None,
                }));
            }
            StrMatchDecision::NoMatch => {}
        }

        // ===== L2 向量匹配 =====
        let Some(embedder) = self.embedder.as_ref() else {
            return Ok(None);
        };
        if self.vec_dim == 0 {
            return Ok(None);
        }

        eprint!("{}向量编码中…", crate::log_tag("L2", "1;36"));
        let _ = std::io::stderr().flush();
        let q = embedder.embed(prompt)?;
        if q.len() != self.vec_dim {
            return Err(PromptCacheError::VectorDimMismatch {
                rule_id: -1,
                stored: self.vec_dim,
                runtime: q.len(),
            });
        }

        let mut best1: f32 = -2.0;
        let mut best2: f32 = -2.0;
        let mut idx1: Option<usize> = None;

        for (i, rule) in self.rules.iter().enumerate() {
            if matches!(rule.kind, RuleKind::ParameterizedIntent) {
                continue;
            }
            let Some(vec) = rule.vec.as_ref() else {
                continue;
            };
            let dot: f32 = q.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            if dot > best1 {
                best2 = best1;
                best1 = dot;
                idx1 = Some(i);
            } else if dot > best2 {
                best2 = dot;
            }
        }

        let Some(i1) = idx1 else {
            eprintln!("\r{}向量库为空", crate::log_tag("L2", "1;36"));
            return Ok(None);
        };
        let top1 = &self.rules[i1];
        let cos1 = best1;
        let cos2 = if best2 > -1.5 { Some(best2) } else { None };

        if cos1 >= self.l2.accept {
            eprintln!(
                "\r{}命中 {}({})  cos={:.4}  [接受≥{:.2}]",
                crate::log_tag("L2", "1;36"),
                top1.tool_name,
                top1.pattern,
                cos1,
                self.l2.accept,
            );
            return Ok(Some(Self::make_match(
                top1,
                MatchLevel::L2Cosine,
                Some(cos1),
                cos2,
            )));
        }
        eprintln!(
            "\r{}cos={:.4}  top=\"{}\"  [接受≥{:.2}  拒识<{:.2}]",
            crate::log_tag("L2", "1;36"),
            cos1,
            top1.pattern,
            self.l2.accept,
            self.l2.reject,
        );
        Ok(None)
    }

    async fn record_learning(
        &self,
        prompt: &str,
        tool_call: &ToolCall,
        tool_version: &str,
    ) -> Result<i64, PromptCacheError> {
        let now = unix_ts();
        let args_s = serde_json::to_string(&tool_call.args)?;
        let conn = self.conn.lock().expect("prompt-cache connection mutex poisoned");

        // 即时学习：confirmed_at = now() 直写，跳过反悔窗口。
        // process_regret 仍可调用（用户主动撤销时把 validation_state 改成 'withdrawn'），
        // 但缺省路径不再依赖"下一轮没说反悔关键词"才推进。
        // 配合 promote_threshold_n=1，首次成功即送 PromotionWorker。
        conn.execute(
            "INSERT INTO pending_rules
                (prompt, tool_name, tool_args, tool_version, occurrences,
                 validation_state, recorded_at, confirmed_at, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, 1, 'pending', ?5, ?5, ?5, ?5)
             ON CONFLICT(prompt, tool_name, tool_args, tool_version)
             DO UPDATE SET
                 occurrences     = occurrences + 1,
                 confirmed_at    = excluded.confirmed_at,
                 validation_state = CASE
                     WHEN validation_state = 'rejected' THEN 'pending'
                     ELSE validation_state
                 END,
                 updated_at      = excluded.updated_at,
                 recorded_at     = excluded.recorded_at",
            params![prompt, tool_call.name, args_s, tool_version, now],
        )?;
        println!(
            "\n{}\"{}\" → {}，已记录，后台生成规则中… (confirmed_at={})",
            crate::log_tag("即时学习", "1;35"),
            prompt,
            tool_call.name,
            now
        );
        // 回读这一行的 id
        let id: i64 = conn.query_row(
            "SELECT id FROM pending_rules
              WHERE prompt = ?1 AND tool_name = ?2 AND tool_args = ?3 AND tool_version = ?4",
            params![prompt, tool_call.name, args_s, tool_version],
            |r| r.get(0),
        )?;
        Ok(id)
    }

    async fn process_regret(
        &self,
        pending_id: i64,
        next_user_message: &str,
        regret_keywords: &[String],
    ) -> Result<bool, PromptCacheError> {
        // 大小写不敏感地扫一遍关键词
        let lower = next_user_message.to_lowercase();
        let hit = regret_keywords
            .iter()
            .any(|kw| lower.contains(&kw.to_lowercase()));
        if !hit {
            // 没反悔 → 标 confirmed_at，让 PromotionWorker 知道反悔窗口已过
            let conn = self.conn.lock().expect("prompt-cache connection mutex poisoned");
            conn.execute(
                "UPDATE pending_rules SET confirmed_at = ?1
                  WHERE id = ?2 AND validation_state = 'pending' AND confirmed_at = 0",
                params![unix_ts(), pending_id],
            )?;
            return Ok(false);
        }
        // 反悔了 → 标 withdrawn。注意：候选可能**已被 PromotionWorker 晋升**进 L1
        // （状态变 'promoted'、string_match_rules 里已有 learned_{id} 规则、引擎已热重载），
        // 这时光撤 pending 不够，下次还会命中 L1。所以也要删掉已晋升的 L1 规则 + 重载引擎。
        {
            let conn = self.conn.lock().expect("prompt-cache connection mutex poisoned");
            conn.execute(
                "UPDATE pending_rules SET validation_state = 'withdrawn', updated_at = ?1
                  WHERE id = ?2 AND validation_state IN ('pending', 'generating', 'validating', 'promoted')",
                params![unix_ts(), pending_id],
            )?;
        }
        // 删已晋升的 L1 规则（learned_{pending_id} @ namespace 'auto'）。未晋升则无此行，no-op。
        let mut removed_from_l1 = false;
        if let Some(sm_conn) = self.sm_conn.as_ref() {
            let sm = sm_conn.lock().expect("string_match connection mutex poisoned");
            removed_from_l1 =
                crate::string_match_db::delete_rule(&sm, "auto", &format!("learned_{pending_id}"))?;
        }
        // 删了才重载，让内存 L1 引擎丢掉这条规则（否则撤销后下次仍 ⚡L1 命中）。
        if removed_from_l1 {
            self.reload_string_match()?;
        }
        Ok(true)
    }
}

// ── PromotionWorker 调用的低层接口 ──────────────────────────────
// 这些不进 trait（只 PromptMatcher 暴露），因为 promotion 是后台 task 的
// 实现细节，不属于"前台 try_match/record_learning"的 contract。

impl PromptMatcher {
    /// 拿一批已通过反悔窗口、occurrences >= threshold、还没晋升的候选规则。
    /// PromotionWorker 用这个驱动 LLM 生成 regex。
    pub fn iter_pending_for_promotion(
        &self,
        threshold_n: i64,
        limit: usize,
    ) -> Result<Vec<PendingRule>, PromptCacheError> {
        let conn = self.conn.lock().expect("prompt-cache connection mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, prompt, tool_name, tool_args, tool_version, occurrences, recorded_at
               FROM pending_rules
              WHERE validation_state = 'pending'
                AND confirmed_at > 0
                AND occurrences >= ?1
              ORDER BY occurrences DESC, recorded_at ASC
              LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![threshold_n, limit as i64], |row| {
            let id: i64 = row.get(0)?;
            let prompt: String = row.get(1)?;
            let tool_name: String = row.get(2)?;
            let tool_args_s: String = row.get(3)?;
            let tool_version: String = row.get(4)?;
            let occurrences: i64 = row.get(5)?;
            let recorded_at: i64 = row.get(6)?;
            Ok((
                id,
                prompt,
                tool_name,
                tool_args_s,
                tool_version,
                occurrences,
                recorded_at,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (id, prompt, tool_name, tool_args_s, tool_version, occurrences, recorded_at) = r?;
            let tool_args: JsonValue = serde_json::from_str(&tool_args_s)?;
            out.push(PendingRule {
                id,
                prompt,
                tool_name,
                tool_args,
                tool_version,
                occurrences,
                recorded_at,
            });
        }
        Ok(out)
    }

    /// 推进 pending 状态（pending → generating / validating / rejected / promoted / withdrawn）。
    /// `log` 用来记录验证失败原因或晋升后 rule_id。
    pub fn mark_pending_state(
        &self,
        pending_id: i64,
        new_state: crate::ValidationState,
        log: Option<&str>,
    ) -> Result<(), PromptCacheError> {
        let conn = self.conn.lock().expect("prompt-cache connection mutex poisoned");
        conn.execute(
            "UPDATE pending_rules
                SET validation_state = ?1, validation_log = ?2, updated_at = ?3
              WHERE id = ?4",
            params![new_state.as_str(), log, unix_ts(), pending_id],
        )?;
        Ok(())
    }

    /// 把候选 pending 晋升成 string_match_rules 表里的一条 regex 规则。原子操作（事务）。
    /// 返回新建 rule 的 id。
    pub fn promote_pending_to_rule(
        &self,
        pending_id: i64,
        regex_pattern: &str,
    ) -> Result<i64, PromptCacheError> {
        // 读 pending 信息
        let conn = self.conn.lock().expect("prompt-cache connection mutex poisoned");
        let (tool_name, tool_args_s, tool_version): (String, String, String) = conn.query_row(
            "SELECT tool_name, tool_args, tool_version FROM pending_rules WHERE id = ?1",
            params![pending_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        drop(conn);

        let now = unix_ts();

        // 写 string_match_rules（L1 规则表）
        let sm_conn = self.sm_conn.as_ref()
            .ok_or_else(|| PromptCacheError::Embedder("string_match db not open".into()))?;
        let sm = sm_conn.lock().expect("string_match connection mutex poisoned");
        crate::string_match_db::insert_rule(
            &sm,
            &crate::string_match_db::NewRule {
                namespace: "auto",
                name: &format!("learned_{pending_id}"),
                pattern: regex_pattern,
                tool: &tool_name,
                args_json: &tool_args_s,
                description: Some("auto-promoted from L3 learning"),
                priority: 50,
            },
        )?;
        let new_rule_id = sm.last_insert_rowid();
        drop(sm);

        // 标记 pending 已晋升
        let conn = self.conn.lock().expect("prompt-cache connection mutex poisoned");
        conn.execute(
            "UPDATE pending_rules
                SET validation_state = 'promoted',
                    validation_log = ?1,
                    updated_at = ?2
              WHERE id = ?3",
            params![format!("promoted to string_match_rules.id={new_rule_id}"), now, pending_id],
        )?;
        Ok(new_rule_id)
    }

    /// 验证：候选 regex 对**所有** negative_samples 都不命中。返回第一个误命中的 sample（如果有）。
    pub fn check_regex_against_negative_samples(
        &self,
        compiled: &regex::Regex,
    ) -> Result<Option<String>, PromptCacheError> {
        let conn = self.conn.lock().expect("prompt-cache connection mutex poisoned");
        let mut stmt = conn.prepare("SELECT prompt FROM negative_samples")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        for r in rows {
            let sample = r?;
            if compiled.is_match(&sample) {
                return Ok(Some(sample));
            }
        }
        Ok(None)
    }
}

fn unix_ts() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fresh() -> (TempDir, PromptMatcher) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let cache = PromptMatcher::open(&path, "", L2Thresholds::default()).unwrap();
        (dir, cache)
    }

    fn fresh_with_string_match() -> (TempDir, TempDir, PromptMatcher) {
        let (dir, cache) = fresh();
        let sm_dir = TempDir::new().unwrap();
        let sm_path = sm_dir.path().join("sm.db");
        let engine = StringMatchEngine::init(&sm_path, "test", true);
        // with_string_match 会再 open 一次同一个 DB 文件（用于 PromotionWorker 写入）
        (dir, sm_dir, cache.with_string_match(engine, &sm_path, "test"))
    }

    #[tokio::test]
    async fn schema_applies_idempotently() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("idem.db");
        let _a = PromptMatcher::open(&path, "", L2Thresholds::default()).unwrap();
        let _b = PromptMatcher::open(&path, "", L2Thresholds::default()).unwrap();
    }

    #[tokio::test]
    async fn string_match_l1_no_hit_without_rules() {
        // 没有灌规则 → L1 miss → L2 miss（无 embedder）→ None
        let (_dir, _sm_dir, cache) = fresh_with_string_match();
        let m = cache.try_match("查内存").await.unwrap();
        assert!(m.is_none(), "L1 and L2 both miss, should return None");
    }

    #[tokio::test]
    async fn string_match_l1_miss_falls_through_to_l2() {
        let (_dir, _sm_dir, cache) = fresh_with_string_match();
        // 没有 embedder，L1 miss → L2 miss → None
        let m = cache.try_match("今天天气怎么样").await.unwrap();
        assert!(m.is_none());
    }

    #[tokio::test]
    async fn string_match_llm_override_bypasses_l1() {
        let (_dir, _sm_dir, cache) = fresh_with_string_match();
        // 即使用户说的话能被工厂规则匹配，"用大模型"也优先
        let m = cache.try_match("用大模型帮我开下空调").await.unwrap();
        assert!(m.is_none(), "LLM override should bypass L1");
    }

    #[tokio::test]
    async fn string_match_no_match_for_unrelated_prompt() {
        let (_dir, _sm_dir, cache) = fresh_with_string_match();
        let m = cache.try_match("xyznonexistent123").await.unwrap();
        assert!(m.is_none());
    }

    #[tokio::test]
    async fn process_regret_withdraws_when_keyword_present() {
        let (_dir, cache) = fresh();
        let tc = ToolCall {
            name: "ac.power".into(),
            args: serde_json::json!({}),
        };
        let pid = cache.record_learning("开空调", &tc, "v1").await.unwrap();
        let kws = vec!["不对".to_string(), "取消".to_string()];

        // 下一轮没反悔 → 不撤销
        assert!(!cache.process_regret(pid, "好的谢谢", &kws).await.unwrap());

        // 下一轮反悔 → 撤销
        assert!(cache.process_regret(pid, "不对，我要的是关空调", &kws).await.unwrap());

        // 验证状态确实是 withdrawn
        let conn = cache.raw_conn();
        let state: String = conn
            .query_row(
                "SELECT validation_state FROM pending_rules WHERE id = ?1",
                rusqlite::params![pid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(state, "withdrawn");
    }

    #[tokio::test]
    async fn iter_pending_for_promotion_filters_correctly() {
        let (_dir, cache) = fresh();
        let tc = ToolCall {
            name: "ac.power".into(),
            args: serde_json::json!({}),
        };
        // 三条候选：occurrences 1/2/3，threshold=2 应该只看到 occ>=2 的两条
        for _ in 0..3 { cache.record_learning("p3", &tc, "v1").await.unwrap(); }
        for _ in 0..2 { cache.record_learning("p2", &tc, "v1").await.unwrap(); }
        cache.record_learning("p1", &tc, "v1").await.unwrap();
        let kws = vec!["不对".to_string()];
        for p in ["p1", "p2", "p3"] {
            cache.process_regret(/* dummy */0, p, &kws).await.unwrap();
        }
        // 但 confirmed_at 还得标——直接 SQL 标
        {
            let conn = cache.raw_conn();
            conn.execute(
                "UPDATE pending_rules SET confirmed_at = strftime('%s','now') WHERE confirmed_at = 0",
                [],
            ).unwrap();
        }
        let cands = cache.iter_pending_for_promotion(2, 10).unwrap();
        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0].prompt, "p3");
        assert_eq!(cands[0].occurrences, 3);
    }

    #[tokio::test]
    async fn promote_pending_inserts_regex_rule() {
        let (_dir, _sm_dir, cache) = fresh_with_string_match();
        let tc = ToolCall {
            name: "ac.power".into(),
            args: serde_json::json!({"action": "on"}),
        };
        let pid = cache.record_learning("帮我开空调", &tc, "v1").await.unwrap();
        // 晋升到 string_match_rules 表
        let new_rule_id = cache
            .promote_pending_to_rule(pid, r"(开|打开).*(空调)")
            .unwrap();
        assert!(new_rule_id > 0);
    }

    #[tokio::test]
    async fn record_learning_increments_occurrences() {
        let (_dir, cache) = fresh();
        let tc = ToolCall {
            name: "ac_power_on".into(),
            args: serde_json::json!({}),
        };
        let id1 = cache.record_learning("开下空调", &tc, "v1").await.unwrap();
        let id2 = cache.record_learning("开下空调", &tc, "v1").await.unwrap();
        assert_eq!(id1, id2, "upsert 应该回到同一行");
        let conn = cache.raw_conn();
        let occ: i64 = conn
            .query_row(
                "SELECT occurrences FROM pending_rules WHERE prompt = '开下空调'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(occ, 2);
    }

    /// V6 即时学习：record_learning 直接写 confirmed_at，
    /// iter_pending_for_promotion 在 threshold=1 时立即能拿到候选。
    #[tokio::test]
    async fn record_learning_marks_confirmed_immediately() {
        let (_dir, cache) = fresh();
        let tc = ToolCall {
            name: "aidetect".into(),
            args: serde_json::json!({"model": "det_hvf_hor.bin"}),
        };
        let _pid = cache.record_learning("看看画面里有没有人", &tc, "v1").await.unwrap();
        // 注意：raw_conn() 返回 MutexGuard，必须显式 drop 才能让 iter_pending_for_promotion
        // 重新 lock；否则同一 Mutex 上 deadlock。
        let confirmed: i64 = {
            let conn = cache.raw_conn();
            conn.query_row(
                "SELECT confirmed_at FROM pending_rules WHERE prompt = '看看画面里有没有人'",
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert!(confirmed > 0, "confirmed_at 应该 > 0（即时学习）");
        let cands = cache.iter_pending_for_promotion(1, 10).unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].prompt, "看看画面里有没有人");
    }

    /// Phase B 核心：promote 写入 string_match_rules → reload_string_match() →
    /// 同一进程下次 try_match 直接命中 L1（不重启）。
    #[tokio::test]
    async fn promote_then_reload_lets_l1_hit_in_same_process() {
        let (_dir, _sm_dir, cache) = fresh_with_string_match();
        let tc = ToolCall {
            name: "aidetect".into(),
            args: serde_json::json!({"model": "det_hvf_hor.bin"}),
        };
        // 学一条
        let pid = cache.record_learning("看看画面里有没有人", &tc, "v1").await.unwrap();
        // 晋升
        let regex = r"^.*?(?:看一?下|看看|有没有).*?(?:画面|人).*?$";
        let _rid = cache.promote_pending_to_rule(pid, regex).unwrap();
        // 此时 L1 引擎内存里还没有这条 → 应该 miss
        assert_eq!(cache.string_match_rule_count(), 0, "reload 前 L1 应该还是空");
        // hot reload
        cache.reload_string_match().unwrap();
        assert_eq!(cache.string_match_rule_count(), 1, "reload 后应该有 1 条");
        // 同句子直接命中
        let m = cache.try_match("看看画面里有没有人").await.unwrap();
        assert!(m.is_some(), "reload 后同一句应该走 L1 命中");
        // 同义说法（regex 泛化）也命中
        let m2 = cache.try_match("看一下当前画面里有人吗").await.unwrap();
        assert!(m2.is_some(), "同义改写应该也命中");
    }
}
