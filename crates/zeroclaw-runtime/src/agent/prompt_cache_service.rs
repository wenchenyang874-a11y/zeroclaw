//! 三层提示词匹配 + V6 自学习的**前端无关**封装。
//!
//! 设计见 doc/3layers_sdk/架构设计/三层匹配统一接入设计.md。
//!
//! 所有 turn 入口（CLI REPL / channel orchestrator / ACP / gateway webhook）都通过
//! `PromptCacheService` 使用三层匹配与学习，保证命中（L1/L2→零 token dispatch）、自学习
//! （L3 成功→record_learning→PromotionWorker 晋升 L1）、override/反悔、humanize 行为**完全一致**，
//! 杜绝多份实现漂移。本组件只管"三层 + 学习"，不碰 I/O（streaming / approval / history）。

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use zeroclaw_providers::Provider;
use zeroclaw_prompt_cache::{PromptCache, PromptMatcher};

use crate::agent::promotion::{PromotionEvent, spawn_promotion_worker};
use crate::tools::Tool;

/// 三层匹配 + 学习服务。`cache=None` 表示未启用（所有方法变透传，前端逻辑不分叉）。
pub struct PromptCacheService {
    cache: Option<Arc<PromptMatcher>>,
    learning_enabled: bool,
    /// PromotionWorker 句柄——随本服务存活，Drop 时 abort（绑定前端生命周期）。
    promotion_handle: Option<JoinHandle<()>>,
    /// 晋升通知源；前端可 subscribe 后打印"🎓 已学到"或仅记日志。
    promotion_tx: Option<broadcast::Sender<PromotionEvent>>,
}

impl PromptCacheService {
    /// 从 config 构建 PromptMatcher（L1 引擎 + L2 embedder）。**不**启动 PromotionWorker
    /// （worker 需前端提供 provider，见 [`start_promotion_worker`]）。enabled=false → cache=None。
    pub fn from_config(
        cfg: &zeroclaw_config::schema::PromptCacheConfig,
        string_match_cfg: &zeroclaw_config::schema::StringMatchConfig,
    ) -> Result<Self> {
        let cache = build_prompt_matcher(cfg, string_match_cfg)?;
        Ok(Self {
            cache,
            learning_enabled: cfg.learning_enabled,
            promotion_handle: None,
            promotion_tx: None,
        })
    }

    /// 禁用实例（cache=None，所有方法透传）。用于不需要三层的入口（如公告投递）。
    pub fn disabled() -> Self {
        Self {
            cache: None,
            learning_enabled: false,
            promotion_handle: None,
            promotion_tx: None,
        }
    }

    /// 三层是否启用（cache 构建成功）。
    pub fn enabled(&self) -> bool {
        self.cache.is_some()
    }

    /// 取底层 PromptMatcher（worker 启动 / 单测用）。
    pub fn cache(&self) -> Option<&Arc<PromptMatcher>> {
        self.cache.as_ref()
    }

    /// 启动 PromotionWorker（仅当 learning_enabled 且 cache 存在且未启动过）。
    /// `worker_provider` 由前端按自己的 provider 配置构建。句柄存入本服务，Drop 即停。
    pub fn start_promotion_worker(
        &mut self,
        worker_provider: Arc<dyn Provider>,
        model: String,
        poll_secs: u64,
        threshold_n: i64,
    ) {
        if self.promotion_handle.is_some() {
            return;
        }
        let Some(cache) = self.cache.clone() else {
            return;
        };
        if !self.learning_enabled {
            return;
        }
        let (tx, _rx) = broadcast::channel::<PromotionEvent>(16);
        println!("{}后台任务已启动", zeroclaw_prompt_cache::log_tag("自学习", "1;35"));
        let handle = spawn_promotion_worker(
            cache,
            worker_provider,
            model,
            poll_secs,
            threshold_n,
            Some(tx.clone()),
        );
        self.promotion_tx = Some(tx);
        self.promotion_handle = Some(handle);
    }

    /// 订阅晋升通知。CLI 订阅后打印"🎓 已学到"；channel 可订阅后推会话或仅日志。
    /// 未启动 worker 时返回 None。
    pub fn subscribe_promotion(&self) -> Option<broadcast::Receiver<PromotionEvent>> {
        self.promotion_tx.as_ref().map(broadcast::Sender::subscribe)
    }

    /// L1/L2 命中 → 执行工具 → humanize → `Some(自然语言)`；未命中 / 未启用 → `None`。
    pub async fn try_dispatch(&self, prompt: &str, tools: &[Box<dyn Tool>]) -> Option<String> {
        let cache = self.cache.as_ref()?;
        let m = match cache.try_match(prompt).await {
            Ok(Some(m)) => m,
            Ok(None) => return None,
            Err(e) => {
                tracing::warn!(error = %e, "prompt-cache try_match errored, falling through to provider");
                return None;
            }
        };
        let tool_name = m.tool_call.name.clone();
        let result = match tools.iter().find(|t| t.name() == tool_name) {
            Some(t) => match t.execute(m.tool_call.args.clone()).await {
                Ok(r) if r.success => crate::agent::loop_::humanize_cached_output(&r.output),
                Ok(r) => format!(
                    "[prompt-cache] tool '{}' failed: {}",
                    tool_name,
                    r.error.unwrap_or(r.output)
                ),
                Err(e) => format!("[prompt-cache] tool '{}' error: {}", tool_name, e),
            },
            None => format!(
                "[prompt-cache] hit rule_id={} ({:?}, score={:?}) but tool '{}' is not registered",
                m.rule_id, m.level, m.score, tool_name
            ),
        };
        Some(result)
    }

    /// 用户是否显式"走大模型"（override 关键词）。命中则前端应直接走 L3 且本轮不学习。
    pub fn is_llm_override(&self, prompt: &str) -> bool {
        self.cache
            .as_ref()
            .is_some_and(|c| c.is_llm_override(prompt))
    }

    /// 反悔检测（有上一轮 pending 的前端用；channel 无人值守可不调）。
    /// 命中反悔 → 撤回 pending（已晋升的连 L1 规则一起删 + 重载）→ 返回 true。
    pub async fn process_regret(&self, pending_id: i64, next_user_message: &str, regret_keywords: &[String]) -> bool {
        let Some(cache) = self.cache.as_ref() else {
            return false;
        };
        match cache.process_regret(pending_id, next_user_message, regret_keywords).await {
            Ok(withdrew) => withdrew,
            Err(e) => {
                tracing::warn!(error = %e, pending_id, "process_regret failed");
                false
            }
        }
    }

    /// L3 成功后按统一策略学习：override 跳过、空 prompt 跳过、成功调用次数 != 1 跳过；
    /// 否则 record_learning。返回 pending_id（学到了）或 None。
    /// `successful` 是本轮**执行成功**的工具调用 (name, args)，由前端在 run_tool_call_loop 时收集。
    pub async fn maybe_learn(
        &self,
        prompt: &str,
        successful: Vec<(String, Value)>,
    ) -> Option<i64> {
        let cache = self.cache.as_ref()?;
        if prompt.trim().is_empty() {
            return None;
        }
        // override（"让大模型…"）：强制走云端的话型不下沉 L1，否则下次被 L1 拦截致 override 失效。
        if cache.is_llm_override(prompt) {
            println!(
                "{}本次不学习（指定走大模型的话型不下沉 L1）",
                zeroclaw_prompt_cache::log_tag("直连大模型", "1;34")
            );
            return None;
        }
        if successful.is_empty() {
            return None;
        }
        // 本轮工具调用次数 > 1 → 跳过：L1 是"单 prompt → 单次 dispatch"，多次调用无法忠实下沉。
        if successful.len() > 1 {
            let names: Vec<&str> = successful.iter().map(|(n, _)| n.as_str()).collect();
            println!(
                "{}本轮 {} 次工具调用（{}），多步不下沉 L1",
                zeroclaw_prompt_cache::log_tag("跳过学习", "2;37"),
                successful.len(),
                names.join(", ")
            );
            return None;
        }
        let (name, args) = successful.into_iter().next().unwrap();
        let tc = zeroclaw_prompt_cache::ToolCall { name, args };
        match cache.record_learning(prompt, &tc, "v0-auto").await {
            Ok(pid) => {
                tracing::debug!(
                    pending_id = pid,
                    tool = %tc.name,
                    prompt = %prompt,
                    "[即时学习] L3 success → pending_rules"
                );
                Some(pid)
            }
            Err(e) => {
                tracing::warn!(error = %e, "record_learning failed");
                None
            }
        }
    }
}

impl Drop for PromptCacheService {
    fn drop(&mut self) {
        if let Some(h) = self.promotion_handle.take() {
            h.abort();
        }
    }
}

/// 按 PromptCacheConfig + StringMatchConfig 构建 PromptMatcher。enabled=false / 缺字段 → Ok(None)。
/// （原 loop_.rs::build_prompt_cache_for_loop，迁此处成为唯一实现。）
fn build_prompt_matcher(
    cfg: &zeroclaw_config::schema::PromptCacheConfig,
    string_match_cfg: &zeroclaw_config::schema::StringMatchConfig,
) -> Result<Option<Arc<PromptMatcher>>> {
    if !cfg.enabled {
        return Ok(None);
    }
    let db_path = cfg
        .db_path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("prompt_cache.db_path required when enabled"))?;
    let model_path = cfg
        .model_path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("prompt_cache.model_path required when enabled"))?;
    let model_id = cfg
        .model_id
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("prompt_cache.model_id required when enabled"))?;
    let pooling = match cfg.pooling.to_ascii_lowercase().as_str() {
        "cls" => zeroclaw_prompt_cache::LlamaPoolingType::Cls,
        "mean" => zeroclaw_prompt_cache::LlamaPoolingType::Mean,
        "last" => zeroclaw_prompt_cache::LlamaPoolingType::Last,
        "none" => zeroclaw_prompt_cache::LlamaPoolingType::None,
        other => anyhow::bail!("prompt_cache.pooling: unknown value '{other}'"),
    };
    let embedder_cfg = zeroclaw_prompt_cache::LlamaEmbedderConfig {
        model_path: model_path.clone(),
        model_id: model_id.to_string(),
        n_ctx: cfg.n_ctx,
        n_threads: cfg.n_threads,
        pooling,
    };
    let embedder = Arc::new(zeroclaw_prompt_cache::LlamaCppEmbedder::new(embedder_cfg)?);
    let l2 = zeroclaw_prompt_cache::prompt_matcher::L2Thresholds {
        accept: cfg.accept,
        reject: cfg.reject,
    };
    let sm_db_path = zeroclaw_config::policy::expand_user_path(&string_match_cfg.db_path);
    let sm_engine = if string_match_cfg.enabled {
        zeroclaw_prompt_cache::StringMatchEngine::init(
            &sm_db_path,
            &string_match_cfg.namespace,
            string_match_cfg.allow_factory_seed,
        )
    } else {
        zeroclaw_prompt_cache::StringMatchEngine::empty()
    };
    let cache = zeroclaw_prompt_cache::PromptMatcher::open(db_path, model_id, l2)?
        .with_string_match(sm_engine, &sm_db_path, &string_match_cfg.namespace)
        .with_embedder(embedder)?;
    Ok(Some(Arc::new(cache)))
}
