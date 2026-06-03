//! 三层提示词匹配缓存（PromptCache）
//!
//! 在 Agent.turn() 之前尝试本地命中：
//!   - L1 字符串匹配（regex → 直调工具）
//!   - L2 向量匹配（cosine，in-process llama-cpp-2）
//!   - L3 云端大模型意图识别
//! 本地 miss 才下放云端。
//!
//! 设计文档：
//! - doc/610llama/架构设计/三级提示词匹配机制设计.md
//! - doc/610llama/架构设计/三级匹配落地方案选型与权衡.md

pub mod embedder;
pub mod error;
pub mod string_match;
pub mod string_match_db;
pub mod schema;
pub mod prompt_matcher;
pub mod types;

pub use embedder::{Embedder, LlamaCppEmbedder, LlamaEmbedderConfig};
pub use error::PromptCacheError;
pub use string_match::{StrMatchDecision, StringMatchEngine, RuntimeRule};
pub use string_match_db::{DbRule, NewRule};
pub use prompt_matcher::PromptMatcher;
pub use types::{
    MatchLevel, MatchResult, PendingRule, RuleKind, ToolCall, ValidationState,
};
// 让外部 crate 不必直接依赖 llama-cpp-2 也能选 pooling
pub use llama_cpp_2::context::params::LlamaPoolingType;

use async_trait::async_trait;

/// 着色的 `[模块]` 日志标签。仅当 stdout 是终端时上色（管道/重定向输出纯文本，
/// 避免 ANSI 码污染）。`ansi` 取 ANSI SGR 参数，如 "1;32"(绿粗) "1;36"(青粗) "1;34"(蓝粗)。
/// 与大模型回复前缀 `[zeroclaw]` 统一风格，便于在密集日志里按模块分层扫读。
pub fn log_tag(label: &str, ansi: &str) -> String {
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        format!("\u{1b}[{ansi}m[{label}]\u{1b}[0m ")
    } else {
        format!("[{label}] ")
    }
}

/// 本地匹配缓存接口（字符串匹配 L1 + L2 embedding + 自学习）。
///
/// `try_match` 在 Agent.turn 开头调；命中即直接构造 ToolCall 走 dispatcher，零 token。
/// `record_learning` 在 L3 命中并执行成功 + 用户非反悔之后调，把候选写进 pending_rules。
#[async_trait]
pub trait PromptCache: Send + Sync {
    /// 尝试 字符串匹配 L1 / L2 匹配。返回 `Ok(Some(...))` 表示本地命中。
    async fn try_match(&self, prompt: &str) -> Result<Option<MatchResult>, PromptCacheError>;

    /// 记录 L3 命中后的候选 (prompt, tool_call)。返回 pending_rules 行 id，
    /// 调用方应该把这个 id 记在会话状态里，下一轮跑 [`process_regret`] 看用户有没有反悔。
    async fn record_learning(
        &self,
        prompt: &str,
        tool_call: &ToolCall,
        tool_version: &str,
    ) -> Result<i64, PromptCacheError>;

    /// 反悔识别：对**上一轮 record_learning 返回的 pending_id**，看本轮用户输入是否
    /// 包含反悔关键词；命中即把该 pending 标 withdrawn，返回 true。否则返回 false。
    async fn process_regret(
        &self,
        pending_id: i64,
        next_user_message: &str,
        regret_keywords: &[String],
    ) -> Result<bool, PromptCacheError>;

    /// 用户是否显式要求"走大模型"（override 关键词，如"问大模型/用 AI/让大模型…"）。
    /// override 的语义是**这一句强制走云端、绕过本地匹配**，因此**不应学进 L1**——
    /// 否则下次同样话型会被 L1 命中拦截，override 形同失效。调用方在 record_learning
    /// 前用它判断跳过学习。默认 false（无 override 能力的实现一律视作非 override）。
    fn is_llm_override(&self, _prompt: &str) -> bool {
        false
    }
}
