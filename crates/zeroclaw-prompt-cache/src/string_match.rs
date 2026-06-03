//! Fast-path: 通过正则匹配常见的用户请求，直接调用工具，跳过 LLM 调用。
//!
//! 数据流：
//!   启动: StringMatchEngine::init(cfg)
//!     1. 打开 l1_string_match.db
//!     2. 若 system namespace 为空且允许 seed → 灌工厂规则
//!     3. 加载 active + system 的 enabled 规则
//!     4. 编译 regex → 安全闸门 → 白名单校验 → RuntimeRule
//!     5. 任何步骤失败 → 降级到内置工厂规则
//!   匹配: engine.match_input(&str)
//!     1. 先 LLM_OVERRIDE_REGEXES（用户说"用大模型"）
//!     2. 再业务规则，命中第一条返回

use regex::Regex;
use std::path::Path;

use crate::error::PromptCacheError;
use crate::string_match_db::{self, DbRule, NewRule};

/// 单条编译后的运行时规则。
#[derive(Debug, Clone)]
pub struct RuntimeRule {
    pub id: i64,
    pub namespace: String,
    pub name: String,
    pub pattern_src: String,
    pub tool: String,
    pub args_json: String,
    pub description: Option<String>,
    pub priority: i64,
    pub updated_at: i64,
    pub regex: Regex,
}

/// 匹配结果。
#[derive(Debug)]
pub enum StrMatchDecision {
    /// 用户明说要用大模型；走 LLM，不论 fast-path 是否能匹配。
    LlmOverride { pattern_idx: usize },
    /// 命中 fast-path 规则，可直接调工具跳过 LLM。
    Match { rule: RuntimeRule },
    /// 没匹配上，走正常 LLM 流程。
    NoMatch,
}

// ── LLM 强制覆盖正则 ────────────────────────────────────────────

const LLM_OVERRIDE_PATTERN_SOURCES: &[&str] = &[
    r"(?i)(用|让|请用|走|经过|通过|借助|问|问一下|叫)\s*(大模型|llm|模型|ai|gpt|deepseek|claude|chatgpt)",
    r"(?i)(大模型|llm|模型|ai|gpt|claude)\s*(查|看|帮|跑|来|执行|分析|想想|，|,|:|：)",
    r"(?i)\b(use|via|with|through|using)\s+(?:\w+\s+){0,3}(llm|model|ai|gpt|claude|chatgpt)\b",
    r"(?i)\b(think|reason|reasoning)\s+(carefully|deeply|step by step)",
];

// ── 工具白名单 ─────────────────────────────────────────────────

const TOOL_WHITELIST: &[&str] = &[
    "shell",
    "memory_recall", "memory_store", "memory_export",
    "calculator", "weather", "weather.today",
    "image_info", "screenshot",
    "hardware_info",
    "aidetect", "get_frame",
    // IoT
    "ac.power", "ac.temp_step", "ac.set_temp", "ac.mode", "ac.fan_speed",
    "light.power",
    "curtain.power",
    "media.play", "media.pause", "media.volume_step",
];

// ── StringMatchEngine ──────────────────────────────────────────────

pub struct StringMatchEngine {
    rules: Vec<RuntimeRule>,
    llm_override_regexes: Vec<Regex>,
}

impl StringMatchEngine {
    /// 空引擎——没有规则，所有输入都 NoMatch。
    pub fn empty() -> Self {
        Self {
            rules: Vec::new(),
            llm_override_regexes: compile_llm_override(),
                    }
    }

    /// 从 l1_string_match.db 初始化。DB 打开失败时返回空引擎（所有输入走 L2/L3）。
    pub fn init(
        db_path: &Path,
        namespace: &str,
        _allow_seed: bool,
    ) -> Self {
        match Self::init_with_db(db_path, namespace) {
            Ok(rules) => {
                println!("{}载入 {} 条规则", crate::log_tag("L1", "1;32"), rules.len());
                Self {
                    rules,
                    llm_override_regexes: compile_llm_override(),
                }
            }
            Err(e) => {
                tracing::error!(
                    error = %e,
                    db_path = %db_path.display(),
                    "L1 init failed; all inputs will miss L1"
                );
                Self {
                    rules: Vec::new(),
                    llm_override_regexes: compile_llm_override(),
                }
            }
        }
    }

    fn init_with_db(
        db_path: &Path,
        namespace: &str,
    ) -> Result<Vec<RuntimeRule>, PromptCacheError> {
        let conn = string_match_db::open(db_path)?;

        let raw_rules = string_match_db::load_active_rules(&conn, namespace)?;
        let mut compiled = Vec::with_capacity(raw_rules.len());
        for r in raw_rules {
            if let Err(e) = check_rule_safety(&r, &r.namespace) {
                tracing::warn!(rule = r.name.as_str(), error = %e, "string_match rule rejected");
                continue;
            }
            match validate_regex_safety(&r.pattern) {
                Ok(regex) => compiled.push(RuntimeRule {
                    id: r.id,
                    namespace: r.namespace,
                    name: r.name,
                    pattern_src: r.pattern,
                    tool: r.tool,
                    args_json: r.args_json,
                    description: r.description,
                    priority: r.priority,
                    updated_at: r.updated_at,
                    regex,
                }),
                Err(e) => {
                    tracing::warn!(rule = r.name.as_str(), error = %e, "string_match regex compile failed; skipping");
                }
            }
        }
        compiled.sort_by(|a, b| {
            a.priority.cmp(&b.priority).then_with(|| a.id.cmp(&b.id))
        });
        Ok(compiled)
    }

    /// 主匹配入口。先 LLM 覆盖，再 string_match 规则。
    pub fn match_input(&self, input: &str) -> StrMatchDecision {
        for (idx, regex) in self.llm_override_regexes.iter().enumerate() {
            if regex.is_match(input) {
                return StrMatchDecision::LlmOverride { pattern_idx: idx };
            }
        }

        for rule in &self.rules {
            if rule.regex.is_match(input) {
                return StrMatchDecision::Match {
                    rule: rule.clone(),
                };
            }
        }
        StrMatchDecision::NoMatch
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// 仅判断输入是否命中 LLM override 正则（"问大模型/用 AI/让大模型…"），
    /// 不查普通规则。用于让调用方在 override 时跳过 V6 学习（强制走云端的话型
    /// 不应下沉 L1，否则下次会被 L1 拦截致 override 失效）。
    pub fn is_llm_override(&self, input: &str) -> bool {
        self.llm_override_regexes.iter().any(|re| re.is_match(input))
    }
}

// ── 辅助函数 ────────────────────────────────────────────────────

fn compile_llm_override() -> Vec<Regex> {
    LLM_OVERRIDE_PATTERN_SOURCES
        .iter()
        .filter_map(|src| {
            Regex::new(src)
                .inspect_err(|e| tracing::error!(pattern = src, error = %e, "invalid LLM_OVERRIDE pattern"))
                .ok()
        })
        .collect()
}

fn validate_regex_safety(pattern: &str) -> Result<Regex, PromptCacheError> {
    let regex = Regex::new(pattern)
        .map_err(|e| PromptCacheError::Embedder(format!("regex compile: {e}")))?;
    const MAX_REGEX_SIZE_BYTES: usize = 1 * 1024 * 1024;
    let size = regex.to_string().len() + regex.capture_names().count() * 32;
    if size > MAX_REGEX_SIZE_BYTES {
        return Err(PromptCacheError::Embedder(format!(
            "regex too large after compile (~{size} bytes); rejected"
        )));
    }
    Ok(regex)
}

fn check_rule_safety(r: &DbRule, namespace: &str) -> Result<(), PromptCacheError> {
    // 只对 system factory 规则做白名单校验；用户 namespace 的规则信任操作者
    if namespace == "system" && !TOOL_WHITELIST.contains(&r.tool.as_str()) {
        return Err(PromptCacheError::Embedder(format!(
            "tool '{}' is not in whitelist", r.tool
        )));
    }
    if let Err(e) = serde_json::from_str::<serde_json::Value>(&r.args_json) {
        return Err(PromptCacheError::Embedder(format!("args_json is not valid JSON: {e}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db_engine() -> (tempfile::TempDir, StringMatchEngine) {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("sm.db");

        // 灌一条 ac.power on 规则
        let conn = string_match_db::open(&db_path).unwrap();
        string_match_db::insert_rule(&conn, &NewRule {
            namespace: "system",
            name: "ac_power_on",
            pattern: r"(打开|开下|开个|帮.*开).*?(空调|冷气)|(空调|冷气).*?(打开|开下|开个|帮.*开)",
            tool: "ac.power",
            args_json: r#"{"action":"on"}"#,
            description: Some("打开空调"),
            priority: 100,
        }).unwrap();
        string_match_db::insert_rule(&conn, &NewRule {
            namespace: "system",
            name: "ac_power_off",
            pattern: r"(关闭|关掉|关).*?(空调|冷气)|(空调|冷气).*?(关闭|关掉)",
            tool: "ac.power",
            args_json: r#"{"action":"off"}"#,
            description: Some("关闭空调"),
            priority: 100,
        }).unwrap();
        drop(conn);

        let engine = StringMatchEngine::init(&db_path, "test", false);
        (dir, engine)
    }

    #[test]
    fn matches_ac_power_on() {
        let (_dir, engine) = db_engine();
        let m = engine.match_input("帮我开下空调");
        assert!(matches!(m, StrMatchDecision::Match { .. }));
        if let StrMatchDecision::Match { rule } = m {
            assert_eq!(rule.tool, "ac.power");
            assert_eq!(rule.args_json, r#"{"action":"on"}"#);
        }
    }

    #[test]
    fn matches_ac_power_off() {
        let (_dir, engine) = db_engine();
        let m = engine.match_input("关闭空调");
        assert!(matches!(m, StrMatchDecision::Match { .. }));
        if let StrMatchDecision::Match { rule } = m {
            assert_eq!(rule.tool, "ac.power");
            assert_eq!(rule.args_json, r#"{"action":"off"}"#);
        }
    }

    #[test]
    fn llm_override_detected() {
        let engine = StringMatchEngine::empty();
        let m = engine.match_input("用大模型帮我查一下");
        assert!(matches!(m, StrMatchDecision::LlmOverride { .. }));
    }

    #[test]
    fn no_match_returns_nomatch() {
        let engine = StringMatchEngine::empty();
        let m = engine.match_input("xyznonexistent123");
        assert!(matches!(m, StrMatchDecision::NoMatch));
    }

    #[test]
    fn empty_engine_all_nomatch() {
        let engine = StringMatchEngine::empty();
        assert!(matches!(engine.match_input("查内存"), StrMatchDecision::NoMatch));
    }

    #[test]
    fn tool_whitelist_rejects_unknown_in_system() {
        let r = DbRule {
            id: 1, namespace: "system".into(), name: "bad".into(),
            pattern: "x".into(), tool: "rm_rf".into(), args_json: "{}".into(),
            description: None, priority: 100, updated_at: 0,
        };
        assert!(check_rule_safety(&r, "system").is_err());
    }

    #[test]
    fn tool_whitelist_allows_unknown_in_user_namespace() {
        let r = DbRule {
            id: 1, namespace: "default".into(), name: "custom".into(),
            pattern: "x".into(), tool: "rm_rf".into(), args_json: "{}".into(),
            description: None, priority: 100, updated_at: 0,
        };
        assert!(check_rule_safety(&r, "default").is_ok());
    }

    #[test]
    fn args_json_must_be_valid() {
        let bad = DbRule {
            id: 1,
            namespace: "test".into(),
            name: "broken".into(),
            pattern: "x".into(),
            tool: "shell".into(),
            args_json: r#"{"unbalanced"#.into(),
            description: None,
            priority: 100,
            updated_at: 0,
        };
        assert!(check_rule_safety(&bad, "system").is_err());
    }
}
