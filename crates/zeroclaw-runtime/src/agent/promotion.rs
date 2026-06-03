//! PromotionWorker —— 自学习闭环的"晋升"环节。
//!
//! 后台 tokio task，定期从 [`PromptMatcher`] 拉一批已通过反悔窗口、occurrences 达
//! 阈值的 `pending_rules` 行，调云端 LLM 生成 regex，跑验证管线，通过则把它晋升到
//! `rules` 表（match_kind='regex'），失败则标 `rejected`。
//!
//! 设计文档对齐：
//! - 三级匹配落地方案选型与权衡.md §6 自学习闭环 (Q4-Q6)
//! - 三级匹配zeroclaw实装与板端部署.md "已知边界" 章节
//!
//! 验证管线（每条候选 regex 必须全过）：
//! 1. 能用 `regex::Regex::new` 编过
//! 2. 编出的 regex 命中原 prompt（防 LLM 幻觉）
//! 3. 不命中任何 `negative_samples`（防误伤）

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::broadcast;
use zeroclaw_providers::{ChatMessage, ChatRequest, Provider};

use zeroclaw_prompt_cache::{PendingRule, PromptMatcher, ValidationState};

/// 晋升事件：单条 (prompt, tool_call) 从 pending 提升到 L1 string_match_rules 表
/// 并 hot-reload 进内存引擎，由 PromotionWorker 通过 broadcast 发出。
/// loop_.rs / channel adapter 订阅以向用户打提示。
#[derive(Debug, Clone)]
pub struct PromotionEvent {
    pub prompt: String,
    pub tool_name: String,
    pub regex: String,
    pub elapsed_ms: u128,
}

/// 后台启动 PromotionWorker。返回 JoinHandle 方便上层 abort。
/// `event_tx` 可选——传 `None` 表示不广播提示（headless / 单测）。
pub fn spawn_promotion_worker(
    cache: Arc<PromptMatcher>,
    provider: Arc<dyn Provider>,
    model: String,
    poll_secs: u64,
    threshold_n: i64,
    event_tx: Option<broadcast::Sender<PromotionEvent>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(poll_secs.max(5)));
        // 第一次 tick 是即刻，跳过等真正间隔再开始
        ticker.tick().await;
        loop {
            ticker.tick().await;
            if let Err(e) = run_one_cycle(
                &cache,
                provider.as_ref(),
                &model,
                threshold_n,
                event_tx.as_ref(),
            )
            .await
            {
                tracing::warn!(error = %e, "PromotionWorker cycle failed; will retry next tick");
            }
        }
    })
}

/// 跑一轮：拉一批候选，逐个走"调 LLM → 验证 → 晋升 or 拒绝"。
async fn run_one_cycle(
    cache: &PromptMatcher,
    provider: &dyn Provider,
    model: &str,
    threshold_n: i64,
    event_tx: Option<&broadcast::Sender<PromotionEvent>>,
) -> Result<()> {
    let candidates = cache.iter_pending_for_promotion(threshold_n, 8)?;
    if candidates.is_empty() {
        return Ok(());
    }
    tracing::debug!(count = candidates.len(), "PromotionWorker: candidates to process");
    println!(
        "{}发现 {} 条候选: {}",
        zeroclaw_prompt_cache::log_tag("学习晋升", "1;35"),
        candidates.len(),
        candidates
            .iter()
            .map(|c| format!("\"{}\"→{}", c.prompt, c.tool_name))
            .collect::<Vec<_>>()
            .join(", ")
    );

    for cand in candidates {
        process_one(cache, provider, model, &cand, event_tx).await;
    }
    Ok(())
}

async fn process_one(
    cache: &PromptMatcher,
    provider: &dyn Provider,
    model: &str,
    cand: &PendingRule,
    event_tx: Option<&broadcast::Sender<PromotionEvent>>,
) {
    let started = Instant::now();
    // 标记 generating
    if let Err(e) = cache.mark_pending_state(cand.id, ValidationState::Generating, None) {
        tracing::warn!(error = %e, id = cand.id, "mark_pending_state(generating) failed");
        return;
    }

    println!(
        "{}为 \"{}\" 生成正则（{}）…",
        zeroclaw_prompt_cache::log_tag("学习晋升", "1;35"),
        cand.prompt, model
    );
    let (regex, paraphrases) = match generate_regex(provider, model, cand).await {
        Ok((re, para)) => {
            println!(
                "{}正则已生成 regex={} paraphrases={}",
                zeroclaw_prompt_cache::log_tag("学习晋升", "1;35"),
                re,
                para.join("|")
            );
            (re, para)
        }
        Err(e) => {
            let log = format!("LLM regex generation failed: {e}");
            let _ = cache.mark_pending_state(cand.id, ValidationState::Rejected, Some(&log));
            println!("{}正则生成失败: {e}", zeroclaw_prompt_cache::log_tag("学习晋升✗", "1;31"));
            return;
        }
    };

    // 验证管线（依次走完才会 promote，任何一关失败就 reject）：
    //   1. 结构闸：regex 不能太空泛（防欠拟合，纯 .* 之类）
    //   2. 编译 OK
    //   3. 命中原 prompt（LLM 没幻觉）
    //   4. 命中所有 LLM 同时给出的同义改写（泛化下限 - 防过拟合到字面）
    //   5. 不命中 negative_samples（特异性上限 - 防欠拟合到太宽）
    let _ = cache.mark_pending_state(cand.id, ValidationState::Validating, None);
    if let Err(reason) = check_regex_structure(&regex) {
        let log = format!("regex structure check failed: {reason} | proposed='{regex}'");
        let _ = cache.mark_pending_state(cand.id, ValidationState::Rejected, Some(&log));
        println!("{}放弃：正则过于宽泛（{reason}），不下沉 L1", zeroclaw_prompt_cache::log_tag("学习晋升✗", "1;31"));
        tracing::debug!(id = cand.id, reason = %reason, "regex too broad (structure)");
        return;
    }
    let compiled = match regex::Regex::new(&regex) {
        Ok(c) => c,
        Err(e) => {
            let log = format!("regex compile failed: {e} | proposed='{regex}'");
            let _ = cache.mark_pending_state(cand.id, ValidationState::Rejected, Some(&log));
            println!("{}放弃：正则编译失败", zeroclaw_prompt_cache::log_tag("学习晋升✗", "1;31"));
            tracing::debug!(error = %e, id = cand.id, "regex compile failed");
            return;
        }
    };
    if !compiled.is_match(&cand.prompt) {
        let log = format!("regex didn't match original prompt | proposed='{regex}'");
        let _ = cache.mark_pending_state(cand.id, ValidationState::Rejected, Some(&log));
        println!("{}放弃：正则匹配不上原句（LLM 幻觉）", zeroclaw_prompt_cache::log_tag("学习晋升✗", "1;31"));
        tracing::debug!(id = cand.id, "regex didn't match its own training prompt");
        return;
    }
    // 同义改写校验：每条 LLM 给的改写都必须命中 → 防过拟合到字面表达
    for p in &paraphrases {
        if p.trim().is_empty() {
            continue;
        }
        if !compiled.is_match(p) {
            let log = format!(
                "regex didn't generalize to paraphrase '{}' | proposed='{}'",
                p.replace('\n', " "),
                regex
            );
            let _ = cache.mark_pending_state(cand.id, ValidationState::Rejected, Some(&log));
            println!("{}放弃：正则过拟合（改写未命中），不下沉 L1", zeroclaw_prompt_cache::log_tag("学习晋升✗", "1;31"));
            tracing::debug!(id = cand.id, paraphrase = %p, "regex overfits (paraphrase miss)");
            return;
        }
    }
    match cache.check_regex_against_negative_samples(&compiled) {
        Ok(Some(neg)) => {
            let log = format!(
                "regex误伤负样本 '{}' | proposed='{}'",
                neg.replace('\n', " "),
                regex
            );
            let _ = cache.mark_pending_state(cand.id, ValidationState::Rejected, Some(&log));
            println!("{}放弃：正则误伤负样本（欠特异），不下沉 L1", zeroclaw_prompt_cache::log_tag("学习晋升✗", "1;31"));
            tracing::debug!(id = cand.id, sample = %neg, "regex hits negative sample");
            return;
        }
        Ok(None) => {}
        Err(e) => {
            let log = format!("negative-sample check failed: {e}");
            let _ = cache.mark_pending_state(cand.id, ValidationState::Rejected, Some(&log));
            return;
        }
    }

    // 晋升
    match cache.promote_pending_to_rule(cand.id, &regex) {
        Ok(rule_id) => {
            tracing::debug!(
                pending_id = cand.id,
                rule_id,
                tool = %cand.tool_name,
                regex = %regex,
                "PromotionWorker: pending → rules.regex"
            );
            // 关键：把新规则 hot-reload 进内存 L1 引擎，无需重启
            if let Err(e) = cache.reload_string_match() {
                tracing::warn!(error = %e, rule_id, "reload_string_match failed; rule persisted but L1 sees it only after restart");
            } else {
                let elapsed_ms = started.elapsed().as_millis();
                tracing::debug!(rule_id, elapsed_ms, "L1 engine reloaded with new rule");
                // 广播提示给前台（REPL / channel adapter）
                if let Some(tx) = event_tx {
                    let _ = tx.send(PromotionEvent {
                        prompt: cand.prompt.clone(),
                        tool_name: cand.tool_name.clone(),
                        regex: regex.clone(),
                        elapsed_ms,
                    });
                }
            }
        }
        Err(e) => {
            let log = format!("promote insert failed: {e}");
            let _ = cache.mark_pending_state(cand.id, ValidationState::Rejected, Some(&log));
            tracing::warn!(error = %e, id = cand.id, "promote_pending_to_rule failed");
        }
    }
}

/// 让云端 LLM 给候选 (prompt, tool_call) 同时给出 regex + 同义改写。
/// 改写用于 process_one 的"泛化下限"校验：regex 必须命中所有 LLM 给出的同义说法，
/// 否则视为过拟合到字面、不能下沉到 L1。
async fn generate_regex(
    provider: &dyn Provider,
    model: &str,
    cand: &PendingRule,
) -> Result<(String, Vec<String>)> {
    let system_prompt = SYSTEM_PROMPT;
    let user_payload = format!(
        "用户原话: {}\n命中的工具: {}\n工具参数: {}\n\n请按要求输出 regex + 同义改写。",
        cand.prompt,
        cand.tool_name,
        serde_json::to_string(&cand.tool_args).unwrap_or_else(|_| "{}".into()),
    );
    let messages = vec![
        ChatMessage::system(system_prompt),
        ChatMessage::user(&user_payload),
    ];
    let resp = provider
        .chat(
            ChatRequest {
                messages: &messages,
                tools: None,
            },
            model,
            0.0,
        )
        .await
        .context("provider.chat for regex generation")?;
    let text = resp.text.unwrap_or_default();
    extract_regex_from_response(&text).context("no usable regex in LLM response")
}

const SYSTEM_PROMPT: &str = r#"你是一个正则表达式工程师。用户会给你"一句话+对应的工具调用"，你需要给这句话写一条 Rust regex crate 兼容的正则（无 backreference / 无 lookaround），用于今后**离线匹配同类意图**的中文表达。

要求：
1. **必须覆盖原句**——给的 regex 必须能命中用户提供的那一句。
2. **泛化得够**：覆盖同义表达和常见中文变体（动作词换位、加礼貌词如"麻烦/帮我/能不能"、上下文铺垫如"我家的/卧室的"、口语化"看看/看一眼"）。
3. **特异性够**：不能把**别的意图**的句子误命中——比如"打开空调"的 regex 不能命中"关闭空调"或"调高空调温度"。
4. **不要纯通配**：禁止 `.*` / `^.*$` / `.+` 这种空泛 regex。必须包含至少一个意图关键词的字面或合理替换组。
5. **相邻组不能共用字符**：如果 regex 是 `(A组).*?(B组)` 的结构（如地点词 + 动作/人），
   A 组的每个备选词末尾不能与 B 组的备选词开头相同——否则"屋里有|有人没"在 "屋里有人没"
   这个输入上会卡住（"有"不能同时属于两边）。解决方案：地点词去掉末尾"有"（`屋里|房间`），
   人称词去掉开头"有"（`人吗|人没|没有人`）。
6. **同时给出 3 条同义改写**：用于回归校验你写的 regex 真的能覆盖同类表达；改写本身要自然、贴近真实用户说法。

**输出格式（严格 JSON，可以用 ```json 围栏包裹）**：
```json
{
  "regex": "...",
  "paraphrases": ["改写1", "改写2", "改写3"]
}
```

例：
输入：用户原话 "把空调打开"，工具 ac.power(on)
输出：
```json
{
  "regex": "^(?:.*?(?:打开|开下|开个|帮.*?开|麻烦.*?开|能.*?开).*?(?:空调|冷气)|.*?(?:空调|冷气).*?(?:打开|开下|开个|帮.*?开)).*?$",
  "paraphrases": ["帮我把空调开一下", "麻烦开下空调", "能不能开个空调"]
}
```"#;

/// 结构闸：禁止显然过宽的 regex（防欠拟合）。
///
/// 规则（任一触发即拒）：
/// - 整体仅由 `.*` / `.+` / `^` / `$` / 空白构成
/// - 不含任何长度 ≥ 2 的字面 token（regex 里非元字符的连续片段）
fn check_regex_structure(re: &str) -> std::result::Result<(), &'static str> {
    let trimmed = re.trim();
    // 一些显然 garbage 的形式直接拒
    let stripped = trimmed.replace(['^', '$', ' '], "");
    if matches!(stripped.as_str(), ".*" | ".+" | ".*?" | ".+?" | "") {
        return Err("regex 是纯通配，不允许");
    }
    // 提取长度 ≥ 2 的字面 token：剥掉常见元字符，看剩下连续字符段
    // 这是粗略检查，但能挡掉 \w+ / [一-鿿]+ 这种"全是字符集"无字面意图词的写法
    let mut buf = String::new();
    let mut had_long_literal = false;
    let mut in_class = false;
    let mut escape = false;
    for ch in trimmed.chars() {
        if escape {
            // \w \d \s \. 等 → 不算字面词
            escape = false;
            if buf.chars().count() >= 2 {
                had_long_literal = true;
                break;
            }
            buf.clear();
            continue;
        }
        match ch {
            '\\' => {
                escape = true;
                if buf.chars().count() >= 2 {
                    had_long_literal = true;
                    break;
                }
                buf.clear();
            }
            '[' => {
                in_class = true;
                buf.clear();
            }
            ']' => {
                in_class = false;
                buf.clear();
            }
            '(' | ')' | '|' | '?' | '*' | '+' | '^' | '$' | '{' | '}' | '.' | ' ' | '\t' => {
                if buf.chars().count() >= 2 {
                    had_long_literal = true;
                    break;
                }
                buf.clear();
            }
            _ if !in_class => buf.push(ch),
            _ => {}
        }
    }
    if !had_long_literal && buf.chars().count() < 2 {
        return Err("regex 缺少长度 ≥2 的字面意图词，泛化过度");
    }
    Ok(())
}

/// 从 LLM 自由文本里抽出 regex + paraphrases。
/// 兼容三种输出：```json{...}```、裸 {...}、上一代单字段 {"regex": "..."}。
fn extract_regex_from_response(text: &str) -> Option<(String, Vec<String>)> {
    // 1. ```json ... ``` 或 ``` ... ```
    if let Some(fence_start) = text.find("```") {
        let rest = &text[fence_start + 3..];
        // 跳过可能的 language tag
        let body_start = rest
            .find('\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        let body = &rest[body_start..];
        if let Some(fence_end) = body.find("```") {
            let json_str = body[..fence_end].trim();
            if let Some(r) = try_parse_regex_json(json_str) {
                return Some(r);
            }
        }
    }
    // 2. 找第一个 { 开始尝试
    if let Some(brace) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if end > brace {
                if let Some(r) = try_parse_regex_json(&text[brace..=end]) {
                    return Some(r);
                }
            }
        }
    }
    None
}

fn try_parse_regex_json(s: &str) -> Option<(String, Vec<String>)> {
    #[derive(serde::Deserialize)]
    struct Body {
        regex: String,
        #[serde(default)]
        paraphrases: Vec<String>,
    }
    serde_json::from_str::<Body>(s)
        .ok()
        .map(|b| (b.regex, b.paraphrases))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_from_json_fence_with_paraphrases() {
        let text = "好的。\n```json\n{\"regex\": \"^开.*?空调$\", \"paraphrases\": [\"开下空调\", \"麻烦把空调开了\"]}\n```";
        let (re, para) = extract_regex_from_response(text).unwrap();
        assert_eq!(re, "^开.*?空调$");
        assert_eq!(para, vec!["开下空调", "麻烦把空调开了"]);
    }

    #[test]
    fn extract_from_bare_json_no_paraphrases_field() {
        let text = "{\"regex\": \"^xyz$\"}";
        let (re, para) = extract_regex_from_response(text).unwrap();
        assert_eq!(re, "^xyz$");
        assert!(para.is_empty());
    }

    #[test]
    fn extract_nothing_useful() {
        assert!(extract_regex_from_response("just text no json").is_none());
    }

    #[test]
    fn structure_check_rejects_pure_wildcard() {
        for re in [".*", "^.*$", ".+", ".*?", "  ^.+$  "] {
            assert!(
                check_regex_structure(re).is_err(),
                "should reject pure wildcard: {re}"
            );
        }
    }

    #[test]
    fn structure_check_accepts_meaningful_regex() {
        for re in [
            "^.*?(?:打开|开下).*?空调.*?$",
            "看看画面",
            "(?:有没有|看一下).*?(?:人|画面)",
        ] {
            assert!(
                check_regex_structure(re).is_ok(),
                "should accept meaningful regex: {re}"
            );
        }
    }

    #[test]
    fn structure_check_rejects_only_char_class() {
        // 全字符集，没有字面意图词
        assert!(check_regex_structure(r"^[\w]+$").is_err());
    }
}
