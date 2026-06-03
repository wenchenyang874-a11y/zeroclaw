//! prompt-cache-cli：三级匹配 demo（phase 2 版，含 L2 in-process）。
//!
//! 用法：
//!     cargo run -p zeroclaw-prompt-cache --bin prompt-cache-cli -- \
//!       --db l2_vector.db \
//!       --model models/embedding/bge-small-zh-v1.5/bge-small-zh-v1.5-q4_k_m.gguf \
//!       --model-id bge-small-zh-v1.5-q4_k_m \
//!       [--pooling cls|mean]   default: cls
//!       [--no-l2]              强制只跑 L1，不加载模型
//!
//! stdin 一行一个查询，stdout 打印命中：
//!     [L1] exact   id=1 tool=ac_power
//!     [L1] fts     id=2 bm25=-1.234 tool=...
//!     [L2] cosine  id=3 cos=0.78 tool=...
//!     [L3] mock dispatch (would call cloud LLM)

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use llama_cpp_2::context::params::LlamaPoolingType;
use zeroclaw_prompt_cache::{
    LlamaCppEmbedder, LlamaEmbedderConfig, MatchLevel, PromptCache,
    prompt_matcher::{L2Thresholds, PromptMatcher},
};

fn parse_pooling(s: &str) -> Result<LlamaPoolingType> {
    match s.to_ascii_lowercase().as_str() {
        "cls" => Ok(LlamaPoolingType::Cls),
        "mean" => Ok(LlamaPoolingType::Mean),
        "last" => Ok(LlamaPoolingType::Last),
        "none" => Ok(LlamaPoolingType::None),
        other => bail!("unknown pooling: {other}"),
    }
}

fn main() -> Result<()> {
    let mut db_path: Option<PathBuf> = None;
    let mut model_path: Option<PathBuf> = None;
    let mut model_id: Option<String> = None;
    let mut pooling = LlamaPoolingType::Cls;
    let mut n_ctx: u32 = 64;
    let mut no_l2 = false;
    // --seed-pending PROMPT TOOL_NAME TOOL_ARGS_JSON：手动塞一条 pending_rules
    // 用于在没接 L3 自动 record_learning 的 V4 阶段测试 PromotionWorker
    let mut seed_pending: Option<(String, String, String)> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--db" => db_path = args.next().map(PathBuf::from),
            "--model" => model_path = args.next().map(PathBuf::from),
            "--model-id" => model_id = args.next(),
            "--pooling" => {
                let v = args.next().context("--pooling needs a value")?;
                pooling = parse_pooling(&v)?;
            }
            "--n-ctx" => {
                n_ctx = args
                    .next()
                    .context("--n-ctx needs a value")?
                    .parse()
                    .context("--n-ctx must be u32")?;
            }
            "--no-l2" => no_l2 = true,
            "--seed-pending" => {
                let prompt = args.next().context("--seed-pending PROMPT TOOL_NAME ARGS_JSON")?;
                let tool_name = args.next().context("--seed-pending PROMPT TOOL_NAME ARGS_JSON")?;
                let args_json = args.next().context("--seed-pending PROMPT TOOL_NAME ARGS_JSON")?;
                seed_pending = Some((prompt, tool_name, args_json));
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: prompt-cache-cli --db PATH \
                     [--model PATH --model-id ID --pooling cls|mean] [--no-l2]\n\
                     \n\
                     测试 V4 PromotionWorker：手动塞一条 pending_rules (跑两次累 occurrences)\n\
                       prompt-cache-cli --db /sqlite/data/l2_vector.db \\\n\
                         --seed-pending '太热了，开下空调' 'ac.power' '{{\"action\":\"on\"}}'\n\
                     然后用一句中性话再触发反悔窗口关闭（让 worker 看到 confirmed_at>0）：\n\
                     ——这部分目前需要在 zeroclaw turn 流程里走，或者直接 SQL UPDATE confirmed_at"
                );
                return Ok(());
            }
            other => bail!("unknown arg: {other}"),
        }
    }

    let db_path = db_path.context("--db required")?;

    // ── --seed-pending 模式：单次插入后退出，不进交互循环 ───────────
    if let Some((prompt, tool_name, args_json)) = seed_pending {
        let cache = PromptMatcher::open(&db_path, "", L2Thresholds::default())
            .context("open prompt cache")?;
        let args_value: serde_json::Value =
            serde_json::from_str(&args_json).context("--seed-pending TOOL_ARGS_JSON 不是合法 JSON")?;
        let tc = zeroclaw_prompt_cache::ToolCall {
            name: tool_name.clone(),
            args: args_value,
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let pending_id = rt
            .block_on(zeroclaw_prompt_cache::PromptCache::record_learning(
                &cache, &prompt, &tc, "seed-v0",
            ))
            .context("record_learning")?;
        eprintln!(
            "[seed] pending_rules row id={} prompt={:?} tool={} args={}",
            pending_id, prompt, tool_name, args_json
        );
        eprintln!(
            "       注意：要让 PromotionWorker 真正处理它，confirmed_at 必须 > 0\n\
                    （正常流程下次用户输入到达时会自动标）。手动调试可：\n\
                    sqlite3 {} \"UPDATE pending_rules SET confirmed_at = strftime('%s','now') WHERE id={};\"",
            db_path.display(),
            pending_id
        );
        return Ok(());
    }

    // 没 --no-l2 且给了 model 才挂 embedder
    let want_embedder = !no_l2 && model_path.is_some();
    let runtime_model_id = model_id.clone().unwrap_or_default();

    let cache = PromptMatcher::open(&db_path, &runtime_model_id, L2Thresholds::default())
        .context("open prompt cache")?;

    let cache = if want_embedder {
        let cfg = LlamaEmbedderConfig {
            model_path: model_path.unwrap(),
            model_id: model_id.unwrap_or_else(|| "unknown".into()),
            n_ctx,
            n_threads: 2,
            pooling,
        };
        let emb = Arc::new(LlamaCppEmbedder::new(cfg).context("init embedder")?);
        cache.with_embedder(emb).context("attach embedder")?
    } else {
        cache
    };

    eprintln!(
        "[cli] db={} rules={} vec_dim={} L2={}",
        db_path.display(),
        cache.rule_count(),
        cache.vec_dim(),
        if no_l2 { "off" } else { "on" }
    );
    eprintln!(
        "[cli] L2 thresholds: accept>={} reject<{}",
        cache.l2.accept, cache.l2.reject
    );
    eprintln!("[cli] type a query and ENTER (Ctrl-D 退出)");

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    for line in stdin.lock().lines() {
        let q = line?;
        let q = q.trim();
        if q.is_empty() {
            continue;
        }
        writeln!(out, "> {q}")?;
        let res = rt.block_on(cache.try_match(q))?;
        match res {
            Some(m) => match m.level {
                MatchLevel::L1Regex => writeln!(
                    out,
                    "  [L1] regex   id={} pattern=\"{}\" tool={}",
                    m.rule_id, m.pattern, m.tool_call.name
                )?,
                MatchLevel::L2Cosine => {
                    let top2 = m
                        .top2_score
                        .map(|c| format!(" top2={c:.4}"))
                        .unwrap_or_default();
                    writeln!(
                        out,
                        "  [L2] cosine  id={} pattern=\"{}\" cos={:.4}{} tool={}",
                        m.rule_id,
                        m.pattern,
                        m.score.unwrap_or(0.0),
                        top2,
                        m.tool_call.name
                    )?;
                }
            },
            None => writeln!(out, "  [L3] mock dispatch (would call cloud LLM)")?,
        }
        out.flush()?;
    }
    Ok(())
}
