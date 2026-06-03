//! prompt-cache-build-db：离线建库工具（Rust 版）。
//!
//! 流程：
//!   1. 用 LlamaCppEmbedder 加载 .gguf
//!   2. 解析 rules.json → INSERT rules + embed → rule_embeddings
//!   3. 可选：--l1-db 灌 L1 规则到 l1_string_match.db
//!   4. 写 meta.embedding_model
//!
//! 用法:
//!   # 灌 L2 向量规则
//!   cargo run -p zeroclaw-prompt-cache --bin prompt-cache-build-db -- \
//!     --db l2_vector.db \
//!     --rules-json rules.json \
//!     --model models/embedding/bge-small-zh-v1.5/bge-small-zh-v1.5-q4_k_m.gguf \
//!     --model-id bge-small-zh-v1.5-q4_k_m \
//!     [--pooling cls|mean|last]   default: cls
//!
//!   # 灌 L1 字符串匹配规则（不需要模型）
//!   cargo run -p zeroclaw-prompt-cache --bin prompt-cache-build-db -- \
//!     --l1-db l1_string_match.db \
//!     --rules-json l1_rules.json

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use llama_cpp_2::context::params::LlamaPoolingType;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use zeroclaw_prompt_cache::{
    Embedder, LlamaCppEmbedder, LlamaEmbedderConfig,
    prompt_matcher::{L2Thresholds, PromptMatcher},
    schema,
    string_match_db,
};

#[derive(Debug, Deserialize, Serialize)]
struct InputRule {
    pattern: String,
    #[serde(default = "default_kind")]
    kind: String,
    tool_name: String,
    #[serde(default = "default_args")]
    tool_args: serde_json::Value,
    #[serde(default)]
    tool_version: String,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    priority: i64,
    #[serde(default)]
    learnable: bool,
}

fn default_kind() -> String {
    "Intent".into()
}
fn default_args() -> serde_json::Value {
    serde_json::json!({})
}

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
    let mut rules_json: Option<PathBuf> = None;
    let mut model_path: Option<PathBuf> = None;
    let mut model_id: Option<String> = None;
    let mut pooling = LlamaPoolingType::Cls;
    let mut n_ctx: u32 = 512;
    let mut l1_db: Option<PathBuf> = None;
    let mut negative_samples_json: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--db" => db_path = args.next().map(PathBuf::from),
            "--rules-json" => rules_json = args.next().map(PathBuf::from),
            "--model" => model_path = args.next().map(PathBuf::from),
            "--model-id" => model_id = args.next(),
            "--l1-db" => l1_db = args.next().map(PathBuf::from),
            "--negative-samples" => negative_samples_json = args.next().map(PathBuf::from),
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
            "-h" | "--help" => {
                eprintln!(
                    "usage: prompt-cache-build-db --db PATH --rules-json PATH \
                     --model PATH --model-id ID [--pooling cls|mean] [--n-ctx 512]\n\
                     \n\
                     L1 规则灌库（不需要模型）：\n  \
                       prompt-cache-build-db --l1-db l1_string_match.db --rules-json l1_rules.json\n\
                     L2 向量灌库（需要模型）：\n  \
                       prompt-cache-build-db --db l2_vector.db --rules-json rules.json --model MODEL --model-id ID"
                );
                return Ok(());
            }
            other => bail!("unknown arg: {other}"),
        }
    }

    // ── L1 灌库模式：不需要模型，从同一份 rules.json 里取 regex 字段 ──
    if let Some(ref l1_path) = l1_db {
        let rules_json = rules_json.as_ref().context("--rules-json required for --l1-db")?;
        let rules_text = std::fs::read_to_string(rules_json)
            .with_context(|| format!("read l1 rules json: {}", rules_json.display()))?;
        let rules: Vec<serde_json::Value> =
            serde_json::from_str(&rules_text).context("parse l1 rules json")?;

        let conn = string_match_db::open(l1_path)?;
        let mut seeded = 0;
        for r in &rules {
            // 跳过没有 regex 字段的规则（如 ParameterizedIntent）
            let regex = match r["regex"].as_str() {
                Some(re) => re,
                None => continue,
            };
            let tool_name = r["tool_name"].as_str().context("rule missing 'tool_name'")?;
            let name = r["pattern"].as_str().context("rule missing 'pattern'")?
                .replace(' ', "_").to_lowercase();
            let args_json = &r["tool_args"];
            let args_str = serde_json::to_string(args_json)?;
            let description = r["description"].as_str();
            let priority = r["priority"].as_i64().unwrap_or(100);

            if let Err(e) = regex::Regex::new(regex) {
                eprintln!("  SKIP {name}: invalid regex: {e}");
                continue;
            }

            match string_match_db::insert_rule(
                &conn,
                &string_match_db::NewRule {
                    namespace: "system",
                    name: &name,
                    pattern: regex,
                    tool: tool_name,
                    args_json: &args_str,
                    description,
                    priority,
                },
            ) {
                Ok(id) => {
                    seeded += 1;
                    eprintln!("  +{id:>3} {name} → {tool_name}");
                }
                Err(e) => eprintln!("  SKIP {name}: {e}"),
            }
        }
        eprintln!("[build_db] L1: {seeded} rules seeded into {}", l1_path.display());
    }

    // ── L2 灌库模式：需要模型 ─────────────────────────────────
    if l1_db.is_some() && db_path.is_none() {
        return Ok(());
    }
    let db_path = db_path.context("--db required (or use --l1-db for L1-only)")?;
    let rules_json = rules_json.context("--rules-json required")?;
    let model_path = model_path.context("--model required")?;
    let model_id = model_id.context("--model-id required")?;

    let rules_text = std::fs::read_to_string(&rules_json)
        .with_context(|| format!("read rules json: {}", rules_json.display()))?;
    let rules: Vec<InputRule> = serde_json::from_str(&rules_text).context("parse rules json")?;

    eprintln!(
        "[build_db] db={} rules={} model={}",
        db_path.display(),
        rules.len(),
        model_path.display()
    );

    // build embedder（在开 cache 之前，方便先发现模型加载错误）
    let embedder_cfg = LlamaEmbedderConfig {
        model_path: model_path.clone(),
        model_id: model_id.clone(),
        n_ctx,
        n_threads: num_cpus_or(2),
        pooling,
    };
    let embedder = Arc::new(LlamaCppEmbedder::new(embedder_cfg)?);
    eprintln!(
        "[build_db] embedder ready: model_id={} dim={}",
        embedder.model_id(),
        embedder.dim()
    );

    // 把 db 文件先建出来 + 应用 schema（不接 embedder，因为这里要写 meta 之后才符合校验）
    let cache = PromptMatcher::open(&db_path, "", L2Thresholds::default())
        .context("open prompt cache for build")?;

    // 写 embedding_model meta
    {
        let conn = cache.raw_conn();
        schema::set_embedding_model(&conn, &model_id)?;
    }

    // 灌 rules + embeddings —— 一个事务批量写
    {
        let mut conn = cache.raw_conn();
        let tx = conn.transaction()?;
        let now = unix_ts();

        for r in &rules {
            // rules
            tx.execute(
                "INSERT INTO rules
                    (pattern, kind, tool_name, tool_args, tool_version,
                     category, priority, learnable, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)",
                params![
                    r.pattern,
                    r.kind,
                    r.tool_name,
                    serde_json::to_string(&r.tool_args)?,
                    if r.tool_version.is_empty() {
                        "demo-v1".to_string()
                    } else {
                        r.tool_version.clone()
                    },
                    r.category,
                    r.priority,
                    if r.learnable { 1 } else { 0 },
                    now,
                ],
            )?;
            let rule_id = tx.last_insert_rowid();

            // embedding
            let v = embedder.embed(&r.pattern)?;
            let mut blob = Vec::with_capacity(v.len() * 4);
            for x in &v {
                blob.extend_from_slice(&x.to_le_bytes());
            }
            tx.execute(
                "INSERT INTO rule_embeddings (rule_id, dim, model, vec)
                 VALUES (?1, ?2, ?3, ?4)",
                params![rule_id, embedder.dim() as i64, model_id, blob],
            )?;

            eprintln!("  +{rule_id:>3} {} → {}", r.pattern, r.tool_name);
        }
        tx.commit()?;
    }

    eprintln!("[build_db] done. {} rules embedded.", rules.len());

    // ── 可选：灌 negative_samples（regex 晋升时验证不命中用）──
    if let Some(neg_path) = negative_samples_json.as_ref() {
        let text = std::fs::read_to_string(neg_path)
            .with_context(|| format!("read negative samples: {}", neg_path.display()))?;
        let neg: Vec<NegSample> = serde_json::from_str(&text)
            .context("parse negative_samples json (array of {prompt, note?})")?;
        let conn = cache.raw_conn();
        let now = unix_ts();
        let mut tx_inserted = 0;
        for s in &neg {
            // 跳过空白条目
            if s.prompt.trim().is_empty() {
                continue;
            }
            match conn.execute(
                "INSERT OR IGNORE INTO negative_samples (prompt, note, created_at)
                 VALUES (?1, ?2, ?3)",
                params![s.prompt, s.note, now],
            ) {
                Ok(n) => tx_inserted += n,
                Err(e) => eprintln!("  neg SKIP '{}': {e}", s.prompt),
            }
        }
        eprintln!(
            "[build_db] negative_samples: {} new (of {}) inserted into {}",
            tx_inserted,
            neg.len(),
            db_path.display()
        );
    }

    Ok(())
}

#[derive(serde::Deserialize)]
struct NegSample {
    prompt: String,
    #[serde(default)]
    note: Option<String>,
}

fn unix_ts() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn num_cpus_or(fallback: i32) -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(fallback)
}
