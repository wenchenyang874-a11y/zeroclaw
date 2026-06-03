//! `zeroclaw l1` 子命令——管理 L1 字符串匹配规则（l1_string_match.db）。
//!
//! 改完规则后需重启 zeroclaw 才能生效。

use anyhow::{Context, Result};
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use zeroclaw_config::schema::StringMatchConfig;
use zeroclaw_prompt_cache::string_match_db;

fn resolve_db_path(cfg: &StringMatchConfig, override_path: Option<PathBuf>) -> PathBuf {
    if let Some(p) = override_path { p }
    else { zeroclaw_config::policy::expand_user_path(&cfg.db_path) }
}

fn open_db(path: &Path) -> Result<Connection> {
    string_match_db::open(path)
        .with_context(|| format!("opening L1 db at {}", path.display()))
}

fn fmt_ts(unix: i64) -> String {
    if unix == 0 { return "-".into(); }
    chrono::DateTime::from_timestamp(unix, 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| "-".into())
}

pub fn handle_list(cfg: &StringMatchConfig, namespace: Option<String>, db: Option<PathBuf>) -> Result<()> {
    let path = resolve_db_path(cfg, db);
    let conn = open_db(&path)?;
    let rules = string_match_db::list_all_rules_with_enabled(&conn, namespace.as_deref())?;

    if rules.is_empty() {
        println!("(no rules)");
        println!("DB: {}", path.display());
        return Ok(());
    }
    println!("{:>4}  {:<10} {:<22} {:<14} {:>5} {:>7}  {:<14}  {}",
        "id", "ns", "name", "tool", "prio", "enabled", "updated", "description");
    println!("{}", "-".repeat(100));
    for (r, enabled) in &rules {
        let desc = r.description.as_deref().unwrap_or("-");
        println!("{:>4}  {:<10} {:<22} {:<14} {:>5} {:>7}  {:<14}  {}",
            r.id, r.namespace, r.name, r.tool, r.priority,
            if *enabled { "yes" } else { "no" },
            fmt_ts(r.updated_at), desc,
        );
    }
    Ok(())
}

pub fn handle_add(
    cfg: &StringMatchConfig,
    name: String,
    pattern: String,
    tool: String,
    args: String,
    description: Option<String>,
    priority: Option<i64>,
    namespace: Option<String>,
    db: Option<PathBuf>,
) -> Result<()> {
    // 校验
    regex::Regex::new(&pattern).context("--pattern is not a valid regex")?;
    serde_json::from_str::<serde_json::Value>(&args).context("--args is not valid JSON")?;

    let ns = namespace.as_deref().unwrap_or(&cfg.namespace);
    if ns == "system" {
        anyhow::bail!("cannot manually add rules to 'system' namespace");
    }

    let path = resolve_db_path(cfg, db);
    let conn = open_db(&path)?;
    let id = string_match_db::insert_rule(&conn, &string_match_db::NewRule {
        namespace: ns,
        name: &name,
        pattern: &pattern,
        tool: &tool,
        args_json: &args,
        description: description.as_deref(),
        priority: priority.unwrap_or(50),
    })?;
    println!("+{} {} → {}", id, name, tool);
    println!("  restart zeroclaw for the new rule to take effect");
    Ok(())
}

pub fn handle_remove(
    cfg: &StringMatchConfig,
    name: String,
    namespace: Option<String>,
    db: Option<PathBuf>,
) -> Result<()> {
    let ns = namespace.as_deref().unwrap_or(&cfg.namespace);
    if ns == "system" {
        anyhow::bail!("cannot remove rules from 'system' namespace");
    }
    let path = resolve_db_path(cfg, db);
    let conn = open_db(&path)?;
    if string_match_db::delete_rule(&conn, ns, &name)? {
        println!("removed '{name}' from namespace '{ns}'");
    } else {
        println!("no rule named '{name}' in namespace '{ns}'");
    }
    Ok(())
}

pub fn handle_set_enabled(
    cfg: &StringMatchConfig,
    name: String,
    enabled: bool,
    namespace: Option<String>,
    db: Option<PathBuf>,
) -> Result<()> {
    let ns = namespace.as_deref().unwrap_or(&cfg.namespace);
    let path = resolve_db_path(cfg, db);
    let conn = open_db(&path)?;
    if string_match_db::set_rule_enabled(&conn, ns, &name, enabled)? {
        println!("{} '{name}' in namespace '{ns}'", if enabled { "enabled" } else { "disabled" });
    } else {
        println!("no rule named '{name}' in namespace '{ns}'");
    }
    Ok(())
}
