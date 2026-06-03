use serde::{Deserialize, Serialize};

/// 哪一级命中。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchLevel {
    /// L1 字符串匹配命中
    L1Regex,
    /// L2 向量 cosine 命中
    L2Cosine,
}

/// 规则类型 —— 关系到 Q3 带参规则隐患的处理。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum RuleKind {
    /// 无参意图（"打开空调"）。L1/L2 可处理。
    Intent,
    /// 带参意图（"调到 28 度"）。v1 阶段 L1/L2 强制跳过，下放 L3，避免 §5 静默错配。
    ParameterizedIntent,
}

impl RuleKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RuleKind::Intent => "Intent",
            RuleKind::ParameterizedIntent => "ParameterizedIntent",
        }
    }
}

impl std::str::FromStr for RuleKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Intent" => Ok(RuleKind::Intent),
            "ParameterizedIntent" => Ok(RuleKind::ParameterizedIntent),
            other => Err(format!("unknown RuleKind: {other}")),
        }
    }
}

/// pending_rules.validation_state 列。详见 migrations/002_learning.sql 头部注释。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationState {
    /// 刚 record，等待 occurrences 累积或反悔窗口结束
    Pending,
    /// 正在调 LLM 生成 regex
    Generating,
    /// regex 已生成，跑验证管线中
    Validating,
    /// 验证失败（编译错 / 不匹配原 prompt / 冲突 / 误伤负样本）
    Rejected,
    /// 已写入 rules 表
    Promoted,
    /// 下一轮用户反悔，已撤销
    Withdrawn,
}

impl ValidationState {
    pub fn as_str(self) -> &'static str {
        match self {
            ValidationState::Pending => "pending",
            ValidationState::Generating => "generating",
            ValidationState::Validating => "validating",
            ValidationState::Rejected => "rejected",
            ValidationState::Promoted => "promoted",
            ValidationState::Withdrawn => "withdrawn",
        }
    }
}

impl std::str::FromStr for ValidationState {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(ValidationState::Pending),
            "generating" => Ok(ValidationState::Generating),
            "validating" => Ok(ValidationState::Validating),
            "rejected" => Ok(ValidationState::Rejected),
            "promoted" => Ok(ValidationState::Promoted),
            "withdrawn" => Ok(ValidationState::Withdrawn),
            other => Err(format!("unknown ValidationState: {other}")),
        }
    }
}

/// 一条等晋升的候选规则快照。PromotionWorker 拿这个去问 LLM 要 regex。
#[derive(Debug, Clone)]
pub struct PendingRule {
    pub id: i64,
    pub prompt: String,
    pub tool_name: String,
    pub tool_args: serde_json::Value,
    pub tool_version: String,
    pub occurrences: i64,
    pub recorded_at: i64,
}

/// 终点统一输出类型。`name` 是 zeroclaw 工具表里的工具名，`args` 是 JSON 对象。
///
/// 与 zeroclaw 现有的 ToolCall 在 phase 3 集成时再对齐（届时本类型会从 zeroclaw-api 借而非自建）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub args: serde_json::Value,
}

/// L1/L2 命中结果。
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub level: MatchLevel,
    pub rule_id: i64,
    pub pattern: String,
    pub tool_call: ToolCall,
    /// L1.1 = None；L1.2 = Some(bm25 rank)；L2 = Some(cosine)。
    pub score: Option<f32>,
    /// L2 only：top2 cosine（用来观察 margin，目前不参与判定）
    pub top2_score: Option<f32>,
}
