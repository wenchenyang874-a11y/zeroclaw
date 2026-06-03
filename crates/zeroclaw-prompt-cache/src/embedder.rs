//! In-process embedder：用 llama-cpp-2 直接加载 .gguf 模型做单条编码。
//!
//! 与 demo/ 那版 shell + subprocess 的差别：
//! - 不 fork 子进程，模型常驻内存，无冷启动开销
//! - 每次 `embed()` 复用同一个 `LlamaContext`，仅清空 KV cache
//! - 输出 L2 归一化向量，cosine 退化成点积（与 PromptMatcher 内存索引对齐）
//!
//! 板上风险：常驻 ~25MB（BGE Q4_K_M 15MB + KV cache + workspace）。
//! 板上 46MB available 下贴上限，必要时切回 subprocess（见 §5 选项 A）——
//! 接口已抽象成 `Embedder` trait，将来加 SubprocessEmbedder 不影响调用侧。

use std::path::Path;
use std::sync::Mutex;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::error::PromptCacheError;

/// 任意 embedding 后端。
pub trait Embedder: Send + Sync {
    /// 编码单条文本，返回 L2 归一化后的 dim 维向量。
    fn embed(&self, text: &str) -> Result<Vec<f32>, PromptCacheError>;
    /// 模型 id（与 db 里 prompt_cache_meta.embedding_model 比对）。
    fn model_id(&self) -> &str;
    /// 向量维度。
    fn dim(&self) -> usize;
}

/// llama.cpp 后端的 embedder 配置。
pub struct LlamaEmbedderConfig {
    pub model_path: std::path::PathBuf,
    /// 显式 model id，落库时写入 meta，运行期与 db 校验。
    /// 推荐用 `<model-name>-<quant>` 格式，例：`bge-small-zh-v1.5-q4_k_m`。
    pub model_id: String,
    /// 上下文窗口。板上 RAM 紧，BGE / MiniLM 短查询 64 够。
    pub n_ctx: u32,
    /// 推理线程数。Hi3516CV610 双核 A7 用 2，宿主机 build_db 用 nproc。
    pub n_threads: i32,
    /// pooling 类型。BGE 用 Cls；MiniLM 用 Mean。
    pub pooling: LlamaPoolingType,
}

impl Default for LlamaEmbedderConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::new(),
            model_id: String::new(),
            n_ctx: 64,
            n_threads: 2,
            pooling: LlamaPoolingType::Cls,
        }
    }
}

/// 进程内 llama.cpp embedder。
///
/// 线程安全策略：`LlamaContext` 不是 `Sync`，整个 context + batch 用 `Mutex` 包，
/// 一次 embed 拿一次锁。板上 query 间隔远大于单条编码耗时，锁竞争可忽略。
pub struct LlamaCppEmbedder {
    /// `LlamaBackend` 全局只允许 init 一次；持续到进程退出。
    /// 用 `'static` 借用让 LlamaContext 能持有它（详见 worker 字段）。
    _backend: &'static LlamaBackend,
    /// 模型本身（共享引用计数式持有）。和 backend 一样取 `'static`。
    model: &'static LlamaModel,
    /// 上下文 + 复用 batch，整体上锁。
    worker: Mutex<EmbedWorker>,
    config_model_id: String,
    dim: usize,
}

/// 真正持有 `LlamaContext` 的内层结构。`LlamaContext` 借用 `model`，必须同生命周期。
struct EmbedWorker {
    ctx: LlamaContext<'static>,
    batch: LlamaBatch<'static>,
}

// SAFETY:
// - LlamaContext / LlamaBatch 内含 NonNull 原始指针，llama-cpp-2 默认不 impl Send/Sync。
// - llama.cpp 的 context / batch 在 *单线程序列化* 访问下是安全的（这是 llama.cpp 的承诺）。
// - 我们把 EmbedWorker 整个包在 Mutex 里（见 LlamaCppEmbedder.worker），任意时刻只有
//   一个线程能持有 &mut，并且我们不在 .await 期间持锁。这等同于"始终在同一线程访问"。
// - 因此把 EmbedWorker 标记为 Send + Sync 是 sound 的。
unsafe impl Send for EmbedWorker {}
unsafe impl Sync for EmbedWorker {}

impl LlamaCppEmbedder {
    /// 加载模型并构建 context。
    ///
    /// 注意：`LlamaBackend::init()` 全局只能调一次。这里用 `OnceLock` 保证。
    pub fn new(cfg: LlamaEmbedderConfig) -> Result<Self, PromptCacheError> {
        let backend = global_backend()?;
        let model = load_model(backend, &cfg.model_path)?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(cfg.n_ctx))
            .with_embeddings(true)
            .with_pooling_type(cfg.pooling)
            .with_n_threads(cfg.n_threads)
            .with_n_threads_batch(cfg.n_threads);

        let ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| PromptCacheError::Embedder(format!("new_context: {e}")))?;

        let dim = model.n_embd() as usize;
        // batch 容量按 n_ctx 走；单条短查询不会爆。
        let batch = LlamaBatch::new(cfg.n_ctx as usize, 1);

        Ok(Self {
            _backend: backend,
            model,
            worker: Mutex::new(EmbedWorker { ctx, batch }),
            config_model_id: cfg.model_id,
            dim,
        })
    }
}

impl Embedder for LlamaCppEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, PromptCacheError> {
        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| PromptCacheError::Embedder(format!("tokenize: {e}")))?;

        if tokens.is_empty() {
            return Err(PromptCacheError::Embedder("empty token sequence".into()));
        }

        let mut w = self
            .worker
            .lock()
            .map_err(|_| PromptCacheError::Embedder("worker mutex poisoned".into()))?;
        // 拆借避免 E0499：ctx 和 batch 同时需要 &mut self
        let EmbedWorker { ctx, batch } = &mut *w;

        // 复用 ctx，每次清空 KV cache 防累积
        ctx.clear_kv_cache();
        batch.clear();

        // 单条序列 seq_id=0；最后一个 token 设 logits=true（pooling 需要）
        for (i, tok) in tokens.iter().enumerate() {
            let last = i == tokens.len() - 1;
            batch
                .add(*tok, i as i32, &[0], last)
                .map_err(|e| PromptCacheError::Embedder(format!("batch.add: {e}")))?;
        }

        ctx.decode(batch)
            .map_err(|e| PromptCacheError::Embedder(format!("decode: {e}")))?;

        let raw = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| PromptCacheError::Embedder(format!("embeddings_seq_ith: {e}")))?;

        if raw.len() != self.dim {
            return Err(PromptCacheError::Embedder(format!(
                "embedding dim {} != expected {}",
                raw.len(),
                self.dim
            )));
        }

        Ok(l2_normalize(raw))
    }

    fn model_id(&self) -> &str {
        &self.config_model_id
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// L2 归一化（in-place 数学等价）。
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

// ---- 全局 backend / model 缓存 ----
//
// LlamaBackend::init 一次。LlamaModel 同一个 path 也只 load 一次（避免 OOM）。
// Context 是每个 embedder 实例独占，model 是共享。

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::RwLock;

fn global_backend() -> Result<&'static LlamaBackend, PromptCacheError> {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    // llama.cpp/ggml 加载日志（llama_model_loader / print_info / create_tensor / sched_reserve …）
    // 默认会直接刷 stderr，一大坨。这里路由进 tracing 并按**全局 tracing 级别**门控：
    //   - 非 --debug（默认 info）→ with_logs_enabled(false)，完全抑制；
    //   - --debug（全局 debug 级）→ 路由进 tracing 显示（target=llama-cpp-2）。
    // 注：send_logs_to_tracing 内部用 OnceLock 仅首次生效，故放在 backend 只初始化一次处。
    let llama_logs_on = tracing::level_filters::LevelFilter::current()
        >= tracing::level_filters::LevelFilter::DEBUG;
    llama_cpp_2::send_logs_to_tracing(
        llama_cpp_2::LogOptions::default().with_logs_enabled(llama_logs_on),
    );
    let b = LlamaBackend::init().map_err(|e| PromptCacheError::Embedder(format!("backend: {e}")))?;
    Ok(BACKEND.get_or_init(|| b))
}

fn load_model(
    backend: &'static LlamaBackend,
    path: &Path,
) -> Result<&'static LlamaModel, PromptCacheError> {
    static MODELS: OnceLock<RwLock<HashMap<std::path::PathBuf, &'static LlamaModel>>> =
        OnceLock::new();
    let registry = MODELS.get_or_init(|| RwLock::new(HashMap::new()));

    {
        let r = registry.read().map_err(|_| {
            PromptCacheError::Embedder("model registry rwlock poisoned".into())
        })?;
        if let Some(m) = r.get(path) {
            return Ok(*m);
        }
    }

    let params = LlamaModelParams::default();
    let owned = LlamaModel::load_from_file(backend, path, &params)
        .map_err(|e| PromptCacheError::Embedder(format!("load_from_file: {e}")))?;
    // Box::leak 把 model 拍成 'static —— 进程生命周期内不释放。
    // 嵌入式场景模型一直要用，与其引计数管理不如 leak 简洁。
    let leaked: &'static LlamaModel = Box::leak(Box::new(owned));

    let mut w = registry
        .write()
        .map_err(|_| PromptCacheError::Embedder("model registry rwlock poisoned".into()))?;
    w.insert(path.to_path_buf(), leaked);
    Ok(leaked)
}

