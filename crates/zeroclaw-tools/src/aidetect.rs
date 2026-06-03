//! aidetect — Hi3516CV610 NPU 目标检测工具（封装板上 sample_aidetect）。
//!
//! 读取一张 YUV 图 + 一个 NPU 模型 .bin，返回检测框列表。
//! image_path 可选：不传用 config.image_path（默认测试图），
//! 传了则用指定路径（与 get_frame 组合：先抓帧 → 再检测）。
//! 模型作为参数暴露给 AI 选。路径全部走 [`AidetectConfig`]。

use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};
use std::sync::OnceLock;
use tokio::process::Command;
use zeroclaw_api::tool::{Tool, ToolResult};
use zeroclaw_config::schema::AidetectConfig;

const MODEL_FILES: &[&str] = &[
    "det_hvf_hor.bin",
    "det_hvf_ver.bin",
    "det_hvf_hor_ll_lite.bin",
    "det_hvf_ver_ll_lite.bin",
    "det_hv_hor.bin",
    "det_hv_ver.bin",
    "det_head_hor.bin",
    "det_nmv_hor.bin",
    "det_nmv_hor_elev.bin",
    "det_pet_hor.bin",
    "det_pkg_hor.bin",
];

fn type_name(t: i32) -> &'static str {
    match t {
        0 => "face",
        1 => "human",
        2 => "vehicle",
        3 => "pet",
        4 => "garbage",
        5 => "bag",
        6 => "wallet",
        7 => "phone",
        8 => "head_shoulder",
        9 => "bicycle",
        10 => "motorcycle",
        11 => "package",
        _ => "unknown",
    }
}

/// 中文目标名（用于人类可读 summary）。优先按 detection 的 `type` 整数映射，
/// 退路按英文 `type_name` 映射（单测里只给 type_name 时也能用）。
fn label_zh(d: &Value) -> &'static str {
    if let Some(t) = d.get("type").and_then(serde_json::Value::as_i64) {
        return match t {
            0 => "人脸",
            1 => "人",
            2 => "车",
            3 => "宠物",
            4 => "垃圾",
            5 => "包",
            6 => "钱包",
            7 => "手机",
            8 => "头肩",
            9 => "自行车",
            10 => "摩托车",
            11 => "包裹",
            _ => "未知目标",
        };
    }
    match d.get("type_name").and_then(serde_json::Value::as_str).unwrap_or("") {
        "face" => "人脸",
        "human" => "人",
        "vehicle" => "车",
        "pet" => "宠物",
        "garbage" => "垃圾",
        "bag" => "包",
        "wallet" => "钱包",
        "phone" => "手机",
        "head_shoulder" => "头肩",
        "bicycle" => "自行车",
        "motorcycle" => "摩托车",
        "package" => "包裹",
        _ => "未知目标",
    }
}

fn frame_line_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\s*type:\s*(\d+)\s+id:\s*(\d+)\s+conf:\s*([0-9.]+)\s*\[\s*x:\s*(-?\d+)\s*,\s*y:\s*(-?\d+)\s*,\s*width:\s*(\d+)\s*,\s*height:\s*(\d+)\s*\]"
        ).expect("frame line regex compiles")
    })
}

pub struct AidetectTool {
    config: AidetectConfig,
}

impl AidetectTool {
    pub fn new(config: AidetectConfig) -> Self {
        Self { config }
    }
}

impl Default for AidetectTool {
    fn default() -> Self {
        Self::new(AidetectConfig::default())
    }
}

#[async_trait]
impl Tool for AidetectTool {
    fn name(&self) -> &str {
        "aidetect"
    }

    fn description(&self) -> &str {
        "Hi3516CV610 板上 NPU 目标检测。读取 YUV 图（NV21），用指定模型跑推理，\
         返回检测框列表（类别、置信度、xywh 坐标）。\
         image 参数可选：不传用默认测试图；传了则读指定文件。\
         默认 model=det_hvf_hor.bin（同时识别人/车/脸，最通用）。\
         每个模型文件支持的检测类别由模型名称中的字母缩写决定（h=human，v=vehicle，f=face，nm=non-motor vehicle，pet=宠物，pkg=包裹）。"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "YUV 图绝对路径（NV21 格式）。不传则用 config 里的默认测试图。通常传 get_frame 的输出路径。"
                },
                "model": {
                    "type": "string",
                    "description": "NPU 模型文件名（不含路径）。默认 det_hvf_hor.bin 同时识别人+车+脸。每个模型文件支持的检测类别由模型名称中的字母缩写决定（h=human，v=vehicle，f=face，nm=non-motor vehicle，pet=宠物，pkg=包裹）。",
                    "enum": [
                        "det_hvf_hor.bin",
                        "det_hvf_ver.bin",
                        "det_hvf_hor_ll_lite.bin",
                        "det_hvf_ver_ll_lite.bin",
                        "det_hv_hor.bin",
                        "det_hv_ver.bin",
                        "det_head_hor.bin",
                        "det_nmv_hor.bin",
                        "det_nmv_hor_elev.bin",
                        "det_pet_hor.bin",
                        "det_pkg_hor.bin"
                    ],
                    "default": "det_hvf_hor.bin"
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        if !self.config.enabled {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("aidetect disabled (config: [aidetect].enabled=false)".into()),
            });
        }

        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("det_hvf_hor.bin");

        // image_path: 参数优先（get_frame 产物），没传则 fallback config 默认测试图
        let image_path = args
            .get("image")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.config.image_path);

        if !MODEL_FILES.contains(&model) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "unknown model '{}'. supported: {}",
                    model,
                    MODEL_FILES.join(", ")
                )),
            });
        }

        let bin_path = &self.config.binary_path;
        let model_path = format!("{}/{}", self.config.models_dir.trim_end_matches('/'), model);

        tracing::debug!(
            "🚀 exec: {bin_path} -m {model_path} -i {image_path} -s {} -v",
            self.config.image_size
        );

        let proc = Command::new(bin_path)
            .arg("-m")
            .arg(&model_path)
            .arg("-i")
            .arg(image_path)
            .arg("-s")
            .arg(&self.config.image_size)
            .arg("-v")
            .output()
            .await;

        let output = match proc {
            Ok(o) => o,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("failed to spawn {}: {}", bin_path, e)),
                });
            }
        };

        if !output.status.success() {
            return Ok(ToolResult {
                success: false,
                output: String::from_utf8_lossy(&output.stdout).into_owned(),
                error: Some(format!(
                    "sample_aidetect exit {:?}, stderr: {}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stderr).trim()
                )),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let detections = parse_frame_section(&stdout);

        let summary = format_summary(model, &detections);
        let result = json!({
            "model": model,
            "image": image_path,
            "image_size": self.config.image_size,
            "detection_count": detections.len(),
            "detections": detections,
            "summary": summary,
        });

        Ok(ToolResult {
            success: true,
            output: serde_json::to_string_pretty(&result).unwrap_or_default(),
            error: None,
        })
    }
}

fn parse_frame_section(stdout: &str) -> Vec<Value> {
    let mut in_frame = false;
    let mut detections = Vec::new();
    let re = frame_line_regex();

    for line in stdout.lines() {
        if line.contains("==================== Frame ====================") {
            in_frame = true;
            continue;
        }
        if !in_frame {
            continue;
        }
        if line.trim_start().starts_with("====================") {
            break;
        }
        if let Some(c) = re.captures(line) {
            let t: i32 = c[1].parse().unwrap_or(-1);
            let id: i32 = c[2].parse().unwrap_or(-1);
            let conf: f32 = c[3].parse().unwrap_or(0.0);
            let x: i32 = c[4].parse().unwrap_or(0);
            let y: i32 = c[5].parse().unwrap_or(0);
            let w: i32 = c[6].parse().unwrap_or(0);
            let h: i32 = c[7].parse().unwrap_or(0);
            detections.push(json!({
                "type": t,
                "type_name": type_name(t),
                "id": id,
                "conf": conf,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }));
        }
    }
    detections
}

fn format_summary(model: &str, detections: &[Value]) -> String {
    if detections.is_empty() {
        return format!("模型 {model} 未检测到任何目标。");
    }
    // 计数概览（中文名）：人脸×2、人×1
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for d in detections {
        *counts.entry(label_zh(d)).or_insert(0) += 1;
    }
    let overview: Vec<String> = counts.iter().map(|(k, v)| format!("{k}×{v}")).collect();

    let mut out = format!(
        "模型 {model} 检测结果如下：共 {} 个目标（{}）。",
        detections.len(),
        overview.join("、")
    );
    // 逐个目标的明细：类别 + 置信度 + 坐标 + 尺寸
    for (i, d) in detections.iter().enumerate() {
        let conf = d.get("conf").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
        let x = d.get("x").and_then(serde_json::Value::as_i64).unwrap_or(0);
        let y = d.get("y").and_then(serde_json::Value::as_i64).unwrap_or(0);
        let w = d.get("width").and_then(serde_json::Value::as_i64).unwrap_or(0);
        let h = d.get("height").and_then(serde_json::Value::as_i64).unwrap_or(0);
        out.push_str(&format!(
            "\n  {}. {}：置信度 {:.0}%，位置 ({x},{y})，尺寸 {w}×{h}",
            i + 1,
            label_zh(d),
            conf * 100.0
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_SAMPLE: &str = "==================== Input Param ====================\n\
Read model from path\n\
Model path: /mnt/aidetect/models/det_hvf_hor.bin\n\
Image path: /mnt/aidetect/data/hvf_image_hor_1920x1080.yuv\n\
Image size: 1920x1080\n\
\n\
[MPP] Version: [HI3516CV610_MPP_V1.0.2.0 B051 Release], Build Time[Apr 16 2025, 12:06:53]\n\
\n\
==================== Model Info ====================\n\
model support frame width: 768, model support frame height: 448\n\
model support class num: 3\n\
model support class type:| 0 | 1 | 2 |\n\
==================== Default Chn Param ====================\n\
preemp_en: 1, priority: 3, priority_up_step_timeout: 0, priority_up_top_timeout: 0\n\
detect num: 3\n\
detect type:0, threshold: 0.400000, track_missing_frame_num: 30\n\
detect type:1, threshold: 0.620000, track_missing_frame_num: 30\n\
detect type:2, threshold: 0.630000, track_missing_frame_num: 30\n\
==================== Default Chn Attr ====================\n\
track num: 3\n\
track type:0, is_tracking: 0\n\
track type:1, is_tracking: 0\n\
track type:2, is_tracking: 0\n\
==================== Frame ====================\n\
type:1  id:0  conf:0.900000 [x: 620, y:  94, width: 375, height: 904] \n";

    #[test]
    fn parses_real_board_output() {
        let dets = parse_frame_section(REAL_SAMPLE);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0]["type"], 1);
        assert_eq!(dets[0]["type_name"], "human");
        assert_eq!(dets[0]["id"], 0);
        let conf = dets[0]["conf"].as_f64().unwrap();
        assert!((conf - 0.9).abs() < 1e-5);
        assert_eq!(dets[0]["x"], 620);
        assert_eq!(dets[0]["y"], 94);
        assert_eq!(dets[0]["width"], 375);
        assert_eq!(dets[0]["height"], 904);
    }

    #[test]
    fn ignores_pre_frame_noise_with_fake_type_lines() {
        let s = "==================== Input Param ====================\n\
                 type:99 id:99 conf:9.0 [x: 0, y: 0, width: 0, height: 0]\n\
                 ==================== Frame ====================\n\
                 type:0 id:0 conf:0.5 [x: 1, y: 2, width: 3, height: 4]\n";
        let dets = parse_frame_section(s);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0]["type"], 0);
    }

    #[test]
    fn empty_frame_section_returns_empty() {
        let s = "==================== Frame ====================\n";
        assert_eq!(parse_frame_section(s).len(), 0);
    }

    #[test]
    fn multiple_detections() {
        let s = "==================== Frame ====================\n\
                 type:0 id:0 conf:0.85 [x: 100, y: 100, width: 50, height: 60]\n\
                 type:1 id:1 conf:0.72 [x: 200, y: 200, width: 150, height: 300]\n\
                 type:2 id:2 conf:0.91 [x: 800, y: 400, width: 400, height: 200]\n";
        let dets = parse_frame_section(s);
        assert_eq!(dets.len(), 3);
        assert_eq!(dets[0]["type_name"], "face");
        assert_eq!(dets[1]["type_name"], "human");
        assert_eq!(dets[2]["type_name"], "vehicle");
    }

    #[test]
    fn summary_formats_counts() {
        let dets = vec![
            json!({"type_name": "human"}),
            json!({"type_name": "human"}),
            json!({"type_name": "vehicle"}),
        ];
        let s = format_summary("det_hvf_hor.bin", &dets);
        assert!(s.contains("3 个目标"));
        assert!(s.contains("人×2"));
        assert!(s.contains("车×1"));
    }

    #[test]
    fn summary_empty() {
        let s = format_summary("det_pet_hor.bin", &[]);
        assert!(s.contains("未检测到"));
    }

    #[test]
    fn schema_lists_all_models() {
        let tool = AidetectTool::default();
        let schema = tool.parameters_schema();
        let enum_vals = schema["properties"]["model"]["enum"].as_array().unwrap();
        assert_eq!(enum_vals.len(), MODEL_FILES.len());
    }
}
