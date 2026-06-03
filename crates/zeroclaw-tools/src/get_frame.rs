//! get_frame — Hi3516CV610 sensor 抓帧工具。
//!
//! 封装板上 `hi3516cv610_get_frame` 二进制。从 sensor 抓一帧并按指定路径落盘，
//! 返回 JSON 含 output 路径。

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;
use zeroclaw_api::tool::{Tool, ToolResult};
use zeroclaw_config::schema::GetFrameConfig;

pub struct GetFrameTool {
    config: GetFrameConfig,
}

impl GetFrameTool {
    pub fn new(config: GetFrameConfig) -> Self {
        Self { config }
    }
}

impl Default for GetFrameTool {
    fn default() -> Self {
        Self::new(GetFrameConfig::default())
    }
}

#[async_trait]
impl Tool for GetFrameTool {
    fn name(&self) -> &str {
        "get_frame"
    }

    fn description(&self) -> &str {
        "Hi3516CV610 摄像头抓帧。从 sensor 读取一帧并保存到文件。\
         返回 output 路径。支持全部 SDK 格式，默认 yuv(NV12) 1920x1080。"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "sensor": {
                    "type": "string",
                    "description": "sensor 型号。目前仅支持 sc4336p。",
                    "enum": ["sc4336p"],
                    "default": "sc4336p"
                },
                "type": {
                    "type": "string",
                    "description": "输出格式。RAW: yuv/NV12(默认) / yvu/NV21 / yuv422sp / yvu422sp / yv12 / gray / rgb / bgr / bgr_planar / rgb565 / bgr565。ENCODED: jpg / mjpg / h264 / h265",
                    "enum": ["yuv", "yvu", "yuv422sp", "yvu422sp", "yv12", "gray", "rgb", "bgr", "bgr_planar", "rgb565", "bgr565", "jpg", "mjpg", "h264", "h265"],
                    "default": "yuv"
                },
                "output": {
                    "type": "string",
                    "description": "输出文件路径。默认 `/tmp/frame.yuv`"
                },
                "size": {
                    "type": "string",
                    "description": "分辨率 WxH。默认 1920x1080",
                    "default": "1920x1080"
                },
                "skip": {
                    "type": "integer",
                    "description": "AE/AWB 收敛丢帧数。室内稳定光 16，强光/户外 30~60。默认 16",
                    "default": 16
                },
                "count": {
                    "type": "integer",
                    "description": "连续抓几帧。默认 1。>1 时文件名自动加序号",
                    "default": 1
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        if !self.config.enabled {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("get_frame disabled (config: [get_frame].enabled=false)".into()),
            });
        }

        let sensor = args
            .get("sensor")
            .and_then(|v| v.as_str())
            .unwrap_or("sc4336p");
        let fmt = args
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("yuv");
        let size = args
            .get("size")
            .and_then(|v| v.as_str())
            .unwrap_or("1920x1080");
        let skip: u32 = args
            .get("skip")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as u32;
        let count: u32 = args
            .get("count")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        let ext = match fmt {
            "jpg" | "jpeg" => "jpg",
            "mjpg" | "mjpeg" => "mjpg",
            "h264" => "h264",
            "h265" | "hevc" => "h265",
            _ => "yuv",
        };
        let default_output = format!("/tmp/frame.{ext}");
        let output = args
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or(&default_output)
            .to_string();

        let bin_path = &self.config.binary_path;

        let n_arg = if count > 1 { format!(" -n {count}") } else { String::new() };
        tracing::debug!("🚀 exec: {bin_path} -s {sensor} -t {fmt} -o {output} -z {size} -k {skip}{n_arg}");

        let mut cmd = Command::new(bin_path);
        cmd.arg("-s").arg(sensor)
            .arg("-t").arg(fmt)
            .arg("-o").arg(&output)
            .arg("-z").arg(size)
            .arg("-k").arg(skip.to_string());
        if count > 1 {
            cmd.arg("-n").arg(count.to_string());
        }
        let proc = cmd.output().await;

        let output_data = match proc {
            Ok(o) => o,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("failed to spawn {}: {}", bin_path, e)),
                });
            }
        };

        if !output_data.status.success() {
            return Ok(ToolResult {
                success: false,
                output: String::from_utf8_lossy(&output_data.stdout).into_owned(),
                error: Some(format!(
                    "get_frame exit {:?}, stderr: {}",
                    output_data.status.code(),
                    String::from_utf8_lossy(&output_data.stderr).trim()
                )),
            });
        }

        let result = json!({
            "output": output,
            "format": fmt,
            "size": size,
        });

        Ok(ToolResult {
            success: true,
            output: serde_json::to_string_pretty(&result).unwrap_or_default(),
            error: None,
        })
    }
}
