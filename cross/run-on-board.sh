#!/bin/sh
# run-on-board.sh —— 在板上**不动 / 根目录**直接跑 zeroclaw
#
# 板上：
#     tar xzf dist-board.tar.gz
#     cd dist-board
#     ./run.sh agent -m "打开空调"      # 把参数原样转给 zeroclaw（推荐用 agent 子命令）
#     ./run.sh status                  # 也行，任何 zeroclaw 子命令都行
#
# 用同目录里的 bin/data/models，临时合成一个 config 目录指过去，跑完不留痕。
# 适合临时验证、"不污染板上根目录"场景。要长期常驻请改用 ./install.sh。

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
GGUF=$(ls "$HERE/models/"*.gguf 2>/dev/null | head -1)
[ -n "$GGUF" ] || { echo "[FATAL] no gguf in $HERE/models/" >&2; exit 1; }

# zeroclaw 要 --config-dir <DIR>（目录），不是 --config <FILE>。临时建个目录放 config.toml。
CONF_DIR=$(mktemp -d -t zeroclaw-portable.XXXXXX 2>/dev/null || echo "/tmp/zeroclaw-portable.$$")
mkdir -p "$CONF_DIR"
trap 'rm -rf "$CONF_DIR"' EXIT

# model_id 留空 → SqlitePromptCache::open 会跳过 db meta 校验。这样无论 build_db
# 时打的 model_id 是什么，run.sh 都能开起来。install.sh 走标准路径才严格校验。
cat > "$CONF_DIR/config.toml" <<EOF
[prompt_cache]
enabled    = true
db_path    = "$HERE/data/l2_vector.db"
model_path = "$GGUF"
pooling    = "cls"
n_ctx      = 64
n_threads  = 2
accept     = 0.66
reject     = 0.54
EOF

# skills 段：把抓帧/检测 skill 注入（prompt_injection_mode=full）。便携模式下 binary 在
# $HERE 下，故把打包 skill 里的 /root/.zeroclaw/tools 改写成 $HERE，并照常裸名可调（见下方 PATH）。
cat >> "$CONF_DIR/config.toml" <<EOF

[skills]
open_skills_enabled = false
allow_scripts       = false
prompt_injection_mode = "full"
EOF

# --config-dir 时工作区 = CONF_DIR/workspace；把 skill 铺进去（sed 改写绝对路径到 $HERE）
if [ -d "$HERE/skills" ]; then
    for d in "$HERE/skills/"*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        mkdir -p "$CONF_DIR/workspace/skills/$name"
        if [ -f "$d/SKILL.md" ]; then
            sed "s|/root/.zeroclaw/tools|$HERE|g" "$d/SKILL.md" \
                > "$CONF_DIR/workspace/skills/$name/SKILL.md"
        fi
    done
fi

# 让两个 binary 裸名可调（便携模式：直接把 $HERE 下工具目录加进 PATH）
export PATH="$HERE/get_frame:$HERE/aidetect:$PATH"

exec "$HERE/bin/zeroclaw" --config-dir "$CONF_DIR" "$@"
