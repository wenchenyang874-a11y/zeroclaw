#!/bin/sh
# install-on-board.sh —— 在 Hi3516CV610 板上把 dist-board 一键铺到位
#
# 板上（busybox sh）跑：
#     tar xzf dist-board.tar.gz
#     cd dist-board
#     ./install.sh                  # 默认部署
#     ./install.sh --dry-run        # 只看会做什么不执行
#
# 装完直接：
#     zeroclaw agent -m "打开空调"
#                                 ^ 不带 --config-dir，因为配置写到了 zeroclaw 默认查找位置
#
# 数据布局（运行时数据统一在 $HOME/.zeroclaw 下，便于管理；见
#           doc/3layers_sdk/架构设计/多工具组合调用-skill+shell方案.md §3.①）：
#     /usr/bin/zeroclaw                          ARM 主二进制（含 prompt-cache）
#     /root/.zeroclaw/config.toml                配置（zeroclaw 默认查找路径，HOME=/root 时）
#     /root/.zeroclaw/data/l2_vector.db          L2 规则 + 向量库
#     /root/.zeroclaw/data/l1_string_match.db    L1 字符串规则库
#     /root/.zeroclaw/models/<gguf>              embedding 模型
#     /root/.zeroclaw/tools/aidetect/            NPU 目标检测 binary + 模型 + 测试图
#     /root/.zeroclaw/tools/get_frame/           sensor 抓帧 binary
#     /root/.zeroclaw/workspace/skills/          snap_detect skill（组合·pipeline，自包含）
#   另外往 /etc/profile 追加 set_path_before，使两个 binary 裸名可调。
#
# 已有配置会自动备份到 config.toml.bak.<timestamp>，安装完成后可以从备份里
# 把你以前的 api_key 拷回去。

set -eu

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'DRY: '
        printf '%s ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

HERE=$(cd "$(dirname "$0")" && pwd)
ZEROCLAW_HOME="${HOME:-/root}/.zeroclaw"
CONFIG_PATH="$ZEROCLAW_HOME/config.toml"
DATA_DIR="$ZEROCLAW_HOME/data"
MODELS_DIR="$ZEROCLAW_HOME/models"
TOOLS_DIR="$ZEROCLAW_HOME/tools"
SKILLS_DIR="$ZEROCLAW_HOME/workspace/skills"

# ============ 0. 平台自检 ============
echo "[install] running on $(uname -m), source: $HERE"
ARCH=$(uname -m)
case "$ARCH" in
    armv7l|armv6l) : ;;
    *)
        if [ "$DRY_RUN" -eq 0 ]; then
            cat >&2 <<EOF
[FATAL] 这台机器架构是 $ARCH，看起来不是 ARM 板。
        install.sh 只能在 Hi3516CV610 板上跑。
        想在宿主机看脚本逻辑请加 --dry-run。
EOF
            exit 1
        fi
        ;;
esac

# ============ 1. 资产齐不齐 ============
echo "[install] checking payload ..."
ZEROCLAW_BIN="$HERE/bin/zeroclaw"
L2_VECTOR_DB="$HERE/data/l2_vector.db"
STRING_MATCH_DB="$HERE/data/l1_string_match.db"
GGUF_FILE=$(ls "$HERE/models/"*.gguf 2>/dev/null | head -1)
TEMPLATE="$HERE/config/zeroclaw.toml"

for f in "$ZEROCLAW_BIN" "$L2_VECTOR_DB" "$STRING_MATCH_DB" "$TEMPLATE"; do
    [ -f "$f" ] || { echo "[FATAL] missing: $f" >&2; exit 1; }
done
[ -n "$GGUF_FILE" ] && [ -f "$GGUF_FILE" ] \
    || { echo "[FATAL] no gguf in $HERE/models/" >&2; exit 1; }

GGUF_NAME=$(basename "$GGUF_FILE")
echo "  zeroclaw:        $(du -h "$ZEROCLAW_BIN" | awk '{print $1}')"
echo "  l2_vector.db:    $(du -h "$L2_VECTOR_DB"     | awk '{print $1}')"
echo "  l1_string_match.db: $(du -h "$STRING_MATCH_DB"  | awk '{print $1}')"
echo "  model:           $(du -h "$GGUF_FILE"    | awk '{print $1}')  ($GGUF_NAME)"

if command -v file >/dev/null 2>&1; then
    if ! file "$ZEROCLAW_BIN" 2>/dev/null | grep -q ARM; then
        echo "[FATAL] zeroclaw binary 不是 ARM ELF" >&2
        file "$ZEROCLAW_BIN" >&2
        exit 1
    fi
fi

# ============ 2. 创建目录 ============
echo
echo "[install] creating dirs ..."
for d in /usr/bin "$ZEROCLAW_HOME" "$DATA_DIR" "$MODELS_DIR" "$TOOLS_DIR" "$SKILLS_DIR"; do
    [ -d "$d" ] || run mkdir -p "$d"
done

# ============ 3. 拷贝产物 ============
echo
echo "[install] copying files ..."
run cp "$ZEROCLAW_BIN" /usr/bin/zeroclaw
run chmod +x /usr/bin/zeroclaw
run cp "$L2_VECTOR_DB" "$DATA_DIR/l2_vector.db"
run cp "$STRING_MATCH_DB" "$DATA_DIR/l1_string_match.db"
run cp "$GGUF_FILE" "$MODELS_DIR/$GGUF_NAME"

# aidetect 工具（可选，dist-board 里有才装；与 [aidetect].binary_path 对齐）
AIDETECT_SRC="$HERE/aidetect"
if [ -d "$AIDETECT_SRC" ] && [ -f "$AIDETECT_SRC/sample_aidetect" ]; then
    echo "  installing aidetect → $TOOLS_DIR/aidetect/"
    run mkdir -p "$TOOLS_DIR/aidetect/models" "$TOOLS_DIR/aidetect/data"
    run cp "$AIDETECT_SRC/sample_aidetect" "$TOOLS_DIR/aidetect/sample_aidetect"
    run chmod +x "$TOOLS_DIR/aidetect/sample_aidetect"
    for f in "$AIDETECT_SRC/models/"*.bin; do
        [ -f "$f" ] && run cp "$f" "$TOOLS_DIR/aidetect/models/$(basename "$f")"
    done
    for f in "$AIDETECT_SRC/data/"*.yuv; do
        [ -f "$f" ] && run cp "$f" "$TOOLS_DIR/aidetect/data/$(basename "$f")"
    done
else
    echo "  (skip aidetect: $AIDETECT_SRC not present)"
fi

# get_frame 工具
GET_FRAME_SRC="$HERE/get_frame"
if [ -d "$GET_FRAME_SRC" ] && [ -f "$GET_FRAME_SRC/hi3516cv610_get_frame" ]; then
    echo "  installing get_frame → $TOOLS_DIR/get_frame/"
    run mkdir -p "$TOOLS_DIR/get_frame"
    run cp "$GET_FRAME_SRC/hi3516cv610_get_frame" "$TOOLS_DIR/get_frame/hi3516cv610_get_frame"
    run chmod +x "$TOOLS_DIR/get_frame/hi3516cv610_get_frame"
else
    echo "  (skip get_frame: $GET_FRAME_SRC not present)"
fi

# skills（放到 zeroclaw 默认工作区，prompt_injection_mode=full 注入）。当前仅 snap_detect
# （组合·pipeline，自包含）。先清空 $SKILLS_DIR 再铺，确保旧版残留的 skill（如已删的
# get_frame/aidetect 原子 skill）在重装时被移除，不会误导 LLM。
SKILLS_SRC="$HERE/skills"
if [ -d "$SKILLS_SRC" ]; then
    echo "  installing skills → $SKILLS_DIR/ (先清空旧 skill)"
    run rm -rf "$SKILLS_DIR"
    for d in "$SKILLS_SRC"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        run mkdir -p "$SKILLS_DIR/$name"
        [ -f "$d/SKILL.md" ] && run cp "$d/SKILL.md" "$SKILLS_DIR/$name/SKILL.md"
    done
else
    echo "  (skip skills: $SKILLS_SRC not present)"
fi

# ============ 3b. PATH：让两个 binary 裸名可调（写 /etc/profile）============
# /etc/profile 已有 set_path_before 习惯用法（目录存在才前插 PATH）。zeroclaw 进程继承
# 启动它的 shell 的 PATH，故登录后启动即可裸名调到（见方案文档 §3.①(2)）。
echo
echo "[install] wiring PATH into /etc/profile ..."
PROFILE=/etc/profile
GF_DIR="$TOOLS_DIR/get_frame"
AD_DIR="$TOOLS_DIR/aidetect"
for line in \
    "set_path_before $AD_DIR" \
    "set_path_before $GF_DIR"; do
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY: ensure '$line' in $PROFILE"
    elif [ -f "$PROFILE" ] && grep -qxF "$line" "$PROFILE"; then
        echo "  already present: $line"
    else
        echo "$line" >> "$PROFILE"
        echo "  appended: $line"
    fi
done
# 安装当前 shell 立即生效（/etc/profile 要重新登录才加载）
export PATH="$GF_DIR:$AD_DIR:$PATH"

# ============ 4. 写默认 config（路径 = zeroclaw 启动时自己找的位置）============
echo
TS=$(date +%Y%m%d-%H%M%S 2>/dev/null || echo bak)
if [ -f "$CONFIG_PATH" ]; then
    BAK="$CONFIG_PATH.bak.$TS"
    echo "[install] backing up existing config: $CONFIG_PATH → $BAK"
    run cp "$CONFIG_PATH" "$BAK"
fi
echo "[install] writing $CONFIG_PATH"
if [ "$DRY_RUN" -eq 0 ]; then
    # 替换模板占位符：__ZC_HOME__ → 数据根目录；__MODEL_PATH__ → 实际 gguf 路径
    sed -e "s|__ZC_HOME__|$ZEROCLAW_HOME|g" \
        -e "s|__MODEL_PATH__|$MODELS_DIR/$GGUF_NAME|g" \
        "$TEMPLATE" > "$CONFIG_PATH"
else
    echo "DRY: install $TEMPLATE → $CONFIG_PATH (sub __ZC_HOME__ → $ZEROCLAW_HOME, __MODEL_PATH__ → $MODELS_DIR/$GGUF_NAME)"
fi

# ============ 完成 ============
cat <<EOF

════════════════════════════════════════════════════════════════
[ok] 已部署完成。

数据布局（统一在 $ZEROCLAW_HOME 下）：
    /usr/bin/zeroclaw                          主二进制
    $DATA_DIR/l2_vector.db          L2 向量规则
    $DATA_DIR/l1_string_match.db    L1 字符串规则
    $MODELS_DIR/$GGUF_NAME
    $TOOLS_DIR/aidetect/            NPU 目标检测 binary（如有打包）
    $TOOLS_DIR/get_frame/           sensor 抓帧 binary（如有打包）
    $SKILLS_DIR/                    snap_detect skill（组合·pipeline）
    $CONFIG_PATH

抓帧检测组合命令靠 skill + shell 一次性下发（binary 经 /etc/profile 加入 PATH，裸名可调）：
    重新登录或 . /etc/profile 后，可直接手敲：
    hi3516cv610_get_frame -s sc4336p -z 1920x1080 -t yuv -o /tmp/f.yuv && \\
    sample_aidetect -m $TOOLS_DIR/aidetect/models/det_hvf_hor.bin -i /tmp/f.yuv -s 1920x1080 -v && rm -f /tmp/f.yuv

直接运行（无需 --config-dir，zeroclaw 自己找 \$HOME/.zeroclaw/config.toml）：
    zeroclaw agent -m "打开空调"

观察匹配日志（L1 命中和 L2 相似度）：
    zeroclaw agent -m "打开空调" 2>&1 | grep -E 'string_match|L2|prompt-cache'

L1 正则命中 / L2 向量命中 → 零 token 零网络，直接调工具。
miss 时走 L3 → 编辑 $CONFIG_PATH 里 [providers.models...] 的 api_key 才能调通云端。
旧配置已备份在 ${CONFIG_PATH}.bak.* —— api_key 等敏感字段从那里拷回来。

清理重装（数据全在一个目录，删一处即净；/etc/profile 的 set_path_before 行无害可留）：
    rm -rf /usr/bin/zeroclaw $ZEROCLAW_HOME
    ./install.sh
════════════════════════════════════════════════════════════════
EOF
