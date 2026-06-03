#!/bin/sh
# build-for-board.sh —— ARM 全链路构建 + 板上部署包
#
# 做的事：
#   1. 设好交叉编译 env（PATH / LIBCLANG_PATH / CMAKE_TOOLCHAIN_FILE / bindgen / 代理）
#   2. 编 ARM zeroclaw（含 prompt-cache）
#   3. 编 x86 prompt-cache-build-db（宿主机灌库用）
#   4. 跑 build_db 生成 l2_vector.db（如果 l2_vector.db 不在或 rules.json 更新过）
#   5. 把要拷到板上的东西归进 target/dist-board/
#   6. tar.gz 一份方便 scp
#
# 用法：
#   ./cross/build-for-board.sh                 # 默认：完整构建 + 打包
#   ./cross/build-for-board.sh --skip-build    # 复用现有 target/ 产物，只重打包
#   ./cross/build-for-board.sh --skip-db       # 不重新灌库，复用现有 l2_vector.db
#   RULES_JSON=path/to/rules.json ./cross/build-for-board.sh   # 用自己的规则集
#
# 输出：
#   target/dist-board/        ← 直接 scp 这个目录
#   target/dist-board.tar.gz  ← 或者 scp 这个 tar 包

set -eu

# ============ 路径常量 ============
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WS_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$WS_DIR/../../.." && pwd)        # 顶层 3layers/

ARM_TOOLCHAIN_DIR=/opt/linux/x86-arm/gcc-20250305-arm-v01c02-linux-musleabi/arm-v01c02-linux-musleabi-gcc
ARM_SYSROOT="$ARM_TOOLCHAIN_DIR/target"
LIBCLANG_DEFAULT=/usr/lib/llvm-14/lib

# 默认资产位置（可被 env 覆盖）
RULES_JSON="${RULES_JSON:-$WS_DIR/crates/zeroclaw-prompt-cache/examples/demo_rules.json}"
NEGATIVE_SAMPLES_JSON="${NEGATIVE_SAMPLES_JSON:-$WS_DIR/crates/zeroclaw-prompt-cache/examples/negative_samples.json}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/3layers_sdk/models/embedding/bge-small-zh-v1.5/bge-small-zh-v1.5-q4_k_m.gguf}"
MODEL_ID="${MODEL_ID:-bge-small-zh-v1.5-q4_k_m}"
POOLING="${POOLING:-cls}"
ACCEPT="${ACCEPT:-0.66}"
REJECT="${REJECT:-0.54}"

DIST_DIR="$REPO_ROOT/3layers_sdk/dist-board"
DIST_TAR="$REPO_ROOT/3layers_sdk/dist-board.tar.gz"

# ============ 参数解析 ============
SKIP_BUILD=0
SKIP_DB=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --skip-db)    SKIP_DB=1 ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ============ Env 自检 + 准备 ============
echo "[1/6] checking host environment"

# ARM 工具链
if [ ! -d "$ARM_TOOLCHAIN_DIR/bin" ]; then
    echo "[FATAL] ARM toolchain not found at $ARM_TOOLCHAIN_DIR/bin" >&2
    echo "       expected to contain arm-v01c02-linux-musleabi-gcc" >&2
    exit 1
fi
export PATH="$ARM_TOOLCHAIN_DIR/bin:$PATH"
echo "  ✓ ARM toolchain: $(arm-v01c02-linux-musleabi-gcc --version | head -1)"

# rustup target
if ! rustup target list --installed 2>/dev/null | grep -q armv7-unknown-linux-musleabi; then
    echo "  ! rustup target armv7-unknown-linux-musleabi 未装；现装"
    rustup target add armv7-unknown-linux-musleabi
fi
echo "  ✓ rust target: armv7-unknown-linux-musleabi"

# libclang for bindgen
if [ -z "${LIBCLANG_PATH:-}" ]; then
    if [ -f "$LIBCLANG_DEFAULT/libclang.so.1" ]; then
        export LIBCLANG_PATH="$LIBCLANG_DEFAULT"
    else
        # try to find any libclang
        FOUND=$(find /usr/lib -maxdepth 3 -name 'libclang.so*' 2>/dev/null | head -1)
        if [ -n "$FOUND" ]; then
            export LIBCLANG_PATH=$(dirname "$FOUND")
        else
            echo "[FATAL] libclang.so not found; set LIBCLANG_PATH or 'apt install libclang-dev'" >&2
            exit 1
        fi
    fi
fi
echo "  ✓ LIBCLANG_PATH=$LIBCLANG_PATH"

# 代理（可选 —— 已 set 就保留）
if [ -n "${http_proxy:-}" ]; then
    echo "  ✓ http_proxy=$http_proxy"
else
    echo "  ! http_proxy 未设；如拉 crates.io 失败请手动 export"
fi

# 交叉编译需要的所有 env
export CMAKE_TOOLCHAIN_FILE="$WS_DIR/cross/hi3516cv610.cmake"
export CXXFLAGS_armv7_unknown_linux_musleabi="-march=armv7-a -mtune=cortex-a7 -mfloat-abi=soft"
export BINDGEN_EXTRA_CLANG_ARGS_armv7_unknown_linux_musleabi="--target=armv7-linux-musleabi --sysroot=$ARM_SYSROOT -mfloat-abi=soft -march=armv7-a -isystem $ARM_SYSROOT/usr/include"
export CARGO_CFG_TARGET_FEATURE=""

# ============ 编 ARM zeroclaw ============
echo
if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "[2/6] cross-compiling zeroclaw + prompt-cache-cli (armv7 musl) ..."
    cd "$WS_DIR"
    cargo build --target armv7-unknown-linux-musleabi -p zeroclawlabs --features agent-runtime --release
    # 顺手编 prompt-cache-cli（板上 seed pending_rules、调试 L1.3 用）
    cargo build --target armv7-unknown-linux-musleabi -p zeroclaw-prompt-cache --bin prompt-cache-cli --release
    echo "  ✓ target/armv7-unknown-linux-musleabi/release/{zeroclaw,prompt-cache-cli}"
else
    echo "[2/6] --skip-build：复用现有 ARM 产物"
    [ -f "$WS_DIR/target/armv7-unknown-linux-musleabi/release/zeroclaw" ] || {
        echo "[FATAL] 没有现成的 zeroclaw ARM binary，去掉 --skip-build" >&2
        exit 1
    }
fi

# ============ 编 x86 build_db（宿主机灌库） ============
echo
if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "[3/6] building host prompt-cache-build-db (x86) ..."
    cargo build -p zeroclaw-prompt-cache --bin prompt-cache-build-db --release
    echo "  ✓ target/release/prompt-cache-build-db"
else
    echo "[3/6] --skip-build：跳过 x86 build_db 编译"
fi

# ============ 灌 l2_vector.db ============
echo
L2_VECTOR_DB_TMP="$WS_DIR/target/l2_vector.db"
if [ "$SKIP_DB" -eq 0 ] || [ ! -f "$L2_VECTOR_DB_TMP" ]; then
    echo "[4/6] populating l2_vector.db ..."
    [ -f "$MODEL_PATH" ] || { echo "[FATAL] model gguf not found: $MODEL_PATH" >&2; exit 1; }
    [ -f "$RULES_JSON" ] || { echo "[FATAL] rules json not found: $RULES_JSON" >&2; exit 1; }
    rm -f "$L2_VECTOR_DB_TMP"
    NEG_ARG=""
    if [ -f "$NEGATIVE_SAMPLES_JSON" ]; then
        NEG_ARG="--negative-samples $NEGATIVE_SAMPLES_JSON"
    fi
    "$WS_DIR/target/release/prompt-cache-build-db" \
        --db "$L2_VECTOR_DB_TMP" \
        --rules-json "$RULES_JSON" \
        --model "$MODEL_PATH" \
        --model-id "$MODEL_ID" \
        --pooling "$POOLING" \
        $NEG_ARG 2>&1 | grep -E '^\[|^  \+|^  neg' || true
    echo "  ✓ $L2_VECTOR_DB_TMP"
else
    echo "[4/6] --skip-db：复用现有 $L2_VECTOR_DB_TMP"
fi

# ============ 灌 L1 l1_string_match.db（与 L2 用同一份 rules.json） ============
echo
STRING_MATCH_DB_TMP="$WS_DIR/target/l1_string_match.db"
if [ "$SKIP_DB" -eq 0 ] || [ ! -f "$STRING_MATCH_DB_TMP" ]; then
    echo "[4.5/6] populating l1_string_match.db ..."
    rm -f "$STRING_MATCH_DB_TMP"
    "$WS_DIR/target/release/prompt-cache-build-db" \
        --l1-db "$STRING_MATCH_DB_TMP" \
        --rules-json "$RULES_JSON" 2>&1 | grep -E '^\[|^  \+' || true
    echo "  ✓ $STRING_MATCH_DB_TMP"
else
    echo "[4.5/6] --skip-db：复用现有 $STRING_MATCH_DB_TMP"
fi

# ============ 组装 dist-board/ ============
echo
echo "[5/6] assembling dist-board/ ..."

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/bin" "$DIST_DIR/data" "$DIST_DIR/models" "$DIST_DIR/config"

# 1. ARM zeroclaw
cp "$WS_DIR/target/armv7-unknown-linux-musleabi/release/zeroclaw" "$DIST_DIR/bin/"

# 1b. ARM prompt-cache-cli（板上调试用）
PROMPT_CLI_ARM="$WS_DIR/target/armv7-unknown-linux-musleabi/release/prompt-cache-cli"
[ -f "$PROMPT_CLI_ARM" ] && cp "$PROMPT_CLI_ARM" "$DIST_DIR/bin/" || true

# 2. l2_vector.db (L2 向量)
cp "$L2_VECTOR_DB_TMP" "$DIST_DIR/data/l2_vector.db"

# 2b. l1_string_match.db (L1 字符串规则)
cp "$STRING_MATCH_DB_TMP" "$DIST_DIR/data/l1_string_match.db"
PROMPT_CLI_ARM="$WS_DIR/target/armv7-unknown-linux-musleabi/release/prompt-cache-cli"
[ -f "$PROMPT_CLI_ARM" ] && cp "$PROMPT_CLI_ARM" "$DIST_DIR/bin/" || true

# 2. l2_vector.db
cp "$L2_VECTOR_DB_TMP" "$DIST_DIR/data/l2_vector.db"

# 3. 模型 gguf（板上 BGE 必备）
cp "$MODEL_PATH" "$DIST_DIR/models/$(basename "$MODEL_PATH")"

# 4. 默认 config.toml 模板（cross/board-default-config.toml 直接进包）
#    install.sh 会用 sed 把 __MODEL_PATH__ 替换成实际安装的 gguf 路径
MODEL_BASENAME=$(basename "$MODEL_PATH")
cp "$SCRIPT_DIR/board-default-config.toml" "$DIST_DIR/config/zeroclaw.toml"

# 5. 板上 install / run 脚本（源码在 cross/ 单独维护，从那里复制过去）
cp "$SCRIPT_DIR/install-on-board.sh" "$DIST_DIR/install.sh"
cp "$SCRIPT_DIR/run-on-board.sh"     "$DIST_DIR/run.sh"
chmod +x "$DIST_DIR/install.sh" "$DIST_DIR/run.sh"

# 6. aidetect 工具（Hi3516CV610 NPU 目标检测；wrapper 通过 [aidetect] 配置寻路）
AIDETECT_SRC="$REPO_ROOT/3layers_sdk/tools/aidetect"
AIDETECT_DST="$DIST_DIR/aidetect"
if [ -d "$AIDETECT_SRC" ] && [ -f "$AIDETECT_SRC/sample_aidetect" ]; then
    mkdir -p "$AIDETECT_DST/models" "$AIDETECT_DST/data"
    cp "$AIDETECT_SRC/sample_aidetect" "$AIDETECT_DST/"
    chmod +x "$AIDETECT_DST/sample_aidetect"
    # 全部 11 个 .bin 模型
    cp "$AIDETECT_SRC/models/"*.bin "$AIDETECT_DST/models/" 2>/dev/null || true
    # 测试阶段只带一张图（与 [aidetect].image_path 对齐）
    if [ -f "$AIDETECT_SRC/data/hvf_image_hor_1920x1080.yuv" ]; then
        cp "$AIDETECT_SRC/data/hvf_image_hor_1920x1080.yuv" "$AIDETECT_DST/data/"
    fi
    AIDETECT_SIZE=$(du -sh "$AIDETECT_DST" | awk '{print $1}')
    echo "  ✓ aidetect/ ($AIDETECT_SIZE: sample_aidetect + $(ls "$AIDETECT_DST/models/" | wc -l) 个模型 + 1 张测试图)"
else
    echo "  ! 未找到 $AIDETECT_SRC/sample_aidetect，跳过 aidetect 打包"
fi

# 6b. get_frame 工具（Hi3516CV610 sensor 抓帧；wrapper 通过 [get_frame] 配置寻路）
GET_FRAME_SRC="$REPO_ROOT/3layers_sdk/tools/get_frame"
GET_FRAME_DST="$DIST_DIR/get_frame"
if [ -d "$GET_FRAME_SRC" ] && [ -f "$GET_FRAME_SRC/hi3516cv610_get_frame" ]; then
    mkdir -p "$GET_FRAME_DST"
    cp "$GET_FRAME_SRC/hi3516cv610_get_frame" "$GET_FRAME_DST/"
    chmod +x "$GET_FRAME_DST/hi3516cv610_get_frame"
    GET_FRAME_SIZE=$(du -sh "$GET_FRAME_DST" | awk '{print $1}')
    echo "  ✓ get_frame/ ($GET_FRAME_SIZE)"
else
    echo "  ! 未找到 $GET_FRAME_SRC/hi3516cv610_get_frame，跳过 get_frame 打包"
fi

# 6c. skills（snap_detect 组合 skill；install.sh 铺到 ~/.zeroclaw/workspace/skills/）
SKILLS_SRC="$REPO_ROOT/3layers_sdk/tools/skills"
SKILLS_DST="$DIST_DIR/skills"
if [ -d "$SKILLS_SRC" ]; then
    mkdir -p "$SKILLS_DST"
    cp -r "$SKILLS_SRC/." "$SKILLS_DST/"
    echo "  ✓ skills/ ($(ls -d "$SKILLS_DST"/*/ 2>/dev/null | wc -l) 个 skill)"
else
    echo "  ! 未找到 $SKILLS_SRC，跳过 skills 打包"
fi

# 7. 板上 README
cat > "$DIST_DIR/README" <<EOF
zeroclaw + 三级提示词匹配 板上部署包

包内容（install.sh 会统一铺到 \$HOME/.zeroclaw/ 下）：
    bin/zeroclaw                 ARM ELF 主二进制（含 StringMatch L1 + L2 embedding）
    data/l2_vector.db            L2 向量库（规则 + embedding）
    data/l1_string_match.db      L1 字符串匹配规则（regex）
    models/$MODEL_BASENAME
    aidetect/                    Hi3516CV610 NPU 目标检测 binary + 11 模型 + 1 测试图
    get_frame/                   Hi3516CV610 sensor 抓帧 binary
    skills/                      snap_detect skill（组合·pipeline）
    config/zeroclaw.toml         默认配置（含 [string_match] + [prompt_cache]）
    install.sh                   铺到 \$HOME/.zeroclaw/ + 写默认配置 + 配 /etc/profile PATH
    run.sh                       原地跑（不动 / 根目录，用临时 config）

正式安装（推荐）：
    ./install.sh
    zeroclaw agent -m "打开空调"

临时跑（不污染板上 / 根目录）：
    ./run.sh agent -m "打开空调"
    ./run.sh agent -m "看看画面里有没有人"      # 抓帧检测组合

L1/L2 命中时 zeroclaw 不会调云端 provider，零 token。
抓帧检测：aidetect/get_frame 的 JSON 工具已停用（enabled=false），改由 skill + shell
一次性组合调用两个 binary（抓帧→检测→清理）；用户问"画面里有没有人/车/脸"时云端 LLM
按 snap_detect skill 下发一条 shell 命令，成功后由 V6 学习下沉 L1。
设计与边界详见仓库 doc/3layers_sdk/架构设计/多工具组合调用-skill+shell方案.md
EOF

# 8. 文件清单
TOTAL=$(du -sh "$DIST_DIR" | awk '{print $1}')
echo "  ✓ $DIST_DIR ($TOTAL)"
ls -la "$DIST_DIR"
echo
echo "  bin/      $(ls -lh "$DIST_DIR/bin/" | tail -n +2 | awk '{print $5, $9}')"
echo "  data/     $(ls -lh "$DIST_DIR/data/" | tail -n +2 | awk '{print $5, $9}')"
echo "  models/   $(ls -lh "$DIST_DIR/models/" | tail -n +2 | awk '{print $5, $9}')"

# ============ 打 tar 包 ============
echo
echo "[6/6] packing tarball ..."
tar -czf "$DIST_TAR" -C "$(dirname "$DIST_DIR")" "$(basename "$DIST_DIR")"
TAR_SIZE=$(du -sh "$DIST_TAR" | awk '{print $1}')
echo "  ✓ $DIST_TAR ($TAR_SIZE)"

cat <<EOF

════════════════════════════════════════════════════════
完成。板上部署：
    scp $DIST_TAR 板IP:/tmp/
    # 板上：
    cd /tmp && tar xzf $(basename "$DIST_TAR") && cd dist-board
    ./run.sh agent -m "打开空调"
    # 或装到 §7 标准路径：
    ./install.sh
════════════════════════════════════════════════════════
EOF
