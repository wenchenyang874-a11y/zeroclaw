# CMake toolchain for Hi3516CV610 (Cortex-A7 / musl / soft-float)
# 与 610llama/doc/610llama/交叉编译/arm-hisilicon-a7.cmake 同源，本地副本启用了 GGML_OPENMP=OFF
# （musl 工具链没有 OpenMP，原文档 §"三条铁律"明确要求）。
#
# llama-cpp-sys-2 build.rs 用 cmake-rs，只透传 CMAKE_* 环境变量；GGML_* 必须从 toolchain 文件
# 注入 cache。所以这里 set(... CACHE BOOL ... FORCE) 是关键。

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_SYSTEM_VERSION 1)

# Cross-compiler toolchain (要求 PATH 包含
# /opt/linux/x86-arm/gcc-20250305-arm-v01c02-linux-musleabi/arm-v01c02-linux-musleabi-gcc/bin)
set(TOOLCHAIN_PREFIX "arm-v01c02-linux-musleabi")
set(CMAKE_C_COMPILER  ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_AR          ${TOOLCHAIN_PREFIX}-ar)
set(CMAKE_RANLIB      ${TOOLCHAIN_PREFIX}-ranlib)

# Search programs only in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Search libs/headers in the target sysroot
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY  ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE  ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE  ONLY)

# A7 + soft-float ABI（绝对不能加 -mfpu=neon-* / -mfloat-abi=hard）
set(ARM_FLAGS "-march=armv7-a -mfloat-abi=soft -mtune=cortex-a7 -O2")
set(CMAKE_C_FLAGS_INIT             "${ARM_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT           "${ARM_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_INIT    "${ARM_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${ARM_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${ARM_FLAGS}")

# llama.cpp 嵌入式必备开关（CLAUDE.md 三条铁律）
set(GGML_OPENMP   OFF CACHE BOOL "Disable OpenMP (musl toolchain has none)" FORCE)
set(GGML_NATIVE   OFF CACHE BOOL "Disable -march=native"                    FORCE)
set(GGML_BLAS     OFF CACHE BOOL "Disable BLAS (no host BLAS, save size)"   FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Static link"                          FORCE)
