#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 当前环境构建；使用部署镜像中的 Python/Torch，避免隔离构建替换 Torch。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/built}"
BUILD_JOBS="${BUILD_JOBS:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
BUILD_WITH_MOONCAKE="${BUILD_WITH_MOONCAKE:-0}"
BUILD_WITH_AEROSPIKE="${BUILD_WITH_AEROSPIKE:-0}"
BUILD_RUST="${BUILD_RUST:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
HTTP_PROXY="${HTTP_PROXY-${http_proxy-http://192.168.10.6:3128}}"
HTTPS_PROXY="${HTTPS_PROXY-${https_proxy-$HTTP_PROXY}}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

usage() {
cat <<'EOF'
在当前环境编译 LMCache CUDA wheel（不创建容器）。
bash build_lmcache_wheel.sh

REPO_DIR / PYTHON       源码目录 / 已安装部署用 Torch 的 Python
OUTPUT_DIR             产物目录，默认源码 built/；不删除以前的 wheel
BUILD_JOBS             编译并行度，默认 nproc / 2，最低 1
TORCH_CUDA_ARCH_LIST   默认 10.0（BNT3 场景）；其他 GPU 请指定目标架构
CUDA_HOME              不指定时，优先匹配 Torch 的 CUDA 主次版本
INSTALL_DEPS           1 自动补齐构建依赖；0 仅检查
HTTP_PROXY/HTTPS_PROXY 依赖代理；失败依次尝试本地 8118、直连
PIP_INDEX_URL          Python 源（未设置时使用清华源）
BUILD_WITH_MOONCAKE    默认 0；1 需要 MOONCAKE_INCLUDE_DIR、MOONCAKE_LIB_DIR
BUILD_WITH_AEROSPIKE   默认 0；1 需要 AEROSPIKE_INCLUDE_DIR、AEROSPIKE_LIBRARY_DIR
BUILD_RUST            默认 0；1 另构建 raw_block wheel，需要已安装 cargo/rustc

成功前会检查 wheel CRC、原生扩展及隔离目录导入；不执行 GPU 或存储业务测试。
不自动安装或升级 Torch，不执行缓存服务或裸盘测试。
EOF
}
[[ "${1:-}" != --help && "${1:-}" != -h ]] || { usage; exit 0; }
log() { printf '[build-lmcache] %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# CI 构建镜像可以不带 git：外层已固定分支/发布版本，仍须在产物中保留可追溯信息。
source_metadata() {
  if command -v git >/dev/null && git -C "$REPO_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
    SOURCE_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
    SOURCE_DESCRIBE="$(git -C "$REPO_DIR" describe --tags --always --dirty)"
    SOURCE_DIRTY=false
    [[ -z "$(git -C "$REPO_DIR" status --porcelain)" ]] || SOURCE_DIRTY=true
    return
  fi

  SOURCE_COMMIT="${SOURCE_COMMIT:-${GIT_COMMIT:-${CI_COMMIT_SHA:-unknown}}}"
  SOURCE_DESCRIBE="${SOURCE_DESCRIBE:-${RELEASE_TAG:-${GIT_BRANCH:-unknown}}}"
  SOURCE_DIRTY="${SOURCE_DIRTY:-unknown}"
  log "git 不可用，使用 CI 版本信息：commit=$SOURCE_COMMIT describe=$SOURCE_DESCRIBE"
}

# setuptools-scm 在元数据阶段也会调用 git。无 git 的 CI 镜像需传入 PEP 440 版本。
configure_setuptools_scm() {
  [[ -n "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-}" ]] && return
  if command -v git >/dev/null && git -C "$REPO_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
    return
  fi

  local version="${RELEASE_TAG:-${RELEASE_NAME:-}}"
  [[ "$version" =~ ^v[0-9] ]] && version="${version#v}"
  if ! "$PYTHON" - "$version" <<'PY' >/dev/null 2>&1
from packaging.version import Version
import sys
Version(sys.argv[1])
PY
  then
    version='0+unknown'
  fi
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="$version"
  log "git 不可用，设置 setuptools-scm 版本：$SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE"
}
[[ $# == 0 ]] || die '使用环境变量配置，不支持位置参数'
for flag in INSTALL_DEPS BUILD_WITH_MOONCAKE BUILD_WITH_AEROSPIKE BUILD_RUST; do
  [[ "${!flag}" =~ ^[01]$ ]] || die "$flag 必须为 0 或 1"
done
[[ -f "$REPO_DIR/setup.py" && -d "$REPO_DIR/setup_extensions" ]] || die 'REPO_DIR 不是支持的 LMCache 源码目录'
command -v "$PYTHON" >/dev/null || die '未找到 PYTHON'
REPO_DIR="$(cd "$REPO_DIR" && pwd -P)"
cd "$REPO_DIR"
source_metadata
TORCH_INFO="$("$PYTHON" - <<'PY'
import sys, torch
if not torch.version.cuda:
    raise SystemExit('需要已安装 CUDA 版 Torch；脚本不会替换 Torch')
print(sys.version.split()[0], torch.__version__, torch.version.cuda,
      int(torch._C._GLIBCXX_USE_CXX11_ABI), sep='|')
PY
)" || die '无法读取 Torch/CUDA 信息'
IFS='|' read -r PYTHON_VERSION TORCH_VERSION TORCH_CUDA TORCH_ABI <<< "$TORCH_INFO"
if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in "/usr/local/cuda-$TORCH_CUDA" /usr/local/cuda "$(dirname "$(dirname "$(command -v nvcc || echo /not-found/bin/nvcc)")")"; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      CUDA_HOME="$candidate"
      break
    fi
  done
fi
[[ -x "${CUDA_HOME:-}/bin/nvcc" ]] || die "没有 CUDA Toolkit；请安装与 Torch CUDA $TORCH_CUDA 匹配的 nvcc"
NVCC_VERSION="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
[[ "$NVCC_VERSION" == "$TORCH_CUDA" ]] || die "CUDA 版本不一致：Torch=$TORCH_CUDA nvcc=$NVCC_VERSION；请指定 CUDA_HOME"
[[ -z "${ENABLE_CXX11_ABI:-}" || "$ENABLE_CXX11_ABI" == "$TORCH_ABI" ]] || die 'ENABLE_CXX11_ABI 与 Torch 不一致'
[[ -z "${LMCACHE_CUDA_MAJOR:-}" || "$LMCACHE_CUDA_MAJOR" == "${TORCH_CUDA%%.*}" ]] || die 'LMCACHE_CUDA_MAJOR 与 Torch 不一致'
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
export ENABLE_CXX11_ABI="$TORCH_ABI" LMCACHE_CUDA_MAJOR="${TORCH_CUDA%%.*}" TORCH_CUDA_ARCH_LIST
[[ "$LMCACHE_CUDA_MAJOR" == 12 || "$LMCACHE_CUDA_MAJOR" == 13 ]] || die '该分支只提供 CUDA 12/13 依赖配置'
# 精简开发环境可由 Torch 配套的 NVIDIA wheel 提供 CUDA 库头文件。
# CUDA 13 的 cuSPARSE 当前也会以无 -cu13 后缀的 nvidia-cusparse 发行。
collect_nvidia_include_paths() {
  "$PYTHON" - "$LMCACHE_CUDA_MAJOR" <<'PY_INCLUDES'
import importlib.metadata as metadata
import pathlib
import sys
major=sys.argv[1]
paths=set()
for distribution in metadata.distributions():
    name=(distribution.metadata.get('Name') or '').lower()
    compatible=name.startswith('nvidia-') and name.endswith('-cu'+major)
    compatible |= major == '13' and name == 'nvidia-cusparse'
    if not compatible:
        continue
    for file in distribution.files or []:
        parts=file.parts
        if 'include' in parts:
            directory=pathlib.Path(distribution.locate_file(pathlib.PurePosixPath(*parts[:parts.index('include')+1])))
            if directory.is_dir(): paths.add(str(directory))
print(':'.join(sorted(paths)))
PY_INCLUDES
}
has_cusparse_header() {
  [[ -f "$CUDA_HOME/include/cusparse.h" ]] && return 0
  local directory
  IFS=: read -r -a directories <<< "$1"
  for directory in "${directories[@]}"; do
    [[ -f "$directory/cusparse.h" ]] && return 0
  done
  return 1
}
torch_cusparse_requirement() {
  "$PYTHON" - "$LMCACHE_CUDA_MAJOR" <<'PY_CUSPARSE'
import importlib.metadata as metadata
import sys
from packaging.requirements import Requirement
major=sys.argv[1]
for raw in metadata.distribution('torch').requires or ():
    requirement=Requirement(raw)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    name=requirement.name.lower()
    if name == 'nvidia-cusparse' or name == f'nvidia-cusparse-cu{major}':
        print(str(requirement))
        break
else:
    print('nvidia-cusparse' if major == '13' else f'nvidia-cusparse-cu{major}')
PY_CUSPARSE
}
export BUILD_WITH_CUDA=1 NO_NATIVE_EXT=0 NO_GPU_EXT=0 NO_CUDA_EXT=0
unset BUILD_WITH_HIP BUILD_WITH_SYCL BUILD_WITH_MUSA
export BUILD_WITH_MOONCAKE BUILD_MOONCAKE="$BUILD_WITH_MOONCAKE"
export BUILD_WITH_AEROSPIKE BUILD_AEROSPIKE="$BUILD_WITH_AEROSPIKE"
if [[ -z "$BUILD_JOBS" ]]; then BUILD_JOBS=$(( $(nproc) / 2 )); (( BUILD_JOBS > 0 )) || BUILD_JOBS=1; fi
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || die 'BUILD_JOBS 必须为正整数'
export MAX_JOBS="$BUILD_JOBS"
# 编译目标由 arch 明确指定；避免 Torch 元数据阶段探测繁忙/不可用的 GPU 驱动。
export CUDA_VISIBLE_DEVICES="" PYTORCH_NVML_BASED_CUDA_CHECK=1
log "python=$PYTHON_VERSION torch=$TORCH_VERSION CUDA=$NVCC_VERSION ABI=$TORCH_ABI arch=$TORCH_CUDA_ARCH_LIST jobs=$MAX_JOBS"

# 先检查可选 SDK，避免编译大半后才发现缺少外部库。
for backend in MOONCAKE AEROSPIKE; do
  enabled="BUILD_WITH_$backend"
  [[ "${!enabled}" == 1 ]] || continue
  inc="${backend}_INCLUDE_DIR"
  lib="${backend}_LIB_DIR"
  [[ "$backend" != AEROSPIKE ]] || lib=AEROSPIKE_LIBRARY_DIR
  [[ -n "${!inc:-}" && -n "${!lib:-}" ]] || die "$backend 需要 $inc 和 $lib（多个目录以分号分隔）"
  IFS=';' read -r -a directories <<< "${!inc};${!lib}"
  for directory in "${directories[@]}"; do [[ -d "$directory" ]] || die "SDK 目录不存在：$directory"; done
  export "$inc" "$lib"
done
if [[ "$BUILD_RUST" == 1 ]]; then
  command -v cargo >/dev/null && command -v rustc >/dev/null || die 'BUILD_RUST=1 需要 cargo/rustc'
fi

# 每次尝试使用独立环境，直连模式不会残留大写/小写代理。
with_proxy() {
  local proxy=$1; shift
  if [[ -n "$proxy" ]]; then
    env -u ALL_PROXY -u all_proxy HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy" http_proxy="$proxy" https_proxy="$proxy" "$@"
  else
    env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy "$@"
  fi
}
install_python_deps() {
  local proxy
  for proxy in "$HTTP_PROXY" http://127.0.0.1:8118 ''; do
    if with_proxy "$proxy" "$PYTHON" -m pip install --disable-pip-version-check --timeout 15 --retries 1 \
      --index-url "$PIP_INDEX_URL" "$@"; then return; fi
  done
  die 'Python 构建依赖安装失败'
}
missing=()
for tool in g++ make; do command -v "$tool" >/dev/null || missing+=(build-essential); done
"$PYTHON" -c 'import sysconfig,pathlib; assert (pathlib.Path(sysconfig.get_path("include"))/"Python.h").is_file()' >/dev/null 2>&1 || missing+=("python${PYTHON_VERSION%.*}-dev")
if (( ${#missing[@]} )); then
  [[ "$INSTALL_DEPS" == 1 ]] || die "缺少系统依赖：${missing[*]}"
  [[ $(id -u) == 0 ]] && command -v apt-get >/dev/null || die '自动补齐系统依赖需要 root 和 apt-get'
  apt_ok=0
  for proxy in "$HTTP_PROXY" http://127.0.0.1:8118 ''; do
    if with_proxy "$proxy" apt-get -o "Acquire::http::Proxy=${proxy:-DIRECT}" -o "Acquire::https::Proxy=${proxy:-DIRECT}" -o Acquire::Retries=1 -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 -o APT::Update::Error-Mode=any update &&
      with_proxy "$proxy" env DEBIAN_FRONTEND=noninteractive apt-get -o "Acquire::http::Proxy=${proxy:-DIRECT}" -o "Acquire::https::Proxy=${proxy:-DIRECT}" -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 install -y --no-install-recommends "${missing[@]}"; then apt_ok=1; break; fi
  done
  [[ "$apt_ok" == 1 ]] || die '系统依赖安装失败，请检查 apt 源及网络'
fi
check_python_deps() {
  "$PYTHON" - <<'PY'
import importlib.metadata as m
from packaging.requirements import Requirement
for line in open('requirements/build.txt'):
    line=line.strip()
    if not line or line.startswith('#'): continue
    r=Requirement(line)
    if r.marker and not r.marker.evaluate(): continue
    try: version=m.version(r.name)
    except m.PackageNotFoundError: raise SystemExit('缺少构建依赖 '+r.name)
    if version not in r.specifier: raise SystemExit('构建依赖版本不满足 '+line)
PY
}
if ! check_python_deps; then
  [[ "$INSTALL_DEPS" == 1 ]] || die '构建依赖检查失败'
  install_python_deps -r requirements/build.txt
  check_python_deps
fi
NVIDIA_INCLUDES="$(collect_nvidia_include_paths)"
if ! has_cusparse_header "$NVIDIA_INCLUDES"; then
  [[ "$INSTALL_DEPS" == 1 ]] || die '缺少 cusparse.h；请安装与 Torch 匹配的 cuSPARSE 开发包'
  CUSPARSE_REQUIREMENT="$(torch_cusparse_requirement)"
  log "缺少 cusparse.h，安装 Torch 匹配依赖：$CUSPARSE_REQUIREMENT"
  install_python_deps "$CUSPARSE_REQUIREMENT"
  NVIDIA_INCLUDES="$(collect_nvidia_include_paths)"
  has_cusparse_header "$NVIDIA_INCLUDES" || die '安装 cuSPARSE 后仍未找到 cusparse.h'
fi
[[ -z "$NVIDIA_INCLUDES" ]] || export CPATH="$NVIDIA_INCLUDES${CPATH:+:$CPATH}"
if [[ "$BUILD_RUST" == 1 ]] && ! "$PYTHON" -c 'import importlib.metadata as m; from packaging.version import Version; assert Version(m.version("maturin")) >= Version("1.8")' >/dev/null 2>&1; then
  [[ "$INSTALL_DEPS" == 1 ]] || die '缺少 maturin>=1.8'
  install_python_deps 'maturin>=1.8'
fi
configure_setuptools_scm
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"
stage="$(mktemp -d "$OUTPUT_DIR/.lmcache-build.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
"$PYTHON" -m pip wheel "$REPO_DIR" --no-build-isolation --no-deps --no-cache-dir --wheel-dir "$stage" -v
if [[ "$BUILD_RUST" == 1 ]]; then
  "$PYTHON" -m maturin build --release --interpreter "$PYTHON" --manifest-path "$REPO_DIR/rust/raw_block/Cargo.toml" --out "$stage"
fi

# 校验本次产物，避免用上次遗留 wheel 冒充成功；不导入源码目录里的旧 .so。
"$PYTHON" - "$stage" "$BUILD_WITH_MOONCAKE" "$BUILD_WITH_AEROSPIKE" "$BUILD_RUST" <<'PY_VALIDATE'
import importlib.machinery
import pathlib
import subprocess
import sys
import tempfile
import zipfile
stage=pathlib.Path(sys.argv[1])
expected=['c_ops','native_storage_ops','lmcache_redis','lmcache_fs']
if sys.argv[2]=='1': expected.append('lmcache_mooncake')
if sys.argv[3]=='1': expected.append('lmcache_aerospike')
wheels=list(stage.glob('lmcache-*.whl'))
if len(wheels)!=1: raise SystemExit('必须恰好生成一个 LMCache wheel')
with tempfile.TemporaryDirectory() as extracted:
    with zipfile.ZipFile(wheels[0]) as archive:
        if archive.testzip(): raise SystemExit('LMCache wheel CRC 校验失败')
        names={i.filename:i.file_size for i in archive.infolist()}
        for module in expected:
            if not any(names.get('lmcache/'+module+suffix,0)>0 for suffix in importlib.machinery.EXTENSION_SUFFIXES):
                raise SystemExit('缺少或为空的扩展：'+module)
        archive.extractall(extracted)
    # 直接载入原生子模块，避免为验证链接而安装整套业务依赖。
    code='''import importlib, pathlib, sys, types, torch
root=pathlib.Path(sys.argv[1]).resolve()
pkg=types.ModuleType('lmcache'); pkg.__path__=[str(root/'lmcache')]; sys.modules['lmcache']=pkg
for name in sys.argv[2:]:
    mod=importlib.import_module('lmcache.'+name)
    assert pathlib.Path(mod.__file__).resolve().is_relative_to(root)
    print('EXTENSION_IMPORT_OK='+name, flush=True)
'''
    subprocess.run([sys.executable,'-I','-c',code,extracted,*expected],cwd=extracted,check=True,timeout=120)
if sys.argv[4]=='1':
    rust=list(stage.glob('lmcache_rust_raw_block_io-*.whl'))
    if len(rust)!=1:raise SystemExit('缺少 Rust wheel')
    with zipfile.ZipFile(rust[0]) as archive:
        if archive.testzip() or not any(i.filename.endswith('.so') and i.file_size>0 for i in archive.infolist()):
            raise SystemExit('Rust wheel 校验失败')
print('WHEEL_CONTENTS_OK=1')
PY_VALIDATE
cat > "$stage/BUILD-INFO" <<EOF
SOURCE_COMMIT=$SOURCE_COMMIT
SOURCE_DESCRIBE=$SOURCE_DESCRIBE
SOURCE_DIRTY=$SOURCE_DIRTY
PYTHON=$PYTHON_VERSION
TORCH=$TORCH_VERSION
CUDA=$NVCC_VERSION
CUDA_HOME=$CUDA_HOME
CXX11_ABI=$TORCH_ABI
TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
BUILD_JOBS=$BUILD_JOBS
BUILD_WITH_MOONCAKE=$BUILD_WITH_MOONCAKE
BUILD_WITH_AEROSPIKE=$BUILD_WITH_AEROSPIKE
BUILD_RUST=$BUILD_RUST
EOF
for wheel in "$stage"/*.whl; do
  target="$OUTPUT_DIR/$(basename "$wheel")"
  mv "$wheel" "$target"
  sha256sum "$target"
  printf 'WHEEL=%s\n' "$target"
done
mv "$stage/BUILD-INFO" "$OUTPUT_DIR/BUILD-INFO"
log '编译与原生扩展导入校验完成'
