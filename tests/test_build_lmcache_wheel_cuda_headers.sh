#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CUDA 13 may publish cuSPARSE as nvidia-cusparse (without a -cu13 suffix).
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SCRIPT="$REPO_DIR/build_lmcache_wheel.sh"
FUNCTION="$(sed -n '/^collect_nvidia_include_paths()/,/^}/p' "$SCRIPT")"
[[ -n "$FUNCTION" ]] || {
  printf 'collect_nvidia_include_paths helper is missing\n' >&2
  exit 1
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/nvidia/cusparse/include" "$tmp/nvidia_cusparse-12.8.2.51.dist-info"
touch "$tmp/nvidia/cusparse/include/cusparse.h"
cat > "$tmp/nvidia_cusparse-12.8.2.51.dist-info/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: nvidia-cusparse
Version: 12.8.2.51
EOF
cat > "$tmp/nvidia_cusparse-12.8.2.51.dist-info/RECORD" <<'EOF'
nvidia/cusparse/include/cusparse.h,,
EOF

output="$(PYTHONPATH="$tmp" PYTHON="$(command -v python3)" LMCACHE_CUDA_MAJOR=13 bash -c "$FUNCTION
collect_nvidia_include_paths")"
expected="$tmp/nvidia/cusparse/include"
[[ ":$output:" == *":$expected:"* ]] || {
  printf 'CUDA 13 nvidia-cusparse include path was not found: %s\n' "$output" >&2
  exit 1
}
