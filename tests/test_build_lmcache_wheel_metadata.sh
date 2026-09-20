#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Verify CI metadata is usable when the builder image intentionally has no git.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SCRIPT="$REPO_DIR/build_lmcache_wheel.sh"
METADATA_FUNCTION="$(sed -n '/^source_metadata()/,/^}/p' "$SCRIPT")"
[[ -n "$METADATA_FUNCTION" ]] || {
  printf 'source_metadata helper is missing\n' >&2
  exit 1
}
SCM_FUNCTION="$(sed -n '/^configure_setuptools_scm()/,/^}/p' "$SCRIPT")"
[[ -n "$SCM_FUNCTION" ]] || {
  printf 'configure_setuptools_scm helper is missing\n' >&2
  exit 1
}

output="$(env -i PATH=/definitely-without-git GIT_BRANCH=v0.5.3-ppfix RELEASE_TAG=v0.1.0 \
  PYTHON="$(command -v python3)" /bin/bash -c "log() { :; }
$METADATA_FUNCTION
$SCM_FUNCTION
source_metadata
configure_setuptools_scm
printf '%s|%s|%s|%s\n' \"\$SOURCE_COMMIT\" \"\$SOURCE_DESCRIBE\" \"\$SOURCE_DIRTY\" \"\$SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE\"")"
[[ "$output" == 'unknown|v0.1.0|unknown|0.1.0' ]] || {
  printf 'unexpected metadata fallback: %s\n' "$output" >&2
  exit 1
}

expected_commit="$(git -C "$REPO_DIR" rev-parse HEAD)"
output="$(REPO_DIR="$REPO_DIR" bash -c "log() { :; }
$METADATA_FUNCTION
source_metadata
printf '%s\n' \"\$SOURCE_COMMIT\"")"
[[ "$output" == "$expected_commit" ]] || {
  printf 'git metadata was not preserved: %s\n' "$output" >&2
  exit 1
}

output="$(REPO_DIR="$REPO_DIR" WHEEL_VERSION=v0.5.3+glm PYTHON="$(command -v python3)" bash -c "log() { :; }
die() { printf '%s\\n' \"\$*\" >&2; exit 1; }
$SCM_FUNCTION
configure_setuptools_scm
printf '%s\n' \"\$SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE\"")"
[[ "$output" == '0.5.3+glm' ]] || {
  printf 'explicit wheel version did not override git metadata: %s\n' "$output" >&2
  exit 1
}

if REPO_DIR="$REPO_DIR" WHEEL_VERSION='not a version!' PYTHON="$(command -v python3)" bash -c "log() { :; }
die() { printf '%s\\n' \"\$*\" >&2; exit 1; }
$SCM_FUNCTION
configure_setuptools_scm" >/dev/null 2>&1; then
  printf 'invalid explicit wheel version was accepted\n' >&2
  exit 1
fi
