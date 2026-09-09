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

output="$(env -i PATH=/definitely-without-git GIT_BRANCH=v0.5.3-ppfix RELEASE_TAG=v0.1.0 \
  /bin/bash -c "log() { :; }
$METADATA_FUNCTION
source_metadata
printf '%s|%s|%s\n' \"\$SOURCE_COMMIT\" \"\$SOURCE_DESCRIBE\" \"\$SOURCE_DIRTY\"")"
[[ "$output" == 'unknown|v0.1.0|unknown' ]] || {
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
