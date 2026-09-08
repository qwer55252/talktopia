#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${TALKTOPIA_PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing $PYTHON_BIN; run ./install.sh first." >&2
  exit 1
fi

export SOTOPIA_STORAGE_BACKEND="${SOTOPIA_STORAGE_BACKEND:-local}"
export CUSTOM_API_KEY="${CUSTOM_API_KEY:-EMPTY}"

for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    exec "$PYTHON_BIN" -m talktopia.pipeline "$@"
  fi
done

SERVERS_READY=0
cleanup() {
  local pipeline_status=$?
  trap - EXIT
  if [[ "$SERVERS_READY" -eq 1 && "${TALKTOPIA_KEEP_LOCAL_API:-0}" != "1" ]]; then
    "$PYTHON_BIN" -m talktopia.models.servers stop || true
  fi
  exit "$pipeline_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$PYTHON_BIN" -m talktopia.models.servers start
SERVERS_READY=1
"$PYTHON_BIN" -m talktopia.pipeline "$@"
