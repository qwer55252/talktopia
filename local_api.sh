#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

PYTHON_BIN="${TALKTOPIA_PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing $PYTHON_BIN; run ./install.sh first." >&2
  exit 1
fi

export CUSTOM_API_KEY="${CUSTOM_API_KEY:-EMPTY}"
"$PYTHON_BIN" -m talktopia.models.servers "${1:-status}"
