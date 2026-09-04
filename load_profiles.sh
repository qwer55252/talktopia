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

export SOTOPIA_STORAGE_BACKEND="${SOTOPIA_STORAGE_BACKEND:-local}"
"$REPO_ROOT/link_sources.sh"
"$PYTHON_BIN" -m talktopia.load_geminilight_profiles "$@"
