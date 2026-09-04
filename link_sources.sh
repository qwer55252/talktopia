#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOTOPIA_DATA_DIR="$HOME/.sotopia/data"
GEMINILIGHT_DATA_DIR="${TALKTOPIA_GEMINILIGHT_DATA_DIR:-$HOME/.cache/talktopia/geminilight_sotopia_dataset}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS:-$HOME/.ollama/models}"

link_source() {
  local target="$1"
  local link="$2"

  if [[ -e "$link" && ! -L "$link" ]]; then
    echo "Refusing to replace non-symlink path: $link" >&2
    exit 1
  fi
  mkdir -p "$target" "$(dirname "$link")"
  ln -sfn "$target" "$link"
  printf '  %-38s -> %s\n' "${link#"$ROOT"/}" "$target"
}

echo "Talktopia project sources:"
link_source "$SOTOPIA_DATA_DIR" "$ROOT/data/sotopia_db"
link_source "$GEMINILIGHT_DATA_DIR" "$ROOT/data/geminilight_sotopia_dataset"
link_source "$OLLAMA_MODELS_DIR" "$ROOT/checkpoints/ollama_models"
