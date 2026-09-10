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
  if [[ "$arg" == "--help" || "$arg" == "-h" || "$arg" == "--dry-run" ]]; then
    exec "$PYTHON_BIN" -m talktopia.pipeline "$@"
  fi
done

# These invocations share the same managed model servers.
exec 9>"$REPO_ROOT/.pipeline.lock"
if ! flock -n 9; then
  echo "Another run_pipeline.sh is already running; no servers were changed." >&2
  exit 1
fi

SERVERS_READY=0
# Validate before starting servers, and restore the frozen server settings on resume.
RUNTIME_SETTINGS=$("$PYTHON_BIN" -c '
import json, os, sys
from talktopia.pipeline import parse_args
args = parse_args(sys.argv[1:])
runtime = {}
if args.resume_run:
    saved = json.loads((args.resume_run / "run_config.json").read_text())
    runtime = {**saved.get("speech_runtime", {}), **saved.get("runtime", {})}
settings = [("OLLAMA_NUM_PARALLEL", "ollama_num_parallel", 2),
            ("OLLAMA_CONTEXT_LENGTH", "ollama_context_length", 32768),
            ("TALKTOPIA_TTS_BATCH_SIZE", "tts_batch_size", 1),
            ("TALKTOPIA_SPEECH_WORKERS_PER_GPU", "speech_workers_per_gpu", 2)]
values = []
for env, key, default in settings:
    value = int(runtime.get(key, os.environ.get(env, default)))
    if value <= 0 or (key in runtime and env in os.environ and int(os.environ[env]) != value):
        raise SystemExit(f"Invalid or conflicting saved setting: {env}")
    values.append(str(value))
print(" ".join(values))
' "$@")
read -r OLLAMA_NUM_PARALLEL OLLAMA_CONTEXT_LENGTH TALKTOPIA_TTS_BATCH_SIZE TALKTOPIA_SPEECH_WORKERS_PER_GPU <<< "$RUNTIME_SETTINGS"
export OLLAMA_NUM_PARALLEL OLLAMA_CONTEXT_LENGTH TALKTOPIA_TTS_BATCH_SIZE TALKTOPIA_SPEECH_WORKERS_PER_GPU

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
# The module runs simulation and automatic evaluation before server cleanup.
"$PYTHON_BIN" -m talktopia.pipeline "$@"
