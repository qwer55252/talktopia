#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$ROOT/engine"
VENV_DIR="${TALKTOPIA_VENV:-$ROOT/.venv}"
PATCH_FILE="$ROOT/patches/sotopia-local-models.patch"
REQUIREMENTS_LOCK="$ROOT/requirements.lock"

source "$ROOT/engine.lock"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
BOOTSTRAP_PYTHON="${TALKTOPIA_BOOTSTRAP_PYTHON:-python3}"

if [[ ! -e "$ENGINE_DIR" ]]; then
  git init -q "$ENGINE_DIR"
  git -C "$ENGINE_DIR" remote add origin "$SOTOPIA_REPOSITORY"
  git -C "$ENGINE_DIR" fetch -q --depth 1 origin "$SOTOPIA_COMMIT"
  git -C "$ENGINE_DIR" checkout -q --detach FETCH_HEAD
elif [[ ! -d "$ENGINE_DIR/.git" ]]; then
  echo "$ENGINE_DIR exists but is not a Git checkout" >&2
  exit 1
fi

ACTUAL_COMMIT="$(git -C "$ENGINE_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$SOTOPIA_COMMIT" ]]; then
  echo "engine is at $ACTUAL_COMMIT; expected $SOTOPIA_COMMIT" >&2
  echo "Move or remove $ENGINE_DIR, then run install.sh again." >&2
  exit 1
fi

if git -C "$ENGINE_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "SOTOPIA patch already applied"
elif git -C "$ENGINE_DIR" apply --check "$PATCH_FILE"; then
  git -C "$ENGINE_DIR" apply "$PATCH_FILE"
  echo "Applied SOTOPIA patch"
else
  echo "SOTOPIA patch is partially applied or conflicts with engine files" >&2
  exit 1
fi
git -C "$ENGINE_DIR" diff --check

if [[ -n "${TALKTOPIA_PYTHON:-}" ]]; then
  PYTHON_BIN="$(command -v "$TALKTOPIA_PYTHON" 2>/dev/null || true)"
  if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "TALKTOPIA_PYTHON is not executable: $TALKTOPIA_PYTHON" >&2
    exit 1
  fi
elif [[ -x "$VENV_DIR/bin/python" ]] && "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  PYTHON_BIN="$VENV_DIR/bin/python"
else
  command -v "$BOOTSTRAP_PYTHON" >/dev/null || {
    echo "$BOOTSTRAP_PYTHON is required" >&2
    exit 1
  }

  VENV_EXISTED=0
  if [[ -e "$VENV_DIR" ]]; then
    VENV_EXISTED=1
  fi

  if "$BOOTSTRAP_PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13)' \
    && "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR" >/dev/null 2>&1; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  else
    if [[ "$VENV_EXISTED" -eq 1 ]]; then
      echo "$VENV_DIR exists but is not a usable Python environment" >&2
      exit 1
    fi
    if [[ -e "$VENV_DIR" ]]; then
      rm -rf -- "$VENV_DIR"
    fi

    echo "Standard venv unavailable; using Conda for Python 3.11"
    CONDA_BIN="${CONDA_EXE:-}"
    if [[ -z "$CONDA_BIN" ]]; then
      CONDA_BIN="$(command -v conda 2>/dev/null || true)"
    fi
    if [[ -z "$CONDA_BIN" && -x "$HOME/miniconda3/bin/conda" ]]; then
      CONDA_BIN="$HOME/miniconda3/bin/conda"
    fi
    if [[ -z "$CONDA_BIN" && -x "$HOME/anaconda3/bin/conda" ]]; then
      CONDA_BIN="$HOME/anaconda3/bin/conda"
    fi
    if [[ -z "$CONDA_BIN" ]]; then
      echo "Could not create a venv and Conda was not found." >&2
      echo "Install python3-venv or set TALKTOPIA_PYTHON to a Python 3.10-3.12 executable." >&2
      exit 1
    fi
    "$CONDA_BIN" create --quiet -y -p "$VENV_DIR" python=3.11 pip
    PYTHON_BIN="$VENV_DIR/bin/python"
  fi
  "$PYTHON_BIN" -m pip install --quiet --upgrade pip
fi

"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Python 3.10-3.12 is required"'
"$PYTHON_BIN" -m pip install --quiet -r "$REQUIREMENTS_LOCK"
"$PYTHON_BIN" -m pip install --quiet --no-deps -e "$ENGINE_DIR"
"$PYTHON_BIN" -m pip install --quiet --no-deps -e "$ROOT"

INSTALLED_PATH="$(SOTOPIA_STORAGE_BACKEND=local "$PYTHON_BIN" -c 'from pathlib import Path; import sotopia; print(Path(sotopia.__file__).resolve())')"
case "$INSTALLED_PATH" in
  "$ENGINE_DIR"/sotopia/*) ;;
  *) echo "sotopia is not imported from editable engine: $INSTALLED_PATH" >&2; exit 1 ;;
esac
SOTOPIA_STORAGE_BACKEND=local "$PYTHON_BIN" -c 'from sotopia.generation_utils.generate import _strip_thinking_tags; assert _strip_thinking_tags("<think>x</think>{\"ok\": true}") == "{\"ok\": true}"'
TALKTOPIA_PATH="$("$PYTHON_BIN" -c 'from pathlib import Path; import talktopia; print(Path(talktopia.__file__).resolve())')"
case "$TALKTOPIA_PATH" in
  "$ROOT"/talktopia/*) ;;
  *) echo "talktopia is not imported from this checkout: $TALKTOPIA_PATH" >&2; exit 1 ;;
esac

"$ROOT/link_sources.sh"

echo
echo "Talktopia installation complete"
echo "  engine:    $SOTOPIA_COMMIT"
echo "  python:    $PYTHON_BIN"
echo "  sotopia:   $INSTALLED_PATH"
echo "  talktopia: $TALKTOPIA_PATH"
