#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH=".krea-2-turbo-mlx/config.json"
CLI_PATH=".venv/bin/krea-2-turbo-mlx"
GUI_ARGS=()
CONFIG_PROVIDED=0

cli_ready() {
  [[ -x ".venv/bin/python" ]] || return 1
  [[ -x "$CLI_PATH" ]] || return 1
  ".venv/bin/python" -c "import sys" >/dev/null 2>&1 || return 1
  "$CLI_PATH" --version >/dev/null 2>&1 || return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "--config requires a path." >&2
        exit 2
      fi
      CONFIG_PATH="$2"
      GUI_ARGS+=("$1" "$2")
      CONFIG_PROVIDED=1
      shift 2
      ;;
    --config=*)
      CONFIG_PATH="${1#--config=}"
      GUI_ARGS+=("$1")
      CONFIG_PROVIDED=1
      shift
      ;;
    *)
      GUI_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$CONFIG_PROVIDED" -eq 0 ]]; then
  GUI_ARGS=(--config "$CONFIG_PATH" "${GUI_ARGS[@]}")
fi

if ! cli_ready || [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Preparing Krea 2 Turbo..."
  # Honor the saved setup choices (including whether to keep the source download);
  # do not force --cleanup-source here or it would override a saved keep preference.
  ./setup.sh --config "$CONFIG_PATH" --accept-defaults
fi

exec "$CLI_PATH" gui "${GUI_ARGS[@]}"
