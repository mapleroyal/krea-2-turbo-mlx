#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "krea-2-turbo-mlx setup requires macOS for MLX."
  exit 1
fi

machine="$(uname -m)"
if [[ "$machine" != "arm64" && "$machine" != "aarch64" ]]; then
  echo "krea-2-turbo-mlx setup requires Apple Silicon; found $machine."
  exit 1
fi

TOOLCHAIN_DIR="${KREA_2_TURBO_MLX_TOOLCHAIN_DIR:-$PWD/.toolchain}"
mkdir -p "$TOOLCHAIN_DIR"
TOOLCHAIN_DIR="$(cd "$TOOLCHAIN_DIR" && pwd -P)"
export KREA_2_TURBO_MLX_TOOLCHAIN_DIR="$TOOLCHAIN_DIR"
export UV_PYTHON_INSTALL_DIR="$TOOLCHAIN_DIR/python"

venv_needs_rebuild() {
  local cfg_home

  [[ -d ".venv" ]] || return 1
  [[ -f ".venv/pyvenv.cfg" ]] || return 0
  [[ -x ".venv/bin/python" ]] || return 0
  ".venv/bin/python" -c "import sys" >/dev/null 2>&1 || return 0

  cfg_home="$(awk -F ' = ' '$1 == "home" { print $2; exit }' ".venv/pyvenv.cfg")"
  if [[ -z "$cfg_home" ]]; then
    return 0
  fi

  case "$cfg_home" in
    "$TOOLCHAIN_DIR"/python/*) return 1 ;;
    *) return 0 ;;
  esac
}

UV_BIN="$(./scripts/toolchain.sh uv)"
./scripts/toolchain.sh python >/dev/null

if venv_needs_rebuild; then
  echo "Rebuilding .venv because it no longer matches this checkout."
  rm -rf ".venv"
fi

"$UV_BIN" venv --allow-existing --python 3.12 .venv
"$UV_BIN" sync --locked --no-dev --extra runtime --python ".venv/bin/python"

if [[ -f "frontend/package.json" && ! -f "frontend/build/client/index.html" ]]; then
  if [[ "${KREA_2_TURBO_MLX_BUILD_FRONTEND:-0}" == "1" ]]; then
    NODE_BIN="$(./scripts/toolchain.sh node)"
    export PATH="$(dirname "$NODE_BIN"):$PATH"
    npm ci --prefix frontend
    npm run build --prefix frontend
  else
    echo "The included React GUI build is missing: frontend/build/client/index.html." >&2
    echo "Use a release checkout that includes the GUI build, or set KREA_2_TURBO_MLX_BUILD_FRONTEND=1 to rebuild it." >&2
    exit 1
  fi
fi

exec ".venv/bin/krea-2-turbo-mlx" setup "$@"
