#!/usr/bin/env bash
set -euo pipefail

UV_VERSION="0.11.26"
UV_ARCHIVE="uv-aarch64-apple-darwin.tar.gz"
UV_SHA256="8f7fbf1708399b921857bce71e1d60f0d3ccf52a30caebc1c1a2f175dce13ab6"
UV_URL="https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/${UV_ARCHIVE}"

NODE_VERSION="24.18.0"
NODE_ARCHIVE="node-v${NODE_VERSION}-darwin-arm64.tar.gz"
NODE_SHA256="e1a97e14c99c803e96c7339403282ea05a499c32f8d83defe9ef5ec66f979ed1"
NODE_URL="https://nodejs.org/download/release/v${NODE_VERSION}/${NODE_ARCHIVE}"

PYTHON_VERSION="3.12"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN_ROOT="${KREA_2_TURBO_MLX_TOOLCHAIN_DIR:-$PROJECT_ROOT/.toolchain}"
DOWNLOAD_DIR="$TOOLCHAIN_ROOT/downloads"
UV_HOME="$TOOLCHAIN_ROOT/uv-$UV_VERSION"
NODE_HOME="$TOOLCHAIN_ROOT/node-v$NODE_VERSION"
PYTHON_INSTALL_DIR="$TOOLCHAIN_ROOT/python"

log() {
  printf '%s\n' "$*" >&2
}

require_macos_arm64() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    log "krea-2-turbo-mlx requires macOS on Apple Silicon."
    exit 1
  fi

  case "$(uname -m)" in
    arm64|aarch64) ;;
    *)
      log "krea-2-turbo-mlx requires Apple Silicon; found $(uname -m)."
      exit 1
      ;;
  esac
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "$1 is required to bootstrap krea-2-turbo-mlx."
    exit 1
  fi
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

verify_file() {
  local file="$1"
  local expected="$2"
  local actual

  [[ -f "$file" ]] || return 1
  actual="$(sha256_file "$file")"
  [[ "$actual" == "$expected" ]]
}

download_verified() {
  local url="$1"
  local expected="$2"
  local dest="$3"
  local tmp

  mkdir -p "$DOWNLOAD_DIR"
  if verify_file "$dest" "$expected"; then
    return 0
  fi

  tmp="$dest.tmp"
  rm -f "$tmp"
  log "Downloading $(basename "$dest")..."
  curl --fail --location --retry 3 --connect-timeout 20 --output "$tmp" "$url"

  if ! verify_file "$tmp" "$expected"; then
    rm -f "$tmp"
    log "Checksum verification failed for $(basename "$dest")."
    exit 1
  fi

  mv "$tmp" "$dest"
}

ensure_uv() {
  local archive="$DOWNLOAD_DIR/$UV_ARCHIVE"
  local tmpdir="$TOOLCHAIN_ROOT/.uv-extract"
  local uv_src
  local uvx_src

  require_macos_arm64
  require_command curl
  require_command shasum
  require_command tar
  require_command awk

  if [[ -x "$UV_HOME/uv" ]] && [[ "$("$UV_HOME/uv" --version 2>/dev/null || true)" == "uv $UV_VERSION"* ]]; then
    printf '%s\n' "$UV_HOME/uv"
    return 0
  fi

  download_verified "$UV_URL" "$UV_SHA256" "$archive"
  rm -rf "$UV_HOME" "$tmpdir"
  mkdir -p "$UV_HOME" "$tmpdir"
  tar -xzf "$archive" -C "$tmpdir"

  uv_src="$(find "$tmpdir" -type f -name uv -print -quit)"
  uvx_src="$(find "$tmpdir" -type f -name uvx -print -quit)"
  if [[ ! -f "$uv_src" ]]; then
    log "Downloaded uv archive did not contain a uv binary."
    exit 1
  fi

  cp "$uv_src" "$UV_HOME/uv"
  chmod 755 "$UV_HOME/uv"
  if [[ -f "$uvx_src" ]]; then
    cp "$uvx_src" "$UV_HOME/uvx"
    chmod 755 "$UV_HOME/uvx"
  fi
  rm -rf "$tmpdir"

  printf '%s\n' "$UV_HOME/uv"
}

ensure_python() {
  local uv_bin

  uv_bin="$(ensure_uv)"
  mkdir -p "$PYTHON_INSTALL_DIR"
  UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR" "$uv_bin" python install --no-bin "$PYTHON_VERSION" >&2
  printf '%s\n' "$PYTHON_VERSION"
}

ensure_node() {
  local archive="$DOWNLOAD_DIR/$NODE_ARCHIVE"
  local tmpdir="$TOOLCHAIN_ROOT/.node-extract"
  local root

  require_macos_arm64
  require_command curl
  require_command shasum
  require_command tar
  require_command awk

  if [[ -x "$NODE_HOME/bin/node" ]] && [[ "$("$NODE_HOME/bin/node" --version 2>/dev/null || true)" == "v$NODE_VERSION" ]]; then
    printf '%s\n' "$NODE_HOME/bin/node"
    return 0
  fi

  download_verified "$NODE_URL" "$NODE_SHA256" "$archive"
  rm -rf "$NODE_HOME" "$tmpdir"
  mkdir -p "$tmpdir"
  tar -xzf "$archive" -C "$tmpdir"
  root="$(find "$tmpdir" -mindepth 1 -maxdepth 1 -type d -name "node-v*" -print -quit)"
  if [[ ! -d "$root" ]]; then
    log "Downloaded Node archive did not contain the expected directory."
    exit 1
  fi
  mv "$root" "$NODE_HOME"
  rm -rf "$tmpdir"

  printf '%s\n' "$NODE_HOME/bin/node"
}

usage() {
  cat >&2 <<'EOF'
Usage: scripts/toolchain.sh uv|python|node|npm
EOF
}

case "${1:-}" in
  uv)
    ensure_uv
    ;;
  python)
    ensure_python
    ;;
  node)
    ensure_node
    ;;
  npm)
    node_bin="$(ensure_node)"
    printf '%s\n' "$(dirname "$node_bin")/npm"
    ;;
  *)
    usage
    exit 2
    ;;
esac
