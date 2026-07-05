from __future__ import annotations

import platform
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = shlex.quote(sys.executable)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def _copy_setup_script_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "scripts").mkdir()
    shutil.copy(ROOT / "setup.sh", project / "setup.sh")
    shutil.copy(ROOT / "scripts" / "launch.sh", project / "scripts" / "launch.sh")
    (project / "setup.sh").chmod(0o755)
    (project / "scripts" / "launch.sh").chmod(0o755)
    return project


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"},
    reason="setup.sh is intentionally macOS Apple Silicon only",
)
def test_setup_rebuilds_relocated_venv(tmp_path: Path) -> None:
    project = _copy_setup_script_project(tmp_path)
    old_project = tmp_path / "attempt-2"
    stale_venv = project / ".venv"
    (stale_venv / "bin").mkdir(parents=True)
    (stale_venv / "stale-marker").write_text("old", encoding="utf-8")
    (stale_venv / "pyvenv.cfg").write_text(
        f"home = {old_project}/.toolchain/python/cpython-3.12/bin\n",
        encoding="utf-8",
    )
    (stale_venv / "bin" / "python").symlink_to(
        old_project / ".toolchain" / "python" / "cpython-3.12" / "bin" / "python3.12"
    )

    _write_executable(
        project / "scripts" / "toolchain.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        case "${1:-}" in
          uv)
            printf '%s\\n' "$PWD/fake-uv"
            ;;
          python)
            mkdir -p "$KREA_2_TURBO_MLX_TOOLCHAIN_DIR/python"
            printf '3.12\\n'
            ;;
          *)
            exit 2
            ;;
        esac
        """,
    )
    _write_executable(
        project / "fake-uv",
        f"""
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\\n' "$*" >> uv.log
        case "${{1:-}}" in
          venv)
            venv="${{@: -1}}"
            mkdir -p "$venv/bin"
            ln -sf {PYTHON} "$venv/bin/python"
            ln -sf python "$venv/bin/python3"
            cat > "$venv/pyvenv.cfg" <<CFG
        home = $PWD/.toolchain/python/cpython-3.12-macos-aarch64-none/bin
        implementation = CPython
        version_info = 3.12
        include-system-site-packages = false
        CFG
            cat > "$venv/bin/krea-2-turbo-mlx" <<'CLI'
        #!/usr/bin/env bash
        printf '%s\\n' "$*" >> setup-command.log
        exit 0
        CLI
            chmod +x "$venv/bin/krea-2-turbo-mlx"
            ;;
          sync)
            ;;
        esac
        """,
    )

    result = subprocess.run(
        [str(project / "setup.sh"), "--accept-defaults"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Rebuilding .venv" in result.stdout
    assert not (project / ".venv" / "stale-marker").exists()
    assert (project / "setup-command.log").read_text(encoding="utf-8").splitlines() == [
        "setup --accept-defaults"
    ]


def test_launcher_runs_setup_when_cli_entrypoint_has_stale_shebang(tmp_path: Path) -> None:
    project = _copy_setup_script_project(tmp_path)
    config = project / ".krea-2-turbo-mlx" / "config.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")
    (project / ".venv" / "bin").mkdir(parents=True)
    (project / ".venv" / "bin" / "python").symlink_to(sys.executable)
    _write_executable(
        project / ".venv" / "bin" / "krea-2-turbo-mlx",
        """
        #!/missing/old-checkout/.venv/bin/python3
        """,
    )
    _write_executable(
        project / "setup.sh",
        f"""
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\\n' "$*" >> setup-command.log
        mkdir -p .venv/bin
        ln -sf {PYTHON} .venv/bin/python
        cat > .venv/bin/krea-2-turbo-mlx <<'CLI'
        #!/usr/bin/env bash
        if [[ "${{1:-}}" == "--version" ]]; then
          exit 0
        fi
        printf '%s\\n' "$*" >> gui-command.log
        exit 0
        CLI
        chmod +x .venv/bin/krea-2-turbo-mlx
        """,
    )

    result = subprocess.run(
        [str(project / "scripts" / "launch.sh"), "--no-browser"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Preparing Krea 2 Turbo..." in result.stdout
    assert (project / "setup-command.log").read_text(encoding="utf-8").splitlines() == [
        "--config .krea-2-turbo-mlx/config.json --accept-defaults"
    ]
    assert (project / "gui-command.log").read_text(encoding="utf-8").splitlines() == [
        "gui --config .krea-2-turbo-mlx/config.json --no-browser"
    ]
