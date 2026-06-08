#!/usr/bin/env python3
"""Bootstrap Argus for local use."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def step(message: str) -> None:
    print(f"[argus-setup] {message}", flush=True)


def venv_python(root: Path) -> Path:
    if platform.system() == "Windows":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def run(command: list[str], root: Path, dry_run: bool) -> None:
    step(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(root), check=True)


def ensure_venv(root: Path, dry_run: bool) -> Path:
    python = venv_python(root)
    if python.exists():
        step(f"using existing virtual environment: {root / '.venv'}")
        return python
    if dry_run:
        step(f"would create virtual environment: {root / '.venv'}")
    else:
        step(f"creating virtual environment: {root / '.venv'}")
        venv.EnvBuilder(with_pip=True).create(root / ".venv")
    return python


def install_dependencies(root: Path, python: Path, dry_run: bool) -> None:
    requirements = root / "requirements.txt"
    if not requirements.exists():
        step("requirements.txt not found; skipping dependency install")
        return
    run([str(python), "-m", "pip", "install", "-r", str(requirements)], root, dry_run)


def ensure_env(root: Path, dry_run: bool) -> None:
    env_path = root / ".env"
    example_path = root / ".env.example"
    if env_path.exists():
        step(".env already exists")
        return
    if not example_path.exists():
        step(".env.example not found; create .env manually")
        return
    if dry_run:
        step("would create .env from .env.example")
    else:
        step("creating .env from .env.example")
        shutil.copyfile(example_path, env_path)
    step("edit .env and add your API keys before starting Argus")


def install_launcher(root: Path, python: Path, force: bool, dry_run: bool) -> None:
    runner = python if python.exists() else Path(sys.executable)
    command = [str(runner), str(root / "scripts" / "install_argus.py")]
    if force:
        command.append("--force")
    if dry_run:
        command.append("--dry-run")
    run(command, root, dry_run=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up Argus locally.")
    parser.add_argument("--force", action="store_true", help="replace an existing argus launcher")
    parser.add_argument("--skip-deps", action="store_true", help="do not install Python dependencies")
    parser.add_argument("--skip-env", action="store_true", help="do not create .env from .env.example")
    parser.add_argument("--dry-run", action="store_true", help="show what would change without writing")
    args = parser.parse_args()

    root = repo_root()
    step(f"repo: {root}")
    python = ensure_venv(root, args.dry_run)
    if not args.skip_deps:
        install_dependencies(root, python, args.dry_run)
    if not args.skip_env:
        ensure_env(root, args.dry_run)
    install_launcher(root, python, args.force, args.dry_run)
    step("setup complete")
    step("after adding keys to .env, start Argus with: argus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
