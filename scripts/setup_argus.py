#!/usr/bin/env python3
"""Bootstrap Argus for local use."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
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


def add_to_process_path(directory: Path) -> None:
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    directory_text = str(directory)
    if directory_text.lower() in {part.lower() for part in path_parts}:
        return
    os.environ["PATH"] = os.pathsep.join([directory_text, *path_parts])


def install_cloudflared_from_github(root: Path, dry_run: bool) -> None:
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    install_dir = Path("C:/cloudflared")
    target = install_dir / "cloudflared.exe"
    if target.exists():
        step(f"cloudflared already installed at {target}")
        add_to_process_path(install_dir)
        return
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if dry_run:
        command_name = powershell or "powershell"
        step(
            f"would run: {command_name} -NoProfile -ExecutionPolicy Bypass "
            f"-Command \"New-Item -ItemType Directory -Force -Path '{install_dir}' | Out-Null; "
            f"Invoke-WebRequest '{url}' -OutFile '{target}'; "
            f"$env:Path += ';{install_dir}'; & '{target}' --version\""
        )
        step(f"would add to PATH for this setup process: {install_dir}")
        return
    if powershell:
        command = (
            f"New-Item -ItemType Directory -Force -Path '{install_dir}' | Out-Null; "
            f"Invoke-WebRequest '{url}' -OutFile '{target}'; "
            f"$env:Path += ';{install_dir}'; "
            f"& '{target}' --version"
        )
        try:
            run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command], root, dry_run=False)
            add_to_process_path(install_dir)
            return
        except subprocess.CalledProcessError as exc:
            step(f"PowerShell cloudflared install failed with exit code {exc.returncode}; trying Python download")
    try:
        install_dir.mkdir(parents=True, exist_ok=True)
        download_path = target.with_suffix(".exe.download")
        step(f"downloading cloudflared from {url}")
        urllib.request.urlretrieve(url, download_path)
        download_path.replace(target)
        add_to_process_path(install_dir)
        step(f"installed cloudflared: {target}")
        run([str(target), "--version"], root, dry_run=False)
    except OSError as exc:
        step(f"direct cloudflared install failed: {exc}")
        step("install it manually, or set CLOUDFLARED_BIN to the full cloudflared.exe path")


def install_cloudflared(root: Path, dry_run: bool) -> None:
    if shutil.which("cloudflared"):
        step("cloudflared already installed")
        return
    if platform.system() == "Windows":
        winget = shutil.which("winget")
        if not winget:
            step(
                "cloudflared not found and winget is unavailable; downloading cloudflared directly"
            )
            install_cloudflared_from_github(root, dry_run)
            return
        run(
            [
                winget,
                "install",
                "--id",
                "Cloudflare.cloudflared",
                "--source",
                "winget",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            root,
            dry_run,
        )
        return
    if not shutil.which("brew"):
        step("Homebrew not found; skipping cloudflared install")
        return
    run(["brew", "install", "cloudflare/cloudflare/cloudflared"], root, dry_run)


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
    parser.add_argument("--skip-cloudflared", action="store_true", help="do not install cloudflared")
    parser.add_argument("--skip-env", action="store_true", help="do not create .env from .env.example")
    parser.add_argument("--dry-run", action="store_true", help="show what would change without writing")
    args = parser.parse_args()

    root = repo_root()
    step(f"repo: {root}")
    python = ensure_venv(root, args.dry_run)
    if not args.skip_deps:
        install_dependencies(root, python, args.dry_run)
    if not args.skip_cloudflared:
        install_cloudflared(root, args.dry_run)
    if not args.skip_env:
        ensure_env(root, args.dry_run)
    install_launcher(root, python, args.force, args.dry_run)
    step("setup complete")
    step("after adding keys to .env, start Argus with: argus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
