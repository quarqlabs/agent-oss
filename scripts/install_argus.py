#!/usr/bin/env python3
"""Install the global `argus` launcher for the current user."""

from __future__ import annotations

import argparse
import os
import platform
import stat
import subprocess
from pathlib import Path


START_MARKER = "# >>> argus cli path >>>"
END_MARKER = "# <<< argus cli path <<<"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def print_step(message: str) -> None:
    print(f"[argus-setup] {message}", flush=True)


def ensure_executable(path: Path, dry_run: bool) -> None:
    if platform.system() == "Windows":
        return
    mode = path.stat().st_mode
    executable_mode = mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    if mode == executable_mode:
        return
    if dry_run:
        print_step(f"would make executable: {path}")
        return
    path.chmod(executable_mode)


def prepend_path(existing: str, directory: Path) -> str:
    parts = [part for part in existing.split(os.pathsep) if part]
    directory_text = str(directory)
    normalized = {str(Path(part).expanduser()) for part in parts}
    if directory_text in normalized:
        return existing
    return os.pathsep.join([directory_text, *parts])


def write_text(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        print_step(f"would write: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_profile_block(profile: Path, block: str, dry_run: bool) -> bool:
    existing = profile.read_text(encoding="utf-8") if profile.exists() else ""
    known_path_mentions = (
        str(Path.home() / ".local" / "bin"),
        "$HOME/.local/bin",
        "~/.local/bin",
    )
    if START_MARKER in existing or any(path in existing for path in known_path_mentions):
        return False
    updated = existing
    if updated and not updated.endswith("\n"):
        updated += "\n"
    updated += f"\n{block}\n"
    write_text(profile, updated, dry_run)
    return True


def detect_shell_profile() -> tuple[Path, str]:
    shell = Path(os.environ.get("SHELL", "")).name
    home = Path.home()
    if shell == "zsh":
        return home / ".zshrc", f'{START_MARKER}\nexport PATH="$HOME/.local/bin:$PATH"\n{END_MARKER}'
    if shell == "bash":
        return home / ".bashrc", f'{START_MARKER}\nexport PATH="$HOME/.local/bin:$PATH"\n{END_MARKER}'
    if shell == "fish":
        return (
            home / ".config" / "fish" / "config.fish",
            f"{START_MARKER}\nfish_add_path $HOME/.local/bin\n{END_MARKER}",
        )
    return home / ".profile", f'{START_MARKER}\nexport PATH="$HOME/.local/bin:$PATH"\n{END_MARKER}'


def install_unix(root: Path, force: bool, no_profile: bool, dry_run: bool) -> int:
    source = root / "argus"
    if not source.exists():
        print_step(f"missing launcher: {source}")
        return 1

    ensure_executable(source, dry_run)

    bin_dir = Path.home() / ".local" / "bin"
    target = bin_dir / "argus"
    if dry_run:
        print_step(f"would create directory: {bin_dir}")
    else:
        bin_dir.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        try:
            current = target.resolve(strict=False)
        except OSError:
            current = None
        if current != source.resolve() and not force:
            print_step(f"{target} already exists and points somewhere else")
            print_step("rerun with --force if you want to replace it")
            return 1
        if current != source.resolve():
            if dry_run:
                print_step(f"would replace existing launcher: {target}")
            else:
                target.unlink()

    if not target.exists() and not target.is_symlink():
        if dry_run:
            print_step(f"would symlink {target} -> {source}")
        else:
            target.symlink_to(source)

    profile_updated = False
    profile = None
    if not no_profile:
        profile, block = detect_shell_profile()
        profile_updated = append_profile_block(profile, block, dry_run)

    print_step(f"installed: {target} -> {source}")
    if str(bin_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        if profile_updated and profile:
            print_step(f"updated PATH for future terminals in {profile}")
        print_step("for this terminal, run: export PATH=\"$HOME/.local/bin:$PATH\"")
        if profile:
            print_step(f"or open a new terminal; zsh/bash users can also run: source {profile}")
    print_step("then start Argus with: argus")
    return 0


def install_windows(root: Path, force: bool, dry_run: bool) -> int:
    bin_dir = Path.home() / "bin"
    cmd_path = bin_dir / "argus.cmd"
    ps1_path = bin_dir / "argus.ps1"
    agent_cli = root / "agent_cli.py"

    if not agent_cli.exists():
        print_step(f"missing CLI entrypoint: {agent_cli}")
        return 1

    if (cmd_path.exists() or ps1_path.exists()) and not force:
        print_step(f"{bin_dir} already contains an argus launcher")
        print_step("rerun with --force if you want to replace it")
        return 1

    root_text = str(root)
    cmd_content = (
        "@echo off\r\n"
        f'set "ARGUS_ROOT={root_text}"\r\n'
        'set "ARGUS_PYTHON=%ARGUS_ROOT%\\.venv\\Scripts\\python.exe"\r\n'
        'if not exist "%ARGUS_PYTHON%" set "ARGUS_PYTHON=python"\r\n'
        'set "PYTHONUTF8=1"\r\n'
        'set "PYTHONIOENCODING=utf-8:replace"\r\n'
        '"%ARGUS_PYTHON%" "%ARGUS_ROOT%\\agent_cli.py" %*\r\n'
    )
    ps1_content = (
        f'$ArgusRoot = "{root_text}"\r\n'
        '$VenvPython = Join-Path $ArgusRoot ".venv\\Scripts\\python.exe"\r\n'
        '$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }\r\n'
        '$env:PYTHONUTF8 = "1"\r\n'
        '$env:PYTHONIOENCODING = "utf-8:replace"\r\n'
        '& $Python (Join-Path $ArgusRoot "agent_cli.py") @args\r\n'
    )

    write_text(cmd_path, cmd_content, dry_run)
    write_text(ps1_path, ps1_content, dry_run)

    if dry_run:
        print_step(f"would add to user PATH: {bin_dir}")
    else:
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                "Environment",
                0,
                winreg.KEY_READ | winreg.KEY_WRITE,
            ) as key:
                try:
                    current_path, value_type = winreg.QueryValueEx(key, "Path")
                except FileNotFoundError:
                    current_path, value_type = "", winreg.REG_EXPAND_SZ
                next_path = prepend_path(current_path, bin_dir)
                if next_path != current_path:
                    winreg.SetValueEx(key, "Path", 0, value_type, next_path)
        except Exception as exc:  # pragma: no cover - Windows-only fallback.
            print_step(f"could not update user PATH automatically: {exc}")
            print_step(f"add this folder to your user PATH manually: {bin_dir}")

    print_step(f"installed: {cmd_path}")
    print_step("open a new terminal, then start Argus with: argus")
    return 0


def smoke_test(root: Path) -> None:
    launcher = root / ("argus.cmd" if platform.system() == "Windows" else "argus")
    if not launcher.exists():
        return
    try:
        subprocess.run(
            [str(launcher), "--help"],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except Exception:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the global argus command.")
    parser.add_argument("--force", action="store_true", help="replace an existing argus launcher")
    parser.add_argument("--no-profile", action="store_true", help="do not update shell profile files")
    parser.add_argument("--dry-run", action="store_true", help="show what would change without writing")
    args = parser.parse_args()

    root = repo_root()
    print_step(f"repo: {root}")
    if platform.system() == "Windows":
        status = install_windows(root, args.force, args.dry_run)
    else:
        status = install_unix(root, args.force, args.no_profile, args.dry_run)

    if status == 0 and not args.dry_run:
        smoke_test(root)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
