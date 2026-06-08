"""Approval-policy helpers for delegated coding agents."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any


DESTRUCTIVE_WORDS = {
    "rm",
    "rmdir",
    "unlink",
    "shred",
    "reset",
    "clean",
    "rebase",
    "push",
    "commit",
    "tag",
    "chmod",
    "chown",
    "sudo",
}
NETWORK_OR_INSTALL_WORDS = {
    "curl",
    "wget",
    "ssh",
    "scp",
    "rsync",
    "pip",
    "npm",
    "pnpm",
    "yarn",
    "brew",
    "apt",
    "gem",
    "cargo",
    "go",
}
SAFE_READ_COMMANDS = {"ls", "pwd", "cat", "sed", "rg", "grep", "find", "head", "tail", "wc", "git"}
SAFE_TEST_COMMANDS = {"pytest", "ruff", "mypy", "npm", "pnpm", "yarn", "python", "python3", "node"}
SENSITIVE_PATH_PARTS = {".env", "id_rsa", "id_ed25519", ".ssh", ".npmrc", ".pypirc", "credentials", "secrets"}


def normalize_workspace(workspace_root: str | Path) -> Path:
    return Path(workspace_root).expanduser().resolve()


def is_inside_workspace(path: str | Path, workspace_root: str | Path) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(normalize_workspace(workspace_root))
        return True
    except ValueError:
        return False


def touches_sensitive_path(value: str | None) -> bool:
    text = str(value or "").lower()
    return any(part in text for part in SENSITIVE_PATH_PARTS)


def command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(str(command or ""))
    except ValueError:
        return str(command or "").split()


def is_safe_test_or_build(tokens: list[str]) -> bool:
    if not tokens or tokens[0] not in SAFE_TEST_COMMANDS:
        return False
    joined = " ".join(tokens)
    safe_markers = (
        "test",
        "pytest",
        "ruff",
        "lint",
        "typecheck",
        "build",
        "-m pytest",
        "-m unittest",
    )
    return any(marker in joined for marker in safe_markers)


def classify_command(command: str, workspace_root: str | Path) -> dict[str, Any]:
    tokens = command_tokens(command)
    if not tokens:
        return {"allow": True, "reason": "empty command has no effect", "risk": "none"}

    executable = Path(tokens[0]).name
    lowered = [Path(token).name.lower() for token in tokens]
    joined = " ".join(tokens).lower()

    if touches_sensitive_path(joined):
        return {"allow": False, "reason": "command touches secrets or env files", "risk": "sensitive_path"}

    if executable == "git" and any(word in lowered for word in {"push", "commit", "reset", "clean", "rebase", "tag"}):
        return {"allow": False, "reason": "git write/history operation needs user approval", "risk": "git_write"}

    if any(word in lowered for word in DESTRUCTIVE_WORDS):
        return {"allow": False, "reason": "destructive shell operation needs user approval", "risk": "destructive"}

    if any(word in lowered for word in NETWORK_OR_INSTALL_WORDS) and not is_safe_test_or_build(tokens):
        return {"allow": False, "reason": "network or dependency operation needs user approval", "risk": "network_or_install"}

    if executable in SAFE_READ_COMMANDS:
        return {"allow": True, "reason": "safe read-only command", "risk": "low"}

    if is_safe_test_or_build(tokens):
        return {"allow": True, "reason": "routine test/build command", "risk": "low"}

    return {"allow": False, "reason": "unknown command needs user approval", "risk": "unknown"}


def classify_file_write(path: str, workspace_root: str | Path) -> dict[str, Any]:
    if touches_sensitive_path(path):
        return {"allow": False, "reason": "secret/env file edits need user approval", "risk": "sensitive_path"}
    if not is_inside_workspace(path, workspace_root):
        return {"allow": False, "reason": "writes outside workspace need user approval", "risk": "outside_workspace"}
    return {"allow": True, "reason": "normal workspace edit", "risk": "low"}


def classify_action(action: dict[str, Any], workspace_root: str | Path) -> dict[str, Any]:
    """Classify a delegated coding-agent action for auto-approval."""

    if action.get("requires_user_confirmation"):
        return {"allow": False, "reason": "provider explicitly requested user confirmation", "risk": "provider_required"}

    command = action.get("command") or action.get("cmd")
    if command:
        return classify_command(str(command), workspace_root)

    path = action.get("path") or action.get("file") or action.get("target_path")
    operation = str(action.get("operation") or action.get("type") or "").lower()
    if path and any(word in operation for word in ("write", "edit", "create", "delete", "patch")):
        return classify_file_write(str(path), workspace_root)

    return {"allow": True, "reason": "low-risk coding-agent progress event", "risk": "low"}
