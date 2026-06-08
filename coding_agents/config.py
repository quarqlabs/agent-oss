"""Configuration helpers for coding-agent delegation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CODEX_ARGS = ("mcp-server",)
ACTIVE_TASK_STATUSES = {"queued", "running", "waiting_user"}
CODING_AGENT_PROVIDERS = {
    "codex": {
        "label": "Codex",
        "status": "available",
        "description": "OpenAI Codex through Codex CLI MCP server.",
    },
    "claude_code": {
        "label": "Claude Code",
        "status": "planned",
        "description": "Planned provider slot for Claude Code delegation.",
    },
    "cursor": {
        "label": "Cursor",
        "status": "planned",
        "description": "Planned provider slot for Cursor delegation.",
    },
}


def parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def safe_id(value: str | None, fallback: str = "local_agent") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(value or "").strip())
    return cleaned or fallback


def resolve_path(
    value: str | None,
    default: Path,
    relative_to: Path | None = None,
) -> Path:
    raw = str(value or "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = (relative_to or BASE_DIR) / path
    return path.resolve()


def launch_cwd() -> Path:
    return resolve_path(os.getenv("ARGUS_LAUNCH_CWD") or os.getcwd(), BASE_DIR)


def parse_args(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_CODEX_ARGS)
    return [item.strip() for item in str(value).split(",") if item.strip()]


@dataclass(frozen=True)
class CodingAgentSettings:
    enabled: bool
    default_provider: str
    codex_command: str
    codex_args: list[str]
    workspace_root: Path
    approval_policy: str
    network_access: bool
    timeout_seconds: int
    memory_root: Path
    agent_id: str

    @property
    def task_root(self) -> Path:
        return self.memory_root / self.agent_id / "coding_agents"


    @property
    def config_path(self) -> Path:
        return self.task_root / "config.json"


def base_settings_from_env() -> CodingAgentSettings:
    memory_root = resolve_path(os.getenv("LOCAL_MEMORY_ROOT"), BASE_DIR / "local_memory")
    workspace_base = launch_cwd()
    return CodingAgentSettings(
        enabled=parse_bool(os.getenv("CODING_AGENTS_ENABLED"), default=True),
        default_provider=(os.getenv("CODING_AGENT_DEFAULT_PROVIDER") or "codex").strip().lower(),
        codex_command=(os.getenv("CODEX_MCP_COMMAND") or "codex").strip(),
        codex_args=parse_args(os.getenv("CODEX_MCP_ARGS")),
        workspace_root=resolve_path(
            os.getenv("CODEX_WORKSPACE_ROOT") or ".",
            workspace_base,
            relative_to=workspace_base,
        ),
        approval_policy=(os.getenv("CODEX_APPROVAL_POLICY") or "argus-safe-auto").strip(),
        network_access=parse_bool(os.getenv("CODEX_NETWORK_ACCESS"), default=True),
        timeout_seconds=int(os.getenv("CODEX_TASK_TIMEOUT_SECONDS") or "1800"),
        memory_root=memory_root,
        agent_id=safe_id(os.getenv("AGENT_ID") or os.getenv("USER_ID")),
    )


def read_local_config(settings: CodingAgentSettings | None = None) -> dict[str, Any]:
    settings = settings or base_settings_from_env()
    path = settings.config_path
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_local_config(updates: dict[str, Any]) -> dict[str, Any]:
    base = base_settings_from_env()
    current = read_local_config(base)
    current.update({key: value for key, value in updates.items() if value is not None})
    if not current.get("default_provider"):
        current.pop("default_provider", None)
    if not current.get("workspace_root"):
        current.pop("workspace_root", None)

    path = base.config_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
    return current


def provider_info(provider: str | None) -> dict[str, Any] | None:
    return CODING_AGENT_PROVIDERS.get(str(provider or "").strip().lower())


def list_providers() -> list[dict[str, Any]]:
    rows = []
    for slug, info in CODING_AGENT_PROVIDERS.items():
        rows.append({"slug": slug, **info})
    return rows


def provider_is_available(provider: str | None) -> bool:
    info = provider_info(provider)
    return bool(info and info.get("status") == "available")


def set_default_provider(provider: str) -> dict[str, Any]:
    provider = str(provider or "").strip().lower()
    info = provider_info(provider)
    if not info:
        raise ValueError(f"Unknown coding agent provider: {provider}")
    if info.get("status") != "available":
        raise ValueError(f"{info['label']} is listed but not implemented yet.")
    write_local_config({"default_provider": provider})
    return coding_config_summary()


def set_workspace_root(workspace_path: str) -> dict[str, Any]:
    path = resolve_path(workspace_path, launch_cwd(), relative_to=launch_cwd())
    write_local_config({"workspace_root": str(path)})
    return coding_config_summary()


def set_network_access(enabled: bool) -> dict[str, Any]:
    write_local_config({"network_access": bool(enabled)})
    return coding_config_summary()


def load_settings() -> CodingAgentSettings:
    base = base_settings_from_env()
    local_config = read_local_config(base)
    if "network_access" in local_config:
        network_access = parse_bool(str(local_config.get("network_access")), default=base.network_access)
    else:
        network_access = base.network_access
    return CodingAgentSettings(
        enabled=base.enabled,
        default_provider=str(local_config.get("default_provider") or base.default_provider).strip().lower(),
        codex_command=base.codex_command,
        codex_args=base.codex_args,
        workspace_root=resolve_path(
            local_config.get("workspace_root"),
            base.workspace_root,
            relative_to=launch_cwd(),
        ),
        approval_policy=base.approval_policy,
        network_access=network_access,
        timeout_seconds=base.timeout_seconds,
        memory_root=base.memory_root,
        agent_id=base.agent_id,
    )


def coding_config_summary() -> dict[str, Any]:
    settings = load_settings()
    provider = provider_info(settings.default_provider) or {}
    return {
        "enabled": settings.enabled,
        "default_provider": settings.default_provider,
        "default_provider_label": provider.get("label", settings.default_provider),
        "default_provider_status": provider.get("status", "unknown"),
        "workspace_root": str(settings.workspace_root),
        "approval_policy": settings.approval_policy,
        "network_access": settings.network_access,
        "timeout_seconds": settings.timeout_seconds,
        "providers": list_providers(),
        "config_path": str(settings.config_path),
    }
