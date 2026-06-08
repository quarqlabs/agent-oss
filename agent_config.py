"""Local agent identity config helpers.

Identity is stored per local agent namespace and uses environment values only
as defaults when the config file or a field is missing.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_AGENT_CONFIG = {
    "agent_name": "Argus",
    "agent_personality": "professional and helpful",
    "agent_use_cases": ["general assistance"],
    "agent_custom_prompt": "",
}


def _safe_agent_id(value: str | None) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value or "local_agent")


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def get_agent_identity_config_path() -> Path:
    """Return the local JSON config path for this agent identity."""

    explicit_path = os.getenv("AGENT_IDENTITY_CONFIG_PATH")
    if explicit_path:
        return _resolve_path(explicit_path)

    memory_root = _resolve_path(os.getenv("LOCAL_MEMORY_ROOT", "local_memory"))
    return memory_root / _safe_agent_id(os.getenv("AGENT_ID")) / "agent_identity.json"


def parse_agent_use_cases(value: Any) -> list[str]:
    """Accept JSON arrays, Python lists, or comma-separated strings."""

    if value is None:
        return []

    if isinstance(value, list):
        return _clean_string_list(value)

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return _clean_string_list(parsed)

    return _clean_string_list(text.split(","))


def _clean_string_list(items: list[Any]) -> list[str]:
    seen = set()
    cleaned = []
    for item in items:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned


def default_agent_config_from_env() -> dict[str, Any]:
    """Build agent identity defaults from env with safe parsing."""

    config = dict(DEFAULT_AGENT_CONFIG)

    if os.getenv("AGENT_NAME"):
        config["agent_name"] = os.getenv("AGENT_NAME", "").strip()

    if os.getenv("AGENT_PERSONALITY"):
        config["agent_personality"] = os.getenv("AGENT_PERSONALITY", "").strip()

    env_use_cases = parse_agent_use_cases(os.getenv("AGENT_USE_CASES"))
    if env_use_cases:
        config["agent_use_cases"] = env_use_cases

    if os.getenv("AGENT_CUSTOM_PROMPT") is not None:
        config["agent_custom_prompt"] = os.getenv("AGENT_CUSTOM_PROMPT", "").strip()

    return normalize_agent_config(config)


def normalize_agent_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return a complete, normalized identity config."""

    source = config or {}
    normalized = dict(DEFAULT_AGENT_CONFIG)

    for key in ("agent_name", "agent_personality", "agent_custom_prompt"):
        value = source.get(key)
        if value is not None:
            normalized[key] = str(value).strip()

    use_cases = parse_agent_use_cases(source.get("agent_use_cases"))
    if use_cases:
        normalized["agent_use_cases"] = use_cases

    if not normalized["agent_name"]:
        normalized["agent_name"] = DEFAULT_AGENT_CONFIG["agent_name"]

    if not normalized["agent_personality"]:
        normalized["agent_personality"] = DEFAULT_AGENT_CONFIG["agent_personality"]

    return normalized


def load_agent_config() -> dict[str, Any]:
    """Load identity from local config, falling back to env defaults."""

    config = default_agent_config_from_env()
    path = get_agent_identity_config_path()

    if not path.exists():
        return config

    try:
        file_config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return config

    if isinstance(file_config, dict):
        config.update(file_config)

    return normalize_agent_config(config)


def save_agent_config(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge and persist identity updates to the local config file."""

    config = load_agent_config()
    config.update(updates)
    config = normalize_agent_config(config)

    path = get_agent_identity_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
    return config
