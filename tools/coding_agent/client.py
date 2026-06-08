"""LangChain tools for Argus coding-agent delegation."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from coding_agents.manager import get_coding_manager


def _runtime_ref(config: RunnableConfig | None) -> str | None:
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    channel = str(configurable.get("channel_type") or "").strip()
    user_id = str(configurable.get("user_id") or "").strip()
    bits = [bit for bit in (channel, user_id) if bit]
    return ":".join(bits) if bits else None


def _format_task_line(task: dict[str, Any]) -> str:
    return (
        f"Task id: {task.get('id')}\n"
        f"Status: {task.get('status')}\n"
        f"Provider: {task.get('provider')}\n"
        f"Workspace: {task.get('workspace_path')}\n"
        f"Network: {'on' if task.get('network_access', True) else 'off'}"
    )


def _format_coding_config(summary: dict[str, Any]) -> str:
    providers = "\n".join(
        f"- {item['label']} (`{item['slug']}`) [{item['status']}] - {item['description']}"
        for item in summary.get("providers", [])
    )
    return (
        f"Default coding agent: {summary.get('default_provider_label')} "
        f"(`{summary.get('default_provider')}`)\n"
        f"Workspace: {summary.get('workspace_root')}\n"
        f"Approval policy: {summary.get('approval_policy')}\n"
        f"Network access: {'on' if summary.get('network_access') else 'off'}\n"
        f"Config file: {summary.get('config_path')}\n\n"
        f"Available coding agents:\n{providers}"
    )


def _parse_network_access(value: str) -> bool | None:
    lowered = str(value or "").strip().lower()
    if lowered in {"on", "true", "1", "yes", "enable", "enabled", "allow", "allowed"}:
        return True
    if lowered in {"off", "false", "0", "no", "disable", "disabled", "block", "blocked"}:
        return False
    return None


@tool
def configure_coding_agent(
    provider: str = "",
    workspace_path: str = "",
    network_access: str = "",
    config: RunnableConfig = None,
) -> str:
    """Show or update the default coding-agent provider, workspace, and network mode."""

    manager = get_coding_manager()
    try:
        summary = manager.get_config()
        if provider:
            summary = manager.set_default_provider(provider)
        if workspace_path:
            summary = manager.set_workspace_root(workspace_path)
        if network_access:
            enabled = _parse_network_access(network_access)
            if enabled is None:
                return "Use `on` or `off` for coding-agent network access."
            summary = manager.set_default_network_access(enabled)
    except ValueError as exc:
        return str(exc)
    return _format_coding_config(summary)


@tool
def start_coding_task(
    task_prompt: str,
    workspace_path: str = "",
    provider: str = "",
    config: RunnableConfig = None,
) -> str:
    """Start a delegated coding task using the configured coding-agent provider."""

    if not str(task_prompt or "").strip():
        return "Please provide the coding task to delegate."

    result = get_coding_manager().start_task(
        task_prompt,
        provider=provider or None,
        workspace_path=workspace_path or None,
        conversation_ref=_runtime_ref(config),
    )
    task = result.get("task") or {}
    if not task:
        return str(result.get("message") or "Coding task could not be started.")

    next_action = f"Follow progress in the CLI or run `/coding-status {task.get('id')}`."
    return f"{result.get('message')}\n\n{_format_task_line(task)}\n\n{next_action}"


@tool
def continue_coding_task(
    message: str,
    task_id: str = "",
    config: RunnableConfig = None,
) -> str:
    """Continue the latest coding task, or a specific task when task_id is provided."""

    if not str(message or "").strip():
        return "Please provide the message to send to the coding task."

    manager = get_coding_manager()
    try:
        if task_id:
            result = manager.reply_to_task(task_id, message)
        else:
            result = manager.reply_to_latest(message, conversation_ref=_runtime_ref(config))
    except KeyError:
        return f"Coding task not found: {task_id}"

    task = result.get("task") or {}
    if not task:
        return str(result.get("message") or "No coding task found.")
    return f"{result.get('message')}\n\n{_format_task_line(task)}"


@tool
def reply_to_coding_task(
    task_id: str,
    message: str,
    config: RunnableConfig = None,
) -> str:
    """Reply to a coding task, including approval responses or clarifications."""

    try:
        result = get_coding_manager().reply_to_task(task_id, message)
    except KeyError:
        return f"Coding task not found: {task_id}"

    task = result.get("task") or {}
    return f"{result.get('message')}\n\n{_format_task_line(task)}"


@tool
def get_coding_task_status(task_id: str, config: RunnableConfig = None) -> str:
    """Get the current status of a coding task."""

    task = get_coding_manager().get_task(task_id)
    if not task:
        return f"Coding task not found: {task_id}"

    lines = [_format_task_line(task)]
    if task.get("approval_request"):
        lines.append("Approval needed: yes")
    if task.get("restart_provider_session"):
        lines.append("Codex session: will restart on next continuation")
    if task.get("changed_files"):
        lines.append("Changed files:\n" + "\n".join(f"- {path}" for path in task["changed_files"]))
    if task.get("error"):
        lines.append(f"Error: {task['error']}")
    if task.get("result_summary"):
        lines.append(f"Summary: {task['result_summary']}")
    return "\n\n".join(lines)


@tool
def get_coding_task_logs(
    task_id: str,
    limit: int = 20,
    config: RunnableConfig = None,
) -> str:
    """Read recent logs for a coding task."""

    logs = get_coding_manager().get_logs(task_id, limit=max(1, min(int(limit or 20), 100)))
    if not logs:
        return f"No logs found for coding task: {task_id}"

    rows = []
    for event in logs:
        rows.append(
            f"- {event.get('time', '')} {event.get('title', 'Log')}: "
            f"{str(event.get('message') or '').strip()}"
        )
    return "\n".join(rows)


@tool
def cancel_coding_task(task_id: str, config: RunnableConfig = None) -> str:
    """Cancel a coding task."""

    try:
        result = get_coding_manager().cancel_task(task_id)
    except KeyError:
        return f"Coding task not found: {task_id}"

    task = result.get("task") or {}
    return f"Coding task cancelled.\n\n{_format_task_line(task)}"
