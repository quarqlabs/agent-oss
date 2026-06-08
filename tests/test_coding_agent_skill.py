from coding_agents.manager import CodingAgentManager
from tools.coding_agent.client import (
    configure_coding_agent,
    get_coding_task_status,
    start_coding_task,
)


def test_disabled_coding_agent_returns_helpful_message(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "disabled_agent")
    monkeypatch.setenv("CODING_AGENTS_ENABLED", "false")

    manager = CodingAgentManager()
    result = manager.start_task("implement something")

    assert result["status"] == "disabled"
    assert "CODING_AGENTS_ENABLED=true" in result["message"]


def test_missing_runtime_creates_failed_task(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "missing_runtime_agent")
    monkeypatch.setenv("CODING_AGENTS_ENABLED", "true")
    monkeypatch.setattr(
        "coding_agents.manager.runtime_check",
        lambda settings: (False, "runtime missing"),
    )

    manager = CodingAgentManager()
    result = manager.start_task("fix the test")

    assert result["status"] == "failed"
    assert result["task"]["status"] == "failed"
    assert result["task"]["error"] == "runtime missing"


def test_coding_tool_functions_return_short_status(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "tool_agent")
    monkeypatch.setenv("CODING_AGENTS_ENABLED", "true")
    monkeypatch.setattr(
        "coding_agents.manager.runtime_check",
        lambda settings: (False, "runtime missing"),
    )

    response = start_coding_task.invoke({"task_prompt": "fix bug"})
    assert "Task id:" in response
    assert "Status: failed" in response

    task_id = response.split("Task id: ", 1)[1].splitlines()[0]
    status = get_coding_task_status.invoke({"task_id": task_id})
    assert "runtime missing" in status


def test_configure_coding_agent_tool_sets_workspace_and_lists_providers(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "config_tool_agent")

    response = configure_coding_agent.invoke({"workspace_path": str(tmp_path), "network_access": "off"})

    assert "Default coding agent: Codex (`codex`)" in response
    assert f"Workspace: {tmp_path}" in response
    assert "Network access: off" in response
    assert "Claude Code (`claude_code`) [planned]" in response
