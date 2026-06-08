import pytest

from coding_agents.codex_runner import (
    codex_task_arguments,
    resolve_codex_command,
    runtime_check,
    should_restart_codex_thread,
)
from coding_agents.config import (
    coding_config_summary,
    load_settings,
    set_default_provider,
    set_network_access,
    set_workspace_root,
)
from coding_agents.manager import CodingAgentManager
from coding_agents.task_store import CodingTaskStore


def test_coding_config_env_defaults_and_local_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "coding_config_agent")
    monkeypatch.setenv("CODING_AGENT_DEFAULT_PROVIDER", "codex")
    monkeypatch.setenv("CODEX_WORKSPACE_ROOT", str(tmp_path / "repo_a"))

    assert load_settings().default_provider == "codex"
    assert load_settings().codex_command == "codex"
    assert load_settings().codex_args == ["mcp-server"]
    assert load_settings().workspace_root == tmp_path / "repo_a"
    assert load_settings().network_access is True

    set_workspace_root(str(tmp_path / "repo_b"))
    set_network_access(False)
    summary = coding_config_summary()

    assert summary["default_provider"] == "codex"
    assert summary["workspace_root"] == str(tmp_path / "repo_b")
    assert summary["network_access"] is False
    assert summary["config_path"].endswith("coding_agents/config.json")


def test_relative_workspace_resolves_from_launch_cwd(monkeypatch, tmp_path):
    launch_dir = tmp_path / "user_project"
    launch_dir.mkdir()
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "portable_workspace_agent")
    monkeypatch.setenv("ARGUS_LAUNCH_CWD", str(launch_dir))
    monkeypatch.setenv("CODEX_WORKSPACE_ROOT", ".")

    assert load_settings().workspace_root == launch_dir

    set_workspace_root("subrepo")
    assert load_settings().workspace_root == launch_dir / "subrepo"


def test_coding_config_rejects_planned_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "planned_provider_agent")

    with pytest.raises(ValueError, match="not implemented yet"):
        set_default_provider("claude_code")


def test_runtime_check_rejects_old_npx_codex_package(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "bad_npx_agent")
    monkeypatch.setenv("CODEX_MCP_COMMAND", "npx")
    monkeypatch.setenv("CODEX_MCP_ARGS", "-y,codex,mcp-server")

    ok, reason = runtime_check(load_settings())

    assert ok is False
    assert "unrelated npm package" in reason
    assert "CODEX_MCP_COMMAND=codex" in reason


def test_codex_task_arguments_carries_network_access():
    enabled = codex_task_arguments("install deps", True)
    disabled = codex_task_arguments("install deps", False)

    assert enabled["sandbox"] == "workspace-write"
    assert enabled["config"]["sandbox_workspace_write"]["network_access"] is True
    assert disabled["config"]["sandbox_workspace_write"]["network_access"] is False


def test_codex_thread_restarts_when_network_mode_is_unknown_or_changed():
    old_task = {"provider_thread_id": "thread_123", "provider_network_access": None}
    matching_task = {"provider_thread_id": "thread_123", "provider_network_access": True}
    changed_task = {"provider_thread_id": "thread_123", "provider_network_access": False}
    flagged_task = {
        "provider_thread_id": "thread_123",
        "provider_network_access": True,
        "restart_provider_session": True,
    }

    assert should_restart_codex_thread(old_task, True) is True
    assert should_restart_codex_thread(old_task, False) is False
    assert should_restart_codex_thread(matching_task, True) is False
    assert should_restart_codex_thread(changed_task, True) is True
    assert should_restart_codex_thread(flagged_task, True) is True


def test_resolve_codex_command_falls_back_to_codex_app(monkeypatch, tmp_path):
    fake_codex = tmp_path / "Codex.app" / "Contents" / "Resources" / "codex"
    fake_codex.parent.mkdir(parents=True)
    fake_codex.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_codex.chmod(0o755)

    monkeypatch.setattr("coding_agents.codex_runner.shutil.which", lambda command: None)
    monkeypatch.setattr("coding_agents.codex_runner.KNOWN_CODEX_COMMANDS", (fake_codex,))

    assert resolve_codex_command("codex") == str(fake_codex)


def test_manager_lists_only_active_tasks(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "active_task_agent")
    monkeypatch.setenv("CODING_AGENTS_ENABLED", "true")
    monkeypatch.setattr(
        "coding_agents.manager.runtime_check",
        lambda settings: (False, "runtime missing"),
    )

    manager = CodingAgentManager()
    failed = manager.start_task("failed task")["task"]

    assert failed["status"] == "failed"
    assert manager.list_tasks(status_filter="active") == []


def test_manager_stores_network_default_on_new_task(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "task_network_agent")
    monkeypatch.setenv("CODING_AGENTS_ENABLED", "true")
    monkeypatch.setenv("CODEX_NETWORK_ACCESS", "false")
    monkeypatch.setattr(
        "coding_agents.manager.runtime_check",
        lambda settings: (False, "runtime missing"),
    )

    manager = CodingAgentManager()
    task = manager.start_task("install deps")["task"]

    assert task["network_access"] is False


def test_manager_marks_existing_codex_thread_for_network_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "network_restart_agent")

    settings = load_settings()
    store = CodingTaskStore(settings.task_root)
    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="download data",
    )
    store.update_task(
        task["id"],
        status="completed",
        provider_thread_id="thread_123",
        provider_network_access=False,
    )

    manager = CodingAgentManager()
    result = manager.set_task_network_access(task["id"], True)

    assert result["task"]["network_access"] is True
    assert result["task"]["restart_provider_session"] is True
    assert "restart from saved task context" in result["message"]


def test_manager_subscribes_task_notifications(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "subscribe_task_agent")

    settings = load_settings()
    store = CodingTaskStore(settings.task_root)
    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="long task",
        conversation_ref="cli",
    )

    manager = CodingAgentManager()
    updated = manager.subscribe_task(task["id"], "telegram:123")

    assert updated["notification_refs"] == ["cli", "telegram:123"]


def test_manager_replies_to_latest_completed_task(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "latest_reply_agent")
    monkeypatch.setenv("CODING_AGENTS_ENABLED", "true")
    monkeypatch.setattr(
        "coding_agents.manager.runtime_check",
        lambda settings: (False, "runtime missing"),
    )

    manager = CodingAgentManager()
    first = manager.start_task("build an app", conversation_ref="cli")["task"]
    second = manager.start_task("fix another app", conversation_ref="telegram")["task"]

    result = manager.reply_to_latest("continue this", conversation_ref="cli")

    assert result["task"]["id"] == first["id"]
    assert result["task"]["pending_prompt"] == "continue this"
    assert manager.latest_task(conversation_ref="telegram")["id"] == second["id"]


def test_manager_deletes_history_and_preserves_active_tasks(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "delete_history_agent")

    settings = load_settings()
    store = CodingTaskStore(settings.task_root)
    active = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="active task",
    )
    old = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="old task",
    )
    store.update_task(old["id"], status="completed")

    manager = CodingAgentManager()
    result = manager.clear_task_history()

    assert result["deleted"] == 1
    assert result["preserved"] == 1
    assert manager.get_task(active["id"]) is not None
    assert manager.get_task(old["id"]) is None


def test_manager_marks_untracked_running_task_as_interrupted(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "interrupted_task_agent")

    settings = load_settings()
    store = CodingTaskStore(settings.task_root)
    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="long coding task",
    )
    store.update_task(task["id"], status="running", current_activity="Codex is working.")

    manager = CodingAgentManager()
    fixed = manager.get_task(task["id"])

    assert fixed["status"] == "failed"
    assert fixed["current_activity"] is None
    assert "interrupted" in fixed["error"]
    assert manager.list_tasks(status_filter="active") == []
    logs = manager.get_logs(task["id"])
    assert logs[-1]["title"] == "Coding interrupted"
