from coding_agents.config import load_settings
from coding_agents.task_store import CodingTaskStore


def test_coding_id_commands_suggest_recent_task_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "cli_suggestion_agent")

    store = CodingTaskStore(load_settings().task_root)
    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="build a small app",
    )

    from agent_cli import command_suggestions

    status_suggestions = command_suggestions("/coding-status ")
    assert status_suggestions[0]["name"] == task["id"]
    assert status_suggestions[0]["insert"] == f"/coding-status {task['id']}"

    reply_suggestions = command_suggestions("/coding-reply ")
    assert reply_suggestions[0]["insert"] == f"/coding-reply {task['id']} "

    network_suggestions = command_suggestions("/coding-network ")
    assert network_suggestions[0]["insert"] == "/coding-network on"
    assert any(item["insert"] == f"/coding-network {task['id']} " for item in network_suggestions)

    allow_network_suggestions = command_suggestions("/coding-allow-network ")
    assert allow_network_suggestions[0]["insert"] == f"/coding-allow-network {task['id']}"

    assert command_suggestions(f"/coding-reply {task['id']} continue") == []
