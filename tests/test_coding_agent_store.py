from coding_agents.task_store import CodingTaskStore


def test_task_store_creates_updates_lists_and_logs(tmp_path):
    store = CodingTaskStore(tmp_path / "coding_agents")

    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="implement a small fix",
        conversation_ref="cli:test",
    )

    assert task["id"].startswith("code_")
    assert task["status"] == "queued"
    assert task["pending_prompt"] == "implement a small fix"
    assert task["network_access"] is True
    assert task["notification_refs"] == ["cli:test"]
    assert task["provider_thread_id"] is None
    assert task["provider_network_access"] is None
    assert task["restart_provider_session"] is False
    assert store.get_task(task["id"])["prompt"] == "implement a small fix"

    updated = store.update_task(task["id"], status="running", changed_files=["app.py"])
    assert updated["status"] == "running"
    assert updated["changed_files"] == ["app.py"]

    event = store.append_event(task["id"], "coding", "Coding log", "hello", {"x": 1})
    assert event["task_id"] == task["id"]

    logs = store.read_logs(task["id"])
    assert any(log["title"] == "Task queued" for log in logs)
    assert logs[-1]["message"] == "hello"

    listed = store.list_tasks()
    assert [item["id"] for item in listed] == [task["id"]]


def test_task_store_deletes_task_and_log(tmp_path):
    store = CodingTaskStore(tmp_path / "coding_agents")
    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="old task",
    )
    store.append_event(task["id"], "coding", "Coding log", "done")

    deleted = store.delete_task(task["id"])

    assert deleted["id"] == task["id"]
    assert store.get_task(task["id"]) is None
    assert store.read_logs(task["id"]) == []


def test_task_store_clears_tasks_with_keep_ids(tmp_path):
    store = CodingTaskStore(tmp_path / "coding_agents")
    kept = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="running task",
    )
    removed = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="old task",
    )
    store.append_event(kept["id"], "coding", "Coding log", "keep")
    store.append_event(removed["id"], "coding", "Coding log", "remove")

    result = store.clear_tasks(keep_ids={kept["id"]})

    assert result["deleted"] == 1
    assert result["preserved"] == 1
    assert store.get_task(kept["id"]) is not None
    assert store.get_task(removed["id"]) is None
    assert store.read_logs(kept["id"])
    assert store.read_logs(removed["id"]) == []
