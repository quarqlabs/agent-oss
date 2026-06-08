import asyncio


def test_telegram_download_limit_message_is_user_facing(monkeypatch):
    import main

    monkeypatch.setattr(main, "CHANNEL_FILE_MAX_BYTES", 20_000_000)
    message = main.telegram_download_limit_message(25_000_000)

    assert "25.0 MB" in message
    assert "20.0 MB" in message
    assert "CHANNEL_FILE_MAX_BYTES" not in message
    assert "Please send a smaller file" in message


def test_attachment_failure_message_lists_files():
    import main

    message = main.format_attachment_failure_message(
        [
            {
                "filename": "large-video.mp4",
                "message": "That file is above this agent's Telegram download limit of 20.0 MB.",
            }
        ]
    )

    assert "large-video.mp4" in message
    assert "20.0 MB" in message


def test_coding_conversation_ref_helpers():
    import main

    assert main.coding_conversation_ref("telegram", "12345") == "telegram:12345"
    assert main.telegram_chat_id_from_conversation_ref("telegram:12345") == 12345
    assert main.telegram_chat_id_from_conversation_ref("cli") is None


def test_connecting_telegram_subscribes_active_coding_tasks(monkeypatch, tmp_path):
    import main
    from coding_agents.config import load_settings
    from coding_agents.manager import get_coding_manager
    from coding_agents.task_store import CodingTaskStore

    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "telegram_subscribe_agent")
    monkeypatch.setattr(main, "TELEGRAM_ALLOWED_USERS", {12345})
    monkeypatch.setattr(main, "list_chat_history_channels", lambda: [])

    sent = []

    async def fake_send(chat_id, text):
        sent.append((chat_id, text))

    monkeypatch.setattr(main, "send_telegram_message", fake_send)

    settings = load_settings()
    store = CodingTaskStore(settings.task_root)
    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="long training task",
        conversation_ref="cli",
    )
    store.update_task(task["id"], status="waiting_user")

    result = asyncio.run(main.subscribe_active_coding_tasks_to_channel("telegram"))
    updated = store.get_task(task["id"])

    assert result["task_count"] == 1
    assert "telegram:12345" in updated["notification_refs"]
    assert sent and sent[0][0] == 12345
