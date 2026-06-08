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


def test_coding_events_are_saved_to_chat_history_and_learning(monkeypatch, tmp_path):
    import main
    from coding_agents.config import load_settings
    from coding_agents.task_store import CodingTaskStore
    from local_channel_store import get_recent_history_items

    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "coding_history_agent")
    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", None)

    learned = []

    async def fake_background_memory_update(
        user_prompt,
        ai_resp,
        semantic_ctx,
        episodic_ctx,
        procedural_ctx,
        current_date=None,
    ):
        learned.append((user_prompt, ai_resp, semantic_ctx, episodic_ctx, procedural_ctx, current_date))

    monkeypatch.setattr(main, "background_memory_update", fake_background_memory_update)

    settings = load_settings()
    store = CodingTaskStore(settings.task_root)
    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="clean data and train model",
        conversation_ref="cli",
    )
    store.update_task(task["id"], notification_refs=["cli", "telegram:12345"])

    async def exercise():
        await main.record_coding_event(
            "coding",
            "Coding progress",
            "Stage: training\nStatus: running\nDetail: epoch 1 started",
            {"task_id": task["id"], "status": "running", "provider": "codex"},
        )
        for _ in range(10):
            if learned:
                return
            await asyncio.sleep(0)

    asyncio.run(exercise())

    cli_history = get_recent_history_items("cli", limit=4)
    telegram_history = get_recent_history_items("telegram", "12345", limit=4)

    assert "clean data and train model" in cli_history[-2]["content"]
    assert "Coding progress" in cli_history[-1]["content"]
    assert "epoch 1 started" in telegram_history[-1]["content"]
    assert learned
    assert "Review and remember this conversation history" in learned[0][0]
    assert "Coding progress" in learned[0][0]
