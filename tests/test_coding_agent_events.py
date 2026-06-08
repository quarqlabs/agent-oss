import asyncio

from coding_agents.codex_runner import (
    CodexRunner,
    activity_message,
    continuation_restart_prompt,
    describe_current_work,
    extract_codex_tool_result,
    extract_text_payload,
    format_progress_record,
    is_codex_session_not_found,
    progress_reporting_prompt,
)
from coding_agents.config import CodingAgentSettings
from coding_agents.task_store import CodingTaskStore


def make_settings(tmp_path):
    return CodingAgentSettings(
        enabled=True,
        default_provider="codex",
        codex_command="codex",
        codex_args=["mcp-server"],
        workspace_root=tmp_path,
        approval_policy="argus-safe-auto",
        network_access=True,
        timeout_seconds=30,
        memory_root=tmp_path / "memory",
        agent_id="event_agent",
    )


def test_extract_text_payload_decodes_final_string():
    assert extract_text_payload('"hello\\nworld"') == "hello\nworld"


def test_extract_codex_tool_result_reads_thread_and_content():
    result = {
        "structuredContent": {
            "threadId": "thread_123",
            "content": "Done.",
        }
    }

    assert extract_codex_tool_result(result) == ("thread_123", "Done.")


def test_extract_codex_tool_result_reads_json_text_content():
    result = {
        "content": [
            {
                "type": "text",
                "text": '{"threadId":"thread_456","content":"Continued."}',
            }
        ]
    }

    assert extract_codex_tool_result(result) == ("thread_456", "Continued.")


def test_detects_missing_codex_session_and_builds_restart_prompt():
    assert is_codex_session_not_found("Session not found for thread_id: abc123")

    prompt = continuation_restart_prompt(
        {
            "prompt": "create a CNN project",
            "result_summary": "Created project in cnn-image-classifier-learning.",
        },
        "install dependencies and run it",
    )

    assert "previous Codex MCP session was no longer available" in prompt
    assert "create a CNN project" in prompt
    assert "Created project" in prompt
    assert "install dependencies and run it" in prompt


def test_codex_activity_updates_task_and_emits_replaceable_event(tmp_path):
    store = CodingTaskStore(tmp_path / "coding_agents")
    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="install packages",
    )
    emitted = []

    async def emit(kind, title, message, data):
        emitted.append((kind, title, message, data))

    runner = CodexRunner(make_settings(tmp_path), store, emit)

    asyncio.run(
        runner._activity(
            task["id"],
            "running fresh Codex session",
            "install packages",
            elapsed_seconds=10,
        )
    )

    stored = store.get_task(task["id"])
    assert "Codex is working" in stored["current_activity"]
    assert stored["current_activity_at"]
    assert emitted[-1][1] == "Codex is working"
    assert emitted[-1][3]["replace_key"] == f"coding:{task['id']}:activity"
    assert emitted[-1][3]["current_work"] == "setting up the project environment and installing dependencies"
    assert activity_message("running tests", "pytest", 5).startswith("Codex is working")
    assert "Current work: running verification commands" in activity_message("running tests", "pytest", 5)


def test_describe_current_work_uses_task_language():
    assert "installing dependencies" in describe_current_work("create venv and pip install packages")
    assert "verification" in describe_current_work("run pytest and fix lint")
    assert "debugging" in describe_current_work("fix traceback in api")
    assert "training" in describe_current_work("train for 2 epochs")
    assert "preprocessing" in describe_current_work("cleaning and preprocessing CIFAR data")


def test_progress_prompt_and_formatting():
    prompt = progress_reporting_prompt("train the model", "code_123", ".argus/coding_progress/code_123.jsonl")

    assert ".argus/coding_progress/code_123.jsonl" in prompt
    assert "Append progress updates as JSON Lines" in prompt
    assert "Stage: preprocessing" in format_progress_record(
        {"stage": "preprocessing", "status": "running", "detail": "normalizing images"}
    )


def test_codex_runner_filters_noisy_stream_events_and_surfaces_errors(tmp_path):
    store = CodingTaskStore(tmp_path / "coding_agents")
    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="suggest a name",
    )
    emitted = []

    async def emit(kind, title, message, data):
        emitted.append((kind, title, message, data))

    runner = CodexRunner(make_settings(tmp_path), store, emit)

    async def exercise():
        await runner._handle_stream_event(
            task["id"],
            {
                "type": "raw_response_event",
                "data": {
                    "type": "response.output_text.delta",
                    "delta": "one-token-fragment",
                },
            },
        )
        await runner._handle_stream_event(
            task["id"],
            {
                "type": "agent_updated_stream_event",
                "new_agent": {"instructions": "large internal JSON"},
            },
        )
        await runner._handle_stream_event(
            task["id"],
            {
                "type": "tool_call_output_item",
                "output": {
                    "type": "text",
                    "text": "Missing environment variable: `OPENAI_API_KEY`.",
                },
            },
        )

    asyncio.run(exercise())

    assert [item[1] for item in emitted] == ["Coding failed"]
    assert store.get_task(task["id"])["status"] == "failed"
    assert "OPENAI_API_KEY" in store.get_task(task["id"])["error"]


def test_codex_runner_emits_progress_file_updates(tmp_path):
    store = CodingTaskStore(tmp_path / "coding_agents")
    task = store.create_task(
        provider="codex",
        workspace_path=str(tmp_path),
        prompt="train model",
    )
    progress_path = tmp_path / ".argus" / "coding_progress" / f"{task['id']}.jsonl"
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(
        '{"stage":"cleaning","status":"running","detail":"removed bad samples"}\n'
        "plain progress line\n",
        encoding="utf-8",
    )
    emitted = []

    async def emit(kind, title, message, data):
        emitted.append((kind, title, message, data))

    runner = CodexRunner(make_settings(tmp_path), store, emit)

    asyncio.run(runner._emit_progress_updates(task["id"], progress_path, {"offset": 0}))

    assert [item[1] for item in emitted] == ["Coding progress", "Coding progress"]
    assert "removed bad samples" in emitted[0][2]
    assert "plain progress line" in emitted[1][2]
    assert "Codex progress" in store.get_task(task["id"])["current_activity"]
