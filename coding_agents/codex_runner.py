"""Codex provider runner backed by Codex CLI over MCP."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from coding_agents.approval import classify_action
from coding_agents.config import CodingAgentSettings
from coding_agents.task_store import CodingTaskStore, utc_now


EmitCallback = Callable[[str, str, str, dict[str, Any]], Awaitable[Any]]


KNOWN_CODEX_COMMANDS = (
    Path("/Applications/Codex.app/Contents/Resources/codex"),
    Path.home() / "Applications" / "Codex.app" / "Contents" / "Resources" / "codex",
)


_CODEX_MCP_SESSIONS: dict[str, "CodexMcpSession"] = {}
PROGRESS_RELATIVE_DIR = Path(".argus") / "coding_progress"


def resolve_codex_command(command: str) -> str | None:
    raw = str(command or "").strip()
    if not raw:
        return None

    found = shutil.which(raw)
    if found:
        return found

    expanded = Path(raw).expanduser()
    if expanded.exists() and os.access(expanded, os.X_OK):
        return str(expanded)

    if raw == "codex":
        for path in KNOWN_CODEX_COMMANDS:
            if path.exists() and os.access(path, os.X_OK):
                return str(path)

    return None


def runtime_check(settings: CodingAgentSettings) -> tuple[bool, str]:
    if not settings.codex_command:
        return False, "CODEX_MCP_COMMAND is empty."
    if (
        settings.codex_command == "npx"
        and len(settings.codex_args) >= 2
        and settings.codex_args[0] == "-y"
        and settings.codex_args[1] == "codex"
    ):
        return False, (
            "`npx -y codex` resolves to an unrelated npm package named `codex`, "
            "not the OpenAI Codex CLI. Install/open OpenAI Codex CLI and set "
            "CODEX_MCP_COMMAND=codex and CODEX_MCP_ARGS=mcp-server."
        )
    if resolve_codex_command(settings.codex_command) is None:
        return False, (
            f"Could not find `{settings.codex_command}` on PATH or in the standard "
            "Codex.app location. Install/open OpenAI Codex CLI, or set "
            "CODEX_MCP_COMMAND to the full executable path that can launch Codex MCP. "
            "For OpenAI Codex CLI, use CODEX_MCP_COMMAND=codex and "
            "CODEX_MCP_ARGS=mcp-server."
        )
    if importlib.util.find_spec("agents") is None:
        return False, (
            "The OpenAI Agents SDK is not installed. Install `openai-agents` "
            "from requirements.txt, then restart Argus."
        )
    return True, ""


def stringify_event(event: Any) -> str:
    for attr in ("delta", "text", "content", "message"):
        value = getattr(event, attr, None)
        if value:
            return str(value)

    data = getattr(event, "data", None)
    if data is not None and data is not event:
        text = stringify_event(data)
        if text and text != repr(data):
            return text

    item = getattr(event, "item", None)
    if item is not None and item is not event:
        text = stringify_event(item)
        if text and text != repr(item):
            return text

    try:
        return json.dumps(to_jsonable(event), ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(event)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return {
            key: to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def event_type(event: Any) -> str:
    return str(getattr(event, "type", "") or getattr(event, "event", "") or "").lower()


def extract_changed_files(text: str) -> list[str]:
    files = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().strip("-* ")
        if not line:
            continue
        if any(line.endswith(suffix) for suffix in (".py", ".js", ".ts", ".tsx", ".md", ".json", ".toml", ".yaml", ".yml")):
            files.append(line)
    return files[:50]


def codex_subprocess_env() -> dict[str, str]:
    return dict(os.environ)


def is_codex_session_not_found(text: str) -> bool:
    lowered = str(text or "").lower()
    return "session not found" in lowered or "thread_id" in lowered and "not found" in lowered


def continuation_restart_prompt(task: dict[str, Any], prompt: str) -> str:
    original_prompt = str(task.get("prompt") or "").strip()
    result_summary = str(task.get("result_summary") or "").strip()
    return (
        "Continue this existing Argus coding task. The previous Codex MCP session "
        "was no longer available, so inspect the workspace and continue from the "
        "files already on disk.\n\n"
        f"Saved task context:\n{original_prompt or '(none)'}\n\n"
        f"Last saved result:\n{result_summary or '(none)'}\n\n"
        f"User continuation request:\n{prompt}"
    )


def progress_relative_path(task_id: str) -> str:
    return str(PROGRESS_RELATIVE_DIR / f"{task_id}.jsonl")


def progress_file_path(workspace_path: str, task_id: str) -> Path:
    return Path(workspace_path).expanduser().resolve() / progress_relative_path(task_id)


def progress_reporting_prompt(prompt: str, task_id: str, progress_path: str) -> str:
    return (
        f"{prompt.strip()}\n\n"
        "Argus progress reporting requirement:\n"
        f"- This task id is `{task_id}`.\n"
        f"- Append progress updates as JSON Lines to `{progress_path}` in the workspace.\n"
        "- Write an update before and after each major step, including planning, file inspection, "
        "data cleaning, preprocessing, training, evaluation, tests, and failures.\n"
        "- Each JSON line should contain keys like `stage`, `step`, `status`, `detail`, "
        "`command`, and `files` when useful.\n"
        "- Keep progress updates short and user-facing. Do not include secrets.\n"
        "- Continue normally even if writing this progress file fails."
    ).strip()


def describe_current_work(prompt: str) -> str:
    text = str(prompt or "").lower()
    if any(word in text for word in ("venv", "virtualenv", "pip install", "install package", "dependency", "dependencies", "package.json", "npm install", "pnpm", "yarn")):
        return "setting up the project environment and installing dependencies"
    if any(word in text for word in ("test", "pytest", "unittest", "jest", "vitest", "lint", "typecheck", "build", "compile")):
        return "running verification commands and checking failures"
    if any(word in text for word in ("train", "training", "epoch", "model.fit", "fine tune", "finetune")):
        return "training the model and watching run metrics"
    if any(word in text for word in ("preprocess", "preprocessing", "process data", "processing", "cleaning", "clean data", "dataset", "cifar", "images")):
        return "cleaning and preprocessing dataset files"
    if any(word in text for word in ("bug", "fix", "error", "exception", "traceback", "debug", "failing")):
        return "debugging the failing path and preparing a fix"
    if any(word in text for word in ("refactor", "cleanup", "clean up", "rename", "restructure")):
        return "refactoring code and checking affected files"
    if any(word in text for word in ("read", "inspect", "analyze", "find out", "understand", "search")):
        return "inspecting project files and existing implementation"
    if any(word in text for word in ("create", "implement", "add", "build", "scaffold", "write", "update")):
        return "planning and applying code changes in the workspace"
    return "working inside the coding workspace"


def activity_message(action: str, prompt: str, elapsed_seconds: int = 0) -> str:
    prompt_line = str(prompt or "").strip().splitlines()[0][:160]
    elapsed = f"\nElapsed: {elapsed_seconds}s" if elapsed_seconds else ""
    current_work = f"\nCurrent work: {describe_current_work(prompt)}"
    request = f"\nRequest: {prompt_line}" if prompt_line else ""
    return f"Codex is working: {action}.{elapsed}{current_work}{request}"


def format_progress_record(record: Any) -> str:
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return str(record).strip()

    parts = []
    for key in ("stage", "step", "status", "detail", "command"):
        value = record.get(key)
        if value:
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {value}")
    files = record.get("files")
    if files:
        if isinstance(files, (list, tuple)):
            files_text = ", ".join(str(item) for item in files[:8])
        else:
            files_text = str(files)
        parts.append(f"Files: {files_text}")
    return "\n".join(parts).strip()


def codex_task_arguments(prompt: str, network_access: bool) -> dict[str, Any]:
    return {
        "approval-policy": "never",
        "sandbox": "workspace-write",
        "config": {
            "sandbox_workspace_write": {
                "network_access": bool(network_access),
            },
        },
        "prompt": prompt,
        "cwd": ".",
    }


def should_restart_codex_thread(task: dict[str, Any], network_access: bool) -> bool:
    if not str(task.get("provider_thread_id") or "").strip():
        return False
    if task.get("restart_provider_session"):
        return True

    provider_network_access = task.get("provider_network_access")
    if provider_network_access is None:
        return bool(network_access)
    return bool(provider_network_access) != bool(network_access)


class CodexMcpSession:
    def __init__(self, task_id: str, params: dict[str, Any], timeout_seconds: int):
        self.task_id = task_id
        self.params = params
        self.timeout_seconds = timeout_seconds
        self._server: Any = None
        self._lock = asyncio.Lock()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        async with self._lock:
            server = await self._ensure_server()
            return await server.call_tool(name, arguments)

    async def close(self) -> None:
        async with self._lock:
            if self._server is None:
                return
            server = self._server
            self._server = None
            await server.__aexit__(None, None, None)

    async def _ensure_server(self) -> Any:
        if self._server is not None:
            return self._server

        from agents.mcp import MCPServerStdio

        server = MCPServerStdio(
            name="Codex CLI",
            params=self.params,
            cache_tools_list=True,
            client_session_timeout_seconds=max(self.timeout_seconds, 3600),
        )
        self._server = await server.__aenter__()
        return self._server


def get_codex_session(
    task_id: str,
    params: dict[str, Any],
    timeout_seconds: int,
) -> CodexMcpSession:
    session = _CODEX_MCP_SESSIONS.get(task_id)
    if session is None:
        session = CodexMcpSession(task_id, params, timeout_seconds)
        _CODEX_MCP_SESSIONS[task_id] = session
    return session


async def close_codex_session(task_id: str) -> None:
    session = _CODEX_MCP_SESSIONS.pop(str(task_id), None)
    if session is not None:
        await session.close()


async def close_all_codex_sessions() -> None:
    task_ids = list(_CODEX_MCP_SESSIONS)
    for task_id in task_ids:
        await close_codex_session(task_id)


def schedule_close_codex_session(task_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(close_codex_session(task_id))


def provider_event_type(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("type", "event"):
        value = data.get(key)
        if value:
            event_name = str(value).lower()
            if event_name in {"raw_response_event", "run_item_stream_event"}:
                break
            return event_name
    for key in ("data", "item", "raw_item"):
        nested = data.get(key)
        nested_type = provider_event_type(nested)
        if nested_type:
            return nested_type
    return ""


def extract_text_payload(value: Any) -> str:
    value = to_jsonable(value)
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            if isinstance(decoded, str):
                return decoded.strip()
        return stripped
    if isinstance(value, list):
        return "\n".join(
            text for text in (extract_text_payload(item) for item in value) if text
        ).strip()
    if not isinstance(value, dict):
        return ""

    direct = value.get("text")
    if isinstance(direct, str):
        return direct.strip()

    for key in ("output", "content", "final_output", "result"):
        text = extract_text_payload(value.get(key))
        if text:
            return text

    for key in ("raw_item", "item", "data", "message"):
        text = extract_text_payload(value.get(key))
        if text:
            return text

    return ""


def extract_codex_tool_result(result: Any) -> tuple[str | None, str]:
    data = to_jsonable(result)

    def walk(value: Any) -> tuple[str | None, str]:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") or text.startswith("["):
                try:
                    return walk(json.loads(text))
                except json.JSONDecodeError:
                    pass
            return None, text
        if isinstance(value, dict):
            thread_id = value.get("threadId") or value.get("conversationId")
            content = value.get("content")
            if isinstance(content, str):
                nested_thread, nested_content = walk(content)
                return (
                    str(thread_id or nested_thread) if (thread_id or nested_thread) else None,
                    nested_content or content.strip(),
                )
            text = value.get("text")
            if isinstance(text, str):
                nested_thread, nested_content = walk(text)
                return (
                    str(thread_id or nested_thread) if (thread_id or nested_thread) else None,
                    nested_content or text.strip(),
                )
            for key in ("structuredContent", "structured_content", "output", "data", "result"):
                nested_thread, nested_content = walk(value.get(key))
                if nested_thread or nested_content:
                    return (str(thread_id or nested_thread) if (thread_id or nested_thread) else None, nested_content)
            for key in ("content", "items"):
                nested_thread, nested_content = walk(value.get(key))
                if nested_thread or nested_content:
                    return (str(thread_id or nested_thread) if (thread_id or nested_thread) else None, nested_content)
        elif isinstance(value, list):
            thread_id = None
            pieces = []
            for item in value:
                item_thread, item_content = walk(item)
                thread_id = thread_id or item_thread
                if item_content:
                    pieces.append(item_content)
            return thread_id, "\n".join(pieces).strip()
        return None, ""

    return walk(data)


def is_noisy_codex_event(provider_type: str) -> bool:
    return provider_type in {
        "agent_updated_stream_event",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_text.delta",
        "response.content_part.added",
        "response.content_part.done",
        "function_call",
        "tool_call_item",
        "message_output_item",
    }


def is_approval_request_event(provider_type: str, data: Any, text: str) -> bool:
    if "approval" in provider_type and "request" in provider_type:
        return True
    if isinstance(data, dict):
        lowered_keys = {str(key).lower() for key in data}
        if {"approval_request", "requires_approval", "approval"} & lowered_keys:
            return True
    lowered_text = text.lower()
    return "approval request" in lowered_text or "requires approval" in lowered_text


class CodexRunner:
    def __init__(
        self,
        settings: CodingAgentSettings,
        store: CodingTaskStore,
        emit: EmitCallback,
    ):
        self.settings = settings
        self.store = store
        self.emit = emit
        self.activity_interval_seconds = 5

    async def run(self, task_id: str) -> None:
        task = self.store.get_task(task_id)
        if not task:
            return

        ok, reason = runtime_check(self.settings)
        if not ok:
            await self._fail(task_id, reason)
            return

        try:
            await asyncio.wait_for(self._run_inner(task), timeout=self.settings.timeout_seconds)
        except asyncio.CancelledError:
            self.store.update_task(task_id, status="cancelled")
            await self._log_emit(task_id, "coding", "Coding cancelled", "Task was cancelled.", {"status": "cancelled"})
            raise
        except asyncio.TimeoutError:
            await self._fail(task_id, f"Task timed out after {self.settings.timeout_seconds} seconds.")
        except Exception as exc:
            await self._fail(task_id, self._friendly_error(exc))

    async def _run_inner(self, task: dict[str, Any]) -> None:
        task_id = str(task["id"])
        workspace_path = str(task.get("workspace_path") or self.settings.workspace_root)
        prompt = str(task.get("pending_prompt") or task.get("last_user_reply") or task.get("prompt") or "").strip()
        provider_thread_id = str(task.get("provider_thread_id") or "").strip() or None
        network_access = bool(task.get("network_access", self.settings.network_access))
        restart_provider_session = should_restart_codex_thread(task, network_access)
        progress_path = progress_file_path(workspace_path, task_id)
        progress_rel_path = progress_relative_path(task_id)
        self._prepare_progress_file(progress_path)
        self.store.update_task(task_id, status="running")
        await self._log_emit(
            task_id,
            "coding",
            "Coding started",
            f"Provider: Codex\nWorkspace: {workspace_path}\nNetwork: {'on' if network_access else 'off'}",
            {
                "status": "running",
                "provider": "codex",
                "workspace_path": workspace_path,
                "network_access": network_access,
            },
        )

        params = {
            "command": resolve_codex_command(self.settings.codex_command) or self.settings.codex_command,
            "args": self.settings.codex_args,
            "cwd": workspace_path,
            "env": codex_subprocess_env(),
        }
        if restart_provider_session:
            await close_codex_session(task_id)
        mcp_session = get_codex_session(task_id, params, self.settings.timeout_seconds)

        if provider_thread_id and restart_provider_session:
            await self._log_emit(
                task_id,
                "coding",
                "Coding session restarted",
                (
                    "Restarting Codex from the saved task context so the current "
                    f"network setting applies: {'on' if network_access else 'off'}."
                ),
                {
                    "status": "running",
                    "provider": "codex",
                    "old_thread_id": provider_thread_id,
                    "network_access": network_access,
                },
            )
            prompt = continuation_restart_prompt(task, prompt)
            provider_thread_id = None
            provider_prompt = progress_reporting_prompt(prompt, task_id, progress_rel_path)
            result = await self._call_codex_tool_with_activity(
                task_id,
                mcp_session,
                "codex",
                codex_task_arguments(provider_prompt, network_access),
                action="restarting Codex session with updated network mode",
                prompt=prompt,
                progress_path=progress_path,
            )
            thread_id, summary = extract_codex_tool_result(result)
        elif provider_thread_id:
            await self._log_emit(
                task_id,
                "coding",
                "Coding continued",
                "Continuing the existing Codex session.",
                {"status": "running", "provider": "codex", "thread_id": provider_thread_id},
            )
            result = await self._call_codex_tool_with_activity(
                task_id,
                mcp_session,
                "codex-reply",
                {
                    "threadId": provider_thread_id,
                    "prompt": progress_reporting_prompt(prompt, task_id, progress_rel_path),
                },
                action="continuing existing Codex session",
                prompt=prompt,
                progress_path=progress_path,
            )
            thread_id, summary = extract_codex_tool_result(result)
            if is_codex_session_not_found(summary):
                await self._log_emit(
                    task_id,
                    "coding",
                    "Coding session restarted",
                    "The previous Codex session was not available, so Codex is continuing from the saved task context and workspace files.",
                    {"status": "running", "provider": "codex", "old_thread_id": provider_thread_id},
                )
                prompt = continuation_restart_prompt(task, prompt)
                provider_thread_id = None
                provider_prompt = progress_reporting_prompt(prompt, task_id, progress_rel_path)
                result = await self._call_codex_tool_with_activity(
                    task_id,
                    mcp_session,
                    "codex",
                    codex_task_arguments(provider_prompt, network_access),
                    action="restarting from saved task context",
                    prompt=prompt,
                    progress_path=progress_path,
                )
                thread_id, summary = extract_codex_tool_result(result)
        else:
            await self._log_emit(
                task_id,
                "coding",
                "Coding running",
                "Codex is working in a fresh session.",
                {"status": "running", "provider": "codex"},
            )
            result = await self._call_codex_tool_with_activity(
                task_id,
                mcp_session,
                "codex",
                codex_task_arguments(
                    progress_reporting_prompt(prompt, task_id, progress_rel_path),
                    network_access,
                ),
                action="running fresh Codex session",
                prompt=prompt,
                progress_path=progress_path,
            )
            thread_id, summary = extract_codex_tool_result(result)

        if summary and "missing environment variable" in summary.lower():
            await self._fail(task_id, summary)
            return

        task = self.store.get_task(task_id) or {}
        if task.get("status") not in {"waiting_user", "cancelled", "failed"}:
            self.store.update_task(
                task_id,
                status="completed",
                result_summary=summary,
                provider_thread_id=thread_id or provider_thread_id,
                provider_network_access=network_access if (thread_id or provider_thread_id) else None,
                restart_provider_session=False,
                pending_prompt=None,
                current_activity=None,
                current_activity_at=None,
            )
            await self._log_emit(
                task_id,
                "coding",
                "Coding completed",
                summary or "Codex completed the task.",
                {
                    "status": "completed",
                    "thread_id": thread_id or provider_thread_id,
                    "network_access": network_access,
                },
                )

    async def _call_codex_tool_with_activity(
        self,
        task_id: str,
        mcp_session: CodexMcpSession,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        action: str,
        prompt: str,
        progress_path: Path | None = None,
    ) -> Any:
        started = time.monotonic()
        progress_state = {"offset": 0}
        call_task = asyncio.create_task(mcp_session.call_tool(tool_name, arguments))
        await self._activity(task_id, action, prompt, elapsed_seconds=0)
        await self._emit_progress_updates(task_id, progress_path, progress_state)

        while not call_task.done():
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(call_task),
                    timeout=self.activity_interval_seconds,
                )
                await self._emit_progress_updates(task_id, progress_path, progress_state)
                return result
            except asyncio.TimeoutError:
                elapsed_seconds = int(time.monotonic() - started)
                await self._emit_progress_updates(task_id, progress_path, progress_state)
                await self._activity(task_id, action, prompt, elapsed_seconds=elapsed_seconds)

        await self._emit_progress_updates(task_id, progress_path, progress_state)
        return await call_task

    def _prepare_progress_file(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        except OSError:
            return

    async def _emit_progress_updates(
        self,
        task_id: str,
        path: Path | None,
        state: dict[str, int],
    ) -> None:
        if path is None:
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(int(state.get("offset") or 0))
                lines = handle.readlines()
                state["offset"] = handle.tell()
        except OSError:
            return

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = line
            message = format_progress_record(record)
            if not message:
                continue
            self.store.update_task(
                task_id,
                current_activity=f"Codex progress:\n{message}",
                current_activity_at=utc_now(),
            )
            await self._log_emit(
                task_id,
                "coding",
                "Coding progress",
                message[:4000],
                {
                    "status": "running",
                    "provider": "codex",
                    "progress": record if isinstance(record, dict) else {"message": message},
                },
            )

    async def _activity(
        self,
        task_id: str,
        action: str,
        prompt: str,
        *,
        elapsed_seconds: int,
    ) -> None:
        message = activity_message(action, prompt, elapsed_seconds=elapsed_seconds)
        self.store.update_task(
            task_id,
            current_activity=message,
            current_activity_at=utc_now(),
        )
        await self._log_emit(
            task_id,
            "coding",
            "Codex is working",
            message,
            {
                "status": "running",
                "provider": "codex",
                "activity": action,
                "current_work": describe_current_work(prompt),
                "elapsed_seconds": elapsed_seconds,
                "replace_key": f"coding:{task_id}:activity",
            },
        )

    async def _handle_stream_event(self, task_id: str, event: Any) -> None:
        typ = event_type(event)
        data = to_jsonable(event)
        provider_type = provider_event_type(data) or typ
        text = extract_text_payload(data) or stringify_event(event).strip()

        if is_approval_request_event(provider_type, data, text):
            decision = classify_action(data if isinstance(data, dict) else {"message": text}, self.settings.workspace_root)
            if decision.get("allow"):
                await self._log_emit(
                    task_id,
                    "coding",
                    "Coding approval",
                    f"Auto-approved: {decision.get('reason')}",
                    {"status": "running", "approval": decision, "provider_event_type": typ},
                )
                return

            request = {"event": data, "message": text, "decision": decision}
            self.store.update_task(task_id, status="waiting_user", approval_request=request)
            await self._log_emit(
                task_id,
                "coding",
                "Coding approval needed",
                text or decision.get("reason") or "Codex needs user approval.",
                {"status": "waiting_user", "approval": decision, "provider_event_type": typ},
            )
            return

        if provider_type == "tool_call_output_item" and text:
            if "missing environment variable" in text.lower() or "error" in text.lower():
                await self._fail(task_id, text)
            return

        if is_noisy_codex_event(provider_type):
            return

        changed_files = extract_changed_files(text)
        if changed_files:
            task = self.store.get_task(task_id) or {}
            existing = list(task.get("changed_files") or [])
            for path in changed_files:
                if path not in existing:
                    existing.append(path)
            self.store.update_task(task_id, changed_files=existing)
            await self._log_emit(
                task_id,
                "coding",
                "Coding changed files",
                "\n".join(changed_files),
                {"changed_files": changed_files, "provider_event_type": typ},
            )
            return

        if text and len(text) > 2:
            await self._log_emit(
                task_id,
                "coding",
                "Coding log",
                text[:4000],
                {"provider_event_type": typ},
            )

    def _final_summary(self, streamed: Any) -> str:
        for attr in ("final_output", "final_response", "output", "result"):
            value = getattr(streamed, attr, None)
            if value:
                text = extract_text_payload(value)
                if text:
                    return text
                return stringify_event(value).strip()
        return ""

    def _friendly_error(self, exc: Exception) -> str:
        message = str(exc).strip() or exc.__class__.__name__
        if message.lower() == "connection closed":
            command = " ".join([self.settings.codex_command, *self.settings.codex_args]).strip()
            return (
                "Codex MCP connection closed before the task completed. "
                f"Check that `{command}` starts successfully from a terminal, "
                "and run `codex` once if the Codex CLI needs sign-in or local setup."
            )
        return message

    async def _fail(self, task_id: str, error: str) -> None:
        self.store.update_task(
            task_id,
            status="failed",
            error=error,
            current_activity=None,
            current_activity_at=None,
        )
        await self._log_emit(
            task_id,
            "coding",
            "Coding failed",
            error,
            {"status": "failed"},
        )

    async def _log_emit(
        self,
        task_id: str,
        kind: str,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        payload = {"task_id": task_id, **(data or {})}
        self.store.append_event(task_id, kind, title, message, payload)
        await self.emit(kind, title, message, payload)
