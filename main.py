# =====================================================
# Argus Agent — Single-Tenant Worker
# =====================================================
# This container serves exactly one user. Identity is injected at
# `docker run` time via the USER_ID environment variable; the Node
# dispatcher forwards prompts to POST /api/chat.

import os
import logging
import asyncio
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agent_connector import get_argus_response
from agent import (
    MEMORY_INGESTION_ACK,
    PENDING_LEARNING_TASKS,
    background_memory_update,
    wipe_all_memories_for_api,
    wrap_memory_ingestion_payload,
)
from agent_tools_config import handle_tool_command, load_enabled_cloud_tools
from coding_agents.codex_runner import close_all_codex_sessions
from coding_agents.config import load_settings as load_coding_settings
from coding_agents.manager import get_coding_manager
from coding_agents.task_store import CodingTaskStore
from local_channel_store import (
    append_attachment_note,
    append_chat_pair,
    decode_base64_payload,
    get_recent_history_items,
    list_chat_history_channels,
    recent_attachment_ids,
    refresh_attachments_for_context,
    render_attachment_context,
    store_attachment_from_bytes,
)
from tools.composio.client import clear_composio_session_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("argus_agent")
for noisy_logger in ("httpx", "httpcore", "openai", "openai._base_client"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


# =====================================================
# CONFIG
# =====================================================
load_dotenv()

AGENT_USER_ID = os.getenv("USER_ID")
if not AGENT_USER_ID:
    raise RuntimeError("USER_ID environment variable is required")

TELEGRAM_WEBHOOK_PATH = "/api/telegram/webhook"
TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAM_MESSAGE_LIMIT = 3900
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")
TELEGRAM_TYPING_INTERVAL_SECONDS = 4
DEFAULT_CHANNEL_FILE_MAX_BYTES = 20_000_000
CHANNEL_FILE_MAX_BYTES = int(os.getenv("CHANNEL_FILE_MAX_BYTES", str(DEFAULT_CHANNEL_FILE_MAX_BYTES)))
CODING_TELEGRAM_NOTIFY_SECONDS = int(os.getenv("CODING_TELEGRAM_NOTIFY_SECONDS", "30"))
EVENT_BUFFER_SIZE = 300
CHAT_HISTORY_WINDOW_MESSAGES = 8


app = FastAPI(title="Argus Agent", version="0.5.0")
EVENTS = deque(maxlen=EVENT_BUFFER_SIZE)
EVENT_LOCK = asyncio.Lock()
EVENT_SEQ = 0
JOBS: dict[str, dict] = {}
JOB_DONE_EVENTS: dict[str, asyncio.Event] = {}
JOB_QUEUE: asyncio.Queue[str] = asyncio.Queue()
JOB_LOCK = asyncio.Lock()
JOB_WORKER_TASK: asyncio.Task | None = None
CHAT_HISTORY_LOCK = asyncio.Lock()
CODING_TELEGRAM_LAST_SENT: dict[str, float] = {}
CODING_LEARNING_SEMAPHORE = asyncio.Semaphore(2)
CODING_HISTORY_TITLES = {
    "Coding task queued",
    "Coding setup needed",
    "Coding started",
    "Coding session restarted",
    "Coding progress",
    "Coding approval needed",
    "Coding reply",
    "Coding completed",
    "Coding failed",
    "Coding interrupted",
    "Coding cancelled",
}
CODING_LEARNING_TITLES = {
    "Coding task queued",
    "Coding setup needed",
    "Coding started",
    "Coding session restarted",
    "Coding progress",
    "Coding approval needed",
    "Coding reply",
    "Coding completed",
    "Coding failed",
    "Coding interrupted",
    "Coding cancelled",
}
CODING_HISTORY_MAX_CHARS = 4000
CODING_LEARNING_MAX_CHARS = 3000


class ChatRequest(BaseModel):
    prompt: str
    channel_type: str = "web"
    skip_learning: bool = False
    current_date: Optional[str] = None
    conversation_id: Optional[str] = None
    attachment_ids: list[str] = Field(default_factory=list)


class FileIngestRequest(BaseModel):
    data_base64: str
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    channel_type: str = "api"
    conversation_id: Optional[str] = None
    source_kind: str = "file"
    source_metadata: dict = Field(default_factory=dict)


class CodingTaskStartRequest(BaseModel):
    prompt: str
    provider: str = "codex"
    workspace_path: Optional[str] = None
    conversation_id: Optional[str] = None


class CodingTaskReplyRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class CodingAgentSelectRequest(BaseModel):
    provider: str


class CodingWorkspaceRequest(BaseModel):
    workspace_path: str


class CodingNetworkRequest(BaseModel):
    enabled: bool


class CodingChannelSubscribeRequest(BaseModel):
    channel_type: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def help_text() -> str:
    return "\n".join(
        [
            "Available commands:",
            "/help - show commands",
            "/status - show agent/API status",
            "/tools - list enabled native and cloud tools",
            "/which-tool <task> - show which tool fits a task",
            "/cloud-tools - list cloud tools available to enable",
            "/add-tool <tool> - enable a cloud tool",
            "/remove-tool <tool> - disable a cloud tool",
            "/coding <task> - start a new coding task",
            "/coding-new <task> - start a fresh coding session",
            "/coding-continue <message> - continue the latest coding session",
            "/coding-tasks - list recent coding task ids",
            "/coding-agents - list coding agents and current default",
            "/coding-use <provider> - set default coding agent",
            "/coding-workspace <path> - set default coding workspace",
            "/coding-network [on|off] - show or set default coding network access",
            "/coding-network <id> on|off - set network access for a coding task",
            "/coding-allow-network <id> - allow network access for a coding task",
            "/coding-status <id> - show coding task status",
            "/coding-log <id> - show recent coding task logs",
            "/coding-reply <id> <message> - reply to a coding task",
            "/coding-cancel <id> - cancel a coding task",
            "/coding-delete <id> - delete a completed/failed/cancelled coding task",
            "/coding-clear - delete all completed/failed/cancelled coding task history",
            "/wipe - clear local memories",
            "/quit - stop the local CLI only; remote channels cannot stop the process",
        ]
    )


def status_payload() -> dict:
    return {
        "status": "ok",
        "user_id": AGENT_USER_ID,
        "telegram_webhook_path": TELEGRAM_WEBHOOK_PATH,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN),
        "telegram_allowed_users_configured": TELEGRAM_ALLOWED_USERS is not None,
        "job_queue_size": JOB_QUEUE.qsize(),
        "enabled_cloud_tools": load_enabled_cloud_tools(),
        "coding_agent": get_coding_manager().get_config(),
        "active_coding_tasks": len(get_coding_manager().list_tasks(status_filter="active", limit=1000)),
        "chat_history_channels": list_chat_history_channels(),
    }


def status_text() -> str:
    payload = status_payload()
    coding_agent = payload.pop("coding_agent", {}) or {}
    rows = [f"{key}: {value}" for key, value in payload.items()]
    if coding_agent:
        rows.append(
            f"coding_agent: {coding_agent.get('default_provider_label')} "
            f"(`{coding_agent.get('default_provider')}`)"
        )
        rows.append(f"coding_workspace: {coding_agent.get('workspace_root')}")
        rows.append(f"coding_network: {'on' if coding_agent.get('network_access') else 'off'}")
    return "\n".join(rows)


def format_coding_agent_config(config: dict) -> str:
    providers = config.get("providers") or []
    rows = [
        f"Default coding agent: {config.get('default_provider_label')} (`{config.get('default_provider')}`)",
        f"Workspace: {config.get('workspace_root')}",
        f"Approval policy: {config.get('approval_policy')}",
        f"Network access: {'on' if config.get('network_access') else 'off'}",
        f"Active tasks: {config.get('active_tasks', 0)}",
        "",
        "Available coding agents:",
    ]
    rows.extend(
        f"- {item['label']} (`{item['slug']}`) [{item['status']}] - {item['description']}"
        for item in providers
    )
    rows.append("")
    rows.append("Use `/coding-use codex`, `/coding-workspace <path>`, and `/coding-network on|off` to change defaults.")
    return "\n".join(rows)


def format_coding_tasks(tasks: list[dict], active_only: bool = False) -> str:
    if not tasks:
        return (
            "No active coding tasks. Start one with `/coding <task>`."
            if active_only
            else "No coding tasks yet. Start one with `/coding <task>`."
        )
    rows = ["Active coding tasks:" if active_only else "Recent coding tasks:"]
    for task in tasks[:20]:
        marker = " needs input" if task.get("approval_request") or task.get("status") == "waiting_user" else ""
        rows.append(
            f"- {task.get('id')} [{task.get('status')}]{marker} "
            f"net={'on' if task.get('network_access', True) else 'off'} "
            f"{str(task.get('prompt') or '').splitlines()[0][:90]}"
        )
    rows.append("")
    rows.append("Continue latest with `/coding-continue <message>`. Start fresh with `/coding-new <task>`.")
    rows.append("Inspect with `/coding-status <id>` or `/coding-log <id>`. Reply to a specific task with `/coding-reply <id> <message>`.")
    rows.append("Delete history with `/coding-delete <id>` or `/coding-clear`.")
    rows.append("In the CLI, type an id command plus a space, choose a task with Ctrl+N/P, then press Tab.")
    return "\n".join(rows)


def format_coding_logs(logs: list[dict]) -> str:
    if not logs:
        return "No coding task logs found."
    rows = []
    for event in logs:
        rows.append(
            f"- {event.get('time', '')} {event.get('title', 'Log')}: "
            f"{str(event.get('message') or '').strip()}"
        )
    return "\n".join(rows)


def format_coding_task(task: dict | None) -> str:
    if not task:
        return "Coding task not found."
    rows = [
        f"Task id: {task.get('id')}",
        f"Status: {task.get('status')}",
        f"Provider: {task.get('provider')}",
        f"Workspace: {task.get('workspace_path')}",
        f"Network: {'on' if task.get('network_access', True) else 'off'}",
    ]
    if task.get("approval_request"):
        rows.append("Approval needed: yes")
    if task.get("provider_thread_id"):
        rows.append("Codex session: connected")
    if task.get("restart_provider_session"):
        rows.append("Codex session: will restart on next continuation")
    if task.get("current_activity"):
        rows.append(f"Current activity: {task.get('current_activity')}")
    if task.get("changed_files"):
        rows.append("Changed files:\n" + "\n".join(f"- {path}" for path in task["changed_files"]))
    if task.get("error"):
        rows.append(f"Error: {task.get('error')}")
    if task.get("result_summary"):
        rows.append(f"Summary: {task.get('result_summary')}")
    return "\n".join(rows)


def coding_conversation_ref(channel_type: str, conversation_id: str | None = None) -> str:
    channel = str(channel_type or "cli").strip() or "cli"
    if conversation_id:
        return f"{channel}:{conversation_id}"
    return channel


def parse_on_off(value: str) -> bool | None:
    lowered = str(value or "").strip().lower()
    if lowered in {"on", "true", "1", "yes", "enable", "enabled", "allow", "allowed"}:
        return True
    if lowered in {"off", "false", "0", "no", "disable", "disabled", "block", "blocked"}:
        return False
    return None


def telegram_chat_id_from_conversation_ref(conversation_ref: str | None) -> int | None:
    prefix = "telegram:"
    if not str(conversation_ref or "").startswith(prefix):
        return None
    value = str(conversation_ref)[len(prefix):]
    try:
        return int(value)
    except ValueError:
        return None


def channel_from_conversation_ref(conversation_ref: str | None) -> tuple[str, str | None] | None:
    text = str(conversation_ref or "").strip()
    if not text:
        return None
    if ":" not in text:
        return text, None
    channel, conversation_id = text.split(":", 1)
    channel = channel.strip() or "web"
    conversation_id = conversation_id.strip() or None
    return channel, conversation_id


def known_telegram_conversation_refs() -> list[str]:
    refs = []
    for channel in list_chat_history_channels():
        if str(channel).startswith("telegram:") and channel not in refs:
            refs.append(str(channel))

    for user_id in TELEGRAM_ALLOWED_USERS or []:
        ref = f"telegram:{int(user_id)}"
        if ref not in refs:
            refs.append(ref)

    return refs


async def subscribe_active_coding_tasks_to_channel(channel_type: str) -> dict:
    channel = str(channel_type or "").strip().lower()
    if channel != "telegram":
        return {
            "channel_type": channel,
            "subscribed_refs": [],
            "task_count": 0,
            "message": f"Channel `{channel}` does not support coding progress subscriptions yet.",
        }

    refs = known_telegram_conversation_refs()
    if not refs:
        return {
            "channel_type": "telegram",
            "subscribed_refs": [],
            "task_count": 0,
            "message": (
                "Telegram is connected, but no Telegram chat id is known yet. "
                "Send `/coding` or `/coding-status <task_id>` from Telegram once to subscribe that chat."
            ),
        }

    manager = get_coding_manager()
    tasks = manager.list_tasks(status_filter="active", limit=100)
    subscribed_task_ids = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        if not task_id:
            continue
        for ref in refs:
            manager.subscribe_task(task_id, ref)
        subscribed_task_ids.append(task_id)

    if subscribed_task_ids:
        summary = (
            "Telegram is now subscribed to active coding task updates.\n"
            + "\n".join(f"- {task_id}" for task_id in subscribed_task_ids[:10])
        )
        for ref in refs:
            chat_id = telegram_chat_id_from_conversation_ref(ref)
            if chat_id is None:
                continue
            try:
                await send_telegram_message(chat_id, summary)
            except Exception as exc:
                logger.debug("Telegram coding subscription notice failed: %s", exc)

    return {
        "channel_type": "telegram",
        "subscribed_refs": refs,
        "task_count": len(subscribed_task_ids),
        "task_ids": subscribed_task_ids,
        "message": (
            f"Subscribed Telegram to {len(subscribed_task_ids)} active coding task(s)."
            if subscribed_task_ids
            else "Telegram is connected. No active coding tasks are running right now."
        ),
    }


def handle_coding_command(
    prompt: str,
    channel_type: str = "cli",
    conversation_id: str | None = None,
) -> str | None:
    stripped = str(prompt or "").strip()
    if not stripped:
        return None
    command, _, rest = stripped.partition(" ")
    command = command.lower().split("@", 1)[0]
    manager = get_coding_manager()
    conversation_ref = coding_conversation_ref(channel_type, conversation_id)

    if command in {"/coding", "/coding-new", "/coding-tasks"}:
        if not rest.strip():
            tasks = manager.list_tasks(status_filter="all", limit=20)
            if channel_type != "cli":
                for task in tasks:
                    if task.get("status") in {"queued", "running", "waiting_user"}:
                        manager.subscribe_task(str(task.get("id")), conversation_ref)
            return format_coding_tasks(
                tasks,
                active_only=False,
            )
        if command == "/coding-tasks":
            tasks = manager.list_tasks(status_filter="all", limit=20)
            if channel_type != "cli":
                for task in tasks:
                    if task.get("status") in {"queued", "running", "waiting_user"}:
                        manager.subscribe_task(str(task.get("id")), conversation_ref)
            return format_coding_tasks(
                tasks,
                active_only=False,
            )
        result = manager.start_task(rest.strip(), conversation_ref=conversation_ref)
        task = result.get("task")
        if not task:
            return str(result.get("message") or "Coding task could not be started.")
        return f"{result.get('message')}\n\n{format_coding_task(task)}"
    if command == "/coding-continue":
        if not rest.strip():
            return "Usage: `/coding-continue <message>`\nOr: `/coding-continue <task_id> <message>`"
        task_id, _, message = rest.partition(" ")
        if task_id.startswith("code_") and message.strip():
            try:
                result = manager.reply_to_task(task_id, message.strip())
                manager.subscribe_task(task_id, conversation_ref)
            except KeyError:
                return f"Coding task not found: {task_id}"
        else:
            result = manager.reply_to_latest(rest.strip(), conversation_ref=conversation_ref)
        return f"{result.get('message')}\n\n{format_coding_task(result.get('task'))}"
    if command in {"/coding-agents", "/coding-config"}:
        return format_coding_agent_config(manager.get_config())
    if command == "/coding-use":
        if not rest.strip():
            return "Usage: `/coding-use <provider>`\nExample: `/coding-use codex`"
        try:
            config = manager.set_default_provider(rest.strip())
        except ValueError as exc:
            return str(exc)
        return format_coding_agent_config(config)
    if command == "/coding-workspace":
        if not rest.strip():
            return "Usage: `/coding-workspace <path>`"
        config = manager.set_workspace_root(rest.strip())
        return format_coding_agent_config(config)
    if command == "/coding-network":
        args = rest.split()
        if not args:
            return format_coding_agent_config(manager.get_config())
        if args[0].startswith("code_"):
            if len(args) < 2:
                return "Usage: `/coding-network <task_id> on|off`"
            enabled = parse_on_off(args[1])
            if enabled is None:
                return "Usage: `/coding-network <task_id> on|off`"
            try:
                result = manager.set_task_network_access(args[0], enabled)
            except KeyError:
                return f"Coding task not found: {args[0]}"
            return f"{result.get('message')}\n\n{format_coding_task(result.get('task'))}"
        enabled = parse_on_off(args[0])
        if enabled is None:
            return "Usage: `/coding-network on|off`\nOr: `/coding-network <task_id> on|off`"
        return format_coding_agent_config(manager.set_default_network_access(enabled))
    if command == "/coding-allow-network":
        task_id = rest.strip()
        if not task_id:
            return "Usage: `/coding-allow-network <task_id>`"
        try:
            result = manager.set_task_network_access(task_id, True)
        except KeyError:
            return f"Coding task not found: {task_id}"
        return f"{result.get('message')}\n\n{format_coding_task(result.get('task'))}"
    if command == "/coding-status":
        if not rest.strip():
            return "Usage: `/coding-status <task_id>`"
        task_id = rest.strip()
        if channel_type != "cli":
            manager.subscribe_task(task_id, conversation_ref)
        return format_coding_task(manager.get_task(task_id))
    if command == "/coding-log":
        task_id = rest.strip()
        if not task_id:
            return "Usage: `/coding-log <task_id>`"
        if channel_type != "cli":
            manager.subscribe_task(task_id, conversation_ref)
        return format_coding_logs(manager.get_logs(task_id, limit=30))
    if command == "/coding-reply":
        task_id, _, message = rest.partition(" ")
        if not task_id or not message.strip():
            return "Usage: `/coding-reply <task_id> <message>`"
        try:
            result = manager.reply_to_task(task_id, message.strip())
            manager.subscribe_task(task_id, conversation_ref)
        except KeyError:
            return f"Coding task not found: {task_id}"
        return f"{result.get('message')}\n\n{format_coding_task(result.get('task'))}"
    if command == "/coding-cancel":
        task_id = rest.strip()
        if not task_id:
            return "Usage: `/coding-cancel <task_id>`"
        try:
            result = manager.cancel_task(task_id)
        except KeyError:
            return f"Coding task not found: {task_id}"
        return f"Coding task cancelled.\n\n{format_coding_task(result.get('task'))}"
    if command in {"/coding-delete", "/coding-rm"}:
        task_id = rest.strip()
        if not task_id:
            return "Usage: `/coding-delete <task_id>`"
        try:
            result = manager.delete_task(task_id)
        except KeyError:
            return f"Coding task not found: {task_id}"
        if result.get("status") == "active":
            return str(result.get("message"))
        return str(result.get("message"))
    if command in {"/coding-clear", "/coding-clear-history"}:
        result = manager.clear_task_history()
        return (
            f"Deleted {result.get('deleted', 0)} coding task(s). "
            f"Preserved {result.get('preserved', 0)} active task(s)."
        )
    return None


async def handle_channel_command(
    prompt: str,
    channel_type: str,
    conversation_id: str | None = None,
) -> Optional[dict]:
    command = prompt.strip().split(maxsplit=1)[0].lower()
    if "@" in command:
        command = command.split("@", 1)[0]

    if command in {"/start", "/help"}:
        response = help_text()
    elif command == "/status":
        response = status_text()
    elif command == "/wipe":
        await record_event("system", "Memory wipe started", f"Requested from {channel_type}.")
        await wipe_all_memories_for_api()
        await record_event("system", "Memory wipe complete", "Local memories were cleared.")
        response = "Local memories were cleared."
    elif command in {"/quit", "/exit"}:
        response = "This command only works in the local CLI. Remote channels cannot stop the local process."
    else:
        response = handle_coding_command(
            prompt,
            channel_type=channel_type,
            conversation_id=conversation_id,
        )
        if response is None:
            response = handle_tool_command(prompt)
        if response is None:
            return None
        if command in {"/add-tool", "/enable-tool", "/remove-tool", "/disable-tool"}:
            clear_composio_session_cache()

    await record_event(
        "system",
        "Command handled",
        response,
        {"channel": channel_type, "command": command},
    )
    return {"response": response, "metrics": {}, "contexts": {}, "command": command}


async def handle_and_store_channel_command(req: ChatRequest) -> Optional[dict]:
    command_result = await handle_channel_command(
        req.prompt,
        req.channel_type,
        conversation_id=req.conversation_id,
    )
    if command_result:
        await append_chat_history(
            req.channel_type,
            req.prompt,
            command_result["response"],
            conversation_id=req.conversation_id,
            attachment_ids=req.attachment_ids,
        )
    return command_result


async def record_event(
    kind: str,
    title: str,
    message: str = "",
    data: Optional[dict] = None,
):
    global EVENT_SEQ

    async with EVENT_LOCK:
        EVENT_SEQ += 1
        event = {
            "id": EVENT_SEQ,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "title": title,
            "message": message,
            "data": data or {},
        }
        EVENTS.append(event)
        return event


async def record_coding_event(
    kind: str,
    title: str,
    message: str = "",
    data: Optional[dict] = None,
):
    event = await record_event(kind, title, message, data)
    await maybe_send_coding_telegram_update(title, message, data or {})
    try:
        await persist_coding_event_to_chat_history_and_learning(title, message, data or {})
    except Exception as exc:
        logger.debug("Coding event history/learning persistence failed: %s", exc)
    return event


def compact_coding_text(value: object, max_chars: int = CODING_HISTORY_MAX_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def coding_event_refs(task: dict) -> list[str]:
    refs = []
    if task.get("conversation_ref"):
        refs.append(str(task["conversation_ref"]))
    for ref in task.get("notification_refs") or []:
        ref_text = str(ref or "").strip()
        if ref_text and ref_text not in refs:
            refs.append(ref_text)
    return refs or ["cli"]


def coding_provider_label(task: dict, data: dict) -> str:
    provider = str(data.get("provider") or task.get("provider") or "coding agent").strip()
    labels = {"codex": "Codex"}
    return labels.get(provider.lower(), provider.replace("_", " ").title())


def coding_event_history_pair(title: str, message: str, task: dict, data: dict) -> tuple[str, str]:
    task_id = str(data.get("task_id") or task.get("id") or "").strip()
    provider = coding_provider_label(task, data)
    status = str(data.get("status") or task.get("status") or "").strip()
    status_suffix = f" [{status}]" if status else ""
    task_prompt = compact_coding_text(task.get("prompt") or task.get("pending_prompt") or "", 1200)
    body = compact_coding_text(message, CODING_HISTORY_MAX_CHARS)

    if title == "Coding reply":
        user_prompt = f"User reply to {provider} coding task {task_id}:\n{body}"
        agent_response = f"Reply recorded for {provider} coding task {task_id}{status_suffix}."
        return user_prompt, agent_response

    if title == "Coding task queued":
        user_prompt = f"User started {provider} coding task {task_id}:\n{task_prompt or body}"
        agent_response = f"{title}{status_suffix}: {body}"
        return user_prompt, agent_response

    user_prompt = (
        f"{provider} coding task {task_id} update for original request:\n"
        f"{task_prompt or 'No original task prompt was saved.'}"
    )
    if title == "Coding completed":
        agent_response = f"{provider} completed coding task {task_id}{status_suffix}:\n{body}"
    elif title == "Coding failed":
        agent_response = f"{provider} failed coding task {task_id}{status_suffix}:\n{body}"
    elif title == "Coding approval needed":
        agent_response = f"{provider} needs user approval for coding task {task_id}{status_suffix}:\n{body}"
    else:
        agent_response = f"{title}{status_suffix}: {body}"
    return user_prompt, agent_response


async def persist_coding_event_to_chat_history_and_learning(title: str, message: str, data: dict) -> None:
    if title not in CODING_HISTORY_TITLES:
        return

    task_id = str(data.get("task_id") or "").strip()
    if not task_id:
        return

    task = CodingTaskStore(load_coding_settings().task_root).get_task(task_id)
    if not task:
        return

    user_prompt, agent_response = coding_event_history_pair(title, message, task, data)
    if not user_prompt.strip() or not agent_response.strip():
        return

    for ref in coding_event_refs(task):
        parsed = channel_from_conversation_ref(ref)
        if not parsed:
            continue
        channel_type, conversation_id = parsed
        await append_chat_history(
            channel_type,
            user_prompt,
            agent_response,
            conversation_id=conversation_id,
        )

    schedule_coding_event_learning(title, user_prompt, agent_response)


def schedule_coding_event_learning(title: str, user_prompt: str, agent_response: str) -> None:
    if title not in CODING_LEARNING_TITLES:
        return

    payload = "\n".join(
        [
            f"user: {compact_coding_text(user_prompt, CODING_LEARNING_MAX_CHARS)}",
            f"assistant: {compact_coding_text(agent_response, CODING_LEARNING_MAX_CHARS)}",
        ]
    )
    ingestion_prompt = wrap_memory_ingestion_payload(payload)

    async def learn_coding_event() -> None:
        async with CODING_LEARNING_SEMAPHORE:
            await background_memory_update(
                ingestion_prompt,
                MEMORY_INGESTION_ACK,
                "",
                "",
                "",
                None,
            )

    task = asyncio.create_task(learn_coding_event())
    PENDING_LEARNING_TASKS.add(task)

    def cleanup_learning_task(done_task: asyncio.Task) -> None:
        PENDING_LEARNING_TASKS.discard(done_task)
        try:
            done_task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("Coding event learning failed: %s", exc)

    task.add_done_callback(cleanup_learning_task)


async def maybe_send_coding_telegram_update(title: str, message: str, data: dict) -> None:
    if not TELEGRAM_BOT_TOKEN:
        return

    task_id = str(data.get("task_id") or "").strip()
    if not task_id:
        return

    task = CodingTaskStore(load_coding_settings().task_root).get_task(task_id)
    if not task:
        return

    refs = []
    if task.get("conversation_ref"):
        refs.append(task.get("conversation_ref"))
    refs.extend(task.get("notification_refs") or [])
    chat_ids = []
    for ref in refs:
        chat_id = telegram_chat_id_from_conversation_ref(ref)
        if chat_id is not None and chat_id not in chat_ids:
            chat_ids.append(chat_id)
    if not chat_ids:
        return

    should_send = False
    if title == "Codex is working":
        now = time.monotonic()
        should_send = True
    elif title in {
        "Coding task queued",
        "Coding started",
        "Coding session restarted",
        "Coding progress",
        "Coding approval needed",
        "Coding completed",
        "Coding failed",
        "Coding interrupted",
        "Coding cancelled",
    }:
        should_send = True

    if not should_send:
        return

    status = str(data.get("status") or task.get("status") or "").strip()
    header = f"{title}"
    if status:
        header += f" [{status}]"
    body = str(message or "").strip()
    if len(body) > 1200:
        body = body[:1200].rstrip() + "\n..."

    for chat_id in chat_ids:
        if title == "Codex is working":
            now = time.monotonic()
            throttle_key = f"{task_id}:{chat_id}"
            last_sent = CODING_TELEGRAM_LAST_SENT.get(throttle_key, 0)
            if now - last_sent < CODING_TELEGRAM_NOTIFY_SECONDS:
                continue
            CODING_TELEGRAM_LAST_SENT[throttle_key] = now
        try:
            await send_telegram_message(chat_id, f"{header}\n{body}".strip())
        except Exception as exc:
            logger.debug("Coding Telegram status update failed: %s", exc)


def serialize_job(job: dict) -> dict:
    return {
        key: value
        for key, value in job.items()
        if key not in {"last_event_key"}
    }


def job_result_payload(job: dict) -> dict:
    result = job.get("result") or {}
    return {
        "response": result.get("response", ""),
        "metrics": result.get("metrics", {}),
        "contexts": result.get("contexts", {}),
    }


def job_event_title(stage: str, data: Optional[dict] = None) -> str:
    data = data or {}
    tool_status = data.get("tool_status")
    if stage == "retrieval":
        return "Retrieving memory"
    if stage == "tool_routing":
        return "Routing tools"
    if stage == "generation":
        return "Generating response"
    if stage == "tool" and tool_status == "completed":
        return "Tool completed"
    if stage == "tool" and tool_status == "failed":
        return "Tool failed"
    if stage == "tool":
        return "Tool is being used"
    if stage == "finalizing":
        return "Finalizing response"
    return "Job status"


def public_tool_name(tool_name: str | None) -> str | None:
    if not tool_name:
        return None
    if str(tool_name).upper().startswith("COMPOSIO_"):
        return "cloud tools"
    if str(tool_name) == "configure_cloud_tools":
        return "cloud tools"
    if str(tool_name) in {
        "coding_agent",
        "configure_coding_agent",
        "start_coding_task",
        "continue_coding_task",
        "reply_to_coding_task",
        "get_coding_task_status",
        "get_coding_task_logs",
        "cancel_coding_task",
    }:
        return "coding agent"
    return str(tool_name)


def public_skill_names(skills: list | None) -> list:
    public_names = []
    for skill in skills or []:
        if str(skill) == "composio":
            public_names.append("cloud tools")
        elif str(skill) == "coding_agent":
            public_names.append("coding agent")
        else:
            public_names.append(skill)
    return public_names


async def get_recent_chat_history(
    channel_type: str,
    conversation_id: str | None = None,
) -> list[BaseMessage]:
    async with CHAT_HISTORY_LOCK:
        items = get_recent_history_items(
            channel_type,
            conversation_id,
            limit=CHAT_HISTORY_WINDOW_MESSAGES,
        )

    messages: list[BaseMessage] = []
    for item in items:
        content = append_attachment_note(
            str(item.get("content") or ""),
            item.get("attachment_ids") or [],
        )
        if item.get("role") == "ai":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


async def append_chat_history(
    channel_type: str,
    user_prompt: str,
    agent_response: str,
    conversation_id: str | None = None,
    attachment_ids: list[str] | None = None,
) -> None:
    async with CHAT_HISTORY_LOCK:
        append_chat_pair(
            channel_type,
            user_prompt,
            agent_response,
            conversation_id=conversation_id,
            attachment_ids=attachment_ids or [],
        )


async def get_job_snapshot(job_id: str) -> dict:
    async with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return serialize_job(job.copy())


async def update_job_progress(
    job_id: str,
    stage: str,
    message: str = "",
    data: Optional[dict] = None,
) -> None:
    data = data or {}
    raw_tool_name = data.get("tool_name")
    display_tool_name = public_tool_name(raw_tool_name)
    title = job_event_title(stage, data)
    event_key = (
        stage,
        message,
        raw_tool_name,
        data.get("tool_status"),
        tuple(data.get("skills") or []),
    )
    display_message = (
        message.replace(str(raw_tool_name), display_tool_name or str(raw_tool_name))
        if raw_tool_name and display_tool_name
        else message
    )
    event_data = {**data}
    if raw_tool_name:
        event_data["tool_name"] = display_tool_name
    if data.get("skills"):
        event_data["skills"] = public_skill_names(data.get("skills"))

    should_record = True
    async with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None or job.get("status") in {"completed", "failed"}:
            return

        if job.get("last_event_key") == event_key:
            should_record = False

        job["status"] = "running"
        job["stage"] = stage
        job["message"] = display_message
        job["tool_name"] = display_tool_name
        job["updated_at"] = now_iso()
        job["last_event_key"] = event_key

    if should_record:
        await record_event(
            "job",
            title,
            display_message,
            {"job_id": job_id, "stage": stage, **event_data},
        )


async def create_completed_job(req: ChatRequest, result: dict) -> dict:
    job_id = str(uuid.uuid4())
    timestamp = now_iso()
    job = {
        "id": job_id,
        "type": "chat",
        "status": "completed",
        "stage": "completed",
        "message": "Command completed.",
        "tool_name": None,
        "request": req.model_dump(),
        "result": {
            "response": result.get("response", ""),
            "metrics": result.get("metrics", {}),
            "contexts": result.get("contexts", {}),
        },
        "error": None,
        "created_at": timestamp,
        "updated_at": timestamp,
        "started_at": timestamp,
        "completed_at": timestamp,
    }
    done_event = asyncio.Event()
    done_event.set()
    async with JOB_LOCK:
        JOBS[job_id] = job
        JOB_DONE_EVENTS[job_id] = done_event
    return serialize_job(job)


async def enqueue_chat_job(req: ChatRequest) -> dict:
    job_id = str(uuid.uuid4())
    timestamp = now_iso()
    job = {
        "id": job_id,
        "type": "chat",
        "status": "queued",
        "stage": "queued",
        "message": "Waiting for the agent worker.",
        "tool_name": None,
        "request": req.model_dump(),
        "result": None,
        "error": None,
        "created_at": timestamp,
        "updated_at": timestamp,
        "started_at": None,
        "completed_at": None,
    }
    async with JOB_LOCK:
        JOBS[job_id] = job
        JOB_DONE_EVENTS[job_id] = asyncio.Event()

    await JOB_QUEUE.put(job_id)
    await record_event(
        "request",
        "Chat request",
        req.prompt,
        {
            "job_id": job_id,
            "channel": req.channel_type,
            "conversation_id": req.conversation_id,
            "skip_learning": req.skip_learning,
            "attachment_count": len(req.attachment_ids),
        },
    )
    return serialize_job(job)


async def complete_job(job_id: str, result: dict) -> None:
    async with JOB_LOCK:
        job = JOBS[job_id]
        job["status"] = "completed"
        job["stage"] = "completed"
        job["message"] = "Response ready."
        job["tool_name"] = None
        job["result"] = result
        job["updated_at"] = now_iso()
        job["completed_at"] = job["updated_at"]
        done_event = JOB_DONE_EVENTS.get(job_id)
        if done_event:
            done_event.set()


async def fail_job(job_id: str, error: str) -> None:
    async with JOB_LOCK:
        job = JOBS[job_id]
        job["status"] = "failed"
        job["stage"] = "failed"
        job["message"] = "The job failed."
        job["tool_name"] = None
        job["error"] = error
        job["updated_at"] = now_iso()
        job["completed_at"] = job["updated_at"]
        done_event = JOB_DONE_EVENTS.get(job_id)
        if done_event:
            done_event.set()


async def wait_for_job(job_id: str, timeout: Optional[float] = None) -> dict:
    async with JOB_LOCK:
        done_event = JOB_DONE_EVENTS.get(job_id)
    if done_event is None:
        raise KeyError(job_id)

    await asyncio.wait_for(done_event.wait(), timeout=timeout)
    return await get_job_snapshot(job_id)


async def run_chat_job(job_id: str) -> None:
    async with JOB_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["stage"] = "starting"
        job["message"] = "Starting agent request."
        job["started_at"] = now_iso()
        job["updated_at"] = job["started_at"]
        request_data = dict(job["request"])

    req = ChatRequest(**request_data)
    started = time.perf_counter()

    async def status_callback(
        stage: str,
        message: str = "",
        data: Optional[dict] = None,
    ) -> None:
        await update_job_progress(job_id, stage, message, data)

    try:
        await update_job_progress(job_id, "retrieval", "Searching memory.")
        chat_history = await get_recent_chat_history(req.channel_type, req.conversation_id)
        context_attachment_ids = []
        for attachment_id in recent_attachment_ids(req.channel_type, req.conversation_id):
            if attachment_id not in context_attachment_ids:
                context_attachment_ids.append(attachment_id)
        for attachment_id in req.attachment_ids:
            if attachment_id not in context_attachment_ids:
                context_attachment_ids.append(attachment_id)
        await refresh_attachments_for_context(context_attachment_ids)
        attachment_context = render_attachment_context(context_attachment_ids)
        response, metrics, contexts = await get_argus_response(
            user_prompt=req.prompt,
            user_id=AGENT_USER_ID,
            channel_type=req.channel_type,
            chat_history=chat_history,
            skip_learning=req.skip_learning,
            current_date=req.current_date,
            status_callback=status_callback,
            attachments_context=attachment_context,
        )
        await append_chat_history(
            req.channel_type,
            req.prompt,
            response,
            conversation_id=req.conversation_id,
            attachment_ids=req.attachment_ids,
        )
        elapsed = time.perf_counter() - started
        result = {"response": response, "metrics": metrics, "contexts": contexts}
        await complete_job(job_id, result)
        await record_event(
            "response",
            "Telegram response" if req.channel_type == "telegram" else "Chat response",
            response or "",
            {
                "job_id": job_id,
                "channel": req.channel_type,
                "conversation_id": req.conversation_id,
                "elapsed": round(elapsed, 2),
                "metrics": metrics,
                "contexts": context_line_counts(contexts),
                "attachment_count": len(req.attachment_ids),
            },
        )
    except Exception as e:
        logger.error("Agent job error for %s: %s", AGENT_USER_ID, e, exc_info=True)
        await fail_job(job_id, str(e))
        await record_event("error", "Chat error", str(e), {"job_id": job_id, "channel": req.channel_type})


async def job_worker() -> None:
    while True:
        job_id = await JOB_QUEUE.get()
        try:
            await run_chat_job(job_id)
        finally:
            JOB_QUEUE.task_done()


def context_line_counts(contexts: dict) -> dict:
    return {
        key: len(str(value or "").splitlines())
        for key, value in (contexts or {}).items()
    }


def parse_allowed_telegram_users() -> set[int] | None:
    raw_value = os.getenv("TELEGRAM_ALLOWED_USERS")
    if not raw_value:
        return None

    allowed = set()
    for item in raw_value.split(","):
        clean_item = item.strip()
        if clean_item:
            try:
                allowed.add(int(clean_item))
            except ValueError:
                logger.warning("Ignoring invalid TELEGRAM_ALLOWED_USERS value: %s", clean_item)
    return allowed


TELEGRAM_ALLOWED_USERS = parse_allowed_telegram_users()


@app.on_event("startup")
async def start_background_job_worker():
    global JOB_WORKER_TASK
    manager = get_coding_manager()
    manager.set_event_callback(record_coding_event)
    manager.reconcile_interrupted_tasks()
    if JOB_WORKER_TASK is None or JOB_WORKER_TASK.done():
        JOB_WORKER_TASK = asyncio.create_task(job_worker())


@app.on_event("shutdown")
async def stop_background_job_worker():
    if JOB_WORKER_TASK is not None:
        JOB_WORKER_TASK.cancel()
        try:
            await JOB_WORKER_TASK
        except asyncio.CancelledError:
            pass
    await close_all_codex_sessions()


def split_telegram_message(text: str) -> list[str]:
    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        chunk = remaining[:TELEGRAM_MESSAGE_LIMIT]
        split_at = chunk.rfind("\n")
        if split_at < TELEGRAM_MESSAGE_LIMIT // 2:
            split_at = chunk.rfind(" ")
        if split_at < TELEGRAM_MESSAGE_LIMIT // 2:
            split_at = TELEGRAM_MESSAGE_LIMIT
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    return [chunk for chunk in chunks if chunk]


async def telegram_api_call(method: str, payload: dict):
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    url = f"{TELEGRAM_API_BASE}/bot{TELEGRAM_BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error: {data}")
        return data


async def send_telegram_message(chat_id: int, text: str):
    for chunk in split_telegram_message(text or "No response was returned."):
        await telegram_api_call("sendMessage", {"chat_id": chat_id, "text": chunk})


async def send_telegram_chat_action(chat_id: int, action: str = "typing"):
    await telegram_api_call("sendChatAction", {"chat_id": chat_id, "action": action})


async def keep_telegram_typing(chat_id: int):
    while True:
        try:
            await send_telegram_chat_action(chat_id)
        except Exception as e:
            logger.debug("Telegram typing action failed: %s", e)
        await asyncio.sleep(TELEGRAM_TYPING_INTERVAL_SECONDS)


def telegram_message_from_update(update: dict) -> tuple[dict, str]:
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        if update.get(key):
            return update[key] or {}, key
    return {}, "unknown"


def format_file_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "unknown size"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} bytes"


def telegram_download_limit_message(size_bytes: int | None = None) -> str:
    limit = format_file_size(CHANNEL_FILE_MAX_BYTES)
    if size_bytes:
        return (
            f"That file is {format_file_size(size_bytes)}, which is above this agent's "
            f"Telegram download limit of {limit}. Please send a smaller file or share "
            "the important text directly."
        )
    return (
        f"That file is above this agent's Telegram download limit of {limit}. "
        "Please send a smaller file or share the important text directly."
    )


def is_telegram_size_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "file is too big",
            "file is too large",
            "request entity too large",
            "above channel_file_max_bytes",
            "telegram download limit",
        )
    )


def public_attachment_error(error: Exception | str) -> str:
    text = str(error)
    if "telegram download limit" in text.lower():
        return text
    if isinstance(error, Exception) and is_telegram_size_limit_error(error):
        return telegram_download_limit_message()
    return "I could not download or read it locally. Please try again with a smaller/common file."


def format_attachment_failure_message(failures: list[dict]) -> str:
    if not failures:
        return ""

    lines = ["I could not process these attachment(s):"]
    for failure in failures[:5]:
        label = failure.get("filename") or failure.get("kind") or "attachment"
        lines.append(f"- {label}: {failure.get('message')}")
    if len(failures) > 5:
        lines.append(f"- {len(failures) - 5} more attachment(s) also failed.")
    return "\n".join(lines)


def telegram_file_references(message: dict) -> list[dict]:
    refs = []

    if message.get("photo"):
        photos = message.get("photo") or []
        best_photo = max(
            photos,
            key=lambda item: item.get("file_size") or (item.get("width", 0) * item.get("height", 0)),
        )
        refs.append(
            {
                "kind": "photo",
                "file_id": best_photo.get("file_id"),
                "filename": f"telegram_photo_{best_photo.get('file_unique_id') or best_photo.get('file_id')}.jpg",
                "mime_type": "image/jpeg",
                "metadata": {
                    "width": best_photo.get("width"),
                    "height": best_photo.get("height"),
                    "file_size": best_photo.get("file_size"),
                },
            }
        )

    field_map = {
        "document": "document",
        "audio": "audio",
        "voice": "voice",
        "video": "video",
        "video_note": "video_note",
        "animation": "animation",
        "sticker": "sticker",
    }
    for field, kind in field_map.items():
        value = message.get(field)
        if not value:
            continue
        refs.append(
            {
                "kind": kind,
                "file_id": value.get("file_id"),
                "filename": value.get("file_name") or f"telegram_{kind}_{value.get('file_unique_id') or value.get('file_id')}",
                "mime_type": value.get("mime_type"),
                "metadata": {
                    key: value.get(key)
                    for key in (
                        "file_size",
                        "duration",
                        "width",
                        "height",
                        "emoji",
                        "set_name",
                    )
                    if value.get(key) is not None
                },
            }
        )

    return [ref for ref in refs if ref.get("file_id")]


async def download_telegram_file(file_id: str) -> tuple[bytes, dict]:
    try:
        file_info = await telegram_api_call("getFile", {"file_id": file_id})
    except Exception as exc:
        if is_telegram_size_limit_error(exc):
            raise RuntimeError(telegram_download_limit_message()) from exc
        raise

    result = file_info.get("result") or {}
    file_size = int(result.get("file_size") or 0)
    if file_size and file_size > CHANNEL_FILE_MAX_BYTES:
        raise RuntimeError(telegram_download_limit_message(file_size))

    file_path = result.get("file_path")
    if not file_path:
        raise RuntimeError("Telegram did not return a downloadable file path.")

    url = f"{TELEGRAM_API_BASE}/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content

    if len(content) > CHANNEL_FILE_MAX_BYTES:
        raise RuntimeError(telegram_download_limit_message(len(content)))
    return content, result


async def store_telegram_attachments(
    refs: list[dict],
    chat_id: int,
    telegram_user_id: int | None,
    username: str,
    update_id: int | None,
) -> tuple[list[dict], list[dict]]:
    records = []
    failures = []
    for ref in refs:
        try:
            content, file_info = await download_telegram_file(ref["file_id"])
            metadata = {
                "telegram_user_id": telegram_user_id,
                "username": username,
                "chat_id": chat_id,
                "update_id": update_id,
                "telegram_file": file_info,
                **(ref.get("metadata") or {}),
            }
            record = await store_attachment_from_bytes(
                content,
                filename=ref.get("filename"),
                mime_type=ref.get("mime_type"),
                channel_type="telegram",
                conversation_id=str(chat_id),
                source_kind=ref.get("kind") or "telegram_file",
                source_metadata=metadata,
            )
            records.append(record)
            await record_event(
                "attachment",
                "Attachment stored",
                f"{record.get('original_filename') or record.get('stored_filename')} saved.",
                {
                    "channel": "telegram",
                    "conversation_id": str(chat_id),
                    "attachment_id": record.get("id"),
                    "mime_type": record.get("mime_type"),
                    "size_bytes": record.get("size_bytes"),
                },
            )
        except Exception as exc:
            await record_event(
                "error",
                "Attachment storage failed",
                str(exc),
                {"channel": "telegram", "file_id": ref.get("file_id"), "kind": ref.get("kind")},
            )
            failures.append(
                {
                    "kind": ref.get("kind"),
                    "filename": ref.get("filename"),
                    "message": public_attachment_error(exc),
                }
            )
    return records, failures


def prompt_for_telegram_message(text: str, attachment_records: list[dict]) -> str:
    if text:
        return text
    labels = [
        str(record.get("original_filename") or record.get("stored_filename") or record.get("id"))
        for record in attachment_records
    ]
    if labels:
        return "I sent these attachment(s). Please read them and respond: " + ", ".join(labels)
    return ""


async def process_telegram_update(update: dict):
    message, update_kind = telegram_message_from_update(update)
    text = str(message.get("text") or message.get("caption") or "").strip()
    file_refs = telegram_file_references(message)
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_id = chat.get("id")
    telegram_user_id = sender.get("id")
    username = sender.get("username") or sender.get("first_name") or "unknown"

    if (not text and not file_refs) or chat_id is None:
        await record_event(
            "telegram",
            "Telegram update ignored",
            "No supported message content was present in the update.",
            {"update_id": update.get("update_id")},
        )
        return

    await record_event(
        "telegram",
        "Telegram edit inbound" if update_kind.startswith("edited_") else "Telegram inbound",
        text or f"{len(file_refs)} attachment(s)",
        {
            "update_kind": update_kind,
            "chat_id": chat_id,
            "telegram_user_id": telegram_user_id,
            "username": username,
            "attachment_count": len(file_refs),
        },
    )

    if (
        TELEGRAM_ALLOWED_USERS is not None
        and int(telegram_user_id or 0) not in TELEGRAM_ALLOWED_USERS
    ):
        await record_event(
            "warning",
            "Telegram blocked",
            f"User {username} is not in TELEGRAM_ALLOWED_USERS.",
            {"telegram_user_id": telegram_user_id, "username": username},
        )
        try:
            await send_telegram_message(
                chat_id,
                "This local agent is not configured for your Telegram account.",
            )
        except Exception as e:
            logger.error("Telegram blocked-user reply failed: %s", e, exc_info=True)
        return

    try:
        attachment_records, attachment_failures = await store_telegram_attachments(
            file_refs,
            chat_id=chat_id,
            telegram_user_id=telegram_user_id,
            username=username,
            update_id=update.get("update_id"),
        )
        if attachment_failures and not attachment_records:
            failure_message = format_attachment_failure_message(attachment_failures)
            await send_telegram_message(
                chat_id,
                failure_message,
            )
            await append_chat_history(
                "telegram",
                text or "I sent attachment(s), but they could not be processed.",
                failure_message,
                conversation_id=str(chat_id),
            )
            return
        if attachment_failures:
            await send_telegram_message(chat_id, format_attachment_failure_message(attachment_failures))

        command_result = None
        if text and not attachment_records:
            command_req = ChatRequest(
                prompt=text,
                channel_type="telegram",
                conversation_id=str(chat_id),
                skip_learning=False,
            )
            command_result = await handle_and_store_channel_command(command_req)
        if command_result:
            await send_telegram_message(chat_id, command_result["response"])
            return

        prompt = prompt_for_telegram_message(text, attachment_records)
        attachment_ids = [record["id"] for record in attachment_records]
        job = await enqueue_chat_job(
            ChatRequest(
                prompt=prompt,
                channel_type="telegram",
                conversation_id=str(chat_id),
                attachment_ids=attachment_ids,
                skip_learning=False,
            )
        )
        typing_task = asyncio.create_task(keep_telegram_typing(chat_id))
        try:
            completed_job = await wait_for_job(job["id"])
        finally:
            typing_task.cancel()
            await asyncio.gather(typing_task, return_exceptions=True)
        if completed_job["status"] == "failed":
            raise RuntimeError(completed_job.get("error") or "Telegram job failed")
        response = job_result_payload(completed_job)["response"]
        await send_telegram_message(chat_id, response)
    except Exception as e:
        logger.error("Telegram processing error: %s", e, exc_info=True)
        await record_event("error", "Telegram processing error", str(e))
        try:
            await send_telegram_message(
                chat_id,
                "The local agent hit an error while processing that message.",
            )
        except Exception as send_error:
            logger.error("Telegram error reply failed: %s", send_error, exc_info=True)


@app.get("/")
async def health():
    return status_payload()


@app.get("/api/events")
async def get_events(after: int = 0):
    async with EVENT_LOCK:
        events = [event for event in EVENTS if event["id"] > after]
    return {"events": events}


@app.post("/api/jobs", status_code=202)
async def create_job(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    command_result = await handle_and_store_channel_command(req)
    if command_result:
        job = await create_completed_job(req, command_result)
        return {"job": job}

    job = await enqueue_chat_job(req)
    return {"job": job}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        job = await get_job_snapshot(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found") from None
    return {"job": job}


@app.get("/api/coding-agents")
async def list_coding_agents():
    return {"coding_agent": get_coding_manager().get_config()}


@app.post("/api/coding-agents/default")
async def select_coding_agent(req: CodingAgentSelectRequest):
    try:
        return {"coding_agent": get_coding_manager().set_default_provider(req.provider)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.post("/api/coding-agents/workspace")
async def set_coding_workspace(req: CodingWorkspaceRequest):
    if not req.workspace_path or not req.workspace_path.strip():
        raise HTTPException(status_code=400, detail="workspace_path is required")
    return {"coding_agent": get_coding_manager().set_workspace_root(req.workspace_path)}


@app.post("/api/coding-agents/network")
async def set_coding_network(req: CodingNetworkRequest):
    return {"coding_agent": get_coding_manager().set_default_network_access(req.enabled)}


@app.get("/api/coding-tasks")
async def list_coding_tasks(limit: int = 50, status: str = "all"):
    return {"tasks": get_coding_manager().list_tasks(limit=limit, status_filter=status)}


@app.delete("/api/coding-tasks")
async def clear_coding_task_history():
    return get_coding_manager().clear_task_history()


@app.post("/api/coding-tasks/subscribe-channel")
async def subscribe_coding_tasks_to_channel(req: CodingChannelSubscribeRequest):
    return await subscribe_active_coding_tasks_to_channel(req.channel_type)


@app.post("/api/coding-tasks", status_code=202)
async def create_coding_task(req: CodingTaskStartRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    result = get_coding_manager().start_task(
        req.prompt,
        provider=req.provider,
        workspace_path=req.workspace_path,
        conversation_ref=req.conversation_id,
    )
    if result.get("status") in {"disabled", "unsupported"}:
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.post("/api/coding-tasks/latest/reply")
async def reply_to_latest_coding_task(req: CodingTaskReplyRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    result = get_coding_manager().reply_to_latest(
        req.message,
        conversation_ref=req.conversation_id,
    )
    if result.get("status") == "missing":
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@app.get("/api/coding-tasks/{task_id}")
async def get_coding_task(task_id: str):
    task = get_coding_manager().get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="coding task not found")
    return {"task": task}


@app.delete("/api/coding-tasks/{task_id}")
async def delete_coding_task(task_id: str):
    try:
        result = get_coding_manager().delete_task(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="coding task not found") from None
    if result.get("status") == "active":
        raise HTTPException(status_code=409, detail=result.get("message"))
    return result


@app.get("/api/coding-tasks/{task_id}/logs")
async def get_coding_task_logs(task_id: str, limit: int = 50):
    if not get_coding_manager().get_task(task_id):
        raise HTTPException(status_code=404, detail="coding task not found")
    return {"logs": get_coding_manager().get_logs(task_id, limit=limit)}


@app.post("/api/coding-tasks/{task_id}/reply")
async def reply_to_coding_task(task_id: str, req: CodingTaskReplyRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    try:
        result = get_coding_manager().reply_to_task(task_id, req.message)
    except KeyError:
        raise HTTPException(status_code=404, detail="coding task not found") from None
    return result


@app.post("/api/coding-tasks/{task_id}/cancel")
async def cancel_coding_task(task_id: str):
    try:
        result = get_coding_manager().cancel_task(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="coding task not found") from None
    return result


@app.post("/api/coding-tasks/{task_id}/network")
async def set_coding_task_network(task_id: str, req: CodingNetworkRequest):
    try:
        return get_coding_manager().set_task_network_access(task_id, req.enabled)
    except KeyError:
        raise HTTPException(status_code=404, detail="coding task not found") from None


@app.post("/api/files", status_code=201)
async def ingest_file(req: FileIngestRequest):
    try:
        content = decode_base64_payload(req.data_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="data_base64 must be valid base64") from None

    if len(content) > CHANNEL_FILE_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"file is {format_file_size(len(content))}, above the configured "
                f"channel limit of {format_file_size(CHANNEL_FILE_MAX_BYTES)}"
            ),
        )

    try:
        record = await store_attachment_from_bytes(
            content,
            filename=req.filename,
            mime_type=req.mime_type,
            channel_type=req.channel_type,
            conversation_id=req.conversation_id,
            source_kind=req.source_kind,
            source_metadata=req.source_metadata,
        )
        await record_event(
            "attachment",
            "Attachment stored",
            f"{record.get('original_filename') or record.get('stored_filename')} saved.",
            {
                "channel": req.channel_type,
                "conversation_id": req.conversation_id,
                "attachment_id": record.get("id"),
                "mime_type": record.get("mime_type"),
                "size_bytes": record.get("size_bytes"),
            },
        )
        return {"attachment": record}
    except Exception as e:
        logger.error("File ingest error: %s", e, exc_info=True)
        await record_event("error", "Attachment storage failed", str(e), {"channel": req.channel_type})
        raise HTTPException(status_code=500, detail="file ingest failed") from e


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        command_result = await handle_and_store_channel_command(req)
        if command_result:
            return command_result

        job = await enqueue_chat_job(req)
        completed_job = await wait_for_job(job["id"])
        if completed_job["status"] == "failed":
            raise RuntimeError(completed_job.get("error") or "agent job failed")
        return job_result_payload(completed_job)
    except Exception as e:
        logger.error(f"Agent error for {AGENT_USER_ID}: {e}", exc_info=True)
        await record_event("error", "Chat error", str(e), {"channel": req.channel_type})
        raise HTTPException(status_code=500, detail="agent processing failed")


@app.post("/api/memories/wipe")
async def wipe_memories():
    try:
        await record_event("system", "Memory wipe started", "Requested from API.")
        await wipe_all_memories_for_api()
        await record_event("system", "Memory wipe complete", "Local memories were cleared.")
        return {"status": "ok", "user_id": AGENT_USER_ID}
    except Exception as e:
        logger.error(f"Memory wipe error for {AGENT_USER_ID}: {e}", exc_info=True)
        await record_event("error", "Memory wipe error", str(e))
        raise HTTPException(status_code=500, detail="memory wipe failed")


@app.post(TELEGRAM_WEBHOOK_PATH)
async def telegram_webhook(
    update: dict,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=503, detail="telegram is not configured")

    if (
        TELEGRAM_WEBHOOK_SECRET
        and x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET
    ):
        raise HTTPException(status_code=403, detail="invalid telegram webhook secret")

    await record_event(
        "telegram",
        "Telegram webhook accepted",
        f"update_id={update.get('update_id')}",
    )
    background_tasks.add_task(process_telegram_update, update)
    return {"status": "accepted"}
