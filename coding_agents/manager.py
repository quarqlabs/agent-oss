"""High-level coding-agent task manager."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable

from coding_agents.codex_runner import CodexRunner, runtime_check, schedule_close_codex_session
from coding_agents.config import (
    ACTIVE_TASK_STATUSES,
    coding_config_summary,
    load_settings,
    provider_info,
    set_default_provider,
    set_network_access,
    set_workspace_root,
)
from coding_agents.task_store import CodingTaskStore


EventCallback = Callable[[str, str, str, dict[str, Any]], Awaitable[Any] | Any]


INTERRUPTED_TASK_STATUSES = {"queued", "running"}
INTERRUPTED_TASK_MESSAGE = (
    "This coding task was interrupted because Quarq restarted or the coding "
    "runner is no longer active. Continue it with `/coding-continue <task_id> "
    "<message>` or start fresh with `/coding-new <task>`."
)


class CodingAgentManager:
    def __init__(self):
        self._event_callback: EventCallback | None = None
        self._active_tasks: dict[str, asyncio.Task] = {}

    def set_event_callback(self, callback: EventCallback | None) -> None:
        self._event_callback = callback

    def start_task(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        workspace_path: str | None = None,
        conversation_ref: str | None = None,
    ) -> dict[str, Any]:
        settings = load_settings()
        provider = (provider or settings.default_provider).strip().lower()
        store = CodingTaskStore(settings.task_root)
        provider_meta = provider_info(provider) or {}

        if not settings.enabled:
            return {
                "status": "disabled",
                "message": "Coding agents are disabled. Set CODING_AGENTS_ENABLED=true and restart Quarq.",
            }
        if provider != "codex":
            return {
                "status": "unsupported",
                "message": (
                    f"Coding provider `{provider_meta.get('label', provider)}` is "
                    "not available yet. Codex is the V1 provider."
                ),
            }

        workspace = self._resolve_workspace(workspace_path, settings.workspace_root)
        task = store.create_task(
            provider=provider,
            workspace_path=str(workspace),
            prompt=str(prompt or "").strip(),
            conversation_ref=conversation_ref,
            network_access=settings.network_access,
        )
        self._schedule_emit(
            "coding",
            "Coding task queued",
            f"Task {task['id']} queued for Codex.",
            {"task_id": task["id"], "status": "queued", "provider": provider},
        )

        ok, reason = runtime_check(settings)
        if not ok:
            store.update_task(task["id"], status="failed", error=reason)
            store.append_event(task["id"], "coding", "Coding setup needed", reason, {"task_id": task["id"], "status": "failed"})
            self._schedule_emit(
                "coding",
                "Coding setup needed",
                reason,
                {"task_id": task["id"], "status": "failed", "provider": provider},
            )
            task = store.get_task(task["id"]) or task
            return {"status": "failed", "task": task, "message": reason}

        self._schedule_runner(task["id"], settings, store)
        return {"status": "queued", "task": task, "message": f"Coding task queued: {task['id']}"}

    def reply_to_task(self, task_id: str, message: str) -> dict[str, Any]:
        settings = load_settings()
        store = CodingTaskStore(settings.task_root)
        task = store.get_task(task_id)
        if not task:
            raise KeyError(task_id)

        message = str(message or "").strip()
        if not message:
            return {"status": task["status"], "task": task, "message": "Reply was empty."}
        if task.get("status") in {"queued", "running"} and task_id in self._active_tasks:
            return {
                "status": task["status"],
                "task": task,
                "message": "This coding task is still running. Wait for it to finish, or cancel it before continuing.",
            }
        if task.get("status") == "cancelled":
            return {
                "status": "cancelled",
                "task": task,
                "message": "That coding task was cancelled. Start a fresh session with `/coding-new <task>`.",
            }

        updated_prompt = (
            f"{task.get('prompt') or ''}\n\n"
            f"User reply for this coding task:\n{message}"
        ).strip()
        task = store.update_task(
            task_id,
            prompt=updated_prompt,
            pending_prompt=message,
            last_user_reply=message,
            approval_request=None,
            error=None,
            status="queued",
        )
        store.append_event(task_id, "coding", "User reply", message, {"task_id": task_id, "status": task["status"]})
        self._schedule_emit("coding", "Coding reply", message, {"task_id": task_id, "status": task["status"]})

        if task_id not in self._active_tasks:
            self._schedule_runner(task_id, settings, store)

        return {"status": task["status"], "task": task, "message": "Reply recorded."}

    def latest_task(self, conversation_ref: str | None = None) -> dict[str, Any] | None:
        settings = load_settings()
        tasks = CodingTaskStore(settings.task_root).list_tasks(limit=None)
        if conversation_ref:
            scoped = [task for task in tasks if task.get("conversation_ref") == conversation_ref]
            if scoped:
                tasks = scoped
        for task in tasks:
            if task.get("status") != "cancelled":
                return task
        return None

    def reply_to_latest(self, message: str, conversation_ref: str | None = None) -> dict[str, Any]:
        task = self.latest_task(conversation_ref=conversation_ref)
        if not task:
            return {
                "status": "missing",
                "task": None,
                "message": "No coding task found yet. Start a new one with `/coding <task>`.",
            }
        return self.reply_to_task(str(task["id"]), message)

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        settings = load_settings()
        store = CodingTaskStore(settings.task_root)
        task = store.get_task(task_id)
        if not task:
            raise KeyError(task_id)

        active = self._active_tasks.pop(task_id, None)
        if active and not active.done():
            active.cancel()

        task = store.update_task(task_id, status="cancelled")
        store.append_event(task_id, "coding", "Coding cancelled", "Task was cancelled by the user.", {"task_id": task_id, "status": "cancelled"})
        self._schedule_emit("coding", "Coding cancelled", f"Task {task_id} was cancelled.", {"task_id": task_id, "status": "cancelled"})
        schedule_close_codex_session(task_id)
        return {"status": "cancelled", "task": task}

    def delete_task(self, task_id: str) -> dict[str, Any]:
        settings = load_settings()
        store = CodingTaskStore(settings.task_root)
        task = store.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        if task_id in self._active_tasks or task.get("status") in ACTIVE_TASK_STATUSES:
            return {
                "status": "active",
                "task": task,
                "message": "That coding task is still active. Cancel it first with `/coding-cancel <task_id>`.",
            }

        deleted = store.delete_task(task_id)
        schedule_close_codex_session(task_id)
        self._schedule_emit(
            "coding",
            "Coding task deleted",
            f"Deleted coding task {task_id}.",
            {"task_id": task_id, "status": "deleted"},
        )
        return {
            "status": "deleted",
            "task": deleted,
            "message": f"Deleted coding task {task_id}.",
        }

    def clear_task_history(self) -> dict[str, Any]:
        settings = load_settings()
        store = CodingTaskStore(settings.task_root)
        tasks = store.list_tasks(limit=None)
        keep_ids = {
            str(task["id"])
            for task in tasks
            if task.get("status") in ACTIVE_TASK_STATUSES or str(task.get("id")) in self._active_tasks
        }
        result = store.clear_tasks(keep_ids=keep_ids)
        for task_id in result.get("deleted_ids") or []:
            schedule_close_codex_session(str(task_id))
        self._schedule_emit(
            "coding",
            "Coding history cleared",
            (
                f"Deleted {result['deleted']} coding task(s). "
                f"Preserved {result['preserved']} active task(s)."
            ),
            {"status": "cleared", **result},
        )
        return {"status": "cleared", **result}

    def list_tasks(
        self,
        limit: int | None = None,
        status_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        settings = load_settings()
        store = CodingTaskStore(settings.task_root)
        self.reconcile_interrupted_tasks(store=store)
        tasks = store.list_tasks(limit=None)
        status = str(status_filter or "").strip().lower()
        if status in {"active", "ongoing", "open"}:
            tasks = [task for task in tasks if task.get("status") in ACTIVE_TASK_STATUSES]
        elif status and status not in {"all", "any"}:
            tasks = [task for task in tasks if str(task.get("status") or "").lower() == status]
        return tasks[:limit] if limit else tasks

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        settings = load_settings()
        store = CodingTaskStore(settings.task_root)
        self.reconcile_interrupted_tasks(store=store)
        return store.get_task(task_id)

    def get_logs(self, task_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        settings = load_settings()
        return CodingTaskStore(settings.task_root).read_logs(task_id, limit=limit)

    def subscribe_task(self, task_id: str, conversation_ref: str | None) -> dict[str, Any] | None:
        conversation_ref = str(conversation_ref or "").strip()
        if not conversation_ref:
            return None
        settings = load_settings()
        store = CodingTaskStore(settings.task_root)
        task = store.get_task(task_id)
        if not task:
            return None
        refs = list(task.get("notification_refs") or [])
        if task.get("conversation_ref") and task["conversation_ref"] not in refs:
            refs.append(task["conversation_ref"])
        if conversation_ref not in refs:
            refs.append(conversation_ref)
            task = store.update_task(task_id, notification_refs=refs)
            store.append_event(
                task_id,
                "coding",
                "Coding notifications subscribed",
                f"Subscribed {conversation_ref} to coding task updates.",
                {"task_id": task_id, "conversation_ref": conversation_ref},
            )
        return task

    def get_config(self) -> dict[str, Any]:
        config = coding_config_summary()
        config["active_tasks"] = len(self.list_tasks(status_filter="active"))
        return config

    def reconcile_interrupted_tasks(self, store: CodingTaskStore | None = None) -> list[dict[str, Any]]:
        settings = load_settings()
        store = store or CodingTaskStore(settings.task_root)
        interrupted = []
        for task in store.list_tasks(limit=None):
            task_id = str(task.get("id") or "")
            if (
                task_id
                and task_id not in self._active_tasks
                and task.get("status") in INTERRUPTED_TASK_STATUSES
            ):
                updated = store.update_task(
                    task_id,
                    status="failed",
                    error=INTERRUPTED_TASK_MESSAGE,
                    current_activity=None,
                    current_activity_at=None,
                )
                store.append_event(
                    task_id,
                    "coding",
                    "Coding interrupted",
                    INTERRUPTED_TASK_MESSAGE,
                    {"task_id": task_id, "status": "failed"},
                )
                interrupted.append(updated)
                self._schedule_emit(
                    "coding",
                    "Coding interrupted",
                    INTERRUPTED_TASK_MESSAGE,
                    {"task_id": task_id, "status": "failed"},
                )
        return interrupted

    def set_default_provider(self, provider: str) -> dict[str, Any]:
        set_default_provider(provider)
        config = self.get_config()
        self._schedule_emit(
            "coding",
            "Coding agent selected",
            f"Default coding agent: {config['default_provider_label']}",
            {"provider": config["default_provider"], "status": "configured"},
        )
        return config

    def set_workspace_root(self, workspace_path: str) -> dict[str, Any]:
        set_workspace_root(workspace_path)
        config = self.get_config()
        self._schedule_emit(
            "coding",
            "Coding workspace updated",
            str(config["workspace_root"]),
            {"workspace_path": config["workspace_root"], "status": "configured"},
        )
        return config

    def set_default_network_access(self, enabled: bool) -> dict[str, Any]:
        set_network_access(enabled)
        config = self.get_config()
        state = "enabled" if config.get("network_access") else "disabled"
        self._schedule_emit(
            "coding",
            "Coding network updated",
            f"Default coding network access: {state}",
            {"network_access": bool(config.get("network_access")), "status": "configured"},
        )
        return config

    def set_task_network_access(self, task_id: str, enabled: bool) -> dict[str, Any]:
        settings = load_settings()
        store = CodingTaskStore(settings.task_root)
        task = store.get_task(task_id)
        if not task:
            raise KeyError(task_id)

        restart_provider_session = bool(task.get("provider_thread_id"))
        task = store.update_task(
            task_id,
            network_access=bool(enabled),
            restart_provider_session=restart_provider_session,
        )
        if restart_provider_session and task.get("status") not in ACTIVE_TASK_STATUSES:
            schedule_close_codex_session(task_id)
        state = "enabled" if task.get("network_access") else "disabled"
        message = (
            f"Network access {state} for task {task_id}. "
            "Codex will restart from saved task context on the next continuation so this applies."
            if restart_provider_session
            else f"Network access {state} for task {task_id}."
        )
        store.append_event(
            task_id,
            "coding",
            "Coding network updated",
            message,
            {"task_id": task_id, "network_access": bool(task.get("network_access"))},
        )
        self._schedule_emit(
            "coding",
            "Coding network updated",
            message,
            {"task_id": task_id, "network_access": bool(task.get("network_access"))},
        )
        return {"status": "configured", "task": task, "message": message}

    def _schedule_runner(self, task_id: str, settings: Any, store: CodingTaskStore) -> None:
        async def run_and_cleanup() -> None:
            try:
                runner = CodexRunner(settings, store, self._emit)
                await runner.run(task_id)
            finally:
                self._active_tasks.pop(task_id, None)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            store.update_task(task_id, status="failed", error="No active event loop was available to start Codex.")
            return

        task = loop.create_task(run_and_cleanup())
        self._active_tasks[task_id] = task

    def _schedule_emit(self, kind: str, title: str, message: str, data: dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._emit(kind, title, message, data))

    async def _emit(self, kind: str, title: str, message: str, data: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        result = self._event_callback(kind, title, message, data)
        if inspect.isawaitable(result):
            await result

    def _resolve_workspace(self, workspace_path: str | None, default_root: Path) -> Path:
        path = Path(workspace_path).expanduser() if workspace_path else default_root
        if not path.is_absolute():
            path = default_root / path
        return path.resolve()


_MANAGER = CodingAgentManager()


def get_coding_manager() -> CodingAgentManager:
    return _MANAGER
