"""Local persistent task and JSONL log storage for coding agents."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_STATUSES = {"queued", "running", "waiting_user", "completed", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CodingTaskStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.tasks_path = self.root / "tasks.json"
        self.logs_dir = self.root / "logs"
        self._lock = threading.RLock()

    def ensure_ready(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.tasks_path.exists():
            self._write_tasks({})

    def create_task(
        self,
        *,
        provider: str,
        workspace_path: str,
        prompt: str,
        conversation_ref: str | None = None,
        network_access: bool = True,
    ) -> dict[str, Any]:
        self.ensure_ready()
        task_id = f"code_{uuid.uuid4().hex[:12]}"
        timestamp = utc_now()
        log_path = self.logs_dir / f"{task_id}.jsonl"
        task = {
            "id": task_id,
            "provider": provider,
            "status": "queued",
            "workspace_path": workspace_path,
            "network_access": bool(network_access),
            "prompt": prompt,
            "pending_prompt": prompt,
            "created_at": timestamp,
            "updated_at": timestamp,
            "conversation_ref": conversation_ref,
            "notification_refs": [conversation_ref] if conversation_ref else [],
            "provider_thread_id": None,
            "provider_network_access": None,
            "restart_provider_session": False,
            "last_user_reply": None,
            "approval_request": None,
            "result_summary": None,
            "current_activity": None,
            "current_activity_at": None,
            "changed_files": [],
            "error": None,
            "log_path": str(log_path),
        }
        with self._lock:
            tasks = self._read_tasks()
            tasks[task_id] = task
            self._write_tasks(tasks)
            self.append_event(task_id, "coding", "Task queued", prompt, {"status": "queued"})
        return task

    def list_tasks(self, limit: int | None = None) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._lock:
            tasks = list(self._read_tasks().values())
        tasks.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return tasks[:limit] if limit else tasks

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        self.ensure_ready()
        with self._lock:
            task = self._read_tasks().get(task_id)
        return dict(task) if task else None

    def update_task(self, task_id: str, **updates: Any) -> dict[str, Any]:
        self.ensure_ready()
        with self._lock:
            tasks = self._read_tasks()
            if task_id not in tasks:
                raise KeyError(task_id)
            if "status" in updates and updates["status"] not in TASK_STATUSES:
                raise ValueError(f"Invalid coding task status: {updates['status']}")
            tasks[task_id].update(updates)
            tasks[task_id]["updated_at"] = utc_now()
            self._write_tasks(tasks)
            return dict(tasks[task_id])

    def delete_task(self, task_id: str) -> dict[str, Any]:
        self.ensure_ready()
        with self._lock:
            tasks = self._read_tasks()
            if task_id not in tasks:
                raise KeyError(task_id)
            task = tasks.pop(task_id)
            self._write_tasks(tasks)

            log_path = self.logs_dir / f"{task_id}.jsonl"
            try:
                log_path.unlink()
            except FileNotFoundError:
                pass

            return dict(task)

    def clear_tasks(self, keep_ids: set[str] | None = None) -> dict[str, Any]:
        self.ensure_ready()
        keep_ids = keep_ids or set()
        with self._lock:
            tasks = self._read_tasks()
            kept = {task_id: task for task_id, task in tasks.items() if task_id in keep_ids}
            removed_ids = [task_id for task_id in tasks if task_id not in keep_ids]
            self._write_tasks(kept)

            for task_id in removed_ids:
                try:
                    (self.logs_dir / f"{task_id}.jsonl").unlink()
                except FileNotFoundError:
                    pass

        return {
            "deleted": len(removed_ids),
            "preserved": len(kept),
            "deleted_ids": removed_ids,
            "preserved_ids": list(kept),
        }

    def append_event(
        self,
        task_id: str,
        kind: str,
        title: str,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.ensure_ready()
        event = {
            "time": utc_now(),
            "task_id": task_id,
            "kind": kind,
            "title": title,
            "message": message,
            "data": data or {},
        }
        log_path = self.logs_dir / f"{task_id}.jsonl"
        with self._lock:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def read_logs(self, task_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        self.ensure_ready()
        log_path = self.logs_dir / f"{task_id}.jsonl"
        if not log_path.exists():
            return []

        rows = []
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"time": "", "task_id": task_id, "kind": "log", "title": "Raw log", "message": line.rstrip(), "data": {}})
        return rows[-limit:] if limit else rows

    def _read_tasks(self) -> dict[str, Any]:
        if not self.tasks_path.exists():
            return {}
        try:
            data = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_tasks(self, tasks: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temp_path = self.tasks_path.with_suffix(f"{self.tasks_path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(tasks, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.tasks_path)
