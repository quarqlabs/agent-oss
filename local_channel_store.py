"""Persistent channel history and attachment storage for local Argus channels."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY_WINDOW_MESSAGES = 8
DEFAULT_ATTACHMENT_CONTEXT_CHARS = 16000
DEFAULT_ATTACHMENT_EXTRACT_CHARS = 24000
DEFAULT_PDF_VISION_MAX_PAGES = 3

TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rst",
    ".text",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_storage_name(value: str | None, default: str = "local") -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", value or default).strip("._")
    return clean or default


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def get_channel_storage_root() -> Path:
    explicit_path = os.getenv("LOCAL_CHANNEL_STORAGE_ROOT")
    if explicit_path:
        return resolve_path(explicit_path)

    memory_root = resolve_path(os.getenv("LOCAL_MEMORY_ROOT", "local_memory"))
    agent_id = safe_storage_name(os.getenv("AGENT_ID"), "local_agent")
    return memory_root / agent_id / "channel_state"


def get_chat_history_path() -> Path:
    return get_channel_storage_root() / "chat_history.json"


def get_attachments_dir() -> Path:
    return get_channel_storage_root() / "attachments"


def get_attachments_index_path() -> Path:
    return get_channel_storage_root() / "attachments_index.json"


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def conversation_key(channel_type: str, conversation_id: str | None = None) -> str:
    channel = safe_storage_name(str(channel_type or "web").lower(), "web")
    if conversation_id:
        return f"{channel}:{safe_storage_name(str(conversation_id), 'conversation')}"
    return channel


def load_chat_history_store() -> dict[str, list[dict[str, Any]]]:
    data = read_json_file(get_chat_history_path(), {})
    if not isinstance(data, dict):
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, list)
    }


def list_chat_history_channels() -> list[str]:
    return sorted(load_chat_history_store())


def get_recent_history_items(
    channel_type: str,
    conversation_id: str | None = None,
    limit: int = DEFAULT_HISTORY_WINDOW_MESSAGES,
) -> list[dict[str, Any]]:
    history = load_chat_history_store().get(conversation_key(channel_type, conversation_id), [])
    return list(history[-limit:])


def append_chat_pair(
    channel_type: str,
    user_prompt: str,
    agent_response: str,
    conversation_id: str | None = None,
    attachment_ids: list[str] | None = None,
) -> None:
    key = conversation_key(channel_type, conversation_id)
    data = load_chat_history_store()
    history = data.setdefault(key, [])
    timestamp = now_iso()
    history.extend(
        [
            {
                "role": "human",
                "content": user_prompt,
                "created_at": timestamp,
                "channel_type": channel_type,
                "conversation_id": conversation_id,
                "attachment_ids": attachment_ids or [],
            },
            {
                "role": "ai",
                "content": agent_response,
                "created_at": now_iso(),
                "channel_type": channel_type,
                "conversation_id": conversation_id,
                "attachment_ids": [],
            },
        ]
    )
    atomic_write_json(get_chat_history_path(), data)


def load_attachment_index() -> dict[str, dict[str, Any]]:
    data = read_json_file(get_attachments_index_path(), {})
    if not isinstance(data, dict):
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, dict)
    }


def save_attachment_index(index: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(get_attachments_index_path(), index)


def attachment_path(record: dict[str, Any]) -> Path:
    relative_path = record.get("relative_path")
    if not relative_path:
        raise ValueError("attachment record is missing relative_path")
    return get_channel_storage_root() / str(relative_path)


def attachment_needs_reprocess(record: dict[str, Any]) -> bool:
    extract = record.get("extract") or {}
    if str(extract.get("text") or "").strip():
        return False
    if extract.get("error") or extract.get("ai_extract_error"):
        return True

    mime_type = str(record.get("mime_type") or "")
    source_kind = str(record.get("source_kind") or "")
    return (
        mime_type.startswith("text/")
        or mime_type in {"application/pdf"}
        or mime_type.startswith("image/")
        or mime_type.startswith("audio/")
        or source_kind in {"audio", "voice", "photo", "document"}
    )


def guess_mime_type(filename: str | None, content: bytes) -> str:
    guessed, _ = mimetypes.guess_type(filename or "")
    if guessed:
        return guessed
    try:
        import filetype

        kind = filetype.guess(content)
        if kind and kind.mime:
            return kind.mime
    except Exception:
        pass
    return "application/octet-stream"


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + "\n[truncated]", True


def extract_text_file(path: Path, max_chars: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text, truncated = truncate_text(text, max_chars)
    return {"extract_type": "text", "text": text, "truncated": truncated}


def extract_pdf_file(path: Path, max_chars: int) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        return {
            "extract_type": "pdf",
            "text": "",
            "error": (
                "PDF text extraction requires pypdf. "
                "Run `pip install -r requirements.txt` in this environment. "
                f"Original error: {exc}"
            ),
        }

    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
            if sum(len(item) for item in pages) >= max_chars:
                break
        text, truncated = truncate_text("\n\n".join(pages).strip(), max_chars)
        error = "" if text else "No embedded PDF text was found; vision fallback is required."
        return {
            "extract_type": "pdf",
            "text": text,
            "truncated": truncated,
            "page_count": len(reader.pages),
            "error": error,
        }
    except Exception as exc:
        return {"extract_type": "pdf", "text": "", "error": str(exc)}


def extract_docx_file(path: Path, max_chars: int) -> dict[str, Any]:
    try:
        from docx import Document
    except Exception as exc:
        return {"extract_type": "docx", "text": "", "error": f"python-docx unavailable: {exc}"}

    try:
        document = Document(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
        text, truncated = truncate_text(text, max_chars)
        return {"extract_type": "docx", "text": text, "truncated": truncated}
    except Exception as exc:
        return {"extract_type": "docx", "text": "", "error": str(exc)}


def extract_image_metadata(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return {
                "extract_type": "image",
                "text": "",
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
            }
    except Exception as exc:
        return {"extract_type": "image", "text": "", "error": str(exc)}


def basic_extract_attachment(path: Path, mime_type: str, max_chars: int) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if mime_type.startswith("text/") or suffix in TEXT_EXTENSIONS:
        return extract_text_file(path, max_chars)
    if mime_type == "application/pdf" or suffix == ".pdf":
        return extract_pdf_file(path, max_chars)
    if suffix == ".docx" or mime_type.endswith("wordprocessingml.document"):
        return extract_docx_file(path, max_chars)
    if mime_type.startswith("image/"):
        return extract_image_metadata(path)
    return {"extract_type": "binary", "text": ""}


async def enrich_image_with_openai(path: Path, mime_type: str, max_chars: int) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}

    try:
        from openai import AsyncOpenAI

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=os.getenv("MULTIMODAL_IMAGE_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image for an assistant that must answer "
                                "the user's current question. Include visible text, key "
                                "objects, people, layout, and any important details."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=700,
        )
        text = response.choices[0].message.content or ""
        text, truncated = truncate_text(text, max_chars)
        return {"text": text, "ai_extract_type": "image_description", "truncated": truncated}
    except Exception as exc:
        return {"ai_extract_error": str(exc)}


async def enrich_audio_with_openai(path: Path, max_chars: int) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        with path.open("rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model=os.getenv("MULTIMODAL_AUDIO_MODEL", "gpt-4o-mini-transcribe"),
                file=audio_file,
            )
        text = getattr(transcript, "text", "") or str(transcript)
        text, truncated = truncate_text(text, max_chars)
        return {"text": text, "ai_extract_type": "audio_transcription", "truncated": truncated}
    except Exception as exc:
        return {"ai_extract_error": str(exc)}


def render_pdf_pages(path: Path, max_pages: int) -> list[dict[str, Any]]:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError(
            "PDF vision fallback requires PyMuPDF. Run `pip install -r requirements.txt` "
            f"in this environment. Original error: {exc}"
        ) from exc

    rendered_pages = []
    document = fitz.open(str(path))
    try:
        page_count = min(len(document), max_pages)
        for page_index in range(page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            rendered_pages.append(
                {
                    "page": page_index + 1,
                    "mime_type": "image/png",
                    "bytes": pixmap.tobytes("png"),
                }
            )
    finally:
        document.close()
    return rendered_pages


async def enrich_pdf_with_openai(path: Path, max_chars: int) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}

    try:
        from openai import AsyncOpenAI

        max_pages = int(os.getenv("PDF_VISION_MAX_PAGES", str(DEFAULT_PDF_VISION_MAX_PAGES)))
        rendered_pages = render_pdf_pages(path, max_pages=max_pages)
        if not rendered_pages:
            return {"ai_extract_error": "PDF contained no renderable pages."}

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Extract the readable text and important structure from this PDF. "
                    "This may be a resume/CV or scanned document. Preserve names, "
                    "headings, roles, dates, skills, links, education, projects, and "
                    "actionable details. Return concise plain text."
                ),
            }
        ]
        for page in rendered_pages:
            encoded = base64.b64encode(page["bytes"]).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{page['mime_type']};base64,{encoded}"},
                }
            )

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=os.getenv("MULTIMODAL_IMAGE_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": content}],
            max_tokens=1600,
        )
        text = response.choices[0].message.content or ""
        text, truncated = truncate_text(text, max_chars)
        return {
            "text": text,
            "ai_extract_type": "pdf_vision_ocr",
            "truncated": truncated,
            "vision_pages": len(rendered_pages),
        }
    except Exception as exc:
        return {"ai_extract_error": str(exc)}


async def enrich_attachment_if_supported(record: dict[str, Any], path: Path, max_chars: int) -> dict[str, Any]:
    mime_type = str(record.get("mime_type") or "")
    if mime_type.startswith("image/"):
        return await enrich_image_with_openai(path, mime_type, max_chars)
    if mime_type == "application/pdf" or path.suffix.lower() == ".pdf":
        extract = record.get("extract") or {}
        if not str(extract.get("text") or "").strip():
            return await enrich_pdf_with_openai(path, max_chars)
    if mime_type.startswith("audio/") or str(record.get("source_kind") or "") in {"audio", "voice"}:
        return await enrich_audio_with_openai(path, max_chars)
    return {}


async def store_attachment_from_bytes(
    content: bytes,
    filename: str | None,
    mime_type: str | None,
    channel_type: str,
    conversation_id: str | None = None,
    source_kind: str = "file",
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    max_extract_chars = int(os.getenv("ATTACHMENT_EXTRACT_MAX_CHARS", DEFAULT_ATTACHMENT_EXTRACT_CHARS))
    attachment_id = f"att_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:10]}"
    safe_filename = safe_storage_name(filename or f"{source_kind}_{attachment_id}", "attachment")
    detected_mime = mime_type or guess_mime_type(filename, content)
    digest = hashlib.sha256(content).hexdigest()
    day_dir = get_attachments_dir() / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = day_dir / f"{attachment_id}_{safe_filename}"

    day_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

    extraction = basic_extract_attachment(path, detected_mime, max_extract_chars)
    record = {
        "id": attachment_id,
        "created_at": now_iso(),
        "channel_type": channel_type,
        "conversation_id": conversation_id,
        "source_kind": source_kind,
        "source_metadata": source_metadata or {},
        "original_filename": filename,
        "stored_filename": path.name,
        "relative_path": str(path.relative_to(get_channel_storage_root())),
        "mime_type": detected_mime,
        "size_bytes": len(content),
        "sha256": digest,
        "extract": extraction,
    }

    ai_extract = await enrich_attachment_if_supported(record, path, max_extract_chars)
    if ai_extract:
        record["extract"].update(ai_extract)
        if str(ai_extract.get("text") or "").strip():
            record["extract"].pop("error", None)

    index = load_attachment_index()
    index[attachment_id] = record
    save_attachment_index(index)
    return record


async def reprocess_attachment_record(record: dict[str, Any]) -> dict[str, Any]:
    path = attachment_path(record)
    if not path.exists():
        record["extract"] = {
            "extract_type": "missing",
            "text": "",
            "error": f"Stored attachment file is missing: {record.get('relative_path')}",
        }
        return record

    max_extract_chars = int(os.getenv("ATTACHMENT_EXTRACT_MAX_CHARS", DEFAULT_ATTACHMENT_EXTRACT_CHARS))
    mime_type = str(record.get("mime_type") or guess_mime_type(record.get("original_filename"), path.read_bytes()))
    extraction = basic_extract_attachment(path, mime_type, max_extract_chars)
    record["mime_type"] = mime_type
    record["extract"] = extraction

    ai_extract = await enrich_attachment_if_supported(record, path, max_extract_chars)
    if ai_extract:
        record["extract"].update(ai_extract)
        if str(ai_extract.get("text") or "").strip():
            record["extract"].pop("error", None)

    record["reprocessed_at"] = now_iso()
    return record


async def refresh_attachments_for_context(attachment_ids: list[str]) -> None:
    if not attachment_ids:
        return

    index = load_attachment_index()
    changed = False
    for attachment_id in attachment_ids:
        record = index.get(attachment_id)
        if not record or not attachment_needs_reprocess(record):
            continue
        index[attachment_id] = await reprocess_attachment_record(record)
        changed = True

    if changed:
        save_attachment_index(index)


def get_attachment_record(attachment_id: str) -> dict[str, Any] | None:
    return load_attachment_index().get(attachment_id)


def recent_attachment_ids(
    channel_type: str,
    conversation_id: str | None = None,
    limit: int = 4,
) -> list[str]:
    ids: list[str] = []
    for item in reversed(get_recent_history_items(channel_type, conversation_id, limit=24)):
        for attachment_id in reversed(item.get("attachment_ids") or []):
            if attachment_id not in ids:
                ids.append(attachment_id)
            if len(ids) >= limit:
                return list(reversed(ids))
    return list(reversed(ids))


def attachment_label(record: dict[str, Any]) -> str:
    return (
        record.get("original_filename")
        or record.get("stored_filename")
        or record.get("id")
        or "attachment"
    )


def append_attachment_note(content: str, attachment_ids: list[str] | None) -> str:
    if not attachment_ids:
        return content
    index = load_attachment_index()
    labels = [
        f"{attachment_label(index[item])} ({item})"
        for item in attachment_ids
        if item in index
    ]
    if not labels:
        return content
    note = "[Attachments: " + "; ".join(labels) + "]"
    return f"{content}\n{note}".strip()


def render_attachment_context(
    attachment_ids: list[str],
    max_chars: int = DEFAULT_ATTACHMENT_CONTEXT_CHARS,
) -> str:
    if not attachment_ids:
        return ""

    index = load_attachment_index()
    blocks = []
    remaining = max_chars
    for attachment_id in attachment_ids:
        record = index.get(attachment_id)
        if not record:
            continue

        extract = record.get("extract") or {}
        text = str(extract.get("text") or "").strip()
        metadata = [
            f"id={record.get('id')}",
            f"name={attachment_label(record)}",
            f"type={record.get('mime_type')}",
            f"size={record.get('size_bytes')} bytes",
        ]
        for key in ("page_count", "width", "height", "format", "mode"):
            if extract.get(key) is not None:
                metadata.append(f"{key}={extract.get(key)}")
        if extract.get("ai_extract_type"):
            metadata.append(f"analysis={extract.get('ai_extract_type')}")
        if extract.get("error"):
            metadata.append(f"extract_error={extract.get('error')}")
        if extract.get("ai_extract_error"):
            metadata.append(f"ai_extract_error={extract.get('ai_extract_error')}")

        header = "Attachment: " + " | ".join(metadata)
        available = max(0, remaining - len(header) - 32)
        if text and available:
            excerpt, _ = truncate_text(text, available)
            block = f"{header}\nExtracted content:\n{excerpt}"
        else:
            block = f"{header}\nExtracted content: [No readable text extracted yet.]"
        blocks.append(block)
        remaining -= len(block)
        if remaining <= 0:
            break

    return "\n\n".join(blocks)


def decode_base64_payload(data_base64: str) -> bytes:
    return base64.b64decode(data_base64.encode("ascii"), validate=True)


async def run_blocking(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)
