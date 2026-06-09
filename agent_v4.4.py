# Argus Agent v0.4.4

import os
import json
import asyncio
import re
import numpy as np
import faiss
from datetime import datetime
from typing import Any, TypedDict, Sequence
from dotenv import load_dotenv
import shutil
import time
import sys

from pydantic import SecretStr
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
import tools.tool_manager as tool_manager
import uuid

from functools import wraps
from agent_config import load_agent_config


def configure_stdio_for_windows() -> None:
    """Avoid Windows legacy code pages crashing on status symbols in logs."""
    if os.name != "nt":
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


configure_stdio_for_windows()

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
load_dotenv()

raw_api_key = os.getenv("OPENAI_API_KEY")


def debug_enabled() -> bool:
    env_value = str(os.getenv("AGENT_DEBUG", "")).lower()
    return env_value in {"1", "true", "yes", "on"} or any(
        arg in {"--debug", "--agent-debug"} for arg in sys.argv[1:]
    )


def debug_print(*args, **kwargs) -> None:
    if debug_enabled():
        print(*args, **kwargs)


CODING_TASK_KEYWORDS = (
    "codex",
    "coding agent",
    "codebase",
    "repo",
    "repository",
    "implement",
    "fix bug",
    "debug",
    "refactor",
    "run tests",
    "failing test",
    "edit code",
    "edit file",
    "pull request",
)


def is_coding_task_prompt(prompt: str) -> bool:
    text = re.sub(r"\s+", " ", str(prompt or "").lower()).strip()
    return any(keyword in text for keyword in CODING_TASK_KEYWORDS)


def coding_retrieval_profile(prompt: str) -> dict[str, Any]:
    text = re.sub(r"\s+", " ", str(prompt or "").strip())
    keywords = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", text):
        lowered = token.lower()
        if lowered not in {"please", "could", "would", "should", "that", "this", "with", "from"}:
            keywords.append(token)
        if len(keywords) >= 4:
            break
    return {
        "vector_queries": [
            f"User's coding task context: {text[:180]}",
            "User's recent coding agent task, repository, implementation, bug, tests",
        ],
        "keywords": keywords,
        "search_mode": "standard",
        "threshold": 0.42,
        "top_k": 4,
        "keyword_top_k": 4,
        "max_lines": 24,
    }



AGENT_ID = os.getenv("AGENT_ID") or "local_agent"
USER_ID = os.getenv("USER_ID")

if not raw_api_key:
    raise ValueError("Missing critical environment variable: OPENAI_API_KEY.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_AGENT_ID = re.sub(r"[^a-zA-Z0-9_.-]", "_", AGENT_ID)
LOCAL_MEMORY_ROOT = os.getenv("LOCAL_MEMORY_ROOT", os.path.join(BASE_DIR, "local_memory"))
LOCAL_AGENT_MEMORY_DIR = os.path.join(LOCAL_MEMORY_ROOT, LOCAL_AGENT_ID)

if not all([raw_api_key, AGENT_ID]):
    raise ValueError(
        "Missing critical environment variables (OPENAI_API_KEY, AGENT_ID)."
    )



# LLM for Text Generation
retrieval_llm = ChatOpenAI(
   api_key=SecretStr(raw_api_key),
    temperature=0,
    model="gpt-4o-mini",
    timeout=30,      
    max_retries=3    

)

gen_llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=SecretStr(raw_api_key),
    temperature=0,
    max_retries=3   
)


learn_llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=SecretStr(raw_api_key),
    temperature=0,
    timeout=60,      
    max_retries=3    
)

EMBED_MODEL = "text-embedding-3-large"

# OpenAI Client for Vector Embeddings
embed_client = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=SecretStr(raw_api_key),
    dimensions=1536,
    timeout=20,     
    max_retries=3    
)


# ==========================================
# GLOBAL CACHE & CONCURRENCY LIMITERS
# ==========================================
AGENT_CONFIG_CACHE = None
LEARNING_SEMAPHORE = asyncio.Semaphore(4)  # 🛠️ NEW: Max 4 concurrent DB saves
INGESTION_LEARNING_LOCK = asyncio.Lock()
PENDING_LEARNING_TASKS = set()  # 🛠️ NEW: Tracks active background tasks



# ==========================================
# ROBUST NETWORK RETRY DECORATOR
# ==========================================
def network_retry(max_retries=4, initial_delay=2.0, timeout=15.0): # 🛠️ ADDED TIMEOUT
    """Automatically retries an async function and prevents infinite thread hanging."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    # 🛠️ FORCE TIMEOUT ON THE FUNCTION CALL
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                except asyncio.TimeoutError:
                    print(f"⚠️ [Network Timeout] {func.__name__} hung for {timeout}s (Attempt {attempt+1}/{max_retries}).")
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"❌ [Fatal Error] {func.__name__} failed after {max_retries} attempts: {e}")
                        raise e 
                    print(f"⚠️ [Network Retry] {func.__name__} failed ({e}). Retrying in {delay}s...")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2 
            raise TimeoutError(f"{func.__name__} completely failed after {max_retries} attempts.")
        return wrapper
    return decorator


def persistent_network_retry(initial_delay=2.0, max_delay=60.0, timeout=None): # 🛠️ Changed default to None
    """
    An infinite retry loop for background tasks. 
    It will NEVER drop the task. It keeps trying forever until it succeeds.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            attempt = 1
            while True:  # 🛠️ INFINITE LOOP
                try:
                    if timeout:
                        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                    else:
                        return await func(*args, **kwargs) # 🛠️ Waits forever without killing the nested DB loop
                except asyncio.TimeoutError:
                    print(f"⚠️ [Persistent Queue] {func.__name__} hung for {timeout}s (Attempt {attempt}). Retrying...")
                except Exception as e:
                    print(f"⚠️ [Persistent Queue] {func.__name__} failed ({e}) (Attempt {attempt}). Retrying in {delay}s...")
                
                # Exponential backoff, capped at `max_delay`
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                attempt += 1
        return wrapper
    return decorator


# ==========================================
# HELPER: ROBUST CONTENT EXTRACTION (Gemini Fix)
# ==========================================
def extract_pure_text(response) -> str:
    """Safely extracts raw text from Gemini's complex list/dict response structures."""
    content = response.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = ""
        for part in content:
            if isinstance(part, dict):
                text += part.get("text", "")
            else:
                text += str(part)
    else:
        text = str(content)
        
    # Strip <thinking> tags if the model returned them
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()

def extract_json_block(text: str, is_array: bool = False) -> str:
    """Extracts the last outermost valid JSON object/array from model text."""
    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"<thinking>.*?</thinking>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    decoder = json.JSONDecoder()
    expected_start = "[" if is_array else "{"
    expected_type = list if is_array else dict

    candidates = []

    for idx, char in enumerate(text):
        if char != expected_start:
            continue

        try:
            parsed, end_offset = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, expected_type):
            candidates.append((idx, idx + end_offset, text[idx : idx + end_offset]))

    if not candidates:
        return ""

    outermost = []
    for start, end, block in candidates:
        is_nested = any(
            other_start < start and end <= other_end
            for other_start, other_end, _ in candidates
        )
        if not is_nested:
            outermost.append((start, end, block))

    return outermost[-1][2] if outermost else candidates[0][2]

def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        clean = re.sub(r"\s+", " ", item).strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def format_current_time(value=None) -> str:
    """Render current-date anchors as 'Month D, YYYY, HH:MM' when parseable."""
    if isinstance(value, datetime):
        return f"{value.strftime('%B')} {value.day}, {value.year}, {value.strftime('%H:%M')}"

    text = str(value or "").strip()
    if not text:
        return format_current_time(datetime.now())

    for fmt in (
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%A, %B %d, %Y at %I:%M %p",
    ):
        try:
            return format_current_time(datetime.strptime(text, fmt))
        except ValueError:
            pass

    return text


MEMORY_INGESTION_PREFIX = "Review and remember this conversation history:"
MEMORY_INGESTION_ACK = "Yes, I have reviewed and remembered everything."


def is_memory_ingestion_prompt(prompt: str) -> bool:
    """Detect dataset/history ingestion turns that should learn but not answer."""
    return str(prompt or "").lstrip().lower().startswith(
        MEMORY_INGESTION_PREFIX.lower()
    )


def unwrap_memory_ingestion_prompt(prompt: str) -> str:
    """Return the conversation-history payload for ingestion prompts."""
    text = str(prompt or "")
    if not is_memory_ingestion_prompt(text):
        return text

    stripped = text.lstrip()
    return stripped[len(MEMORY_INGESTION_PREFIX) :].strip()


def wrap_memory_ingestion_payload(payload: str) -> str:
    """Build a memory-ingestion prompt around an already isolated payload."""
    return f"{MEMORY_INGESTION_PREFIX}\n\n{str(payload or '').strip()}"


def split_memory_ingestion_pairs(prompt: str) -> list[str]:
    """Split an ingestion payload into chronological user/assistant pairs."""
    payload = unwrap_memory_ingestion_prompt(prompt)
    if not payload:
        return []

    turns: list[dict] = []
    current_role = None
    current_lines: list[str] = []

    def flush_turn() -> None:
        nonlocal current_role, current_lines
        if current_role and current_lines:
            turns.append(
                {
                    "role": current_role,
                    "content": "\n".join(current_lines).strip(),
                }
            )
        current_role = None
        current_lines = []

    for raw_line in payload.splitlines():
        match = re.match(
            r"^\s*(user|human|assistant|ai)\s*:\s*(.*)$",
            raw_line,
            flags=re.IGNORECASE,
        )
        if match:
            flush_turn()
            raw_role = match.group(1).lower()
            current_role = "user" if raw_role in {"user", "human"} else "assistant"
            current_lines = [match.group(2)]
            continue

        if current_role:
            current_lines.append(raw_line)
        elif raw_line.strip():
            current_role = "user"
            current_lines = [raw_line]

    flush_turn()

    if not turns:
        return [payload.strip()]

    pairs: list[str] = []
    idx = 0
    while idx < len(turns):
        turn = turns[idx]

        if turn["role"] == "user":
            parts = [f"user: {turn['content']}"]
            if idx + 1 < len(turns) and turns[idx + 1]["role"] == "assistant":
                parts.append(f"assistant: {turns[idx + 1]['content']}")
                idx += 2
            else:
                idx += 1
            pairs.append("\n".join(parts).strip())
            continue

        pairs.append(f"assistant: {turn['content']}".strip())
        idx += 1

    return [pair for pair in pairs if pair]


def extract_target_queries_from_thinking(text: str) -> list[str]:
    """Extract target nouns from thinking lines like target_nouns:{museum}."""
    if not text:
        return []

    thinking_match = re.search(
        r"<thinking>(.*?)</thinking>",
        str(text),
        flags=re.DOTALL | re.IGNORECASE,
    )
    source = thinking_match.group(1) if thinking_match else str(text)

    target_terms = []

    target_noun_stopwords = {
        "action",
        "after",
        "all",
        "amount",
        "answer",
        "before",
        "count",
        "date",
        "dates",
        "difference",
        "duration",
        "earliest",
        "eaten",
        "from",
        "gap",
        "how",
        "latest",
        "list",
        "order",
        "ordered",
        "recent",
        "relation",
        "seen",
        "six",
        "target",
        "thing",
        "things",
        "time",
        "timeline",
        "to",
        "total",
        "visit",
        "visited",
        "visits",
        "when",
        "where",
        "which",
    }

    keyword_blocks = re.findall(
        r"target(?:_nouns?|_keywords?)?\s*:\s*\{\{?([^{}\n]+?)\}?\}",
        source,
        flags=re.IGNORECASE,
    )

    for block in keyword_blocks:
        for term in re.split(r"[,;|]", block):
            for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", term):
                clean = word.strip().lower()
                if (
                    clean
                    and clean not in target_noun_stopwords
                    and clean not in {"nouns", "noun", "keywords", "none", "unknown"}
                ):
                    target_terms.append(clean)

    # Be tolerant if the model ignored braces and wrote:
    # "Target: description. target_nouns: museum, visits, order."
    for line in source.splitlines():
        if not re.search(r"\btarget\b", line, flags=re.IGNORECASE):
            continue
        matches = list(re.finditer(r"\btarget(?:_nouns?|_keywords?)?\s*:", line, flags=re.IGNORECASE))
        if len(matches) < 2:
            continue
        start = matches[-1].end()
        raw = line[start:].split(".")[0]
        for term in re.split(r"[,;|]", raw):
            for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", term):
                clean = word.strip().lower()
                if (
                    clean
                    and clean not in target_noun_stopwords
                    and clean not in {"nouns", "noun", "keywords", "none", "unknown"}
                ):
                    target_terms.append(clean)

    return _dedupe_keep_order(target_terms)


MAX_STRUCTURED_ARTIFACT_MEMORIES = 60
MAX_ARTIFACT_LIST_ITEMS_PER_BLOCK = 10


def _table_cells(line: str) -> list[str]:
    line = line.strip().strip("|")
    return [c.strip() for c in line.split("|")]


def _is_table_separator(cells: list[str]) -> bool:
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells
    )


def extract_table_rows(text: str, current_time: str, artifact_context: str) -> list[str]:
    lines = text.splitlines()
    memories = []
    i = 0

    while i < len(lines) - 1:
        if "|" not in lines[i] or "|" not in lines[i + 1]:
            i += 1
            continue

        headers = _table_cells(lines[i])
        sep = _table_cells(lines[i + 1])

        if len(headers) < 2 or not _is_table_separator(sep):
            i += 1
            continue

        i += 2
        while i < len(lines) and "|" in lines[i]:
            row = _table_cells(lines[i])
            if len(row) >= 2 and not _is_table_separator(row):
                row_label = row[0] or "row"
                pairs = []
                for h, v in zip(headers[1:], row[1:]):
                    if h and v:
                        pairs.append(f"{h} = {v}")

                if pairs:
                    memories.append(
                        f"On {current_time}, in {artifact_context}, table row {row_label}: "
                        + "; ".join(pairs)
                        + "."
                    )
            i += 1

    return memories


def extract_heading_sections(text: str, current_time: str, artifact_context: str) -> list[str]:
    lines = text.splitlines()
    memories = []

    heading_re = re.compile(
        r"^\s*(#{1,6}\s+.+|chapter\s+\d+[:.\-].+|\d+[.)]\s+.+|[A-Z][^.!?]{3,80}:)\s*$",
        re.IGNORECASE,
    )

    current_heading = None
    buffer = []

    def flush():
        if current_heading and buffer:
            content = " ".join(x.strip() for x in buffer if x.strip())
            content = re.sub(r"\s+", " ", content).strip()
            if content:
                memories.append(
                    f"On {current_time}, in {artifact_context}, section {current_heading}: {content}"
                )

    for line in lines:
        clean = line.strip()
        if heading_re.match(clean):
            flush()
            current_heading = clean.lstrip("#").strip().rstrip(":")
            buffer = []
        elif current_heading:
            buffer.append(clean)

    flush()
    return memories


def extract_list_items(text: str, current_time: str, artifact_context: str) -> list[str]:
    memories = []
    current_heading = "the artifact"

    heading_re = re.compile(r"^\s*(#{1,6}\s+.+|[A-Z][^.!?]{3,80}:)\s*$")
    item_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$")

    for line in text.splitlines():
        clean = line.strip()

        if heading_re.match(clean):
            current_heading = clean.lstrip("#").strip().rstrip(":")
            continue

        match = item_re.match(clean)
        if match:
            memories.append(
                f"On {current_time}, in {artifact_context}, under {current_heading}, list item: {match.group(1).strip()}"
            )

    return memories


def extract_labeled_lines(text: str, current_time: str, artifact_context: str) -> list[str]:
    memories = []

    patterns = [
        re.compile(r"^::\s*(?P<label>[^:\n]+?)\s*::\s*==\s*(?P<value>.+)$"),
        re.compile(r"^(?P<label>[A-Za-z][^:\n]{2,80})\s*[:=]\s*(?P<value>.+)$"),
    ]

    for line in text.splitlines():
        clean = line.strip()
        for pattern in patterns:
            match = pattern.match(clean)
            if match:
                label = match.group("label").strip()
                value = match.group("value").strip()
                if label.lower() in {"user", "assistant", "ai", "human", "system"}:
                    continue
                memories.append(
                    f"On {current_time}, in {artifact_context}, labeled item {label}: {value}"
                )
                break

    return memories


def extract_explicit_artifact_block_memories(
    text: str,
    current_time: str,
    artifact_context: str,
) -> list[str]:
    """Extract only explicit artifact blocks like `::title:: == description`."""
    memories = []
    marker_re = re.compile(r"::\s*(?P<label>[^:\n]{1,120}?)\s*::\s*==")

    for line in text.splitlines():
        matches = list(marker_re.finditer(line))
        if not matches:
            continue

        for idx, match in enumerate(matches):
            label = re.sub(r"\s+", " ", match.group("label")).strip()
            value_start = match.end()
            value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            value = re.sub(r"\s+", " ", line[value_start:value_end]).strip()

            if not label or not value:
                continue

            memories.append(
                f"On {current_time}, in {artifact_context}, explicit artifact {label}: {value}"
            )

    return _dedupe_keep_order(memories)


def _strip_inline_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    return text.strip()


def _split_named_list_item(body: str) -> tuple[str, str] | None:
    body = _strip_inline_markdown(re.sub(r"\s+", " ", body).strip())
    if not body:
        return None

    match = re.match(
        r"(?P<title>.{2,120}?)(?:\s+[-–—]\s+|:\s+)(?P<desc>.+)$",
        body,
    )
    if not match:
        return None

    title = _strip_inline_markdown(match.group("title")).strip(" .:-")
    desc = _strip_inline_markdown(match.group("desc")).strip()

    if not title or len(desc) < 8:
        return None

    return title, desc


def _has_named_list_anchor(title: str) -> bool:
    """Return true for likely named entities, not generic step labels."""
    clean = re.sub(r"[^A-Za-z0-9&'+ ]", " ", title)
    words = [w for w in clean.split() if w]
    if not words:
        return False

    generic_starters = {
        "add", "assemble", "bake", "blend", "bring", "brush", "check",
        "choose", "consider", "create", "cut", "do", "don't", "dont",
        "enjoy", "explore", "factor", "have", "knead", "make", "map",
        "order", "pace", "pick", "plan", "preheat", "roll", "share",
        "stay", "try", "use", "visit", "watch", "wear", "whisk",
    }

    capitalized = [
        w for w in words
        if re.match(r"^[A-Z][A-Za-z0-9&'+-]*$", w) and w.lower() not in {"a", "an", "the"}
    ]
    acronyms_or_numbers = [
        w for w in words
        if re.search(r"\d", w) or (len(w) > 1 and w.isupper())
    ]

    if len(capitalized) >= 2 or acronyms_or_numbers:
        return True

    first = words[0].lower()
    if len(capitalized) == 1 and first not in generic_starters:
        return True

    return False


def extract_artifact_list_item_memories(
    text: str,
    current_time: str,
    artifact_context: str,
) -> list[str]:
    """
    Extract high-signal bullet/numbered list items without turning every list
    into memory. This is meant for recommendation/artifact lists like
    `1. The Sugar Factory - ...`, not generic step-by-step instructions.
    """
    memories = []
    item_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$")
    context_signal_re = re.compile(
        r"\b(recommend|suggest|option|idea|place|spot|restaurant|shop|activity|"
        r"things to do|where to|itinerary|list|example|dessert|dining|"
        r"family-friendly|museum|exhibition|attraction|venue|operator|trail|"
        r"recipe)\b",
        re.IGNORECASE,
    )

    lines = text.splitlines()
    current_context = artifact_context
    block_count = 0

    for line in lines:
        clean = line.strip()
        match = item_re.match(clean)

        if not match:
            if clean:
                current_context = _clean_table_context(clean)[:180] or artifact_context
            block_count = 0
            continue

        context_blob = f"{artifact_context} {current_context} {clean}"
        if not context_signal_re.search(context_blob):
            continue

        if block_count >= MAX_ARTIFACT_LIST_ITEMS_PER_BLOCK:
            continue

        split = _split_named_list_item(match.group(1))
        if not split:
            continue

        title, desc = split
        if not _has_named_list_anchor(title):
            continue

        memories.append(
            f"On {current_time}, in {artifact_context}, under {current_context}, "
            f"list item {title}: {desc}"
        )
        block_count += 1

    return _dedupe_keep_order(memories)


def extract_factual_numbered_item_memories(
    text: str,
    current_time: str,
    artifact_context: str,
) -> list[str]:
    """Preserve numbered factual examples that carry answerable details."""
    memories = []
    item_re = re.compile(r"^\s*(?:\d+[.)])\s+(.+?)\s*$")

    lines = text.splitlines()
    current_context = artifact_context
    block_count = 0

    for line in lines:
        clean = line.strip()
        match = item_re.match(clean)

        if not match:
            if clean:
                current_context = _clean_table_context(clean)[:180] or artifact_context
            block_count = 0
            continue

        body = _strip_inline_markdown(re.sub(r"\s+", " ", match.group(1)).strip())
        has_answerable_detail = bool(
            re.search(r"\d|[$€£¥%]", body)
            or re.search(r"[\"'“”][^\"'“”]{2,80}[\"'“”]", body)
            or re.search(r"\b[A-Z]{2,}\b", body)
            or len(re.findall(r"\b[A-Z][A-Za-z0-9&'+-]*\b", body)) >= 2
        )
        if not has_answerable_detail:
            continue

        if block_count >= MAX_ARTIFACT_LIST_ITEMS_PER_BLOCK:
            continue

        memories.append(
            f"On {current_time}, in {artifact_context}, under {current_context}, "
            f"factual numbered item: {body}"
        )
        block_count += 1

    return _dedupe_keep_order(memories)


ORDERED_SEQUENCE_MARKER_RE = re.compile(
    r"(?<![\w.])(?P<num>\d{1,4})(?P<sep>\.{1,3}|[):])\s+"
)


def _clip_exact_unit(text: str, max_len: int = 260) -> str:
    clean = re.sub(r"\s+", " ", text).strip(" ;")
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "..."


def extract_ordered_sequence_memories(
    text: str,
    current_time: str,
    artifact_context: str,
) -> list[str]:
    """Preserve exact ordered sequence units that LLM summaries often compress."""
    normalized = re.sub(r"\s+", " ", text.replace("…", "...")).strip()
    if not normalized:
        return []

    memories: list[str] = []
    markers = list(ORDERED_SEQUENCE_MARKER_RE.finditer(normalized))
    items: list[dict] = []

    for idx, match in enumerate(markers):
        body_start = match.end()
        body_end = markers[idx + 1].start() if idx + 1 < len(markers) else len(normalized)
        body = _clip_exact_unit(normalized[body_start:body_end])
        if not body:
            continue

        items.append(
            {
                "num": int(match.group("num")),
                "sep": match.group("sep"),
                "start": match.start(),
                "end": min(body_end, body_start + len(body)),
                "body": body,
            }
        )

    for current, next_item in zip(items, items[1:]):
        if next_item["num"] != current["num"] + 1:
            continue
        if not (0 < next_item["start"] - current["start"] <= 900):
            continue

        memories.append(
            f"On {current_time}, in {artifact_context}, ordered sequence transition: "
            f"{current['num']}{current['sep']} {current['body']}; "
            f"next item {next_item['num']}{next_item['sep']} {next_item['body']}."
        )

    run: list[dict] = []

    def flush_run() -> None:
        if len(run) < 2:
            return
        excerpt_start = run[0]["start"]
        excerpt_end = min(run[-1]["end"], excerpt_start + 700)
        excerpt = _clip_exact_unit(normalized[excerpt_start:excerpt_end], max_len=700)
        if excerpt:
            memories.append(
                f"On {current_time}, in {artifact_context}, ordered sequence excerpt: {excerpt}"
            )

    for item in items:
        if not run:
            run = [item]
            continue

        previous = run[-1]
        if item["num"] == previous["num"] + 1 and item["start"] - previous["start"] <= 900:
            run.append(item)
            continue

        flush_run()
        run = [item]

    flush_run()

    for line in text.splitlines():
        clean = _strip_inline_markdown(line)
        clean = re.sub(r"^(?:user|assistant|ai|human)\s*:\s*", "", clean, flags=re.IGNORECASE).strip()
        if not clean or len(clean) > 280:
            continue
        if not ORDERED_SEQUENCE_MARKER_RE.match(clean):
            continue

        memories.append(
            f"On {current_time}, in {artifact_context}, exact ordered statement: {clean}"
        )

    return _dedupe_keep_order(memories)


def extract_structured_artifact_memories(
    text: str,
    current_time: str,
    artifact_context: str,
) -> list[str]:
    memories = []
    memories.extend(extract_markdown_table_row_memories(text, current_time))
    memories.extend(
        extract_explicit_artifact_block_memories(
            text,
            current_time,
            artifact_context,
        )
    )
    memories.extend(extract_artifact_list_item_memories(text, current_time, artifact_context))
    memories.extend(
        extract_factual_numbered_item_memories(text, current_time, artifact_context)
    )
    memories.extend(extract_ordered_sequence_memories(text, current_time, artifact_context))

    # Keep deterministic extraction high-signal only. Generic headings, numbered
    # sections, and `Label: value` lines are intentionally left to the learning
    # model because extracting all of them floods Episodic memory. List extraction
    # is restricted to named recommendation/artifact items and capped per block.
    return _dedupe_keep_order(memories)[:MAX_STRUCTURED_ARTIFACT_MEMORIES]


def _split_markdown_table_row(line: str) -> list[str]:
    """Split a simple markdown table row into cells."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _is_markdown_table_line(line: str) -> bool:
    return "|" in line and len(_split_markdown_table_row(line)) >= 2


def _is_markdown_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False

    for cell in cells:
        compact = cell.replace(" ", "")
        if not re.fullmatch(r":?-{3,}:?", compact):
            return False
    return True


def _clean_table_context(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(User|AI|Assistant):\s*", "", text, flags=re.IGNORECASE)
    return text.strip(" :-")


def extract_markdown_table_row_memories(text: str, current_time: str = "") -> list[str]:
    """
    Convert markdown tables into row-level memory strings.

    The learning LLM may summarize a table and drop row/cell mappings. These
    deterministic row memories preserve the artifact shape before compression.
    """
    lines = text.splitlines()
    row_memories: list[str] = []
    i = 0

    while i < len(lines):
        if not _is_markdown_table_line(lines[i]):
            i += 1
            continue

        if i + 1 >= len(lines):
            break

        headers = _split_markdown_table_row(lines[i])
        separator = _split_markdown_table_row(lines[i + 1])

        if not _is_markdown_separator_row(separator):
            i += 1
            continue

        table_start = i
        data_rows: list[list[str]] = []
        i += 2

        while i < len(lines) and _is_markdown_table_line(lines[i]):
            row = _split_markdown_table_row(lines[i])
            if not _is_markdown_separator_row(row):
                data_rows.append(row)
            i += 1

        if not data_rows:
            continue

        context_lines: list[str] = []
        context_idx = table_start - 1
        while context_idx >= 0 and len(context_lines) < 3:
            candidate = lines[context_idx].strip()
            if candidate and not _is_markdown_table_line(candidate):
                context_lines.append(candidate)
            context_idx -= 1

        context = _clean_table_context(" ".join(reversed(context_lines)))
        if not context:
            context = "the table"

        for row_num, row in enumerate(data_rows, start=1):
            max_len = max(len(headers), len(row))
            padded_headers = headers + [""] * (max_len - len(headers))
            padded_row = row + [""] * (max_len - len(row))

            first_header = padded_headers[0].strip() or "row"
            first_value = padded_row[0].strip()

            cell_pairs: list[str] = []
            for idx in range(1, max_len):
                header = padded_headers[idx].strip() or f"Column {idx + 1}"
                value = padded_row[idx].strip()
                if not value:
                    continue
                cell_pairs.append(f"{header} = {value}")

            if not cell_pairs:
                for idx in range(max_len):
                    header = padded_headers[idx].strip() or f"Column {idx + 1}"
                    value = padded_row[idx].strip()
                    if value:
                        cell_pairs.append(f"{header} = {value}")

            if not cell_pairs:
                continue

            row_label = f"{first_header} {first_value}".strip()
            if row_label == "row":
                row_label = f"row {row_num}"

            prefix = f"On {current_time}, " if current_time else ""
            row_memories.append(
                f"{prefix}in {context}, table row {row_label}: "
                + "; ".join(cell_pairs)
                + "."
            )

    return row_memories
# ==========================================
# HELPER: TOKEN EXTRACTION (Breakdown)
# ==========================================
def get_token_metrics(response) -> dict:
    """Extracts input, output, and total token usage from LangChain AIMessage."""
    metrics = {"input": 0, "output": 0, "total": 0}

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        metrics["input"] = response.usage_metadata.get("input_tokens", 0)
        metrics["output"] = response.usage_metadata.get("output_tokens", 0)
        metrics["total"] = response.usage_metadata.get("total_tokens", 0)
    elif (
        hasattr(response, "response_metadata")
        and "token_usage" in response.response_metadata
    ):
        usage = response.response_metadata["token_usage"]
        # Safe check: ensures 'usage' is a dict before calling .get()
        if isinstance(usage, dict):
            metrics["input"] = usage.get("prompt_tokens", 0)
            metrics["output"] = usage.get("completion_tokens", 0)
            metrics["total"] = usage.get("total_tokens", 0)

    return metrics




class VectorMemoryManager:
    """Manages Semantic and Episodic memories locally with FAISS + JSON."""

    def __init__(self, memory_type: str):
        self.memory_type = memory_type
        self.dim = 1536
        self.folder = os.path.join(
            LOCAL_AGENT_MEMORY_DIR, f"{memory_type.lower()}_memory"
        )
        self.index_file = os.path.join(self.folder, "index.faiss")
        self.store_file = os.path.join(self.folder, "memories.json")
        self.lock = asyncio.Lock()

        os.makedirs(self.folder, exist_ok=True)
        self.memories = self._load_memories()
        self.index = self._load_or_rebuild_index()

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _load_memories(self) -> list[dict]:
        if not os.path.exists(self.store_file):
            return []
        try:
            with open(self.store_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"⚠️ [Memory] Failed to load {self.memory_type} store: {e}")
            return []

    def _save_memories(self):
        os.makedirs(self.folder, exist_ok=True)
        with open(self.store_file, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

        faiss.write_index(self.index, self.index_file)

    def _vector_from_embedding(self, embedding: list[float]) -> np.ndarray:
        vector = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vector)
        return vector

    def _new_index(self):
        return faiss.IndexFlatIP(self.dim)

    def _rebuild_index(self):
        self.index = self._new_index()
        vectors = []

        for memory in self.memories:
            embedding = memory.get("embedding")
            if embedding:
                vectors.append(np.array(embedding, dtype="float32"))

        if vectors:
            matrix = np.array(vectors, dtype="float32")
            faiss.normalize_L2(matrix)
            self.index.add(matrix)

    def _load_or_rebuild_index(self):
        if os.path.exists(self.index_file):
            try:
                index = faiss.read_index(self.index_file)
                if index.ntotal == len(self.memories):
                    return index
            except Exception as e:
                print(f"⚠️ [Memory] Failed to load FAISS index: {e}")

        self.index = self._new_index()
        self._rebuild_index()
        self._save_memories()
        return self.index

    def _format_memory(self, memory: dict) -> str:
        return (
            f"[STORED_AT: {memory.get('created_at', self._now())}] "
            f"[ID: {memory['id']}] {memory['content']}"
        )

    def _action_entry_time(self, action: dict, default_time: str) -> str:
        for key in ("created_at", "entry_time", "timestamp", "stored_at"):
            value = action.get(key)
            if value:
                return str(value)
        return default_time

    async def _execute_mutation_action_locked(self, action: dict) -> dict | None:
        act_type = action.get("action", "").upper()
        content = action.get("content", "")
        raw_id = action.get("id")

        record_id = None
        if raw_id:
            match = re.search(r"([0-9a-fA-F\-]{36})", raw_id)
            if match:
                record_id = match.group(1)

        if act_type == "DELETE" and record_id:
            before = len(self.memories)
            self.memories = [m for m in self.memories if m.get("id") != record_id]
            if len(self.memories) != before:
                self._rebuild_index()
                self._save_memories()
                debug_print(f"🗑️ [Memory] DELETED {self.memory_type} memory: {record_id}")
                return {"action": "DELETE", "id": record_id}
            return None

        if not content.strip():
            return None

        embedding = await embed_client.aembed_query(text=content)
        if not embedding:
            return None

        vector = self._vector_from_embedding(embedding)

        if act_type == "ADD":
            if self.index.ntotal > 0:
                scores, _ = self.index.search(vector, 1)
                if scores[0][0] >= 0.95:
                    debug_print(f"🔄 [Memory] Skipped duplicate {self.memory_type} ADD.")
                    return None

            now = self._action_entry_time(action, self._now())
            memory = {
                "id": str(uuid.uuid4()),
                "agent_id": AGENT_ID,
                "memory_type": self.memory_type,
                "content": content,
                "embedding": embedding,
                "created_at": now,
                "updated_at": now,
            }
            self.memories.append(memory)
            self.index.add(vector)
            self._save_memories()
            debug_print(f"✅ [Memory] ADDED {self.memory_type}: {content[:30]}...")
            return {"action": "ADD", "id": memory["id"], "memory": memory.copy()}

        if act_type == "UPDATE" and record_id:
            for memory in self.memories:
                if memory.get("id") == record_id:
                    memory["content"] = content
                    memory["embedding"] = embedding
                    memory["updated_at"] = self._now()
                    self._rebuild_index()
                    self._save_memories()
                    debug_print(f"✏️ [Memory] UPDATED {self.memory_type} memory: {record_id}")
                    return {"action": "UPDATE", "id": record_id, "memory": memory.copy()}

        return None

    async def _execute_add_batch_locked(
        self, actions: list[dict], entry_time: str
    ) -> list[dict]:
        unique_actions = []
        seen_contents = {
            re.sub(r"\s+", " ", str(memory.get("content", "")).strip()).lower()
            for memory in self.memories
        }

        for action in actions:
            content = str(action.get("content", "")).strip()
            if not content:
                continue

            normalized_content = re.sub(r"\s+", " ", content).lower()
            if normalized_content in seen_contents:
                debug_print(f"🔄 [Memory] Skipped exact duplicate {self.memory_type} ADD.")
                continue

            seen_contents.add(normalized_content)
            unique_actions.append((action, content))

        if not unique_actions:
            return []

        contents = [content for _, content in unique_actions]
        embeddings = await embed_client.aembed_documents(texts=contents)
        if not embeddings:
            return []

        results: list[dict] = []
        now = entry_time or self._now()

        for (action, content), embedding in zip(unique_actions, embeddings):
            if not embedding:
                continue

            vector = self._vector_from_embedding(embedding)

            if self.index.ntotal > 0:
                scores, _ = self.index.search(vector, 1)
                if scores[0][0] >= 0.95:
                    debug_print(f"🔄 [Memory] Skipped duplicate {self.memory_type} ADD.")
                    continue

            created_at = self._action_entry_time(action, now)
            memory = {
                "id": str(uuid.uuid4()),
                "agent_id": AGENT_ID,
                "memory_type": self.memory_type,
                "content": content,
                "embedding": embedding,
                "created_at": created_at,
                "updated_at": created_at,
            }
            self.memories.append(memory)
            self.index.add(vector)
            results.append({"action": "ADD", "id": memory["id"], "memory": memory.copy()})
            debug_print(f"✅ [Memory] ADDED {self.memory_type}: {content[:30]}...")

        if results:
            self._save_memories()

        return results

    @persistent_network_retry(initial_delay=2.0, max_delay=30.0, timeout=30.0)
    async def execute_actions_with_results(
        self, actions: list[dict], batch_size: int = 32
    ) -> list[dict]:
        if not actions:
            return []

        results: list[dict] = []
        default_entry_time = self._now()
        add_buffer = []
        add_buffer_time = None

        async def flush_adds() -> None:
            nonlocal results, add_buffer, add_buffer_time
            if not add_buffer:
                return

            async with self.lock:
                results.extend(
                    await self._execute_add_batch_locked(add_buffer, add_buffer_time)
                )

            add_buffer = []
            add_buffer_time = None

        for action in actions:
            if not isinstance(action, dict):
                continue

            act_type = action.get("action", "").upper()
            if act_type == "ADD":
                entry_time = self._action_entry_time(action, default_entry_time)
                if (
                    add_buffer
                    and (entry_time != add_buffer_time or len(add_buffer) >= batch_size)
                ):
                    await flush_adds()

                add_buffer.append(action)
                add_buffer_time = entry_time
                continue

            await flush_adds()

            async with self.lock:
                result = await self._execute_mutation_action_locked(action)
                if result:
                    results.append(result)

        await flush_adds()
        return results

    async def execute_actions(self, actions: list[dict], batch_size: int = 32) -> int:
        results = await self.execute_actions_with_results(actions, batch_size=batch_size)
        return len(results)

    async def execute_action(self, action: dict):
        return await self.execute_actions([action], batch_size=1)

    @network_retry(max_retries=4, initial_delay=4.0)
    async def search(self, query: str, top_k: int = 10, threshold: float = 0.38) -> str:
        if not query.strip() or self.index.ntotal == 0:
            return ""

        embedding = await embed_client.aembed_query(text=query)
        if not embedding:
            return ""

        vector = self._vector_from_embedding(embedding)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue
            if idx < len(self.memories):
                results.append(self._format_memory(self.memories[idx]))

        return "\n".join(results)

    @network_retry(max_retries=3, initial_delay=1.0)
    async def keyword_search(self, keywords: list[str], top_k: int = 10) -> str:
        if not keywords:
            return ""

        results = {}
        for kw in keywords:
            clean_kw = kw.strip().lower()
            if len(clean_kw) < 3:
                continue

            for memory in self.memories:
                if clean_kw in memory.get("content", "").lower():
                    results[memory["id"]] = self._format_memory(memory)
                    if len(results) >= top_k:
                        break

        return "\n".join(results.values())

    @network_retry(max_retries=3, initial_delay=2.0)
    async def clear(self):
        async with self.lock:
            self.memories = []
            self.index = self._new_index()
            if os.path.exists(self.folder):
                shutil.rmtree(self.folder)
            os.makedirs(self.folder, exist_ok=True)
            self._save_memories()
            print(f"🧹 [Memory] Wiped all {self.memory_type} memories for agent {AGENT_ID}.")


semantic_db = VectorMemoryManager("Semantic")
episodic_db = VectorMemoryManager("Episodic")


def get_formatted_rules_with_ids(rules: list, limit: int = 15) -> str:
    """Formats a specific list of rule objects with their IDs and DB Timestamps for LLM context."""
    if not rules:
        return ""
    limited_rules = rules[:limit]

    formatted = []
    for r in limited_rules:
        raw_time = r.get("created_at", datetime.now().isoformat())
        dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        formatted.append(f"[STORED_AT: {dt}] [ID: {r['id']}] {r['rule']}")

    return "\n".join(formatted)


PROCEDURAL_DIR = os.path.join(LOCAL_AGENT_MEMORY_DIR, "procedural_memory")
PROCEDURAL_FILE = os.path.join(PROCEDURAL_DIR, "rules.json")


def _load_rules_file() -> list:
    if not os.path.exists(PROCEDURAL_FILE):
        return []
    try:
        with open(PROCEDURAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"⚠️ [Rules] Failed to load rules: {e}")
        return []


def _save_rules_file(rules: list):
    os.makedirs(PROCEDURAL_DIR, exist_ok=True)
    with open(PROCEDURAL_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)


@network_retry(max_retries=4, initial_delay=2.0, timeout=15.0)
async def load_procedural_rules() -> list:
    rules = await asyncio.to_thread(_load_rules_file)
    return sorted(
        rules,
        key=lambda r: r.get("created_at", ""),
        reverse=True,
    )


def save_procedural_rules(valid_rules: list):
    rules = _load_rules_file()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for rule_obj in valid_rules:
        rules.append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": AGENT_ID,
                "rule": rule_obj.get("rule"),
                "reasoning": rule_obj.get("reasoning", ""),
                "target_entity": rule_obj.get("target_entity", "global"),
                "tags": rule_obj.get("tags", []),
                "created_at": now,
                "updated_at": now,
            }
        )

    _save_rules_file(rules)


@network_retry(max_retries=4, initial_delay=2.0, timeout=15.0)
async def clear_procedural_rules():
    await asyncio.to_thread(_save_rules_file, [])


async def wait_for_pending_learning_before_wipe():
    """Let in-flight background learning finish before an API-triggered wipe."""
    global PENDING_LEARNING_TASKS

    if not PENDING_LEARNING_TASKS:
        return

    pending_tasks = list(PENDING_LEARNING_TASKS)
    print(
        f"⏳ [Memory Wipe] Waiting for {len(pending_tasks)} pending learning tasks..."
    )
    results = await asyncio.gather(*pending_tasks, return_exceptions=True)
    failures = [result for result in results if isinstance(result, Exception)]

    if failures:
        print(
            f"⚠️ [Memory Wipe] {len(failures)} pending learning task(s) failed before wipe."
        )


async def wipe_all_memories_for_api():
    """API-safe memory wipe used by benchmark/external callers."""
    await wait_for_pending_learning_before_wipe()
    await wipe_all_memories()


async def wipe_all_memories():
    """Wipes all vectors and rules for this agent from the DB."""
    try:
        await semantic_db.clear()
        await episodic_db.clear()
        
        # 🛠️ USE THE NEW PROTECTED FUNCTION
        await clear_procedural_rules()
        
        print(f"🧹 [Rules] Wiped all procedural rules for agent {AGENT_ID}.")
        print(f"✅ Agent {AGENT_ID} is now completely blank and ready for the next test.")
    except Exception as e:
        print(f"❌ [Error] Failed to wipe all memories: {e}")

def sort_memories_by_recency(memory_block: str,max_lines: int = 15) -> str:
    """Parses timestamps in a block of text and sorts lines newest-to-oldest."""
    if not memory_block or memory_block == "None":
        return "None"

    lines = memory_block.strip().split("\n")

    def extract_timestamp(line):
        match = re.search(
        r"\[(?:STORED_AT:\s*)?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]",
        line,
    )
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            except Exception:
                return datetime.min
        return datetime.min

    # Sort lines by timestamp descending
    sorted_lines = sorted(lines, key=extract_timestamp, reverse=True)
    # 🛠️ TOKEN PROTECTION: Keep only the most recent N lines
    capped_lines = sorted_lines[:max_lines]
    
    return "\n".join(capped_lines)


def format_memory_context_for_prompt(memory_block: str, keep_ids: bool = False) -> str:
    """Strip storage metadata and add ordinal recency labels for LLM prompts."""
    if not memory_block or memory_block == "None":
        return "None"

    section_header_re = re.compile(r"^\[(?:SEMANTIC|EPISODIC|PROCEDURAL)\]$", re.IGNORECASE)

    def clean_memory_line(line: str) -> str:
        cleaned = re.sub(r"\[STORED_AT:\s*[^\]]+\]\s*", "", line).strip()
        if not keep_ids:
            cleaned = re.sub(r"\[ID:\s*[^\]]+\]\s*", "", cleaned).strip()
        return re.sub(r"\s+", " ", cleaned).strip()

    def format_lines(lines: list[str]) -> list[str]:
        if not lines:
            return []

        formatted = []
        total = len(lines)
        for idx, line in enumerate(lines, start=1):
            cleaned = clean_memory_line(line)
            if not cleaned:
                continue

            if total == 1:
                label = "Memory 1 - newest and oldest"
            elif idx == 1:
                label = "Memory 1 - newest"
            elif idx == total:
                label = f"Memory {idx} - oldest"
            else:
                label = f"Memory {idx}"

            formatted.append(f"[{label}] {cleaned}")

        return formatted

    output = []
    pending_lines = []
    current_header = None

    def flush_pending() -> None:
        nonlocal pending_lines, current_header
        formatted = format_lines(pending_lines)
        if formatted:
            if current_header:
                output.append(current_header)
            output.extend(formatted)
        pending_lines = []

    for raw_line in memory_block.strip().split("\n"):
        line = raw_line.strip()
        if not line or line == "None":
            continue

        if section_header_re.match(line):
            flush_pending()
            current_header = line
            continue

        pending_lines.append(line)

    flush_pending()
    return "\n".join(output) if output else "None"


MEMORY_CONTEXT_ID_RE = re.compile(r"\[ID:\s*([0-9a-fA-F\-]{36})\]")


def new_vector_staging_state() -> dict:
    return {"adds": {}, "updates": {}, "deletes": set()}


def extract_action_record_id(action: dict) -> str | None:
    raw_id = action.get("id")
    if not raw_id:
        return None

    match = re.search(r"([0-9a-fA-F\-]{36})", str(raw_id))
    return match.group(1) if match else None


def format_staged_memory_line(memory_id: str, created_at: str, content: str) -> str:
    return f"[STORED_AT: {created_at}] [ID: {memory_id}] {content}"


def stage_vector_actions_on_context(
    current_context: str,
    actions: list[dict],
    db: VectorMemoryManager,
    staged_state: dict,
    max_lines: int = 70,
) -> str:
    """Stage vector actions in RAM and return the updated prompt context."""
    line_by_id: dict[str, str] = {}
    passthrough_lines: list[str] = []

    for raw_line in str(current_context or "").splitlines():
        line = raw_line.strip()
        if not line or line == "None":
            continue

        match = MEMORY_CONTEXT_ID_RE.search(line)
        if match:
            line_by_id[match.group(1)] = line
        else:
            passthrough_lines.append(line)

    adds = staged_state.setdefault("adds", {})
    updates = staged_state.setdefault("updates", {})
    deletes = staged_state.setdefault("deletes", set())

    for action in actions:
        if not isinstance(action, dict):
            continue

        action_type = str(action.get("action", "")).upper()
        content = re.sub(r"\s+", " ", str(action.get("content", "")).strip())

        if action_type == "ADD":
            if not content:
                continue

            memory_id = str(uuid.uuid4())
            created_at = db._action_entry_time(action, db._now())
            adds[memory_id] = {"content": content, "created_at": created_at}
            line_by_id[memory_id] = format_staged_memory_line(
                memory_id, created_at, content
            )
            continue

        record_id = extract_action_record_id(action)
        if not record_id:
            continue

        if action_type == "DELETE":
            if record_id in adds:
                adds.pop(record_id, None)
            else:
                deletes.add(record_id)
                updates.pop(record_id, None)
            line_by_id.pop(record_id, None)
            continue

        if action_type == "UPDATE" and content:
            if record_id in adds:
                created_at = adds[record_id].get("created_at") or db._now()
                adds[record_id] = {"content": content, "created_at": created_at}
                line_by_id[record_id] = format_staged_memory_line(
                    record_id, created_at, content
                )
                continue

            if record_id in deletes:
                continue

            updates[record_id] = {"content": content}
            created_at = db._now()
            existing_line = line_by_id.get(record_id, "")
            stored_at_match = re.search(r"\[STORED_AT:\s*([^\]]+)\]", existing_line)
            if stored_at_match:
                created_at = stored_at_match.group(1)
            line_by_id[record_id] = format_staged_memory_line(
                record_id, created_at, content
            )

    combined = "\n".join(passthrough_lines + list(line_by_id.values()))
    if not combined.strip():
        return ""

    return sort_memories_by_recency(combined, max_lines=max_lines)


def build_staged_vector_commit_actions(staged_state: dict) -> list[dict]:
    actions: list[dict] = []
    deletes = staged_state.get("deletes", set())
    updates = staged_state.get("updates", {})
    adds = staged_state.get("adds", {})

    for record_id in deletes:
        actions.append({"action": "DELETE", "id": record_id})

    for record_id, data in updates.items():
        if record_id in deletes:
            continue
        content = str(data.get("content", "")).strip()
        if content:
            actions.append({"action": "UPDATE", "id": record_id, "content": content})

    for data in adds.values():
        content = str(data.get("content", "")).strip()
        if not content:
            continue
        action = {"action": "ADD", "content": content}
        if data.get("created_at"):
            action["created_at"] = data["created_at"]
        actions.append(action)

    return actions


def add_token_metrics(total: dict, metrics: dict) -> dict:
    """Accumulate token usage dictionaries without assuming all keys exist."""
    for key in ("input", "output", "total"):
        total[key] = total.get(key, 0) + int((metrics or {}).get(key, 0) or 0)
    return total


def debug_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def debug_memory_block(label: str, value: str) -> None:
    clean_value = str(value or "").strip()
    debug_print(f"\n[{label}]")
    debug_print(clean_value if clean_value else "None")


def public_tool_display_name(name: str | None) -> str:
    text = str(name or "")
    if text == "composio" or text.upper().startswith("COMPOSIO_"):
        return "cloud tools"
    if text == "configure_cloud_tools":
        return "cloud tools"
    if text == "coding_agent" or text in {
        "configure_coding_agent",
        "start_coding_task",
        "reply_to_coding_task",
        "get_coding_task_status",
        "get_coding_task_logs",
        "cancel_coding_task",
    }:
        return "coding agent"
    return text


def public_skill_display_names(names: list[str]) -> list[str]:
    return [public_tool_display_name(name) for name in names]


async def report_job_status(
    state: dict,
    stage: str,
    message: str = "",
    data: dict | None = None,
) -> None:
    callback = state.get("job_status_callback")
    if not callback:
        return

    try:
        result = callback(stage, message, data or {})
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:
        debug_print(f"[Warning] Failed to update job status: {exc}")


# ==========================================
# 4. GRAPH STATE
# ==========================================
class AgentState(TypedDict):
    user_prompt: str
    chat_history: Sequence[BaseMessage]
    semantic_context: str
    episodic_context: str
    procedural_context: str
    hyde_queries: list[str]         # 🛠️ NEW: Pass queries to the next node
    selected_skills: list[str]  # UPDATE: Now a list of strings
    skill_markdown: str  # NEW: Documentation for active tools
    final_response: str
    skip_learning: bool
    user_id: str  # NEW: Unique identifier for the user
    channel_type: str  # NEW: e.g., 'telegram', 'whatsapp', 'terminal'
    metrics: dict
    current_date: str  # 🛠️ ADDED: To explicitly pass the benchmark date
    job_status_callback: Any
    attachments_context: str


# ==========================================
# 5. RETRIEVAL NODE (Robust Tagging)
# ==========================================
async def retrieve_memories_node(state: AgentState):
    start_time = time.time()  # START TIMER
    await report_job_status(state, "retrieval", "Building memory queries.")

    # 🛠️ BENCHMARK SYNC: If this is a final benchmark question, wait for all background memories to save FIRST!
    global PENDING_LEARNING_TASKS
    if (
        state.get("channel_type") == "benchmark"
        and is_memory_ingestion_prompt(state.get("user_prompt", ""))
        and INGESTION_LEARNING_LOCK.locked()
    ):
        print("⏳ [Benchmark Ingestion] Waiting for previous ingestion learning before retrieval...")
        async with INGESTION_LEARNING_LOCK:
            pass
        print("✅ [Benchmark Ingestion] Previous ingestion learning finished. Proceeding with retrieval.")

    if state.get("skip_learning", False) and PENDING_LEARNING_TASKS:
        print(
            f"⏳ [Benchmark Sync] Waiting for {len(PENDING_LEARNING_TASKS)} background chunks to finish saving to DB..."
        )
        await asyncio.gather(*PENDING_LEARNING_TASKS, return_exceptions=True)
        print("✅ [Benchmark Sync] All memories saved. Proceeding with retrieval.")

    in_tokens = 0
    out_tokens = 0

    user_prompt = state["user_prompt"]
    history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state["chat_history"][-5:]]
    )
    coding_profile = (
        state.get("channel_type") != "benchmark"
        and is_coding_task_prompt(user_prompt)
    )
    retrieval_profile = coding_retrieval_profile(user_prompt) if coding_profile else {}

    # 🛠️ NEW: Use the provided state date, fallback to system clock if none provided
    if state.get("current_date"):
        current_time_str = state["current_date"]
    else:
        current_time_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    hyde_time_str = format_current_time(current_time_str)

    preference_hyde_hint = """
    If the user's prompt asks for a preference, recommendation, advice, suggestion, or choice, generate queries and keywords to retrieve the user's likes, dislikes, preferences, constraints, avoidances, and related preference signals.
    """.rstrip()

    hyde_prompt = f"""
    You are an AI Search Query Optimizer. 
    Analyze the recent chat history and the user's latest prompt.
    You must generate distinct search queries to maximize the chances of finding the right memory in a hybrid database (Vector + Keyword).
    {preference_hyde_hint}



    Query Definitions:
    - Query 1 (Comprehensive Baseline): A concise, 3rd-person factual statement capturing the core subject of the user's intent. (This is the primary direct search).
    - Query 2 (Entity Focus / Relational Anchor):  A keyword list of specific objects, tools, brands, or places. If the prompt is relational (e.g., 'X before Y'), focus this query strictly on the anchor (Y).
    - Query 3 (Action Focus / Relational Target):  A keyword list of verbs, milestones, or thematic concepts. If the prompt is relational (e.g., 'X before Y'), focus this query strictly on the target detail (X). 
    - Query 4 (Literal Unit & Noun Net): A raw, comma-separated list of ONLY the exact nouns, numbers, and quantitative units (e.g., hours, dollars, three) from the prompt. DO NOT add the word "User". Keep it strictly to the user's literal vocabulary. DO NOT include verbs. DO NOT use synonyms. Keep it under 6 words.
    - Element 5 (Exact Keywords): A raw, comma-separated string of 2 to 4 highly specific proper nouns, names, or rare keywords from the prompt for direct text matching.
    

    DO NOT answer the user's question. Just state the context for a database search.
    
    CRITICAL RULES:
    1. PERSPECTIVE: Convert 1st-person ("I", "my") into 3rd-person ("User", "User's") for Queries 1, 2, and 3. Do NOT add "User" to Query 4.
       Queries 1, 2 MUST literally start with either:
        - "User"
        - "User's"
        This is mandatory.
        Query 3 FORMAT RULES:
        - MUST start with "User's "
        - MUST contain EXACTLY 1-2 comma-separated concepts after "User's"
        - Each concept MUST be 1-3 words maximum
        - NO additional commas
        - NO explanations
        - NO expansions
        - Example valid outputs:
        - "User's leadership, management"
        - "User's purchases, spending"
    2. AGGREGATION & MONEY (CRITICAL): If the user asks for a total sum, count, or cost (e.g., "how much money", "total hours", "how many"):
       - For money/expenses: Query 4 MUST include literal financial symbols and terms: "$, cost, paid, price, bought, purchased, spent".
       - For time/counts: Query 4 MUST include unit words: "hours, times, total, count, days".
       Prefer verb forms like "attended", "visited" over abstract nouns like "attendance".
    3. RELATIONAL DECONSTRUCTION & NEGATIVE CONSTRAINTS (CRITICAL): 
       - If the prompt relates two different things (e.g., 'What did I do before X?', 'Who was at Y with me?', 'X during Y'), Queries 2 and 3 MUST search for those two components INDEPENDENTLY. 
         * Query 2: Search for the context/anchor event (The meeting, the doctor, the concert).
         * Query 3: Search for the specific detail or action (The food, the bedtime, the companion).
       - If the prompt asks for an aggregation over a state change or maintenance action, Query 1 and Query 3 must describe the underlying transition, not only the literal verb in the prompt. Include the source/prior object or state, the resulting object or state, and the concrete target category when these can be inferred from the prompt. This is a conceptual decomposition rule, not a keyword-expansion rule.
       - If the user asks for advice, recommendations, suggestions, or ideas (e.g., "Can you suggest...", "Any advice?", "Can you recommend..."), Query 3 MUST ALSO explicitly search for the user's past struggles, dislikes, negative constraints, or explicit non-interests related to the topic to ensure the agent knows what to avoid (e.g., "User's dislikes, struggles, avoids, not interested in, strictly prefers").
    4. CONDITIONAL TIME RESOLUTION (CRITICAL): The current system date is {hyde_time_str}. 
       - ONLY append dates if the user's intent is temporally bound (e.g., specific past events, action times, time comparisons, durations).
       - If the user's prompt contains relative time (e.g., "last weekend", "yesterday", "recently"), you MUST explicitly calculate the exact target date or year (Month and Day/Year) and put THAT DATE in Query 1 and Query 2. 
       - NEVER use the actual words "last weekend" or "yesterday" in your generated queries. Replace them completely with the calculated absolute date (e.g., "May 9").
       - HOW-LONG-AGO QUERY GUARD: If the user asks "how long ago", "how many weeks ago", "how many days ago", or similar, this rule overrides the relative-date calculation rule above. DO NOT invent or calculate the target event date inside the search query. Search for the named event/entity itself. The current date is only the calculation anchor after retrieval, not a guessed event date.
       - DO NOT append dates or timestamps if the user is asking about permanent facts, timeless attributes, demographics, or general preferences (e.g., "What is my ethnicity?", "Do I like dogs?").
    5. CATEGORICAL , GEOGRAPHICAL & HYPONYMS EXPANSION (CRITICAL): If the user asks about a broad category, an action, or a geographical region, your queries MUST include the category/region name PLUS at least 4 to 5 specific physical examples or sub-locations to catch exact matches in the database. 
       - Noun Example: "electronics" -> "electronics, phone, laptop, TV, headphones, monitor"
       - Verb Example: "read" -> "read, book, novel, article, magazine, textbook"
       - Geography Example: "Europe" -> "Europe, Paris, Rome, London, Germany, Spain"
       - Geography Example: "Hawaii" -> "Hawaii, Maui, Oahu, Kauai, Honolulu"
       Do not just use abstract synonyms; you MUST list specific physical items or sub-locations.
    6. NO ANSWERS: Do not try to answer the question. Only provide the search context.
    7. DYNAMIC SEARCH MODE & JSON FORMAT ONLY (CRITICAL): You must classify the required search depth based on the fundamental nature of the user's question. This determines how wide the database search net will be cast.
       
       Use "deep" (Wide Net / High Recall) IF the prompt involves ANY of the following:
       - Aggregations & Lists: Questions asking for totals, counts, calculations, or exhaustive lists (e.g., "how many", "total cost", "list all").
       - Temporal Spans & Histories: Questions requiring timelines, durations, or finding the origin/transition point of a current state (e.g., "how long have I", "when did I start", "history of").
       - Broad Categories: Questions asking about a general class of items where the database might hold highly specific variants (e.g., asking about "vehicles" when the database might hold "sedan" or "bicycle").
       - Exploratory & Advisory: Open-ended requests for recommendations, advice, or suggestions, which require pulling historical dislikes, past constraints, and broad preferences.
       
       Use "standard" (Strict Net / High Precision) ONLY IF the prompt is:
       - Point-Fact Retrieval: Direct, highly specific questions looking for a single established fact, name, or location (e.g., "What is the name of my pet?", "Where did I leave my keys?", "Who is my manager?").
       
       You MUST return a valid JSON object matching this exact schema. Do not include markdown blocks, just the raw JSON:
       {{
           "vector_queries": ["Query 1", "Query 2", "Query 3", "Query 4"],
           "keywords": "exact, proper, nouns",
           "search_mode": "standard" // or "deep"
       }}

    Recent Chat History:
    {history_text if history_text else "None"}
    
    User's Latest Prompt: "{user_prompt}"
    
    ---
    EXAMPLES OF OPTIMIZED QUERIES:

    Input: "What did I eat before the meeting?" (Relational / Point-Fact)
    Output: {{
        "vector_queries": [
            "User's food consumption prior to the meeting",
            "User's meeting history and schedule",
            "User's meals, diet, and what they ate",
            "eat, meeting"
        ],
        "keywords": "eating, ate, meeting",
        "search_mode": "standard"
    }}

    Input: "How many hours in total did I spend driving?" (Aggregation)
    Output: {{
        "vector_queries": [
            "User total driving duration and travel history",
            "User vehicle, road trip, transit records",
            "User travel milestones, driving time calculation",
            "hours, road trip, destinations"
        ],
        "keywords": "driving, hours, total",
        "search_mode": "deep"
    }}

    Input: "Can you recommend a new hobby for me?" (Exploratory & Advisory)
    Output: {{
        "vector_queries": [
            "User's current hobbies, interests, and recreational activities",
            "User's pastimes, skills, free time",
            "User's dislikes, struggles, quit doing, avoids, negative constraints",
            "hobby, interests, activities, free time"
        ],
        "keywords": "hobby, interests, activities",
        "search_mode": "deep"
    }}
    
    """

    if retrieval_profile:
        debug_print("HYDE skipped: using lightweight coding retrieval profile.")
        content = json.dumps(
            {
                "vector_queries": retrieval_profile["vector_queries"],
                "keywords": ",".join(retrieval_profile["keywords"]),
                "search_mode": retrieval_profile["search_mode"],
            }
        )
    else:
        hyde_response = await retrieval_llm.ainvoke([HumanMessage(content=hyde_prompt)])
        # 🛠️ FIXED: Use robust extractor
        content = extract_pure_text(hyde_response)
        
        

        # Track HyDE tokens
        m_hyde = get_token_metrics(hyde_response)
        in_tokens += m_hyde["input"]
        out_tokens += m_hyde["output"]

    
    # Clean JSON
    if content.startswith("```json"): content = content[7:]
    if content.startswith("```"): content = content[3:]
    if content.endswith("```"): content = content[:-3]
    content = content.strip()

    # ---------------------------------------------------------
    # 🛠️ BULLETPROOF JSON PARSING & DYNAMIC THRESHOLD LOGIC
    # ---------------------------------------------------------
    search_queries = []
    keywords = []
    search_mode = "standard"

    try:
        # Isolate just the JSON object explicitly
        json_str = extract_json_block(content, is_array=False)
        parsed_data = json.loads(json_str)
        
        if isinstance(parsed_data, dict):
            search_queries = parsed_data.get("vector_queries", [])
            
            # Handle keywords (safely parsing string or list)
            keyword_data = parsed_data.get("keywords", "")
            if isinstance(keyword_data, list):
                keyword_data = ",".join(keyword_data)
            keywords = [k.strip() for k in keyword_data.split(",") if len(k.strip()) >= 3]
            
            # Extract Mode
            search_mode = str(parsed_data.get("search_mode", "standard")).lower()
            
        elif isinstance(parsed_data, list):
            # Safe Fallback in case LLM generates the old array format
            if len(parsed_data) == 5:
                keyword_str = str(parsed_data.pop())
                keywords = [k.strip() for k in keyword_str.split(",") if len(k.strip()) >= 3]
            search_queries = parsed_data[:4]
            
    except Exception as e:
        print(f"⚠️ [HyDE] JSON parse failed ({e}). Attempting text fallback.")
        
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith("here are") and not line.lower().startswith("output:"):
                clean_line = re.sub(r"^[\d\.\-\*\s]+", "", line).strip().strip('"').strip("'")
                if clean_line:
                    cleaned_lines.append(clean_line)
        
        if cleaned_lines:
            search_queries = cleaned_lines[:4]
            if len(cleaned_lines) > 4:
                keywords = [k.strip() for k in cleaned_lines[4].split(",") if len(k.strip()) >= 3]
        else:
            print("⚠️ [HyDE] Text fallback failed. Reverting to raw user prompt.")
            search_queries = [user_prompt]

    # Ensure we don't have empty queries
    search_queries = [sq for sq in search_queries if sq.strip()]
    if not search_queries:
        search_queries = [user_prompt]

    if retrieval_profile:
        search_queries = retrieval_profile["vector_queries"]
        keywords = retrieval_profile["keywords"]
        search_mode = "standard"

    # 🛠️ DYNAMIC THRESHOLD APPLICATION
    current_threshold = retrieval_profile.get(
        "threshold",
        0.28 if search_mode == "deep" else 0.38,
    )

    current_top_k = retrieval_profile.get(
        "top_k",
        20 if search_mode == "deep" else 10,
    )
    keyword_top_k = retrieval_profile.get("keyword_top_k", 15)
    retrieval_max_lines = retrieval_profile.get("max_lines", 70)

    await report_job_status(
        state,
        "retrieval",
        f"Searching memory in {search_mode.upper()} mode.",
        {"search_mode": search_mode, "threshold": current_threshold},
    )

    debug_print(f"HYDE Mode: [{search_mode.upper()}] (Threshold: {current_threshold})")
    debug_print("HYDE Vector Queries:", search_queries)
    if keywords:
        debug_print("HYDE Direct Keywords:", keywords)


    # 🛠️ CHANGED: CONCURRENT SEARCH FOR ALL QUERIES USING DYNAMIC THRESHOLD
    semantic_tasks = [semantic_db.search(sq, top_k=current_top_k, threshold=current_threshold) for sq in search_queries]
    episodic_tasks = [episodic_db.search(sq, top_k=current_top_k, threshold=current_threshold) for sq in search_queries]

    if keywords:
        # Add the Keyword Search tasks to the concurrent pool
        semantic_tasks.append(semantic_db.keyword_search(keywords, top_k=keyword_top_k))
        episodic_tasks.append(episodic_db.keyword_search(keywords, top_k=keyword_top_k))

    all_semantic_results = await asyncio.gather(*semantic_tasks)
    all_episodic_results = await asyncio.gather(*episodic_tasks)

    # 🛠️ NEW: DEDUPLICATE RESULTS BASED ON DATABASE ID
    def deduplicate_memories(results_list):
        unique_memories = {}
        for result_block in results_list:
            if not result_block: continue
            lines = result_block.split('\n')
            for line in lines:
                # Extract ID to use as a unique key
                match = re.search(r"\[ID: ([0-9a-fA-F\-]{36})\]", line)
                if match:
                    unique_memories[match.group(1)] = line
        return "\n".join(unique_memories.values())

    combined_semantic = deduplicate_memories(all_semantic_results)
    combined_episodic = deduplicate_memories(all_episodic_results)

    # Apply Temporal Sorting
    semantic_result = sort_memories_by_recency(combined_semantic,max_lines=retrieval_max_lines)
    episodic_result = sort_memories_by_recency(combined_episodic,max_lines=retrieval_max_lines)

    all_rules=[]
      # 🛠️ UPDATED: Procedural Tag Routing with CoT and Fallback
    try:
        all_rules = await load_procedural_rules()
    except Exception:
        print("⚠️ [Warning] Procedural rules completely failed to load after 4 retries. Moving on without them.")
        all_rules = []


    procedural_result = ""

    if all_rules:

        known_tags = list(
            set(
                tag
                for rule in all_rules
                if isinstance(rule, dict)
                for tag in rule.get("tags", [])
            )
        )
        tag_prompt = f"""
        You are an intelligent Routing AI. Your task is to determine which behavioral rules the agent needs to answer the user's prompt correctly.
        
        Current User Prompt: "{user_prompt}"
        Recent Semantic Context: {semantic_result}
        Recent Episodic Context: {episodic_result}
        
        Available Rule Tags: {known_tags}
        
        CHAIN OF THOUGHT REQUIREMENTS:
        1. Analyze the user's intent. Is it a greeting? A coding request? A technical architecture question?
        2. Select tags that apply to this intent.
        3. ALWAYS include the "global" tag, as it contains universal personality traits.
        
        You MUST respond EXACTLY with a valid JSON object matching this schema:
        {{
            "reasoning": "Briefly explain what the user wants and why you selected these tags.",
            "tags": ["global", "tag1", "tag2"]
        }}
        """

        response = await retrieval_llm.ainvoke([HumanMessage(content=tag_prompt)])
        # TRACK BREAKDOWN
        m = get_token_metrics(response)
        in_tokens += m["input"]
        out_tokens += m["output"]

        # 🛠️ FIXED: Use robust extractor
        content = extract_pure_text(response)
        

        try:
            # Isolate just the JSON object
            json_str = extract_json_block(content, is_array=False)
            parsed_data = json.loads(json_str)

            # CRITICAL FIX: Verify parsed_data is a dictionary
            if isinstance(parsed_data, dict):
                requested_tags = [
                    str(tag).lower() for tag in parsed_data.get("tags", [])
                ]
            else:
                # If LLM returned a list or string, fallback to global
                requested_tags = ["global"]

            matched_rules = []
            for rule in all_rules:
                # ARMOR: Skip if the JSON file got corrupted with raw strings
                if not isinstance(rule, dict):
                    continue

                rule_tags = [t.lower() for t in rule.get("tags", [])]
                if (
                    any(tag in rule_tags for tag in requested_tags)
                    or "global" in rule_tags
                ):
                    matched_rules.append(rule)

            # 🛠️ CHANGED: Use the formatter to attach IDs and apply the Hard Limit
            procedural_result = get_formatted_rules_with_ids(matched_rules, limit=8)

        except Exception as e:
            print(f"[Warning] Failed to parse procedural tags: {e}")
            # Fallback: Just show the 10 most recent rules if routing fails
            procedural_result = get_formatted_rules_with_ids(all_rules, limit=8)

    procedural_result = sort_memories_by_recency(procedural_result)

    end_time = time.time()  # END TIMER

    debug_print("\n--- Memory Retrieval Complete ---")
    # if semantic_result:
    #     print(f"Semantic Found:\n{semantic_result}")
    # if episodic_result:
    #     print(f"Episodic Found:\n{episodic_result}")
    # if procedural_result:
    #     print(f"Procedural Found:\n{procedural_result}")
    # else:
    #     print("Procedural Found: None (No relevant tags found)")

    print(
        f"⏱️ [Metrics] Time: {end_time - start_time:.2f}s | Tokens: In({in_tokens}) Out({out_tokens})"
    )
    debug_print("---------------------------------\n")
    await report_job_status(
        state,
        "retrieval",
        "Memory retrieval complete.",
        {
            "semantic_lines": len(str(semantic_result or "").splitlines()),
            "episodic_lines": len(str(episodic_result or "").splitlines()),
            "procedural_lines": len(str(procedural_result or "").splitlines()),
        },
    )

    return {
        "semantic_context": semantic_result,
        "episodic_context": episodic_result,
        "procedural_context": procedural_result,
        "hyde_queries": search_queries, # 🛠️ NEW
        "metrics": {"retrieval_in": in_tokens, "retrieval_out": out_tokens},
    }


# ==========================================
# 5b. TOOL ROUTING NODE
# ==========================================
async def route_tools_node(state: AgentState):
    """Pick skills for this turn using the tool_manager."""
    start_time = time.time()
    await report_job_status(state, "tool_routing", "Checking whether a tool is needed.")

    def print_tool_metrics() -> None:
        end_time = time.time()
        router_metrics = tool_manager.get_last_router_metrics()
        print(
            f"⏱️ [Metrics] Tool Routing Time: {end_time - start_time:.2f}s | Tokens: In({router_metrics.get('input', 0)}) Out({router_metrics.get('output', 0)})"
        )
        debug_print("---------------------------------\n")

    # NEW: Disable tool calling entirely during benchmarks
    if state.get("channel_type") == "benchmark":
        debug_print("--- Tool Routing: Skipped (Benchmark Mode) ---")
        print_tool_metrics()
        await report_job_status(
            state,
            "tool_routing",
            "Tool routing skipped for benchmark mode.",
            {"skills": []},
        )
        return {"selected_skills": [], "skill_markdown": ""}

    if is_memory_ingestion_prompt(state["user_prompt"]):
        debug_print("--- Tool Routing: Skipped (Memory Ingestion Mode) ---")
        print_tool_metrics()
        await report_job_status(
            state,
            "tool_routing",
            "Tool routing skipped for memory ingestion.",
            {"skills": []},
        )
        return {"selected_skills": [], "skill_markdown": ""}

    # We pass the history to the tool router for better intent detection
    history_text = "\n".join(
        [f"{m.type}: {m.content}" for m in state["chat_history"][-6:]]
    )

    # --- NEW: Compile memory context for the router ---
    memory_context = f"""
    [Semantic]: {state.get('semantic_context', 'None')}
    [Episodic]: {state.get('episodic_context', 'None')}
    [Procedural]: {state.get('procedural_context', 'None')}
        """.strip()

    chosen_skills = await tool_manager.select_skills(
        user_prompt=state["user_prompt"],
        recent_history=history_text,
        memory_context=memory_context,
    )

    if not chosen_skills:
        debug_print("--- Tool Routing: No skill selected ---")
        print_tool_metrics()
        await report_job_status(
            state,
            "tool_routing",
            "No tool needed for this request.",
            {"skills": []},
        )
        return {"selected_skills": [], "skill_markdown": ""}

    debug_print(
        f"--- Tool Routing: Skills Selected -> '{public_skill_display_names(chosen_skills)}' ---"
    )
    await report_job_status(
        state,
        "tool_routing",
        "Tool skill selected.",
        {"skills": chosen_skills},
    )

    combined_markdown = ""
    for skill in chosen_skills:
        loaded = tool_manager.load_skill(skill)
        combined_markdown += f"\n### {skill.upper()} SKILL\n{loaded['markdown']}\n"

    print_tool_metrics()
    return {"selected_skills": chosen_skills, "skill_markdown": combined_markdown}


# ==========================================
# 6. GENERATION NODE (Robust CoT Generation & ReAct Loop)
# ==========================================
async def generate_response_node(state: AgentState):
    start_time = time.time()
    await report_job_status(state, "generation", "Generating response.")

    # 1. INITIALIZE variables at the top of the function scope
    in_tokens = 0
    out_tokens = 0

    # 🛠️ TRACK CONTEXT FOR STATE UPDATES & LEARNING
    sem_ctx_to_save = state.get("semantic_context", "")
    epi_ctx_to_save = state.get("episodic_context", "")

    global AGENT_CONFIG_CACHE  # 🛠️ Pull in the global cache

    if AGENT_CONFIG_CACHE is None:
        debug_print("📁 [Local Config] Loading Agent Config from local identity config...")
        AGENT_CONFIG_CACHE = load_agent_config()

    cfg = AGENT_CONFIG_CACHE
    name = cfg.get("agent_name") or "Argus Agent"
    personality = cfg.get("agent_personality") or "professional and helpful"
    use_cases = ", ".join(cfg.get("agent_use_cases") or ["general assistance"])
    custom_prompt = cfg.get("agent_custom_prompt") or ""



    # 🛠️ NEW: Pass current date down to the LLM
    if state.get("current_date"):
        current_time_str = state["current_date"]
    else:
        current_time_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    prompt_time_str = format_current_time(current_time_str)


    identity_instruction = f"""
    [IDENTITY & PERSONA]
    Your Name: {name}
    Personality/Tone: {personality}
    Core Objectives: You are specifically optimized for: {use_cases}.
    {f'Custom User Directives: {custom_prompt}' if custom_prompt else ''}
    CURRENT SYSTEM DATE/TIME: {prompt_time_str} 
    

    Your responses must strictly align with this identity and tone.
                """.strip()

    is_benchmark_channel = state.get("channel_type") == "benchmark"
    normal_context_policy_instruction = ""
    normal_general_question_hint = ""
    if not is_benchmark_channel:
        normal_context_policy_instruction = """
    [NORMAL CHANNEL CONTEXT POLICY]
    Normal channel mode is not benchmark mode.
    - For user-specific memory questions about the user's past, preferences, possessions, plans, purchases, dates, current/latest personal state, or things they previously told you, retrieved memory is authoritative. If the required personal fact is not in retrieved memory, say that you do not have enough memory/context.
    - For general conversation, advice, brainstorming, explanations, coding help, research-style thinking, or common-knowledge questions, answer normally using your own reasoning and knowledge. Retrieved memory may personalize the answer, but it is not a hard limit.
    - REQUIRED_DATA is for missing user-memory facts, unresolved personal references, or missing values needed for a user-specific calculation. Do not trigger REQUIRED_DATA merely because a general advice question lacks personal memories.
        """.strip()
        normal_general_question_hint = """
    [NORMAL-CHANNEL GENERAL QUESTION OVERRIDE]
    If the user is not asking for a stored personal fact, do not force unrelated memories into the answer and do not build a strict memory ledger over irrelevant context.
    In the <thinking> block, briefly state that this is a general question and whether any retrieved memory is relevant for personalization. Then answer directly in agent_response.
        """.strip()

    system_instruction = f"""You are a highly advanced, disciplined AI assistant created by the Argus team.

    {identity_instruction}

    TOOL USE PROTOCOL:
    - If tools are available and needed, you may call them sequentially.
    - If a tool result shows that another tool call is needed, call the next tool with the new information.
    - If you use a tool, provide a helpful response to the user after tool execution is complete. Never respond with only tool calls.
    - After all necessary tool calls are complete, provide the final response in the required JSON format.

    [REQUIRED_DATA] PROTOCOL & HYDE QUERY GENERATION (CRITICAL):
    You must maintain strict skepticism about your retrieved context. You MUST trigger the REQUIRED_DATA flag in your final JSON if you encounter ANY of the following gaps:
    - Missing Variables & Scope Mismatches: The user asks for a calculation or comparison for a SPECIFIC target. If your context contains the necessary numbers, but they belong to a DIFFERENT target or storyline than the one the user requested, you cannot use them. You must trigger REQUIRED_DATA to search for the correct target's exact variables.
    - Broad vs. Specific Gap: The user asks for a highly specific duration, phase, or item, but your context only provides the broad, overarching total (meaning a transition or sub-phase is missing).
    - Recommendation Blindspots: The user asks for open-ended recommendations, but you lack data on their explicit dislikes or past constraints.
    - Unresolved References & Scope Mismatch: The user asks about a specific reference using generic pronouns/terms (e.g., "my latest vehicle", "my current project") but the context only contains overarching categorical data (e.g., "has been driving for 10 years", "works in software").
    - Ambiguous References & Cross-Contamination (CRITICAL): The user asks about a generic location, event, or item (e.g., "the airport", "the hotel", "the wedding") and your retrieved context contains MULTIPLE distinct entities of that type (e.g., a trip to Mumbai AND a trip to Tokyo). You are FORBIDDEN from mixing facts, prices, or timelines between different storylines to force an answer. You must trigger REQUIRED_DATA to clarify which specific entity they mean.

    When triggering this protocol:
    - Leave "agent_response" empty.
    - Add "REQUIRED_DATA" to the "flags" array.
    - Provide exactly 3 highly specific keyword queries in the "hyde_queries" array (maximum 2-3 words per query).

    HOW TO CONNECT THE DOTS FOR QUERY GENERATION:
    1. Identify the logical gap between the user's prompt and your current knowledge.
    2. Deduce the exact transition, opposite, or missing component required to bridge that gap.
    3. Distill that deduction into targeted keywords.

    Examples of connecting the dots:
    - Gap (Entity Hijacking Prevention): User asks for the total cost of the "Marketing software". Context shows the "HR software" costs $500, but mentions no price for the Marketing software.
      Dot-Connecting: I cannot substitute the HR software price for the Marketing software just because they are both software. The exact data for the target entity is missing.
      Queries: ["marketing software cost", "marketing software price", "paid marketing software"]

    - Gap (Ambiguous Reference): User asks about "the airport". Context mentions Mumbai Airport and Narita Airport.
      Dot-Connecting: I cannot mix Mumbai prices with a Tokyo hotel. I need to know the specific airport or if the specific bus price exists for the target hotel.
      Queries: ["airport bus", "bus fare", "hotel transit"]

    - Gap (Exact Noun Mismatch): User asks about "my leased apartment". Context shows they lived in a "purchased condo", but a lease and a purchase are legally distinct.
      Dot-Connecting: I cannot assume the purchased condo is the leased apartment. I must search specifically for the leased apartment.
      Queries: ["leased apartment", "rented apartment", "leasing"]

    - Gap (Unresolved Reference): User asks about "my latest car". Context shows they have been driving for 10 years, but does not name a specific car model.
      Dot-Connecting: I cannot apply the overarching 10-year driving history to the "latest car" without knowing what the latest car is and when it was purchased. I need to find the specific car acquisition.
      Queries: ["new car", "bought vehicle", "latest vehicle"]
    
    - Gap (Missing Entity): User asks for total cost of Item A and Item B. Context only shows Item A.
      Dot-Connecting: I cannot do the math. I need the exact purchase event for Item B.
      Queries: ["bought [Item B]", "cost [Item B]", "paid [Item B]"]

    - Gap (Negative Constraints): User asks for new hobby recommendations. Context only shows what they currently do.
      Dot-Connecting: To give good advice, I need to know what they dislike, abandoned, or actively avoid so I don't recommend the wrong thing.
      Queries: ["dislikes", "quit doing", "avoids"]

    FINAL OUTPUT FORMAT (CRITICAL):
    Your final response MUST end with a valid JSON object matching this schema. You may include a <thinking> block immediately before the JSON when requested, but do not output any other raw text outside the JSON block.
    {{
        "agent_response": "Your final conversational answer to the user.",
        "flags": ["REQUIRED_DATA"], 
        "hyde_queries": ["keyword one", "keyword two", "keyword three"] 
    }}
    """
    if not is_benchmark_channel:
        system_instruction = system_instruction.replace(
            "\n\n    TOOL USE PROTOCOL:",
            f"\n\n    {normal_context_policy_instruction}\n\n    TOOL USE PROTOCOL:",
            1,
        )
        system_instruction = system_instruction.replace(
            "    - Recommendation Blindspots: The user asks for open-ended recommendations, but you lack data on their explicit dislikes or past constraints.",
            "    - Recommendation Blindspots: For user-specific personalization requests, the user asks for open-ended recommendations, but you lack data on their explicit dislikes or past constraints. For broad/general advice in normal channel mode, answer normally with a transparent caveat instead of triggering REQUIRED_DATA solely for missing preference memories.",
            1,
        )

    preference_answer_hint = """
        [QUESTION-TYPE GUIDANCE]
        If the user asks for a preference, recommendation, advice, suggestion, or choice, answer as a personalized recommendation rather than a neutral factual summary.
        Start the answer with preference-oriented wording such as "Based on your preferences...", "Given your preferences...", "You would likely prefer...", or "My recommendation would be...".

        Use retrieved likes, dislikes, preferences, constraints, avoidances, current pain points, workarounds, priorities, recent purchases, and desired outcomes as evidence.
        Do not reject supporting current-state/workaround evidence merely because it is not the exact target product or noun.
        Do not recommend options that conflict with retrieved dislikes, avoidances, or negative constraints.
        Do not introduce wait-for-sale, budget, new-model, or delay advice unless retrieved evidence supports that concern.
        """.rstrip()

    knowledge_update_answer_hint = """
        [QUESTION-TYPE GUIDANCE]
        If the user asks a knowledge-update or current-state question, expect that older memories may be superseded, modified, or contextualized by later memories about the same target.
        Reconstruct the target's timeline and answer with the latest supported state interpretation.
        If the user asks a current-state question, use the user question time as the anchor unless another anchor is stated.
        If a newer same-target update is an intention/plan with a concrete target state and its effective date/window is before the anchor, include that newer intended state first and mention the older confirmed state as prior context instead of answering only with the older state.
        """.rstrip()

    assistant_recall_answer_hint = """
        [QUESTION-TYPE GUIDANCE]
        If the user asks a recall question about information the assistant previously provided, treat assistant-provided facts in episodic memory as valid evidence.
        This includes factual list items, study details, counts, sample sizes, journal names, article titles, examples, recommendations, ordered sequences, tables, and route or step details.
        Do not reject evidence merely because it came from an assistant answer rather than a user statement.
        """.rstrip()

    benchmark_span_aggregation_hint = ""
    if is_benchmark_channel:
        benchmark_span_aggregation_hint = """
        [BENCHMARK TEMPORAL CONSISTENCY HINT]
        Some benchmark histories contain wrapper/question timestamps that conflict with narrative dates inside user-stated completed events.
        For broad relative-span aggregations such as "last N months", "past N months", or "in recent months", do not exclude a retrieved event solely because its narrative date appears after the wrapper question timestamp if the memory explicitly describes the event as completed/attended/paid/bought/received and it otherwise matches the requested target.
        Use the completed-event wording and exact amount/count from the memory. Still reject explicit future plans, intentions, recommendations, or unexecuted possibilities.
        """.rstrip()

    generation_semantic_context = format_memory_context_for_prompt(
        sem_ctx_to_save, keep_ids=False
    )
    generation_episodic_context = format_memory_context_for_prompt(
        epi_ctx_to_save, keep_ids=False
    )
    generation_procedural_context = format_memory_context_for_prompt(
        state.get("procedural_context", ""), keep_ids=False
    )
    attachment_context = str(state.get("attachments_context") or "").strip()
    attachment_prompt_block = ""
    if attachment_context:
        attachment_prompt_block = f"""
    [CURRENT MESSAGE ATTACHMENTS]
    The user attached file(s) to the current or recent channel message. Treat this attachment context as current-message evidence.
    Use extracted text, transcript, image description, and metadata below when answering file-related questions.
    If no readable text was extracted, say what was received and what is missing instead of inventing the file contents.

    {attachment_context}
        """.rstrip()
        system_instruction += """

    [ATTACHMENT HANDLING]
    When CURRENT MESSAGE ATTACHMENTS are present, they are part of the user's current message. Use them directly for summaries, extraction, analysis, and follow-up references such as "this file", "that PDF", "the image", or "the audio".
        """.rstrip()

    debug_print(f"""

    Retrieved context:
    Within each memory block, memories are listed newest-to-oldest by storage recency.
    [Memory 1 - newest] is the newest retrieved memory in that block; the line marked oldest is the oldest.

    [SEMANTIC - User Facts]
    {generation_semantic_context}

    [EPISODIC - Past Events]
    {generation_episodic_context}

    [PROCEDURAL - Strict Rules]
    {generation_procedural_context}

    """)

    question_time_str = prompt_time_str

    final_user_prompt = f"""
    Deduce the response for the user question using only the retrieved context below.

    Retrieved context:
    Within each memory block, memories are listed newest-to-oldest by storage recency.
    [Memory 1 - newest] is the newest retrieved memory in that block; the line marked oldest is the oldest.

    [SEMANTIC - User Facts]
    {generation_semantic_context}

    [EPISODIC - Past Events]
    {generation_episodic_context}

    [PROCEDURAL - Strict Rules]
    {generation_procedural_context}

    {attachment_prompt_block}

    Grounding rules:
    - Do not use outside knowledge, assumptions, guesses, or hallucinated facts.
    - Bind the exact requested target before selecting evidence: answer type, requested count/list size, exact noun/category, qualifiers, relation/action, and time/scope.
    - Accept only entities whose actual name or type matches the requested target noun/category. Reject sibling or nearby categories.
    - Do not broaden or substitute categories unless the user asks for suggestions, recommendations, opinions, ideas, examples, or related items. For example: museum != gallery, fruit != vegetable, chocolate bar type != candy, hotel != restaurant, shirt != all clothing.
    - For category-specific list/order/count questions, a candidate must either contain the requested category in its name/type or be explicitly classified as that category by the memory. Do not infer category membership from cultural, travel, tour, venue, historical, educational, or recommendation context alone.
    - For exact-category questions, if a candidate's own name or stated type contains a sibling category noun that is not the requested category, reject it even if the activity/relation resembles the requested category. If the requested category is one noun, a different venue/type noun is not accepted unless the memory explicitly states it belongs to the requested category.
    - If the user asks for exactly N targets, first filter and deduplicate by exact category, completed relation/action, and requested scope. If exactly N valid targets remain, answer with all N valid targets and do not replace any valid target with a near-miss. If more or fewer than N valid targets remain after strict filtering, explain the supported ledger or request more data according to the REQUIRED_DATA rule; never choose the first N from a mixed list that still contains rejected near-misses.
    - Any entity written under rejected near-misses is banned from calculation and from being stated as the direct answer.
    - Use narrative dates inside memory content as event/state dates. Use ordinal memory labels only as storage-recency hints, not as historical event dates.
    - For event ordering, separate event dates from report dates. Phrases such as "reported having", "mentioned having", "recalled", "recently", or "as of" often mark when the event was discussed, not when it happened. When several memories describe the same real-world event, use the earliest exact completed-event date directly tied to that event; do not move the event later because it was re-mentioned later.
    - Preserve the memory's action/verb semantics exactly. Do not convert, promote, or complete one stated action into another stronger or different action unless the memory explicitly states that stronger action happened.
    - For state updates, a supporting date/window memory may be used if it is clearly part of the same target update chain, even if it does not repeat the target noun. Example: "store old notebooks in a shelf box" plus "organize the shelf on Jan 5" can supply Jan 5 as the effective window for the shelf-box update. Do not use unrelated same-category plans this way.
    - For current-state questions, the first sentence of agent_response must answer the current state at the anchor. Put previous/historical states after that, not before.
    - For ordered sequences, games, logs, score sheets, or notation, exact recorded sequence items are valid evidence. If the question asks what came after an anchor and an ACCEPT row explicitly contains that anchor followed by the next item, answer with that next item.
    - For monetary, count, duration, or total aggregation questions, build a complete ledger of every ACCEPT row with its date, target, and numeric value before calculating. Do not answer from only the most recent or most salient row. Sum/count every unique ACCEPT row in scope, including older rows within the requested window.
    - For numeric comparison questions with multiple candidate values for the same side of the comparison, separate user-provided/recalled values and direct corrections of those values from assistant-generated general estimates or alternative-option estimates. Prefer the user-specific value or its direct correction when it matches the requested option. Treat broader, newer, bundled, or assistant-only estimates as caveat/competing evidence unless the user explicitly asks for that combined option or later adopts that estimate.
    - For numeric difference, savings, or comparison questions, if the accepted evidence produces a bounded range, include both the supported range and a natural midpoint/average or rounded estimate when that estimate is meaningful. Do not average open-ended, categorical, asymmetric, or incompatible quantities.
    - For count/times aggregation questions, deduplicate repeated memories that describe the same real-world event before counting. Count unique completed events, not every retrieved mention or summary of the same event.
    - For list/order/count questions, if a later memory repeats a completed event using report wording and an earlier memory gives a more direct completed-event date for the same target/action, merge them as one event under the direct event date.
    - For count/times aggregation questions, a completed attempt still counts even if the result was poor, failed, disappointing, or later retried, unless the user explicitly asks only for successful outcomes.
    - For state-change aggregation questions, accept evidence that preserves the same real-world transition even when it is expressed through old/source state plus new/result state rather than the exact wording in the question. Keep each changed target distinct; do not count generic advice, future plans, or unrelated ownership as a completed transition.
    - For current possession/count questions about durable objects, treat earlier owned/acquired/used/setup objects as still current unless later evidence says they were sold, discarded, returned, replaced, transferred away, or otherwise no longer owned/kept by the user.
    - For compound target categories with modifiers, bind the modifier as part of the category. Accept only evidence explicitly linked to the modified category or its normal function/context. Reject nearby items when the modifier is not supported. Do not infer a domain/location modifier merely because an item could plausibly be associated with that domain.
    - For aggregation questions where exact requested qualifiers are partially missing but the retrieved context contains a small, unambiguous set of matching completed targets with the needed numeric values, do not refuse solely because the qualifier label is missing. If the question asks for N targets and the retrieved context contains exactly N matching completed targets with the needed values, compute the result with a caveat about the missing qualifier label. Answer transparently: state that the requested qualifier was not found/preserved, list the matching completed targets actually found with their available dates/values, and then provide the supported total. Do not invent missing qualifiers.

    MEMORY STATE RESOLUTION RULE:
    Retrieved memories may describe states, events, observations, plans, intentions, preferences, beliefs, relationships, possessions, locations, activities, or updates over time.
    When multiple memories refer to the same entity, attribute, object, situation, preference, project, relationship, possession, or location:
    1. Construct a timeline using the narrative dates found in memory content.
    2. Do not automatically discard older memories because newer memories exist.
    3. Classify each relevant memory as exactly one of: OBSERVED_STATE, REPORTED_STATE, EVENT, PLAN, INTENTION, PREFERENCE, STATE_UPDATE.
    4. Identify whether later memories confirm an earlier state, modify it, contradict it, replace it, or merely discuss a future possibility.
    5. Do not assume that a PLAN or INTENTION was executed by default. However, for current-state questions, if a later same-target PLAN/INTENTION gives a concrete target state/location/value and its planned effective date or scheduled window is before the user question time, treat that target state as the best current interpretation unless later evidence contradicts it.
       - Mark it as CURRENT_CANDIDATE, not merely POSSIBLE_UPDATE.
       - Lead the final answer with this best current interpretation.
       - Then mention the older confirmed state as previous context if it helps explain the update.
       - Do not use this rule for plans to repair, donate, sell, discard, buy, research, compare, contact, or ask about something unless the user asks about that planned action.
    6. For current-state questions ("currently", "now", "today", "where is", "where do I keep", "what do I use", "what do I prefer", "what am I working on", etc.), the user question time is the anchor unless a clearer anchor is stated. Prefer the most recent relevant evidence describing that state at or before the anchor.
    7. If multiple temporally valid states exist and a later intended state has not yet reached its effective date/window, do not invent a resolution. Explain the state progression and identify the last confirmed state, the most recent intended state, and any remaining uncertainty.
    8. Prefer transparent timeline reasoning over false certainty.

    CONFLICT RESOLUTION RULE:
    If multiple memories describe different values for the same attribute, build a conflict table, compare dates, compare memory types, determine whether the newer memory supersedes the older memory, and only return REQUIRED_DATA when the conflict cannot be resolved from retrieved evidence.

    REQUIRED_DATA RULE:
    Return REQUIRED_DATA only when critical entities are missing, critical dates/values are missing, the requested count cannot be satisfied, or competing interpretations cannot be resolved from retrieved evidence.
    Do not return REQUIRED_DATA merely because multiple historical states exist. Explain the progression when that answers the question better.
    Do not return REQUIRED_DATA solely because a requested qualifier label is missing when a small, unambiguous set of matching completed targets and numeric values was retrieved. In that case, answer with a transparent caveat and the supported ledger.

    TEMPORAL ANCHOR RULE:
    - Use an explicit anchor from the user prompt when present: a date/time, "as of", "when I...", "at the time I...", "by the time I...", "before/after", "since/until", "from X to Y", or "between X and Y".
    - If the prompt asks about the relationship, order, or gap between multiple events, compare those event dates to each other; do not use user question time unless it is explicitly one of the events.
    - If a relative phrase like "ago", "currently", "now", "today", or "how long have I been" has no clearer event/date anchor in the prompt, use the user question time as the anchor.
    - If an "ago" question also names another event as a reference clause, use that named event as the anchor instead of user question time.
    - For "last N months" / "past N months" aggregation questions, use the question time as anchor and include the N calendar months ending at the question month unless the user gives an exact date range. For Feb 26, 2023, "last four months" includes November 2022, December 2022, January 2023, and February 2023.
    - If a memory only gives Month YYYY and that month overlaps the accepted range, include it unless contradictory evidence shows it is outside the range.

    Before the JSON response, write a <thinking> block with:
    1. Target: <short string description of the exact request>. target_nouns:{{noun,noun}}.
       target_nouns must contain only target category nouns from the user request, lowercase, single words, no actions, no relations, no time/order words. Good: {{fruit}}. Bad: {{fruit,eaten,order,date}}.
    2. Evidence table:
       Evidence | Date | Memory type | Relation match | Category match | Temporal role | Decision | Reason
       Before writing the table, collapse duplicate retrieved memories into unique evidence units by real-world target/event/state transition. Do not list the same unit repeatedly. If duplicates were retrieved, mention the duplicate count briefly in the Reason for the one merged row.
       The table has a hard limit of 25 rows total. Never write a 26th evidence row. Choose the 25 most relevant unique evidence units: all unique ACCEPT rows needed for the answer first, then only the most important REJECT rows needed to prevent category or scope mistakes. If more than 25 evidence units exist, write one short "omitted" phrase outside the table and immediately continue to step 3.
       Memory type must be one of: OBSERVED_STATE, REPORTED_STATE, EVENT, PLAN, INTENTION, PREFERENCE, STATE_UPDATE.
       Temporal role must be one of: CURRENT_CANDIDATE, HISTORICAL_STATE, POSSIBLE_UPDATE, CONFLICTING_STATE, IRRELEVANT.
       Decision must be ACCEPT or REJECT. ACCEPT only if the evidence matches the target noun/category, relation/action, and requested time/scope. REJECT sibling/nearby categories. For durable current-possession counts, do not reject older owned/acquired/used/setup objects solely because the memory is older or uses past-tense wording unless later evidence removes them. For count/times questions, merge duplicate descriptions of the same real-world event into one evidence row; do not spend table rows listing duplicate rejects.
    3. Accepted targets: write only ACCEPT rows here. target_seen:{{Exact Name A,Exact Name B}}. Rejected near-misses: list only the important rejected categories or scopes briefly; do not repeat duplicates.
    4. Timeline analysis:
       Build a chronological timeline using only ACCEPT rows. For each state transition identify the previous state, later state, and whether the later state is confirmed, inferred, planned, or uncertain. If multiple states exist, explain how they relate instead of selecting one arbitrarily.
       If a separate same-update support row provides the planned effective date/window, connect it to the concrete target state in the timeline.
    5. Anchor:
       Identify the temporal reference point. For current-state questions, use the user question time unless a clearer anchor is stated. Determine the best current interpretation first, then the last confirmed previous state, then any later intended/proposed state. If a planned effective date/window is before the anchor, classify the planned target state as CURRENT_CANDIDATE and use it first unless contradicted.
    6. Calculation or resolution:
       Use only ACCEPT rows. For ordering/counting, sort/count only unique ACCEPT rows by exact event date and requested scope after deduplicating repeated mentions of the same event. For knowledge updates, preferences, opinions, locations, possessions, projects, jobs, relationships, and other stateful questions, resolve the latest supported state at the anchor and include important historical/intended-state context after the direct answer when needed to avoid false certainty.
    7. Final check:
       Verify every factual claim comes from ACCEPT evidence, all conflicts were analyzed, state transitions were considered, no rejected evidence appears as the answer, current-state answers begin with the anchor-time current interpretation, uncertainty is explicitly stated when present, and REQUIRED_DATA is used only when evidence is genuinely insufficient.
       The final answer must be exactly the ACCEPT rows after strict filtering. Never include a REJECT row in target_seen, timeline, calculation, or final answer. If target_seen contains a rejected item, or if the timeline/calculation contains more or fewer accepted items than the final answer, redo the filtering before answering.
    8. Conclusion: state the answer you will put in agent_response or state that REQUIRED_DATA is needed.

    Neutral target format example:
    User asks: "Where do I currently keep my old notebooks?"
    Retrieved context contains an older confirmed location and a newer intended storage update.
    Correct thinking format:
    <thinking>
    1. Target: current storage location of old notebooks. target_nouns:{{notebook}}
    2. Evidence table:
       old notebooks under desk | Jan 1 | REPORTED_STATE | storage location yes | notebook yes | HISTORICAL_STATE | ACCEPT | confirmed older location
       old notebooks in shelf box | Jan 5 | INTENTION | storage location yes | notebook yes | POSSIBLE_UPDATE | ACCEPT | newer intended storage location
       shelf organization planned | Jan 5 | PLAN | effective window support yes | same notebook storage update | CURRENT_CANDIDATE | ACCEPT | supplies passed effective date for notebook storage update
       old magazines in shelf box | Jan 6 | REPORTED_STATE | storage location yes | notebook no | IRRELEVANT | REJECT | magazine, not notebook
    3. Accepted targets: target_seen:{{old notebooks under desk,old notebooks in shelf box,shelf organization planned}}. Rejected near-misses: old magazines in shelf box.
    4. Timeline analysis: Jan 1 confirmed notebooks under desk. Jan 5 user intended to move/store them in shelf box; no later evidence confirms execution, but it is the latest same-target storage update.
    5. Anchor: user asks current location at question time. Jan 5 is before the question anchor, so the shelf-box plan is the best current interpretation; under desk is previous context.
    6. Calculation or resolution: answer first with the newer shelf-box target state, then mention the older under-desk state as previous context.
    7. Final check: all claims come from accepted notebook evidence; the answer leads with the anchor-time interpretation and preserves prior-state context.
    8. Conclusion: answer with the current interpretation first, then the timeline context.
    </thinking>

    After </thinking>, return only this JSON shape:
    {{
        "agent_response": "final answer for the user",
        "flags": [],
        "hyde_queries": []
    }}

    If more data is required, return:
    {{
        "agent_response": "",
        "flags": ["REQUIRED_DATA"],
        "hyde_queries": ["query one", "query two", "query three"]
    }}

    {preference_answer_hint}
    {knowledge_update_answer_hint}
    {assistant_recall_answer_hint}
    {benchmark_span_aggregation_hint}

    User question at this time ({question_time_str}):
    {state["user_prompt"]}
    """
    if not is_benchmark_channel:
        final_user_prompt = final_user_prompt.replace(
            "    Deduce the response for the user question using only the retrieved context below.",
            "    Answer the user question. Use retrieved context for user-specific memory facts, and use your own reasoning and knowledge for general conversation or advice.",
            1,
        )
        final_user_prompt = final_user_prompt.replace(
            "    - Do not use outside knowledge, assumptions, guesses, or hallucinated facts.",
            """    - For user-specific memory facts, do not use outside assumptions: ground the answer in ACCEPT evidence from retrieved context or say the memory is insufficient.
    - For general questions/advice/conversation, retrieved context is optional personalization, not a hard limit; answer normally from your own reasoning and knowledge.
    - Do not return REQUIRED_DATA just because no personal memory is needed for a general question.""",
            1,
        )
        final_user_prompt = final_user_prompt.replace(
            "    Before the JSON response, write a <thinking> block with:",
            f"    {normal_general_question_hint}\n\n    Before the JSON response, write a <thinking> block with:",
            1,
        )

    debug_print("user prompt:")

    debug_print(f"""User question at this time ({question_time_str}):
    {state["user_prompt"]} """)

    selected_skills = state.get("selected_skills", [])

    if state.get("skill_markdown"):
        system_instruction += (
            f"\n\n[ACTIVE SKILL INSTRUCTIONS]:\n{state['skill_markdown']}"
        )
        system_instruction += """

    [ACTIVE TOOL USE REQUIREMENT]
    The current request has already been routed to one or more active skills. You must call the relevant bound tool before giving the final answer. Do not answer from memory, assumptions, or an apology about missing access until the tool result confirms what happened. If the tool returns an auth or connect link, give that link to the user as the next step.
        """.rstrip()

    messages = (
        [SystemMessage(content=system_instruction)]
        + list(state["chat_history"])
        + [HumanMessage(content=final_user_prompt)]
    )

    last_response = None
    is_memory_ingestion = is_memory_ingestion_prompt(state["user_prompt"])

    if is_memory_ingestion:
        debug_print("🧠 [Ingestion] Memory review prompt detected; skipping generation LLM.")
        last_response = AIMessage(
            content=json.dumps(
                {
                    "agent_response": MEMORY_INGESTION_ACK,
                    "flags": [],
                    "hyde_queries": [],
                }
            )
        )

    elif selected_skills:

        tools_list = []
        runtime_config = {
            "user_id": state.get("user_id"),
            "channel_type": state.get("channel_type"),
        }
        for skill in selected_skills:
            skill_data = tool_manager.load_skill(skill, runtime_config=runtime_config)
            tools_list.extend(skill_data["tools"])

        forced_tool_llm = gen_llm.bind_tools(tools_list, tool_choice="any")
        llm_with_tools = gen_llm.bind_tools(tools_list)

        # --- NEW: ReAct Loop ---
        MAX_ITERATIONS = 5  # Prevent infinite loops if the LLM gets stuck
        iteration = 0

        while iteration < MAX_ITERATIONS:

            # --- Pass 1: Intent & Initial Tool Call ---
            active_llm = forced_tool_llm if iteration == 0 else llm_with_tools
            response = await active_llm.ainvoke(messages)

            # Track Tokens
            m = get_token_metrics(response)
            in_tokens += m["input"]
            out_tokens += m["output"]
            last_response = response

            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # If the LLM didn't call any tools, it means it has formulated its final answer. Break the loop.
                break

            # CRITICAL: Must append the LLM's request to history before executing
            messages.append(response)

            tool_msgs = []
            for call in tool_calls:
                fn = next((t for t in tools_list if t.name == call["name"]), None)
                display_tool_name = public_tool_display_name(call["name"])
                if fn:
                    try:
                        await report_job_status(
                            state,
                            "tool",
                            f"Using tool: {display_tool_name}",
                            {
                                "tool_name": call["name"],
                                "tool_status": "running",
                                "loop": iteration + 1,
                            },
                        )
                        debug_print(
                            f"🔧 [Loop {iteration+1}] Executing Tool: {display_tool_name}..."
                        )

                        # 🚀 NEW: CACHE INVALIDATION INTERCEPTOR
                        # If the agent uses the identity update tool, wipe the cache!
                        if call["name"] == "update_agent_identity":
                            debug_print(
                                "🔄 [Cache] Agent identity updated. Invalidating config cache."
                            )
                            AGENT_CONFIG_CACHE = None

                        # --- NEW: SECURE CONTEXT INJECTION ---
                        # Pass backend variables securely without exposing them to the LLM
                        run_config = {
                            "configurable": {
                                "user_id": state.get("user_id"),
                                "channel_type": state.get("channel_type"),
                            }
                        }

                        result = fn.invoke(call["args"], config=run_config)
                        await report_job_status(
                            state,
                            "tool",
                            f"Tool completed: {display_tool_name}",
                            {
                                "tool_name": call["name"],
                                "tool_status": "completed",
                                "loop": iteration + 1,
                            },
                        )
                    except Exception as e:
                        result = f"Error: {e}"
                        await report_job_status(
                            state,
                            "tool",
                            f"Tool failed: {display_tool_name}",
                            {
                                "tool_name": call["name"],
                                "tool_status": "failed",
                                "loop": iteration + 1,
                            },
                        )
                else:
                    result = "Tool not found."
                    await report_job_status(
                        state,
                        "tool",
                        f"Tool not found: {display_tool_name}",
                        {
                            "tool_name": call["name"],
                            "tool_status": "failed",
                            "loop": iteration + 1,
                        },
                    )

                tool_msgs.append(
                    ToolMessage(content=str(result), tool_call_id=call["id"])
                )

            messages.extend(tool_msgs)

            iteration += 1

        # --- FIX FOR PROBLEM 2: FORCE TEXT GENERATION ON TIMEOUT ---
        if iteration == MAX_ITERATIONS:
            print(
                "⚠️ ReAct loop reached maximum iterations. Forcing final text generation."
            )
            # Unbind tools by using the plain `llm` so it is FORCED to output text
            messages.append(
                SystemMessage(
                    content="SYSTEM INSTRUCTION: You have reached the maximum allowed tool execution limit. You must immediately provide a final text response to the user summarizing what you accomplished and what you couldn't finish. Do NOT attempt to call any more tools."
                )
            )

            final_response = await gen_llm.ainvoke(messages)

            m_final = get_token_metrics(final_response)
            in_tokens += m_final["input"]
            out_tokens += m_final["output"]
            last_response = final_response

    else:
        # Normal execution
        response = await gen_llm.ainvoke(messages)
        m = get_token_metrics(response)
        in_tokens = m["input"]
        out_tokens = m["output"]
        last_response = response

    # =====================================================================
    # 🛠️ NEW: JSON PARSING & DYNAMIC REQUIRED_DATA FALLBACK
    # =====================================================================
    raw_text = str(last_response.content) 
    json_str = extract_json_block(raw_text)


     # 🐞 ADD THIS DEBUG BLOCK 🐞
    debug_print("\n" + "="*50)
    debug_print("🐞 [DEBUG] RAW PASS 1 OUTPUT (WITH THINKING):")
    debug_print(last_response.content)
    debug_print("="*50 + "\n")
    
    final_output = ""
    flags = []
    dynamic_queries = []

    try:
        if json_str:
            data = json.loads(json_str)
            final_output = data.get("agent_response", "")
            flags = data.get("flags", [])
            dynamic_queries = data.get("hyde_queries", []) or []
        else:
            final_output = extract_pure_text(last_response)
    except Exception as e:
        print(f"⚠️ [JSON Parse Error] Falling back to raw text. {e}")
        final_output = extract_pure_text(last_response)

    target_queries = extract_target_queries_from_thinking(raw_text)
    if target_queries:
        debug_print("Target Queries:")
        debug_print(target_queries)

    if "REQUIRED_DATA" in flags and target_queries:
        dynamic_queries = _dedupe_keep_order(list(dynamic_queries) + target_queries)

    debug_print("New dynamic Queries:")
    debug_print(dynamic_queries)

    if "REQUIRED_DATA" in flags and dynamic_queries:
        debug_print(f"🔄 [Self-Correction] Agent requested specific data. Queries: {dynamic_queries}")
        
        # Run targeted search using the LLM's exact keyword pairs
        fb_sem_tasks = [semantic_db.search(sq, top_k=20, threshold=0.28) for sq in dynamic_queries]
        fb_epi_tasks = [episodic_db.search(sq, top_k=20, threshold=0.28) for sq in dynamic_queries]
        
        fb_sem_results = await asyncio.gather(*fb_sem_tasks)
        fb_epi_results = await asyncio.gather(*fb_epi_tasks)
        
        def fallback_dedupe(results_list):
            unique = {}
            for block in results_list:
                if not block: continue
                for line in block.split('\n'):
                    match = re.search(r"\[ID: ([0-9a-fA-F\-]{36})\]", line)
                    if match: unique[match.group(1)] = line
            return "\n".join(unique.values())
        
        # 🛠️ FIXED: Pass both the new results AND the old context into the deduplicator!
        combined_sem = fallback_dedupe(fb_sem_results + [sem_ctx_to_save])
        combined_epi = fallback_dedupe(fb_epi_results + [epi_ctx_to_save])

        new_sem_ctx = sort_memories_by_recency(combined_sem, max_lines=70)
        new_epi_ctx = sort_memories_by_recency(combined_epi, max_lines=70)

        # Update context variables for state tracking
        sem_ctx_to_save = new_sem_ctx
        epi_ctx_to_save = new_epi_ctx

        fallback_semantic_context = format_memory_context_for_prompt(
            sem_ctx_to_save, keep_ids=False
        )
        fallback_episodic_context = format_memory_context_for_prompt(
            epi_ctx_to_save, keep_ids=False
        )
        fallback_procedural_context = format_memory_context_for_prompt(
            state.get("procedural_context", ""), keep_ids=False
        )


        # Force the LLM to answer using the expanded context
        override_command = """
CRITICAL OVERRIDE: You are currently in FALLBACK mode. You have been provided with the expanded database search you requested. You are FORBIDDEN from using the "REQUIRED_DATA" flag again. 

FINAL VERIFICATION (STRICT ENTITY ISOLATION): Look at the expanded data provided. Does it contain the EXACT required variables explicitly linked to the specific target requested by the user?
- EAGER MATCHING BAN: You are STRICTLY FORBIDDEN from taking data (dates, numbers, facts) attached to a generic, unnamed, or broadly described entity and applying it to a highly specific Proper Noun or named entity. Even if contextual clues strongly suggest they might be the same thing, you cannot merge them to force an answer.
- CATEGORY MISMATCH DISCLOSURE: For source/giver questions only, if no exact category match exists but exactly one date-matched received/got/acquired/was-given event exists with an explicit source person, answer transparently with the actual item and source. Do not claim the item belongs to the user's requested category.
- PARTIAL QUALIFIER DISCLOSURE: For aggregation questions only, if exact requested qualifier labels are missing but a small, unambiguous set of matching completed targets has the needed numeric values, do not claim the missing labels are proven. Answer transparently with what was actually found, the available dates/values, and the supported total.
- If YES (the required data is explicitly attached to the EXACT named target): Provide your final answer in the "agent_response" field.
- If NO (the exact target is STILL missing its required variables, or the variables are only attached to generic/unnamed entities, and PARTIAL QUALIFIER DISCLOSURE does not apply): You MUST explicitly state that the information provided is not enough to answer the question. Do not guess or assume.
"""

        new_user_prompt = f"""
You are answering after REQUIRED_DATA fallback with expanded retrieved context.

{override_command}

Deduce the response for the user question using only the expanded retrieved context below.

Grounding rules:
- Do not use outside knowledge, assumptions, guesses, or hallucinated facts.
- Bind the exact requested target before selecting evidence: answer type, requested count/list size, exact noun/category, qualifiers, relation/action, and time/scope.
- Accept only entities whose actual name or type matches the requested target noun/category. Reject sibling or nearby categories.
- Do not broaden or substitute categories unless the user asks for suggestions, recommendations, opinions, ideas, examples, or related items. For example: museum != gallery, fruit != vegetable, chocolate bar type != candy, hotel != restaurant, shirt != all clothing.
- For category-specific list/order/count questions, a candidate must either contain the requested category in its name/type or be explicitly classified as that category by the memory. Do not infer category membership from cultural, travel, tour, venue, historical, educational, or recommendation context alone.
- For exact-category questions, if a candidate's own name or stated type contains a sibling category noun that is not the requested category, reject it even if the activity/relation resembles the requested category. If the requested category is one noun, a different venue/type noun is not accepted unless the memory explicitly states it belongs to the requested category.
- If the user asks for exactly N targets, first filter and deduplicate by exact category, completed relation/action, and requested scope. If exactly N valid targets remain, answer with all N valid targets and do not replace any valid target with a near-miss. If more or fewer than N valid targets remain after strict filtering, explain the supported ledger or request more data according to the REQUIRED_DATA rule; never choose the first N from a mixed list that still contains rejected near-misses.
- Any entity written under rejected near-misses is banned from calculation and from being stated as the direct answer.
- Use narrative dates inside memory content as event/state dates. Use ordinal memory labels only as storage-recency hints, not as historical event dates.
- For event ordering, separate event dates from report dates. Phrases such as "reported having", "mentioned having", "recalled", "recently", or "as of" often mark when the event was discussed, not when it happened. When several memories describe the same real-world event, use the earliest exact completed-event date directly tied to that event; do not move the event later because it was re-mentioned later.
- Preserve the memory's action/verb semantics exactly. Do not convert, promote, or complete one stated action into another stronger or different action unless the memory explicitly states that stronger action happened.
- For state updates, a supporting date/window memory may be used if it is clearly part of the same target update chain, even if it does not repeat the target noun. Example: "store old notebooks in a shelf box" plus "organize the shelf on Jan 5" can supply Jan 5 as the effective window for the shelf-box update. Do not use unrelated same-category plans this way.
- For current-state questions, the first sentence of agent_response must answer the current state at the anchor. Put previous/historical states after that, not before.
- For ordered sequences, games, logs, score sheets, or notation, exact recorded sequence items are valid evidence. If the question asks what came after an anchor and an ACCEPT row explicitly contains that anchor followed by the next item, answer with that next item.
- For monetary, count, duration, or total aggregation questions, build a complete ledger of every ACCEPT row with its date, target, and numeric value before calculating. Do not answer from only the most recent or most salient row. Sum/count every unique ACCEPT row in scope, including older rows within the requested window.
- For numeric comparison questions with multiple candidate values for the same side of the comparison, separate user-provided/recalled values and direct corrections of those values from assistant-generated general estimates or alternative-option estimates. Prefer the user-specific value or its direct correction when it matches the requested option. Treat broader, newer, bundled, or assistant-only estimates as caveat/competing evidence unless the user explicitly asks for that combined option or later adopts that estimate.
- For numeric difference, savings, or comparison questions, if the accepted evidence produces a bounded range, include both the supported range and a natural midpoint/average or rounded estimate when that estimate is meaningful. Do not average open-ended, categorical, asymmetric, or incompatible quantities.
- For count/times aggregation questions, deduplicate repeated memories that describe the same real-world event before counting. Count unique completed events, not every retrieved mention or summary of the same event.
- For list/order/count questions, if a later memory repeats a completed event using report wording and an earlier memory gives a more direct completed-event date for the same target/action, merge them as one event under the direct event date.
- For count/times aggregation questions, a completed attempt still counts even if the result was poor, failed, disappointing, or later retried, unless the user explicitly asks only for successful outcomes.
- For state-change aggregation questions, accept evidence that preserves the same real-world transition even when it is expressed through old/source state plus new/result state rather than the exact wording in the question. Keep each changed target distinct; do not count generic advice, future plans, or unrelated ownership as a completed transition.
- For current possession/count questions about durable objects, treat earlier owned/acquired/used/setup objects as still current unless later evidence says they were sold, discarded, returned, replaced, transferred away, or otherwise no longer owned/kept by the user.
- For compound target categories with modifiers, bind the modifier as part of the category. Accept only evidence explicitly linked to the modified category or its normal function/context. Reject nearby items when the modifier is not supported. Do not infer a domain/location modifier merely because an item could plausibly be associated with that domain.
- For aggregation questions where exact requested qualifiers are partially missing but the retrieved context contains a small, unambiguous set of matching completed targets with the needed numeric values, do not refuse solely because the qualifier label is missing. Answer transparently: state that the requested qualifier was not found/preserved, list the matching completed targets actually found with their available dates/values, and then provide the supported total. Do not invent missing qualifiers.

MEMORY STATE RESOLUTION RULE:
Retrieved memories may describe states, events, observations, plans, intentions, preferences, beliefs, relationships, possessions, locations, activities, or updates over time.
When multiple memories refer to the same entity, attribute, object, situation, preference, project, relationship, possession, or location:
1. Construct a timeline using the narrative dates found in memory content.
2. Do not automatically discard older memories because newer memories exist.
3. Classify each relevant memory as exactly one of: OBSERVED_STATE, REPORTED_STATE, EVENT, PLAN, INTENTION, PREFERENCE, STATE_UPDATE.
4. Identify whether later memories confirm an earlier state, modify it, contradict it, replace it, or merely discuss a future possibility.
5. Do not assume that a PLAN or INTENTION was executed by default. However, for current-state questions, if a later same-target PLAN/INTENTION gives a concrete target state/location/value and its planned effective date or scheduled window is before the user question time, treat that target state as the best current interpretation unless later evidence contradicts it.
   - Mark it as CURRENT_CANDIDATE, not merely POSSIBLE_UPDATE.
   - Lead the final answer with this best current interpretation.
   - Then mention the older confirmed state as previous context if it helps explain the update.
   - Do not use this rule for plans to repair, donate, sell, discard, buy, research, compare, contact, or ask about something unless the user asks about that planned action.
6. For current-state questions ("currently", "now", "today", "where is", "where do I keep", "what do I use", "what do I prefer", "what am I working on", etc.), the user question time is the anchor unless a clearer anchor is stated. Prefer the most recent relevant evidence describing that state at or before the anchor.
7. If multiple temporally valid states exist and a later intended state has not yet reached its effective date/window, do not invent a resolution. Explain the state progression and identify the last confirmed state, the most recent intended state, and any remaining uncertainty.
8. Prefer transparent timeline reasoning over false certainty.

CONFLICT RESOLUTION RULE:
If multiple memories describe different values for the same attribute, build a conflict table, compare dates, compare memory types, determine whether the newer memory supersedes the older memory, and state the best-supported resolution. Because this is fallback mode, if the conflict still cannot be resolved, say the information is not enough.

TEMPORAL ANCHOR RULE:
- Use an explicit anchor from the user prompt when present: a date/time, "as of", "when I...", "at the time I...", "by the time I...", "before/after", "since/until", "from X to Y", or "between X and Y".
- If the prompt asks about the relationship, order, or gap between multiple events, compare those event dates to each other; do not use user question time unless it is explicitly one of the events.
- If a relative phrase like "ago", "currently", "now", "today", or "how long have I been" has no clearer event/date anchor in the prompt, use the user question time as the anchor.
- If an "ago" question also names another event as a reference clause, use that named event as the anchor instead of user question time.
- Because this is fallback mode, do not return REQUIRED_DATA again. If critical evidence is still missing, state in agent_response that the information is not enough.
- For "last N months" / "past N months" aggregation questions, use the question time as anchor and include the N calendar months ending at the question month unless the user gives an exact date range. For Feb 26, 2023, "last four months" includes November 2022, December 2022, January 2023, and February 2023.
- If a memory only gives Month YYYY and that month overlaps the accepted range, include it unless contradictory evidence shows it is outside the range.

Before the JSON response, write a <thinking> block with:
1. Target: <short string description of the exact request>. target_nouns:{{noun,noun}}.
   target_nouns must contain only target category nouns from the user request, lowercase, single words, no actions, no relations, no time/order words. Good: {{fruit}}. Bad: {{fruit,eaten,order,date}}.
2. Evidence table:
   Evidence | Date | Memory type | Relation match | Category match | Temporal role | Decision | Reason
   Before writing the table, collapse duplicate retrieved memories into unique evidence units by real-world target/event/state transition. Do not list the same unit repeatedly. If duplicates were retrieved, mention the duplicate count briefly in the Reason for the one merged row.
   The table has a hard limit of 25 rows total. Never write a 26th evidence row. Choose the 25 most relevant unique evidence units: all unique ACCEPT rows needed for the answer first, then only the most important REJECT rows needed to prevent category or scope mistakes. If more than 25 evidence units exist, summarize the omitted duplicates or near-misses in one short phrase outside the table instead of enumerating them.
   Memory type must be one of: OBSERVED_STATE, REPORTED_STATE, EVENT, PLAN, INTENTION, PREFERENCE, STATE_UPDATE.
   Temporal role must be one of: CURRENT_CANDIDATE, HISTORICAL_STATE, POSSIBLE_UPDATE, CONFLICTING_STATE, IRRELEVANT.
   Decision must be ACCEPT or REJECT. ACCEPT only if the evidence matches the target noun/category, relation/action, and requested time/scope. REJECT sibling/nearby categories. For durable current-possession counts, do not reject older owned/acquired/used/setup objects solely because the memory is older or uses past-tense wording unless later evidence removes them. For count/times questions, either merge duplicate descriptions of the same real-world event into one ACCEPT row or mark later duplicates as REJECT with duplicate stated as the reason.
3. Accepted targets: write only ACCEPT rows here. target_seen:{{Exact Name A,Exact Name B}}. Rejected near-misses: list only the important rejected categories or scopes briefly; do not repeat duplicates.
4. Timeline analysis:
   Build a chronological timeline using only ACCEPT rows. For each state transition identify the previous state, later state, and whether the later state is confirmed, inferred, planned, or uncertain. If multiple states exist, explain how they relate instead of selecting one arbitrarily.
   If a separate same-update support row provides the planned effective date/window, connect it to the concrete target state in the timeline.
5. Anchor:
   Identify the temporal reference point. For current-state questions, use the user question time unless a clearer anchor is stated. Determine the best current interpretation first, then the last confirmed previous state, then any later intended/proposed state. If a planned effective date/window is before the anchor, classify the planned target state as CURRENT_CANDIDATE and use it first unless contradicted.
6. Calculation or resolution:
   Use only ACCEPT rows. For ordering/counting, sort/count only unique ACCEPT rows by exact event date and requested scope after deduplicating repeated mentions of the same event. For knowledge updates, preferences, opinions, locations, possessions, projects, jobs, relationships, and other stateful questions, resolve the latest supported state at the anchor and include important historical/intended-state context after the direct answer when needed to avoid false certainty.
7. Final check:
   Verify every factual claim comes from ACCEPT evidence, all conflicts were analyzed, state transitions were considered, no rejected evidence appears as the answer, current-state answers begin with the anchor-time current interpretation, uncertainty is explicitly stated when present, and insufficiency is stated only when evidence is genuinely insufficient.
   The final answer must be exactly the ACCEPT rows after strict filtering. Never include a REJECT row in target_seen, timeline, calculation, or final answer. If target_seen contains a rejected item, or if the timeline/calculation contains more or fewer accepted items than the final answer, redo the filtering before answering.
8. Conclusion: state the answer you will put in agent_response or state that the information is not enough.

Neutral target format example:
User asks: "Where do I currently keep my old notebooks?"
Expanded context contains an older confirmed location and a newer intended storage update.
Correct thinking format:
<thinking>
1. Target: current storage location of old notebooks. target_nouns:{{notebook}}
2. Evidence table:
   old notebooks under desk | Jan 1 | REPORTED_STATE | storage location yes | notebook yes | HISTORICAL_STATE | ACCEPT | confirmed older location
   old notebooks in shelf box | Jan 5 | INTENTION | storage location yes | notebook yes | POSSIBLE_UPDATE | ACCEPT | newer intended storage location
   shelf organization planned | Jan 5 | PLAN | effective window support yes | same notebook storage update | CURRENT_CANDIDATE | ACCEPT | supplies passed effective date for notebook storage update
   old magazines in shelf box | Jan 6 | REPORTED_STATE | storage location yes | notebook no | IRRELEVANT | REJECT | magazine, not notebook
3. Accepted targets: target_seen:{{old notebooks under desk,old notebooks in shelf box,shelf organization planned}}. Rejected near-misses: old magazines in shelf box.
4. Timeline analysis: Jan 1 confirmed notebooks under desk. Jan 5 user intended to move/store them in shelf box; no later evidence confirms execution, but it is the latest same-target storage update.
5. Anchor: user asks current location at question time. Jan 5 is before the question anchor, so the shelf-box plan is the best current interpretation; under desk is previous context.
6. Calculation or resolution: answer first with the newer shelf-box target state, then mention the older under-desk state as previous context.
7. Final check: all claims come from accepted notebook evidence; the answer leads with the anchor-time interpretation and preserves prior-state context.
8. Conclusion: answer with the current interpretation first, then the timeline context.
</thinking>

After </thinking>, return only this JSON shape:
{{
    "agent_response": "final answer for the user",
    "flags": [],
    "hyde_queries": []
}}

Expanded retrieved context:
Within each memory block, memories are listed newest-to-oldest by storage recency.
[Memory 1 - newest] is the newest retrieved memory in that block; the line marked oldest is the oldest.

[SEMANTIC - User Facts]
{fallback_semantic_context}

[EPISODIC - Past Events]
{fallback_episodic_context}

[PROCEDURAL - Strict Rules]
{fallback_procedural_context}

{attachment_prompt_block}

{preference_answer_hint}
{knowledge_update_answer_hint}
{assistant_recall_answer_hint}
{benchmark_span_aggregation_hint}

User question at this time ({question_time_str}):
{state["user_prompt"]}
"""
        new_messages = [SystemMessage(content=system_instruction)] + list(state["chat_history"]) + [HumanMessage(content=new_user_prompt)]
        
        debug_print("🧠 [Self-Correction] New context loaded. Re-generating JSON response...")
        fallback_response = await gen_llm.ainvoke(new_messages)
        
        # Update metrics
        m_fb = get_token_metrics(fallback_response)
        in_tokens += m_fb["input"]
        out_tokens += m_fb["output"]

        # 🐞 ADD THIS DEBUG BLOCK 🐞
        debug_print("\n" + "="*50)
        debug_print("🐞 [DEBUG] RAW PASS 2 (FALLBACK) OUTPUT:")
        debug_print(fallback_response.content)
        debug_print("="*50 + "\n")
        
        # Parse the Fallback JSON
        fb_json_str = extract_json_block(str(fallback_response.content))
        try:
            if fb_json_str:
                fb_data = json.loads(fb_json_str)
                final_output = fb_data.get("agent_response", "")
            else:
                final_output = extract_pure_text(fallback_response)
        except Exception:
            final_output = extract_pure_text(fallback_response)

    # Clean up any residual markdown or thinking blocks inside the agent_response string
    final_output = re.sub(r"<thinking>.*?</thinking>", "", final_output, flags=re.DOTALL).strip()

    if not final_output:
        final_output = "I have processed that request using my tools, but I don't have a specific summary to display. Please let me know if you need anything else."

    await report_job_status(state, "finalizing", "Finalizing response.")

    if state["channel_type"] != "terminal":
        print(f"Agent Response :{final_output}")

    end_time = time.time()
    print(
        f"\n⏱️ [Metrics] Generation Time: {end_time - start_time:.2f}s | Tokens: In({in_tokens}) Out({out_tokens})"
    )

    current_metrics = state.get("metrics", {})
    current_metrics.update({"generation_in": in_tokens, "generation_out": out_tokens})

    # 🚀 NEW: FIRE BACKGROUND LEARNING TO REDUCE LATENCY
    if not state.get("skip_learning", False):

        is_ingestion_learning = is_memory_ingestion_prompt(state["user_prompt"])
        run_ingestion_inline = (
            state.get("channel_type") == "benchmark" and is_ingestion_learning
        )
        
        # Define a safe wrapper that respects the 4-task limit
        async def bounded_learning():
            if is_ingestion_learning:
                async with INGESTION_LEARNING_LOCK:
                    await background_memory_update(
                        state["user_prompt"],
                        final_output,
                        sem_ctx_to_save,  # 🛠️ Pass updated Semantic context here
                        epi_ctx_to_save,  # 🛠️ Pass updated Episodic context here
                        state.get("procedural_context", ""),
                        state.get("current_date")
                    )
                return

            async with LEARNING_SEMAPHORE:
                await background_memory_update(
                    state["user_prompt"],
                    final_output,
                    sem_ctx_to_save,  # 🛠️ Pass updated Semantic context here
                    epi_ctx_to_save,  # 🛠️ Pass updated Episodic context here
                    state.get("procedural_context", ""),
                    state.get("current_date")
                )

        if run_ingestion_inline:
            debug_print("🧠 [Benchmark Ingestion] Running learning inline before returning response.")
            await bounded_learning()
        else:
            # Fire-and-forget, but track it in the global set
            task = asyncio.create_task(bounded_learning())
            PENDING_LEARNING_TASKS.add(task)
            task.add_done_callback(PENDING_LEARNING_TASKS.discard)

    # 🛠️ RETURN THE UPDATED CONTEXTS SO LANGGRAPH STATE UPDATES EVERYWHERE
    return {
            "final_response": final_output, 
            "metrics": current_metrics,
            "semantic_context": sem_ctx_to_save,
            "episodic_context": epi_ctx_to_save
        }


# ==========================================
# 7. LEARNING NODE (Robust Extraction)
# ==========================================
@persistent_network_retry(initial_delay=3.0, max_delay=60.0, timeout=None) # 🛠️ NEVER DROP
async def learn_vector_memory(
    db: VectorMemoryManager,
    memory_type: str,
    user_prompt: str,
    ai_response: str,
    current_context: str,
    current_date: str = None, 
    related_context: str = "",
    return_action_results: bool = False,
    return_actions: bool = False,
    execute_mutations: bool = True,
) -> tuple[int, dict] | tuple[int, dict, list[dict]] | tuple[int, dict, list[dict], list[dict]]:
    """Extracts facts/episodes with strict isolation and an importance threshold."""

    if memory_type == "Semantic":
        definition = (
            "Atomic, standalone facts that build the user's core identity profile. These are permanent or long-term "
            "attributes (e.g., name, age, demographics, ethnicity, origins, job title, health/dietary needs, specific tech/creative preferences, relationships, core routines and many more such things)."
            "Extract any enduring, long-term information that defines WHO the user is, WHAT they do, or HOW they live."
        )
        instruction_extension = (
            "ATOMICITY & ENTITY-PRESERVATION RULE (CRITICAL): Separate unrelated facts, BUT NEVER sever relational links. "
            "NEVER drop or omit any proper nouns, names of people, or specific places mentioned by the user. If the user mentions a couple or multiple people (e.g., 'Jen and Tom', 'Rachel and Mike'), you MUST include ALL NAMES in the extracted memory. " # 🛠️ ADDED THIS SENTENCE
            "EVIDENCE-BOUND SEMANTIC SCOPE (CRITICAL): Extract durable user attributes, stable preferences, relationships, owned resources, current active statuses, tracked inventories, and unresolved obligations when directly supported by the user's message or conversation-history payload. "
            "Do not promote one-time completed historical events into standalone Semantic facts unless they establish a current durable state, current ownership, current inventory, active obligation, or ongoing endeavor. "
            "Do not infer locations, affiliations, ownership, completion, replacement, dates, counts, or event statuses from assistant suggestions or from merely related wording. "
            "If information reveals the user's background, origins, physical/mental traits, or strict lifestyle parameters (things that will likely still be true 5 years from now), extract it as a standalone fact. "
            "If a status involves a specific count, pending logistical requirement, current possession, or multi-stage endeavor, capture the specific quantity, item, and status. "
            "If the user says 'I am 22 and my friend Sarah likes destiny', issue TWO 'ADD' actions: 1. 'User is 22 years old.' 2. 'User has a friend named Sarah who is interested in destiny.' "
            "NEVER replace specific names with generic pronouns like 'a friend' or 'they'."
        )
        example = (
            '{ "action": "ADD", "content": "User\'s name is Sourav." }, '
            '{ "action": "ADD", "content": "User works at Argus Labs." }'
            '{ "action": "ADD", "content": "User goes to Blue Bottle Coffee." }, '
            '{ "action": "ADD", "content": "User uses a Nespresso machine at home." }, '
            '{ "action": "ADD", "content": "User\'s high school friend is named Sarah." }, '
        )
        exclusion_rule = "DO NOT extract temporary states, emotions, or the narrative of the conversation. DO NOT extract greetings, pleasantries, or basic conversational filler."
    else:
        definition = (
            "The narrative occurrence of the conversation. Focus on WHAT happened during the "
            "interaction (e.g., User introduced themselves, Agent provided code, User corrected a mistake)."
        )
        instruction_extension = (
            "EVENT-ONLY & ENTITY-PRESERVATION RULE (CRITICAL): Focus on the milestone achieved in the conversation. "
            "HOWEVER, you MUST preserve specific names, identities, dates, locations, and entities within the event. "
            "If a date is available in the interaction and is tied to a specific event, or can be resolved from the provided current system time for a direct present-day phrase, attach that date directly to the exact event being stored. "
            "Never store a named event as undated if the interaction provides a date for it. "
            "Do not duplicate facts that belong in Semantic memory. "
            "ASSISTANT-ANSWER FACT PRESERVATION (CRITICAL): If the assistant provided factual recall material that the user may later ask about, preserve the exact answer-bearing details. "
            "This includes study/article titles, journal names, author names, organizations, product names, numbers, counts, dates, durations, prices, sample sizes, rankings, and quoted labels. "
            "Do not compress a list of studies, citations, recommendations, or examples into only a generic sentence like 'assistant provided examples'; keep each high-signal item with its numbers and names. "
            "EXACT ORDERED SEQUENCE PRESERVATION (CRITICAL): If the interaction contains a recorded game, score sheet, ordered notation, step list, timeline, numbered answer, route, or other sequence, preserve exact item numbers, symbols, labels, and adjacent transitions. "
            "For example, store the exact transition 'after item 27, item 28 was ...' rather than only 'the sequence continued'."
        )
        example = (
            '{ "action": "ADD", "content": "User introduced themselves for the first time and established their professional role." }'
            '{ "action": "ADD", "content": "User transitioned from exploring existential theory to actively applying meaning after a conversation with their friend Sourav." }'
            '{ "action": "ADD", "content": "User is doing xyz project about xxx" }'
        )
        exclusion_rule = "CRITICAL: DO NOT extract temporary states, greetings, or small talk. (Note: You ARE allowed and encouraged to include specific names, brands, or places if they are directly involved in the event)."

    # 🛠️ FIXED: Use the provided date
    if current_date:
        current_time = current_date
        # Extract just the year from the string if possible, else fallback to system year
        try:
            current_year = re.search(r"\d{4}", current_date).group()
        except Exception:
            current_year = datetime.now().year
    else:
        current_year = datetime.now().year 
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    prompt_current_time = format_current_time(current_time)

    ingestion_turn = is_memory_ingestion_prompt(user_prompt)
    source_user_prompt = unwrap_memory_ingestion_prompt(user_prompt)
    source_ai_response = (
        ""
        if ingestion_turn and str(ai_response).strip() == MEMORY_INGESTION_ACK
        else ai_response
    )

    structured_artifact_memories = []

    prompt_current_context = format_memory_context_for_prompt(
        current_context, keep_ids=True
    )
    prompt_related_context = format_memory_context_for_prompt(
        related_context, keep_ids=True
    )

    learning_prompt = f"""
    You are a Cognitive Memory Editor managing a {memory_type} database.
    Your job is to consolidate information by issuing ADD, UPDATE, or DELETE commands.

    DEFINITION OF {memory_type.upper()} MEMORY:
    {definition}

    {instruction_extension}

    CURRENT ACTIVE MEMORIES (With Database IDs):
    {prompt_current_context}

    CURRENT RELATED MEMORIES FROM OTHER MEMORY TYPES (READ-ONLY):
    {prompt_related_context}

    Current memories are listed newest-to-oldest by storage recency within each block.
    [Memory 1 - newest] is the newest memory in that block; the line marked oldest is the oldest.
    IDs are retained only so you can target UPDATE or DELETE actions.
    
    {"MEMORY INGESTION MODE: The original user prompt was only a wrapper asking the agent to review and remember conversation history. Extract memories only from the conversation-history payload below. Ignore the wrapper and ignore the fixed acknowledgement response." if ingestion_turn else ""}

    Interaction:
    {"Conversation history to remember" if ingestion_turn else "User"}: {source_user_prompt}
    This is AI response to user message: {source_ai_response if source_ai_response else "(fixed ingestion acknowledgement omitted)"}

    CRITICAL TIME RESOLUTION: The current system time is {prompt_current_time}. 
    If the user uses relative time words (e.g., "yesterday", "last month", "tomorrow"), you MUST convert them into absolute dates within the "content" string you generate.
    If the user mentions an event but doesn't specify the year (e.g., "in August", "for my birthday"), you MUST append the current year ({current_year}) to the date. Example: "in August" -> "in August {current_year}". 
    Example: If user says "I started a diet yesterday", store "User started a diet on [Calculated Date]."

    CURRENT TURN EVENT DATE ANCHORING:
    The current system time is the timestamp of the conversation being learned.
    If the original conversation text describes a specific user event as happening today, now, just now, earlier today, just completed, just came back from, just got back from, or returned from, and no explicit event date is written, use the current system date as the narrative date for that exact event.
    For Episodic memory, attach that date directly to the exact named event/entity in the same sentence.
    Do not use the current system date merely because the wrapper instruction says to review, remember, ingest, or summarize the conversation.

    VAGUE RECENCY HANDLING:
    Vague recency words such as "recently", "lately", "earlier", "before", "a while ago", and "in the past" are report-time context, not exact event dates by themselves.
    Unless the wording directly indicates same-day completion, return, or current completion, store vague recency as "As of [current system date], user reported recently ..." rather than "On [current system date], user did ...".
    If the user is still recovering from, still thinking about, or still dealing with a prior event, preserve the current date as the report/status date and preserve the prior event details without inventing its completion date.

    SPECIFIC RELATIVE CALENDAR RESOLUTION:
    Specific relative calendar expressions such as "yesterday", "today", "this morning", "last night", "last weekend", "last Sunday", "last Thursday", "last month", "earlier this week", and "next Friday" are resolvable temporal anchors.
    Resolve them to absolute dates using the current system time and preserve the resolved date on the exact event they modify.
    If a relative calendar expression names a multi-day period, such as "last weekend", "this weekend", "next weekend", "last month", or "earlier this week", preserve the resolved date range or period unless the text supplies a specific day inside that period.
    Do not collapse a multi-day period to one exact day without direct evidence.

    TRANSFER / ACQUISITION EVENT ANCHORING:
    When the user says they got, received, acquired, bought, found, inherited, borrowed, were given, picked up, started, joined, or completed something, treat that as a concrete event.

    If the acquisition/transfer/completion phrase uses a specific same-day or calendar phrase such as "today", "now", "just", "this morning", "yesterday", "last Sunday", or "this week", resolve it using the current system time and attach the resolved date or date period directly to the event.
    Do not treat vague recency words such as "recently" as exact event dates.

    Preserve all transfer anchors:
    - recipient/actor,
    - item/object,
    - source/giver/seller/place if present,
    - date/time,
    - provenance/background if present.

    Do not weaken a dated transfer event into only a timeless ownership fact.

    EVIDENCE BOUNDARY & ACTION LEDGER:
    Store facts that are explicitly stated by the user, plus direct resolutions of relative-time phrases using the current system time.
    Attach every date, count, price, status, location, and obligation only to the specific event, item, action, or clause it directly modifies.
    If one sentence mentions multiple roles or events, preserve the roles separately instead of converting one role into another.
    Relationship phrases may explain timing or context, but they are not standalone proof of the related event's exact date, status, or completion unless the text directly states that event.
    For each user event, action, current status, or unresolved obligation worth storing, include the actor, action verb, object, date/time when known, location/source/counterparty when known, and status/outcome when known.
    Keep distinct action roles distinct: ordering, arrival, pickup, return, exchange, replacement, repair, sale, visit, offer, rejection, start, finish, ownership, and donation are separate roles unless the text explicitly states one transition connecting them.

    CROSS-SESSION TEMPORAL LEDGER SUPPORT:
    When the user reports a dated start, finish, visit, participation, viewing, travel, ownership, replacement, repair, or completed attempt that could later be counted, ordered, compared, or used in duration reasoning, preserve a standalone event-level memory for that dated action.
    Preserve both endpoints for durations when the user states a start and a finish in different chunks; do not update one endpoint into a generic summary that loses the original date.
    Preserve unsuccessful but completed attempts, rejected offers, non-selected options, and evaluated options with their stated reasons when the user gives the reason.

    ACTIVE OBLIGATION EXTRACTION:
    If the user says they need to, still need to, have to, should, plan to, are going to, or intend to pick up, return, exchange, repair, replace, collect, drop off, or retrieve an item, store that as a current active obligation in Semantic memory and as an event/plan in Episodic memory.
    Use one memory per unresolved action. If one statement contains both returning an original item and picking up a replacement item, create separate action-level memories for each obligation.
    Each obligation memory must include the action verb, item, store/location/counterparty when stated, and current status.

    SAME-EVENT DATE CONFLICT HANDLING:
    Before adding a dated event, compare against current and related memories for the same user action and same object/item/title/place.
    If the new text appears to re-mention the same event and does not explicitly state that it is another separate occurrence, do not create a second event with a conflicting date.
    If the existing memory has an exact date and the new text supplies only a relative or repeated mention, preserve the existing exact date. Use UPDATE only if useful details are added without losing the existing date; otherwise return no action for that duplicate event.
    If the text clearly describes a separate repeated occurrence, store it as a separate event with its own date.

    STRICT EXTRACTION & CONSOLIDATION RULES:
    1. ATOMICITY: Each 'ADD' or 'UPDATE' action must contain exactly ONE independent piece of information.
    2. TYPE ISOLATION: {exclusion_rule}
    3. HIGH-FIDELITY: Preserve specific entities, brand names, and proper nouns (for Semantic). 
    4. NEW INFO: If new information is provided that doesn't exist, use "ADD".
    5. DUPLICATE & UPDATE DECISION:
    Before issuing an ADD, scan both CURRENT ACTIVE MEMORIES and CURRENT RELATED MEMORIES FROM OTHER MEMORY TYPES.

    Treat memories as the same real-world event/item when they share the same canonical anchors:
    - actor/entity,
    - action/property,
    - event/item/object,
    - date/time,
    - location,
    - named counterpart or organization.

    Ignore wording differences, memory type labels, and exact-vs-qualified numeric wording when deciding whether two memories refer to the same real-world event/item.

    If the same event/item already exists in the current memory type, use UPDATE instead of ADD.
    If the same event/item exists only in another memory type, do not create a conflicting numeric version. Preserve the most precise known numeric form if you create or update a memory in this memory type.

    
    6. CORRECTIONS, EVOLUTIONS & CONTEXTUAL MERGING (CRITICAL): 
       - If the user provides new details about an existing topic, use "UPDATE" with the ID of the old memory to combine the facts. 
       - If the user mentions an event, project, or item, and then provides specific details (e.g., duration, cost, names) about the same event role in subsequent sentences within the chunk, merge those details into a single comprehensive sentence.
       - Do not merge distinct event roles, distinct dated actions, or distinct obligations into one ambiguous sentence. Preserve separate action roles separately.
       - NOUN & REFERENT RECOVERY (CRITICAL): Users often use casual or incomplete grammar, leaving adjectives or quantifiers hanging without their noun (e.g., "the blue one", "the 400-page"). If you see a lone descriptor or measurement, you MUST look at the surrounding context to find the specific noun it describes and explicitly attach it when merging. Example: "I hated the 3-hour" -> "User hated the 3-hour movie." You MUST preserve all specific entities and names. Never delete a name or detail just because the user didn't repeat it in the next sentence!
    7. DATE PRESERVATION & EVENT ANCHORING (CRITICAL):
        If the interaction contains any explicit date, resolved relative date, weekday, month, year, or temporal phrase tied to an event, the extracted memory MUST preserve that date in the same sentence as the exact named event/entity.
        Do not store a named event without its date when the date is directly tied to that same event anywhere in the interaction.
        Do not move a date from one event, role, or clause onto another event, role, or clause.
        Do not move a date onto a generic summary if a specific named event is present.
        For Episodic memory, event memories should generally begin with the narrative date when one is known.
        If multiple events happen on different dates, create separate actions or an UPDATE that preserves each event with its own date.
        If an item/object is received, acquired, bought, found, inherited, borrowed, or given on a known date, the extracted memory must preserve that date with the acquisition/transfer verb, not only with later discussion, research, maintenance, or ownership.

    8. QUANTITATIVE FIDELITY:
    Preserve every number with its owner, measured action/property, event/item, and qualifier. Do not drop numbers during ADD or UPDATE.
    If the user's text contains a number or measurement (e.g., "5-hour", "$40", "3 weeks", "12 items"), that exact number must appear in the extracted memory. Resolve dangling numbers to their noun using surrounding context.
    Values in subordinate or relative clauses belong to the grammatical subject of that clause, not automatically to the user or nearest person.

    9. STATE-TRANSITION FIDELITY:
    If the interaction describes a change from a prior/source object, state, tool, place, habit, role, or plan into a later/result object, state, tool, place, habit, role, or plan, preserve the transition itself with both sides and the resolved date. Do not compress transitions into only passive ownership, current use, or a generic summary.
    If an old/source entity is removed, transferred away, retired, no longer used, or superseded while a new/result entity is acquired, used, installed, adopted, or otherwise becomes the successor, store that as one explicit old-to-new transition when the text supports the link.
    Replacement rule: If the user says they got, acquired, bought, received, installed, started using, or were gifted a new item and also got rid of, donated, threw away, retired, stopped using, or replaced an old item with the same function, learn it as "User replaced their old <old item> with <new item>," even if the word "replaced" is not explicitly used.
    The transition memory must name both the old/source item and the new/result item, and it must preserve giver/source, disposal destination, and date when stated.
    Do not leave the replacement relation only implicit across separate acquisition and disposal memories.
    Only infer replacement facts from the user's own message or conversation-history payload; do not learn assistant-suggested items as user-owned, replaced, fixed, acquired, or discarded items.
    If the user attempted and completed an activity but reports that the outcome was poor, failed, disappointing, or needed retrying, still preserve the attempted completed activity and its date. Do not drop it merely because it was unsuccessful.

    10. NUMERIC PRECISION CONSOLIDATION:
    For the same real-world event/item, prefer the most precise numeric form:
    exact unqualified value > qualified value ("about", "over", "at least") > range > vague quantity.

    Never replace an existing exact unqualified value with a less precise restatement.
    If new text repeats the same event/item with a less precise number and adds no new non-numeric details, output no action.
    If new text adds useful details, update the memory while preserving the most precise known number and the new details.

    General example:
    Old memory: "User bought a monitor for $300 on May 2, 2023."
    New text: "The monitor cost about $300 and has a USB-C hub."
    Correct update: "User bought a monitor for $300 on May 2, 2023; it was later described as costing about $300 and has a USB-C hub."
    Wrong update: "User bought a monitor for about $300."

    11. REVOCATION: If the user explicitly revokes information, use "DELETE" on the old ID.
    12. NO PREFIXES: Do not use "Semantic:" or "Episodic:" labels in the content.
    13. NO GREETINGS: Ignore "Hello", "How are you", etc.

    OUTPUT FORMAT:
    You must return a raw JSON object with an "actions" array. Return exactly `{{"actions": []}}` if no changes are needed.
    {{
        "actions": [
            {{ "action": "ADD", "content": "Fact or Event 1" }},
            {{ "action": "ADD", "content": "Fact or Event 2" }},
            {{ "action": "UPDATE", "id": "uuid", "content": "Updated Fact preserving old names" }},
            {{ "action": "DELETE", "id": "uuid" }}
        ]
    }}
    Example for this task:
    {example}
    """

    response = await learn_llm.ainvoke([HumanMessage(content=learning_prompt)])


    # 🛠️ FIXED: Use robust extractor
    content = extract_pure_text(response)

    # print("learning content:")
    # print(content)

    # print("data:",content, current_context)
    m = get_token_metrics(response)


    actions_executed = 0
    action_results: list[dict] = []
    parsed_actions: list[dict] = []

    if (
        (content and content.upper() != "NONE" and content != '{"actions": []}')
        or (memory_type == "Episodic" and structured_artifact_memories)
    ):
        try:
            # Isolate the JSON object explicitly
            json_str = extract_json_block(content, is_array=False)
            data = json.loads(json_str) if json_str else {"actions": []}
            actions = data.get("actions", [])

            

            if memory_type == "Episodic":
                existing_contents = {
                    str(act.get("content", "")).strip()
                    for act in actions
                    if isinstance(act, dict)
                }
                for memory in structured_artifact_memories:
                    if memory not in existing_contents:
                        actions.append({"action": "ADD", "content": memory})
                        existing_contents.add(memory)

            parsed_actions = [act for act in actions if isinstance(act, dict)]
            if execute_mutations:
                results = await db.execute_actions_with_results(parsed_actions)
                actions_executed += len(results)
                action_results.extend(results)

        except json.JSONDecodeError:
            # 2. FALLBACK PARSER: If the LLM just dumped raw text instead of JSON
            print(
                f"⚠️ [Warning] {memory_type} Editor returned raw text, falling back to ADD action."
            )

            # Strip accidental prefixes
            clean_content = re.sub(
                r"^(Semantic|Episodic):\s*",
                "",
                content,
                flags=re.IGNORECASE | re.MULTILINE,
            )

            lines = clean_content.split("\n")

            fallback_actions = [
                {
                    "action": "ADD",
                    "content": line.strip(),  # 🛠️ REMOVED THE TIMESTAMP INJECTION LOGIC HERE
                }
                for line in lines
                if line.strip()
            ]
            parsed_actions = fallback_actions
            if execute_mutations:
                results = await db.execute_actions_with_results(fallback_actions)
                actions_executed += len(results)
                action_results.extend(results)

        except Exception as e:
            print(f"❌ [Error] Memory Execution failed: {e}")
            raise e  # 🛠️ ADD THIS: Signal the decorator to retry the whole process

    if return_action_results and return_actions:
        return actions_executed, m, action_results, parsed_actions

    if return_action_results:
        return actions_executed, m, action_results

    if return_actions:
        return actions_executed, m, parsed_actions

    return actions_executed, m


@persistent_network_retry(initial_delay=3.0, max_delay=60.0, timeout=None)
async def learn_broad_episodic_memory(
    user_prompt: str,
    ai_response: str,
    current_context: str,
    current_date: str = None,
    related_context: str = "",
) -> tuple[int, dict]:
    """
    Store broader high-fidelity episodic capsules alongside atomic memories.

    Atomic learning is optimized for clean facts; this pass is optimized for
    recallability of exact details that users later ask about.
    """
    if current_date:
        current_time = current_date
        try:
            current_year = re.search(r"\d{4}", current_date).group()
        except Exception:
            current_year = datetime.now().year
    else:
        current_year = datetime.now().year
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    prompt_current_time = format_current_time(current_time)

    ingestion_turn = is_memory_ingestion_prompt(user_prompt)
    source_user_prompt = unwrap_memory_ingestion_prompt(user_prompt)
    source_ai_response = (
        ""
        if ingestion_turn and str(ai_response).strip() == MEMORY_INGESTION_ACK
        else ai_response
    )

    artifact_context = re.sub(r"\s+", " ", source_user_prompt).strip()
    artifact_context = artifact_context[:220] if artifact_context else "the conversation"
    artifact_source_text = f"{source_user_prompt}\n\n{source_ai_response}".strip()
    structured_units = extract_structured_artifact_memories(
        artifact_source_text,
        prompt_current_time,
        artifact_context,
    )
    structured_section = (
        "\n".join(f"- {m}" for m in structured_units)
        if structured_units
        else "None"
    )

    prompt_current_context = format_memory_context_for_prompt(
        current_context, keep_ids=True
    )
    prompt_related_context = format_memory_context_for_prompt(
        related_context, keep_ids=True
    )

    broad_prompt = f"""
    You are a Broad Episodic Memory Editor.

    Atomic memory extraction has a different job: it compresses the interaction into small facts.
    Your job is the complementary job: preserve a small number of broader, high-fidelity recall capsules so future questions can recover exact answer-bearing context.

    CURRENT SYSTEM TIME: {prompt_current_time}
    CURRENT YEAR: {current_year}

    CURRENT EPISODIC MEMORIES (With Database IDs):
    {prompt_current_context}

    RELATED MEMORY CONTEXT (READ-ONLY):
    {prompt_related_context}

    Current memories are listed newest-to-oldest by storage recency within each block.
    [Memory 1 - newest] is the newest memory in that block; the line marked oldest is the oldest.
    IDs are retained only so you can target UPDATE or DELETE actions.

    {"MEMORY INGESTION MODE: The original user prompt was only a wrapper asking the agent to review and remember conversation history. Store recall capsules only from the conversation-history payload below. Ignore the wrapper and ignore the fixed acknowledgement response." if ingestion_turn else ""}

    Interaction:
    {"Conversation history to remember" if ingestion_turn else "User"}: {source_user_prompt}
    This is AI response to user message: {source_ai_response if source_ai_response else "(fixed ingestion acknowledgement omitted)"}

    STRUCTURED EXACT UNITS:
    {structured_section}

    What to preserve:
    - Exact assistant-provided facts the user may later ask to recall.
    - Numbers, counts, sample sizes, dates, durations, prices, rankings, quantities, and units.
    - Names, titles, journals, organizations, products, places, people, labels, and quoted phrases.
    - Ordered sequences, move lists, timelines, routes, steps, score sheets, and adjacent transitions.
    - State changes with both prior/source and later/result sides, including unsuccessful but completed attempts.
    - Replacement transitions where the user's own message says they got/acquired/received/started using a new item and got rid of/donated/retired/stopped using an old item with the same function, even if the word "replaced" is not explicit.
    - User corrections and the corrected final state.
    - The surrounding question/answer context needed to understand what each exact detail belongs to.

    Evidence and temporal fidelity:
    - Attach dates, counts, prices, statuses, locations, and obligations only to the specific event, item, action, or clause they directly modify.
    - Keep distinct action roles distinct: ordering, arrival, pickup, return, exchange, replacement, repair, sale, visit, offer, rejection, start, finish, ownership, and donation are separate roles unless the text explicitly connects them as one transition.
    - Vague recency words such as "recently", "lately", "earlier", "before", "a while ago", and "in the past" are report-time context, not exact event dates by themselves.
    - Specific relative calendar expressions such as "yesterday", "today", "last weekend", "last Sunday", "last Thursday", "last month", and "next Friday" must be resolved from the current system time and preserved on the exact event they modify.
    - If a relative calendar expression names a multi-day period, preserve the resolved date range or period unless the text supplies a specific day inside that period.
    - Preserve dated starts, finishes, visits, participations, viewings, travels, ownership changes, replacements, repairs, rejected offers, non-selected options, and completed attempts as ledger-ready details when they could later be counted, ordered, compared, or used in duration reasoning.
    - If the user describes unresolved pickup, return, exchange, repair, replacement, collection, drop-off, or retrieval actions, preserve each unresolved action with the item, store/location/counterparty when stated, and current status.
    - If a new item/tool is acquired or started and an old item/tool with the same practical role is donated, discarded, removed, retired, or stopped, preserve an explicit replacement transition naming both sides.
    - Before adding a dated event, compare against current and related memories for the same user action and same object/item/title/place. Do not create a second conflicting dated event unless the text clearly describes a separate occurrence.

    How this differs from atomic learning:
    - You may store one broader memory containing several tightly related details from the same interaction.
    - Do not over-compress into vague summaries like "assistant discussed studies" or "the game continued".
    - Do not explode every sentence into tiny facts; preserve the smallest useful capsule that would answer future follow-up questions.
    - If a STRUCTURED EXACT UNIT is listed, preserve it as an ADD unless the same exact unit already exists.
    - Use UPDATE when an existing broad episode describes the same interaction/topic but lacks newly available exact details.
    - Use DELETE when an existing broad episode is clearly duplicated, superseded by a corrected broad episode, or explicitly revoked by the user.
    - Do not DELETE useful atomic-looking memories merely because you are adding a broader capsule. Delete only stale/duplicate broad capsules or explicitly revoked information.

    Return no more than 6 actions.
    Return exactly {{"actions": []}} for greetings, small talk, or interactions with no future recall value.

    OUTPUT FORMAT:
    {{
        "actions": [
            {{ "action": "ADD", "content": "On [date], user and assistant discussed ...; exact details preserved include ..." }},
            {{ "action": "UPDATE", "id": "uuid", "content": "Updated broader episode preserving old and new exact details." }},
            {{ "action": "DELETE", "id": "uuid" }}
        ]
    }}
    """

    response = await learn_llm.ainvoke([HumanMessage(content=broad_prompt)])
    content = extract_pure_text(response)
    m = get_token_metrics(response)
    actions_executed = 0

    try:
        json_str = extract_json_block(content, is_array=False)
        data = json.loads(json_str) if json_str else {"actions": []}
        actions = data.get("actions", [])
        actions = [act for act in actions if isinstance(act, dict)][:6]

        existing_contents = {
            str(act.get("content", "")).strip()
            for act in actions
            if isinstance(act, dict)
        }
        for memory in structured_units:
            if memory not in existing_contents:
                actions.append({"action": "ADD", "content": memory})
                existing_contents.add(memory)

        actions_executed += await episodic_db.execute_actions(actions)

    except json.JSONDecodeError:
        print("⚠️ [Warning] Broad Episodic Editor returned raw text; storing as broad capsule.")
        clean_content = re.sub(
            r"^(Broad Episodic|Episodic):\s*",
            "",
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()
        if clean_content:
            actions_executed += await episodic_db.execute_actions(
                [{"action": "ADD", "content": clean_content}]
            )
    except Exception as e:
        print(f"❌ [Error] Broad Episodic Learning failed: {e}")
        raise e

    return actions_executed, m


async def execute_procedural_action(act: dict) -> bool:
    act_type = act.get("action", "").upper()
    raw_id = act.get("id")

    record_id = None
    if raw_id:
        match = re.search(r"([0-9a-fA-F\-]{36})", raw_id)
        if match:
            record_id = match.group(1)

    def mutate_rules():
        rules = _load_rules_file()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if act_type == "DELETE" and record_id:
            new_rules = [r for r in rules if r.get("id") != record_id]
            _save_rules_file(new_rules)
            return len(new_rules) != len(rules)

        if act_type == "ADD" and act.get("rule"):
            rules.append(
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": AGENT_ID,
                    "rule": act.get("rule"),
                    "reasoning": act.get("reasoning", ""),
                    "target_entity": act.get("target_entity", "global"),
                    "tags": act.get("tags", ["general"]),
                    "created_at": now,
                    "updated_at": now,
                }
            )
            _save_rules_file(rules)
            return True

        if act_type == "UPDATE" and record_id and act.get("rule"):
            for rule in rules:
                if rule.get("id") == record_id:
                    rule["rule"] = act.get("rule")
                    rule["reasoning"] = act.get("reasoning", "")
                    rule["target_entity"] = act.get("target_entity", "global")
                    rule["tags"] = act.get("tags", ["general"])
                    rule["updated_at"] = now
                    _save_rules_file(rules)
                    return True

        return False

    return await asyncio.to_thread(mutate_rules)



@persistent_network_retry(initial_delay=3.0, max_delay=60.0, timeout=None) # 🛠️ NEVER DROP
async def learn_procedural_memory(
    user_prompt: str, ai_response: str, current_context: str
) -> tuple[int, dict]:
    """Extracts, updates, or deletes explicit behavioral rules with strict formatting constraints."""

    ingestion_turn = is_memory_ingestion_prompt(user_prompt)
    source_user_prompt = unwrap_memory_ingestion_prompt(user_prompt)
    source_ai_response = (
        ""
        if ingestion_turn and str(ai_response).strip() == MEMORY_INGESTION_ACK
        else ai_response
    )

    prompt_current_context = format_memory_context_for_prompt(
        current_context, keep_ids=True
    )

    learning_prompt = f"""
    You are a Cognitive Procedural Database Editor.
    Your job is to identify, consolidate, and clean up behavioral rules, personas, and formatting constraints.

    CRITICAL BOUNDARIES (WHAT NOT TO EXTRACT):
    1. DO NOT extract factual information about the user (e.g., their name, job, or projects). That belongs in Semantic Memory.
    2. DO NOT extract specific details of the current task (e.g., "The user needed an email about being late"). That belongs in Episodic Memory.
    3. ONLY extract a rule if the user expressed a general preference for HOW you should behave, format, or generate outputs in the future (e.g., "Always use a formal tone", "Never use emojis", "Format code in snake_case").

    ENTITY SPECIFICITY (CRITICAL RULE):
    - If the user specifies that a rule applies to a specific person (e.g., "Elias"), project, or context, you MUST explicitly include that name/context at the very beginning of the rule text (e.g., "When generating content for Elias..."). Set the "target_entity" field to this name.
    - If no target is specified, leave "target_entity" as "global".

    CURRENT ACTIVE RULES (With Database IDs):
    {prompt_current_context}

    Current rules are listed newest-to-oldest by storage recency.
    [Memory 1 - newest] is the newest rule; the line marked oldest is the oldest.
    IDs are retained only so you can target UPDATE or DELETE actions.
    
    {"MEMORY INGESTION MODE: The original user prompt was only a wrapper asking the agent to review and remember conversation history. Extract procedural rules only from the conversation-history payload below. Ignore the wrapper and ignore the fixed acknowledgement response." if ingestion_turn else ""}

    Interaction:
    {"Conversation history to remember" if ingestion_turn else "User"}: {source_user_prompt}
    This is AI response to user message: {source_ai_response if source_ai_response else "(fixed ingestion acknowledgement omitted)"}

    STRICT CONSOLIDATION RULES:
    1. If the user establishes a NEW rule, use "ADD".
    2. CONTRADICTION REMOVAL & LOSSY-UPDATE PREVENTION (CRITICAL): If the user corrects a past rule (e.g., "Stop calling me that" or fixing a typo like 'todasy' to 'today'), you MUST use "UPDATE" and provide the ID of the old rule. When updating, you MUST preserve all existing specific entities and examples from the old rule.  DO NOT "ADD" the correction, as leaving the old rule active will confuse the generation agent.
         - Example: If an old rule says "Avoid allergens (e.g., peanuts, dairy)" and the user adds "gluten", the updated rule MUST say "(e.g., peanuts, dairy, gluten)". Do not drop old examples unless explicitly revoked.
    3. If the user explicitly revokes a rule, use "DELETE" on the old ID.

    OUTPUT FORMAT:
    Return a raw JSON object with an "actions" array. Return exactly `{{"actions": []}}` if no changes are needed.
    {{
        "actions": [
            {{
                "action": "ADD", 
                "rule": "[Timestamp] New rule here.",
                "target_entity": "global",
                "tags": ["formatting"],
                "reasoning": "Why this rule was added."
            }},
            {{
                "action": "UPDATE", 
                "id": "uuid-from-context",
                "rule": "[Timestamp] Corrected rule here.",
                "target_entity": "global",
                "tags": ["formatting"],
                "reasoning": "Why it was updated."
            }},
            {{
                "action": "DELETE", 
                "id": "uuid-to-delete",
                "reasoning": "Why it was removed."
            }}
        ]
    }}
    """

    response = await learn_llm.ainvoke([HumanMessage(content=learning_prompt)])
     # 🛠️ FIXED: Use robust extractor
    content = extract_pure_text(response)
    m = get_token_metrics(response)

    

    actions_executed = 0

    if content and content.upper() != "NONE" and content != '{"actions": []}':
        try:
            # Isolate the JSON object explicitly
            json_str = extract_json_block(content, is_array=False)
            data = json.loads(json_str)
            actions = data.get("actions", [])

            for act in actions:
                act_type = act.get("action", "").upper()

                changed = await execute_procedural_action(act)
                if changed:
                    debug_print(f"✅ [Rules] {act_type} completed locally.")
                    actions_executed += 1

        except json.JSONDecodeError:
            # 2. FALLBACK PARSER: If the LLM dumped raw text instead of JSON
            print(
                "⚠️ [Warning] Procedural Editor returned raw text, falling back to basic ADD action."
            )

            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            lines = content.split("\n")

            for line in lines:
                if line.strip() and len(line.strip()) > 10:
                    fallback_act = {
                        "action": "ADD",
                        "rule": f"{timestamp} {line.strip()}",
                        "reasoning": "Fallback extraction",
                        "target_entity": "global",
                        "tags": ["fallback"],
                    }
                    changed = await execute_procedural_action(fallback_act)
                    if changed:
                        actions_executed += 1

        except Exception as e:
            print(f"❌ [Error] Rule Execution failed: {e}")
            raise e  # 🛠️ ADD THIS: Signal the decorator to retry the whole process

    return actions_executed, m


def has_memory_context(context: str) -> bool:
    return bool(str(context or "").strip() and str(context).strip() != "None")


async def learn_ingestion_memory_pairs(
    user_prompt: str,
    ai_resp: str,
    semantic_ctx: str,
    episodic_ctx: str,
    procedural_ctx: str,
    current_date: str = None,
) -> list[tuple[int, dict]]:
    """Learn ingestion/history chunks one user-assistant pair at a time."""
    pair_payloads = split_memory_ingestion_pairs(user_prompt)
    if not pair_payloads:
        zero = {"input": 0, "output": 0, "total": 0}
        return [(0, zero.copy()), (0, zero.copy()), (0, zero.copy())]

    working_semantic_ctx = semantic_ctx or ""
    working_episodic_ctx = episodic_ctx or ""
    working_procedural_ctx = procedural_ctx or ""

    semantic_stage = new_vector_staging_state()
    episodic_stage = new_vector_staging_state()

    semantic_count = 0
    episodic_count = 0
    procedural_count = 0

    semantic_tokens = {"input": 0, "output": 0, "total": 0}
    episodic_tokens = {"input": 0, "output": 0, "total": 0}
    procedural_tokens = {"input": 0, "output": 0, "total": 0}

    debug_print("\n" + "=" * 80)
    debug_print(f"🧠 [Ingestion Learning] Split chunk into {len(pair_payloads)} pair(s).")
    debug_print("🧠 [Ingestion Learning] Vector memories will be staged in RAM and committed after all pairs.")
    debug_print("=" * 80)

    for pair_index, pair_payload in enumerate(pair_payloads, start=1):
        pair_prompt = wrap_memory_ingestion_payload(pair_payload)

        debug_print("\n" + "-" * 80)
        debug_print(f"🧠 [Ingestion Learning] Feeding pair {pair_index}/{len(pair_payloads)}")
        debug_print("[Pair Payload]")
        debug_print(pair_payload)
        debug_memory_block("Working Semantic BEFORE pair", working_semantic_ctx)
        debug_memory_block("Working Episodic BEFORE pair", working_episodic_ctx)
        debug_memory_block("Working Procedural BEFORE pair", working_procedural_ctx)

        semantic_related_ctx = (
            f"[Episodic]\n{working_episodic_ctx}"
            if has_memory_context(working_episodic_ctx)
            else ""
        )
        episodic_related_ctx = (
            f"[Semantic]\n{working_semantic_ctx}"
            if has_memory_context(working_semantic_ctx)
            else ""
        )

        pair_results = await asyncio.gather(
            learn_vector_memory(
                semantic_db,
                "Semantic",
                pair_prompt,
                ai_resp,
                working_semantic_ctx,
                current_date,
                semantic_related_ctx,
                return_actions=True,
                execute_mutations=False,
            ),
            learn_vector_memory(
                episodic_db,
                "Episodic",
                pair_prompt,
                ai_resp,
                working_episodic_ctx,
                current_date,
                episodic_related_ctx,
                return_actions=True,
                execute_mutations=False,
            ),
            learn_procedural_memory(pair_prompt, ai_resp, working_procedural_ctx),
        )

        _, sem_metrics, sem_actions = pair_results[0]
        _, epi_metrics, epi_actions = pair_results[1]
        pro_count, pro_metrics = pair_results[2]

        debug_print(f"\n[Pair {pair_index}/{len(pair_payloads)} Semantic Actions]")
        debug_print(debug_json(sem_actions))
        debug_print(f"\n[Pair {pair_index}/{len(pair_payloads)} Episodic Actions]")
        debug_print(debug_json(epi_actions))
        debug_print(f"\n[Pair {pair_index}/{len(pair_payloads)} Procedural Changes]")
        debug_print(pro_count)

        procedural_count += pro_count

        add_token_metrics(semantic_tokens, sem_metrics)
        add_token_metrics(episodic_tokens, epi_metrics)
        add_token_metrics(procedural_tokens, pro_metrics)

        working_semantic_ctx = stage_vector_actions_on_context(
            working_semantic_ctx,
            sem_actions,
            semantic_db,
            semantic_stage,
            max_lines=70,
        )
        working_episodic_ctx = stage_vector_actions_on_context(
            working_episodic_ctx,
            epi_actions,
            episodic_db,
            episodic_stage,
            max_lines=70,
        )

        if pro_count:
            all_rules = await load_procedural_rules()
            working_procedural_ctx = get_formatted_rules_with_ids(all_rules, limit=15)

        debug_memory_block("Working Semantic AFTER pair", working_semantic_ctx)
        debug_memory_block("Working Episodic AFTER pair", working_episodic_ctx)
        debug_memory_block("Working Procedural AFTER pair", working_procedural_ctx)
        debug_print(f"🧠 [Ingestion Learning] Pair {pair_index}/{len(pair_payloads)} complete; updated working memory will be passed to the next pair.")
        debug_print("-" * 80)

    semantic_commit_actions = build_staged_vector_commit_actions(semantic_stage)
    episodic_commit_actions = build_staged_vector_commit_actions(episodic_stage)

    debug_print("\n" + "=" * 80)
    debug_print("🧠 [Ingestion Learning] All pairs complete. Final working memories before DB commit:")
    debug_memory_block("Final Working Semantic", working_semantic_ctx)
    debug_memory_block("Final Working Episodic", working_episodic_ctx)
    debug_memory_block("Final Working Procedural", working_procedural_ctx)
    debug_print("\n[Final Semantic Actions Committed To DB]")
    debug_print(debug_json(semantic_commit_actions))
    debug_print("\n[Final Episodic Actions Committed To DB]")
    debug_print(debug_json(episodic_commit_actions))
    debug_print("=" * 80)

    semantic_commit, episodic_commit = await asyncio.gather(
        semantic_db.execute_actions_with_results(semantic_commit_actions),
        episodic_db.execute_actions_with_results(episodic_commit_actions),
    )
    semantic_count = len(semantic_commit)
    episodic_count = len(episodic_commit)

    debug_print("\n" + "=" * 80)
    debug_print(f"🧠 [Ingestion Learning] DB commit complete: Semantic={semantic_count}, Episodic={episodic_count}, Procedural={procedural_count}")
    debug_print("=" * 80)

    return [
        (semantic_count, semantic_tokens),
        (episodic_count, episodic_tokens),
        (procedural_count, procedural_tokens),
    ]


async def update_memories_node(state: AgentState):
    start_time = time.time()

    u_prompt = state["user_prompt"]
    ai_resp = state["final_response"]

    if is_memory_ingestion_prompt(u_prompt):
        results = await learn_ingestion_memory_pairs(
            u_prompt,
            ai_resp,
            state.get("semantic_context", ""),
            state.get("episodic_context", ""),
            state.get("procedural_context", ""),
            state.get("current_date"),
        )
    else:
        results = await asyncio.gather(
            learn_vector_memory(
                semantic_db,
                "Semantic",
                u_prompt,
                ai_resp,
                state.get("semantic_context", ""),
                state.get("current_date"),
            ),
            learn_vector_memory(
                episodic_db,
                "Episodic",
                u_prompt,
                ai_resp,
                state.get("episodic_context", ""),
                state.get("current_date"),
            ),
            learn_procedural_memory(u_prompt, ai_resp, state.get("procedural_context", "")),
        )

    # Unpack the tuples
    sem_content, sem_tokens = results[0]
    epi_content, epi_tokens = results[1]
    broad_epi_content = ""
    pro_content, pro_tokens = results[2]

    # Unpack and Sum
    total_in = sum(r[1]["input"] for r in results)
    total_out = sum(r[1]["output"] for r in results)

    # Fetch existing metrics (from retrieval + generation) and add learning stats
    current_metrics = state.get("metrics", {})
    current_metrics.update({"learning_in": total_in, "learning_out": total_out})

    end_time = time.time()  # END TIMER

    debug_print("\n--- Memory Learning Complete ---")
    if sem_content:
        debug_print(f"💡 Learned Semantic: {sem_content}")
    if epi_content:
        debug_print(f"💡 Learned Episodic: {epi_content}")
    if broad_epi_content:
        debug_print(f"💡 Learned Broad Episodic: {broad_epi_content}")
    if pro_content:
        debug_print(f"💡 Learned Procedural: {pro_content}")
    if not any([sem_content, epi_content, broad_epi_content, pro_content]):
        debug_print("No new memories learned this turn.")

    debug_print(
        f"⏱️ [Metrics] Learning Time: {end_time - start_time:.2f}s | Total Tokens: In({total_in}) Out({total_out})"
    )
    debug_print("--------------------------------\n")
    return {"metrics": current_metrics}


# ==========================================
# 7b. BACKGROUND LEARNING PROCESS
# ==========================================
async def background_memory_update(
    user_prompt: str,
    ai_resp: str,
    semantic_ctx: str,
    episodic_ctx: str,
    procedural_ctx: str,
    current_date: str = None, # 🛠️ ADDED
):
    """Runs memory extraction silently in the background so the user doesn't wait."""
    start_time = time.time()

    semantic_related_ctx = f"[Episodic]\n{episodic_ctx}" if episodic_ctx else ""
    episodic_related_ctx = f"[Semantic]\n{semantic_ctx}" if semantic_ctx else ""

    if is_memory_ingestion_prompt(user_prompt):
        results = await learn_ingestion_memory_pairs(
            user_prompt,
            ai_resp,
            semantic_ctx,
            episodic_ctx,
            procedural_ctx,
            current_date,
        )
    else:
        results = await asyncio.gather(
            learn_vector_memory(
                semantic_db, "Semantic", user_prompt, ai_resp, semantic_ctx, current_date,semantic_related_ctx
            ),
            learn_vector_memory(
                episodic_db, "Episodic", user_prompt, ai_resp, episodic_ctx, current_date,episodic_related_ctx
            ),
            learn_procedural_memory(user_prompt, ai_resp, procedural_ctx),
        )

    # print("result:" ,results)

    # Unpack the tuples
    sem_content, sem_tokens = results[0]
    epi_content, epi_tokens = results[1]
    broad_epi_content = ""
    pro_content, pro_tokens = results[2]

    total_in = sum(r[1]["input"] for r in results)
    total_out = sum(r[1]["output"] for r in results)

    end_time = time.time()

    debug_print("\n--- Background Memory Learning Complete ---")
    if sem_content:
        debug_print(f"💡 Learned Semantic: {sem_content}")
    if epi_content:
        debug_print(f"💡 Learned Episodic: {epi_content}")
    if broad_epi_content:
        debug_print(f"💡 Learned Broad Episodic: {broad_epi_content}")
    if pro_content:
        debug_print(f"💡 Learned Procedural: {pro_content}")
    if not any([sem_content, epi_content, broad_epi_content, pro_content]):
        debug_print("No new memories learned this turn.")

    debug_print(
        f"⏱️ [Metrics] Background Learning Time: {end_time - start_time:.2f}s | Total Tokens: In({total_in}) Out({total_out})"
    )
    debug_print("--------------------------------\n")


# ==========================================
# 8. BUILD GRAPH & INTERFACES
# ==========================================
def route_after_generation(state: AgentState):
    return END if state.get("skip_learning", False) else "update_memories"


workflow = StateGraph(AgentState)
workflow.add_node("retrieve_memories", retrieve_memories_node)
workflow.add_node("route_tools", route_tools_node)  # NEW
workflow.add_node("generate_response", generate_response_node)
# workflow.add_node("update_memories", update_memories_node)

workflow.add_edge(START, "retrieve_memories")
workflow.add_edge("retrieve_memories", "route_tools")  # NEW PATH
workflow.add_edge("route_tools", "generate_response")  # NEW PATH
# workflow.add_conditional_edges("generate_response", route_after_generation)
# workflow.add_edge("update_memories", END)
workflow.add_edge("generate_response", END)  # 🚀 Returns to the user instantly!

app = workflow.compile()


async def main_chat_loop():
    print(
        "🤖 Argus Agent  V4 - Self Reflective loop ,  Cognitive Memory Editor, Temporal Truth Protocol, and Background Learning"
    )
    chat_history = []
    while True:
        u_input = input("\nYou: ")
        if u_input.lower() in ["exit", "quit"]:
            break

        initial_state: AgentState = {
            "user_prompt": u_input,
            "chat_history": chat_history,
            "semantic_context": "",
            "episodic_context": "",
            "procedural_context": "",
            "selected_skills": [],
            "skill_markdown": "",
            "final_response": "",
            "skip_learning": False,
            "user_id": "local_terminal_user",  # ADDED
            "channel_type": "terminal",  # ADDED
            "metrics": {},
            "current_date": None,
            "job_status_callback": None,
            "attachments_context": "",
        }
        final_state = await app.ainvoke(initial_state)
        print(f"\n🤖 Agent: {final_state['final_response']}")

        chat_history.extend(
            [
                HumanMessage(content=u_input),
                AIMessage(content=final_state["final_response"]),
            ]
        )


if __name__ == "__main__":
    asyncio.run(main_chat_loop())
