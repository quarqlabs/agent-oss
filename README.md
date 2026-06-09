# Argus Agent

**Local memory. Hybrid retrieval. Self-correcting reasoning. Benchmark-grade recall.**

Argus Agent is a memory-first AI agent for long-context personal intelligence, grounded recall, temporal reasoning, quantitative reasoning, and tool use.

It is designed as an open, inspectable alternative to memory agents such as Hermes or OpenClaw, with a stronger emphasis on durable local memory, strict attribution, self-correcting retrieval, and benchmark-grade long-term recall.

The current local implementation keeps normal semantic, episodic, and procedural learning in `agent.py`. Deterministic structured-artifact extractor code exists in the repo, but it is disabled in the active learning path while benchmark memory quality is being tuned.

Local LongMemEval-S reports are checkpoints while learning and generation behavior is being validated. Treat checked-in report files as local progress snapshots, not final published benchmark numbers.

Benchmark cost warning: a full 500-question LongMemEval-S run with the current model mix has cost about `$2,500` in practice, or about `$5` per average question. Run a 1-question or small-sample benchmark first before starting the full dataset.

## Contents

- [Why Argus Exists](#why-argus-exists)
- [What's New In v0.5.0](#whats-new-in-v050)
- [What Makes It Different](#what-makes-it-different)
- [Highlights](#highlights)
- [Architecture](#architecture)
- [Memory System](#memory-system)
- [Structured Artifact Extractors](#structured-artifact-extractors)
- [Local Storage Layout](#local-storage-layout)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Temporal Reasoning](#temporal-reasoning)
- [Quantitative Reasoning](#quantitative-reasoning)
- [Self-Correcting Retrieval](#self-correcting-retrieval)
- [Learning Pipeline](#learning-pipeline)
- [Tool System](#tool-system)
- [Coding Agent Delegation](#coding-agent-delegation)
- [Benchmarks](#benchmarks)
- [Benchmark Cost Planning](#benchmark-cost-planning)
- [Current Local Metrics](#current-local-metrics)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Control Console](#control-console)
- [Agent Identity Config](#agent-identity-config)
- [Channel Integrations](#channel-integrations)
- [API Job Queue](#api-job-queue)
- [Environment Variables](#environment-variables)
- [Repository Map](#repository-map)
- [Design Principles](#design-principles)
- [Status](#status)
- [License](#license)

## Why Argus Exists

Most agents can chat. Fewer can remember. Almost none can remember carefully.

Argus Agent is built around a simple idea: memory is not just vector search. A serious memory agent needs to know what a memory means, when it happened, what numbers belong to, which entity a fact is attached to, when evidence is incomplete, and when it must search again instead of guessing.

Argus combines:

- local FAISS vector memory
- semantic, episodic, and procedural memory separation
- hybrid vector plus keyword retrieval
- HyDE-style query expansion
- dynamic recall depth
- strict temporal grounding
- numeric attribution and exact aggregation rules
- structured artifact extractor code for table rows, lists, blocks, quotes, budgets, timelines, metrics, ratios, and other evidence-shaped outputs, currently disabled in the active learning path
- self-correcting fallback retrieval
- background memory consolidation
- LangGraph orchestration
- progressive tool routing

The result is an agent that behaves less like a stateless chatbot and more like a disciplined cognitive system.

## What's New In v0.5.0

This release turns Argus into a much more complete local agent runtime:

- **Triage-first normal runtime:** normal chat now starts with a fast LLM triage layer that chooses direct response, tool routing, memory retrieval, or retrieval-before-tools. Information-only updates and questions already answerable from recent chat can skip foreground retrieval entirely.
- **Benchmark path preserved:** benchmark traffic still uses the original retrieval, tool-routing, generation, and learning prompts/flow so memory benchmark behavior remains comparable.
- **Learning-safe optional retrieval:** when foreground retrieval is skipped, background learning still performs a learning-related retrieval pass before memory editing so `UPDATE` and `DELETE` actions have prior memory IDs/context.
- **Clearer metrics:** triage, direct response, foreground retrieval, tool routing, generation, and learning-related retrieval are timed separately, and terminal background metrics avoid overwriting the active input prompt.
- **Argus control console:** a Codex-style CLI with a fixed bottom input row, scrollable transcript, Markdown rendering, command palette, multiline compose, live status header, global `argus` launcher, and one-command setup scripts for macOS/Linux and Windows.
- **API job queue:** chat requests now create jobs, emit status events while work is happening, and return final responses when the job completes. The CLI polls events instead of blocking silently.
- **On-demand channels:** Telegram connects only when requested with `/connect telegram`, supports startup defaults, shows typing indicators, retries registration cleanly, and can receive coding-task progress after connecting mid-run.
- **Durable channel context:** CLI and Telegram chat history are stored locally, command responses are saved into history, and only the latest four user/assistant pairs are passed into each agent request.
- **Multimodal input storage:** incoming Telegram/API files are stored under local channel state, indexed, and passed into the agent with best-effort text extraction or AI-assisted image/audio/PDF understanding.
- **Local identity management:** agent name, personality, use cases, and custom prompt can be updated through a local identity config file instead of Supabase-backed identity tools.
- **Cloud-tool expansion:** external SaaS actions are routed through a single cloud-tool skill with configurable toolkits, user-facing `/tools`, `/which-tool`, `/cloud-tools`, `/add-tool`, and `/remove-tool` commands.
- **Coding-agent delegation:** Argus can delegate software work to Codex, persist tasks and logs, continue or start fresh sessions, delete task history, configure workspace/provider/network mode, and stream progress into CLI and subscribed Telegram chats.
- **Coding safety and UX:** coding tasks use shallow retrieval, portable workspace defaults, task-id suggestions, network-on defaults for package registries, safe restart when Codex sandbox settings change, and progress-file polling for long training/build tasks.
- **Cloudflared setup fixes:** setup can install `cloudflared` with Homebrew on macOS/Linux, `winget` on Windows, or a direct PowerShell download fallback to `C:\cloudflared\cloudflared.exe`.

## What Makes It Different

Argus is not a wrapper around a vector database. It is a full memory reasoning loop.

Standard RAG systems usually fail long-memory tasks for one of four reasons:

1. They retrieve the wrong memory.
2. They retrieve the right memory but attach it to the wrong entity.
3. They confuse storage time with event time.
4. They calculate with nearby numbers that do not belong to the question.

Argus directly attacks those failure modes with retrieval decomposition, evidence attribution, temporal guardrails, numeric scope checks, and a second-pass recovery path when the first context is incomplete.

## Highlights

- Local-first memory: no Supabase pgvector dependency. Memories and rules are saved under `local_memory/<AGENT_ID>/`.
- Three memory types: semantic facts, episodic events, and procedural behavioral rules.
- FAISS-backed retrieval: normalized OpenAI embeddings with `IndexFlatIP` cosine-style similarity.
- Hybrid search: every retrieval pass combines vector search and direct keyword matching.
- HyDE query optimizer: rewrites the user prompt into multiple retrieval probes before search.
- Dynamic thresholds: wide-net `deep` mode for aggregation, timelines, broad categories, and recommendations; stricter `standard` mode for point facts.
- Required-data fallback: the model can request a targeted second retrieval pass when evidence is missing.
- Temporal truth protocol: separates database storage time from narrative event time.
- Quantitative fidelity: numbers are stored and used with owner, property, item, and exactness.
- Benchmark ingestion learning: history chunks are split into individual user/assistant pairs, learned sequentially, staged in RAM between pairs, and committed after the chunk is complete.
- Duplicate protection: batch writes skip exact duplicate content before embedding, then use the normal vector duplicate check to avoid repeated memories.
- Triage-first normal chat: information-only updates, casual chat, and questions answerable from current chat can skip foreground retrieval.
- Retrieval-before-tools: when an action genuinely needs stored memory before a tool can run, triage routes retrieval first and then tool selection.
- Background learning: normal user responses return immediately while memory extraction runs asynchronously.
- Learning-related retrieval: optional foreground retrieval does not weaken memory editing, because background learning can retrieve prior context before issuing `ADD`, `UPDATE`, or `DELETE`.
- Benchmark ingestion synchronization: benchmark memory-ingestion turns learn synchronously before returning, guarded by an ingestion lock.
- Progressive tool loading: tool docs are only injected when a skill is selected.
- Benchmark mode: disables tool routing, synchronously learns memory-ingestion chunks, and waits for any pending learning before final evaluation.
- Local control console: starts the FastAPI worker, shows structured request/job/channel events, supports multiline input, command completion, and a scrollable transcript.
- On-demand channel connections: channels are connected only when requested, starting with Telegram through a temporary Cloudflare tunnel and automatic webhook registration.
- Local identity config: agent name, personality, use cases, and custom directives can be updated by tool call into a local JSON file instead of Supabase.
- Coding-agent delegation: Argus can start durable Codex tasks, stream coding progress into the CLI, let the user reply, and persist task logs locally.

## Architecture

```text
User / API
    |
    v
LangGraph StateGraph
    |
    +-- triage_request  (normal channels)
    |     |
    |     +-- direct_response
    |     +-- route_tools
    |     +-- retrieve_memories
    |     +-- retrieve_memories -> route_tools
    |
    +-- retrieve_memories  (benchmark channel always starts here)
    |     |
    |     +-- HyDE query generation
    |     +-- semantic FAISS search
    |     +-- episodic FAISS search
    |     +-- keyword search
    |     +-- procedural rule routing
    |
    +-- route_tools
    |     |
    |     +-- skill catalog router
    |     +-- progressive skill markdown loading
    |
    +-- generate_response
          |
          +-- grounded answer synthesis
          +-- optional ReAct tool loop
          +-- REQUIRED_DATA fallback retrieval
          +-- memory learning
                +-- normal chat: background async
                +-- benchmark ingestion: inline sync
```

The normal interactive graph is triage-first:

```python
START -> triage_request -> direct_response -> END
START -> triage_request -> route_tools -> generate_response -> END
START -> triage_request -> retrieve_memories -> generate_response -> END
START -> triage_request -> retrieve_memories -> route_tools -> generate_response -> END
```

Benchmark traffic intentionally keeps the original high-recall route:

```python
START -> retrieve_memories -> route_tools -> generate_response -> END
```

Learning is launched from `generate_response` or `direct_response`. Normal chat keeps the interactive path fast by learning in the background. If normal chat skipped foreground retrieval, background learning performs a learning-related retrieval pass before memory editing so updates and deletes still see prior memory context. Benchmark memory-ingestion prompts learn inline before the response returns so the next history chunk retrieves against the latest committed memories.

## Memory System

Argus uses three memory layers.

### Semantic Memory

Semantic memory stores durable user facts:

- identity
- preferences
- relationships
- routines
- long-term projects
- possessions
- stable traits
- active statuses and inventories

Example:

```text
User owns a crystal chandelier that originally belonged to their great-grandmother and was given to them by their aunt.
```

### Episodic Memory

Episodic memory stores events and interaction history:

- what happened
- when it happened
- who was involved
- what was decided
- what the user asked for
- what changed

Example:

```text
On March 4, 2023, user received a crystal chandelier from their aunt that originally belonged to their great-grandmother.
```

### Procedural Memory

Procedural memory stores behavioral rules:

- tone preferences
- formatting preferences
- project-specific instructions
- forbidden wording
- content generation constraints

Procedural rules are tagged and routed, so the model sees only the relevant rules for the current prompt instead of carrying every rule forever.

## Structured Artifact Extractors

The local agent keeps the original normal-text learning path intact. Full user and assistant turns are still passed to the learning model, so ordinary narrative details, decisions, preferences, and summaries can become semantic, episodic, or procedural memories.

Structured extractor code is present for artifact-shaped content that summarization can otherwise compress too aggressively. In the current active runtime, these extractors are disabled and their outputs are not injected into the learning prompt or appended to episodic memory. This keeps benchmark learning focused on the model-generated memory actions from the actual user/assistant pair.

When enabled experimentally, the extractor layer can create deterministic episodic units for high-signal data such as:

- markdown table rows
- explicit artifact blocks such as `::title:: == description`
- numbered sections for objectives, parameters, methods, options, steps, recommendations, and similar headings
- recommendation, remedy, dish, shop, restaurant, and product list items
- ingredient and material items
- budget, cost, allocation, and campaign plan rows
- timeline and dated event clauses
- implementation or "uses algorithm/tool" relationships
- attributed quotations and exact source claims
- metric, percentage, improvement, and score relationships
- ratios, dilutions, and mixture instructions
- music sections, chord/note style rows, and chess move notation
- counted entity headings such as encounter counts, party sizes, item totals, or named grouped entities

The extractor layer is intentionally capped and high-signal. It is not meant to memorize every sentence. Its job is to preserve compact data-bearing rows and items that future recall questions often target verbatim.

In the current runtime, `STRUCTURED ARTIFACT UNITS` are not passed to the learning prompt. Vector writes still pass through the normal local action execution path, including exact duplicate blocking, batch embedding, and vector duplicate checks.

## Local Storage Layout

The current runtime stores memory locally:

```text
local_memory/
  <AGENT_ID>/
    semantic_memory/
      index.faiss
      memories.json
    episodic_memory/
      index.faiss
      memories.json
    procedural_memory/
      rules.json
    channel_state/
      chat_history.json
      attachments_index.json
      attachments/
    coding_agents/
      tasks.json
      logs/
        <task_id>.jsonl
```

`AGENT_ID` determines which memory folder is used. Reusing the same `AGENT_ID` reuses the same memory. Changing `AGENT_ID` gives you a clean isolated agent profile.

Channel state is also durable. CLI, Telegram, and future channels store full
conversation history in `channel_state/chat_history.json`, while each agent
request receives only the last eight messages (four user/assistant pairs) as
short-term chat context. Command responses such as `/cloud-tools` are saved
there too, so follow-ups can refer to them.

Incoming files are saved under `channel_state/attachments/` and indexed in
`attachments_index.json`. Text, Markdown, JSON, CSV, PDFs, DOCX files, images,
and audio get best-effort local extraction or AI-assisted description/transcript
when supported. The stored file remains available even when readable text cannot
yet be extracted.

PDF extraction uses `pypdf` for embedded text. If a PDF has no embedded text or
behaves like a scanned/image document, Argus can render the first few pages with
`PyMuPDF` and send them through the configured multimodal image model for
vision OCR. Make sure `pip install -r requirements.txt` has been run in the same
virtual environment that starts `main.py` or `agent_cli.py`.

Coding-agent task state is durable too. Each delegated coding task is stored in
`coding_agents/tasks.json`, and every progress/update/approval/completion event
is appended to that task's JSONL log under `coding_agents/logs/`.

Each semantic and episodic memory record includes:

- UUID
- agent ID
- memory type
- content
- embedding
- created timestamp
- updated timestamp

The formatted retrieval output intentionally preserves this shape:

```text
[STORED_AT: 2026-05-25 14:00:00] [ID: <uuid>] <memory content>
```

Downstream deduplication, recency sorting, contradiction handling, and memory update logic all depend on that stable format.

## Retrieval Pipeline

Argus does not simply embed the latest user prompt and hope for the best.

For normal chat, retrieval is now optional on the foreground path. A fast triage LLM first decides whether the latest turn is:

- `direct`: answer or acknowledge from the current message and recent chat history
- `tool`: route to the tool system without memory retrieval
- `retrieval`: retrieve memory before generation
- `retrieval + tool`: retrieve memory first, then route tools with that memory context

The triage layer is semantic rather than regex-based. It treats information-only updates as direct responses, because newly supplied facts do not need old memory to be acknowledged. It chooses retrieval when the user is asking for an answer that depends on stored personal memory not already present in recent chat. It chooses tools when the user asks Argus to perform an external action or delegate work. For action requests, tool-only is the default unless a concrete required action input is missing and stored memory is the likely source.

Benchmark traffic bypasses triage and still runs the original retrieval-first path.

The retrieval node first asks a lightweight model to produce a structured search plan:

```json
{
  "vector_queries": [
    "User total driving duration and travel history",
    "User vehicle, road trip, transit records",
    "User travel milestones, driving time calculation",
    "hours, road trip, destinations"
  ],
  "keywords": "driving, hours, total",
  "search_mode": "deep"
}
```

Then it performs:

1. Semantic vector search
2. Episodic vector search
3. Semantic keyword search
4. Episodic keyword search
5. ID-based deduplication
6. recency sorting
7. procedural rule routing

Search modes:

- `standard`: strict retrieval for point facts, threshold `0.38`
- `deep`: wide recall for totals, timelines, histories, recommendations, and broad categories, threshold `0.28`

This is why Argus can answer questions that require multiple memories rather than only nearest-neighbor recall.

## Temporal Reasoning

Argus treats time as evidence, not decoration.

The agent distinguishes:

- storage timestamp: when the memory was saved
- narrative date: when the event actually happened
- benchmark current date: the simulated date for an evaluation question
- relative dates: phrases like "yesterday", "today", "last month"

The Temporal Truth Protocol prevents common long-memory errors:

- using database timestamps as event dates
- borrowing dates from nearby but unrelated memories
- assuming a discussion date is the same as an event date
- calculating date gaps from guessed anchors

For "how long ago" questions, Argus searches for the named event first and only uses the current date as the calculation anchor after retrieval.

## Quantitative Reasoning

Long-memory benchmarks often punish sloppy number handling. Argus's numeric protocol is built to avoid that.

For totals, counts, durations, prices, quantities, or money questions, the model must identify:

- actor or entity
- measured action or property
- event or item
- exactness

It excludes numbers that are merely nearby or topically related.

Example:

```text
User helped organize a concert, which raised over $5,000.
```

The amount belongs to the concert, not automatically to the user. It is also a lower-bound value, not an exact addend.

For exact totals, Argus sums only exact unqualified values unless the user explicitly asks for a minimum, estimate, or range.

## Self-Correcting Retrieval

If the first retrieval pass does not contain enough evidence, Argus can emit:

```json
{
  "agent_response": "",
  "flags": ["REQUIRED_DATA"],
  "hyde_queries": ["aunt meetup", "received chandelier", "chandelier handoff"]
}
```

The runtime then performs a targeted fallback search and regenerates the answer with expanded context.

The fallback pass has a strict final verification rule: if the exact target is still missing, the agent must say the information is not available instead of guessing.

This makes the agent aggressive about recall but conservative about truth.

## Learning Pipeline

After every normal response, Argus starts background learning.

Because foreground retrieval can now be skipped, background learning has its own safety step. If the user-facing path did not retrieve memory, the learning worker runs a retrieval pass for the actual user prompt before calling the memory editor. That retrieved context is used only for learning, so memory updates and deletes still have the prior memory lines and IDs they need without forcing every user-facing response through retrieval.

Benchmark memory-ingestion prompts are the exception. They are learned synchronously before the ingestion response returns, so `run_dataset_evals.py` does not feed the next history chunk until the previous chunk has been learned and committed.

The learning model extracts:

- semantic memories
- episodic memories
- procedural rules

It can issue:

```json
{
  "actions": [
    {"action": "ADD", "content": "New memory"},
    {"action": "UPDATE", "id": "uuid", "content": "Updated memory"},
    {"action": "DELETE", "id": "uuid"}
  ]
}
```

Important learning behaviors:

- preserves specific names and proper nouns
- resolves relative dates using the current date
- anchors transfer and acquisition events
- preserves every number and qualifier
- avoids duplicate memories across semantic and episodic layers
- prefers exact values over approximate or bounded restatements
- updates existing records instead of creating conflicting duplicates
- keeps normal prose learning active
- splits benchmark ingestion chunks into individual user/assistant pairs
- stages semantic and episodic `ADD`, `UPDATE`, and `DELETE` actions in RAM between pairs
- commits staged semantic and episodic vector actions after all pairs in the chunk have been processed
- reloads procedural context between ingestion pairs when procedural rules change

Background learning is protected by:

- learning-related retrieval before memory editing when foreground retrieval was skipped
- persistent retry loop
- exponential backoff
- concurrency limit of 4 learning tasks for normal background learning
- benchmark ingestion lock for synchronous memory-ingestion learning
- benchmark synchronization before final questions

## Tool System

Argus includes a progressive skill router.

Instead of injecting all tool instructions into every prompt, the router sees a compact catalog and selects only the relevant skills. The generation model then receives the full markdown and bound tools for those selected skills.

Current included skills:

- agent identity management
- cloud app actions
- coding-agent delegation

Tool execution uses a ReAct loop with a maximum of 5 iterations. If the loop reaches the limit, the model is forced to stop calling tools and produce a final text response.

Cloud tools are the external-action layer for app and SaaS integrations such as GitHub, Gmail, Google Calendar, Slack, Notion, and Linear. Argus keeps its own local identity tool native, while external app auth, tool search, and execution flow through the cloud-tool session.

Users can inspect and expand the enabled cloud toolkit at runtime:

```text
/tools
/which-tool check my unread emails
/cloud-tools
/add-tool gmail
/remove-tool slack
```

Enabled cloud tools are stored in `local_memory/<AGENT_ID>/agent_tools.json`, with `.env` values used as startup defaults.
`/cloud-tools` fetches the available cloud-tool catalog when credentials are
configured, then falls back to the small local catalog if the remote catalog is
unavailable.

Native coding-agent commands:

```text
/coding implement the failing auth test
/coding-new start a fresh implementation pass for the dashboard
/coding-continue now add tests for the same change
/coding-continue <task_id> use this exact older session
/coding or /coding-tasks
/coding-agents
/coding-use codex
/coding-workspace /absolute/path/to/repo
/coding-network on
/coding-network off
/coding-network <task_id> on
/coding-allow-network <task_id>
/coding-status <task_id>
/coding-log <task_id>
/coding-reply <task_id> approved, continue
/coding-cancel <task_id>
/coding-delete <task_id>
/coding-clear
```

The coding-agent skill is provider-neutral. V1 uses Codex; future providers can
plug into the same task store, event feed, and API command surface.
The current default provider, workspace, and network mode are visible in the CLI
header.

## Coding Agent Delegation

Argus delegates coding work instead of editing files directly in the chat flow.
The first provider is Codex, launched through Codex CLI's MCP server with the
default command:

```bash
codex mcp-server
```

Delegated tasks are durable:

- `tasks.json` stores task status, provider, workspace, network mode, prompt, changed files, errors, and summary.
- `logs/<task_id>.jsonl` stores every progress event.
- `.argus/coding_progress/<task_id>.jsonl` inside the coding workspace is a provider-side progress file Argus asks Codex to update during long tasks. Argus polls it and streams those progress rows to the CLI and subscribed channels.
- `config.json` stores local overrides for default provider, workspace, and coding-network default when changed from CLI/API/tool calls.
- The CLI subscribes to `/api/events` and renders coding events in the transcript.
- The user can reply to a waiting task with `/coding-reply <task_id> <message>`.
- The user can continue the most recent completed Codex session with `/coding-continue <message>`.
- The user can start a deliberately fresh Codex session with `/coding-new <task>`.
- The user can delete a completed/failed/cancelled task with `/coding-delete <task_id>`, or clear all non-active task history with `/coding-clear`.

Provider/workspace control:

- `.env` provides startup defaults such as `CODING_AGENT_DEFAULT_PROVIDER` and `CODEX_WORKSPACE_ROOT`.
- `CODEX_WORKSPACE_ROOT=.` is portable and resolves to the directory where the user launched the Argus CLI/API.
- `/coding-agents` lists supported providers and shows the current default.
- `/coding-use <provider>` selects the default provider. V1 supports `codex`; `claude_code` and `cursor` are reserved provider slots for later.
- `/coding-workspace <path>` sets the default workspace without editing `.env`.
- `/coding-network on|off` sets whether new coding tasks can use network access. It defaults to `on` so package registries such as PyPI/npm work during coding tasks.
- `/coding-network <task_id> on|off` changes the stored network mode for a specific task. `/coding-allow-network <task_id>` is a shortcut for turning it on.
- If an older Codex thread was created before network was enabled, Argus restarts Codex from the saved task context on the next continuation so the new network mode actually reaches the provider shell.
- If a Telegram chat lists or inspects a running coding task, that chat is subscribed to future progress/completion updates for the task.
- When `/connect telegram` succeeds, Argus also subscribes known Telegram chats from `TELEGRAM_ALLOWED_USERS` or previous Telegram history to currently active coding tasks.
- Asking the agent to change the coding agent, workspace, or network mode can use the same native coding-agent config tool.

Common Codex session flow:

```text
/coding build an ml project scaffold
/coding-continue add a README and run the tests
/coding-new investigate a separate bug in the API
/coding-continue <task_id> continue an older task explicitly
```

Each Codex task stores the provider session id when Codex returns one, so
`/coding-continue` can resume the same Codex thread instead of starting from a
blank session.

Codex MCP returns the final tool result at the end of a provider call rather
than streaming every shell stdout line to Argus. For long work, Argus injects a
progress-reporting instruction and polls the workspace progress file above; if
Codex cannot write it, Argus still sends periodic heartbeat/status events.

For commands that need a task id, the CLI provides task-id selection. Type the
command followed by a space, choose a recent task with `Ctrl+N` / `Ctrl+P`, then
press `Tab` to insert the selected id:

```text
/coding-status <space>
/coding-log <space>
/coding-reply <space>
/coding-cancel <space>
/coding-delete <space>
/coding-continue <space>
/coding-network <space>
/coding-allow-network <space>
```

Task statuses:

```text
queued
running
waiting_user
completed
failed
cancelled
```

Default safety policy:

- Auto-approve low-risk read-only operations, normal edits inside the configured workspace, and routine test/build commands.
- Coding-task network access defaults to `on` inside the workspace-write sandbox. Turn it off globally with `/coding-network off`, or for a task with `/coding-network <task_id> off`.
- Ask for user confirmation before destructive commands, writes outside the workspace, secret/env edits, git commits, git pushes, credential access, or anything the provider explicitly marks as confirmation-required.

Coding requests use a shallow retrieval profile in normal channels. Argus still
retrieves recent local context, but it skips the HyDE LLM call and forces
standard mode with small `top_k` instead of broad/deep memory search. Benchmark
mode is unchanged.

### Adding A New Tool

Tool expansion is deliberately simple. To add a new capability, drop a new folder inside `tools/` and follow the existing skill convention.

```text
tools/
  your_tool/
    skill.md
    __init__.py
    tools.py
```

`skill.md` defines the router-facing metadata:

```markdown
---
name: your_tool
description: One-line description of what this skill can do.
triggers: keyword one, keyword two, natural language trigger
---

# Your Tool Skill

Describe when to use it, when not to use it, available tools, and operating rules.
```

`tools.py` defines LangChain tool callables, and `__init__.py` exports them using the folder-name convention:

```python
from .tools import your_first_tool, your_second_tool

YOUR_TOOL_TOOLS = [your_first_tool, your_second_tool]
```

That is it. `tools/__init__.py` automatically scans every subdirectory with a `skill.md`, imports the package, reads the frontmatter, and registers the exported `<FOLDER_NAME>_TOOLS` list. No central registry edit is required.

For user-scoped dynamic tools, a skill package can also export `<FOLDER_NAME>_TOOLS_FACTORY(runtime_config)`. This is how the cloud-tools skill creates session tools for the active `user_id` without hard-coding every external app schema into the prompt.

This gives Argus a clean capability expansion path: add a folder, describe the skill, export the tools, restart the process, and the agent can route to the new capability.

## Benchmarks

Argus includes a LongMemEval runner:

```bash
python run_dataset_evals.py
```

The benchmark pipeline:

1. Loads `eval_datasets/longmemeval_s_cleaned.json`
2. Splits haystack sessions into chunks
3. Feeds each chunk through the agent with learning enabled
4. Learns benchmark memory-ingestion chunks synchronously before returning the ingestion ACK
5. Asks the benchmark question with learning disabled
6. Judges with a binary evaluator
7. Stores `question_type` with each result row for reporting, while benchmark agent calls use the same runtime interface as normal agent calls
8. Writes results to `reports/longmemeval_results.json`

For each benchmark history chunk, the agent splits the chunk into individual user/assistant pairs. Semantic and episodic actions are staged in RAM after each pair, so the next pair sees the updated working memory. After all pairs in the chunk are processed, staged semantic and episodic actions are committed to the vector stores. Procedural learning also runs per pair and refreshes procedural context when rules change.

### Parallel Evaluation

For faster local benchmark runs, Argus also includes a process-based parallel evaluator:

```bash
EVAL_WORKERS=5 python run_dataset_evals_parallel.py
```

The parallel runner is designed for long benchmark runs where you do not want to lose completed progress. It first loads all completed question IDs from:

```text
reports/longmemeval_results.json
reports/longmemeval_results.worker*.json
```

Then it calculates the remaining questions, splits only those questions across the requested number of workers, and launches one isolated process per worker.

Each worker receives its own `AGENT_ID`:

```text
<AGENT_ID>_eval_worker_0
<AGENT_ID>_eval_worker_1
<AGENT_ID>_eval_worker_2
...
```

That means each worker gets an isolated FAISS memory folder under `local_memory/`, so multiple questions can be learned and evaluated at the same time without memory collision.

Worker outputs are written independently:

```text
reports/longmemeval_results.worker0.json
reports/longmemeval_results.worker1.json
reports/longmemeval_results.worker2.json
```

When all workers finish, the runner merges worker outputs back into:

```text
reports/longmemeval_results.json
```

The default is `5` workers. Choose `EVAL_WORKERS` based on the machine and API limits, then increase until OpenAI rate limits or local CPU pressure become the bottleneck.

### Prompt Regression Samples

For faster iteration on prompt and retrieval changes, use the sample runner:

```bash
python run_dataset_evals_sample.py
```

It reuses the parallel evaluator, selects a deterministic sample, writes a manifest of sampled questions, keeps per-worker checkpoints, and merges results into a sample-specific report. Useful environment variables:

```text
EVAL_SAMPLE_NAME=prompt_regression
EVAL_SAMPLE_SIZE=60
EVAL_SAMPLE_SOURCE_LIMIT=500
EVAL_SAMPLE_SEED=test6
EVAL_SAMPLE_QUESTION_IDS=<comma-separated ids>
EVAL_SAMPLE_FRESH=1
EVAL_WORKERS=20
```

To monitor live results across both the main and worker result files:

```bash
python monitor_results.py
```

Current local report files:

```text
reports/longmemeval_results.json
reports/longmemeval_results.worker*.json
```

### Benchmark Cost Planning

LongMemEval-S is an expensive benchmark for this agent because each question
feeds many conversation-history chunks, and each chunk can trigger retrieval
planning, memory learning, embeddings, and final answer generation.

Observed cost with the current model mix:

| Run size | Approximate cost |
| --- | ---: |
| 1 average question | about `$5` |
| 10 questions | about `$50` |
| 100 questions | about `$500` |
| Full 500-question LongMemEval-S run | about `$2,500` |

The current benchmark model mix is:

| Component | Model | Input / 1M | Output / 1M | Cached input / 1M |
| --- | --- | ---: | ---: | ---: |
| Retrieval planning | `gpt-4o-mini` | `$0.15` | `$0.60` | `$0.075` |
| Generation | `gpt-4.1` | `$2.00` | `$8.00` | `$0.50` |
| Memory learning | `gpt-4.1` | `$2.00` | `$8.00` | `$0.50` |
| Embeddings | `text-embedding-3-large` | `$0.13` | n/a | n/a |
| Benchmark judge | `gpt-5` | `$1.25` | `$10.00` | `$0.125` |

For the current LongMemEval-S dataset, the benchmark runner sees about 41,813
chunks total, or about 83.6 chunks per question. The direct chunk-ingestion
traffic alone is about 62.3M estimated input tokens, but that is only a lower
bound. The full cost is higher because the agent re-reads chunk content during
learning and creates embeddings for retrieval and memory writes.

Run a 1-question or small-sample benchmark first and inspect recorded usage
metrics before running all 500 questions.

### Current Local Metrics

Current local LongMemEval-S metrics, computed from `reports/longmemeval_results.json` and joined with `eval_datasets/longmemeval_s_cleaned.json` by `question_id`:

| Question type | Correct | Incorrect | Total | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Overall | 491 | 9 | 500 | 98.20% |
| knowledge-update | 77 | 1 | 78 | 98.72% |
| multi-session | 129 | 4 | 133 | 96.99% |
| single-session-assistant | 56 | 0 | 56 | 100.00% |
| single-session-preference | 30 | 0 | 30 | 100.00% |
| single-session-user | 70 | 0 | 70 | 100.00% |
| temporal-reasoning | 129 | 4 | 133 | 96.99% |

These metrics represent the current LongMemEval-S progress while Argus Agent is actively being improved. Some answers may change as failing or uncertain questions are rerun and fixes are added.

## Requirements

- Python 3.11 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)
- Optional for Telegram: a Telegram bot token from `@BotFather`
- Optional for Telegram without a domain: `cloudflared`. Setup can install it with Homebrew, Windows `winget`, or a direct Windows PowerShell download fallback.
- Optional for coding delegation: OpenAI Codex CLI on your `PATH`

## Quick Start

Clone the repo, then run the setup script from the repo root.

```bash
git clone https://github.com/quarqlabs/argus.git
cd argus
```

macOS/Linux:

```bash
python3 scripts/setup_argus.py
```

Windows PowerShell:

```powershell
py scripts\setup_argus.py
```

The setup script:

- creates `.venv`
- installs `requirements.txt`
- creates `.env` from `.env.example` if missing
- installs `cloudflared` when possible
- installs the global `argus` launcher for your user
- updates the user `PATH` for future terminals when possible

`cloudflared` setup behavior:

- macOS/Linux with Homebrew: `brew install cloudflare/cloudflare/cloudflared`
- Windows with `winget`: `winget install --id Cloudflare.cloudflared --source winget --accept-package-agreements --accept-source-agreements`
- Windows without `winget`: download the latest Cloudflare binary to `C:\cloudflared\cloudflared.exe`, temporarily add `C:\cloudflared` to the setup process `PATH`, and verify with `cloudflared.exe --version`

The CLI also checks `C:\cloudflared\cloudflared.exe` directly on Windows, so `/connect telegram` can work even before a new terminal picks up a permanent PATH update.

After setup, edit `.env` and fill at least:

```bash
OPENAI_API_KEY=your_api_key
USER_ID=local_user
AGENT_ID=local_agent
LOCAL_MEMORY_ROOT=local_memory
```

Open a new terminal, or run the PATH refresh command printed by the setup
script. Then start Argus from any directory:

```bash
argus
```

If you only want to install or repair the global launcher without reinstalling
dependencies, run the lighter launcher installer.

macOS/Linux:

```bash
python3 scripts/install_argus.py --force
```

Windows PowerShell:

```powershell
py scripts\install_argus.py --force
```

The global launcher points back to the cloned repo. If you move or delete that
repo folder, rerun the launcher installer from the new location.

On Windows, the launcher sets Python UTF-8 mode so CLI/API status symbols do
not crash under legacy `cmd.exe` code pages. If you see a `charmap` error such
as `can't encode character '\u274c'`, pull the latest code and rerun:

```powershell
py scripts\install_argus.py --force
```

The control console starts `main:app` for you, connects the CLI to the API job queue, and shows structured events as requests move through triage, retrieval, tool routing, generation, tool use, and final response.

For coding-agent delegation, the default Codex provider launches:

```bash
codex mcp-server
```

The Python side uses the OpenAI Agents SDK dependency from `requirements.txt`.
If the Codex CLI or the Agents SDK is missing, Argus records a failed coding
task with a setup message instead of crashing. On macOS, if `codex` is not on
the API worker's `PATH`, Argus also checks the standard Codex.app binary at
`/Applications/Codex.app/Contents/Resources/codex`.

The API worker keeps process-lifetime chat history per channel and passes the
last four user/assistant pairs into each agent request. This preserves short
references such as "done" after an auth link or "now check calendar" without
stuffing the full conversation into every prompt.

You can still run the raw terminal agent directly:

```bash
python agent.py
```

Or run only the API server:

```bash
uvicorn main:app --reload
```

Call the API:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What do you remember about me?", "channel_type": "web"}'
```

## Control Console

`agent_cli.py` is the recommended local entrypoint. It provides a Codex-style terminal surface around the local FastAPI worker:

- starts `main:app` on `127.0.0.1:8000`
- hides noisy HTTP client logs
- shows a scrollable transcript of structured events
- shows the current model label, working directory, API URL, connected channels, startup channels, default coding agent, coding workspace, and coding network mode
- reads the agent name from the live local identity config, so a rename through `agent_identity_manager` updates the header without restarting
- supports formatted Markdown in agent responses
- supports multiline input: `Enter` sends, `Shift+Enter` inserts a newline
- supports command suggestions when you type `/`, with `Tab` completing the first suggestion

Console commands:

```text
/help
/status
/tools
/which-tool <task>
/cloud-tools
/add-tool <tool>
/remove-tool <tool>
/coding <task>
/coding-new <task>
/coding-continue <message>
/coding-continue <task_id> <message>
/coding-tasks
/coding-agents
/coding-use <provider>
/coding-workspace <path>
/coding-network [on|off]
/coding-network <task_id> on|off
/coding-allow-network <task_id>
/coding-status <task_id>
/coding-log <task_id>
/coding-reply <task_id> <message>
/coding-cancel <task_id>
/coding-delete <task_id>
/coding-clear
/connect telegram
set-default start-channel telegram
set-default start-channel none
/wipe
/quit
```

`/connect telegram` starts the Telegram connection pipeline only when you ask for it. `set-default start-channel telegram` stores a local startup preference in `local_memory/<AGENT_ID>/agent_cli.json`, so future CLI launches can connect that channel automatically. `set-default start-channel none` clears that preference.

## Agent Identity Config

Argus no longer needs Supabase for local identity updates. Runtime identity uses a local JSON config file, with `.env` values as defaults.

Default location:

```text
local_memory/<AGENT_ID>/agent_identity.json
```

Override location:

```bash
AGENT_IDENTITY_CONFIG_PATH=local_memory/local_agent/agent_identity.json
```

Supported identity fields:

```json
{
  "agent_name": "Argus",
  "agent_personality": "professional and helpful",
  "agent_use_cases": ["general assistance"],
  "agent_custom_prompt": ""
}
```

Initial values can come from `.env`:

```bash
AGENT_NAME=Argus
AGENT_PERSONALITY="friendly, precise, high-energy"
AGENT_USE_CASES=["coding","research","life-long memory"]
AGENT_CUSTOM_PROMPT="Be concise, grounded, and useful."
```

When the user asks the agent to rename itself, change its personality, update its main use cases, or change global instructions, the `agent_identity_manager` tool writes the update to the JSON config file. The env values remain fallback defaults for a new agent profile or a missing config file.

## Channel Integrations

Channels are API-facing integrations. The first supported channel is Telegram; the design leaves room for WhatsApp and other channels later.

Telegram supports text, captions, photos, documents, PDFs, DOCX files, audio,
voice notes, videos, stickers, and other Telegram file objects. Every received
file is downloaded into local channel storage, indexed with source metadata, and
passed to the current agent job as attachment context. Future channel adapters
can use the same generic `POST /api/files` endpoint, then include the returned
attachment IDs in `/api/jobs` or `/api/chat`.

Telegram may allow a user to send a larger file into the chat, but the official
bot download path is limited. Argus defaults `CHANNEL_FILE_MAX_BYTES` to
`20000000` bytes (about 20 MB) for channel attachments. If a file is too large
or cannot be downloaded/read, the Telegram bot replies with a clear attachment
failure message instead of silently answering from the caption alone.

### Telegram

1. Create a Telegram bot with `@BotFather`.
2. Put the bot token in `.env`.
3. Put your Telegram numeric user ID in `TELEGRAM_ALLOWED_USERS`.
4. Set a random `TELEGRAM_WEBHOOK_SECRET`.
5. Install `cloudflared` if you do not have a public domain. `scripts/setup_argus.py` can install it automatically, including the direct Windows fallback to `C:\cloudflared\cloudflared.exe`.
6. Run `python agent_cli.py`.
7. In the console, run `/connect telegram`.

Example `.env` values:

```bash
TELEGRAM_BOT_TOKEN=123456789:replace_with_botfather_token
TELEGRAM_ALLOWED_USERS=123456789
TELEGRAM_WEBHOOK_SECRET=replace_with_random_secret
```

What `/connect telegram` does:

1. Starts a temporary Cloudflare tunnel to the local API.
2. Builds the public webhook URL as `<tunnel-url>/api/telegram/webhook`.
3. Calls Telegram `setWebhook` with the webhook secret.
4. Shows channel registration progress in the CLI.

Telegram messages are processed through the same API job queue as CLI messages. While a response is generating, the API sends Telegram `typing` chat actions so the chat feels alive instead of silent.

Channel commands also work from Telegram:

```text
/help
/status
/tools
/which-tool <task>
/cloud-tools
/add-tool <tool>
/remove-tool <tool>
/coding <task>
/coding-new <task>
/coding-continue <message>
/coding-continue <task_id> <message>
/coding-tasks
/coding-agents
/coding-use <provider>
/coding-workspace <path>
/coding-network [on|off]
/coding-network <task_id> on|off
/coding-allow-network <task_id>
/coding-status <task_id>
/coding-log <task_id>
/coding-reply <task_id> <message>
/coding-cancel <task_id>
/coding-delete <task_id>
/coding-clear
/wipe
/quit
```

`/quit` only stops the local CLI when typed in the console. From Telegram it returns a safety message because remote channels should not stop the local process.

## API Job Queue

The FastAPI worker exposes both synchronous and job-based paths.

Synchronous compatibility route:

```text
POST /api/chat
```

Job queue routes:

```text
POST /api/jobs
GET  /api/jobs/{job_id}
GET  /api/events?after=<event_id>
```

Coding-task routes:

```text
GET  /api/coding-agents
POST /api/coding-agents/default
POST /api/coding-agents/workspace
POST /api/coding-agents/network
GET  /api/coding-tasks
DELETE /api/coding-tasks
POST /api/coding-tasks/subscribe-channel
POST /api/coding-tasks
POST /api/coding-tasks/latest/reply
GET  /api/coding-tasks/{task_id}
DELETE /api/coding-tasks/{task_id}
GET  /api/coding-tasks/{task_id}/logs
POST /api/coding-tasks/{task_id}/reply
POST /api/coding-tasks/{task_id}/cancel
POST /api/coding-tasks/{task_id}/network
```

The CLI uses the job routes. A request is enqueued, the single worker processes jobs one by one, and status events are emitted for:

- triage
- retrieval
- tool routing
- generation
- tool running/completed/failed
- coding-agent progress
- final response

This is what lets the console show useful loader text such as memory retrieval, response generation, and active tool usage instead of blocking silently until the final answer arrives.

## Environment Variables

| Variable | Required | Description |
|---|---:|---|
| `OPENAI_API_KEY` | yes | Used for generation, retrieval planning, learning, and embeddings. |
| `AGENT_ID` | no | Selects the local memory namespace. Defaults to `local_agent`. |
| `USER_ID` | API only | Required by `main.py` for the FastAPI worker. |
| `LOCAL_MEMORY_ROOT` | no | Root folder for local memory. Defaults to `local_memory`. |
| `LOCAL_CHANNEL_STORAGE_ROOT` | no | Optional override for durable channel chat history and attachment storage. Defaults to `local_memory/<AGENT_ID>/channel_state`. |
| `CHANNEL_FILE_MAX_BYTES` | no | Max accepted channel attachment size in bytes. Defaults to `20000000` (about 20 MB, matching the practical Telegram bot download ceiling). |
| `ATTACHMENT_EXTRACT_MAX_CHARS` | no | Max extracted text saved from an attachment. Defaults to `24000`. |
| `MULTIMODAL_IMAGE_MODEL` | no | Optional OpenAI model for image descriptions. Defaults to `gpt-4o-mini`. |
| `MULTIMODAL_AUDIO_MODEL` | no | Optional OpenAI model for audio transcription. Defaults to `gpt-4o-mini-transcribe`. |
| `PDF_VISION_MAX_PAGES` | no | Max PDF pages rendered for vision OCR fallback when embedded text extraction fails. Defaults to `3`. |
| `AGENT_IDENTITY_CONFIG_PATH` | no | Optional override for the local identity config file. Defaults to `local_memory/<AGENT_ID>/agent_identity.json`. |
| `AGENT_NAME` | no | Default persona name when no local identity config exists. |
| `AGENT_PERSONALITY` | no | Default tone/personality when no local identity config exists. |
| `AGENT_USE_CASES` | no | Default use-case description. Accepts a JSON array or comma-separated string. |
| `AGENT_CUSTOM_PROMPT` | no | Default custom behavior instructions when no local identity config exists. |
| `ARGUS_AGENT_VERSION` | no | Display-only version label for the control console. Defaults to `v0.5.0`. |
| `ARGUS_MODEL_LABEL` | no | Display-only model label for the control console. Falls back to generation model labels. |
| `ARGUS_REASONING_EFFORT` | no | Optional display suffix for the console model label. |
| `AGENT_DEBUG` | no | Set to `true`/`1` to show verbose debug logs from `agent.py`; metrics still print without debug. |
| `CLOUD_TOOLS_API_KEY` | cloud tools only | Required to use external app tools through the cloud-tool session. |
| `CLOUD_TOOLKITS` | no | Comma-separated cloud-tool slugs. Defaults to `github,gmail,googlecalendar,slack,notion,linear`. |
| `CLOUD_TOOLS_CONFIG_PATH` | no | Optional override for enabled cloud-tool config. Defaults to `local_memory/<AGENT_ID>/agent_tools.json`. |
| `CLOUD_TOOLS_CACHE_DIR` | no | Writable cloud-tool SDK cache directory. Defaults to `local_memory/cloud_tools_cache`. |
| `CODING_AGENTS_ENABLED` | no | Enables coding-agent delegation. Defaults to `true`. |
| `CODING_AGENT_DEFAULT_PROVIDER` | no | Default coding provider. V1 supports `codex`. |
| `CODEX_MCP_COMMAND` | no | Command used to launch Codex MCP. Defaults to `codex`. |
| `CODEX_MCP_ARGS` | no | Comma-separated args for Codex MCP. Defaults to `mcp-server`. |
| `CODEX_WORKSPACE_ROOT` | no | Workspace path for delegated coding work. `.env.example` uses `.`, resolved from the user's launch directory. |
| `CODEX_APPROVAL_POLICY` | no | Approval policy label. Defaults to `argus-safe-auto`. |
| `CODEX_NETWORK_ACCESS` | no | Allows network access for new Codex coding tasks inside the workspace-write sandbox. Defaults to `true`; use `/coding-network off` to disable locally. |
| `CODEX_TASK_TIMEOUT_SECONDS` | no | Max runtime for a delegated coding task. Defaults to `1800`. |
| `TELEGRAM_BOT_TOKEN` | Telegram only | Bot token from `@BotFather`. Required for `/connect telegram`. |
| `TELEGRAM_ALLOWED_USERS` | Telegram recommended | Comma-separated numeric Telegram user IDs allowed to use the local agent. |
| `TELEGRAM_WEBHOOK_SECRET` | Telegram recommended | Secret token sent to Telegram `setWebhook` and verified by `/api/telegram/webhook`. |

Agent identity updates are local-first. The `agent_identity_manager` tool writes
to the JSON config file above, while env values remain startup defaults/fallbacks.

## Repository Map

```text
agent.py                  Core LangGraph agent, memory, retrieval, generation, learning
agent_cli.py              Local control console for API, jobs, events, and channels
agent_config.py           Local agent identity config loader/saver
agent_connector.py        Public async integration gateway
main.py                   FastAPI single-tenant worker
local_channel_store.py    Durable channel history and attachment storage
coding_agents/            Provider-neutral coding task store, policy, manager, Codex runner
run_dataset_evals.py      LongMemEval evaluation runner
run_dataset_evals_parallel.py
                          Parallel LongMemEval evaluation runner
monitor_results.py        Benchmark monitoring helper
tools/                    Skill registry and tool implementations
tools/coding_agent/       Native skill for coding-agent delegation
eval_datasets/            Cleaned LongMemEval dataset
reports/                  Evaluation outputs and checkpoints
local_memory/             Local FAISS and JSON memory stores
```

## Design Principles

Argus is built around a few hard rules:

- Retrieve broadly, reason narrowly.
- Store memories with ownership, dates, and qualifiers intact.
- Prefer saying "missing data" over inventing an answer.
- Treat temporal and numeric claims as evidence-bound operations.
- Keep normal user-facing latency low by learning in the background, while keeping benchmark ingestion deterministic by learning synchronously.
- Keep the context window clean with routing and progressive disclosure.
- Make the memory system portable, local, and easy to inspect.

## Status

Argus Agent v0.5.0 is an active OSS release candidate.

The current version is optimized for long-memory evaluation and single-user local memory. The next natural steps are:

- package cleanup
- dependency trimming
- unit tests for memory storage and retrieval
- reproducible benchmark scripts
- Docker packaging
- memory compaction and archival policies
- multi-user serving with isolated local stores

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
