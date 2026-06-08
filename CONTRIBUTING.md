# Contributing to Argus Agent

Thank you for your interest in contributing. This document covers how to set up a development environment, the conventions we follow, and the process for submitting changes.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository and clone your fork.
2. Create a new branch for your change:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. Make your changes, commit, and open a pull request against `main`.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your OPENAI_API_KEY and other values in .env
```

Run the agent locally:

```bash
python agent.py
```

Run the API server:

```bash
uvicorn main:app --reload
```

## Project Structure

| Path | Role |
|------|------|
| `agent.py` | Core LangGraph agent (v0.4.0) — memory, retrieval, generation, learning |
| `agent_connector.py` | Public async API gateway |
| `main.py` | FastAPI single-tenant worker |
| `tools/` | Progressive skill registry (email, calendar, PDF, identity) |
| `run_dataset_evals.py` | LongMemEval benchmark runner |
| `monitor_results.py` | Benchmark result monitor |
| `content_check.py` | Debug utility for Supabase memory inspection |

`agent_v1.py`, `agent_v2.py`, `agent_v3.py` are legacy iterations kept for reference; `agent.py` is the canonical version.

## Adding a New Tool

Drop a new folder under `tools/` following the existing skill convention:

```text
tools/
  your_tool/
    skill.md        # Router-facing metadata (YAML frontmatter + instructions)
    __init__.py     # Exports <FOLDER_NAME>_TOOLS list
    tools.py        # LangChain @tool callables
```

The skill registry in `tools/__init__.py` auto-discovers any folder that contains a `skill.md`. No central registry edit is required.

## Submitting Changes

- Keep pull requests focused: one logical change per PR.
- Write a clear PR description explaining _why_ the change is needed.
- If you are fixing a bug, reference the issue number.
- Ensure your branch is up to date with `main` before opening a PR.
- For large changes, open an issue first to discuss the approach.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python.
- Prefer explicit over implicit; avoid magic globals.
- Keep new dependencies minimal — justify any addition in the PR.
- Do not commit `.env`, OAuth tokens, memory data, or evaluation datasets.

## Reporting Issues

Please use [GitHub Issues](https://github.com/quarqlabs/argus/issues) to report bugs or request features. Include:

- A clear description of the problem or request.
- Steps to reproduce (for bugs).
- The Python version and OS you are running.
- Relevant log output (redact any API keys).
