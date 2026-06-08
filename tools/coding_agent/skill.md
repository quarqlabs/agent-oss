---
name: coding_agent
description: Delegate coding, repo, edit, debug, refactor, and test tasks to a persistent coding agent.
triggers: codex, coding agent, fix bug, implement, refactor, run tests, edit code, repo, repository, codebase, debug, failing test
---

# Coding Agent Skill

Use this skill when the user asks Quarq to perform software engineering work in a repository:

- implement a feature
- fix a bug
- refactor code
- inspect a codebase
- run tests or builds
- edit files
- delegate work to Codex or a coding agent
- list or select the default coding-agent provider
- update the default coding workspace
- show or update coding-agent network access

## Operating Rules

- Use `configure_coding_agent` when the user asks which coding agents are available, what the default coding agent/workspace/network mode is, or asks to set the default provider/workspace/network mode.
- Use `start_coding_task` for a new coding task. Keep the task prompt concrete and include the repository/workspace path when the user provides one.
- Use `continue_coding_task` when the user asks to continue the latest coding task without providing a task id.
- Use `reply_to_coding_task` when the user is responding to a specific coding task id, approval request, or coding-agent question.
- Use `get_coding_task_status` or `get_coding_task_logs` when the user asks what happened, asks for progress, or provides a task id.
- Use `cancel_coding_task` only when the user asks to stop/cancel a coding task.
- Tool responses are intentionally short. Give the user the task id and next command or next action.
- Do not directly claim that code was changed unless the coding-agent task status/logs say so.
- Coding-agent network access is enabled by default so package registries can work during delegated coding tasks. Use `configure_coding_agent` with `network_access="off"` only when the user asks to block network.
- Destructive actions, secret/env edits, git commits, and pushes require user confirmation unless the user explicitly requested them and the coding agent confirms they are safe.
