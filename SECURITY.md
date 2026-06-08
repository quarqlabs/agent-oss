# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.4.x (current) | Yes |
| < 0.4 | No |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report them through **GitHub Security Advisories for `quarqlabs/argus`**. Include:

- A description of the vulnerability and its potential impact.
- Step-by-step reproduction instructions.
- Any proof-of-concept code or screenshots (if applicable).

You will receive an acknowledgement within 48 hours. We aim to triage and release a fix within 14 days for critical issues.

## Secrets and Credentials

This repository is designed to keep secrets out of source code:

- All API keys and credentials are loaded via environment variables (see `.env.example`).
- OAuth tokens (`token.json`, `token_calendar.json`), service account keys (`gcs_service_account.json`), and local memory data are listed in `.gitignore` and must never be committed.
- If you believe a secret was accidentally exposed in a commit, rotate the affected credential immediately and report it so we can scrub the history.
