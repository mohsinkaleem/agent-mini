# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue.
2. Email **mohsin.kaleem512@gmail.com** with details of the vulnerability.
3. Include steps to reproduce if possible.

You should receive a response within 48 hours. We'll work with you to understand the issue and coordinate a fix before any public disclosure.

## Security Considerations

Agent Mini executes shell commands and file operations on behalf of the user. The built-in sandbox levels (`unrestricted`, `workspace`, `readonly`) control access:

- **Always use `workspace` or `readonly` mode** in shared or untrusted environments.
- **Never expose the agent to untrusted input** without enabling `restrictToWorkspace`.
- Dangerous commands (`rm -rf /`, `sudo`, `mkfs`, etc.) are blocked by default.
- API keys are stored in `~/.agent-mini/config.json` — ensure this file has appropriate permissions (`chmod 600`).
