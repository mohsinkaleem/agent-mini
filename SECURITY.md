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

- **Always use `workspace` or `readonly` mode** in shared or untrusted environments. An unknown `sandboxLevel` value is rejected at startup.
- **Never expose the agent to untrusted input** with `sandboxLevel: unrestricted`.
- Use **approval mode** (`tools.confirm`) so shell commands and file edits need your OK. `agent-mini init` enables it; in the Telegram gateway those tools are denied.
- A Telegram bot with an empty `allowFrom` is public and runs `readonly` unless `allowShell: true` is set. Use numeric user IDs. With more than one allowed user, memory is off unless `sharedMemory: true`.
- Dangerous commands (`rm -rf /`, `sudo`, `mkfs`, etc.) are blocked by default. The blocklist is friction, not a boundary.
- `shell_exec` does not inherit environment variables that look like secrets (`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*CREDENTIAL*`) unless listed in `tools.shellEnvAllow`.
- Web content is wrapped in `<untrusted_content>` and the model is told not to follow instructions in it. This reduces prompt-injection risk; it does not remove it.
- `AGENTS.md` / `.agent-mini.md` in the workspace are added to the system prompt. Treat them like code when you open an untrusted repository.
- `config.json`, `memory.json` and sessions are written with `0600` permissions. API keys can come from `AGENT_MINI_API_KEY` / `OPENAI_API_KEY` and the bot token from `TELEGRAM_BOT_TOKEN` instead of the file. `agent-mini doctor` warns when the config is readable by others.
