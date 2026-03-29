"""CLI entry-point — chat, gateway, init, status, login."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import (
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_WORKSPACE,
    MEMORY_FILE,
    load_config,
    save_config,
)
from .sessions import (
    generate_session_id,
    list_sessions,
    load_session,
    save_session,
)

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ======================================================================
# Top-level group
# ======================================================================


@click.group()
@click.version_option(__version__, prog_name="agent-mini")
def cli():
    """Agent Mini — ultra-lightweight personal AI agent."""


# ======================================================================
# init
# ======================================================================


@cli.command()
def init():
    """Initialise config and workspace with interactive onboarding."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_WORKSPACE.mkdir(parents=True, exist_ok=True)

    if CONFIG_FILE.exists():
        console.print(f"[yellow]Config already exists at {CONFIG_FILE}[/yellow]")
        overwrite = click.confirm("Overwrite with a fresh config?", default=False)
        if not overwrite:
            console.print("[dim]Keeping existing config.[/dim]")
            return

    console.print(Panel(
        "[bold green]Agent Mini[/bold green] — Setup Wizard",
        border_style="green",
        padding=(0, 1),
    ))
    console.print()

    # ── Step 1: Provider ────────────────────────────────────────────
    console.print("[bold]1. Choose your LLM provider[/bold]\n")
    providers = [
        ("ollama", "Local models via Ollama (recommended)"),
        ("openai", "OpenAI API (GPT-4o, o1, etc.)"),
        ("local", "Any OpenAI-compatible server (LM Studio, vLLM, llama.cpp)"),
    ]
    for i, (name, desc) in enumerate(providers, 1):
        console.print(f"  [cyan]{i}[/cyan]) [bold]{name}[/bold] — {desc}")
    console.print()

    choice = click.prompt(
        "Select provider",
        type=click.IntRange(1, len(providers)),
        default=1,
    )
    provider_name = providers[choice - 1][0]
    console.print(f"  → [green]{provider_name}[/green]\n")

    # ── Step 2: Provider-specific config ────────────────────────────
    console.print("[bold]2. Provider settings[/bold]\n")

    providers_cfg: dict = {}

    if provider_name == "ollama":
        base_url = click.prompt(
            "  Ollama base URL",
            default="http://localhost:11434",
        )
        model = click.prompt("  Model name", default="llama3.1")
        think = click.confirm("  Enable thinking mode?", default=False)
        providers_cfg["ollama"] = {
            "baseUrl": base_url,
            "model": model,
            "think": think,
        }

    elif provider_name == "openai":
        api_key = click.prompt("  OpenAI API key", hide_input=True)
        model = click.prompt("  Model name", default="gpt-4o")
        providers_cfg["openai"] = {"apiKey": api_key, "model": model}

    elif provider_name == "local":
        base_url = click.prompt(
            "  Server base URL",
            default="http://localhost:8080/v1",
        )
        api_key = click.prompt("  API key (or 'no-key')", default="no-key")
        model = click.prompt("  Model name", default="local-model")
        providers_cfg["local"] = {
            "baseUrl": base_url,
            "apiKey": api_key,
            "model": model,
        }

    console.print()

    # ── Step 3: Workspace ───────────────────────────────────────────
    console.print("[bold]3. Workspace[/bold]\n")
    workspace = click.prompt(
        "  Workspace directory",
        default=str(DEFAULT_WORKSPACE),
    )
    console.print()

    # ── Step 4: Memory ──────────────────────────────────────────────
    console.print("[bold]4. Memory[/bold]\n")
    memory_enabled = click.confirm("  Enable persistent memory?", default=True)
    console.print()

    # ── Step 5: Telegram (optional) ─────────────────────────────────
    console.print("[bold]5. Telegram gateway (optional)[/bold]\n")
    telegram_enabled = click.confirm("  Set up Telegram bot?", default=False)
    telegram_cfg = {
        "enabled": False,
        "token": "",
        "allowFrom": [],
        "streamResponses": True,
    }
    if telegram_enabled:
        tg_token = click.prompt("  Bot token (from @BotFather)")
        tg_users = click.prompt(
            "  Allowed user IDs (comma-separated, or leave blank)",
            default="",
        )
        allow_list = [u.strip() for u in tg_users.split(",") if u.strip()]
        telegram_cfg = {
            "enabled": True,
            "token": tg_token,
            "allowFrom": allow_list,
            "streamResponses": True,
        }
    console.print()

    # ── Build & save config ─────────────────────────────────────────
    config = {
        "provider": provider_name,
        "providers": providers_cfg,
        "agent": {
            "maxIterations": 20,
            "temperature": 0.7,
            "systemPrompt": "",
        },
        "channels": {"telegram": telegram_cfg},
        "tools": {"restrictToWorkspace": False},
        "memory": {"enabled": memory_enabled, "maxEntries": 1000},
        "workspace": workspace,
    }
    save_config(config)

    console.print(Panel(
        f"[green]✓ Config saved → {CONFIG_FILE}[/green]\n"
        f"[green]✓ Workspace  → {workspace}[/green]",
        border_style="green",
        padding=(0, 1),
    ))
    console.print()
    console.print("[bold]Ready![/bold] Start chatting:")
    console.print("  [cyan]agent-mini chat[/cyan]")
    if telegram_enabled:
        console.print("  [cyan]agent-mini gateway[/cyan]  (start Telegram bot)")


# ======================================================================
# chat  (interactive or single-shot)
# ======================================================================


@cli.command()
@click.option("-m", "--message", default=None, help="Single message (non-interactive).")
@click.option("-v", "--verbose", is_flag=True, help="Show debug logs.")
@click.option("--no-markdown", is_flag=True, help="Plain-text output.")
@click.option("-s", "--session", default=None, help="Resume a session by ID.")
def chat(message: str | None, verbose: bool, no_markdown: bool, session: str | None):
    """Chat with the agent."""
    _setup_logging(verbose)
    config = load_config()
    if not config:
        console.print("[red]No config found. Run 'agent-mini init' first.[/red]")
        raise SystemExit(1)
    asyncio.run(_chat(config, message, no_markdown, session))


async def _chat(config: dict, message: str | None, plain: bool, session_id: str | None) -> None:
    from .agent import AgentLoop, Memory, ToolEvent
    from .providers import create_provider

    provider = create_provider(config)
    memory = Memory(
        MEMORY_FILE,
        max_entries=config.get("memory", {}).get("maxEntries", 1000),
    )
    agent = AgentLoop(provider, config, memory)

    # Session handling — resume or create new
    if session_id:
        loaded = load_session(session_id)
        conversation: list[dict] = loaded if loaded is not None else []
        if loaded is not None:
            console.print(f"[dim]Resumed session {session_id} ({len(conversation)} messages)[/dim]")
    else:
        session_id = generate_session_id()
        conversation = []

    # Track tool timing and turn timing
    turn_start = 0.0

    # Tool event callback for visualization
    async def _on_tool_event(event: ToolEvent) -> None:
        if event.arguments is not None:
            # Tool call start
            args_preview = json.dumps(event.arguments, ensure_ascii=False)[:120]
            console.print(
                f"  [dim cyan]⚡[/dim cyan] [bold dim]{event.name}[/bold dim][dim]({args_preview})[/dim]"
            )
        elif event.result_preview is not None:
            # Tool call result
            if event.is_error:
                console.print(f"    [red]✗ Error:[/red] [dim red]{event.result_preview[:120]}[/dim red]")
            else:
                preview = event.result_preview.replace("\n", " ")[:100]
                console.print(f"    [green]✓[/green] [dim]{preview}[/dim]")

    try:
        # Display styled header
        header = Text()
        header.append("Agent Mini", style="bold green")
        header.append(f" v{__version__}", style="dim")
        header.append(" • ", style="dim")
        header.append(f"{provider.name}", style="bold cyan")
        header.append(f" ({provider.model_name})", style="cyan")
        console.print(Panel(header, border_style="dim green", padding=(0, 1)))

        if message:
            with console.status("[dim]Thinking…[/dim]", spinner="dots"):
                response = await agent.run(
                    message, conversation, on_tool_event=_on_tool_event
                )
            _render(response, plain)
            return

        # Interactive REPL
        console.print(
            "[dim]Type 'exit' to quit · '/help' for commands[/dim]\n"
        )

        while True:
            try:
                user_input = _read_input()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "/exit", "/quit", ":q"}:
                console.print("[dim]Goodbye![/dim]")
                break

            # --- Slash commands ---
            if user_input.startswith("/"):
                handled = await _handle_slash_command(
                    user_input, conversation, agent, config, provider, memory, plain
                )
                if handled:
                    continue

            turn_start = time.monotonic()
            response = await agent.run(
                user_input, conversation, on_tool_event=_on_tool_event
            )
            turn_elapsed = time.monotonic() - turn_start

            # Auto-save session after each turn
            save_session(session_id, conversation)

            console.print()
            _render(response, plain)
            # Show token usage and timing
            info_parts = []
            if agent.turn_usage["total_tokens"] > 0:
                u = agent.turn_usage
                info_parts.append(
                    f"{u['prompt_tokens']}→ {u['completion_tokens']}← ({u['total_tokens']} total)"
                )
            info_parts.append(f"{turn_elapsed:.1f}s")
            console.print(f"[dim]  {'  •  '.join(info_parts)}[/dim]")
            console.print()
    finally:
        await agent.close()


def _read_input() -> str:
    """Read user input with multi-line support using triple-quote delimiters."""
    first_line = console.input("[bold blue]You:[/bold blue] ").strip()

    # Check for multi-line delimiter
    for delim in ('"""', "'''"):
        if first_line.startswith(delim):
            lines = [first_line[len(delim):]]
            while True:
                try:
                    line = console.input("[dim]...:[/dim] ")
                except (EOFError, KeyboardInterrupt):
                    break
                if delim in line:
                    lines.append(line[: line.index(delim)])
                    break
                lines.append(line)
            return "\n".join(lines)

    # Line continuation with backslash
    result = first_line
    while result.endswith("\\"):
        result = result[:-1] + "\n"
        try:
            result += console.input("[dim]...:[/dim] ")
        except (EOFError, KeyboardInterrupt):
            break

    return result


async def _handle_slash_command(
    command: str,
    conversation: list[dict],
    agent,
    config: dict,
    provider,
    memory,
    plain: bool,
) -> bool:
    """Handle a slash command. Returns True if handled."""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/clear":
        conversation.clear()
        console.print("[dim]Conversation cleared.[/dim]")
        return True

    if cmd == "/help":
        help_table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        help_table.add_column("Command", style="cyan bold", no_wrap=True)
        help_table.add_column("Description", style="dim")
        help_table.add_row("/clear", "Reset conversation")
        help_table.add_row("/model <name>", "Switch provider/model (e.g. ollama/llama3.1)")
        help_table.add_row("/tools", "List available tools")
        help_table.add_row("/memory [query]", "Browse/search stored memories")
        help_table.add_row("/status", "Show current config")
        help_table.add_row("/save [file]", "Export conversation as Markdown")
        help_table.add_row("/sessions", "List saved sessions")
        help_table.add_row("/load <id>", "Load a saved session")
        help_table.add_row("/help", "Show this help")
        console.print(Panel(help_table, title="[bold]Commands[/bold]", border_style="dim", padding=(1, 1)))
        console.print(
            "[dim]Multi-line: start with \"\"\" or ''' and end with the same delimiter.\n"
            "Line continuation: end a line with \\ to continue on the next line.[/dim]"
        )
        return True

    if cmd == "/tools":
        tool_defs = agent.tools.get_tool_defs()
        tools_table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        tools_table.add_column("Tool", style="cyan bold", no_wrap=True)
        tools_table.add_column("Description", style="dim")
        for td in tool_defs:
            func = td["function"]
            tools_table.add_row(func["name"], func.get("description", ""))
        console.print(Panel(tools_table, title="[bold]Available Tools[/bold]", border_style="dim", padding=(1, 1)))
        return True

    if cmd == "/memory":
        if arg:
            result = memory.recall(arg)
            console.print(result)
        else:
            recent = memory.get_recent(10)
            if not recent:
                console.print("[dim]No memories stored yet.[/dim]")
            else:
                mem_table = Table(show_header=True, box=None, padding=(0, 1, 0, 0))
                mem_table.add_column("Time", style="dim", no_wrap=True)
                mem_table.add_column("Key", style="cyan bold")
                mem_table.add_column("Value")
                for entry in recent:
                    ts = entry.get("timestamp", "?")[:16]
                    mem_table.add_row(ts, entry["key"], entry["value"])
                console.print(Panel(
                    mem_table,
                    title="[bold]Recent Memories[/bold]",
                    border_style="dim",
                    padding=(1, 1),
                ))
        return True

    if cmd == "/status":
        prov = config.get("provider", "ollama")
        model = config.get("providers", {}).get(prov, {}).get("model", "?")
        sandbox = config.get("tools", {}).get("sandboxLevel", "workspace")
        u = agent.session_usage
        status_table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
        status_table.add_column("Key", style="bold", no_wrap=True)
        status_table.add_column("Value")
        status_table.add_row("Provider", f"[cyan]{prov}[/cyan] ({model})")
        status_table.add_row("Workspace", config.get("workspace", str(DEFAULT_WORKSPACE)))
        status_table.add_row("Sandbox", sandbox)
        status_table.add_row("History", f"{len(conversation)} messages")
        status_table.add_row(
            "Tokens",
            f"{u['total_tokens']} total ({u['prompt_tokens']}→ {u['completion_tokens']}←)"
        )
        console.print(Panel(status_table, title="[bold]Status[/bold]", border_style="dim", padding=(1, 1)))
        return True

    if cmd == "/model":
        if not arg:
            console.print("[yellow]Usage: /model <provider/model> or /model <model>[/yellow]")
            return True
        # Parse provider/model or just model
        if "/" in arg:
            new_prov, new_model = arg.split("/", 1)
        else:
            new_prov = config.get("provider", "ollama")
            new_model = arg
        config["provider"] = new_prov
        config.setdefault("providers", {}).setdefault(new_prov, {})["model"] = new_model
        # Persist the change
        save_config(config)
        # Re-create provider
        from .providers import create_provider
        try:
            new_provider = create_provider(config)
            await agent.provider.close()
            agent.provider = new_provider
            console.print(
                f"[green]Switched to {new_prov}/{new_model}[/green]"
            )
        except Exception as e:
            console.print(f"[red]Failed to switch: {e}[/red]")
        return True

    if cmd == "/save":
        filename = arg or f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        _export_conversation(conversation, filename)
        return True

    if cmd == "/sessions":
        sessions = list_sessions()
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
        else:
            sess_table = Table(show_header=True, box=None, padding=(0, 1, 0, 0))
            sess_table.add_column("ID", style="cyan bold")
            sess_table.add_column("Messages", justify="right")
            sess_table.add_column("Updated", style="dim")
            sess_table.add_column("Preview", style="dim")
            for s in sessions[:20]:
                sess_table.add_row(
                    s["id"], str(s["messages"]),
                    s["updated"][:16], s["preview"]
                )
            console.print(Panel(sess_table, title="[bold]Sessions[/bold]", border_style="dim", padding=(1, 1)))
        return True

    if cmd == "/load":
        if not arg:
            console.print("[yellow]Usage: /load <session_id>[/yellow]")
            return True
        loaded = load_session(arg)
        if loaded is None:
            console.print(f"[red]Session '{arg}' not found.[/red]")
        else:
            conversation.clear()
            conversation.extend(loaded)
            console.print(f"[green]Loaded session {arg} ({len(loaded)} messages)[/green]")
        return True

    console.print(f"[yellow]Unknown command: {cmd}. Type /help for commands.[/yellow]")
    return True


def _export_conversation(conversation: list[dict], filename: str) -> None:
    """Export conversation history as Markdown."""
    lines = ["# Agent Mini Conversation\n", f"*Exported {datetime.now().isoformat()}*\n\n"]

    for msg in conversation:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"## You\n\n{content}\n\n")
        elif role == "assistant":
            lines.append(f"## Assistant\n\n{content}\n\n")
            if msg.get("tool_calls"):
                lines.append("<details>\n<summary>Tool calls</summary>\n\n")
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    lines.append(f"- **{func.get('name', '?')}**({func.get('arguments', '')})\n")
                lines.append("\n</details>\n\n")
        elif role == "tool":
            lines.append(
                f"<details>\n<summary>Tool result: {msg.get('name', 'tool')}</summary>\n\n"
                f"```\n{content[:2000]}\n```\n\n</details>\n\n"
            )

    path = Path(filename).resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines))
        console.print(f"[green]✓ Conversation saved → {path}[/green]")
    except OSError as e:
        console.print(f"[red]Failed to save: {e}[/red]")


def _render(text: str, plain: bool) -> None:
    if plain:
        console.print(text)
    else:
        try:
            console.print(Markdown(text))
        except Exception:
            console.print(text)


# ======================================================================
# gateway  (Telegram)
# ======================================================================


@cli.command()
@click.option("-v", "--verbose", is_flag=True, help="Show debug logs.")
def gateway(verbose: bool):
    """Start the messaging gateway (Telegram)."""
    _setup_logging(verbose)
    config = load_config()
    if not config:
        console.print("[red]No config found. Run 'agent-mini init' first.[/red]")
        raise SystemExit(1)
    asyncio.run(_gateway(config))


async def _gateway(config: dict) -> None:
    from .agent import AgentLoop, Memory
    from .bus import MessageBus
    from .channels import TelegramChannel
    from .providers import create_provider

    provider = create_provider(config)
    memory = Memory(
        MEMORY_FILE,
        max_entries=config.get("memory", {}).get("maxEntries", 1000),
    )
    agent = AgentLoop(provider, config, memory)
    bus = MessageBus(agent)

    channels = []
    ch_cfg = config.get("channels", {})

    # Telegram
    tg = ch_cfg.get("telegram", {})
    if tg.get("enabled") and tg.get("token"):
        channels.append(
            TelegramChannel(
                token=tg["token"],
                allow_from=tg.get("allowFrom", []),
                stream_responses=tg.get("streamResponses", True),
            )
        )

    if not channels:
        console.print(
            "[red]No channels enabled. Edit "
            f"{CONFIG_FILE} and set 'enabled': true.[/red]"
        )
        raise SystemExit(1)

    for ch in channels:
        console.print(f"[green]Starting {ch.name}…[/green]")
        await ch.start(bus.handle_message)

    console.print(
        f"\n[bold green]Gateway running[/bold green] "
        f"— {len(channels)} channel(s) active"
    )
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        console.print("\n[yellow]Shutting down…[/yellow]")
        for ch in channels:
            await ch.stop()
        await agent.close()


# ======================================================================
# status
# ======================================================================


@cli.command()
def status():
    """Show configuration status."""
    config = load_config()
    if not config:
        console.print("[red]Not initialised. Run 'agent-mini init'.[/red]")
        return

    console.print(f"[bold]Agent Mini[/bold] v{__version__}\n")

    # Provider
    prov = config.get("provider", "ollama")
    model = config.get("providers", {}).get(prov, {}).get("model", "?")
    console.print(f"  Provider : [cyan]{prov}[/cyan] ({model})")

    # Channels
    console.print("  Channels :")
    for name, cfg in config.get("channels", {}).items():
        tag = "[green]enabled[/green]" if cfg.get("enabled") else "[dim]disabled[/dim]"
        console.print(f"    {name:12s} {tag}")

    # Tools
    console.print("  Web search: [green]enabled[/green] (DuckDuckGo)")
    console.print("  Web fetch : [green]enabled[/green]")
    mem = config.get("memory", {}).get("enabled", True)
    console.print(
        f"  Memory    : {'[green]enabled[/green]' if mem else '[dim]disabled[/dim]'}"
    )
    console.print(f"  Workspace : {config.get('workspace', str(DEFAULT_WORKSPACE))}")
