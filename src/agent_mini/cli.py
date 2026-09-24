"""CLI entry-point — init, chat, gateway, doctor."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import sys
import time

import click
import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
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
# Turn stats in -m mode go here so stdout holds only the reply.
err_console = Console(stderr=True)

# Third-party loggers that would otherwise spam the chat UI with one line
# per HTTP request. Only unmuted with --verbose.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "asyncio", "telegram")


def _setup_logging(verbose: bool = False) -> None:
    """Route logs through Rich so they blend with the chat UI.

    Only ``agent-mini``'s own logger is chatty by default; everything else
    is muted below WARNING so the transcript stays readable.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=console,
                show_time=verbose,
                show_path=verbose,
                markup=False,
                rich_tracebacks=True,
            )
        ],
        force=True,
    )
    logging.getLogger("agent-mini").setLevel(
        logging.DEBUG if verbose else logging.INFO
    )
    if not verbose:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


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

    # ── Step 5: Safety ────────────────────────────────────────────────────
    console.print("[bold]5. Safety[/bold]\n")
    ask_first = click.confirm(
        "  Ask before running shell commands and changing files?", default=True
    )
    console.print()

    # ── Step 6: Telegram (optional) ─────────────────────────────────
    console.print("[bold]6. Telegram gateway (optional)[/bold]\n")
    telegram_enabled = click.confirm("  Set up Telegram bot?", default=False)
    telegram_cfg = {
        "enabled": False,
        "token": "",
        "allowFrom": [],
        "streamResponses": True,
    }
    if telegram_enabled:
        tg_token = click.prompt("  Bot token (from @BotFather)")
        console.print(
            "  [yellow]If you leave the next field blank, ANYONE who finds the bot can use it.[/yellow]\n"
            "  [dim]A public bot runs read-only (no shell, no file writes) unless you set\n"
            "  channels.telegram.allowShell: true. Prefer numeric user IDs over usernames:\n"
            "  usernames can be changed and then claimed by someone else. Message\n"
            "  @userinfobot on Telegram to get your ID.[/dim]"
        )
        tg_users = click.prompt(
            "  Allowed numeric user IDs (comma-separated, blank = public)",
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

    # ── Build & save config (only the choices; everything else uses code defaults) ──
    config = {
        "provider": provider_name,
        "providers": providers_cfg,
        "channels": {"telegram": telegram_cfg},
        "tools": {
            "sandboxLevel": "workspace",
            "confirm": list(_DIFF_TOOLS) + ["shell_exec"] if ask_first else [],
        },
        "memory": {"enabled": memory_enabled},
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

# Exit codes for single-shot runs, so scripts and the eval harness can tell
# "wrong answer" from "never finished" from "provider down".
EXIT_OK = 0
EXIT_MAX_ITERATIONS = 2
EXIT_PROVIDER_ERROR = 3
EXIT_STUCK = 4
EXIT_CANCELLED = 130
_FINISH_EXIT_CODES = {
    "max_iterations": EXIT_MAX_ITERATIONS,
    "provider_error": EXIT_PROVIDER_ERROR,
    "stuck": EXIT_STUCK,
}


@cli.command()
@click.option("-m", "--message", default=None, help="Single message (non-interactive). '-' reads stdin.")
@click.option("-v", "--verbose", is_flag=True, help="Show debug logs.")
@click.option("--no-markdown", is_flag=True, help="Plain-text output.")
@click.option("-s", "--session", default=None, help="Resume a session by ID.")
@click.option(
    "--workspace",
    default=None,
    help="Override workspace directory for this run (isolates eval / one-shot tasks).",
)
@click.option("--provider", "provider_name", default=None, help="Override the provider for this run.")
@click.option("--model", default=None, help="Override the model for this run.")
@click.option("-y", "--yes", is_flag=True, help="Approve every tool call without asking (tools.confirm).")
@click.option("--no-stream", is_flag=True, help="Show the answer only once it is complete.")
@click.option("--show-thinking", is_flag=True, help="Stream the model's thinking (dimmed).")
def chat(
    message: str | None,
    verbose: bool,
    no_markdown: bool,
    session: str | None,
    workspace: str | None,
    provider_name: str | None,
    model: str | None,
    yes: bool,
    no_stream: bool,
    show_thinking: bool,
):
    """Chat with the agent.

    With -m, exits 0 on success, 2 on max iterations, 3 on a provider error,
    4 when the agent got stuck repeating itself and 130 when interrupted.
    """
    _setup_logging(verbose)
    config = load_config()
    if not config:
        console.print("[red]No config found. Run 'agent-mini init' first.[/red]")
        raise SystemExit(1)
    if message == "-":
        message = sys.stdin.read()
    # CLI --workspace or AGENT_MINI_WORKSPACE env var overrides config,
    # so callers (eval runner, scripts) can pin an isolated sandbox
    # without touching ~/.agent-mini/config.json.
    override = workspace or os.environ.get("AGENT_MINI_WORKSPACE")
    if override:
        config["workspace"] = override
        # Force restrictToWorkspace when an override is provided so the
        # agent stays inside the isolated dir.
        config.setdefault("tools", {})["restrictToWorkspace"] = True
    if provider_name:
        config["provider"] = provider_name
    if model:
        prov = config.get("provider", "ollama")
        config.setdefault("providers", {}).setdefault(prov, {})["model"] = model
    if yes:
        config.setdefault("tools", {})["confirm"] = []
    # Streaming redraws the screen, so only do it on a real terminal.
    stream = console.is_terminal and not no_stream
    show_thinking = stream and (show_thinking or bool(config.get("agent", {}).get("showThinking")))
    try:
        code = asyncio.run(_chat(config, message, no_markdown, session, stream, show_thinking))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        code = EXIT_CANCELLED
    if code:
        raise SystemExit(code)


def _one_line(text: str, limit: int) -> str:
    """Collapse whitespace, escape Rich markup, and clip *text* to *limit* chars.

    Tool output routinely contains square brackets (``[dir]``, log lines,
    JSON) which Rich would otherwise swallow as style tags.
    """
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return escape(flat)


def _format_tool_args(arguments: dict) -> str:
    """Render tool arguments as compact ``key=value`` pairs.

    Raw JSON is hard to scan in a terminal; this keeps the signal
    (which file, which query) and drops the punctuation.
    """
    parts = []
    for key, value in arguments.items():
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parts.append(f"{key}={_one_line(raw, 60)}")
    return "  ".join(parts)


# Tool results are Markdown (web_search emits **bold** titles); the preview
# line is plain text, so the syntax is just noise.
_MD_NOISE = re.compile(r"\*\*|__|`+|^\s*#{1,6}\s*", re.MULTILINE)


def _format_tool_result(preview: str) -> str:
    """Render a tool result preview as a single clean line."""
    return _one_line(_MD_NOISE.sub("", preview), 96)


def _build_agent(config: dict):
    """Create provider, memory and agent; exit cleanly on a config error."""
    from .agent import AgentLoop, Memory
    from .providers import create_provider

    memory = Memory(
        MEMORY_FILE,
        max_entries=config.get("memory", {}).get("maxEntries", 1000),
    )
    try:
        provider = create_provider(config)
        agent = AgentLoop(provider, config, memory)
    except ValueError as e:
        console.print(f"[red]Config error: {escape(str(e))}[/red]")
        console.print(f"[dim]Fix it in {CONFIG_FILE}[/dim]")
        raise SystemExit(1) from None
    return agent


async def _detect_model(agent) -> None:
    """Refine tier budgets from what the server reports; warn if the model can't call tools."""
    info = await agent.detect_model()
    if info and info.capabilities is not None and "tools" not in info.capabilities:
        console.print(
            f"[yellow]⚠ {escape(agent.provider.model_name)} does not report tool support. "
            "Tool calls may fail; agent-mini will try to parse calls written as text.[/yellow]"
        )


class _TurnView:
    """Spinner while waiting, live output while the answer streams, pauses for prompts."""

    def __init__(self, plain: bool):
        self._plain = plain
        self._status = console.status("[dim]Thinking…[/dim]", spinner="dots")
        self._live: Live | None = None
        self._text = ""
        self._thinking = ""
        # The streamed text of the reply that ended the turn (not frozen earlier segments).
        self.final_streamed = ""

    def __enter__(self) -> _TurnView:
        self._status.start()
        return self

    def __exit__(self, *exc) -> None:
        self.final_streamed = self._text
        self._stop_live()
        self._status.stop()

    def _renderable(self):
        parts = []
        if self._thinking:
            parts.append(Text(self._thinking.strip(), style="dim italic"))
        if self._text:
            parts.append(Text(self._text) if self._plain else Markdown(self._text))
        return Group(*parts)

    def _show(self) -> None:
        if self._live is None:
            self._status.stop()
            self._live = Live(
                self._renderable(), console=console, refresh_per_second=8,
                vertical_overflow="visible",
            )
            self._live.start()
        else:
            self._live.update(self._renderable())

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.update(self._renderable(), refresh=True)
            self._live.stop()
            self._live = None

    async def on_delta(self, delta: str) -> None:
        self._text += delta
        self._show()

    async def on_thinking(self, delta: str) -> None:
        self._thinking += delta
        self._show()

    def _freeze(self) -> None:
        """Leave streamed text on screen and start a fresh segment."""
        if self._live is not None:
            self._stop_live()
            self._text = self._thinking = ""
            self._status.start()

    def status(self, text: str) -> None:
        """Freeze any streamed text (a tool is about to run) and show *text* on the spinner."""
        self._freeze()
        self._status.update(text)

    def pause(self) -> None:
        self._freeze()
        self._status.stop()

    def resume(self) -> None:
        self._status.start()


_DIFF_TOOLS = ("write_file", "code_edit")


def _ask_approval(name: str, arguments: dict, tools, always: set[str]) -> bool:
    """Prompt y / n / always / diff for a tool listed in tools.confirm. Empty answer = no."""
    console.print(f"  [yellow]?[/yellow] [bold]{name}[/bold] needs approval")
    if name == "shell_exec":
        console.print(Text(f"    $ {arguments.get('command', '')}", style="bold"))
    else:
        console.print(f"    [dim]{_format_tool_args(arguments)}[/dim]")
    has_diff = name in _DIFF_TOOLS
    choices = "[y]es / [N]o / [a]lways" + (" / [d]iff" if has_diff else "")
    while True:
        try:
            answer = console.input(f"    Allow? {escape(choices)}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n    [dim]No answer (non-interactive?) — denied. Use --yes to approve all.[/dim]")
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("a", "always"):
            always.add(name)
            return True
        if answer in ("", "n", "no"):
            return False
        if has_diff and answer in ("d", "diff"):
            diff = tools.preview_change(name, arguments) or "(no preview)"
            console.print(Syntax(diff, "diff", theme="ansi_dark", background_color="default"))


async def _cancellable(coro):
    """Await *coro* so that Ctrl+C cancels it (raising CancelledError) instead of exiting."""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(coro)
    try:
        loop.add_signal_handler(signal.SIGINT, task.cancel)
    except (NotImplementedError, RuntimeError, ValueError):
        return await task  # Windows, or not on the main thread
    try:
        return await task
    finally:
        loop.remove_signal_handler(signal.SIGINT)


async def _chat(
    config: dict,
    message: str | None,
    plain: bool,
    session_id: str | None,
    stream: bool = False,
    show_thinking: bool = False,
) -> int:
    from .agent import ToolEvent

    agent = _build_agent(config)
    await _detect_model(agent)

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
    # Display for the current turn, so tool events and prompts can use it.
    view: _TurnView | None = None
    always_allowed: set[str] = set()

    async def _approve(name: str, arguments: dict) -> bool:
        if name in always_allowed:
            return True
        if view:
            view.pause()
        try:
            return _ask_approval(name, arguments, agent.tools, always_allowed)
        finally:
            if view:
                view.resume()

    agent.tools.approver = _approve

    # Tool event callback for visualization
    async def _on_tool_event(event: ToolEvent) -> None:
        if event.arguments is not None:
            # Tool call start
            if view:
                view.status(f"[dim]Running {event.name}…[/dim]")
            console.print(
                f"  [cyan]⚡[/cyan] [bold]{event.name}[/bold] "
                f"[dim]{_format_tool_args(event.arguments)}[/dim]"
            )
        elif event.result_preview is not None:
            # Tool call result
            timing = f" [dim]({event.duration:.1f}s)[/dim]" if event.duration >= 0.1 else ""
            preview = _format_tool_result(event.result_preview)
            if event.is_error:
                console.print(f"    [red]✗[/red] [red]{preview}[/red]{timing}")
            else:
                console.print(f"    [green]✓[/green] [dim]{preview}[/dim]{timing}")
            if view:
                view.status("[dim]Thinking…[/dim]")

    async def _run_turn(text: str) -> str:
        """Run one agent turn, then print the reply unless it was already streamed."""
        nonlocal view
        view = _TurnView(plain)
        try:
            with view:
                response = await agent.run(
                    text,
                    conversation,
                    on_stream=view.on_delta if stream else None,
                    on_tool_event=_on_tool_event,
                    on_thinking=view.on_thinking if show_thinking else None,
                )
            if response.strip() != view.final_streamed.strip():
                _render(response, plain)
            return response
        finally:
            view = None

    try:
        # Display styled header
        header = Text()
        header.append("Agent Mini", style="bold green")
        header.append(f" v{__version__}", style="dim")
        header.append(" • ", style="dim")
        header.append(f"{agent.provider.name}", style="bold cyan")
        header.append(f" ({agent.provider.model_name})", style="cyan")
        header.append(f" · {agent.profile.tier}", style="dim")
        console.print(Panel(header, border_style="dim green", padding=(0, 1)))

        if message:
            turn_start = time.monotonic()
            await _run_turn(message)
            save_session(session_id, conversation)
            err_console.print(f"[dim]  {_turn_footer(agent, time.monotonic() - turn_start)}[/dim]")
            return _FINISH_EXIT_CODES.get(agent.finish_reason, EXIT_OK)

        # Interactive REPL
        console.print(
            "[dim]Type 'exit' to quit · '/help' for commands · Ctrl+C stops a turn[/dim]\n"
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
                await _handle_slash_command(user_input, conversation, agent, config)
                continue

            turn_start = time.monotonic()
            console.print()
            try:
                await _cancellable(_run_turn(user_input))
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                # Cancel the turn, not the session.
                console.print("\n[yellow]Interrupted.[/yellow]\n")
                continue
            turn_elapsed = time.monotonic() - turn_start

            # Auto-save session after each turn
            save_session(session_id, conversation)

            console.print(f"[dim]  {_turn_footer(agent, turn_elapsed)}[/dim]")
            console.print()
    finally:
        await agent.close()
    return EXIT_OK


def _turn_footer(agent, elapsed: float) -> str:
    """One-line turn stats; evals/run.py parses this format."""
    n = agent.turn_iterations
    parts = [f"{n} iteration{'' if n == 1 else 's'}"]
    u = agent.turn_usage
    if u["total_tokens"] > 0:
        parts.append(f"{u['prompt_tokens']:,} in")
        parts.append(f"{u['completion_tokens']:,} out")
    parts.append(f"{elapsed:.1f}s")
    return "  •  ".join(parts)


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


def _panel(table: Table, title: str) -> None:
    console.print(Panel(table, title=f"[bold]{title}[/bold]", border_style="dim", padding=(1, 1)))


async def _handle_slash_command(command: str, conversation: list[dict], agent, config: dict) -> None:
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    match cmd:
        case "/clear":
            conversation.clear()
            console.print("[dim]Conversation cleared.[/dim]")

        case "/undo":
            console.print(f"[green]{escape(agent.tools.undo())}[/green]")
            if agent.tools.undo_depth:
                console.print(f"[dim]{agent.tools.undo_depth} more change(s) can be undone.[/dim]")
            console.print("[dim]Changes made through shell_exec can't be undone.[/dim]")

        case "/help":
            table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
            table.add_column("Command", style="cyan bold", no_wrap=True)
            table.add_column("Description", style="dim")
            for row in (
                ("/clear", "Reset conversation"),
                ("/undo", "Revert the last file change made by the agent"),
                ("/model <name>", "Switch provider/model (e.g. ollama/qwen3:8b)"),
                ("/tools", "List available tools"),
                ("/memory [query]", "Browse/search stored memories"),
                ("/status", "Show config, tier and token usage"),
                ("/sessions", "List saved sessions"),
                ("/load <id>", "Load a saved session"),
                ("/help", "Show this help"),
            ):
                table.add_row(*row)
            _panel(table, "Commands")
            console.print(
                "[dim]Multi-line: start with \"\"\" or ''' and end with the same delimiter.\n"
                "Line continuation: end a line with \\ to continue on the next line.[/dim]"
            )

        case "/tools":
            table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
            table.add_column("Tool", style="cyan bold", no_wrap=True)
            table.add_column("Description", style="dim")
            for td in agent.tools.get_tool_defs():
                table.add_row(td["function"]["name"], td["function"].get("description", ""))
            _panel(table, "Available Tools")

        case "/memory" if arg:
            console.print(agent.memory.recall(arg))

        case "/memory":
            recent = agent.memory.get_recent(10)
            if not recent:
                console.print("[dim]No memories stored yet.[/dim]")
                return
            table = Table(show_header=True, box=None, padding=(0, 1, 0, 0))
            table.add_column("Time", style="dim", no_wrap=True)
            table.add_column("Key", style="cyan bold")
            table.add_column("Value")
            for entry in recent:
                table.add_row(entry.get("timestamp", "?")[:16], entry["key"], entry["value"])
            _panel(table, "Recent Memories")

        case "/status":
            p, u = agent.profile, agent.session_usage
            table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
            table.add_column("Key", style="bold", no_wrap=True)
            table.add_column("Value")
            table.add_row("Provider", f"[cyan]{agent.provider.name}[/cyan] ({agent.provider.model_name})")
            table.add_row("Tier", f"{p.tier} · {p.context:,} token budget · {agent.max_iterations} iterations")
            table.add_row("Workspace", str(agent.tools.workspace))
            table.add_row("Sandbox", agent.tools.sandbox_level)
            table.add_row("Approval", ", ".join(sorted(agent.tools.confirm_tools)) or "off")
            table.add_row("History", f"{len(conversation)} messages")
            table.add_row("Tokens", f"{u['total_tokens']} total ({u['prompt_tokens']}→ {u['completion_tokens']}←)")
            _panel(table, "Status")

        case "/model" if arg:
            if "/" in arg:
                new_prov, new_model = arg.split("/", 1)
            else:
                new_prov, new_model = config.get("provider", "ollama"), arg
            config["provider"] = new_prov
            config.setdefault("providers", {}).setdefault(new_prov, {})["model"] = new_model
            # Persist only this change: the in-memory config also carries
            # per-run overrides (--workspace, --model) that must not be saved.
            persisted = load_config()
            persisted["provider"] = new_prov
            persisted.setdefault("providers", {}).setdefault(new_prov, {})["model"] = new_model
            save_config(persisted)
            from .providers import create_provider
            try:
                new_provider = create_provider(config)
            except Exception as e:
                console.print(f"[red]Failed to switch: {escape(str(e))}[/red]")
                return
            # The agent recomputes tier, budgets and num_ctx.
            await agent.set_provider(new_provider)
            await _detect_model(agent)
            console.print(
                f"[green]Switched to {escape(new_prov)}/{escape(new_model)}[/green] "
                f"[dim](tier {agent.profile.tier})[/dim]"
            )

        case "/model":
            console.print("[yellow]Usage: /model <provider/model> or /model <model>[/yellow]")

        case "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
                return
            table = Table(show_header=True, box=None, padding=(0, 1, 0, 0))
            table.add_column("ID", style="cyan bold")
            table.add_column("Messages", justify="right")
            table.add_column("Updated", style="dim")
            table.add_column("Preview", style="dim")
            for s in sessions[:20]:
                table.add_row(s["id"], str(s["messages"]), s["updated"][:16], s["preview"])
            _panel(table, "Sessions")

        case "/load" if arg:
            loaded = load_session(arg)
            if loaded is None:
                console.print(f"[red]Session '{arg}' not found.[/red]")
                return
            conversation[:] = loaded
            console.print(f"[green]Loaded session {arg} ({len(loaded)} messages)[/green]")

        case "/load":
            console.print("[yellow]Usage: /load <session_id>[/yellow]")

        case _:
            console.print(f"[yellow]Unknown command: {cmd}. Type /help for commands.[/yellow]")


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
    from .bus import MessageBus
    from .channels import TelegramChannel

    channels = []
    ch_cfg = config.get("channels", {})

    # Telegram
    tg = ch_cfg.get("telegram", {})
    token = tg.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if tg.get("enabled") and token:
        allow_from = [str(u) for u in tg.get("allowFrom", [])]
        if not allow_from or "*" in allow_from:
            if tg.get("allowShell", False):
                console.print(Panel(
                    "[bold red]⚠ Telegram bot is PUBLIC and shell_exec is enabled.[/bold red]\n"
                    "Anyone who finds the bot can run commands on this machine.\n"
                    "Set channels.telegram.allowFrom to your numeric user ID.",
                    border_style="red",
                ))
            else:
                config = {
                    **config,
                    "tools": {**config.get("tools", {}), "sandboxLevel": "readonly"},
                }
                console.print(Panel(
                    "[bold yellow]⚠ Telegram bot is PUBLIC:[/bold yellow] anyone who finds it can use it.\n"
                    "Running in [bold]readonly[/bold] sandbox (no shell, no file writes).\n"
                    "Set channels.telegram.allowFrom to your numeric user ID, or\n"
                    "channels.telegram.allowShell: true to accept the risk.",
                    border_style="yellow",
                ))
        elif any(not u.isdigit() for u in allow_from):
            console.print(
                "[yellow]⚠ channels.telegram.allowFrom contains usernames. Usernames can be "
                "changed and then claimed by someone else; prefer numeric user IDs.[/yellow]"
            )
        # One memory store would leak one user's facts into everyone's prompt.
        shared = len(allow_from) != 1 or "*" in allow_from
        if shared and not tg.get("sharedMemory", False):
            config = {**config, "memory": {**config.get("memory", {}), "enabled": False}}
            console.print(
                "[dim]Memory is off for the gateway because several users can reach it "
                "(set channels.telegram.sharedMemory: true to share one store).[/dim]"
            )
        channels.append(
            TelegramChannel(
                token=token,
                allow_from=allow_from,
                stream_responses=tg.get("streamResponses", True),
            )
        )

    if not channels:
        console.print(
            "[red]No channels enabled. Edit "
            f"{CONFIG_FILE} and set 'enabled': true.[/red]"
        )
        raise SystemExit(1)

    agent = _build_agent(config)
    agent.allow_local_images = False
    await _detect_model(agent)
    confirm = sorted(agent.tools.confirm_tools)
    if confirm:
        console.print(
            f"[dim]No one can approve tool calls in the gateway, so these are denied: "
            f"{', '.join(confirm)} (tools.confirm).[/dim]"
        )
    bus = MessageBus(agent)

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
# doctor
# ======================================================================

_OK, _WARN, _FAIL = "[green]✓[/green]", "[yellow]![/yellow]", "[red]✗[/red]"


@cli.command()
def doctor():
    """Check config, provider, model and tools. Exits 1 if something is broken."""
    raise SystemExit(asyncio.run(_doctor()))


async def _doctor() -> int:
    from .agent import AgentLoop, Memory
    from .providers import create_provider

    rows: list[tuple[str, str, str]] = []

    def report() -> int:
        table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
        for mark, check, detail in rows:
            table.add_row(mark, f"[bold]{check}[/bold]", detail)
        console.print(table)
        return 1 if any(mark == _FAIL for mark, _, _ in rows) else 0

    try:
        config = load_config()
    except json.JSONDecodeError as e:
        rows.append((_FAIL, "Config", f"{CONFIG_FILE} is not valid JSON: {escape(str(e))}"))
        return report()
    if not config:
        rows.append((_FAIL, "Config", f"{CONFIG_FILE} not found — run: agent-mini init"))
        return report()
    mode = CONFIG_FILE.stat().st_mode & 0o777
    if mode & 0o077:
        rows.append((_WARN, "Config", f"{CONFIG_FILE} is readable by others; run: chmod 600 {CONFIG_FILE}"))
    else:
        rows.append((_OK, "Config", str(CONFIG_FILE)))

    try:
        # A throwaway memory path: doctor must not touch the real store.
        agent = AgentLoop(create_provider(config), config, Memory(CONFIG_DIR / ".doctor-memory.json"))
    except ValueError as e:
        rows.append((_FAIL, "Config values", escape(str(e))))
        return report()

    try:
        tools = agent.tools
        writable = os.access(tools.workspace, os.W_OK)
        if writable:
            rows.append((_OK, "Workspace", str(tools.workspace)))
        else:
            rows.append((_FAIL, "Workspace", f"{tools.workspace} is not writable"))
        approval = ", ".join(sorted(tools.confirm_tools)) or "off"
        rows.append((_OK, "Sandbox", f"{tools.sandbox_level} · approval: {approval}"))
        rows.extend(await _doctor_provider(agent))
        p = agent.profile
        rows.append((
            _OK, "Budgets",
            f"tier {p.tier} · {p.context:,}-token budget · num_ctx {agent.provider.context_window:,} · "
            f"{agent.max_iterations} iterations",
        ))
        if shutil.which("rg"):
            rows.append((_OK, "ripgrep", "installed"))
        else:
            rows.append((_WARN, "ripgrep", "not installed; search_files falls back to grep"))
        tg = config.get("channels", {}).get("telegram", {})
        if tg.get("enabled"):
            allow = [str(u) for u in tg.get("allowFrom", [])]
            if not (tg.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN")):
                rows.append((_FAIL, "Telegram", "enabled but no token (config or TELEGRAM_BOT_TOKEN)"))
            elif not allow or "*" in allow:
                rows.append((_WARN, "Telegram", "bot is public: allowFrom is empty"))
            else:
                rows.append((_OK, "Telegram", f"{len(allow)} allowed user(s)"))
    finally:
        await agent.close()
    return report()


async def _doctor_provider(agent) -> list[tuple[str, str, str]]:
    """Reachability, model presence and tool support for the configured provider."""
    provider = agent.provider
    model = escape(provider.model_name)
    base = getattr(provider, "_base_url", "")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            if provider.name == "ollama":
                resp = await client.get(f"{base}/api/version")
            else:
                resp = await client.get(f"{base}/models", headers=getattr(provider, "_headers", {}))
        data = resp.json() if resp.status_code < 400 else {}
    except (httpx.HTTPError, ValueError) as e:
        return [(_FAIL, "Provider", f"{provider.name} at {base} is unreachable ({type(e).__name__})")]

    if provider.name == "ollama":
        rows = [(_OK, "Provider", f"ollama {data.get('version', '?')} at {base}")]
        info = await agent.detect_model()
        if info is None:
            rows.append((_FAIL, "Model", f"{model} not found — run: ollama pull {model}"))
            return rows
        facts = [model]
        if info.params_b:
            facts.append(f"{info.params_b:.3g}B params")
        if info.context_length:
            facts.append(f"{info.context_length:,}-token context")
        rows.append((_OK, "Model", " · ".join(facts)))
        if info.capabilities is not None:
            if "tools" in info.capabilities:
                rows.append((_OK, "Tool calling", "supported"))
            else:
                rows.append((_FAIL, "Tool calling", f"{model} has no tool support; pick another model"))
        return rows

    if resp.status_code in (401, 403):
        return [(_FAIL, "Provider", f"{base} rejected the API key (HTTP {resp.status_code})")]
    if resp.status_code >= 400:
        return [(_WARN, "Provider", f"{base} answered HTTP {resp.status_code} for /models")]
    ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
    rows = [(_OK, "Provider", f"{provider.name} at {base}")]
    if ids and provider.model_name not in ids:
        rows.append((_WARN, "Model", f"{model} is not in the server's model list"))
    else:
        rows.append((_OK, "Model", model))
    return rows
