"""Command line: init, chat, doctor, gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path

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
from .config import DEFAULTS, PROVIDERS, agent_home, config_file, read_config, save_config, with_defaults
from .sessions import generate_session_id, list_sessions, load_session, save_session

# The UI: header, tool lines, spinner, prompts. With -m into a pipe it moves to
# stderr, so stdout holds only the reply.
console = Console()
reply_console = Console()
err_console = Console(stderr=True)

# Exit codes for `chat -m`, so scripts and the eval harness can tell
# "wrong answer" from "never finished" from "provider down".
EXIT_OK, EXIT_MAX_ITERATIONS, EXIT_PROVIDER_ERROR, EXIT_STUCK, EXIT_CANCELLED = 0, 2, 3, 4, 130
_EXIT_CODES = {"max_iterations": EXIT_MAX_ITERATIONS, "provider_error": EXIT_PROVIDER_ERROR, "stuck": EXIT_STUCK}
_FILE_TOOLS = ("write_file", "edit_file")


def _setup_logging(verbose: bool = False) -> None:
    """Logs go through Rich. Only agent-mini's own logger talks unless --verbose."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING, format="%(message)s", force=True,
        handlers=[RichHandler(console=console, show_time=verbose, show_path=verbose, markup=False)],
    )
    logging.getLogger("agent-mini").setLevel(logging.DEBUG if verbose else logging.INFO)
    if not verbose:
        for name in ("httpx", "httpcore", "asyncio", "telegram"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _load(overrides: dict | None = None) -> dict:
    """The user's config with *overrides* and defaults applied. Exits cleanly when it is missing or invalid."""
    try:
        raw = read_config()
    except json.JSONDecodeError as e:
        console.print(f"[red]{config_file()} is not valid JSON: {escape(str(e))}[/red]")
        raise SystemExit(1) from None
    if raw is None:
        console.print("[red]No config found. Run 'agent-mini init' first.[/red]")
        raise SystemExit(1)
    for path, value in (overrides or {}).items():
        node = raw
        *parents, leaf = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    try:
        return with_defaults(raw)
    except ValueError as e:
        console.print(f"[red]Config error: {escape(str(e))}[/red]\n[dim]Fix it in {config_file()}[/dim]")
        raise SystemExit(1) from None


def _build_agent(config: dict):
    from .loop import AgentLoop
    from .memory import Memory
    from .providers import create_provider

    memory = Memory(agent_home() / "memory.json", max_entries=config["memory"]["maxEntries"])
    try:
        return AgentLoop(create_provider(config), config, memory)
    except ValueError as e:
        console.print(f"[red]Config error: {escape(str(e))}[/red]\n[dim]Fix it in {config_file()}[/dim]")
        raise SystemExit(1) from None


async def _detect_model(agent) -> None:
    """Refine budgets from what the server reports; warn when the model can't call tools."""
    info = await agent.detect_model()
    if info and info.capabilities is not None and "tools" not in info.capabilities:
        console.print(f"[yellow]⚠ {escape(agent.provider.model)} does not report tool support. Tool calls "
                      "may fail; agent-mini will try to parse calls written as text.[/yellow]")


@click.group()
@click.version_option(__version__, prog_name="agent-mini")
def cli():
    """Agent Mini: a small local-first AI agent."""


# ── init ─────────────────────────────────────────────────────────────


@cli.command()
def init():
    """Create the config and workspace, asking a few questions."""
    path = config_file()
    if path.exists() and not click.confirm(f"{path} already exists. Overwrite it?", default=False):
        console.print("[dim]Keeping the existing config.[/dim]")
        return
    console.print(Panel("[bold green]Agent Mini[/bold green] setup", border_style="green", padding=(0, 1)))
    choices = [("ollama", "local models via Ollama (recommended)"), ("openai", "OpenAI API"),
               ("local", "any OpenAI-compatible server: LM Studio, vLLM, llama.cpp")]
    for i, (name, desc) in enumerate(choices, 1):
        console.print(f"  [cyan]{i}[/cyan]) [bold]{name}[/bold]: {desc}")
    name = choices[click.prompt("Provider", type=click.IntRange(1, len(choices)), default=1) - 1][0]
    defaults = DEFAULTS["providers"][name]
    chosen: dict = {}
    if name != "openai":
        chosen["baseUrl"] = click.prompt("Server URL", default=defaults["baseUrl"])
    chosen["model"] = click.prompt("Model", default=defaults["model"])
    if name != "ollama":
        key = click.prompt("API key (blank: read AGENT_MINI_API_KEY / OPENAI_API_KEY)", default="",
                           hide_input=True, show_default=False)
        if key:
            chosen["apiKey"] = key
    workspace = click.prompt("Workspace directory", default=str(agent_home() / "workspace"))
    ask = click.confirm("Ask before running shell commands and changing files?", default=True)
    Path(workspace).expanduser().mkdir(parents=True, exist_ok=True)
    save_config({
        "provider": name,
        "providers": {name: chosen},
        "workspace": workspace,
        "tools": {"confirm": ["shell", *_FILE_TOOLS] if ask else []},
    })
    console.print(f"\n[green]✓ Config saved to {path}[/green]\nNext: [cyan]agent-mini doctor[/cyan], then "
                  "[cyan]agent-mini chat[/cyan]. Every other setting is in the README.")


# ── chat ─────────────────────────────────────────────────────────────


@cli.command()
@click.option("-m", "--message", default=None, help="Send one message and exit. '-' reads stdin.")
@click.option("-s", "--session", default=None, help="Resume a session by ID.")
@click.option("--workspace", default=None, help="Workspace directory for this run (also AGENT_MINI_WORKSPACE).")
@click.option("--provider", "provider_name", default=None, help="Provider for this run.")
@click.option("--model", default=None, help="Model for this run.")
@click.option("-y", "--yes", is_flag=True, help="Approve every tool call without asking.")
@click.option("--no-stream", is_flag=True, help="Show the answer only once it is complete.")
@click.option("--show-thinking", is_flag=True, help="Stream the model's thinking (dimmed).")
@click.option("--no-markdown", is_flag=True, help="Plain-text output.")
@click.option("--report", "report_path", default=None, help="With -m: write a JSON report of the run here.")
@click.option("-v", "--verbose", is_flag=True, help="Show debug logs.")
def chat(message, session, workspace, provider_name, model, yes, no_stream, show_thinking, no_markdown,
         report_path, verbose):
    """Chat with the agent.

    With -m, exits 0 on success, 2 on max iterations, 3 on a provider error,
    4 when the agent got stuck repeating itself and 130 when interrupted.
    """
    global console
    if message and not reply_console.is_terminal:
        console = err_console
    _setup_logging(verbose)
    overrides: dict = {}
    if ws := workspace or os.environ.get("AGENT_MINI_WORKSPACE"):
        overrides["workspace"] = ws
    if provider_name:
        overrides["provider"] = provider_name
    if yes:
        overrides["tools.confirm"] = []
    config = _load(overrides)
    if "workspace" in overrides and config["tools"]["sandboxLevel"] == "unrestricted":
        config["tools"]["sandboxLevel"] = "workspace"  # a per-run workspace is a sandbox
    if model:
        config["providers"][config["provider"]]["model"] = model
    if message == "-":
        message = sys.stdin.read()
    # Streaming redraws the screen, so only on a real terminal.
    stream = reply_console.is_terminal and not no_stream
    thinking = stream and (show_thinking or config["agent"]["showThinking"])
    try:
        code = asyncio.run(_chat(config, message, no_markdown, session, stream, thinking, report_path))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        code = EXIT_CANCELLED
    if code:
        raise SystemExit(code)


def _one_line(text: str, limit: int) -> str:
    """Whitespace collapsed, clipped, and escaped: tool output is full of [brackets] Rich would eat."""
    flat = " ".join(text.split())
    return escape(flat[: limit - 1] + "…" if len(flat) > limit else flat)


def _format_args(arguments: dict) -> str:
    return "  ".join(
        f"{k}={_one_line(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), 60)}"
        for k, v in arguments.items()
    )


class _TurnView:
    """A spinner while waiting, live text while the answer streams, and pauses for approval prompts."""

    def __init__(self, plain: bool):
        self._plain = plain
        self._status = console.status("[dim]Thinking…[/dim]", spinner="dots")
        self._live: Live | None = None
        self._text = self._thinking = ""
        self.final_streamed = ""  # streamed text of the reply that ended the turn

    def __enter__(self) -> _TurnView:
        self._status.start()
        return self

    def __exit__(self, *exc) -> None:
        self.final_streamed = self._text
        self._stop_live()
        self._status.stop()

    def _renderable(self):
        parts = [Text(self._thinking.strip(), style="dim italic")] if self._thinking else []
        if self._text:
            parts.append(Text(self._text) if self._plain else Markdown(self._text))
        return Group(*parts)

    def _show(self) -> None:
        if self._live is None:
            self._status.stop()
            self._live = Live(self._renderable(), console=reply_console, refresh_per_second=8,
                              vertical_overflow="visible")
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

    def status(self, text: str) -> None:
        """Leave streamed text on screen (a tool is about to run) and show *text* on the spinner."""
        if self._live is not None:
            self._stop_live()
            self._text = self._thinking = ""
            self._status.start()
        self._status.update(text)

    def pause(self) -> None:
        self.status("")
        self._status.stop()

    def resume(self) -> None:
        self._status.start()


def _ask_approval(name: str, arguments: dict, tools, always: set[str]) -> bool:
    """Ask y / N / always / diff for a tool listed in tools.confirm. No answer means no."""
    console.print(f"  [yellow]?[/yellow] [bold]{name}[/bold] needs approval")
    if name == "shell":
        console.print(Text(f"    $ {arguments.get('command', '')}", style="bold"))
    else:
        console.print(f"    [dim]{_format_args(arguments)}[/dim]")
    has_diff = name in _FILE_TOOLS
    choices = "[y]es / [N]o / [a]lways" + (" / [d]iff" if has_diff else "")
    # asyncio's Ctrl+C handler can't interrupt a blocking input(): use Python's
    # own while the prompt is open. Ctrl+C then means "no" (and the turn stops).
    try:
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    except ValueError:  # not the main thread
        previous = None
    try:
        return _approval_answer(name, arguments, tools, always, has_diff, choices)
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def _approval_answer(name: str, arguments: dict, tools, always: set[str], has_diff: bool, choices: str) -> bool:
    while True:
        try:
            answer = console.input(f"    Allow? {escape(choices)}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n    [dim]No answer: denied. Use --yes to approve all.[/dim]")
            return False
        if answer in ("y", "yes", "a", "always"):
            if answer.startswith("a"):
                always.add(name)
            return True
        if answer in ("", "n", "no"):
            return False
        if has_diff and answer in ("d", "diff"):
            diff = tools.preview(name, arguments) or "(no preview)"
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


async def _chat(config: dict, message: str | None, plain: bool, session_id: str | None,
                stream: bool = False, show_thinking: bool = False, report_path: str | None = None) -> int:
    from .loop import ToolEvent

    agent = _build_agent(config)
    await _detect_model(agent)
    conversation: list[dict] = []
    if session_id and (loaded := load_session(session_id)) is not None:
        conversation = loaded
        console.print(f"[dim]Resumed session {session_id} ({len(conversation)} messages)[/dim]")
    session_id = session_id or generate_session_id()
    view: _TurnView | None = None
    always: set[str] = set()

    async def approve(name: str, arguments: dict) -> bool:
        if name in always:
            return True
        if view:
            view.pause()
        try:
            return _ask_approval(name, arguments, agent.tools, always)
        finally:
            if view:
                view.resume()

    agent.tools.approver = approve

    async def on_tool_event(event: ToolEvent) -> None:
        if event.arguments is not None:
            if view:
                view.status(f"[dim]Running {event.name}…[/dim]")
            console.print(f"  [cyan]⚡[/cyan] [bold]{event.name}[/bold] [dim]{_format_args(event.arguments)}[/dim]")
            return
        timing = f" [dim]({event.duration:.1f}s)[/dim]" if event.duration >= 0.1 else ""
        preview = _one_line(event.result or "", 96)
        mark = "[red]✗[/red] [red]" if event.is_error else "[green]✓[/green] [dim]"
        console.print(f"    {mark}{preview}[/]{timing}")
        if view:
            view.status("[dim]Thinking…[/dim]")

    async def run_turn(text: str) -> float:
        nonlocal view
        started = time.monotonic()
        view = _TurnView(plain)
        try:
            with view:
                reply = await agent.run(text, conversation, on_stream=view.on_delta if stream else None,
                                        on_tool_event=on_tool_event,
                                        on_thinking=view.on_thinking if show_thinking else None)
            if reply.strip() != view.final_streamed.strip():
                _render(reply, plain)
        finally:
            view = None
        save_session(session_id, conversation)
        return time.monotonic() - started

    try:
        header = Text.assemble(("Agent Mini", "bold green"), (f" v{__version__} • ", "dim"),
                               (agent.provider.name, "bold cyan"), (f" ({agent.provider.model})", "cyan"),
                               (f" · {agent.profile.tier}", "dim"))
        console.print(Panel(header, border_style="dim green", padding=(0, 1)))
        if message:
            started = time.monotonic()
            try:
                elapsed = await run_turn(message)
            finally:  # also when interrupted, so a timed-out eval run can be read
                if report_path:
                    _write_report(report_path, agent, session_id, time.monotonic() - started)
            err_console.print(f"[dim]  {_turn_footer(agent, elapsed)}[/dim]")
            return _EXIT_CODES.get(agent.finish_reason, EXIT_OK)

        console.print("[dim]Type 'exit' to quit · '/help' for commands · Ctrl+C stops a turn[/dim]\n")
        while True:
            try:
                text = _read_input()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text.lower() in {"exit", "quit", "/exit", "/quit", ":q"}:
                break
            if text.startswith("/"):
                await _slash(text, conversation, agent, config)
                continue
            console.print()
            try:
                elapsed = await _cancellable(run_turn(text))
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                console.print("\n[yellow]Interrupted.[/yellow]\n")  # cancel the turn, not the session
                continue
            console.print(f"[dim]  {_turn_footer(agent, elapsed)}[/dim]\n")
        console.print("[dim]Goodbye![/dim]")
    finally:
        await agent.close()
    return EXIT_OK


def _turn_footer(agent, elapsed: float) -> str:
    n, u = agent.turn_iterations, agent.turn_usage
    parts = [f"{n} iteration{'' if n == 1 else 's'}"]
    if u["total_tokens"]:
        parts += [f"{u['prompt_tokens']:,} in", f"{u['completion_tokens']:,} out"]
    return "  •  ".join([*parts, f"{elapsed:.1f}s"])


def _write_report(path: str, agent, session_id: str, elapsed: float) -> None:
    """JSON summary of a one-shot run (read by evals/run.py)."""
    report = {"model": agent.provider.model, "provider": agent.provider.name, "tier": agent.profile.tier,
              "session": session_id, "elapsed_s": round(elapsed, 2), "iterations": agent.turn_iterations,
              **agent.last_run}
    Path(path).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def _read_input() -> str:
    """One line, or several: wrap them in triple quotes, or end lines with a backslash."""
    first = console.input("[bold blue]You:[/bold blue] ").strip()
    for delim in ('"""', "'''"):
        if first.startswith(delim):
            lines = [first[len(delim):]]
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
    text = first
    while text.endswith("\\"):
        try:
            text = text[:-1] + "\n" + console.input("[dim]...:[/dim] ")
        except (EOFError, KeyboardInterrupt):
            break
    return text


_COMMANDS = (
    ("/clear", "Reset the conversation"),
    ("/undo", "Revert the last file change made by the agent"),
    ("/model <name>", "Switch model, e.g. qwen3.5:9b or ollama/qwen3.5:9b"),
    ("/tools", "List available tools"),
    ("/memory [query]", "Show or search memories"),
    ("/forget <key>", "Delete a memory"),
    ("/status", "Show model, tier, sandbox and token usage"),
    ("/sessions", "List saved sessions"),
    ("/load <id>", "Resume a saved session"),
    ("/help", "Show this help"),
)


def _table(rows, title: str, header: tuple[str, ...] | None = None) -> None:
    table = Table(show_header=header is not None, box=None, padding=(0, 2, 0, 0))
    for i, col in enumerate(header or ("", "")):
        table.add_column(col, style="cyan bold" if i == 0 else None, no_wrap=i == 0)
    for row in rows:
        table.add_row(*map(str, row))
    console.print(Panel(table, title=f"[bold]{title}[/bold]", border_style="dim", padding=(1, 1)))


async def _slash(command: str, conversation: list[dict], agent, config: dict) -> None:
    cmd, _, arg = command.partition(" ")
    arg = arg.strip()
    match cmd.lower():
        case "/clear":
            conversation.clear()
            console.print("[dim]Conversation cleared.[/dim]")
        case "/undo":
            console.print(f"[green]{escape(agent.tools.undo())}[/green]")
            if agent.tools.undo_depth:
                console.print(f"[dim]{agent.tools.undo_depth} more change(s) can be undone.[/dim]")
            console.print("[dim]Changes made through the shell can't be undone.[/dim]")
        case "/help":
            _table(_COMMANDS, "Commands")
            console.print("[dim]Multi-line: start with \"\"\" or ''' and end with the same. "
                          "Or end a line with \\ to continue it.[/dim]")
        case "/tools":
            _table([(d["function"]["name"], d["function"]["description"]) for d in agent.tools.definitions()],
                   "Tools")
        case "/memory" if arg:
            console.print(escape(agent.memory.recall(arg)))
        case "/memory":
            recent = agent.memory.recent(10)
            if recent:
                _table([(e["timestamp"][:16], e["key"], e["value"]) for e in recent], "Recent memories",
                       ("Time", "Key", "Value"))
            else:
                console.print("[dim]No memories yet.[/dim]")
        case "/forget" if arg:
            console.print(escape(agent.memory.forget(arg)))
        case "/status":
            p, u = agent.profile, agent.session_usage
            _table([
                ("Model", f"{agent.provider.name}/{agent.provider.model}"),
                ("Tier", f"{p.tier} · {p.context:,}-token budget · {agent.max_iterations} iterations · "
                         f"temperature {agent.temperature}"),
                ("Workspace", agent.tools.workspace),
                ("Sandbox", agent.tools.sandbox_level),
                ("Approval", ", ".join(sorted(agent.tools.confirm)) or "off"),
                ("History", f"{len(conversation)} messages"),
                ("Tokens", f"{u['prompt_tokens']:,} in · {u['completion_tokens']:,} out"),
            ], "Status")
        case "/model" if arg:
            prefix, sep, rest = arg.partition("/")
            # "ollama/qwen3.5:9b" names a provider; "hf.co/org/model" is just a model name.
            provider_name, model = (prefix, rest) if sep and prefix in PROVIDERS else (config["provider"], arg)
            from .providers import create_provider
            try:
                switched = with_defaults({**config, "provider": provider_name})
                switched["providers"][provider_name]["model"] = model
                provider = create_provider(switched)
            except ValueError as e:
                console.print(f"[red]Can't switch: {escape(str(e))}[/red]")
                return
            config.update(switched)
            # Save only this change: the running config also holds per-run overrides.
            raw = read_config() or {}
            raw["provider"] = provider_name
            raw.setdefault("providers", {}).setdefault(provider_name, {})["model"] = model
            save_config(raw)
            await agent.set_provider(provider)
            await _detect_model(agent)
            console.print(f"[green]Switched to {escape(provider_name)}/{escape(model)}[/green] "
                          f"[dim](tier {agent.profile.tier})[/dim]")
        case "/sessions":
            rows = [(s["id"], s["messages"], s["updated"][:16], s["preview"]) for s in list_sessions()[:20]]
            if rows:
                _table(rows, "Sessions", ("ID", "Messages", "Updated", "Preview"))
            else:
                console.print("[dim]No saved sessions.[/dim]")
        case "/load" if arg:
            loaded = load_session(arg)
            if loaded is None:
                console.print(f"[red]Session '{escape(arg)}' not found.[/red]")
                return
            conversation[:] = loaded
            console.print(f"[green]Loaded session {escape(arg)} ({len(loaded)} messages)[/green]")
        case "/model" | "/load" | "/forget":
            console.print(f"[yellow]Usage: {next(c for c, _ in _COMMANDS if c.startswith(cmd))}[/yellow]")
        case _:
            console.print(f"[yellow]Unknown command: {escape(cmd)}. Type /help for commands.[/yellow]")


def _render(text: str, plain: bool) -> None:
    try:
        reply_console.print(text if plain else Markdown(text), markup=False)
    except Exception:
        reply_console.print(text, markup=False)


# ── gateway ──────────────────────────────────────────────────────────


@cli.command()
@click.option("-v", "--verbose", is_flag=True, help="Show debug logs.")
def gateway(verbose: bool):
    """Run the Telegram bot."""
    _setup_logging(verbose)
    asyncio.run(_gateway(_load()))


async def _gateway(config: dict) -> None:
    from .telegram import TelegramGateway, gateway_config

    tg = config["channels"]["telegram"]
    token = tg["token"] or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not (tg["enabled"] and token):
        console.print(f"[red]Telegram is not set up: set channels.telegram.enabled and token in {config_file()} "
                      "(or TELEGRAM_BOT_TOKEN).[/red]")
        raise SystemExit(1)
    config, warnings = gateway_config(config)
    for warning in warnings:
        console.print(Panel(escape(warning), border_style="yellow"))
    agent = _build_agent(config)
    agent.allow_local_images = False
    await _detect_model(agent)
    if agent.tools.confirm:
        console.print(f"[dim]Nobody can approve tool calls here, so these are denied: "
                      f"{', '.join(sorted(agent.tools.confirm))} (tools.confirm).[/dim]")
    bot = TelegramGateway(agent, token, tg["allowFrom"], tg["streamResponses"])
    await bot.start()
    console.print("[bold green]Telegram gateway running.[/bold green] [dim]Ctrl+C stops it.[/dim]")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await bot.stop()
        await agent.close()


# ── doctor ───────────────────────────────────────────────────────────

_OK, _WARN, _FAIL = "[green]✓[/green]", "[yellow]![/yellow]", "[red]✗[/red]"


@cli.command()
def doctor():
    """Check config, provider, model and tools. Exits 1 if something is broken."""
    raise SystemExit(asyncio.run(_doctor()))


async def _doctor() -> int:
    from .loop import AgentLoop
    from .memory import Memory
    from .providers import create_provider

    rows: list[tuple[str, str, str]] = []

    def report() -> int:
        table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
        for mark, check, detail in rows:
            table.add_row(mark, f"[bold]{check}[/bold]", detail)
        console.print(table)
        return 1 if any(mark == _FAIL for mark, _, _ in rows) else 0

    path = config_file()
    try:
        raw = read_config()
    except json.JSONDecodeError as e:
        rows.append((_FAIL, "Config", f"{path} is not valid JSON: {escape(str(e))}"))
        return report()
    if raw is None:
        rows.append((_FAIL, "Config", f"{path} not found; run: agent-mini init"))
        return report()
    if path.stat().st_mode & 0o077:
        rows.append((_WARN, "Config", f"{path} is readable by others; run: chmod 600 {path}"))
    else:
        rows.append((_OK, "Config", str(path)))
    try:
        config = with_defaults(raw)
        # A throwaway memory file: doctor must not touch the real store.
        agent = AgentLoop(create_provider(config), config, Memory(agent_home() / ".doctor-memory.json"))
    except ValueError as e:
        rows.append((_FAIL, "Config values", escape(str(e))))
        return report()

    try:
        tools = agent.tools
        if os.access(tools.workspace, os.W_OK):
            rows.append((_OK, "Workspace", str(tools.workspace)))
        else:
            rows.append((_FAIL, "Workspace", f"{tools.workspace} is not writable"))
        rows.append((_OK, "Sandbox", f"{tools.sandbox_level} · approval: {', '.join(sorted(tools.confirm)) or 'off'}"))
        rows.extend(await _doctor_provider(agent))
        p = agent.profile
        rows.append((_OK, "Budgets", f"tier {p.tier} · {p.context:,}-token budget · "
                                     f"num_ctx {agent.provider.context_window:,} · {agent.max_iterations} iterations"))
        rows.append((_OK, "ripgrep", "installed") if shutil.which("rg") else
                    (_WARN, "ripgrep", "not installed; search_files falls back to grep"))
        tg = config["channels"]["telegram"]
        if tg["enabled"]:
            allow = [str(u) for u in tg["allowFrom"]]
            if not (tg["token"] or os.environ.get("TELEGRAM_BOT_TOKEN")):
                rows.append((_FAIL, "Telegram", "enabled but no token (config or TELEGRAM_BOT_TOKEN)"))
            elif not allow or "*" in allow:
                rows.append((_WARN, "Telegram", "the bot is public: allowFrom is empty"))
            else:
                rows.append((_OK, "Telegram", f"{len(allow)} allowed user(s)"))
    finally:
        await agent.close()
    return report()


async def _doctor_provider(agent) -> list[tuple[str, str, str]]:
    """Reachability, model presence and tool support."""
    provider = agent.provider
    model, base = escape(provider.model), provider.base_url
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            if provider.name == "ollama":
                resp = await client.get(f"{base}/api/version")
            else:
                resp = await client.get(f"{base}/models", headers=provider.headers)
        data = resp.json() if resp.status_code < 400 else {}
    except (httpx.HTTPError, ValueError) as e:
        return [(_FAIL, "Provider", f"{provider.name} at {base} is unreachable ({type(e).__name__})")]

    if provider.name == "ollama":
        rows = [(_OK, "Provider", f"ollama {data.get('version', '?')} at {base}")]
        info = await agent.detect_model()
        if info is None:
            return [*rows, (_FAIL, "Model", f"{model} not found; run: ollama pull {model}")]
        facts = [model] + ([f"{info.params_b:.3g}B params"] if info.params_b else []) + \
                ([f"{info.context_length:,}-token context"] if info.context_length else [])
        rows.append((_OK, "Model", " · ".join(facts)))
        if info.capabilities is not None:
            rows.append((_OK, "Tool calling", "supported") if "tools" in info.capabilities else
                        (_FAIL, "Tool calling", f"{model} has no tool support; pick another model"))
        return rows
    if resp.status_code in (401, 403):
        return [(_FAIL, "Provider", f"{base} rejected the API key (HTTP {resp.status_code})")]
    if resp.status_code >= 400:
        return [(_WARN, "Provider", f"{base} answered HTTP {resp.status_code} for /models")]
    ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
    known = not ids or provider.model in ids
    return [(_OK, "Provider", f"{provider.name} at {base}"),
            (_OK, "Model", model) if known else (_WARN, "Model", f"{model} is not in the server's model list")]

