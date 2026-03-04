"""CLI entry-point — chat, gateway, init, status, login."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown

from . import __version__
from .config import (
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_WORKSPACE,
    MEMORY_FILE,
    load_config,
    save_config,
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
    """Initialise config and workspace."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_WORKSPACE.mkdir(parents=True, exist_ok=True)

    if CONFIG_FILE.exists():
        console.print(f"[yellow]Config already exists at {CONFIG_FILE}[/yellow]")
    else:
        default = {
            "provider": "ollama",
            "providers": {
                "ollama": {
                    "baseUrl": "http://localhost:11434",
                    "model": "llama3.1",
                    "think": False,
                },
                "gemini": {"apiKey": "", "model": "gemini-2.0-flash"},
                "github_copilot": {"token": "", "model": "gpt-4o"},
                "local": {
                    "baseUrl": "http://localhost:8080/v1",
                    "apiKey": "no-key",
                    "model": "local-model",
                },
            },
            "agent": {
                "maxIterations": 20,
                "temperature": 0.7,
                "systemPrompt": "",
            },
            "channels": {
                "telegram": {
                    "enabled": False,
                    "token": "",
                    "allowFrom": [],
                    "streamResponses": True,
                },
                "whatsapp": {"enabled": False, "allowFrom": []},
            },
            "tools": {
                "restrictToWorkspace": False,
            },
            "memory": {"enabled": True, "maxEntries": 1000},
            "workspace": str(DEFAULT_WORKSPACE),
        }
        save_config(default)
        console.print(f"[green]✓ Config created → {CONFIG_FILE}[/green]")

    console.print(f"[green]✓ Workspace → {DEFAULT_WORKSPACE}[/green]")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print(f"  1. Edit config:  [cyan]$EDITOR {CONFIG_FILE}[/cyan]")
    console.print("  2. Set your provider and API keys")
    console.print("  3. Start chatting: [cyan]agent-mini chat[/cyan]")


# ======================================================================
# chat  (interactive or single-shot)
# ======================================================================


@cli.command()
@click.option("-m", "--message", default=None, help="Single message (non-interactive).")
@click.option("-v", "--verbose", is_flag=True, help="Show debug logs.")
@click.option("--no-markdown", is_flag=True, help="Plain-text output.")
def chat(message: str | None, verbose: bool, no_markdown: bool):
    """Chat with the agent."""
    _setup_logging(verbose)
    config = load_config()
    if not config:
        console.print("[red]No config found. Run 'agent-mini init' first.[/red]")
        raise SystemExit(1)
    asyncio.run(_chat(config, message, no_markdown))


async def _chat(config: dict, message: str | None, plain: bool) -> None:
    from .providers import create_provider
    from .agent import AgentLoop, Memory

    provider = create_provider(config)
    memory = Memory(
        MEMORY_FILE,
        max_entries=config.get("memory", {}).get("maxEntries", 1000),
    )
    agent = AgentLoop(provider, config, memory)
    conversation: list[dict] = []

    try:
        console.print(
            f"[bold green]Agent Mini[/bold green] v{__version__}"
            f" • Provider: [cyan]{provider.name}[/cyan]"
            f" ({provider.model_name})"
        )

        if message:
            response = await agent.run(message, conversation)
            _render(response, plain)
            return

        # Interactive REPL
        console.print(
            "[dim]Type 'exit' to quit · '/clear' to reset conversation[/dim]\n"
        )

        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "/exit", "/quit", ":q"}:
                console.print("[dim]Goodbye![/dim]")
                break
            if user_input == "/clear":
                conversation.clear()
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            with console.status("[bold cyan]Thinking…[/bold cyan]"):
                response = await agent.run(user_input, conversation)

            console.print()
            _render(response, plain)
            console.print()
    finally:
        await agent.close()


def _render(text: str, plain: bool) -> None:
    if plain:
        console.print(text)
    else:
        try:
            console.print(Markdown(text))
        except Exception:
            console.print(text)


# ======================================================================
# gateway  (Telegram / WhatsApp)
# ======================================================================


@cli.command()
@click.option("-v", "--verbose", is_flag=True, help="Show debug logs.")
def gateway(verbose: bool):
    """Start the messaging gateway (Telegram / WhatsApp)."""
    _setup_logging(verbose)
    config = load_config()
    if not config:
        console.print("[red]No config found. Run 'agent-mini init' first.[/red]")
        raise SystemExit(1)
    asyncio.run(_gateway(config))


async def _gateway(config: dict) -> None:
    from .providers import create_provider
    from .agent import AgentLoop, Memory
    from .bus import MessageBus
    from .channels import TelegramChannel, WhatsAppChannel

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

    # WhatsApp
    wa = ch_cfg.get("whatsapp", {})
    if wa.get("enabled"):
        channels.append(
            WhatsAppChannel(
                allow_from=wa.get("allowFrom", []),
                bridge_dir=wa.get("bridgeDir"),
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
# login  (GitHub Copilot OAuth device-flow)
# ======================================================================


@cli.command()
@click.argument("provider_name", default="github_copilot")
def login(provider_name: str):
    """Authenticate with a provider (e.g. github_copilot)."""
    if provider_name not in ("github_copilot", "github-copilot"):
        console.print(
            f"[yellow]Login is only needed for github_copilot. "
            f"For {provider_name}, set the API key in config.[/yellow]"
        )
        return

    from .providers.github_copilot import GitHubCopilotProvider

    console.print("[bold]GitHub Copilot — OAuth Device Flow[/bold]\n")
    try:
        token = GitHubCopilotProvider.device_flow_login()
    except Exception as e:
        console.print(f"[red]Login failed: {e}[/red]")
        raise SystemExit(1)

    config = load_config()
    config.setdefault("providers", {}).setdefault("github_copilot", {})["token"] = token
    save_config(config)

    console.print(f"\n[green]✓ Token saved to {CONFIG_FILE}[/green]")
    console.print(
        "  Set [cyan]\"provider\": \"github_copilot\"[/cyan] in config to use it."
    )


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
    console.print(
        f"  Workspace : {config.get('workspace', str(DEFAULT_WORKSPACE))}"
    )
