"""WhatsApp channel — bridges to a Node.js whatsapp-web.js process."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path

import httpx

from .base import BaseChannel, MessageHandler

log = logging.getLogger("agent-mini")

# Default ports for the bridge ↔ agent communication
_BRIDGE_PORT = 18902  # Node.js bridge listens here (send API)
_WEBHOOK_PORT = 18901  # Python webhook server listens here (receive)

# Default bridge location (relative to package root)
_BRIDGE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "whatsapp-bridge"
_PACKAGED_BRIDGE_DIR = Path(__file__).resolve().parent.parent / "whatsapp_bridge"


class WhatsAppChannel(BaseChannel):
    """WhatsApp integration via a local whatsapp-web.js Node.js bridge.

    Communication flow::

        WhatsApp ←→ whatsapp-web.js (Node) ←→ HTTP ←→ Agent (Python)

    The bridge sends incoming messages to a Python webhook and exposes
    a ``POST /send`` endpoint for outgoing messages.

    Requires ``aiohttp`` (for the webhook) and **Node.js ≥ 18**.
    Install with::

        pip install agent-mini[whatsapp]
    """

    def __init__(
        self,
        allow_from: list[str] | None = None,
        bridge_dir: str | None = None,
    ):
        self._allow_from: set[str] | None = (
            set(allow_from) if allow_from else None
        )
        self._bridge_dir = Path(bridge_dir) if bridge_dir else _BRIDGE_DIR
        self._handler: MessageHandler | None = None
        self._bridge_proc: subprocess.Popen | None = None
        self._runner = None  # aiohttp.web.AppRunner
        self._client = httpx.AsyncClient(timeout=30)

    @property
    def name(self) -> str:
        return "whatsapp"

    def _is_allowed(self, phone: str) -> bool:
        if not self._allow_from or "*" in self._allow_from:
            return True
        return phone in self._allow_from

    # ------------------------------------------------------------------
    # Webhook (receives messages from the bridge)
    # ------------------------------------------------------------------

    async def _webhook_handler(self, request):
        """Handle POST /webhook from the Node.js bridge."""
        from aiohttp import web

        try:
            data = await request.json()
            phone = data.get("from", "")
            text = data.get("body", "")

            if not text or not self._is_allowed(phone):
                return web.json_response({"ok": True})

            log.info("[whatsapp] %s: %s", phone, text[:120])

            if self._handler:
                response = await self._handler("whatsapp", phone, text, None)
                await self.send(phone, response)

            return web.json_response({"ok": True})
        except Exception as e:
            log.error("[whatsapp] Webhook error: %s", e)
            from aiohttp import web as _web

            return _web.json_response({"error": str(e)}, status=500)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, on_message: MessageHandler) -> None:
        from aiohttp import web

        self._handler = on_message
        bridge_dir = self._resolve_bridge_dir()

        # 1. Start the Python webhook server
        app = web.Application()
        app.router.add_post("/webhook", self._webhook_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", _WEBHOOK_PORT)
        await site.start()
        self._runner = runner
        log.info(
            "[whatsapp] Webhook listening on http://localhost:%d", _WEBHOOK_PORT
        )

        # 2. Choose bridge entrypoint:
        #    - Prefer prebuilt bundle if packaged with Python wheel.
        #    - Fallback to source mode (requires npm install).
        bundle_entry = bridge_dir / "dist" / "index.cjs"
        source_entry = bridge_dir / "index.js"
        if bundle_entry.exists():
            entrypoint = bundle_entry
            log.info("[whatsapp] Using bundled bridge: %s", bundle_entry)
        else:
            entrypoint = source_entry
            if not (bridge_dir / "node_modules").exists():
                log.info("[whatsapp] Installing bridge dependencies (npm install)…")
                subprocess.run(
                    ["npm", "install"],
                    cwd=str(bridge_dir),
                    env={**os.environ, "PUPPETEER_SKIP_DOWNLOAD": "true"},
                    check=True,
                    capture_output=True,
                )

        # 3. Launch the bridge process
        env = {
            **os.environ,
            "WEBHOOK_URL": f"http://localhost:{_WEBHOOK_PORT}/webhook",
            "PORT": str(_BRIDGE_PORT),
        }
        self._bridge_proc = subprocess.Popen(
            ["node", str(entrypoint)],
            cwd=str(bridge_dir),
            env=env,
        )
        log.info(
            "[whatsapp] Bridge started (PID %d) on port %d",
            self._bridge_proc.pid,
            _BRIDGE_PORT,
        )

    async def send(self, user_id: str, text: str) -> None:
        """Send a message via the bridge HTTP API."""
        try:
            # Split long messages (WhatsApp has a practical ~65k limit)
            for i in range(0, len(text), 60_000):
                await self._client.post(
                    f"http://localhost:{_BRIDGE_PORT}/send",
                    json={"to": user_id, "message": text[i : i + 60_000]},
                )
        except Exception as e:
            log.error("[whatsapp] Send error: %s", e)

    async def stop(self) -> None:
        if self._bridge_proc:
            log.info("[whatsapp] Stopping bridge (PID %d)…", self._bridge_proc.pid)
            self._bridge_proc.terminate()
            try:
                self._bridge_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._bridge_proc.kill()
        if self._runner:
            await self._runner.cleanup()

    def _resolve_bridge_dir(self) -> Path:
        """Resolve bridge directory from configured, repo, or packaged paths."""
        if self._bridge_dir.exists():
            return self._bridge_dir
        if _PACKAGED_BRIDGE_DIR.exists():
            return _PACKAGED_BRIDGE_DIR
        raise FileNotFoundError(
            "WhatsApp bridge directory not found. "
            "Set channels.whatsapp.bridgeDir or reinstall package."
        )
