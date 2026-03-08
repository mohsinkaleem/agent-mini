"""GitHub Copilot provider — uses OAuth device-flow for authentication."""

from __future__ import annotations

import json
import time

import httpx

from .base import (
    BaseProvider,
    ChatResponse,
    StreamCallback,
    ToolCall,
    parse_arguments,
    parse_openai_tool_calls,
)

_COPILOT_API = "https://api.githubcopilot.com"
_GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"  # VS Code Copilot client ID


class GitHubCopilotProvider(BaseProvider):
    """Calls the GitHub Copilot chat completions endpoint.

    Authentication can be done in two ways:

    1. **Pre-existing token** — pass ``token`` directly (e.g. from a prior
       device-flow login stored in config).
    2. **Device flow** — call ``login()`` to interactively authenticate,
       then the token is cached for subsequent calls.
    """

    def __init__(self, token: str = "", model: str = "gpt-4o"):
        self._github_token = token
        self._copilot_token: str | None = None
        self._copilot_token_expires: float = 0
        self._model = model
        self._client = httpx.AsyncClient(timeout=300)

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def name(self) -> str:
        return "github_copilot"

    @property
    def model_name(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # OAuth device-flow login
    # ------------------------------------------------------------------

    @staticmethod
    def device_flow_login() -> str:
        """Interactive OAuth device-flow.  Returns a GitHub access token.

        This is a *blocking* helper meant to be called from a CLI command
        before the async event loop starts.
        """
        with httpx.Client() as client:
            # Step 1 — request device & user codes
            resp = client.post(
                "https://github.com/login/device/code",
                data={
                    "client_id": _GITHUB_CLIENT_ID,
                    "scope": "read:user",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

            device_code = data["device_code"]
            user_code = data["user_code"]
            verification_uri = data["verification_uri"]
            interval = data.get("interval", 5)

            print(f"\n  Open: {verification_uri}")
            print(f"  Enter code: {user_code}\n")
            print("  Waiting for authorization…")

            # Step 2 — poll for access token
            while True:
                time.sleep(interval)
                resp = client.post(
                    "https://github.com/login/oauth/access_token",
                    data={
                        "client_id": _GITHUB_CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers={"Accept": "application/json"},
                )
                resp.raise_for_status()
                token_data = resp.json()

                if "access_token" in token_data:
                    return token_data["access_token"]

                error = token_data.get("error")
                if error == "authorization_pending":
                    continue
                if error == "slow_down":
                    interval += 5
                    continue
                raise RuntimeError(f"OAuth error: {error}")

    # ------------------------------------------------------------------
    # Copilot session token (short-lived)
    # ------------------------------------------------------------------

    async def _ensure_copilot_token(self) -> str:
        """Exchange the long-lived GitHub token for a short-lived Copilot token."""
        if self._copilot_token and time.time() < self._copilot_token_expires - 60:
            return self._copilot_token

        resp = await self._client.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers={
                "Authorization": f"token {self._github_token}",
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._copilot_token = data["token"]
        self._copilot_token_expires = data.get("expires_at", time.time() + 1800)
        return self._copilot_token  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ChatResponse:
        copilot_token = await self._ensure_copilot_token()

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        resp = await self._client.post(
            f"{_COPILOT_API}/chat/completions",
            headers={
                "Authorization": f"Bearer {copilot_token}",
                "Content-Type": "application/json",
                "Editor-Version": "vscode/1.95.0",
                "Copilot-Integration-Id": "agent-mini",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or None
        tool_calls = parse_openai_tool_calls(msg.get("tool_calls"))
        usage = data.get("usage")

        return ChatResponse(content=content, tool_calls=tool_calls, usage=usage)

    async def chat_stream(
        self,
        messages: list[dict],
        on_delta: StreamCallback,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        on_thinking: StreamCallback | None = None,
    ) -> ChatResponse:
        copilot_token = await self._ensure_copilot_token()

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        content_parts: list[str] = []
        tool_calls_by_idx: dict[int, dict] = {}

        async with self._client.stream(
            "POST",
            f"{_COPILOT_API}/chat/completions",
            headers={
                "Authorization": f"Bearer {copilot_token}",
                "Content-Type": "application/json",
                "Editor-Version": "vscode/1.95.0",
                "Copilot-Integration-Id": "agent-mini",
            },
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                if raw.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})

                text = delta.get("content") or ""
                if text:
                    content_parts.append(text)
                    await on_delta(text)

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_by_idx:
                        tool_calls_by_idx[idx] = {
                            "id": tc.get("id", f"call_{idx}"),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": "",
                        }
                    if tc.get("function", {}).get("name"):
                        tool_calls_by_idx[idx]["name"] = tc["function"]["name"]
                    if tc.get("function", {}).get("arguments"):
                        tool_calls_by_idx[idx]["arguments"] += tc["function"]["arguments"]
                    if tc.get("id"):
                        tool_calls_by_idx[idx]["id"] = tc["id"]

        content = "".join(content_parts) or None
        tcs: list[ToolCall] | None = None
        if tool_calls_by_idx:
            tcs = [
                ToolCall(
                    id=v["id"],
                    name=v["name"],
                    arguments=parse_arguments(v["arguments"]),
                )
                for v in sorted(tool_calls_by_idx.values(), key=lambda x: x["id"])
            ]
        return ChatResponse(content=content, tool_calls=tcs)
