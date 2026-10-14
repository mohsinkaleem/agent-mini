"""Web search parsing, SSRF guard and fetching."""

from pathlib import Path
from urllib.parse import quote

import httpx
import pytest

from agent_mini.memory import Memory
from agent_mini.tools import Tools
from agent_mini.web import html_to_text


def _tools(workspace: Path, memory: Memory, handler) -> Tools:
    t = Tools({"workspace": str(workspace)}, memory)
    t._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return t


def _page(text: str):
    return lambda request: httpx.Response(200, text=text)


async def test_search_parses_lite_results(workspace, memory):
    target = quote("https://example.com/btc", safe="")
    page = (f"<table><tr><td><a rel='nofollow' href='//duckduckgo.com/l/?uddg={target}' "
            "class='result-link'>Bitcoin &amp; USD</a></td></tr>"
            "<tr><td class='result-snippet'>Live price</td></tr></table>")
    result = await _tools(workspace, memory, _page(page)).execute("web_search", {"query": "bitcoin"})
    assert "Bitcoin & USD\nhttps://example.com/btc\nLive price" in result
    assert result.startswith("<untrusted_content")


async def test_blocked_search_says_do_not_retry(workspace, memory):
    page = "<html><body><div class='anomaly-modal'>…</div></body></html>"
    result = await _tools(workspace, memory, _page(page)).execute("web_search", {"query": "q"})
    assert result.startswith("Error:") and "Do NOT retry" in result


async def test_empty_search_is_not_an_error(workspace, memory):
    page = "<html><body>No results found for that query.</body></html>"
    result = await _tools(workspace, memory, _page(page)).execute("web_search", {"query": "zzzqqq"})
    assert result.startswith("No results found")


@pytest.mark.parametrize("url, reason", [
    ("http://localhost:8080/admin", "loopback"),
    ("http://169.254.169.254/latest/meta-data/", "link-local"),
    ("http://192.168.1.1/", "private"),
    ("http://[::1]/", "private"),
    ("http://100.100.100.200/latest/meta-data/", "private"),  # CGNAT; Alibaba Cloud metadata
    ("http://0.0.0.0/", "private"),
    ("file:///etc/passwd", "scheme"),
])
async def test_fetch_refuses_private_hosts(tools, url, reason):
    result = await tools.execute("web_fetch", {"url": url})
    assert result.startswith("Error:") and reason in result


async def test_fetch_checks_every_redirect_before_requesting(workspace, memory):
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    result = await _tools(workspace, memory, handler).execute("web_fetch", {"url": "http://93.184.216.34/start"})
    assert result.startswith("Error:") and "after redirect" in result
    assert requested == ["http://93.184.216.34/start"]


async def test_fetch_marks_content_untrusted_and_strips_html(workspace, memory):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><head><title>t</title></head><body><p>Hello</p><script>x()</script>"
                                        "</body></html>", headers={"content-type": "text/html"})

    result = await _tools(workspace, memory, handler).execute("web_fetch", {"url": "http://93.184.216.34/page"})
    assert result == '<untrusted_content source="http://93.184.216.34/page">\nHello\n</untrusted_content>'


def test_html_to_text():
    assert html_to_text("<div>a</div><div>b<br>c</div><style>x</style>") == "a\n\nb\nc"
