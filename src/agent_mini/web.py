"""Web tools: DuckDuckGo search and page fetching, stdlib HTML parsing only."""

from __future__ import annotations

import asyncio
import ipaddress
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import unquote, urlparse

import httpx

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
MAX_REDIRECTS = 5
MAX_FETCH_BYTES = 2_000_000


def untrusted(source: str, text: str) -> str:
    """Mark external content so the model can tell data from instructions."""
    return f'<untrusted_content source="{source.replace(chr(34), "%22")}">\n{text}\n</untrusted_content>'


# ── HTML → text ──────────────────────────────────────────────────────


class _HTMLToText(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript", "svg", "head"})
    _BLOCK = frozenset({"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "blockquote", "section", "article"})

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._skip += tag in self._SKIP
        if tag == "br" or tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _HTMLToText()
    parser.feed(html)
    text = re.sub(r"\n{3,}", "\n\n", "".join(parser.parts))
    return re.sub(r"[ \t]+", " ", text).strip()


# ── Search ───────────────────────────────────────────────────────────


def _strip_tags(fragment: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _unwrap(raw: str) -> str:
    """DuckDuckGo wraps result links in a redirect: pull out the real URL."""
    match = re.search(r"[?&]uddg=([^&]+)", raw)
    return unquote(match.group(1)) if match else unescape(raw)


def _external(url: str) -> bool:
    return url.startswith("http") and "duckduckgo.com" not in url


def parse_ddg_lite(page: str) -> list[dict[str, str]]:
    """Results from lite.duckduckgo.com (table markup)."""
    anchors = re.findall(r"<a\b([^>]*class=['\"]?result-link['\"]?[^>]*)>(.*?)</a>", page, re.DOTALL | re.IGNORECASE)
    snippets = re.findall(r"class=['\"]?result-snippet['\"]?[^>]*>(.*?)</td>", page, re.DOTALL | re.IGNORECASE)
    results = []
    for i, (attrs, title) in enumerate(anchors):
        href = re.search(r"href=['\"]([^'\"]+)", attrs)
        url = _unwrap(href.group(1)) if href else ""
        if _external(url):
            snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
            results.append({"title": _strip_tags(title) or url, "url": url, "snippet": snippet})
    return results[:8]


def parse_ddg_html(page: str) -> list[dict[str, str]]:
    """Results from html.duckduckgo.com (div markup)."""
    blocks = re.findall(
        r'<div[^>]*class="[^"]*result[_ ]results_links[^"]*"[^>]*>(.*?)</div>\s*</div>', page, re.DOTALL
    ) or re.findall(
        r'<div[^>]*class="[^"]*links_main[^"]*"[^>]*>(.*?)(?=<div[^>]*class="[^"]*links_main|$)', page, re.DOTALL
    )
    results = []
    for block in blocks:
        href = re.search(r'href="([^"]+)"', block)
        url = _unwrap(href.group(1)) if href else ""
        if not _external(url):
            continue
        title = re.search(r'class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', block, re.DOTALL)
        snippet = re.search(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</[at]>', block, re.DOTALL)
        results.append({
            "title": _strip_tags(title.group(1)) if title else url,
            "url": url,
            "snippet": _strip_tags(snippet.group(1)) if snippet else "",
        })
    return results[:8]


# Served instead of results when DuckDuckGo rate-limits or challenges the
# client (common behind corporate proxies and VPNs).
_BLOCKED = ("anomaly-modal", "unfortunately, bots use duckduckgo", "detected unusual activity",
            "challenge-form", "please try again later")
_EMPTY = ("no results", "not many great matches")

# The lite endpoint is smaller and survives rate limiting more often.
_ENDPOINTS = (
    ("https://lite.duckduckgo.com/lite/", parse_ddg_lite),
    ("https://html.duckduckgo.com/html/", parse_ddg_html),
)


async def web_search(client: httpx.AsyncClient, query: str) -> str:
    """Search DuckDuckGo (free, no key).

    Failures say not to retry: small models otherwise repeat the same query
    until they run out of iterations.
    """
    failures = []
    for url, parse in _ENDPOINTS:
        host = urlparse(url).hostname
        try:
            resp = await client.post(url, data={"q": query}, headers={"User-Agent": _BROWSER_UA},
                                     follow_redirects=True, timeout=15)
        except httpx.HTTPError as e:
            failures.append(f"{host}: {type(e).__name__}")
            continue
        if resp.status_code >= 400:
            failures.append(f"{host}: HTTP {resp.status_code}")
            continue
        head = resp.text[:8000].lower()
        if any(marker in head for marker in _BLOCKED):
            failures.append(f"{host}: rate-limited or blocked by a network filter")
            continue
        if results := parse(resp.text):
            return untrusted("web_search", "\n\n".join(
                f"{r['title']}\n{r['url']}" + (f"\n{r['snippet']}" if r["snippet"] else "") for r in results
            ))
        if any(marker in head for marker in _EMPTY):
            return f"No results found for '{query}'. Try different or broader keywords."
        failures.append(f"{host}: response had no parsable results")
    return (
        f"Error: web_search is unavailable ({'; '.join(failures)}). Do NOT retry it. "
        "Use web_fetch on a URL you already know, or answer from your own knowledge "
        "and say it may be out of date."
    )


# ── Fetch ────────────────────────────────────────────────────────────


def _private(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Not a public address: private, loopback, link-local, CGNAT (100.64/10, some cloud metadata), multicast, …"""
    return not ip.is_global or ip.is_multicast


async def ssrf_error(url: str) -> str | None:
    """Why *url* must not be fetched (non-http, or a private/loopback host), or None.

    Anything not globally routable is refused: localhost, RFC-1918, link-local,
    CGNAT and cloud metadata addresses. DNS rebinding can still get through.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"scheme '{parsed.scheme or '?'}' is not allowed (use http/https)"
    host = parsed.hostname
    if not host:
        return "URL missing hostname"
    try:
        if _private(ipaddress.ip_address(host)):
            return f"host {host} is a private/loopback/link-local address"
        return None
    except ValueError:
        pass
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError as e:
        return f"cannot resolve host {host}: {e}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _private(ip):
            return f"host {host} resolves to private/loopback address {ip}"
    return None


async def web_fetch(client: httpx.AsyncClient, url: str) -> str:
    """Fetch *url* as text. Redirects are followed by hand so each hop is checked before it is requested."""
    headers = {"User-Agent": _BROWSER_UA, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}
    for hop in range(MAX_REDIRECTS + 1):
        if blocked := await ssrf_error(url):
            return f"Error: {blocked}" + (" (after redirect)" if hop else "")
        async with client.stream("GET", url, headers=headers, timeout=20) as resp:
            if resp.is_redirect and resp.headers.get("location"):
                url = str(resp.url.join(resp.headers["location"]))
                continue
            if resp.status_code >= 400:
                return f"Error: HTTP {resp.status_code} fetching {url}"
            content_type = resp.headers.get("content-type", "").lower()
            body = b""
            async for chunk in resp.aiter_bytes():
                body += chunk
                if len(body) >= MAX_FETCH_BYTES:
                    break
            text = body[:MAX_FETCH_BYTES].decode(resp.encoding or "utf-8", errors="replace")
        break
    else:
        return f"Error: too many redirects (more than {MAX_REDIRECTS})"
    if "html" in content_type:
        text = html_to_text(text)
        if not text:
            return "(page returned no readable text)"
    return untrusted(url, text)
