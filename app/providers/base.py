"""Pluggable search-provider interface + shared HTTP (retry + Retry-After).

    class SearchProvider:
        name: str
        search(query) -> list[SearchResult]
        fetch(url)    -> Page

Concrete providers: BraveProvider (primary), WikipediaProvider + WikidataProvider
(structured), DuckDuckGoProvider (fallback). The orchestrator routes between them.
"""
from __future__ import annotations

import abc
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from .. import config
from ..utils.htmltext import strip_inline_noise
from . import cache
from .stats import STATS

# Process-wide cap on outbound requests in flight at once -- see
# config.MAX_CONCURRENT_REQUESTS for why this exists and why it's global
# rather than per-provider or per-caller.
_REQUEST_SLOTS = threading.Semaphore(config.MAX_CONCURRENT_REQUESTS)


class HardTimeout(Exception):
    """A call exceeded _call_with_hard_timeout's deadline. Treated as any
    other request failure by request_with_retry's except clause -- this is
    not a new failure MODE for callers to handle, just a way to guarantee
    request_with_retry actually returns."""


def _call_with_hard_timeout(fn, timeout: float):
    """Run `fn()` on a background daemon thread; return its result or raise
    HardTimeout if it hasn't finished within `timeout` seconds.

    Exists because httpx's own `timeout=` is not trustworthy in every
    environment: measured live, a DuckDuckGo call sat in raw
    socket.connect() for 40+ seconds -- well past its configured 8s timeout
    -- because the host was unreachable in a way that never produced a
    response httpx could time out ON (packets silently dropped rather than
    rejected). This function is the backstop: it does not depend on httpx,
    the OS, or the network behaving.

    The thread is NOT killed on timeout -- Python has no safe way to do that
    -- it's abandoned as a daemon and keeps running (still stuck in
    socket.connect() until the OS eventually gives up, however long that
    takes). Deliberately accepted: a leaked daemon thread costs nothing but
    memory and doesn't block process shutdown, whereas the alternative --
    the calling thread hanging forever -- costs the entire request that
    thread was serving, with no way to recover it at all.
    """
    result_q: "queue.Queue" = queue.Queue(maxsize=1)

    def runner():
        try:
            result_q.put(("ok", fn()))
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, re-raised below
            result_q.put(("error", exc))

    threading.Thread(target=runner, daemon=True).start()
    try:
        status, value = result_q.get(timeout=timeout)
    except queue.Empty:
        raise HardTimeout(f"call exceeded hard timeout of {timeout}s")
    if status == "error":
        raise value
    return value


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url,
                "snippet": self.snippet, "provider": self.provider}

    @staticmethod
    def from_dict(d: dict) -> "SearchResult":
        return SearchResult(d.get("title", ""), d.get("url", ""),
                            d.get("snippet", ""), d.get("provider", ""))


@dataclass
class Page:
    url: str
    content: str = ""          # raw HTML or plain text
    status_code: int = 0
    from_cache: bool = False
    meta: Dict[str, object] = field(default_factory=dict)


def _normalize_cache_query(query: str) -> str:
    """Case-fold + collapse whitespace for the cache key only -- the provider
    still receives the original query verbatim. Near-duplicate queries
    ("Elon Musk" / "elon musk" / a double space) would otherwise bypass the
    cache and trigger a redundant network request each time."""
    return " ".join((query or "").strip().lower().split())


def _normalize_cache_url(url: str) -> str:
    """Strip whitespace and a URL fragment for the cache key only -- a
    fragment never changes what the server returns, so "url#comments" and
    "url#top" would otherwise be cached (and fetched) as unrelated pages."""
    return (url or "").strip().split("#", 1)[0]


def make_client() -> httpx.Client:
    transport = httpx.HTTPTransport(retries=1)  # connection-level only
    return httpx.Client(
        headers={"User-Agent": config.USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=config.HTTP_TIMEOUT,
        follow_redirects=True,
        transport=transport,
    )


def _do_single_request(method: str, url: str, **kwargs) -> httpx.Response:
    with make_client() as c:
        return c.request(method, url, **kwargs)


def _retry_after_seconds(resp: httpx.Response) -> Optional[float]:
    val = resp.headers.get("Retry-After")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None  # HTTP-date form: ignore, fall back to backoff


def request_with_retry(
    method: str,
    url: str,
    *,
    provider: str,
    limiter=None,
    breaker=None,
    **kwargs,
) -> Optional[httpx.Response]:
    """HTTP with: a global concurrent-request cap, provider rate limiting,
    circuit breaker, jittered exponential backoff on 429/5xx only, and
    Retry-After honoring. Records latency in STATS. Returns None when all
    attempts fail or the breaker is open.

    Holds ONE _REQUEST_SLOTS permit for the ENTIRE call -- every retry and
    backoff sleep included -- not just the moment of the actual request. A
    slot released between retries would let a fresh, unrelated request pile
    onto an already-struggling host while this one waits its turn to try
    again; holding it instead makes MAX_CONCURRENT_REQUESTS a true ceiling on
    "requests being attempted", which is the number that actually matters to
    a rate-limited provider, not just "requests on the wire this instant"."""
    if breaker is not None and not breaker.allow():
        return None

    with _REQUEST_SLOTS:
        last: Optional[httpx.Response] = None
        for attempt in range(config.HTTP_RETRIES + 1):
            if limiter is not None:
                limiter.acquire()
            start = time.monotonic()
            try:
                resp = _call_with_hard_timeout(
                    lambda: _do_single_request(method, url, **kwargs),
                    config.HTTP_TIMEOUT + config.HTTP_HARD_TIMEOUT_BUFFER)
            except Exception:
                if breaker is not None:
                    breaker.record_failure()
                _sleep_backoff(attempt)
                continue
            STATS.record_call(provider, time.monotonic() - start)

            if resp.status_code in config.HTTP_RETRY_STATUS:
                last = resp
                if breaker is not None:
                    breaker.record_failure()
                wait = _retry_after_seconds(resp)
                if attempt < config.HTTP_RETRIES:
                    if wait is not None:
                        time.sleep(min(wait, 30.0) + _jitter())
                    else:
                        _sleep_backoff(attempt)
                continue

            if breaker is not None:
                breaker.record_success()
            return resp
        return last


def _jitter() -> float:
    return random.uniform(0, config.HTTP_BACKOFF_JITTER)


def _sleep_backoff(attempt: int) -> None:
    if attempt < config.HTTP_RETRIES:
        time.sleep(config.HTTP_BACKOFF_BASE * (2 ** attempt) + _jitter())


def fetch_page(url: str, render: bool = False) -> Page:
    """Cache-first page fetch, shared across providers and keyed by URL.

    `render=True` uses a headless browser (see browser.py) to execute the
    page's JavaScript, recovering the DOM of a single-page-app team roster
    that a plain GET returns empty. Rendered and plain results are cached
    under SEPARATE keys, so a caller can cheaply try the plain fetch first
    and only pay the render cost when it came back empty. A failed render is
    never cached — a transient browser hiccup must not disable a page for
    CACHE_TTL_PAGE days. Rendering is fully optional: when Playwright isn't
    installed, `render=True` degrades to ("", 0) and the caller's own
    fallback logic uses the plain fetch instead.
    """
    kind = "rendered" if render else "fetch"
    key = cache.make_key("page", kind, _normalize_cache_url(url))
    cached = cache.get(key)
    if cached is not None:
        return Page(url=url, content=cached.get("content", ""),
                    status_code=cached.get("status_code", 0), from_cache=True)

    if render:
        from .browser import render_html
        content, status = render_html(url)
        if content:
            cache.set(key, "page", {"content": content, "status_code": status},
                      config.CACHE_TTL_PAGE)
        return Page(url=url, content=content, status_code=status, from_cache=False)

    resp = request_with_retry("GET", url, provider="fetch")
    content, status = "", 0
    if resp is not None:
        status = resp.status_code
        if status == 200:
            # Strip inline script/style bodies BEFORE truncating. On a modern
            # page those routinely fill the first several hundred KB, so the
            # cap was being spent on markup that cannot contain a roster:
            # uncorkcapital.com/team yielded 0 names truncated and 14 whole,
            # bvp.com/team 65 vs 252. Stripping first also roughly halves what
            # goes into the page cache. JSON-LD is preserved -- see
            # htmltext.strip_inline_noise.
            content = strip_inline_noise(resp.text)[: config.MAX_HTML_CHARS]
    cache.set(key, "page", {"content": content, "status_code": status},
              config.CACHE_TTL_PAGE)
    return Page(url=url, content=content, status_code=status, from_cache=False)


class SearchProvider(abc.ABC):
    name: str = "base"
    cache_ttl: int = config.CACHE_TTL_SEARCH

    @abc.abstractmethod
    def _search_uncached(self, query: str) -> List[SearchResult]:
        ...

    def available(self) -> bool:
        return True

    def search(self, query: str) -> List[SearchResult]:
        """Cache-first search. Never repeats an identical query within TTL."""
        key = cache.make_key(self.name, "search", _normalize_cache_query(query))
        cached = cache.get(key)
        if cached is not None:
            return [SearchResult.from_dict(d) for d in cached]
        results = self._search_uncached(query)
        cache.set(key, "search", [r.to_dict() for r in results], self.cache_ttl)
        return results

    def fetch(self, url: str) -> Page:
        """Cache-first page fetch (shared across providers, keyed by URL)."""
        return fetch_page(url)
