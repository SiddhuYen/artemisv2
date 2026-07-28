"""Brave Search REST API provider (PRIMARY web search).

API key from BRAVE_API_KEY. Respects a per-second rate limit (free tier ~1 q/s)
and a best-effort monthly quota; on quota/credit exhaustion (HTTP 429/402) it
marks itself unavailable for the run so the orchestrator falls back. Every
response is cached by the base class.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import List

from .. import config
from . import cache
from .base import SearchProvider, SearchResult, _call_with_hard_timeout, make_client
from .ratelimit import IntervalLimiter
from .stats import STATS


def _do_request(query: str):
    with make_client() as c:
        return c.get(
            config.BRAVE_ENDPOINT,
            headers={"X-Subscription-Token": config.BRAVE_API_KEY,
                     "Accept": "application/json"},
            params={"q": query, "count": config.RESULTS_PER_QUERY},
        )


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _state_key() -> str:
    return cache.make_key("bravestate", _current_month(), "s")


def _mark_state(state: str) -> None:
    """Persist a month-scoped Brave outage state (survives across runs/processes)
    so the UI can warn users that results are degraded."""
    try:
        cache.set(_state_key(), "bravestate", {"state": state}, 40 * 86400)
    except Exception:
        pass


def brave_status() -> dict:
    """Brave availability for the UI: monthly usage + any outage state.
    state one of: ok | exhausted | invalid_key | not_configured."""
    used = cache.get_counter(cache.make_key("bravequota", _current_month(), "count"))
    quota = config.BRAVE_MONTHLY_QUOTA
    persisted = (cache.get(_state_key(), track=False) or {}).get("state")
    if not config.BRAVE_API_KEY:
        state = "not_configured"
    elif persisted in ("exhausted", "invalid_key"):
        state = persisted
    elif used >= quota:
        state = "exhausted"
    else:
        state = "ok"
    return {"ok": state == "ok", "state": state,
            "used": used, "quota": quota, "remaining": max(0, quota - used)}


class BraveProvider(SearchProvider):
    name = "brave"
    cache_ttl = config.CACHE_TTL_SEARCH

    def __init__(self) -> None:
        interval = 1.0 / config.BRAVE_QPS if config.BRAVE_QPS > 0 else 0.0
        self._limiter = IntervalLimiter(interval)
        self._lock = threading.Lock()
        # PERSISTENT monthly quota: survives across runs so the free tier can't
        # be silently exhausted a few runs at a time (keyed by UTC year-month).
        self._quota_key = cache.make_key("bravequota", _current_month(), "count")
        self._used = cache.get_counter(self._quota_key)
        self._exhausted = False

    def available(self) -> bool:
        return bool(config.BRAVE_API_KEY) and not self._exhausted \
            and self._used < config.BRAVE_MONTHLY_QUOTA

    def _search_uncached(self, query: str) -> List[SearchResult]:
        if not self.available():
            return []
        self._limiter.acquire()
        start = time.monotonic()
        try:
            # Bypasses request_with_retry deliberately -- Brave's own status
            # codes mean something request_with_retry's generic retry-on-429
            # would get wrong (429/402 here means QUOTA EXHAUSTED, not
            # "transient, retry me"; see below). The hard-timeout backstop
            # still applies: measured live, a sibling provider (DuckDuckGo)
            # hung in raw socket.connect() well past httpx's configured
            # timeout, and this direct-call pattern is equally exposed.
            resp = _call_with_hard_timeout(
                lambda: _do_request(query),
                config.HTTP_TIMEOUT + config.HTTP_HARD_TIMEOUT_BUFFER)
        except Exception:
            return []  # no response reached us -- don't burn quota on a network failure
        STATS.record_call(self.name, time.monotonic() - start)
        with self._lock:
            self._used = cache.incr_counter(self._quota_key)

        if resp.status_code in (401, 403):
            self._exhausted = True  # bad/expired key -> stop trying this run
            _mark_state("invalid_key")
            return []
        if resp.status_code in (429, 402):
            self._exhausted = True  # rate/quota/credit exhausted -> fall back
            _mark_state("exhausted")
            return []
        if resp.status_code != 200:
            return []
        _mark_state("ok")  # clear any prior degraded state now that a call succeeded

        out: List[SearchResult] = []
        try:
            for item in (resp.json().get("web", {}) or {}).get("results", []) or []:
                title = item.get("title", "") or ""
                url = item.get("url", "") or ""
                snippet = item.get("description", "") or ""
                if url and title:
                    out.append(SearchResult(_clean(title), url, _clean(snippet), self.name))
        except Exception:
            return []
        return out[: config.RESULTS_PER_QUERY]


def _clean(s: str) -> str:
    # Brave descriptions can contain <strong> highlight tags.
    from bs4 import BeautifulSoup
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
