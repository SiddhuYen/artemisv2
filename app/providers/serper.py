"""Serper.dev (Google SERP) provider — PRIMARY web search.

Cheap Google-backed results (~$0.10-1 / 1k, 2,500/mo free). Key from
SERPER_API_KEY; absent => unavailable, so the orchestrator falls back to Brave
and then DuckDuckGo. Persistent monthly quota + outage state mirror Brave's, so
the UI can warn when the primary search degrades. Every response is cached by
the base class.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import List

from .. import config
from . import cache
from .base import SearchProvider, SearchResult, make_client
from .stats import STATS
from .ratelimit import IntervalLimiter


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _state_key() -> str:
    return cache.make_key("serperstate", _current_month(), "s")


def _mark_state(state: str) -> None:
    try:
        cache.set(_state_key(), "serperstate", {"state": state}, 40 * 86400)
    except Exception:
        pass


def serper_status() -> dict:
    """Serper availability for the UI: monthly usage + any outage state.
    state one of: ok | exhausted | invalid_key | not_configured."""
    used = cache.get_counter(cache.make_key("serperquota", _current_month(), "count"))
    quota = config.SERPER_MONTHLY_QUOTA
    persisted = (cache.get(_state_key(), track=False) or {}).get("state")
    if not config.SERPER_API_KEY:
        state = "not_configured"
    elif persisted in ("exhausted", "invalid_key"):
        state = persisted
    elif used >= quota:
        state = "exhausted"
    else:
        state = "ok"
    return {"ok": state == "ok", "state": state,
            "used": used, "quota": quota, "remaining": max(0, quota - used)}


class SerperProvider(SearchProvider):
    name = "serper"
    cache_ttl = config.CACHE_TTL_SEARCH

    def __init__(self) -> None:
        interval = 1.0 / config.SERPER_QPS if config.SERPER_QPS > 0 else 0.0
        self._limiter = IntervalLimiter(interval)
        self._lock = threading.Lock()
        self._quota_key = cache.make_key("serperquota", _current_month(), "count")
        self._used = cache.get_counter(self._quota_key)
        self._exhausted = False

    def available(self) -> bool:
        return bool(config.SERPER_API_KEY) and not self._exhausted \
            and self._used < config.SERPER_MONTHLY_QUOTA

    def _search_uncached(self, query: str) -> List[SearchResult]:
        if not self.available():
            return []
        self._limiter.acquire()
        with self._lock:
            self._used = cache.incr_counter(self._quota_key)
        start = time.monotonic()
        try:
            with make_client() as c:
                resp = c.post(
                    config.SERPER_ENDPOINT,
                    headers={"X-API-KEY": config.SERPER_API_KEY,
                             "Content-Type": "application/json"},
                    content=json.dumps({"q": query, "num": config.RESULTS_PER_QUERY}),
                )
        except Exception:
            return []
        STATS.record_call(self.name, time.monotonic() - start)

        if resp.status_code in (401, 403):
            self._exhausted = True  # bad/expired key
            _mark_state("invalid_key")
            return []
        if resp.status_code in (429, 402):
            self._exhausted = True  # rate/quota/credit exhausted
            _mark_state("exhausted")
            return []
        if resp.status_code != 200:
            return []

        out: List[SearchResult] = []
        try:
            for item in (resp.json().get("organic", []) or []):
                title = item.get("title", "") or ""
                url = item.get("link", "") or ""
                snippet = item.get("snippet", "") or ""
                if url and title:
                    out.append(SearchResult(title, url, snippet, self.name))
        except Exception:
            return []
        return out[: config.RESULTS_PER_QUERY]
