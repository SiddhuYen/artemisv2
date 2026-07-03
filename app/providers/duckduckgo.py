"""DuckDuckGo HTML provider (FALLBACK only).

Used only when Brave is unavailable/exhausted/empty. Protected by a token
bucket + jittered spacing and a circuit breaker that trips on repeated 429s
(stops all DDG traffic during a cooldown, then auto-retries). Cache-first.
"""
from __future__ import annotations

import time
from typing import List
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup

from .. import config
from .base import SearchProvider, SearchResult, make_client
from .ratelimit import CircuitBreaker, IntervalLimiter, TokenBucket
from .stats import STATS

_ENDPOINTS = [
    "https://html.duckduckgo.com/html/",
    "https://lite.duckduckgo.com/lite/",
]


def _decode_ddg_url(href: str) -> str:
    if not href:
        return href
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return href


class DuckDuckGoProvider(SearchProvider):
    name = "duckduckgo"

    def __init__(self) -> None:
        self._bucket = TokenBucket(rate_per_sec=1.0 / max(config.DDG_MIN_INTERVAL, 0.01),
                                   capacity=config.DDG_BUCKET_CAPACITY)
        self._jitter = IntervalLimiter(0.0, jitter=config.DDG_JITTER)
        self._breaker = CircuitBreaker(
            threshold=config.DDG_BREAKER_THRESHOLD,
            cooldown=config.DDG_BREAKER_COOLDOWN,
            on_trip=STATS.breaker_tripped,
        )

    def available(self) -> bool:
        return self._breaker.allow()

    def _search_uncached(self, query: str) -> List[SearchResult]:
        if not self._breaker.allow():
            return []  # circuit open: skip DDG entirely
        results: List[SearchResult] = []
        for endpoint in _ENDPOINTS:
            self._bucket.acquire()
            self._jitter.acquire()
            start = time.monotonic()
            try:
                with make_client() as c:
                    resp = c.post(endpoint, data={"q": query})
            except Exception:
                self._breaker.record_failure()
                continue
            STATS.record_call(self.name, time.monotonic() - start)
            if resp.status_code == 429:
                self._breaker.record_failure()
                continue
            if resp.status_code != 200:
                continue
            self._breaker.record_success()
            results = self._parse(resp.text)
            if results:
                break
        return results[: config.RESULTS_PER_QUERY]

    def _parse(self, html: str) -> List[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        out: List[SearchResult] = []
        for res in soup.select("div.result, div.web-result"):
            a = res.select_one("a.result__a")
            if not a:
                continue
            url = _decode_ddg_url(a.get("href", ""))
            title = a.get_text(" ", strip=True)
            snip_el = res.select_one(".result__snippet")
            snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
            if url and title:
                out.append(SearchResult(title, url, snippet, self.name))
        if not out:
            for a in soup.select("a.result-link"):
                url = _decode_ddg_url(a.get("href", ""))
                title = a.get_text(" ", strip=True)
                if url and title:
                    out.append(SearchResult(title, url, "", self.name))
        return out
