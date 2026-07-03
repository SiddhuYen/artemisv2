"""Wikipedia provider (SECONDARY, structured knowledge).

- search(query): MediaWiki search -> results (cache-first via base).
- summary(title): plain-text lead extract.
- links(title): important outbound page links.
- wikidata_id(title): the page's Wikidata QID (feeds WikidataProvider).
Light rate limiter (Wikipedia is generous). All responses cached.
"""
from __future__ import annotations

from typing import List, Optional

from bs4 import BeautifulSoup

from .. import config
from . import cache
from .base import SearchProvider, SearchResult, request_with_retry
from .ratelimit import IntervalLimiter

_API = "https://en.wikipedia.org/w/api.php"
_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

_LIMITER = IntervalLimiter(config.WIKI_MIN_INTERVAL)


class WikipediaProvider(SearchProvider):
    name = "wikipedia"
    cache_ttl = config.CACHE_TTL_WIKI

    def _search_uncached(self, query: str) -> List[SearchResult]:
        _LIMITER.acquire()
        params = {"action": "query", "list": "search", "srsearch": query,
                  "format": "json", "srlimit": config.RESULTS_PER_QUERY}
        resp = request_with_retry("GET", _API, provider=self.name, params=params)
        out: List[SearchResult] = []
        if resp is not None and resp.status_code == 200:
            try:
                for hit in resp.json().get("query", {}).get("search", []):
                    title = hit.get("title", "")
                    snippet = _strip_html(hit.get("snippet", ""))
                    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
                    out.append(SearchResult(title, url, snippet, self.name))
            except Exception:
                pass
        return out

    def summary(self, title: str) -> str:
        key = cache.make_key(self.name, "summary", title)
        cached = cache.get(key)
        if cached is not None:
            return cached.get("text", "")
        _LIMITER.acquire()
        resp = request_with_retry("GET", _SUMMARY + title.replace(" ", "_"),
                                  provider=self.name)
        text = ""
        if resp is not None and resp.status_code == 200:
            try:
                text = resp.json().get("extract", "") or ""
            except Exception:
                text = ""
        cache.set(key, "summary", {"text": text}, self.cache_ttl)
        return text

    def article_text(self, title: str) -> str:
        """Full plain-text article (far more named people than the lead summary)."""
        key = cache.make_key(self.name, "article", title)
        cached = cache.get(key)
        if cached is not None:
            return cached.get("text", "")
        _LIMITER.acquire()
        params = {"action": "query", "prop": "extracts", "explaintext": 1,
                  "titles": title, "format": "json", "redirects": 1}
        resp = request_with_retry("GET", _API, provider=self.name, params=params)
        text = ""
        if resp is not None and resp.status_code == 200:
            try:
                pages = resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    text = page.get("extract", "") or ""
                    if text:
                        break
            except Exception:
                text = ""
        text = text[: config.MAX_PAGE_CHARS]
        cache.set(key, "article", {"text": text}, self.cache_ttl)
        return text

    def links(self, title: str, limit: int = 60) -> List[str]:
        key = cache.make_key(self.name, "links", title)
        cached = cache.get(key)
        if cached is not None:
            return cached.get("links", [])
        _LIMITER.acquire()
        params = {"action": "query", "prop": "links", "titles": title,
                  "pllimit": limit, "plnamespace": 0, "format": "json"}
        resp = request_with_retry("GET", _API, provider=self.name, params=params)
        links: List[str] = []
        if resp is not None and resp.status_code == 200:
            try:
                pages = resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    for lk in page.get("links", []) or []:
                        if lk.get("title"):
                            links.append(lk["title"])
            except Exception:
                pass
        cache.set(key, "links", {"links": links[:limit]}, self.cache_ttl)
        return links[:limit]

    def wikidata_id(self, title: str) -> Optional[str]:
        key = cache.make_key(self.name, "wdid", title)
        cached = cache.get(key)
        if cached is not None:
            return cached.get("qid")
        _LIMITER.acquire()
        params = {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
                  "titles": title, "format": "json", "redirects": 1}
        resp = request_with_retry("GET", _API, provider=self.name, params=params)
        qid = None
        if resp is not None and resp.status_code == 200:
            try:
                pages = resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    qid = page.get("pageprops", {}).get("wikibase_item")
                    if qid:
                        break
            except Exception:
                qid = None
        cache.set(key, "wdid", {"qid": qid}, self.cache_ttl)
        return qid

    def best_title(self, name: str) -> Optional[str]:
        """Top search hit title for a name (the person's likely page)."""
        results = self.search(name)
        return results[0].title if results else None


def _strip_html(s: str) -> str:
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
