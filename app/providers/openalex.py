"""OpenAlex provider (SECONDARY, structured) — academic coauthors.

Free, no API key, person-centric: resolve a name to an OpenAlex author, then
collect their coauthors across works. High-recall, clean person→person
professional links for anyone who has published (researchers, many execs,
authors, clinicians). Cached.
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import List

from .. import config
from . import cache
from .base import request_with_retry
from .ratelimit import IntervalLimiter
from ..utils.names import person_norm_key


def _name_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, person_norm_key(a), person_norm_key(b)).ratio()

_AUTHORS = "https://api.openalex.org/authors"
_WORKS = "https://api.openalex.org/works"
_LIMITER = IntervalLimiter(config.WIKI_MIN_INTERVAL)

_MAX_WORKS = 50
_MAX_COAUTHORS = 30


class OpenAlexProvider:
    name = "openalex"

    def coauthors(self, name: str) -> List[dict]:
        """Return [{name, count}] of the person's most frequent coauthors."""
        if not name:
            return []
        key = cache.make_key(self.name, "coauthors", name.lower())
        cached = cache.get(key)
        if cached is not None:
            return cached.get("coauthors", [])

        author_id = self._resolve_author(name)
        result: List[dict] = []
        if author_id:
            counts: Counter = Counter()
            _LIMITER.acquire()
            resp = request_with_retry(
                "GET", _WORKS, provider=self.name,
                params={"filter": f"author.id:{author_id}", "per_page": _MAX_WORKS,
                        "select": "authorships"},
            )
            if resp is not None and resp.status_code == 200:
                try:
                    for work in resp.json().get("results", []) or []:
                        for a in work.get("authorships", []) or []:
                            dn = (a.get("author") or {}).get("display_name")
                            if dn and dn.lower() != name.lower():
                                counts[dn] += 1
                except Exception:
                    counts = Counter()
            result = [{"name": n, "count": c} for n, c in counts.most_common(_MAX_COAUTHORS)]
        cache.set(key, "coauthors", {"coauthors": result}, config.CACHE_TTL_WIKI)
        return result

    def coauthors_text(self, subject: str, coauthors: List[dict]) -> str:
        return " ".join(f"{subject} coauthor of {c['name']}." for c in coauthors)

    def _resolve_author(self, name: str):
        _LIMITER.acquire()
        resp = request_with_retry(
            "GET", _AUTHORS, provider=self.name,
            params={"search": name, "per_page": 1,
                    "select": "id,display_name,works_count"},
        )
        if resp is None or resp.status_code != 200:
            return None
        try:
            results = resp.json().get("results", []) or []
            if not results:
                return None
            top = results[0]
            # guard against namesakes: require a real publication record and a
            # close name match (reduces matching a researcher to a non-academic).
            if (top.get("works_count") or 0) < config.OPENALEX_MIN_WORKS:
                return None
            if _name_ratio(name, top.get("display_name", "")) < config.OPENALEX_NAME_SIM:
                return None
            return top["id"].rsplit("/", 1)[-1]
        except Exception:
            return None
