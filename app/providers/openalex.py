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

        author_id, _identity = self._resolve_author(name)
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

    def identity_text(self, name: str) -> str:
        """A short description of whoever `name` actually resolved to at
        OpenAlex -- their last known institutional affiliation, if any.

        This exists so a caller can verify the match BEFORE trusting the
        coauthor list this same resolution produces (see graph.expansion +
        graph.disambiguate.domain_conflict) -- the works_count/name-similarity
        guard in _resolve_author narrows candidates, it doesn't confirm
        identity, and a bare name search has no way to tell "Prantik
        Chakraborty, VP Sales at a chemicals company" from "Prantik
        Chakraborty, ISRO radar researcher" without something like this.
        """
        if not name:
            return ""
        _author_id, identity = self._resolve_author(name)
        return identity

    def _resolve_author(self, name: str):
        """Returns (author_id, identity_text) -- (None, "") on no match.

        Both come from, and are cached with, the same lookup: identity_text
        is whatever this resolution already told us about who it resolved
        to, at no extra request cost.
        """
        key = cache.make_key(self.name, "resolve", name.lower())
        cached = cache.get(key)
        if cached is not None:
            return cached.get("id"), cached.get("identity_text", "")

        _LIMITER.acquire()
        resp = request_with_retry(
            "GET", _AUTHORS, provider=self.name,
            params={"search": name, "per_page": 1,
                    "select": "id,display_name,works_count,last_known_institutions"},
        )
        author_id, identity = None, ""
        if resp is not None and resp.status_code == 200:
            try:
                results = resp.json().get("results", []) or []
                if results:
                    top = results[0]
                    # guard against namesakes: require a real publication record
                    # and a close name match (reduces matching a researcher to a
                    # non-academic) -- narrows candidates, doesn't confirm identity.
                    if ((top.get("works_count") or 0) >= config.OPENALEX_MIN_WORKS
                            and _name_ratio(name, top.get("display_name", ""))
                                >= config.OPENALEX_NAME_SIM):
                        author_id = top["id"].rsplit("/", 1)[-1]
                        insts = [i.get("display_name") for i in
                                (top.get("last_known_institutions") or [])
                                if i.get("display_name")]
                        # "an academic author" is explicit and load-bearing, not
                        # filler: passing the works_count/name-similarity guard
                        # above IS a real published record, so this is a true
                        # statement regardless of institution data -- and it's
                        # the word disambiguate.domain_conflict's lexicon
                        # actually matches on. An institution's bare NAME (e.g.
                        # "Indian Space Research Organisation") does NOT contain
                        # a profession keyword on its own; without this, the
                        # domain check silently never fires, which is exactly
                        # what happened testing this against the live API for
                        # the motivating case before this line was added.
                        identity = f"{top.get('display_name', name)}, an academic author"
                        if insts:
                            identity += f" affiliated with {', '.join(insts)}"
                        identity += "."
            except Exception:
                pass
        cache.set(key, "resolve", {"id": author_id, "identity_text": identity},
                 config.CACHE_TTL_WIKI)
        return author_id, identity
