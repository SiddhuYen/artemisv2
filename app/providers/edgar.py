"""SEC EDGAR provider (SECONDARY, structured) — public-company insider networks.

Free, no key (declared User-Agent required). Person-centric via EDGAR full-text
search of Form 4 (insider) filings:
  person  -> Form 4 filings -> issuer companies
  company -> Form 4 filers  -> co-insiders (fellow directors / officers / owners)

Returns co-insiders as 'coworker' edges (conservative; they're company
leadership/ownership). Cached.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from .. import config
from . import cache
from .base import request_with_retry
from .ratelimit import IntervalLimiter
from ..utils.names import looks_like_org_name

_FTS = "https://efts.sec.gov/LATEST/search-index"
_LIMITER = IntervalLimiter(config.EDGAR_MIN_INTERVAL)
_TICKER = re.compile(r"\([A-Z]{1,6}\)")

_MAX_COMPANIES = 4
_MAX_INSIDERS_PER_COMPANY = 20
_MAX_TOTAL = 30


def _clean(dn: str) -> str:
    # "MICROSOFT CORP  (MSFT)  (CIK 0000789019)" -> "MICROSOFT CORP"
    return dn.split("  (")[0].strip()


def _is_company(dn: str) -> bool:
    return bool(_TICKER.search(dn)) or looks_like_org_name(_clean(dn))


def _person_display(dn: str) -> str:
    # EDGAR uses "LAST FIRST MIDDLE" -> "First Middle Last", title-cased
    name = _clean(dn)
    parts = name.split()
    if len(parts) >= 2:
        name = " ".join(parts[1:] + parts[:1])
    return name.title()


class EdgarProvider:
    name = "edgar"

    def available(self) -> bool:
        return bool(config.EDGAR_ENABLED)

    def officer_colleagues(self, name: str) -> List[dict]:
        if not name or not self.available():
            return []
        key = cache.make_key(self.name, "colleagues", name.lower())
        cached = cache.get(key)
        if cached is not None:
            return cached.get("colleagues", [])

        companies = self._companies_for_person(name)
        results: List[dict] = []
        seen = set()
        for company in companies[:_MAX_COMPANIES]:
            for insider in self._insiders_for_company(company):
                k = insider.lower()
                if k == name.lower() or k in seen:
                    continue
                seen.add(k)
                results.append({"name": insider, "relationship_type": "coworker",
                                "company": company, "phrase": "coworker of"})
                if len(results) >= _MAX_TOTAL:
                    break
            if len(results) >= _MAX_TOTAL:
                break
        cache.set(key, "colleagues", {"colleagues": results}, config.CACHE_TTL_WIKI)
        return results

    def colleagues_text(self, subject: str, colleagues: List[dict]) -> str:
        return " ".join(f"{subject} coworker of {c['name']}." for c in colleagues)

    # --- internals --------------------------------------------------------
    def _fts_hits(self, **params):
        _LIMITER.acquire()
        resp = request_with_retry("GET", _FTS, provider=self.name, params=params,
                                  headers={"User-Agent": config.EDGAR_USER_AGENT})
        if resp is None or resp.status_code != 200:
            return []
        try:
            return resp.json().get("hits", {}).get("hits", [])
        except Exception:
            return []

    def _companies_for_person(self, name: str) -> List[str]:
        comps: List[str] = []
        seen = set()
        for h in self._fts_hits(q=f'"{name}"', forms="4"):
            for dn in h.get("_source", {}).get("display_names", []):
                if _is_company(dn):
                    c = _clean(dn)
                    if c and c.lower() not in seen:
                        seen.add(c.lower())
                        comps.append(c)
        return comps

    def _insiders_for_company(self, company: str) -> List[str]:
        people: List[str] = []
        seen = set()
        for h in self._fts_hits(q=f'"{company}"', forms="4"):
            for dn in h.get("_source", {}).get("display_names", []):
                if _is_company(dn):
                    continue
                p = _person_display(dn)
                if p and p.lower() not in seen:
                    seen.add(p.lower())
                    people.append(p)
                if len(people) >= _MAX_INSIDERS_PER_COMPANY:
                    break
        return people
