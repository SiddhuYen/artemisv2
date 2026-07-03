"""OpenCorporates provider (SECONDARY, structured) — company officer networks.

Person-centric business relationships, including NON-famous people who will
never appear on Wikipedia/Wikidata: resolve a name to officer positions, then
pull co-officers of those companies (fellow directors/board/execs).

Requires OPENCORPORATES_API_TOKEN (free tier; anonymous access is throttled).
Gracefully no-ops when the token is absent or calls fail. Cached.
"""
from __future__ import annotations

from typing import List

from .. import config
from . import cache
from .base import request_with_retry
from .ratelimit import IntervalLimiter

_BASE = "https://api.opencorporates.com/v0.4"
_LIMITER = IntervalLimiter(config.OPENCORP_MIN_INTERVAL)

_MAX_COMPANIES = 5
_MAX_OFFICERS_PER_COMPANY = 15
_MAX_TOTAL = 30


def _relationship_for(position: str) -> str:
    p = (position or "").lower()
    if any(k in p for k in ("director", "board", "trustee", "chair")):
        return "board_member"
    if any(k in p for k in ("ceo", "president", "officer", "secretary",
                            "treasurer", "manager", "executive", "partner")):
        return "employee"
    return "coworker"


class OpenCorporatesProvider:
    name = "opencorporates"

    def available(self) -> bool:
        return bool(config.OPENCORPORATES_API_TOKEN)

    def officer_colleagues(self, name: str) -> List[dict]:
        """Co-officers of companies where `name` is an officer.
        Returns [{name, relationship_type, company, phrase}]."""
        if not name or not self.available():
            return []
        key = cache.make_key(self.name, "colleagues", name.lower())
        cached = cache.get(key)
        if cached is not None:
            return cached.get("colleagues", [])

        companies = self._companies_for_officer(name)
        results: List[dict] = []
        seen = set()
        for jur, num, company_name in companies[:_MAX_COMPANIES]:
            for off_name, position in self._company_officers(jur, num):
                if off_name.lower() == name.lower() or off_name.lower() in seen:
                    continue
                seen.add(off_name.lower())
                rel = _relationship_for(position)
                results.append({"name": off_name, "relationship_type": rel,
                                "company": company_name,
                                "phrase": _PHRASE.get(rel, "coworker of")})
                if len(results) >= _MAX_TOTAL:
                    break
            if len(results) >= _MAX_TOTAL:
                break
        cache.set(key, "colleagues", {"colleagues": results}, config.CACHE_TTL_WIKI)
        return results

    def colleagues_text(self, subject: str, colleagues: List[dict]) -> str:
        return " ".join(f"{subject} {c['phrase']} {c['name']}." for c in colleagues)

    # --- internals --------------------------------------------------------
    def _companies_for_officer(self, name: str):
        _LIMITER.acquire()
        resp = request_with_retry(
            "GET", f"{_BASE}/officers/search", provider=self.name,
            params={"q": name, "per_page": 20,
                    "api_token": config.OPENCORPORATES_API_TOKEN},
        )
        out = []
        if resp is None or resp.status_code != 200:
            return out
        try:
            for item in resp.json().get("results", {}).get("officers", []) or []:
                off = item.get("officer", {})
                comp = off.get("company", {}) or {}
                jur = comp.get("jurisdiction_code")
                num = comp.get("company_number")
                cname = comp.get("name", "")
                if jur and num:
                    out.append((jur, num, cname))
        except Exception:
            return []
        return out

    def _company_officers(self, jurisdiction: str, number: str):
        _LIMITER.acquire()
        resp = request_with_retry(
            "GET", f"{_BASE}/companies/{jurisdiction}/{number}", provider=self.name,
            params={"api_token": config.OPENCORPORATES_API_TOKEN},
        )
        out = []
        if resp is None or resp.status_code != 200:
            return out
        try:
            officers = resp.json().get("results", {}).get("company", {}).get("officers", [])
            for item in officers[:_MAX_OFFICERS_PER_COMPANY]:
                off = item.get("officer", {})
                nm = off.get("name", "")
                if nm:
                    out.append((nm, off.get("position", "")))
        except Exception:
            return []
        return out


_PHRASE = {"board_member": "board member with", "employee": "coworker of",
           "coworker": "coworker of"}
