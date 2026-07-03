"""ProPublica Nonprofit Explorer provider (SECONDARY, structured) — 990 boards.

Nonprofit board members / officers / key employees, including non-famous people.
The API only exposes aggregate financials (no names), so we resolve the EIN via
the API (clean) and scrape the names+titles from the org's HTML page (the 990
Part VII table). Org-centric: fed by the subject's nonprofit affiliations.
Cached.
"""
from __future__ import annotations

import re
from typing import List

from bs4 import BeautifulSoup

from .. import config
from . import cache
from .base import request_with_retry
from .ratelimit import IntervalLimiter
from ..utils.names import normalize

_SEARCH = "https://projects.propublica.org/nonprofits/api/v2/search.json"
_ORG_HTML = "https://projects.propublica.org/nonprofits/organizations/{ein}"
_LIMITER = IntervalLimiter(config.PROPUBLICA_MIN_INTERVAL)
_MAX_PEOPLE = 20


def _relationship_for(title: str) -> str:
    t = (title or "").lower()
    if any(k in t for k in ("director", "trustee", "board", "chair")):
        return "board_member"
    if any(k in t for k in ("ceo", "president", "officer", "cfo", "treasurer",
                            "secretary", "executive", "director")):
        return "employee"
    return "coworker"


def _phrase(rel: str) -> str:
    return "board member with" if rel == "board_member" else "coworker of"


class ProPublicaProvider:
    name = "propublica"

    def available(self) -> bool:
        return bool(config.PROPUBLICA_ENABLED)

    def board_members(self, org_name: str) -> List[dict]:
        """Resolve a nonprofit by name, return its board/officers as contacts."""
        if not org_name or not self.available():
            return []
        key = cache.make_key(self.name, "board", normalize(org_name))
        cached = cache.get(key)
        if cached is not None:
            return cached.get("people", [])

        ein, matched = self._resolve_ein(org_name)
        people: List[dict] = []
        if ein:
            for nm, title in self._scrape_officers(ein):
                rel = _relationship_for(title)
                people.append({"name": nm, "relationship_type": rel,
                               "org": matched or org_name, "phrase": _phrase(rel)})
                if len(people) >= _MAX_PEOPLE:
                    break
        cache.set(key, "board", {"people": people}, config.CACHE_TTL_WIKI)
        return people

    def colleagues_text(self, subject: str, people: List[dict]) -> str:
        return " ".join(f"{subject} {p['phrase']} {p['name']}." for p in people)

    # --- internals --------------------------------------------------------
    def _resolve_ein(self, org_name: str):
        _LIMITER.acquire()
        resp = request_with_retry("GET", _SEARCH, provider=self.name,
                                  params={"q": org_name})
        if resp is None or resp.status_code != 200:
            return None, None
        try:
            orgs = resp.json().get("organizations", []) or []
            if not orgs:
                return None, None
            top = orgs[0]
            # name-match guard: avoid pulling a wrong nonprofit's board
            if not _token_overlap(org_name, top.get("name", "")):
                return None, None
            return top.get("ein"), top.get("name")
        except Exception:
            return None, None

    def _scrape_officers(self, ein):
        _LIMITER.acquire()
        resp = request_with_retry(
            "GET", _ORG_HTML.format(ein=ein), provider=self.name,
            headers={"User-Agent": config.USER_AGENT})
        if resp is None or resp.status_code != 200:
            return []
        out = []
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            for table in soup.find_all("table"):
                head = table.get_text(" ", strip=True)[:80]
                if "Officer" not in head and "Key Employees" not in head \
                        and "Trustee" not in head and "Director" not in head:
                    continue
                for tr in table.find_all("tr"):
                    cells = tr.find_all("td")
                    if not cells:
                        continue
                    full = cells[0].get_text(" ", strip=True)
                    # cell holds "Name (Title)"; title may also be in cell 2
                    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", full)
                    if m:
                        name, title = m.group(1).strip(), m.group(2).strip()
                    else:
                        name = full
                        title = cells[1].get_text(" ", strip=True).strip("()") \
                            if len(cells) > 1 else ""
                    if name and len(name.split()) >= 2 and not name[0].isdigit():
                        out.append((name, title))
                if out:
                    break
        except Exception:
            return []
        return out


def _token_overlap(a: str, b: str) -> bool:
    # bidirectional: the matched org must not have lots of extra tokens, so
    # "Ford Foundation" doesn't match "Foundation For Rocky Ford Schools".
    ta = {t for t in normalize(a).split() if len(t) > 2}
    tb = {t for t in normalize(b).split() if len(t) > 2}
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    return inter / len(ta) >= 0.6 and inter / len(tb) >= 0.6
