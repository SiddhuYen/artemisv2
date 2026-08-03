"""Firm roster scraping (SECONDARY, structured-ish) — VC/company team pages.

Search is only ever allowed to LOCATE a page. The roster on that page is the
structural assertion: a page that lists two people as its team is a much
stronger signal than a search-engine snippet mentioning them together.

The two PAGE-level guards this module used to define — "does it LOOK like a
roster" and "does it BELONG to the firm" — now live in rosters.py, shared
with the org-keyed directory lookup (directory.py). They moved unchanged;
see that module for the bug each one exists to prevent. What stays here is
what is genuinely firm-specific: locating a roster from a PERSON's name, and
Guard 3 (the roster must actually NAME that person — see roster_colleagues).

Requires a search callable to LOCATE candidate roster URLs (person/firm name
-> web search); given a URL directly, `roster()` needs no search at all.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional
from urllib.parse import urlparse

from .. import config
from ..utils.htmltext import jsonld_names, text_blocks
from ..utils.names import org_norm_key, person_norm_key
from . import cache
from .base import Page, SearchResult
from .rosters import (
    clean_roster_names,
    fetch_readable,
    is_roster_url,
    org_name_from_page,
    page_belongs_to_org,
)


_PHRASE = {"coworker": "coworker of", "board_member": "board member with",
           "employee": "coworker of"}


class FirmsProvider:
    """Locate and scrape firm team-roster pages, and resolve a person to the
    colleagues a roster lists alongside them.

    `search` is a callable(query: str) -> List[SearchResult] used ONLY to
    LOCATE candidate pages (e.g. SearchOrchestrator.search); scraping a known
    roster URL via `roster()` needs no search at all.
    """

    name = "firms"

    def __init__(self, search: Optional[Callable[[str], List[SearchResult]]] = None) -> None:
        self._search = search

    def available(self) -> bool:
        return bool(config.FIRMS_ENABLED) and self._search is not None

    # --- firm -> roster ---------------------------------------------------
    def find_team_page(self, firm_name: str) -> Optional[str]:
        """The firm's own roster URL, or None. Verified by Guard 2."""
        if not firm_name or not self.available():
            return None
        key = cache.make_key(self.name, "teampage", org_norm_key(firm_name))
        cached = cache.get(key)
        if cached is not None:
            return cached.get("url") or None

        candidates: List[str] = []
        for query in (f"{firm_name} team page",
                      f"{firm_name} our team partners",
                      f'"{firm_name}" about the team'):
            for result in self._search(query):
                if is_roster_url(result.url) and result.url not in candidates:
                    candidates.append(result.url)
            if candidates:
                break

        # Prefer the shallowest path: "/team" is the roster, "/team/andy" is
        # one person's bio page.
        candidates.sort(key=lambda u: len(urlparse(u).path.strip("/").split("/")))

        # Fetch every candidate concurrently, then pick the FIRST one (in the
        # shallowest-path-preferred sort above) that verifies -- order still
        # decides which page wins, only the I/O is parallelized.
        pages: List[Page] = []
        if candidates:
            with ThreadPoolExecutor(max_workers=min(len(candidates), config.SEARCH_WORKERS)) as ex:
                pages = list(ex.map(fetch_readable, candidates))

        url = ""
        for candidate, page in zip(candidates, pages):
            if page.status_code == 200 and page.content and \
                    page_belongs_to_org(candidate, page.content, firm_name):
                url = candidate
                break
        cache.set(key, "teampage", {"url": url}, config.CACHE_TTL_PAGE)
        return url or None

    def roster(self, url: str, firm_name: str = "") -> dict:
        """Scrape a roster page given its URL directly.

        Returns {firm, url, members[], overflow}. `overflow` is True when the
        page lists more people than the edge cap permits, so the caller can
        record membership without materializing a false clique.
        """
        out = {"firm": firm_name, "url": url, "members": [], "overflow": False}
        if not is_roster_url(url):
            return out  # Guard 1: a non-roster page asserts no roster

        key = cache.make_key(self.name, "roster", f"{org_norm_key(firm_name)}::{url}")
        cached = cache.get(key)
        if cached is not None:
            return cached

        page = fetch_readable(url)
        if page.status_code != 200 or not page.content:
            return out
        if firm_name and not page_belongs_to_org(url, page.content, firm_name):
            return out  # Guard 2: this page is not this firm's

        # Prefer schema.org Person data when present — machine-readable and
        # survives JS rendering intact even when visible text does not.
        jsonld = clean_roster_names(jsonld_names(page.content, "Person"))

        # Else per-element blocks, never one flattened string: flattening
        # glues neighbouring "Email" / "LinkedIn" labels onto a name.
        blocks = text_blocks(page.content)
        if not jsonld and not blocks:
            return out  # a JS-rendered shell asserts nothing we can read

        scraped = clean_roster_names(blocks)
        names, seen = [], set()
        for n in jsonld + scraped:  # JSON-LD names are authoritative; union
            k = person_norm_key(n)
            if k and k not in seen:
                seen.add(k)
                names.append(n)

        out["firm"] = firm_name or org_name_from_page(page.content, url)
        out["members"] = names[: config.MAX_ROSTER_MEMBERS]
        out["overflow"] = len(names) > config.MAX_ROSTER_MEMBERS
        cache.set(key, "roster", out, config.CACHE_TTL_PAGE)
        return out

    def roster_for_firm(self, firm_name: str) -> dict:
        url = self.find_team_page(firm_name)
        if not url:
            return {"firm": firm_name, "url": "", "members": [], "overflow": False}
        result = self.roster(url, firm_name)
        result["firm"] = firm_name or result.get("firm", "")
        return result

    # --- person -> firm colleagues -----------------------------------------
    def roster_colleagues(self, person_name: str) -> List[dict]:
        """Roster-mate colleagues of `person_name`, across at most
        MAX_FIRMS_PER_PERSON firms.

        Guard 3: the person's name must appear on the roster we scraped — a
        page a search merely returned for their name asserts nothing about
        them. Returns [{name, relationship_type, company, phrase, url}].
        """
        if not person_name or not self.available():
            return []
        target = person_norm_key(person_name)
        if not target:
            return []
        key = cache.make_key(self.name, "colleagues", target)
        cached = cache.get(key)
        if cached is not None:
            return cached.get("colleagues", [])

        candidates: List[str] = []
        for query in (f'"{person_name}" venture capital team',
                      f'"{person_name}" partner venture firm team page',
                      f'"{person_name}" team'):
            for result in self._search(query):
                if is_roster_url(result.url) and result.url not in candidates:
                    candidates.append(result.url)
        candidates.sort(key=lambda u: len(urlparse(u).path.strip("/").split("/")))

        # Keep the FULLEST verified roster per firm: a bio page and the
        # roster both name the person and share a domain, but the bio page
        # lists a few colleagues where the roster lists the whole team.
        urls = candidates[: 3 * config.MAX_FIRMS_PER_PERSON]
        best: dict = {}
        if urls:
            # Independent per-URL scrapes -- no ordering dependency (each
            # candidate only competes on member count within its own firm
            # key), so fetch them all concurrently.
            with ThreadPoolExecutor(max_workers=min(len(urls), config.SEARCH_WORKERS)) as ex:
                rosters = list(ex.map(self.roster, urls))  # no firm name yet; derived from the page
            for roster in rosters:
                members = roster.get("members") or []
                if target not in {person_norm_key(m) for m in members}:
                    continue  # Guard 3: this roster does not name them
                if not roster.get("firm"):
                    continue
                firm_key = org_norm_key(roster["firm"])
                if len(members) > len(best.get(firm_key, {}).get("members") or []):
                    best[firm_key] = roster

        found = sorted(best.values(), key=lambda r: -len(r["members"]))[
            : config.MAX_FIRMS_PER_PERSON]

        results: List[dict] = []
        seen = set()
        for roster in found:
            for member in roster["members"]:
                if person_norm_key(member) == target:
                    continue
                k = (person_norm_key(member), org_norm_key(roster["firm"]))
                if k in seen:
                    continue
                seen.add(k)
                results.append({
                    "name": member, "relationship_type": "coworker",
                    "company": roster["firm"], "phrase": _PHRASE["coworker"],
                    "url": roster["url"],
                })
        cache.set(key, "colleagues", {"colleagues": results}, config.CACHE_TTL_WIKI)
        return results

    def colleagues_text(self, subject: str, colleagues: List[dict]) -> str:
        return " ".join(f"{subject} {c['phrase']} {c['name']} at {c['company']}."
                        for c in colleagues)
