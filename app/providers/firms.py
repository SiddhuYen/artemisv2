"""Firm roster scraping (SECONDARY, structured-ish) — VC/company team pages.

Search is only ever allowed to LOCATE a page. The roster on that page is the
structural assertion: a page that lists two people as its team is a much
stronger signal than a search-engine snippet mentioning them together.

Two guards, each of which cost a real bug on a sibling project:

  1. A page must LOOK like a roster (`/team`, `/people`, ...), not a firm's
     homepage — a homepage interleaves partners with quoted portfolio
     founders, and neither NER nor name-shape filtering can tell them apart.
  2. A page must BELONG to the firm, established by IDENTITY (the domain, or
     the name the page declares for itself) rather than keyword presence.
     "Storm Ventures" is a substring of a rival firm's domain "calmstorm.vc",
     and a search for "Homebrew team page" returns the package manager's
     cask index — keyword presence alone would attach the wrong roster.

Requires a search callable to LOCATE candidate roster URLs (person/firm name
-> web search); given a URL directly, `roster()` needs no search at all.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional
from urllib.parse import urlparse

from .. import config
from ..utils.htmltext import jsonld_names, soup_of, text_blocks
from ..utils.names import (
    is_noise_name,
    looks_like_person_name,
    normalize,
    org_norm_key,
    person_norm_key,
)
from . import cache
from .base import Page, SearchResult, fetch_page

# Path segments that mark a page as a roster of people.
#
# Deliberately EXCLUDES "/about" and "/founders": an about page interleaves
# the team with portfolio companies (proper nouns that look like people to a
# shape-only filter), so treating it as a roster invites false members.
_ROSTER_HINTS = ("team", "people", "our-team", "ourteam", "partners", "staff",
                 "leadership", "who-we-are", "whoweare", "members",
                 "our-firm", "ourfirm", "crew", "humans")

# Never a roster, even when a hint appears elsewhere in the path.
_NEGATIVE_HINTS = ("portfolio", "blog", "post", "news", "careers", "jobs",
                   "contact", "privacy", "terms", "press", "insights")

# Aggregators and socials: real pages, but never the firm's own roster.
_BLOCKED_HOSTS = ("linkedin.com", "twitter.com", "x.com", "facebook.com",
                  "crunchbase.com", "pitchbook.com", "dealroom.co", "f6s.com",
                  "reddit.com", "wikipedia.org", "medium.com", "youtube.com")

# Tokens too generic to identify a firm on a page.
_GENERIC_FIRM_TOKENS = {"ventures", "capital", "partners", "fund", "funds",
                        "group", "management", "the", "and", "vc", "llc", "lp"}

# Leading role words scraped rosters glue onto a name in one text node
# ("Partner Alex Harris" -> "Alex Harris"). Stripped before judging name
# shape, not treated as disqualifying — outright rejection silently drops
# real team members.
_ROLE_PREFIXES = (
    "general partner", "managing partner", "venture partner", "founding partner",
    "operating partner", "managing director", "senior partner",
    "partner", "principal", "associate", "analyst",
    "chief executive officer", "chief operating officer", "chief financial officer",
    "co-founder", "cofounder", "founder", "president", "chairman", "chairwoman",
    "ceo", "coo", "cfo", "cto", "gp", "vp", "svp", "evp",
)

_TLD_TOKENS = {"co", "com", "vc", "io", "ai", "net", "org", "fund", "capital"}


def _host(url: str) -> str:
    # NB: not .lstrip("www."), which strips any leading 'w'/'.' characters and
    # would turn "wework.com" into "ework.com".
    host = (urlparse(url).netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


def _domain_stem(url: str) -> str:
    """"www.hustlefund.vc" -> "hustlefund"; "btv.vc" -> "btv"."""
    host = _host(url)
    parts = [p for p in host.split(".") if p]
    return re.sub(r"[^a-z0-9]", "", parts[0]) if parts else ""


def is_roster_url(url: str) -> bool:
    """True when the URL path looks like a team/people page (not a homepage)."""
    if not url:
        return False
    host = _host(url)
    if any(bad in host for bad in _BLOCKED_HOSTS):
        return False
    path = (urlparse(url).path or "/").strip("/").lower()
    if not path:
        return False  # bare homepage
    if any(neg in path for neg in _NEGATIVE_HINTS):
        return False
    return any(hint in path.split("/") or hint in path for hint in _ROSTER_HINTS)


def firm_tokens(firm_name: str) -> set:
    """Distinctive words of a firm name ("Uncork Capital" -> {"uncork"})."""
    return {t for t in normalize(firm_name).split()
            if t and t not in _GENERIC_FIRM_TOKENS and len(t) > 2}


def _page_title(html: str) -> str:
    title = soup_of(html).title
    return title.get_text(" ", strip=True) if title else ""


def firm_name_from_page(html: str, url: str) -> str:
    """Display name of the firm behind a roster page, VERIFIED by the domain.

    A <title> is often "BTV | Sheel Mohnot" or "Our team | Hustle Fund".
    Taking the longest segment risks a person's name or a tagline, so only
    the title segment whose letters match the registrable domain is
    accepted. Falls back to the domain stem, which is always at least honest.
    """
    stem = _domain_stem(url)
    title = _page_title(html)
    for segment in re.split(r"[|\-–—:·]", title):
        segment = segment.strip()
        if not segment or looks_like_person_name(segment):
            continue
        tokens = [t for t in normalize(segment).split() if t]
        while tokens and tokens[-1] in _TLD_TOKENS and len(tokens) > 1:
            tokens.pop()
        key = "".join(tokens)
        if key and stem and (key == stem or key in stem or stem in key):
            display = re.sub(r"\.(co|com|vc|io|ai|net|org)$", "", segment,
                             flags=re.I).strip(" .")
            return display or segment.strip()
    return stem.title() if stem else ""


def page_belongs_to_firm(url: str, html: str, firm_name: str) -> bool:
    """Guard 2. The page must BE this firm's, established by identity — the
    domain, or the name the page declares for itself — never keyword presence.

    So: the domain must begin with a distinctive token of the firm's name, or
    the page's own declared name must equal that firm (allowing an initialism,
    since "btv.vc" declares itself "BTV" and means Better Tomorrow Ventures).
    """
    tokens = firm_tokens(firm_name)
    if not tokens:
        return True  # nothing distinctive to check against; fall back to Guard 1

    stem = _domain_stem(url)
    domain_hit = bool(stem) and any(stem.startswith(tok) or tok.startswith(stem)
                                    for tok in tokens)
    # A bare domain match settles a single-token firm ("Homebrew" -> homebrew.co).
    # It does NOT settle a multi-word one — a business unit of a much bigger
    # company can share the parent's domain stem while being a different desk.
    if domain_hit and len(tokens) == 1:
        return True

    declared = firm_name_from_page(html, url)
    if not declared:
        return False
    if org_norm_key(declared) == org_norm_key(firm_name):
        return True
    initials = "".join(word[0] for word in normalize(firm_name).split() if word)
    return normalize(declared).replace(" ", "") == initials


def _strip_role_prefix(text: str) -> str:
    t = text.strip()
    low = t.lower()
    for role in sorted(_ROLE_PREFIXES, key=len, reverse=True):
        if low.startswith(role + " "):
            return t[len(role):].strip(" ,.-")
    return t


def _clean_roster_names(candidates: List[str]) -> List[str]:
    """Deterministic name-shape filter over roster text blocks; dedups on the
    person key. Never an LLM — see utils/names.looks_like_person_name."""
    seen, out = set(), []
    for raw in candidates:
        name = _strip_role_prefix((raw or "").strip())
        if is_noise_name(name) or not looks_like_person_name(name):
            continue
        key = person_norm_key(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _fetch_readable(url: str) -> Page:
    """Fetch `url`, falling back to a headless render if the plain GET returns
    a JavaScript shell (no readable text blocks). Rendering is optional: when
    the browser is unavailable this is just the plain fetch."""
    page = fetch_page(url)
    if page.status_code == 200 and text_blocks(page.content):
        return page
    from .browser import available as _browser_available
    if _browser_available():
        rendered = fetch_page(url, render=True)
        if rendered.content and text_blocks(rendered.content):
            return rendered
    return page


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
                pages = list(ex.map(_fetch_readable, candidates))

        url = ""
        for candidate, page in zip(candidates, pages):
            if page.status_code == 200 and page.content and \
                    page_belongs_to_firm(candidate, page.content, firm_name):
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

        page = _fetch_readable(url)
        if page.status_code != 200 or not page.content:
            return out
        if firm_name and not page_belongs_to_firm(url, page.content, firm_name):
            return out  # Guard 2: this page is not this firm's

        # Prefer schema.org Person data when present — machine-readable and
        # survives JS rendering intact even when visible text does not.
        jsonld = _clean_roster_names(jsonld_names(page.content, "Person"))

        # Else per-element blocks, never one flattened string: flattening
        # glues neighbouring "Email" / "LinkedIn" labels onto a name.
        blocks = text_blocks(page.content)
        if not jsonld and not blocks:
            return out  # a JS-rendered shell asserts nothing we can read

        scraped = _clean_roster_names(blocks)
        names, seen = [], set()
        for n in jsonld + scraped:  # JSON-LD names are authoritative; union
            k = person_norm_key(n)
            if k and k not in seen:
                seen.add(k)
                names.append(n)

        out["firm"] = firm_name or firm_name_from_page(page.content, url)
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
