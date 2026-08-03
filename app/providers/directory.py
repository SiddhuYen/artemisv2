"""Org-keyed staff/professional directory scraping.

The sibling of firms.py. firms.py answers "who does this PERSON work with"
by locating a roster from the person's own name; this answers "who works at
this ORG", starting from an organization the graph already established the
subject belongs to. Both share one implementation of the page-level guards
(rosters.py) so the directory path inherits, rather than re-earns, every bug
those guards were written for.

Why this exists at all: for a non-famous person, their employer's own
directory is the densest source of real professional connections available
anywhere -- far denser than news, publications or events, which are silos
built around people who get written about. Alpha's whole premise is walking
UP from an ordinary person, and this is the source that actually makes that
possible.

It also replaces a broken implementation of the same idea. The
`current_employer_leadership` search-strategy angle used to fire
'"{org}" leadership team OR executives' into the ordinary PROSE extraction
path, which meant the proximity gate (extraction/spacy_extractor) decided
the outcome: on a long leadership page that never names a VP-level subject
it dropped every extracted entity, and on a short one it accepted all of
them, wiring the entire exec roster to the subject as unevidenced
colleagues. A roster is a structural assertion; running it through a
sentence-proximity heuristic could only ever produce one of those two
failures.

WHAT THIS MODULE DOES NOT DECIDE: whether the people it finds are the
subject's colleagues. It reports who a verified directory page lists, and
whether the subject is among them. The caller (graph.expansion phase 0e)
owns the edge-emission rule, because that rule is about evidence, not
scraping -- see `directory()`'s return contract.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional
from urllib.parse import urlparse

from .. import config
from ..utils.htmltext import jsonld_names, text_blocks
from ..utils.names import org_norm_key, person_norm_key
from . import cache
from .base import SearchResult
from .rosters import (
    DIRECTORY_HINTS,
    clean_roster_names,
    fetch_readable,
    is_leadership_url,
    is_roster_url,
    org_name_from_page,
    page_belongs_to_org,
)


def pack_for(industry: str) -> dict:
    """The sector query pack matching `industry`, or the default pack.

    Keyword match, not a model call: node_profiler's `industry` is already a
    grounded free-text phrase ("Oracle ERP consulting"), and the job here is
    only to route it to one of a handful of pre-written query sets. Falling
    through to "default" is a normal outcome, not a failure -- an unknown or
    unmatched industry still gets the generic team-page queries.
    """
    text = (industry or "").lower()
    if text and text != "unknown":
        for key, pack in config.DIRECTORY_PACKS.items():
            if key == "default":
                continue
            if any(token in text for token in pack["match"]):
                return pack
    return config.DIRECTORY_PACKS["default"]


class DirectoryProvider:
    """Locate and scrape an organization's own staff/professional directory.

    `search` is a callable(query: str) -> List[SearchResult] used ONLY to
    LOCATE candidate pages -- the same contract as FirmsProvider. The page
    itself is always the evidence; search never asserts a relationship.
    """

    name = "directory"

    def __init__(self, search: Optional[Callable[[str], List[SearchResult]]] = None,
                 official_domain: Optional[Callable[[str], str]] = None) -> None:
        self._search = search
        # Injected rather than imported so this provider stays a leaf: it
        # never reaches back into the orchestrator, and a test can exercise
        # the Wikidata-verified path without any network.
        self._official_domain_fn = official_domain

    def available(self) -> bool:
        return bool(config.DIRECTORY_ENABLED) and self._search is not None

    def _official_domain(self, org_name: str) -> str:
        """Registrable domain of the org's Wikidata official website, or "".

        Best-effort in the same sense as everything else here: a failure means
        Guard 2 falls back to name matching alone, exactly as before.
        """
        if not self._official_domain_fn:
            return ""
        try:
            return self._official_domain_fn(org_name) or ""
        except Exception:
            return ""

    # --- locate ------------------------------------------------------------
    def _plan(self, org_name: str, industry: str, size_tier: str):
        """(queries, url_predicate) for this org's size and industry.

        A large org (or one whose profile never grounded, which is treated the
        same way on purpose) gets ONLY its leadership page: its full staff
        directory is thousands of people who are weak bridges individually and
        would swamp the node caps collectively. Small/mid orgs get the sector
        pack, because at that size the directory genuinely is the subject's
        professional circle.
        """
        if size_tier in config.DIRECTORY_FULL_SIZE_TIERS:
            queries = [q.format(org=org_name) for q in pack_for(industry)["queries"]]
            return queries, (lambda url: is_roster_url(url, extra_hints=DIRECTORY_HINTS))
        queries = [q.format(org=org_name) for q in config.DIRECTORY_LEADERSHIP_QUERIES]
        return queries, is_leadership_url

    def find_directory_page(self, org_name: str, industry: str = "",
                            size_tier: str = "", official_domain: str = "") -> Optional[str]:
        """The org's own directory URL, or None. Verified by Guard 2."""
        if not org_name or not self.available():
            return None
        queries, accept = self._plan(org_name, industry, size_tier)

        candidates: List[str] = []
        for query in queries:
            try:
                results = self._search(query)
            except Exception:
                continue
            for result in results:
                if accept(result.url) and result.url not in candidates:
                    candidates.append(result.url)
            if candidates:
                break

        # Prefer the shallowest path: "/team" is the directory, "/team/andy"
        # is one person's bio page.
        candidates.sort(key=lambda u: len(urlparse(u).path.strip("/").split("/")))
        candidates = candidates[: config.DIRECTORY_MAX_CANDIDATES]
        if not candidates:
            return None

        # Fetch every candidate concurrently, then pick the FIRST one that
        # verifies -- order still decides, only the I/O is parallelized.
        with ThreadPoolExecutor(max_workers=min(len(candidates),
                                                config.SEARCH_WORKERS)) as ex:
            pages = list(ex.map(fetch_readable, candidates))
        for candidate, page in zip(candidates, pages):
            if (page.status_code == 200 and page.content
                    and page_belongs_to_org(candidate, page.content, org_name,
                                            official_domain=official_domain)):
                return candidate
        return None

    # --- scrape ------------------------------------------------------------
    def directory(self, org_name: str, industry: str = "",
                  size_tier: str = "") -> dict:
        """{org, url, members, overflow, leadership_only} for `org_name`.

        `overflow` is True when the page lists more people than
        DIRECTORY_MAX_MEMBERS. It matters: a 200-person directory must never
        be materialized as a 200-clique of mutual colleagues, so the caller
        uses this to fall back to membership-only edges (see expansion's
        phase 0e). firms.py computes the same flag and its caller ignores it;
        this one does not.

        `members` is who the page LISTS. It is deliberately not filtered to
        "colleagues of the subject" -- this module has no subject.
        """
        out = {"org": org_name, "url": "", "members": [], "overflow": False,
               "leadership_only": size_tier not in config.DIRECTORY_FULL_SIZE_TIERS,
               "status": "ok"}
        if not org_name or not self.available():
            out["status"] = "disabled"
            return out

        key = cache.make_key(self.name, "directory",
                             f"{org_norm_key(org_name)}::{size_tier}::{industry}")
        cached = cache.get(key)
        if cached is not None:
            return cached

        # One authoritative identity lookup per org, reused by both guards.
        official_domain = self._official_domain(org_name)
        url = self.find_directory_page(org_name, industry, size_tier, official_domain)
        if not url:
            # Nothing survived the shape check + Guard 2. Distinguishing this
            # from the other zero-member outcomes is the whole point of
            # `status`: a silent empty result is indistinguishable from "this
            # org has no directory", which is exactly how the Wikimedia
            # outage hid for so long.
            out["status"] = "no_verified_page"
            cache.set(key, "directory", out, config.CACHE_TTL_PAGE)
            return out

        page = fetch_readable(url)
        if page.status_code != 200 or not page.content:
            out["status"] = f"fetch_{page.status_code}"  # 403 => bot-blocked
            out["url"] = url
            return out
        # Guard 2 again on the page we actually scrape -- find_directory_page
        # verified a page, but re-verifying here keeps `directory()` safe to
        # call with a URL that came from anywhere.
        if not page_belongs_to_org(url, page.content, org_name,
                                   official_domain=official_domain):
            out["status"] = "identity_mismatch"
            return out

        # Prefer schema.org Person data when present -- machine-readable and
        # survives JS rendering intact even when visible text does not.
        jsonld = clean_roster_names(jsonld_names(page.content, "Person"))
        blocks = text_blocks(page.content)
        if not jsonld and not blocks:
            # No readable text nodes at all: a JS-rendered shell. THIS is the
            # case a headless browser would recover, and the only one -- see
            # config.MAX_HTML_CHARS for why most apparent "SPAs" were really
            # just truncation. Recorded so a future bench can measure how
            # often rendering would actually pay for itself.
            out["status"] = "js_shell"
            out["url"] = url
            return out

        scraped = clean_roster_names(blocks)
        names, seen = [], set()
        for n in jsonld + scraped:  # JSON-LD names are authoritative; union
            k = person_norm_key(n)
            if k and k not in seen:
                seen.add(k)
                names.append(n)

        out["org"] = org_name or org_name_from_page(page.content, url)
        out["url"] = url
        out["members"] = names[: config.DIRECTORY_MAX_MEMBERS]
        out["overflow"] = len(names) > config.DIRECTORY_MAX_MEMBERS
        # A verified page that yields no names is its own diagnosis: the
        # markup was readable and simply had no person-shaped text (or the
        # junk filter took all of it).
        out["status"] = "ok" if names else "no_names_found"
        cache.set(key, "directory", out, config.CACHE_TTL_PAGE)
        return out

    # NB: this provider deliberately exposes NO "render the roster to prose"
    # helper, unlike firms.colleagues_text. Its caller builds edges directly
    # from `members` (see expansion phase 4f). Rendering scraped roster names
    # to a sentence and re-extracting them puts spaCy NER between the graph
    # and names that a deterministic shape filter already accepted -- and
    # en_core_web_sm's PERSON/ORG confusion on non-Anglo names would then
    # silently drop exactly the people this feature exists to find.
