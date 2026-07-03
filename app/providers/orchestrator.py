"""Search orchestration: provider routing + query deduplication + enrichment.

- dedup(pairs): collapse equivalent queries across silos to unique requests,
  keeping a mapping back to every originating silo.
- search(query, is_person): try providers in the configured route order,
  returning the first useful (non-empty) result set. Brave is primary; DDG is
  the last-resort fallback; Wikipedia participates for person queries.
- enrich_person(name): structured Wikipedia summary/links + Wikidata
  relationships, consulted before HTML scraping.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .. import config
from ..utils.names import normalize, person_norm_key
from .brave import BraveProvider
from .duckduckgo import DuckDuckGoProvider
from .edgar import EdgarProvider
from .openalex import OpenAlexProvider
from .opencorporates import OpenCorporatesProvider
from .propublica import ProPublicaProvider
from .stats import STATS
from .wikipedia import WikipediaProvider
from .wikidata import WikidataProvider

_PUNCT = re.compile(r"[^\w\s]")


def _canon(query: str) -> str:
    """Canonical dedup key: lowercase, punctuation-free, token-sorted."""
    s = _PUNCT.sub(" ", query.lower())
    tokens = sorted(t for t in s.split() if t)
    return " ".join(tokens)


def _name_matches(name: str, title: str) -> bool:
    """Does a Wikipedia page title plausibly belong to this person? Requires most
    of the person's name tokens to appear in the title — so 'Fred Volinsky' does
    NOT match '2026 United States House of Representatives elections...'."""
    n = set(person_norm_key(name).split())
    t = set(normalize(title).split())
    if not n:
        return False
    return len(n & t) / len(n) >= 0.6


class SearchOrchestrator:
    def __init__(self) -> None:
        self.brave = BraveProvider()
        self.wikipedia = WikipediaProvider()
        self.wikidata = WikidataProvider()
        self.openalex = OpenAlexProvider()
        self.opencorporates = OpenCorporatesProvider()
        self.edgar = EdgarProvider()
        self.propublica = ProPublicaProvider()
        self.duckduckgo = DuckDuckGoProvider()
        self._providers = {
            "brave": self.brave,
            "wikipedia": self.wikipedia,
            "duckduckgo": self.duckduckgo,
        }

    # --- query dedup ------------------------------------------------------
    def dedup(self, pairs: List[Tuple[object, str]]) -> Tuple[List[str], Dict[str, List[object]]]:
        """pairs: [(silo, query)]. Returns (unique_queries, query -> [silos]).

        Equivalent queries (same canonical form) across silos collapse to one
        request; every originating silo is preserved in the mapping.
        """
        seen: Dict[str, str] = {}
        q2silos: Dict[str, List[object]] = {}
        seen_keys: Dict[str, set] = {}
        removed = 0
        for silo, query in pairs:
            k = _canon(query)
            if k in seen:
                rep = seen[k]
                removed += 1
            else:
                seen[k] = query
                rep = query
                q2silos[rep] = []
                seen_keys[rep] = set()
            if silo.key not in seen_keys[rep]:
                seen_keys[rep].add(silo.key)
                q2silos[rep].append(silo)
        STATS.removed_by_dedup(removed)
        return list(q2silos.keys()), q2silos

    # --- routed search ----------------------------------------------------
    def search(self, query: str, is_person: bool = True) -> List["object"]:
        route = config.ROUTE_PERSON if is_person else config.ROUTE_DEFAULT
        for pname in route:
            if pname == "wikidata":
                continue  # structured-only; handled by enrich_person
            provider = self._providers.get(pname)
            if provider is None or not provider.available():
                continue
            results = provider.search(query)
            if results:
                return results
        return []

    def fetch(self, url: str):
        return self.duckduckgo.fetch(url)  # shared cache-first fetch

    def coauthors_enrichment(self, name: str) -> dict:
        """Academic coauthors (OpenAlex) for any person — independent of Wikipedia."""
        coauthors = self.openalex.coauthors(name)
        return {
            "coauthors": coauthors,
            "coauthors_text": self.openalex.coauthors_text(name, coauthors) if coauthors else "",
        }

    def officer_enrichment(self, name: str) -> dict:
        """Company co-officers (OpenCorporates) — business contacts incl. non-famous."""
        cols = self.opencorporates.officer_colleagues(name)
        return {
            "officers": cols,
            "officers_text": self.opencorporates.colleagues_text(name, cols) if cols else "",
        }

    def edgar_enrichment(self, name: str) -> dict:
        """SEC EDGAR co-insiders (public-company directors/officers/owners)."""
        cols = self.edgar.officer_colleagues(name)
        return {
            "edgar": cols,
            "edgar_text": self.edgar.colleagues_text(name, cols) if cols else "",
        }

    def notable_set(self, names: List[str]) -> set:
        """Return the subset of names that are 'notable' — i.e. have their own
        Wikidata-backed Wikipedia page. Used as a fame signal for reachability
        ranking. Cached (each name resolved at most once)."""
        notable = set()
        for name in dict.fromkeys(names):
            if not name:
                continue
            try:
                if self.wikipedia.wikidata_id(name):
                    notable.add(name)
            except Exception:
                continue
        return notable

    # --- structured enrichment -------------------------------------------
    def enrich_person(self, name: str) -> Optional[dict]:
        """Structured enrichment for a notable person (Tier-1 high recall):
          - Wikipedia full article (many more named people than the summary)
          - Wikipedia page links filtered to actual PEOPLE
          - Wikidata relationships + reverse colleague lookups (shared org/school)
        Returns None if the person has no Wikipedia page that is actually about
        THIS person (guards against tangential top-hits / namesakes)."""
        title = self.wikipedia.best_title(name)
        if not title or not _name_matches(name, title):
            return None  # top hit isn't this person's page (e.g. an election page)
        qid = self.wikipedia.wikidata_id(title)
        if not qid or not self.wikidata.is_human(qid):
            return None  # the page isn't a human entity (election/company/song/...)
        summary = self.wikipedia.summary(title)
        article = self.wikipedia.article_text(title)

        rels = self.wikidata.relationships(qid) if qid else []
        wd_text = self.wikidata.evidence_text(name, rels) if rels else ""

        # reverse: colleagues / classmates via shared org / school (high recall)
        colleagues = self.wikidata.colleagues(qid) if qid else []
        colleagues_text = self.wikidata.colleagues_text(name, colleagues) if colleagues else ""

        # nonprofit boards (ProPublica 990) for the subject's affiliated orgs
        nonprofit_text = ""
        if config.PROPUBLICA_ENABLED:
            org_names = list(dict.fromkeys(
                [c["org"] for c in colleagues if c.get("org")]
                + [r["name"] for r in rels if r.get("relationship_type") in
                   ("board_member", "cofounder")]
            ))[: config.PROPUBLICA_MAX_ORGS]
            board = []
            for org in org_names:
                board.extend(self.propublica.board_members(org))
            if board:
                nonprofit_text = self.propublica.colleagues_text(name, board)

        return {
            "title": title, "summary": summary, "article": article,
            "qid": qid, "wikidata_relationships": rels, "wikidata_text": wd_text,
            "colleagues": colleagues, "colleagues_text": colleagues_text,
            "nonprofit_text": nonprofit_text,
        }
