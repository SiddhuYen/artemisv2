"""A one-query test for whether a contact has any web footprint at all.

Most people in a real contact export are not written about anywhere: a phone
contact, a recruiter, someone met once at a conference. Enriching them costs
the same ~35 queries as enriching a founder and returns nothing. The probe
turns that into ONE query, and only the contacts that pass go on to the full
sweep — the single largest saving available on a long-tail run.

The query is built exactly the way expansion._process_person builds its
first silo query (silo template + the disambiguating context), which makes the
probe close to free when it PASSES: the provider layer is cache-first, so the
full sweep that follows reuses the cached response instead of re-issuing it.
A contact that fails cost one query instead of thirty-five. If the silo
definitions ever drift out of step with this, the cost is one extra query per
contact, not a broken probe.

Deliberately NOT a notability check. ORCH.notable_set is cheaper still, but it
answers "is this person on Wikipedia", and virtually every real contact fails
that while plenty of them have news, company pages and press releases worth
extracting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .. import config
from ..silos import SILOS
from ..utils.names import normalize


@dataclass
class ProbeResult:
    has_footprint: bool
    hits: int
    query: str
    sample_url: str = ""


def probe_query(name: str, context: str = "") -> str:
    """The query to probe with — the first silo query, context appended.

    Mirrors _process_person's construction (`f"{query} {context}"`) so the
    result lands in the same cache entry the real sweep will look for.
    """
    template = SILOS[0].render_queries(name)[0] if SILOS else f'"{name}"'
    return f"{template} {context}".strip() if context else template


def _mentions(name: str, text: str) -> bool:
    """Whether `text` plausibly refers to `name`.

    Every meaningful token of the name has to appear, not just one: a search
    for a person at a company routinely returns pages about the COMPANY that
    never mention the person, and counting those as a footprint would defeat
    the whole point of probing.
    """
    haystack = normalize(text)
    if not haystack:
        return False
    tokens = [t for t in normalize(name).split() if len(t) > 1]
    return bool(tokens) and all(t in haystack for t in tokens)


def probe_footprint(name: str, context: str = "", search=None) -> ProbeResult:
    """One search; decide whether the full sweep is worth paying for.

    `search` is injected for testing and defaults to the shared orchestrator.
    Any provider failure counts as a PASS, not a fail — a transient outage
    must not permanently mark a real contact as having no web presence.
    """
    query = probe_query(name, context)
    if search is None:
        from ..graph.expansion import ORCH
        search = lambda q: ORCH.search(q, is_person=True)  # noqa: E731

    try:
        results = search(query) or []
    except Exception:
        return ProbeResult(True, 0, query)

    hits: List[object] = [
        r for r in results
        if _mentions(name, f"{getattr(r, 'title', '')} {getattr(r, 'snippet', '')}")
    ]
    sample: Optional[str] = getattr(hits[0], "url", "") if hits else ""
    return ProbeResult(
        has_footprint=len(hits) >= config.ENRICH_PROBE_MIN_HITS,
        hits=len(hits), query=query, sample_url=sample or "",
    )
