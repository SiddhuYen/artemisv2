"""Which of this node's pages are worth reading in full.

Per-source extraction sends a whole page to the strong model, once per fetched
result. Nothing decides whether a given page is worth that. On one measured
/connect -- Charlie Warren -> Donald Trump, depth 2 -- it was 1,043 Sonnet
calls, 1.72M input tokens, $10.41 of a $10.49 route. The 218 searches cost
$0.22. Reading the pages was 99% of the bill, and the answer was "no route".

Every guard built around it governs which NODES to walk or which NAMES to keep.
None governs which PAGES to read, so a node that survives to expansion has all
of its pages read at full price, whatever they turn out to contain -- and for a
famous-adjacent node that is dozens of news articles about matters with no
bearing on the route.

This is one call per NODE that returns which pages deserve the deep read.
Everything else falls back to spaCy, which is the extractor this pipeline used
by default before Claude extraction existed and is still the documented second
tier: worse at relationships, free, and entirely adequate for a page that
turned out to be about a lawsuit.

Selection is BY INDEX into the caller's own list, so the model chooses among
pages that were actually fetched and cannot conjure a URL. It is judging titles
and snippets -- the same thing a person scanning search results judges -- and
the cost of a wrong keep is one extraction, while a wrong drop costs the
relationships on one page, recoverable next time that page is fetched.

BOUNDED ON FAILURE, not open-ended. An unreachable model does not silently
restore the old bill: the caller falls back to the highest-ranked
EXTRACT_DEEP_MAX_PAGES results, so the worst case is capped rather than
uncapped. That is the opposite choice from the frontier triage, and
deliberately -- there, failing open costs a handful of searches; here it costs
ten dollars.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .. import config
from .claude_client import call_json, claude_available

_PROMPT = """You are deciding which web pages are worth reading in full.

We are mapping the professional network of: {subject}
In order to find a connection to: {target}

Each page below was returned by a search for {subject}. Reading one in full
costs a large model call; skipping it means a cheaper reader handles it and
only obvious relationships are picked up.

Keep a page when its title or snippet suggests it NAMES OTHER PEOPLE connected
to {subject} -- colleagues, co-founders, board members, family, co-authors,
people at the same event or organisation. Those are what a route is built from.

Skip a page that is about {subject} alone, or about a topic rather than a
relationship: product announcements, market commentary, legal proceedings,
opinion pieces, profiles that name nobody else, listings, and anything whose
relevance is only that the name appears.

Pages:
{pages}

Return the numbers of the pages worth reading in full, best first, at most
{limit}. Return an empty list if none of them look like they name anyone.

why: one short clause explaining the cut."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "array", "items": {"type": "integer"}},
        "why": {"type": "string"},
    },
    "required": ["keep", "why"],
    "additionalProperties": False,
}


def is_active() -> bool:
    return bool(config.EXTRACT_PAGE_TRIAGE) and claude_available()


def select(subject: str, target: str, pages: Sequence[dict]) -> Optional[List[int]]:
    """`pages`: [{title, snippet}] -> indices worth a deep read, or None.

    None means "no verdict" and is distinct from an empty list, which is a real
    verdict that nothing here names anyone. The caller must treat them
    differently: None falls back to a bounded default, [] skips deep extraction
    entirely for this node.
    """
    if not is_active() or not pages:
        return None
    limit = max(1, int(config.EXTRACT_DEEP_MAX_PAGES))
    rendered = "\n".join(
        f"  {i}. {(p.get('title') or '(untitled)')[:120]}\n"
        f"     {(p.get('snippet') or '')[:220]}"
        for i, p in enumerate(pages, 1))
    payload = call_json(
        _PROMPT.format(subject=subject, target=target or "anyone",
                       pages=rendered, limit=limit),
        schema=_SCHEMA, model=config.PAGE_TRIAGE_MODEL,
        max_tokens=16 * limit + 256)
    if not payload:
        return None

    keep: List[int] = []
    for idx in (payload.get("keep") or []):
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        # Out of range dropped, not clamped -- a clamped index spends a
        # whole-page call on whichever result sat at the boundary.
        if 1 <= i <= len(pages) and (i - 1) not in keep:
            keep.append(i - 1)
        if len(keep) >= limit:
            break
    return keep
