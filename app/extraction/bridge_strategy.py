"""Hop-0 reasoning: which of the operator's own contacts to walk first.

network/ranking.score_contacts answers "who scores highest" with a fixed
additive formula -- seniority, shared employers, a company-coverage decay. It
is free, deterministic and encodes real signal, but it cannot reason about the
SHAPE of a particular reach: that a mid-size Oracle-consulting shop's VP of
Sales is a better bridge to Larry Ellison than a more senior contact in an
unrelated industry is a judgment about two contexts, not a sum of weights.

This is search_strategy's idea (Alpha step 6) moved to the front of the walk,
where it decides who to spend the first queries on rather than which queries to
spend on an already-chosen person.

Two containment rules carried over unchanged, because they are what make the
per-node version safe:

  1. The model PICKS, it never writes. Candidates are referenced by index into
     a shortlist the caller built; anything outside that range is discarded, so
     a hallucinated name cannot become a node to expand. The angle likewise
     comes from a fixed enum.

  2. Judge only what you're given. The prompt states the contacts' employers
     and titles and the target's known organizations; it explicitly refuses
     outside knowledge, because "everyone knows a Goldman VP can reach anyone"
     is exactly the training-data prior that produces confident, ungrounded
     picks.

The shortlist is deterministic-first for a practical reason too: a 1,188-contact
export does not fit in a prompt, and asking the model to rank all of them would
trade a cheap exact computation for an expensive approximate one. Rank, then
reason over the top of the ranking.

Fails closed. No key, a refusal, a timeout, a malformed pick -- every path
returns None and the caller keeps the deterministic order it already had.
"""
from __future__ import annotations

from typing import List, Optional

from .. import config
from .claude_client import call_json, claude_available

# Same vocabulary as search_strategy's angles, plus the two that only make
# sense when choosing a PERSON rather than a query: the reach may be social
# rather than professional, and the best bridge may be the one who shares the
# target's own institution regardless of anything else about them.
_ANGLES = [
    "shared_employer",       # someone inside the target's own organization
    "industry_adjacency",    # same professional world, different employer
    "shared_institution",    # school, board, or program in common
    "social_proximity",      # a personal rather than professional tie
    "generic",               # nothing in the facts justifies an angle
]

_PROMPT = """Someone wants to reach a target person through their own network. \
Choose which of their contacts to search FIRST.

Reaching FROM: {origin}{origin_context_line}
Trying to reach: {target}{target_context_line}
Target's known organizations: {target_orgs}

Their contacts (index. name -- employer -- title):
{candidates}

Decide, using ONLY the facts above -- not outside knowledge about any person, \
company, or how careers usually work:

1. angle: which kind of connection is most likely to bridge to THIS target.
   - shared_employer: a contact inside the target's own organization.
   - industry_adjacency: same professional world, different employer.
   - shared_institution: a school, board, or program in common.
   - social_proximity: a personal rather than professional tie.
   - generic: nothing in the facts above justifies preferring any angle. Say \
this rather than inventing a rationale -- a wrong angle sends the first and \
most expensive searches in the wrong direction.

2. picks: the {n_picks} contact INDEXES most worth searching first, best \
first. Choose only from the indexes listed above. Fewer than {n_picks} is \
fine, and an empty list is the right answer when nothing listed is a \
plausible bridge.

3. why: one sentence, grounded only in the facts above.
"""

_ORIGIN_CONTEXT_LINE = " ({origin_context})"
_TARGET_CONTEXT_LINE = " ({target_context})"

_SCHEMA = {
    "type": "object",
    "properties": {
        "angle": {"type": "string", "enum": _ANGLES},
        "picks": {"type": "array", "items": {"type": "integer"}},
        "why": {"type": "string"},
    },
    "required": ["angle", "picks", "why"],
    "additionalProperties": False,
}


def is_active() -> bool:
    return bool(config.BRIDGE_STRATEGY_ENABLED) and claude_available()


def _candidate_line(i: int, contact) -> str:
    """One shortlist row. Carries the deterministic ranker's own findings
    (`bridge_reasons`: shared_employer / shared_school / silo_affinity) because
    those are FACTS computed from the data, not opinions -- handing them over
    lets the model reason about which of several real overlaps matters most
    for this particular target, instead of re-deriving overlap from names."""
    line = f"{i}. {contact.display_name} -- {contact.context or 'unknown employer'}"
    # getattr, not attribute access: `candidates` is typed as a plain list and
    # these two are populated only once a contact has been through
    # contact_profiler, so a shortlist built before the backfill ran (or by a
    # caller passing its own objects) must degrade to the shorter line rather
    # than raise inside a front that is meant to fail soft.
    domain = getattr(contact, "domain", None)
    footprint = getattr(contact, "footprint", None)
    if domain:
        line += f" -- domain: {domain}"
    if footprint:
        line += f" -- searchable: {footprint}"
    if contact.bridge_reasons:
        line += f" -- overlaps with target: {', '.join(contact.bridge_reasons)}"
    return line


def choose(origin_name: str, origin_context: str, target_name: str,
           target_context: str, target_orgs: List[str], candidates: List,
           n_picks: Optional[int] = None) -> Optional[dict]:
    """{angle, picks, why} or None when inactive/failed.

    `picks` are indexes INTO `candidates`, already validated: out-of-range and
    duplicate values are dropped rather than corrected, since a model that
    returned index 40 for a list of 15 was not making a near-miss judgment
    about contact 14 and pretending otherwise would invent a decision nobody
    made.
    """
    if not is_active() or not candidates:
        return None
    n_picks = n_picks or config.BRIDGE_PRIORITY_PICKS
    prompt = _PROMPT.format(
        origin=origin_name,
        origin_context_line=_ORIGIN_CONTEXT_LINE.format(origin_context=origin_context)
        if origin_context else "",
        target=target_name,
        target_context_line=_TARGET_CONTEXT_LINE.format(target_context=target_context)
        if target_context else "",
        target_orgs=", ".join(target_orgs) if target_orgs else "unknown",
        candidates="\n".join(_candidate_line(i, c) for i, c in enumerate(candidates)),
        n_picks=n_picks,
    )
    payload = call_json(prompt, schema=_SCHEMA,
                        model=config.BRIDGE_STRATEGY_MODEL, max_tokens=512)
    if payload is None:
        return None

    angle = payload.get("angle", "generic")
    if angle not in _ANGLES:
        angle = "generic"

    picks, seen = [], set()
    for raw in (payload.get("picks") or []):
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            picks.append(idx)
        if len(picks) >= n_picks:
            break
    return {"angle": angle, "picks": picks,
            "why": str(payload.get("why", "") or "")[:300]}
