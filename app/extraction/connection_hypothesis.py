"""Where is this node likely to be connected? Asked before it is searched.

Expansion's default move on arriving at a node is to fire the same ~35 generic
silo templates at it -- '"X" interview', '"X" board of directors', '"X"
university' -- and then read whatever comes back. The templates know nothing
about X. They are the same for a US senator, a postdoc and a regional sales
manager, and for most people most of them are structurally hopeless (see
network/silo_weights, which trims them from the CSV row alone).

This asks a different question first, using what the node's own free structured
enrichment already established -- their employer, their board seats, the
colleagues Wikidata and the roster pages just named: given THIS person, who or
what are they most likely to be publicly documented alongside, and which of
those is worth proving? The searches that follow are then aimed at confirming
specific, named connections rather than trawling for whatever a category query
happens to return.

THE CONTAINMENT RULE IS THE SAME ONE bridge_hypothesis USES, and it is the
whole design. The model's answer never becomes an edge. It becomes a SEARCH
QUERY, rendered from a fixed template in config.NODE_HYPOTHESIS_QUERIES -- the
model names entities, code writes queries. Every edge still arrives the
ordinary way: a page is fetched, and the extractor reads the relationship off
the sentence. So a confidently wrong guess costs one search, never a fabricated
connection.

Three further constraints follow from that rule, all enforced by the caller
(graph/expansion.py phase 0e) rather than trusted to the prompt:

  - The `relationship` a hypothesis carries is an EXPECTATION, never a label.
    It is recorded for inspection and used to order the work; it is never
    written onto an edge. What the tie actually is gets read off the evidence
    by the same relation_classifier that types every other edge.
  - Pages fetched to check a hypothesis are extracted under HYPOTHESIS_SILO, at
    a confidence multiplier of exactly 1.0. Predicting a connection must not
    make its confirmation cheaper to believe.
  - Only edges about the entity that was hypothesised are kept from those
    pages. A page returned by '"X" "Y"' may name a dozen other people, and
    those belong to whichever query genuinely surfaces them -- harvesting them
    here would let a guess about Y quietly seed the graph with everyone who
    happened to share Y's page.

Fails closed like every other stage here: no key, a refusal, a timeout, a
malformed answer -- all return an empty list, and the caller runs exactly the
silo search it would have run anyway.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

from .. import config
from ..models import RELATIONSHIP_TYPES
from ..providers import cache
from ..utils.names import org_norm_key, person_norm_key
from .claude_client import call_json, claude_available

# 'linkedin_1st' and 'podcast_guest' are excluded for the same reason
# relation_classifier excludes them: each is asserted only by its own
# structural source (an uploaded CSV row, an RSS episode) and must never be
# reachable by guessing. The rest are expectations only -- see the module
# docstring on why no value here ever reaches an edge.
_EXPECTED_TYPES = [t for t in RELATIONSHIP_TYPES
                   if t not in ("linkedin_1st", "podcast_guest")]

_PROMPT = """A relationship graph is being grown outward from one person. \
Before any searches are spent on them, decide WHERE they are most likely to be \
documented as connected -- and to whom it is worth proving.

Subject: {subject}{context}
{facts_block}{target_block}
Name specific people or organizations that the subject is likely to be \
PUBLICLY DOCUMENTED with: named together on the same page in news, a company \
or leadership page, a board listing, a paper, a team roster, a regulatory \
filing, a conference programme or an interview.

Each candidate is checked by a web search for the subject and that name \
together, and only what a fetched page actually states becomes a connection. \
So a candidate that is merely plausible costs a wasted search and helps \
nobody, and a candidate you are confident about but that nobody has written \
down is worth exactly as little.

Rules:
- Name a specific person (full name, as it would appear in an article) or a \
specific organization. Never a job title, a team, a category or a group \
("the board", "his colleagues", "several investors").
- Do not name the subject.
- Use the facts above where they help. You may go beyond them, but only for \
ties you believe are actually written down somewhere.
- Prefer ties that are specific and load-bearing -- a named cofounder, an \
employer's named leadership, a named board, a named coauthor -- over ties that \
merely put the subject in the same room as someone famous.
- relationship: the tie you EXPECT. It is used to prioritise the search, never \
to label a connection; the evidence decides that.
- If you do not know of anyone whose tie to the subject is actually \
documented, return an empty list. An empty list is a useful answer here; a \
guess is not.

Return at most {limit} candidates, best first."""

_CONTEXT = " ({context})"

_FACTS_BLOCK = """
Already known about the subject (from structured sources, this run):
{facts}
"""

_TARGET_BLOCK = """
This walk is trying to reach: {target}{target_context}
Prefer candidates that plausibly shorten the distance to them -- someone or \
something documented with BOTH -- but do not invent a link to them. A solid, \
well-documented tie that goes nowhere near the target is still worth more than \
a speculative one that would be perfect if it were real.
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["person", "org"]},
                    "relationship": {"type": "string", "enum": _EXPECTED_TYPES},
                    "why": {
                        "type": "string",
                        "description": "One clause naming the documented thing "
                                       "that ties them to the subject.",
                    },
                },
                "required": ["name", "kind", "relationship", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def is_active() -> bool:
    return bool(config.NODE_HYPOTHESIS_ENABLED) and claude_available()


def _key(subject: str, context: str, target: str, facts: List[str]) -> str:
    """Cache key. The FACTS are part of it, deliberately.

    A node re-encountered with more known about it than last time is a
    different question -- that is the entire premise of asking from what the
    structured enrichment found -- so widening the evidence must not be served
    a verdict formed without it.
    """
    blob = "||".join([subject, context, target] + list(facts))
    h = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    return cache.make_key("nodehypo", "v1", h)


def _render(subject_name: str, context: str, facts: List[str],
            target_name: str, target_context: str, limit: int) -> str:
    facts_block = ""
    if facts:
        facts_block = _FACTS_BLOCK.format(
            facts="\n".join(f"- {f}" for f in facts))
    target_block = ""
    if target_name:
        target_block = _TARGET_BLOCK.format(
            target=target_name,
            target_context=_CONTEXT.format(context=target_context)
            if target_context else "")
    return _PROMPT.format(
        subject=subject_name,
        context=_CONTEXT.format(context=context) if context else "",
        facts_block=facts_block, target_block=target_block, limit=limit)


def propose(subject_name: str, context: str = "",
            facts: Optional[List[str]] = None, target_name: str = "",
            target_context: str = "", exclude: Optional[set] = None) -> List[dict]:
    """[{name, kind, relationship, why}] -- candidates to search for. Never edges.

    `facts` are short grounded lines about the subject that this run already
    holds (employer, board seats, colleagues named by Wikidata/rosters). They
    are what makes this a judgment about THIS node rather than a recall of
    whatever the model knows about the name.

    `exclude` is a set of normalized keys (person_norm_key / org_norm_key) the
    caller has nothing left to prove about -- the subject themselves, and
    counterparts already strongly evidenced. Filtered here rather than trusted
    to the prompt.

    Cached per (subject, context, target, facts): re-expanding a node with the
    same evidence in hand asks the same question, and should not pay twice.
    """
    if not is_active() or not (subject_name or "").strip():
        return []
    limit = max(1, int(config.NODE_HYPOTHESIS_MAX))
    facts = [f for f in (facts or []) if f][:max(0, config.NODE_HYPOTHESIS_FACTS)]
    key = _key(subject_name, context, target_name, facts)
    payload = cache.get(key, track=False)
    if payload is None:
        payload = call_json(
            _render(subject_name, context, facts, target_name, target_context, limit),
            schema=_SCHEMA, model=config.NODE_HYPOTHESIS_MODEL,
            max_tokens=160 * limit + 256)
        if payload is None:
            return []
        cache.set(key, "nodehypo", payload, config.CACHE_TTL_WIKI)

    exclude = exclude or set()
    subject_key = person_norm_key(subject_name)
    out: List[dict] = []
    seen = set()
    for row in (payload.get("candidates") or []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "") or "").strip()
        kind = row.get("kind") if row.get("kind") in ("person", "org") else "person"
        if not name or len(name) > 120:
            continue
        # A single token is a first name, a category or a fragment when it is
        # meant to be a person -- searching it returns noise, not a person. An
        # ORG can legitimately be one word ("Oracle", "Netflix"), so the rule
        # applies only to the kind it is actually true of.
        if kind == "person" and len(name.split()) < 2:
            continue
        norm = person_norm_key(name) if kind == "person" else org_norm_key(name)
        if not norm or norm == subject_key or norm in exclude or (kind, norm) in seen:
            continue
        seen.add((kind, norm))
        relationship = row.get("relationship")
        if relationship not in _EXPECTED_TYPES:
            relationship = "unknown"
        out.append({
            "name": name, "kind": kind, "norm": norm,
            "relationship": relationship,
            "why": str(row.get("why", "") or "")[:200],
        })
        if len(out) >= limit:
            break
    return out
