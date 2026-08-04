"""Who might stand between these two? Asked of the model, answered by search.

The cheap pair search asks one question -- are A and B named together -- and
for a pair three hops apart the honest answer is no. Sanjay Ghemawat and Larry
Page are the case that motivated this: nine results for the pair query, not one
of them stating a direct tie, and Jeff Dean named on every single one. The
intermediary was on the page; nothing was looking for an intermediary.

Expansion would eventually find him, at ~35 queries per node across two
neighborhoods. This asks instead, for one model call and a handful of searches:
who plausibly sits in the middle, and is that borne out?

THE CONTAINMENT RULE IS THE WHOLE DESIGN. The model's answer never becomes an
edge. It becomes a SEARCH QUERY. Every edge that reaches the graph still comes
from _direct_pair_search reading a fetched page, exactly as if the operator had
typed the intermediary's name themselves. So the failure mode of a confidently
wrong guess is a wasted search, not a fabricated connection -- which matters
more here than anywhere else in the codebase, because "which famous people know
each other" is precisely where a language model's training prior is most fluent
and least accountable.

That is also why the prompt asks for people whose connection to BOTH endpoints
is publicly documented, rather than for people who are plausibly connected. The
first is checkable by the step that follows; the second is a vibe.

Fails closed, like every other stage here: no key, a refusal, a timeout, a
malformed answer -- all return an empty list, and the caller proceeds to the
expansion it would have run anyway.
"""
from __future__ import annotations

from typing import List

from .. import config
from ..utils.names import person_norm_key
from .claude_client import call_json, claude_available

_PROMPT = """Two people may be connected through someone else. Name the people \
most likely to sit BETWEEN them.

Person A: {name_a}{context_a}
Person B: {name_b}{context_b}

Name real, specific individuals whose connection to BOTH A and B is PUBLICLY \
DOCUMENTED -- written about in news, company pages, papers, board listings or \
biographies. Each name you give will be checked with a web search, so a name \
that is merely plausible costs a wasted search and helps nobody.

Prefer someone who worked directly with both, sat on a board with both, \
co-authored with both, or is repeatedly written about alongside both. Do not \
name an organization, a job title, or a group -- only a person. Do not name A \
or B themselves.

If you do not know of anyone whose link to both is actually documented, return \
an empty list. An empty list is a useful answer here; a guess is not.

Return at most {limit} names, best first. For each: the person's full name as \
it would appear in an article, and one short clause naming what documented \
thing ties them to A and to B."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["name", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def is_active() -> bool:
    return bool(config.CONNECT_ASK_CLAUDE_BRIDGE) and claude_available()


def _context_line(label: str, context: str) -> str:
    return f" ({context.strip()})" if (context or "").strip() else ""


def propose(name_a: str, name_b: str, context_a: str = "",
            context_b: str = "") -> List[dict]:
    """[{name, why}] -- candidate intermediaries, best first. Never edges.

    The returned names are search terms. Nothing here asserts that any of them
    is actually connected to anyone; that is the caller's next step to settle.
    """
    if not is_active() or not (name_a or "").strip() or not (name_b or "").strip():
        return []
    limit = max(1, int(config.CONNECT_BRIDGE_HYPOTHESES))
    payload = call_json(
        _PROMPT.format(name_a=name_a, name_b=name_b,
                       context_a=_context_line("A", context_a),
                       context_b=_context_line("B", context_b),
                       limit=limit),
        schema=_SCHEMA, model=config.BRIDGE_HYPOTHESIS_MODEL,
        max_tokens=120 * limit + 256)
    if not payload:
        return []

    # Drop anything that cannot be searched as a person, and anything naming an
    # endpoint back at us -- "A is connected to B via A" is not a bridge, and
    # would send the pair search off to re-answer the question that just failed.
    endpoints = {person_norm_key(name_a), person_norm_key(name_b)}
    out: List[dict] = []
    seen = set()
    for row in (payload.get("candidates") or []):
        name = str((row or {}).get("name", "") or "").strip()
        key = person_norm_key(name)
        if not key or key in endpoints or key in seen:
            continue
        # A single token is almost always a first name, an org, or a fragment;
        # searching it produces noise rather than a person.
        if len(name.split()) < 2 or len(name) > 120:
            continue
        seen.add(key)
        out.append({"name": name, "why": str(row.get("why", "") or "")[:200]})
        if len(out) >= limit:
            break
    return out
