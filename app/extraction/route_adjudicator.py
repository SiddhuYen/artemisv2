"""Before answering "no connection", show the model the work and ask.

connect_people used to return not-connected the moment its own machinery ran
out: the pathfinder found nothing, or found candidates that hop verification
threw out, and that was the answer. Nothing ever looked at what the walk had
actually turned up and asked whether it had stopped too early.

That is a bad place to stop, because by then Artemis is holding exactly the
context needed to judge: who was explored on each side, which routes were
proposed, and the verifier's own words for why each was rejected. A person
reading that page would often say "you never checked X against Y" -- and be
right. Charlie Warren -> Donald Trump is the case: the walk quit while Sam
Altman sat unexpanded in the graph with 34 edges, and neither "Charlie Warren
and Sam Altman are both Y Combinator" nor "Sam Altman has met Trump repeatedly"
was ever a query.

TWO ACTIONS, PRICED DIFFERENTLY, AND THAT ASYMMETRY IS THE CONTAINMENT.

  probe  - "search these two people together". One query. The model may name
           anyone, including someone not in the graph, because the answer comes
           back from a fetched page either way: a wrong guess costs a search.

  expand - "walk this node's neighborhood". ~35 queries. Restricted to nodes
           the walk ALREADY ranked and handed to the model, referenced by index
           into that shortlist. A hallucinated name cannot become an expansion.

So the model steers spending it cannot invent, and every edge that results is
still read off a page by the ordinary search path. Nothing here writes to the
graph.

Fails closed to `none`, which is the old behavior -- an unreachable model, a
refusal, a malformed answer and an honest "there is nothing here" all end the
same way, with the caller reporting no route.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import config
from ..utils.names import person_norm_key
from .claude_client import call_json, claude_available

ACTIONS = ("probe", "expand", "none")

_PROMPT = """A search for a connection between two people has come up empty. \
Decide whether it stopped too early.

FROM: {name_a}{context_a}
TO:   {name_b}{context_b}

Explored around {name_a}: {explored_a}
Explored around {name_b}: {explored_b}

Candidate routes that were found and then REJECTED on inspection:
{rejected}

Nodes available to expand (reference by number):
{shortlist}

You have two moves.

"probe" -- name pairs of people to search for TOGETHER, when you believe their \
connection is publicly documented. Cheap: one search each. You may name anyone, \
including people not listed above, because the search is what decides. Prefer \
pairs where one side is already near one endpoint and the other side is near, \
or is, the other endpoint.

"expand" -- list numbers from the shortlist whose wider network is worth \
mapping. Expensive: about 35 searches each. Only use this when no specific pair \
suggests itself and a node is clearly central to the gap.

"none" -- the two are genuinely unlikely to be connected within reach, or the \
rejected routes were rejected correctly and nothing else is promising.

Prefer "probe" over "expand". Prefer a specific documented pair over a hopeful \
one; each probe you name that returns nothing is a wasted search.

Give at most {max_probes} pairs and at most {max_expand} numbers.

why: one sentence naming the specific gap you are trying to close."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
        "expand": {"type": "array", "items": {"type": "integer"}},
        "why": {"type": "string"},
    },
    "required": ["action", "pairs", "expand", "why"],
    "additionalProperties": False,
}


def is_active() -> bool:
    return bool(config.CONNECT_ADJUDICATE_NO_ROUTE) and claude_available()


def _ctx(text: str) -> str:
    return f" ({text.strip()})" if (text or "").strip() else ""


def _render(names: List[str], limit: int) -> str:
    return ", ".join(names[:limit]) if names else "(nothing)"


def decide(name_a: str, name_b: str, context_a: str = "", context_b: str = "",
           explored_a: Optional[List[str]] = None,
           explored_b: Optional[List[str]] = None,
           rejected: Optional[List[str]] = None,
           shortlist: Optional[List[str]] = None) -> Optional[dict]:
    """-> {action, pairs, expand, why} or None.

    `shortlist` is the ONLY thing an "expand" may name -- returned as resolved
    names, already validated against it, so the caller never has to trust an
    index it did not hand out.
    """
    if not is_active():
        return None
    shortlist = shortlist or []
    max_probes = max(1, int(config.CONNECT_ADJUDICATE_MAX_PROBES))
    max_expand = max(0, int(config.CONNECT_ADJUDICATE_MAX_EXPAND))

    payload = call_json(
        _PROMPT.format(
            name_a=name_a, name_b=name_b,
            context_a=_ctx(context_a), context_b=_ctx(context_b),
            explored_a=_render(explored_a or [], 40),
            explored_b=_render(explored_b or [], 40),
            rejected="\n".join(f"  - {r}" for r in (rejected or [])) or "  (none)",
            shortlist="\n".join(f"  {i}. {n}" for i, n in enumerate(shortlist, 1))
                      or "  (none)",
            max_probes=max_probes, max_expand=max_expand),
        schema=_SCHEMA, model=config.ROUTE_ADJUDICATOR_MODEL,
        max_tokens=140 * (max_probes + max_expand) + 512)
    if not payload:
        return None

    action = payload.get("action")
    if action not in ACTIONS:
        return None

    endpoints = {person_norm_key(name_a), person_norm_key(name_b)}
    pairs: List[Dict[str, str]] = []
    seen = set()
    for row in (payload.get("pairs") or []):
        a = str((row or {}).get("a", "") or "").strip()
        b = str((row or {}).get("b", "") or "").strip()
        ka, kb = person_norm_key(a), person_norm_key(b)
        # A pair that is just the two endpoints re-asks the question that
        # already failed; a pair with itself asks nothing.
        if not ka or not kb or ka == kb or {ka, kb} == endpoints:
            continue
        if (ka, kb) in seen or (kb, ka) in seen:
            continue
        seen.add((ka, kb))
        pairs.append({"a": a, "b": b})
        if len(pairs) >= max_probes:
            break

    # Out of range is DROPPED, not clamped: a clamped index silently spends ~35
    # queries on whichever node happened to sit at the boundary.
    expand: List[str] = []
    for idx in (payload.get("expand") or []) if max_expand else []:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(shortlist) and shortlist[i - 1] not in expand:
            expand.append(shortlist[i - 1])
        if len(expand) >= max_expand:
            break

    if action == "probe" and not pairs:
        action = "none"
    if action == "expand" and not expand:
        action = "none"
    return {"action": action, "pairs": pairs, "expand": expand,
            "why": str(payload.get("why", "") or "")[:300]}
