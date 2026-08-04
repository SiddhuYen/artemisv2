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

_PROMPT = """A search for a connection between two people came up empty. You are \
being shown both sides of it. Say which people on the LEFT could plausibly know \
which people on the RIGHT.

TRYING TO CONNECT: {name_a}{context_a}  →  {name_b}{context_b}

LEFT — people around {name_a} (their contacts and who the search found near them):
{left}

RIGHT — {name_b}, and the people the search found near {name_b}:
{right}

Candidate routes that were found and then REJECTED on inspection:
{rejected}

For each pairing you believe is PUBLICLY DOCUMENTED, give the left number and \
the right number. Each one becomes a single web search for those two names \
together, and the search is what decides -- so a pairing you are unsure of \
costs one query, and a pairing you invent to be helpful wastes it.

Pair across the two lists only. Two people from the same list tell us nothing \
about the other endpoint, however well they know each other. Pairing directly \
with {name_b} (right number 1) is usually the most valuable question you can \
ask, because it closes the gap in one hop.

Text in parentheses after a name is that person's company or role, not a person.

You may also ask to EXPAND a left-hand person's wider network, by their left \
number. That costs about 35 searches, so use it only when no specific pairing \
suggests itself and that person is clearly central to the gap.

Reply "none" if these two are genuinely unlikely to be connected through anyone \
shown, or if the rejected routes were rejected correctly and nothing here is \
promising.

At most {max_probes} pairings and {max_expand} expansions.

why: one sentence naming the specific gap you are trying to close."""


_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"left": {"type": "integer"},
                               "right": {"type": "integer"}},
                "required": ["left", "right"],
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
           left: Optional[List[str]] = None, right: Optional[List[str]] = None,
           rejected: Optional[List[str]] = None) -> Optional[dict]:
    """-> {action, pairs:[{a,b}], expand:[name], why} or None.

    BOTH lists are supplied by the caller and BOTH sides of every pairing are
    chosen by index into them. The model cannot name anyone at all, which is a
    stronger rule than this stage started with and closes two live failures in
    one go: a free-text pair produced "Convex" (the CONTEXT string the operator
    typed for Charlie Warren) searched against Donald Trump, and repeatedly
    paired the origin with whatever famous names happened to be in front of it.

    `right` must lead with name_b itself. Pairing someone on the left directly
    with the target is the single most valuable question available -- it closes
    the gap in one hop -- and making it index 1 is what makes it askable.
    """
    if not is_active():
        return None
    left = [n for n in (left or []) if (n or "").strip()]
    right = [n for n in (right or []) if (n or "").strip()]
    if not left or not right:
        return None
    max_probes = max(1, int(config.CONNECT_ADJUDICATE_MAX_PROBES))
    max_expand = max(0, int(config.CONNECT_ADJUDICATE_MAX_EXPAND))

    payload = call_json(
        _PROMPT.format(
            name_a=name_a, name_b=name_b,
            context_a=_ctx(context_a), context_b=_ctx(context_b),
            left="\n".join(f"  {i}. {n}" for i, n in enumerate(left, 1)),
            right="\n".join(f"  {i}. {n}" for i, n in enumerate(right, 1)),
            rejected="\n".join(f"  - {r}" for r in (rejected or [])) or "  (none)",
            max_probes=max_probes, max_expand=max_expand),
        schema=_SCHEMA, model=config.ROUTE_ADJUDICATOR_MODEL,
        max_tokens=90 * (max_probes + max_expand) + 512)
    if not payload:
        return None

    action = payload.get("action")
    if action not in ACTIONS:
        return None

    def _at(seq, idx):
        """1-based lookup. Out of range is DROPPED, never clamped -- a clamped
        index silently spends a query, or ~35 of them, on whichever entry
        happened to sit at the boundary."""
        try:
            i = int(idx)
        except (TypeError, ValueError):
            return None
        return seq[i - 1] if 1 <= i <= len(seq) else None

    pairs: List[Dict[str, str]] = []
    seen = set()
    for row in (payload.get("pairs") or []):
        a = _at(left, (row or {}).get("left"))
        b = _at(right, (row or {}).get("right"))
        ka, kb = person_norm_key(a or ""), person_norm_key(b or "")
        if not ka or not kb or ka == kb:
            continue
        if (ka, kb) in seen or (kb, ka) in seen:
            continue
        seen.add((ka, kb))
        pairs.append({"a": a, "b": b})
        if len(pairs) >= max_probes:
            break

    expand: List[str] = []
    for idx in (payload.get("expand") or []) if max_expand else []:
        who = _at(left, idx)
        if who and who not in expand:
            expand.append(who)
        if len(expand) >= max_expand:
            break

    if action == "probe" and not pairs:
        action = "none"
    if action == "expand" and not expand:
        action = "none"
    return {"action": action, "pairs": pairs, "expand": expand,
            "why": str(payload.get("why", "") or "")[:300]}
