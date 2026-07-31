"""Claude search-strategy call -- Alpha step 6 ("run reasoning to identify
best type of search").

Given what node_profiler (step 4/5) established about the subject's own org
-- size, industry -- and who the walk is ultimately trying to reach, decide
which angle of the subject's professional network is most likely to bridge
toward the target. Concrete motivating case: Prantik Chakraborty (VP Sales
at a mid-sized Oracle consulting shop) walking toward Larry Ellison (Oracle)
-- a human would reason "he works in Oracle's own partner ecosystem, so
check who at his company deals directly with Oracle leadership" rather than
firing the same generic silo templates used for every subject regardless of
context.

Constrained to a FIXED enum of angles, not free-form text -- the model picks
WHICH pre-written query angle applies, it never writes query text itself.
That keeps the actual search surface fully deterministic and inspectable
(see config.STRATEGY_ANGLE_QUERIES), so a bad strategy CALL can only ever
pick the wrong known option, not invent an ungrounded direction -- the same
containment principle node_profiler applies to org identity, just one level
up.

Gated (by the caller, expansion.py) on node_profiler having already produced
a GROUNDED profile -- deciding a strategy from an ungrounded/absent profile
would just be reasoning on top of a guess, compounding exactly the risk
node_profiler's own grounding check exists to contain.
"""
from __future__ import annotations

from typing import Optional

from .. import config
from .claude_client import call_json, claude_available

_PROMPT = """A person is trying to reach a target through their professional \
network. Decide which ONE search angle is most likely to surface the \
strongest bridge -- using ONLY the facts given below, not general assumptions \
about how careers usually work.

Subject: {subject}
Subject's organization: {org} -- industry: {industry}, size: {size_tier}
({org_summary})

Target: {target}{target_context_line}

Angles (pick exactly one):
- current_employer_leadership: the subject's own company's leadership/exec \
team is the most likely bridge (subject's org plausibly has direct dealings \
with the target's world).
- past_employers: a previous employer is more likely to bridge than the \
subject's current one.
- industry_peers: the subject's industry has a small, identifiable circle of \
senior people, worth searching directly rather than through any one employer.
- board_or_advisory: the subject's board/advisory ties are the more promising \
angle than their day-to-day employer.
- generic: none of the above is clearly better than the existing broad silo \
search -- don't force an angle that isn't actually supported by the facts \
given.

why: one sentence, grounded only in the facts above.
"""

_TARGET_CONTEXT_LINE = " (context: {target_context})"

_ANGLES = ["current_employer_leadership", "past_employers", "industry_peers",
           "board_or_advisory", "generic"]

_SCHEMA = {
    "type": "object",
    "properties": {
        "angle": {"type": "string", "enum": _ANGLES},
        "why": {"type": "string"},
    },
    "required": ["angle", "why"],
    "additionalProperties": False,
}


def is_active() -> bool:
    return bool(config.STRATEGY_ENABLED) and claude_available()


def decide_angle(subject_name: str, org_name: str, profile: dict,
                 target_name: str, target_context: str = "") -> Optional[dict]:
    """{angle, why} or None (inactive, or the call failed).

    `profile` is node_profiler's output for `org_name` -- callers should only
    invoke this with a `grounded` profile; an ungrounded one has nothing real
    to reason from and this module doesn't re-check that itself (the
    grounding gate belongs to whoever produced the profile, not to every
    downstream reader of it).
    """
    if not is_active():
        return None
    target_line = _TARGET_CONTEXT_LINE.format(target_context=target_context) \
        if target_context else ""
    prompt = _PROMPT.format(
        subject=subject_name, org=org_name,
        industry=profile.get("industry", "unknown"),
        size_tier=profile.get("size_tier", "unknown"),
        org_summary=profile.get("summary", ""),
        target=target_name, target_context_line=target_line,
    )
    payload = call_json(prompt, schema=_SCHEMA, model=config.STRATEGY_MODEL, max_tokens=256)
    if payload is None:
        return None
    angle = payload.get("angle", "generic")
    if angle not in _ANGLES:
        angle = "generic"
    return {"angle": angle, "why": str(payload.get("why", "") or "")[:300]}
