"""Claude coauthor-plausibility gate -- runs BEFORE the OpenAlex call, not
after it.

Phase 4b's existing domain_conflict check (see graph.expansion, graph.
disambiguate) catches a homonym AFTER OpenAlex has already resolved a bare
name and returned a coauthor list -- useful, but it only fires once we've
already paid for the network round trip and already run the risk of a
merge slipping past the lexicon (see the live "Donald Trump" mixup this
closes a layer earlier). This module asks a cheaper, prior question using
data already in hand from phases 0-4: given what's ALREADY known about this
subject, would they plausibly have real academic publications with
coauthors at all? A sales executive, a software engineer at a tech company,
a professional athlete -- none of these people are likely to show up in
OpenAlex as themselves, so there's no reason to even attempt the lookup.

Conservative by design, same philosophy as disambiguate.domain_conflict:
silent (proceeds with OpenAlex) whenever the evidence is too thin or
ambiguous to judge either way. This is a cost/risk-reduction layer, not a
safety gate -- Claude being unavailable, or genuinely unsure, must never
block a search that would otherwise have been fine; only a CLEAR signal
against academic authorship skips the call.
"""
from __future__ import annotations

from typing import Optional

from .. import config
from .claude_client import call_json, claude_available

_PROMPT = """Based ONLY on the evidence below, is it plausible this person has \
published academic research with named coauthors?

Consider: some professions routinely produce peer-reviewed, co-authored \
publications (research scientists, professors, clinician-researchers, PhD \
students). Most don't (sales/business executives, engineers at tech \
companies, athletes, politicians, artists) -- though a mid-career \
professional occasionally publishing outside their main role isn't rare \
either, so don't rule someone out just because their day job isn't research.

If the evidence is too thin, vague, or gives no real signal either way, \
answer plausible=true -- only say false when the evidence clearly points \
away from academic authorship. A missed check here just means the existing \
identity check downstream still has to catch a bad match; a wrongly-skipped \
search loses real data outright.

Subject: {subject}
Known context: {context}
Evidence gathered so far: {signal}

why: one sentence, grounded only in the evidence above.
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "plausible": {"type": "boolean"},
        "why": {"type": "string"},
    },
    "required": ["plausible", "why"],
    "additionalProperties": False,
}


def is_active() -> bool:
    return bool(config.COAUTHOR_PLAUSIBILITY_ENABLED) and claude_available()


def check(subject_name: str, context: str, signal: str) -> Optional[dict]:
    """{plausible, why}, or None when inactive / nothing to judge / the call
    failed -- callers must treat None as "proceed with OpenAlex as before,"
    the same fail-open contract as every other stage through
    claude_client.call_json. Only an explicit plausible=False should skip
    the coauthor lookup.
    """
    if not is_active():
        return None
    combined = f"{context} {signal}".strip()
    if not combined:
        # Nothing at all to judge from -- a brand-new node with zero prior
        # evidence. Same "no signal -> fail open" default the rest of the
        # homonym-guard machinery already uses.
        return None
    payload = call_json(
        _PROMPT.format(subject=subject_name, context=context or "(none given)",
                       signal=signal or "(none gathered yet)"),
        schema=_SCHEMA, model=config.COAUTHOR_PLAUSIBILITY_MODEL, max_tokens=200,
    )
    if payload is None:
        return None
    return {
        "plausible": bool(payload.get("plausible", True)),
        "why": str(payload.get("why", "") or "")[:300],
    }
