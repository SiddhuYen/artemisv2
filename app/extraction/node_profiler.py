"""Claude org-profiling call -- Alpha step 4/5 ("understand current node").

Answers "how big is this org, what industry is it in" from a handful of
fetched snippets. This is deliberately the most tightly-leashed stage in
Alpha: unlike relation_classifier (one evidence sentence, one label from a
fixed enum), this asks for an open-ended judgment (size tier, industry) that
an LLM can easily fill from its own training-data priors about "what a
company like this is probably like" when the fetched snippets actually say
nothing concrete -- exactly the shape of failure this whole project has
spent this session chasing out of the deterministic extractors.

Two guards against that, both enforced in code, not just the prompt:
  - the model must set `grounded: false` if it inferred ANYTHING beyond what
    the snippets literally state, and
  - an ungrounded response is downgraded to "unknown" here regardless of
    what size_tier/industry the model still filled in -- a plausible-
    sounding guess the model itself flagged as unsupported must not survive
    to look like a fact downstream.

Cached on the Organization row's own `meta` column (no TTL, same convention
as `openalex_rejected` elsewhere) -- once profiled, an org's profile is
reused across every person who works there, this run and future ones.

A THIRD guard, added after live verification surfaced a real hit: a search
for "Trinamix" also pulled in "trinamiX GmbH", an unrelated German
biometric-sensor company sharing a near-identical name -- an org-level
homonym, the same bug class phase 4b's OpenAlex identity gate exists to
catch at the person level. `known_context` (the evidence snippet from the
subject's own affiliation edge, e.g. "...Vice President Sales & Strategy at
Trinamix Inc...") gives the model something concrete to check each fetched
snippet against, and instructs it to silently drop any snippet describing a
different organization rather than blend it into the profile.
"""
from __future__ import annotations

from typing import List, Optional

from .. import config
from .claude_client import call_json, claude_available

_IDENTITY_BLOCK = """
IMPORTANT -- identity check: the snippets below come from a web search for \
the organization's name, and a search can pull in an UNRELATED organization \
that merely shares a similar name (this has happened live: a search for \
"Trinamix" also returned "trinamiX GmbH", an unrelated German sensor-\
hardware company). Known context about the REAL organization, from the \
subject's own profile: "{known_context}"

Before using ANY snippet, check whether it plausibly describes the SAME \
organization as that known context (same industry / line of business). If a \
snippet clearly describes something else, ignore it completely -- do not let \
it influence size_tier, industry, or summary, and do not count it toward \
`grounded`.
"""

_PROMPT = """Describe the size and industry of this organization using ONLY \
the snippets below. Do not use anything you already know about this \
organization from training -- if the snippets don't say it, you don't know it.
{identity_block}
Rules:
- size_tier: "small", "mid", or "large" ONLY if a snippet (that passed the \
identity check, if one was given) states or clearly implies employee count, \
revenue, or scale. Otherwise "unknown".
- industry: a short phrase (e.g. "Oracle ERP consulting") ONLY if a snippet \
that passed the identity check states what the organization actually does. \
Otherwise "unknown".
- summary: one sentence, grounded only in identity-checked snippets. If \
nothing useful survives, say so plainly instead of guessing.
- grounded: false if you had to infer or estimate ANYTHING beyond what the \
snippets literally state, OR if no snippet passed the identity check. true \
only if every field above is directly supported by identity-checked text.

Organization: {org}

Snippets:
{snippets}
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "size_tier": {"type": "string", "enum": ["small", "mid", "large", "unknown"]},
        "industry": {"type": "string"},
        "summary": {"type": "string"},
        "grounded": {"type": "boolean"},
    },
    "required": ["size_tier", "industry", "summary", "grounded"],
    "additionalProperties": False,
}


def is_active() -> bool:
    return bool(config.NODE_PROFILE_ENABLED) and claude_available()


def profile_org(org_name: str, snippets: List[str], known_context: str = "") -> Optional[dict]:
    """{size_tier, industry, summary, grounded} from `snippets`, or None.

    `known_context` is evidence already on hand about the REAL organization
    (e.g. the subject's own affiliation edge: "...Vice President Sales &
    Strategy at Trinamix Inc..."), used to catch a search pulling in an
    unrelated same-named organization -- see the module docstring for the
    live case (Trinamix / trinamiX GmbH) this guards against. Without it,
    every fetched snippet is trusted equally, which is exactly what let
    that mixup through.

    None means "no call worth making, or the call failed" -- same failure
    contract as everything through claude_client.call_json: callers treat it
    as "couldn't profile this run," not as a verdict of any kind.
    """
    if not snippets or not is_active():
        return None
    text = "\n\n".join(
        f"[{i + 1}] {s[:config.NODE_PROFILE_SNIPPET_CHARS]}"
        for i, s in enumerate(snippets[:config.NODE_PROFILE_MAX_SNIPPETS])
    )
    identity_block = _IDENTITY_BLOCK.format(known_context=known_context) if known_context else ""
    payload = call_json(
        _PROMPT.format(org=org_name, snippets=text, identity_block=identity_block),
        schema=_SCHEMA,
        model=config.NODE_PROFILE_MODEL,
        max_tokens=512,
    )
    if payload is None:
        return None

    grounded = bool(payload.get("grounded", False))
    size_tier = payload.get("size_tier", "unknown")
    if size_tier not in ("small", "mid", "large", "unknown"):
        size_tier = "unknown"
    industry = str(payload.get("industry", "") or "").strip() or "unknown"

    if not grounded:
        # The model itself said this wasn't actually supported by the text --
        # a guess admitting it's a guess is still a guess. Don't let it
        # through looking like a fact just because it filled the fields in.
        size_tier = "unknown"
        industry = "unknown"

    return {
        "size_tier": size_tier,
        "industry": industry,
        "summary": str(payload.get("summary", "") or "")[:400],
        "grounded": grounded,
    }
