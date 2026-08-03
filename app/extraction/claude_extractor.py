"""Claude-backed extractor (opt-in, structured output).

Asks Claude for the named entities and subject-anchored relationships in one
scraped page. Anything that fails (no key, timeout, refusal) returns None so
the caller transparently falls back to the spaCy/heuristic extractor.

Emits the same ExtractionOutput contract as the heuristic extractor, with the
confidence model applied identically (silo multiplier × keyword strength,
evidence-rule ceilings) — the extractor decides WHAT was found, never how much
to trust it.

NOTE: structured extraction only — NOT the (deferred) Claude path-verification
stage, which judges whether a whole route is real.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from .. import config
from ..utils.names import detect_org_type, normalize
from .claude_client import claude_available, call_json
from .confidence import (
    classify_with_signal,
    compute_confidence,
    keyword_strength_factor,
    sentence_cooccurrence,
)
from .schemas import EdgeSignals, ExtractedEdge, ExtractionOutput

_PROMPT_TEMPLATE = """Extract the named entities and relationships in the text below.

Rules:
- Only extract NAMED entities explicitly present in the text.
- Only describe relationships to the subject person: "{subject}".
- For each relationship, quote the span of the text that states it as `evidence`.
- Do NOT guess or infer relationships that are not stated in the text.
- If you are unsure about an entity or a relationship, omit it.

TEXT:
\"\"\"
{text}
\"\"\"
"""

# Structured-output schema. Every object needs `additionalProperties: false`
# and an explicit `required` list, so a successful call cannot come back in a
# shape the loop below has to defend against.
_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Full names of specific people named in the text.",
        },
        "organizations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of specific organizations named in the text.",
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "other": {
                        "type": "string",
                        "description": "The person or organization the subject is connected to.",
                    },
                    "kind": {"type": "string", "enum": ["person", "organization"]},
                    "evidence": {
                        "type": "string",
                        "description": "Short quote from the text stating the relationship.",
                    },
                },
                "required": ["other", "kind", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["people", "organizations", "relationships"],
    "additionalProperties": False,
}


# Bump on ANY change to _PROMPT_TEMPLATE or _SCHEMA. A verdict cached under an
# older prompt is indistinguishable from a current one and would be replayed for
# the whole TTL -- the same trap config.NODE_PROFILE_VERSION exists for, solved
# the same way relation_classifier's own "v2" key segment solves it.
_CACHE_VERSION = "v1"


def _verdict_key(subject: str, body: str, model: str) -> str:
    from ..providers import cache

    h = hashlib.sha1(
        "{}||{}||{}".format(model, subject, body).encode("utf-8")
    ).hexdigest()[:24]
    return cache.make_key("claudeextract", _CACHE_VERSION, h)


def _extract_verdict(subject_person: str, body: str) -> Optional[dict]:
    """The model's raw verdict for one (subject, page) pair. Cached; None on failure.

    Split out from the edge-building below because the prompt depends ONLY on
    the subject and the page text: `silo`, `evidence` and `source_url` never
    reach the model (see _PROMPT_TEMPLATE), they only shape the edges built
    from a verdict afterwards.

    That distinction is what makes caching here safe AND worth doing.
    expansion._process_person's phase 4 iterates (query, result, silo) tuples,
    not unique pages -- page FETCHES are deduped through a set, extractions are
    not -- so one URL returned by five of the ~35 silo queries used to be five
    full-page requests carrying a byte-identical prompt and getting back a
    byte-identical answer, which _dedup_and_cap then collapsed anyway by
    (counterpart, type, source_url). Every caller still runs its own
    post-processing against the verdict; only the request is shared.

    The model is part of the key: CLAUDE_EXTRACT_MODEL is a knob (a cheaper
    model is the obvious lever once this stage is the budget), and verdicts
    from the previous one must not be served after it changes.
    """
    # Imported inside the function, not at module scope, because
    # extraction/__init__ imports THIS module at package load while
    # app.providers pulls in the whole provider stack (bs4, httpx, the
    # orchestrator). Keeping it local means importing the extraction layer
    # still costs nothing it didn't already -- same reason extract() takes
    # `from .. import config` locally.
    from ..providers import cache

    key = _verdict_key(subject_person, body, config.CLAUDE_EXTRACT_MODEL)
    hit = cache.get(key, track=False)
    if hit is not None:
        return hit
    verdict = call_json(
        _PROMPT_TEMPLATE.format(subject=subject_person, text=body),
        schema=_SCHEMA,
        model=config.CLAUDE_EXTRACT_MODEL,
        max_tokens=8192,
    )
    # Never cache a failure. call_json returns None for a timeout, a rate limit
    # past the SDK's retries, a refusal or a truncated response, and the caller
    # treats that as "fall back to spaCy for this page". Persisting a
    # non-verdict would pin the page to the deterministic extractor for the
    # whole TTL on one bad response -- the same rule entity_filter and
    # relation_classifier apply to their own missing verdicts.
    if verdict is not None:
        cache.set(key, "claudeextract", verdict, config.CACHE_TTL_PAGE)
    return verdict


def claude_extract(
    subject_person: str, text: str, silo, evidence: str = "", source_url: str = ""
) -> Optional[ExtractionOutput]:
    if not text:
        return ExtractionOutput(extractor="claude")
    payload = _extract_verdict(subject_person, text[: config.MAX_PAGE_CHARS])
    if payload is None:
        return None

    out = ExtractionOutput(extractor="claude")
    subj_norm = normalize(subject_person)

    for name in payload.get("people", []) or []:
        if isinstance(name, str) and name.strip():
            out.entities.people.append(name.strip())
    for name in payload.get("organizations", []) or []:
        if isinstance(name, str) and name.strip():
            out.entities.organizations.append(name.strip())

    for rel in payload.get("relationships", []) or []:
        if not isinstance(rel, dict):
            continue
        other = (rel.get("other") or "").strip()
        if not other:
            out.add_rejected("relationship missing named counterpart", str(rel))
            continue
        if normalize(other) == subj_norm:
            continue
        kind = rel.get("kind", "person")
        kind = kind if kind in ("person", "organization") else "person"
        ev = (rel.get("evidence") or evidence)[:400]

        rel_type, explicit, _kw = classify_with_signal(ev or text, silo)
        factor, found = keyword_strength_factor(ev or text)
        cooc = sentence_cooccurrence(subject_person, other, ev)
        base = config.CLAUDE_BASE_CONFIDENCE
        adjusted = compute_confidence(base, silo.confidence_multiplier, factor, explicit, cooc)

        out.edges.append(
            ExtractedEdge(
                person_a=subject_person,
                person_b=other if kind == "person" else "",
                organization=other if kind == "organization" else "",
                other_kind=kind,
                org_type=detect_org_type(other) if kind == "organization" else "unknown",
                relationship_type=rel_type,
                method=f"claude extraction ({config.CLAUDE_EXTRACT_MODEL}) in silo '{silo.key}'",
                evidence_snippet=ev,
                source_url=source_url,
                confidence_base=base,
                confidence_adjusted=adjusted,
                signals=EdgeSignals(
                    explicit_keyword_match=explicit,
                    sentence_cooccurrence=cooc,
                    strength_keywords_found=found,
                ),
            )
        )

    return out


__all__ = ["claude_available", "claude_extract"]
