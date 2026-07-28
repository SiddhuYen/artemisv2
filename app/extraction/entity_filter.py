"""Claude-backed entity validation filter.

Decides whether extracted candidate strings are REAL named entities (a specific
person, or a real organization) versus heuristic junk — headings, places,
products, job titles, fragments ("UW Regents", "Share Copied Bill Gates",
"First Interstate Bank" mislabeled as a person, etc.).

- Auto-enabled when an Anthropic key resolves; otherwise a transparent no-op
  (returns all names as valid, so the pipeline is unaffected).
- Batched + cached (30-day TTL) so each distinct name is judged, and paid for,
  exactly once.

This is NOT the deferred Claude relationship-verification stage — it only
validates entity-hood.
"""
from __future__ import annotations

from typing import Dict, List, Set

from .. import config
from ..providers import cache
from ..utils.names import is_noise_name, normalize
from .claude_client import claude_available, call_json

_PROMPT = """You validate named-entity extraction from messy web text.

For each candidate string below, decide if it is a REAL, specific {kind}.
{rule}
Mark false for headings, navigation text, dates, job titles, generic phrases,
fragments, or the wrong entity category.

Echo each candidate back EXACTLY as given, once, in the order listed.

Candidates:
{items}
"""

_RULES = {
    "person": "A real person is a specific human individual's name (first + last, "
              "or a well-known mononym). NOT an organization, place, or product.",
    "organization": "A real organization is a specific named company, school, "
                    "nonprofit, agency, or institution. NOT a person or a generic word.",
}

# A list of {name, valid} pairs rather than a name-keyed object: a JSON schema
# cannot both fix the key set to the batch AND set additionalProperties:false,
# and the array form is what lets the model echo the candidate verbatim.
_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The candidate string, echoed exactly as given.",
                    },
                    "valid": {
                        "type": "boolean",
                        "description": "True if this is a real, specific named entity of the requested kind.",
                    },
                },
                "required": ["name", "valid"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def _judge(names: List[str], kind: str) -> Dict[str, bool]:
    """Verdicts for one batch, keyed by the candidate string. Empty on failure."""
    items = "\n".join("- {}".format(n) for n in names)
    payload = call_json(
        _PROMPT.format(kind=kind, rule=_RULES.get(kind, ""), items=items),
        schema=_SCHEMA,
        model=config.CLAUDE_FILTER_MODEL,
        # ~30 tokens per verdict, plus headroom: a truncated response is
        # discarded wholesale by call_json, which would waste the batch.
        max_tokens=64 * max(len(names), 1) + 512,
    )
    if payload is None:
        return {}
    out: Dict[str, bool] = {}
    for verdict in payload.get("verdicts", []) or []:
        if not isinstance(verdict, dict):
            continue
        name = verdict.get("name")
        if isinstance(name, str) and name:
            out[name] = bool(verdict.get("valid"))
    return out


def validate(names: List[str], kind: str = "person") -> Set[str]:
    """Return the subset of `names` that are real named entities of `kind`.

    No-op (returns all) when filtering is disabled or no Claude key resolves.
    """
    # deterministic pre-filter: drop scraped boilerplate ("Cookie Policy",
    # "User Agreement", "... Profile") regardless of whether Claude is set up.
    uniq = [n for n in dict.fromkeys(names) if n and n.strip() and not is_noise_name(n)]
    if not uniq:
        return set()
    if not config.CLAUDE_FILTER or not claude_available():
        return set(uniq)

    valid: Set[str] = set()
    pending: List[str] = []
    for n in uniq:
        key = cache.make_key("claudefilter", kind, normalize(n))
        hit = cache.get(key, track=False)
        if hit is not None:
            if hit.get("valid"):
                valid.add(n)
        else:
            pending.append(n)

    for i in range(0, len(pending), config.CLAUDE_FILTER_BATCH):
        # Re-checked per batch, not just once up front: an auth failure latches
        # Claude off mid-run, and without this the remaining batches would each
        # re-attempt and re-fail. Whatever is left is kept (the same
        # conservative default as a missing verdict) and left uncached.
        if not claude_available():
            valid.update(pending[i:])
            break
        batch = pending[i:i + config.CLAUDE_FILTER_BATCH]
        verdicts = _judge(batch, kind)
        # match by normalized key too: a model asked to echo a name back can
        # still normalize its whitespace or punctuation, and an exact-string
        # lookup would miss the verdict and silently default to KEEP (letting
        # junk through).
        norm_verdicts = {normalize(k): v for k, v in verdicts.items()}
        for n in batch:
            v = verdicts.get(n)
            if v is None:
                v = norm_verdicts.get(normalize(n))
            if v is None:
                # No verdict at all (the call failed, or the model dropped this
                # item) -- keep for THIS run (conservative) but don't cache it:
                # it's not a real judgment, and persisting it for CACHE_TTL_WIKI
                # would silently keep junk for 30 days on one bad response.
                valid.add(n)
                continue
            if v:
                valid.add(n)
            cache.set(cache.make_key("claudefilter", kind, normalize(n)),
                      "claudefilter", {"valid": v}, config.CACHE_TTL_WIKI)
    return valid


def is_filtering_active() -> bool:
    return bool(config.CLAUDE_FILTER and claude_available())
