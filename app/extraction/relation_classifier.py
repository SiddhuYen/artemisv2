"""Claude relationship classifier.

spaCy/heuristic extraction tells us THAT two entities are connected; this turns
the weak 'unknown' edges into typed ones (coworker / cofounder / board_member …)
by asking Claude to label the relationship from the evidence sentence alone.
Batched + cached; no-op when no Anthropic key resolves.

The label matters well beyond display: an edge's type sets its strength in
pathfinding (see network/paths.REL_STRENGTH, where 'unknown' is 0.4 against
0.8–1.0 for a typed tie), so every edge left untyped distorts route ranking.

This is relationship *typing*, not the deferred Claude *verification* stage —
it never asserts a path is real, only labels the documented co-mention.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from .. import config
from ..models import RELATIONSHIP_TYPES
from ..providers import cache
from .claude_client import claude_available, call_json

# Excludes 'linkedin_1st' / 'podcast_guest': those are asserted only by their
# own structural source (a CSV row, an RSS feed episode) and must never be
# guessed onto a plain co-occurrence sentence by the classifier.
_ALLOWED = [t for t in RELATIONSHIP_TYPES if t not in ("linkedin_1st", "podcast_guest")]

_PROMPT = """Label the relationship between two entities using ONLY the evidence sentence.

Rules:
- Pick the single best type that the evidence actually supports.
- Use "unknown" if the sentence doesn't state a clear relationship.
- confidence is 0..1: how clearly the evidence supports the type you picked.
- Return one entry per item, using the item's own number.

Items:
{items}
"""

# `index` ties a verdict back to its item; the enum makes an out-of-vocabulary
# relationship type unrepresentable rather than something to sanitize later.
_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The item number this label is for.",
                    },
                    "type": {"type": "string", "enum": _ALLOWED},
                    "confidence": {
                        "type": "number",
                        "description": "0..1, how clearly the evidence supports the type.",
                    },
                },
                "required": ["index", "type", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def is_active() -> bool:
    return bool(config.CLAUDE_CLASSIFY_RELATIONS) and claude_available()


def _key(a: str, b: str, evidence: str) -> str:
    h = hashlib.sha1("{}||{}||{}".format(a, b, evidence).encode("utf-8")).hexdigest()[:16]
    # v2: v1 entries were labelled by a different model; don't inherit them.
    return cache.make_key("relclassify", "v2", h)


def _ask(items: List[dict]) -> Dict[int, dict]:
    """Verdicts for one batch, keyed by 1-based item number. Empty on failure."""
    lines = []
    for i, it in enumerate(items, 1):
        ev = (it["evidence"] or "")[:240].replace("\n", " ")
        lines.append('{}. A="{}" B="{}" evidence="{}"'.format(i, it["a"], it["b"], ev))
    payload = call_json(
        _PROMPT.format(items="\n".join(lines)),
        schema=_SCHEMA,
        model=config.CLAUDE_CLASSIFY_MODEL,
        max_tokens=64 * max(len(items), 1) + 512,
    )
    if payload is None:
        return {}
    out: Dict[int, dict] = {}
    for label in payload.get("labels", []) or []:
        if not isinstance(label, dict):
            continue
        try:
            idx = int(label["index"])
        except (KeyError, TypeError, ValueError):
            continue
        out[idx] = label
    return out


def classify(items: List[dict]) -> List[dict]:
    """items: [{a, b, evidence}] -> [{type, confidence}] aligned by index.

    No-op (all 'unknown') when inactive. Cached per (a, b, evidence).
    """
    results: List[dict] = [{"type": "unknown", "confidence": 0.0} for _ in items]
    if not items or not is_active():
        return results

    pending = []  # (orig_index, item)
    for idx, it in enumerate(items):
        cached = cache.get(_key(it["a"], it["b"], it["evidence"]), track=False)
        if cached is not None:
            results[idx] = cached
        else:
            pending.append((idx, it))

    chunks = [pending[start:start + config.CLAUDE_CLASSIFY_BATCH]
             for start in range(0, len(pending), config.CLAUDE_CLASSIFY_BATCH)]
    if not chunks:
        return results

    def _apply(chunk, verdicts: Dict[int, dict]) -> None:
        for n, (orig_idx, it) in enumerate(chunk, 1):
            v = verdicts.get(n)
            if v is None:
                # No verdict for this item (the whole call failed, or the
                # model dropped it) -- `results[orig_idx]` already defaults to
                # 'unknown' for THIS run, but don't cache it: persisting a
                # non-judgment for CACHE_TTL_WIKI would permanently downgrade a
                # real edge to 'unknown' for a month on one bad response.
                continue
            rtype = v.get("type", "unknown")
            if rtype not in _ALLOWED:
                rtype = "unknown"
            try:
                conf = float(v.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            out = {"type": rtype, "confidence": max(0.0, min(conf, 1.0))}
            results[orig_idx] = out
            cache.set(_key(it["a"], it["b"], it["evidence"]), "relclassify", out,
                      config.CACHE_TTL_WIKI)

    # See entity_filter.validate: the first chunk runs alone so a broken key
    # costs exactly one wasted call (test_auth_failure_latches_claude_off),
    # not one per chunk. Once it succeeds, the rest -- independent calls with
    # no shared state -- run concurrently instead of being awaited one at a
    # time.
    first, rest = chunks[0], chunks[1:]
    _apply(first, _ask([it for _idx, it in first]))
    if not rest:
        return results
    if not claude_available():
        return results

    with ThreadPoolExecutor(max_workers=min(4, len(rest))) as ex:
        futures = {ex.submit(_ask, [it for _idx, it in chunk]): chunk for chunk in rest}
        for future in as_completed(futures):
            chunk = futures[future]
            if not claude_available():
                continue
            _apply(chunk, future.result())
    return results
