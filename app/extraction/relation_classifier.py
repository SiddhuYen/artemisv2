"""Ollama relationship classifier.

spaCy/heuristic extraction tells us THAT two entities are connected; this turns
the weak 'unknown' edges into typed ones (coworker / cofounder / board_member …)
by asking the local Ollama model to label the relationship from the evidence
sentence alone. Batched + cached; no-op when Ollama is unavailable.

This is relationship *typing*, not the deferred Claude *verification* stage —
it never asserts a path is real, only labels the documented co-mention.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List

import httpx

from .. import config
from ..models import RELATIONSHIP_TYPES
from ..providers import cache
from .ollama_extractor import ollama_available

_ALLOWED = [t for t in RELATIONSHIP_TYPES]  # includes 'unknown'

_PROMPT = """You label the relationship between two entities using ONLY the evidence sentence.

Allowed types: {allowed}.
Rules:
- Pick the single best type that the evidence actually supports.
- Use "unknown" if the sentence doesn't state a clear relationship.
- confidence is 0..1 (how clearly the evidence supports the type).

Return ONLY JSON mapping each item number to {{"type": "...", "confidence": 0.x}}:
{{"1": {{"type": "coworker", "confidence": 0.8}}, ...}}

Items:
{items}
"""


def is_active() -> bool:
    return bool(config.OLLAMA_CLASSIFY_RELATIONS) and ollama_available()


def _key(a: str, b: str, evidence: str) -> str:
    h = hashlib.sha1(f"{a}||{b}||{evidence}".encode("utf-8")).hexdigest()[:16]
    return cache.make_key("relclassify", "v1", h)


def _ask(items: List[dict]) -> Dict[str, dict]:
    lines = []
    for i, it in enumerate(items, 1):
        ev = (it["evidence"] or "")[:240].replace("\n", " ")
        lines.append(f'{i}. A="{it["a"]}" B="{it["b"]}" evidence="{ev}"')
    prompt = _PROMPT.format(allowed=", ".join(_ALLOWED), items="\n".join(lines))
    try:
        resp = httpx.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": config.OLLAMA_FILTER_MODEL, "prompt": prompt,
                  "stream": False, "format": "json", "options": {"temperature": 0.0}},
            timeout=config.HTTP_TIMEOUT * 8,
        )
        if resp.status_code != 200:
            return {}
        raw = resp.json().get("response", "")
        data = json.loads(raw) if raw.strip().startswith("{") else _loose(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _loose(raw: str):
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


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

    for start in range(0, len(pending), config.OLLAMA_CLASSIFY_BATCH):
        chunk = pending[start:start + config.OLLAMA_CLASSIFY_BATCH]
        verdicts = _ask([it for _idx, it in chunk])
        for n, (orig_idx, it) in enumerate(chunk, 1):
            v = verdicts.get(str(n)) or {}
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
    return results
