"""Ollama-backed entity validation filter.

Decides whether extracted candidate strings are REAL named entities (a specific
person, or a real organization) versus heuristic junk — headings, places,
products, job titles, fragments ("UW Regents", "Share Copied Bill Gates",
"First Interstate Bank" mislabeled as a person, etc.).

- Auto-enabled when Ollama is reachable; otherwise a transparent no-op (returns
  all names as valid, so the pipeline is unaffected).
- Batched + cached (30-day TTL) so each distinct name is judged once.

This is NOT the deferred Claude relationship-verification stage — it only
validates entity-hood, using the local Ollama model already in the stack.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Set

import httpx

from .. import config
from ..providers import cache
from ..utils.names import is_noise_name, normalize
from .ollama_extractor import ollama_available

_PROMPT = """You validate named-entity extraction from messy web text.

For each candidate string below, decide if it is a REAL, specific {kind}.
{rule}
Mark false for headings, navigation text, dates, job titles, generic phrases,
fragments, or the wrong entity category.

Return ONLY a JSON object mapping each candidate EXACTLY as given to true/false:
{{"Candidate A": true, "Candidate B": false}}

Candidates:
{items}
"""

_RULES = {
    "person": "A real person is a specific human individual's name (first + last, "
              "or a well-known mononym). NOT an organization, place, or product.",
    "organization": "A real organization is a specific named company, school, "
                    "nonprofit, agency, or institution. NOT a person or a generic word.",
}


def _judge(names: List[str], kind: str) -> Dict[str, bool]:
    items = "\n".join(f"- {n}" for n in names)
    prompt = _PROMPT.format(kind=kind, rule=_RULES.get(kind, ""), items=items)
    try:
        resp = httpx.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": config.OLLAMA_FILTER_MODEL, "prompt": prompt,
                  "stream": False, "format": "json", "options": {"temperature": 0.0}},
            timeout=config.HTTP_TIMEOUT * 6,
        )
        if resp.status_code != 200:
            return {}
        raw = resp.json().get("response", "")
        data = json.loads(raw) if raw.strip().startswith("{") else _loose_json(raw)
    except Exception:
        return {}
    out: Dict[str, bool] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            out[k] = bool(v) if isinstance(v, bool) else str(v).lower() in ("true", "yes", "1")
    return out


def _loose_json(raw: str):
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def validate(names: List[str], kind: str = "person") -> Set[str]:
    """Return the subset of `names` that are real named entities of `kind`.

    No-op (returns all) when filtering is disabled or Ollama is unreachable.
    """
    # deterministic pre-filter: drop scraped boilerplate ("Cookie Policy",
    # "User Agreement", "... Profile") regardless of whether Ollama is available.
    uniq = [n for n in dict.fromkeys(names) if n and n.strip() and not is_noise_name(n)]
    if not uniq:
        return set()
    if not config.OLLAMA_FILTER or not ollama_available():
        return set(uniq)

    valid: Set[str] = set()
    pending: List[str] = []
    for n in uniq:
        key = cache.make_key("ollamafilter", kind, normalize(n))
        hit = cache.get(key, track=False)
        if hit is not None:
            if hit.get("valid"):
                valid.add(n)
        else:
            pending.append(n)

    for i in range(0, len(pending), config.OLLAMA_FILTER_BATCH):
        batch = pending[i:i + config.OLLAMA_FILTER_BATCH]
        verdicts = _judge(batch, kind)
        # match by normalized key too: a small local model often reformats the
        # name it echoes back, so an exact-string lookup would miss the verdict
        # and silently default to KEEP (letting junk through).
        norm_verdicts = {normalize(k): v for k, v in verdicts.items()}
        for n in batch:
            v = verdicts.get(n)
            if v is None:
                v = norm_verdicts.get(normalize(n))
            # default to KEEP only when truly absent (conservative)
            keep = True if v is None else v
            if keep:
                valid.add(n)
            cache.set(cache.make_key("ollamafilter", kind, normalize(n)),
                      "ollamafilter", {"valid": keep}, config.CACHE_TTL_WIKI)
    return valid


def is_filtering_active() -> bool:
    return bool(config.OLLAMA_FILTER and ollama_available())
