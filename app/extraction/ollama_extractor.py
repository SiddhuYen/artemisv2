"""Optional Ollama-backed extractor (hardened, structured output).

If a local Ollama daemon is reachable, ask it for STRICT JSON relationships.
Anything that fails (daemon down, bad JSON, timeout) returns None so the
caller transparently falls back to the heuristic extractor.

Emits the same ExtractionOutput contract as the heuristic extractor, with the
confidence model applied identically (silo multiplier × keyword strength,
evidence-rule ceilings).

NOTE: structured extraction only — NOT the (deferred) Claude verification stage.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from .. import config
from ..utils.names import normalize
from .confidence import (
    classify_with_signal,
    compute_confidence,
    keyword_strength_factor,
    sentence_cooccurrence,
)
from .schemas import EdgeSignals, ExtractedEdge, ExtractionOutput

_PROMPT_TEMPLATE = """You extract structured relationships from text.

Return ONLY valid JSON of the form:
{{
  "people": ["Full Name", ...],
  "organizations": ["Org Name", ...],
  "relationships": [
    {{"other": "Name or Org", "kind": "person|organization", "evidence": "short quote"}}
  ]
}}

Rules:
- Only extract NAMED entities explicitly present in the text.
- Only describe relationships to the subject person: "{subject}".
- Do NOT guess or infer relationships not stated in the text.
- No hallucination. If unsure, omit it.

TEXT:
\"\"\"
{text}
\"\"\"
"""

_availability_cache: Optional[bool] = None


def ollama_available() -> bool:
    global _availability_cache
    if _availability_cache is not None:
        return _availability_cache
    try:
        resp = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=2.0)
        _availability_cache = resp.status_code == 200
    except Exception:
        _availability_cache = False
    return _availability_cache


def _extract_json(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def ollama_extract(
    subject_person: str, text: str, silo, evidence: str = "", source_url: str = ""
) -> Optional[ExtractionOutput]:
    if not text:
        return ExtractionOutput(extractor="ollama")
    prompt = _PROMPT_TEMPLATE.format(subject=subject_person, text=text[: config.MAX_PAGE_CHARS])
    try:
        resp = httpx.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0}},
            timeout=config.HTTP_TIMEOUT * 4,
        )
        if resp.status_code != 200:
            return None
        payload = _extract_json(resp.json().get("response", ""))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    out = ExtractionOutput(extractor="ollama")
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
            out.add_rejected("relationship missing named counterpart", json.dumps(rel))
            continue
        if normalize(other) == subj_norm:
            continue
        kind = rel.get("kind", "person")
        kind = kind if kind in ("person", "organization") else "person"
        ev = (rel.get("evidence") or evidence)[:400]

        rel_type, explicit, _kw = classify_with_signal(ev or text, silo)
        factor, found = keyword_strength_factor(ev or text)
        cooc = sentence_cooccurrence(subject_person, other, ev)
        base = config.OLLAMA_BASE_CONFIDENCE
        adjusted = compute_confidence(base, silo.confidence_multiplier, factor, explicit, cooc)

        out.edges.append(
            ExtractedEdge(
                person_a=subject_person,
                person_b=other if kind == "person" else "",
                organization=other if kind == "organization" else "",
                other_kind=kind,
                relationship_type=rel_type,
                method=f"ollama extraction ({config.OLLAMA_MODEL}) in silo '{silo.key}'",
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
