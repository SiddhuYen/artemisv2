"""spaCy NER extractor (Tier 4) — grammar-aware entity extraction.

Replaces the capitalized-token heuristic's biggest failure mode: it understands
sentence structure, so "Following Microsoft, he…" yields ORG=Microsoft (not a
person named "Following Microsoft"), and titles/fragments aren't mistaken for
names. Emits the same ExtractionOutput contract; the confidence model is applied
identically.

Loaded lazily; if spaCy/the model isn't installed, the caller falls back to the
heuristic extractor.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from .. import config
from ..utils.names import (
    detect_org_type,
    is_noise_name,
    org_norm_key,
    person_norm_key,
)
from .confidence import (
    classify_with_signal,
    compute_confidence,
    keyword_strength_factor,
    sentence_cooccurrence,
)
from .schemas import EdgeSignals, ExtractedEdge, ExtractionOutput

MAX_ENTITIES_PER_TEXT = 25

_nlp = None
_loaded = False
_LEADING = ("the ", "The ", "a ", "an ")


def spacy_available() -> bool:
    global _nlp, _loaded
    if _loaded:
        return _nlp is not None
    _loaded = True
    if not config.SPACY_EXTRACT:
        _nlp = None
        return False
    try:
        import spacy
        # only need NER + sentence boundaries; drop nothing required for ents
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        _nlp = None
    return _nlp is not None


def _clean(text: str) -> str:
    t = text.strip()
    for lead in _LEADING:
        if t.startswith(lead):
            t = t[len(lead):]
    return t.strip(" .,'\"")


def spacy_extract(
    subject_person: str, text: str, silo, evidence: str = "", source_url: str = ""
) -> ExtractionOutput:
    out = ExtractionOutput(extractor="spacy")
    if not text or not spacy_available():
        return out

    doc = _nlp(text[: config.MAX_PAGE_CHARS])
    subj_norm = person_norm_key(subject_person)

    people: Counter = Counter()
    orgs: Counter = Counter()
    display: dict = {}
    ev: dict = {}  # norm -> evidence sentence

    for ent in doc.ents:
        name = _clean(ent.text)
        if not name or is_noise_name(name):
            continue
        if ent.label_ == "PERSON":
            norm = person_norm_key(name)
            if not norm or norm == subj_norm or len(norm.split()) < 2:
                continue  # require a full name (drops bare first/last names)
            people[norm] += 1
        elif ent.label_ == "ORG":
            norm = org_norm_key(name)
            if not norm or norm == subj_norm:
                continue
            orgs[norm] += 1
        else:
            continue
        display.setdefault(norm, name)
        if norm not in ev:
            ev[norm] = ent.sent.text.strip()[:400]

    for norm, count in people.most_common(MAX_ENTITIES_PER_TEXT):
        out.entities.people.append(display[norm])
        out.edges.append(_edge(subject_person, display[norm], "person", "unknown",
                               ev.get(norm, evidence), source_url, silo, count))
    for norm, count in orgs.most_common(MAX_ENTITIES_PER_TEXT):
        name = display[norm]
        out.entities.organizations.append(name)
        out.edges.append(_edge(subject_person, name, "organization",
                               detect_org_type(name), ev.get(norm, evidence),
                               source_url, silo, count))
    return out


def _edge(subject, name, kind, org_type, evidence_sent, source_url, silo, count):
    rel_type, explicit, _kw = classify_with_signal(evidence_sent, silo)
    factor, found = keyword_strength_factor(evidence_sent)
    cooc = sentence_cooccurrence(subject, name, evidence_sent)
    base = round(min(config.SPACY_BASE_CONFIDENCE + 0.03 * (count - 1),
                     config.SPACY_BASE_CONFIDENCE + 0.15), 3)
    adjusted = compute_confidence(base, silo.confidence_multiplier, factor, explicit, cooc)
    return ExtractedEdge(
        person_a=subject,
        person_b=name if kind == "person" else "",
        organization=name if kind == "organization" else "",
        other_kind=kind, org_type=org_type,
        relationship_type=rel_type,
        method=f"spaCy NER in silo '{silo.key}'",
        evidence_snippet=evidence_sent,
        source_url=source_url,
        confidence_base=base, confidence_adjusted=adjusted,
        signals=EdgeSignals(explicit_keyword_match=explicit,
                            sentence_cooccurrence=cooc, strength_keywords_found=found),
    )
