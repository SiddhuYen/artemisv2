"""Heuristic (no-LLM) extractor — hardened.

Deliberately conservative and LOW confidence. It surfaces named-entity
candidates from text without inferring anything it can't see:
  - capitalised multi-word tokens -> candidate people,
  - tokens with org suffixes (Inc, LLC, University, Foundation...) -> orgs.

Every edge it emits is source-grounded (evidence sentence + source URL) and
carries a fully-derived confidence plus the signals that justify it. Candidates
that fail the entity rule are recorded in `rejected`, not silently dropped.
"""
from __future__ import annotations

import re
from collections import Counter

from .. import config
from ..utils.names import (
    ORG_SUFFIXES,
    detect_org_type,
    is_noise_name,
    looks_like_org_name,
    looks_like_person_name,
    normalize,
    person_norm_key,
)
from .confidence import (
    classify_with_signal,
    compute_confidence,
    keyword_strength_factor,
    sentence_cooccurrence,
)
from .schemas import EdgeSignals, ExtractedEdge, ExtractionOutput

# capitalised run of 1..5 tokens (allowing &, ., -)
_CANDIDATE = re.compile(r"\b[A-Z][A-Za-z.&\-]*(?:\s+[A-Z][A-Za-z.&\-]*){0,4}\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

MAX_ENTITIES_PER_TEXT = 15
MAX_REJECTED_PER_TEXT = 20

_TRIM_TOKENS = {"he", "she", "they", "it", "we", "i", "his", "her", "their",
                "this", "that", "these", "those", "and", "the", "a", "an"}


def _truncate_org(phrase: str) -> str:
    parts = phrase.split()
    last = -1
    for i, p in enumerate(parts):
        if normalize(p) in ORG_SUFFIXES:
            last = i
    return " ".join(parts[: last + 1]) if last >= 0 else phrase


def _trim_person(phrase: str) -> str:
    parts = phrase.split()
    while parts and normalize(parts[0]) in _TRIM_TOKENS:
        parts.pop(0)
    while parts and normalize(parts[-1]) in _TRIM_TOKENS:
        parts.pop()
    return " ".join(parts)


def _evidence_for(name: str, text: str, fallback: str) -> str:
    for sent in _SENT_SPLIT.split(text):
        if name.lower() in sent.lower():
            return sent.strip()[:400]
    return fallback[:400]


def _base_score(count: int) -> float:
    base = config.HEURISTIC_BASE_CONFIDENCE
    return round(min(base + 0.03 * (count - 1), base + 0.15), 3)


def heuristic_extract(
    subject_person: str, text: str, silo, evidence: str = "", source_url: str = ""
) -> ExtractionOutput:
    out = ExtractionOutput(extractor="heuristic")
    if not text:
        return out

    subj_norm = person_norm_key(subject_person)
    person_counts: Counter = Counter()
    org_counts: Counter = Counter()
    display: dict = {}  # norm -> original display form

    for sentence in _SENT_SPLIT.split(text):
        for match in _CANDIDATE.finditer(sentence):
            raw_phrase = match.group(0).strip(" .,-&")
            if is_noise_name(raw_phrase):
                continue  # scraped boilerplate ("Cookie Policy", "... Profile")
            if looks_like_org_name(raw_phrase):
                phrase = _truncate_org(raw_phrase)
                norm = normalize(phrase)
                if not norm or norm == subj_norm:
                    continue
                org_counts[norm] += 1
                display.setdefault(norm, phrase)
            else:
                phrase = _trim_person(raw_phrase)
                norm = person_norm_key(phrase)
                if not norm or norm == subj_norm:
                    continue
                if looks_like_person_name(phrase):
                    person_counts[norm] += 1
                    display.setdefault(norm, phrase)
                elif len(phrase.split()) >= 2 and len(out.rejected) < MAX_REJECTED_PER_TEXT:
                    # plausible-but-rejected: violates the named-person entity rule
                    out.add_rejected("failed named-person entity rule", phrase)

    for norm, count in person_counts.most_common(MAX_ENTITIES_PER_TEXT):
        name = display[norm]
        out.entities.people.append(name)
        out.edges.append(
            _build_edge(subject_person, name, "person", "unknown",
                        text, evidence, source_url, silo, count)
        )

    for norm, count in org_counts.most_common(MAX_ENTITIES_PER_TEXT):
        name = display[norm]
        otype = detect_org_type(name)
        out.entities.organizations.append(name)
        out.edges.append(
            _build_edge(subject_person, name, "organization", otype,
                        text, evidence, source_url, silo, count)
        )

    return out


def _build_edge(subject, name, kind, org_type, text, evidence, source_url, silo, count):
    ev = _evidence_for(name, text, evidence)
    rel_type, explicit, _kw = classify_with_signal(ev, silo)
    factor, found = keyword_strength_factor(ev)
    cooc = sentence_cooccurrence(subject, name, ev)
    base = _base_score(count)
    adjusted = compute_confidence(base, silo.confidence_multiplier, factor, explicit, cooc)
    return ExtractedEdge(
        person_a=subject,
        person_b=name if kind == "person" else "",
        organization=name if kind == "organization" else "",
        other_kind=kind,
        org_type=org_type,
        relationship_type=rel_type,
        method=f"heuristic {kind} match in silo '{silo.key}'",
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
