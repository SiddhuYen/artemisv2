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

import threading
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
_load_lock = threading.Lock()
_LEADING = ("the ", "The ", "a ", "an ")


def spacy_available() -> bool:
    """Double-checked locking: `_loaded` and `_nlp` must flip together. Without
    the lock, a concurrent caller (nodes now process in parallel — see
    expansion.py's per-hop worker pool) can observe `_loaded=True` while
    spacy.load() is still running on another thread, read `_nlp` as still
    None, and silently fall back to the weaker heuristic extractor for that
    call even though spaCy IS available. Single-threaded callers pay one
    uncontended lock acquisition; the fast path above avoids even that once
    loaded."""
    global _nlp, _loaded
    if _loaded:
        return _nlp is not None
    with _load_lock:
        if _loaded:  # a racing thread may have just finished loading
            return _nlp is not None
        if not config.SPACY_EXTRACT:
            _nlp = None
            _loaded = True
            return False
        try:
            import spacy
            # only need NER + sentence boundaries; drop nothing required for ents
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = None
        _loaded = True
    return _nlp is not None


def sentence_split(text: str) -> Optional[list]:
    """Abbreviation-aware sentence segmentation, when spaCy is available.

    Returns None (not an empty list) when spaCy isn't available, so callers
    can distinguish "no sentences" from "fall back to a cruder splitter" --
    a naive regex splitter (splitting on '. ') breaks on abbreviations like
    "U.S." or "Dr.", turning one real sentence into two fragments and
    silently pushing two co-mentioned names further apart than they really
    are in the text.
    """
    if not text or not spacy_available():
        return None
    doc = _nlp(text[: config.MAX_PAGE_CHARS])
    return [s.text.strip() for s in doc.sents if s.text.strip()]


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

    # A page's html_to_text() result (or this function's own slice below) may
    # already be cut to exactly config.MAX_PAGE_CHARS -- confirmed live: a
    # real fetched page had the subject's OWN bio section past that cutoff
    # while an unrelated person's sentence, earlier in the page, survived
    # it. `>=` (not `>`) because by the time this function sees `text`, an
    # upstream truncation to exactly the cap is indistinguishable from a
    # text that just happens to be exactly that length.
    was_truncated = len(text) >= config.MAX_PAGE_CHARS
    doc = _nlp(text[: config.MAX_PAGE_CHARS])
    subj_norm = person_norm_key(subject_person)

    # Proximity gate (see config.ENTITY_PROXIMITY_WINDOW's comment for the
    # live failure this closes): without it, EVERY entity anywhere on the
    # page becomes a "subject -> entity" edge using that entity's OWN nearby
    # sentence as evidence, with no check that the subject is mentioned
    # anywhere near it -- on a large multi-person page, that wires the
    # subject to everyone else's unrelated context. Sentences are indexed by
    # start_char (stable, hashable, cheap) rather than the Span object
    # itself.
    sent_list = list(doc.sents)
    sent_index = {s.start_char: i for i, s in enumerate(sent_list)}
    subject_lower = subject_person.lower()
    subject_sent_idx = {i for i, s in enumerate(sent_list) if subject_lower in s.text.lower()}

    def _near_subject(sent) -> bool:
        if not subject_sent_idx:
            # No mention of the subject anywhere in the (possibly truncated)
            # text -- no proximity signal to check against. Falling back to
            # "accept everything" is only safe when nothing was actually cut
            # off: a short, genuinely subject-free text (a synthetic
            # enrichment string, a pronoun-heavy paragraph) still plausibly
            # concerns the subject. A TRUNCATED text with no subject mention
            # is exactly the live failure case -- their own section may have
            # simply been cut, and accepting everything here would silently
            # reintroduce the bug this whole gate exists to close.
            return not was_truncated
        idx = sent_index.get(sent.start_char)
        if idx is None:
            return True
        return any(abs(idx - si) <= config.ENTITY_PROXIMITY_WINDOW for si in subject_sent_idx)

    people: Counter = Counter()
    orgs: Counter = Counter()
    display: dict = {}
    ev: dict = {}  # norm -> evidence sentence

    for ent in doc.ents:
        name = _clean(ent.text)
        if not name or is_noise_name(name):
            continue
        if not _near_subject(ent.sent):
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
