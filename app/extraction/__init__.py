"""Extraction layer: Claude when a key is configured, heuristic fallback otherwise.

Every extractor returns the same hardened ExtractionOutput contract.
"""
from __future__ import annotations

from .confidence import (
    classify_with_signal,
    compute_confidence,
    keyword_strength_factor,
    tier,
)
from .claude_extractor import claude_available, claude_extract
from .heuristic import heuristic_extract
from .spacy_extractor import spacy_available, spacy_extract
from .schemas import (
    EdgeSignals,
    Entities,
    ExtractedEdge,
    ExtractionOutput,
    RejectedItem,
)


def extract(
    subject_person: str, text: str, silo, evidence: str = "", source_url: str = ""
) -> ExtractionOutput:
    """Run the best available extractor for one (subject, text, silo) unit.

    Precedence: Claude per-source extraction (opt-in, costly, cleanest) ->
    spaCy NER (grammar-aware, default when installed) -> capitalized-token
    heuristic (last-resort fallback).
    """
    from .. import config
    if config.CLAUDE_EXTRACT and claude_available():
        result = claude_extract(subject_person, text, silo, evidence, source_url)
        if result is not None:
            return result
    if spacy_available():
        return spacy_extract(subject_person, text, silo, evidence, source_url)
    return heuristic_extract(subject_person, text, silo, evidence, source_url)


__all__ = [
    "extract",
    "ExtractionOutput",
    "ExtractedEdge",
    "EdgeSignals",
    "Entities",
    "RejectedItem",
    "classify_with_signal",
    "compute_confidence",
    "keyword_strength_factor",
    "tier",
    "heuristic_extract",
    "claude_extract",
    "claude_available",
    "spacy_extract",
    "spacy_available",
]
