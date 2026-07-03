"""Extraction layer: Ollama when available, heuristic fallback otherwise.

Both extractors return the same hardened ExtractionOutput contract.
"""
from __future__ import annotations

from .confidence import (
    classify_with_signal,
    compute_confidence,
    keyword_strength_factor,
    tier,
)
from .heuristic import heuristic_extract
from .ollama_extractor import ollama_available, ollama_extract
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

    Precedence: Ollama per-source extraction (opt-in, slow, cleanest) ->
    spaCy NER (grammar-aware, default when installed) -> capitalized-token
    heuristic (last-resort fallback).
    """
    from .. import config
    if config.OLLAMA_EXTRACT and ollama_available():
        result = ollama_extract(subject_person, text, silo, evidence, source_url)
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
    "ollama_extract",
    "ollama_available",
    "spacy_extract",
    "spacy_available",
]
