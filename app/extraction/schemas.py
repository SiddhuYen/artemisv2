"""Structured, strictly-typed extraction output contract (pydantic).

This is the hardened extractor output. Every edge is source-grounded
(evidence_snippet + source_url) and carries its confidence derivation and the
signals that justify it. Items that fail the evidence/entity rules are not
silently dropped — they go to `rejected` with a reason.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class EdgeSignals(BaseModel):
    explicit_keyword_match: bool = False
    sentence_cooccurrence: bool = False
    strength_keywords_found: List[str] = Field(default_factory=list)
    # the counterpart came from a STRUCTURED source (Wikidata/EDGAR/ProPublica/
    # OpenAlex) with a clean canonical name -> skip the Ollama entity filter.
    trusted: bool = False


class ExtractedEdge(BaseModel):
    person_a: str                      # subject (always a person)
    person_b: str = ""                 # set when the counterpart is a person
    organization: str = ""             # set when the counterpart is an org
    relationship_type: str = "unknown"
    method: str = ""
    evidence_snippet: str = ""
    source_url: str = ""
    confidence_base: float = 0.0
    confidence_adjusted: float = 0.0
    signals: EdgeSignals = Field(default_factory=EdgeSignals)

    # routing metadata (not part of the public contract but handy downstream)
    other_kind: str = "person"         # "person" | "organization"
    org_type: str = "unknown"

    @property
    def counterpart(self) -> str:
        return self.person_b or self.organization


class RejectedItem(BaseModel):
    reason: str
    text: str


class Entities(BaseModel):
    people: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)


class ExtractionOutput(BaseModel):
    entities: Entities = Field(default_factory=Entities)
    edges: List[ExtractedEdge] = Field(default_factory=list)
    rejected: List[RejectedItem] = Field(default_factory=list)
    extractor: str = "heuristic"

    def add_rejected(self, reason: str, text: str) -> None:
        self.rejected.append(RejectedItem(reason=reason, text=text[:200]))
