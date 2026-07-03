"""ORM models for the relationship graph.

Enum-like columns are stored as TEXT for SQLite friendliness; the allowed
values are documented by the constant tuples below and validated in the
schema / builder layers.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- controlled vocabularies ----------------------------------------------
ORG_TYPES = ("company", "nonprofit", "school", "government", "event", "unknown")

RELATIONSHIP_TYPES = (
    "coworker", "cofounder", "board_member", "advisor", "investor",
    "employee", "speaker", "author", "student", "faculty",
    "family_social", "interview", "coauthor", "appointee", "unknown",
)

EDGE_STATUSES = ("weak", "candidate", "strong", "raw", "rejected", "accepted")

PROVIDERS = ("duckduckgo", "wikipedia", "scrape")


class Person(Base):
    __tablename__ = "people"

    id = Column(String, primary_key=True, default=_uuid)
    canonical_name = Column(String, nullable=False)
    norm_name = Column(String, index=True, unique=True, nullable=False)
    aliases = Column(JSON, default=list)
    meta = Column("metadata", JSON, default=dict)
    created_at = Column(String, default=lambda: _now().isoformat())


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    norm_name = Column(String, index=True, unique=True, nullable=False)
    type = Column(String, default="unknown")  # one of ORG_TYPES
    meta = Column("metadata", JSON, default=dict)
    created_at = Column(String, default=lambda: _now().isoformat())


class Source(Base):
    __tablename__ = "sources"

    id = Column(String, primary_key=True, default=_uuid)
    url = Column(String, index=True)
    title = Column(String)
    snippet = Column(Text)
    full_text = Column(Text, nullable=True)
    provider = Column(String)  # one of PROVIDERS
    query_used = Column(String)
    created_at = Column(String, default=lambda: _now().isoformat())


class RelationshipEdge(Base):
    __tablename__ = "relationship_edges"

    id = Column(String, primary_key=True, default=_uuid)
    person_a_id = Column(String, ForeignKey("people.id"), nullable=False, index=True)
    person_b_id = Column(String, ForeignKey("people.id"), nullable=True, index=True)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=True, index=True)
    relationship_type = Column(String, default="unknown")  # one of RELATIONSHIP_TYPES
    method = Column(Text)            # how the relationship was inferred
    evidence_snippet = Column(Text)
    source_id = Column(String, ForeignKey("sources.id"), nullable=True)
    confidence_base = Column(Float, default=0.0)
    confidence_raw = Column(Float, default=0.0)  # the adjusted/final confidence
    signals = Column(JSON, default=dict)         # EdgeSignals dump
    depth = Column(Integer, default=0)
    status = Column(String, default="weak")  # one of EDGE_STATUSES (tier)
    created_at = Column(String, default=lambda: _now().isoformat())

    person_a = relationship("Person", foreign_keys=[person_a_id])
    person_b = relationship("Person", foreign_keys=[person_b_id])
    organization = relationship("Organization", foreign_keys=[organization_id])
    source = relationship("Source", foreign_keys=[source_id])


# ===========================================================================
# Local network (uploaded CSV) — stage: graph matching (no Claude yet)
# ===========================================================================

MATCH_TYPES = ("exact_name", "name_company", "name_school", "org_overlap", "fuzzy_name")
PATH_STATUSES = ("unverified", "verified", "rejected")  # only 'unverified' is set here


class LocalProfile(Base):
    __tablename__ = "local_profiles"

    id = Column(String, primary_key=True, default=_uuid)
    canonical_name = Column(String, nullable=False)
    norm_name = Column(String, index=True)  # person_norm_key, for matching
    aliases = Column(JSON, default=list)
    email = Column(String, nullable=True, index=True)
    linkedin_url = Column(String, nullable=True)
    companies = Column(JSON, default=list)
    titles = Column(JSON, default=list)
    schools = Column(JSON, default=list)
    locations = Column(JSON, default=list)
    notes = Column(Text, nullable=True)
    raw_row = Column(JSON, default=dict)
    created_at = Column(String, default=lambda: _now().isoformat())


class LocalEdge(Base):
    __tablename__ = "local_edges"

    id = Column(String, primary_key=True, default=_uuid)
    # from_profile_id NULL == "You" (the network owner)
    from_profile_id = Column(String, ForeignKey("local_profiles.id"), nullable=True)
    to_profile_id = Column(String, ForeignKey("local_profiles.id"), nullable=False)
    edge_type = Column(String, default="uploaded_network")
    confidence = Column(Float, default=1.0)
    source = Column(String, default="uploaded_csv")
    created_at = Column(String, default=lambda: _now().isoformat())


class GraphMatch(Base):
    __tablename__ = "graph_matches"

    id = Column(String, primary_key=True, default=_uuid)
    local_profile_id = Column(String, ForeignKey("local_profiles.id"), nullable=False, index=True)
    public_person_id = Column(String, ForeignKey("people.id"), nullable=True, index=True)
    public_org_id = Column(String, ForeignKey("organizations.id"), nullable=True, index=True)
    match_type = Column(String)  # one of MATCH_TYPES
    confidence = Column(Float, default=0.0)
    explanation = Column(Text)
    created_at = Column(String, default=lambda: _now().isoformat())


class CandidatePath(Base):
    __tablename__ = "candidate_paths"

    id = Column(String, primary_key=True, default=_uuid)
    target_person_id = Column(String, ForeignKey("people.id"), nullable=False, index=True)
    local_profile_id = Column(String, ForeignKey("local_profiles.id"), nullable=False)
    public_person_id = Column(String, ForeignKey("people.id"), nullable=True)
    path_json = Column(JSON, default=dict)
    score = Column(Float, default=0.0)
    status = Column(String, default="unverified")  # NEVER 'accepted' at this stage
    created_at = Column(String, default=lambda: _now().isoformat())
