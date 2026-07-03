"""Graph persistence: dedup-aware upserts for people, orgs, sources, edges.

Dedup keys:
  - people -> person_norm_key (normalised, middle-initials stripped); surface
              variants are auto-stored as aliases.
  - orgs   -> org_norm_key (normalised, trailing legal/structural suffix stripped).
  - sources-> url
  - edges  -> (person_a, counterpart, relationship_type, source_url) -> discard dup.

Every edge carries source URL + evidence snippet + base & adjusted confidence +
signals + status tier. No relationship is ever auto-set to 'accepted'
(verification is deferred).
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import config
from ..extraction import tier
from ..extraction.schemas import ExtractedEdge
from ..models import (
    CandidatePath,
    GraphMatch,
    Organization,
    Person,
    RelationshipEdge,
    Source,
)
from ..providers.base import SearchResult
from ..utils.names import name_variants, org_norm_key, person_norm_key


def reset_public_graph(db: Session) -> None:
    """Clear the PUBLIC graph + derived matches/paths, preserving the uploaded
    local network (local_profiles / local_edges). Children first for FK safety."""
    db.query(CandidatePath).delete()
    db.query(GraphMatch).delete()
    db.query(RelationshipEdge).delete()
    db.query(Source).delete()
    db.query(Organization).delete()
    db.query(Person).delete()
    db.commit()


# --- node-count cap --------------------------------------------------------
def node_count(db: Session) -> int:
    # Cap on PEOPLE only — they're the expandable, path-relevant nodes.
    # Orgs are cheap leaf metadata and shouldn't starve the person budget.
    return db.scalar(select(func.count()).select_from(Person)) or 0


def at_node_cap(db: Session) -> bool:
    return node_count(db) >= config.MAX_TOTAL_NODES


# --- entity upserts --------------------------------------------------------
def get_or_create_person(db: Session, name: str, allow_create: bool = True) -> Optional[Person]:
    norm = person_norm_key(name)
    if not norm:
        return None
    existing = db.execute(
        select(Person).where(Person.norm_name == norm)
    ).scalar_one_or_none()
    if existing:
        _merge_aliases(existing, name)
        return existing
    if not allow_create:
        return None
    person = Person(
        canonical_name=name.strip(),
        norm_name=norm,
        aliases=sorted(v for v in name_variants(name) if v != name.strip()),
        meta={},
    )
    db.add(person)
    db.flush()
    return person


def _merge_aliases(person: Person, surface: str) -> None:
    aliases = set(person.aliases or [])
    for v in name_variants(surface):
        if v and v != person.canonical_name:
            aliases.add(v)
    # prefer the longest surface form as the canonical display name
    if len(surface.strip()) > len(person.canonical_name):
        aliases.add(person.canonical_name)
        person.canonical_name = surface.strip()
        aliases.discard(person.canonical_name)
    if aliases != set(person.aliases or []):
        person.aliases = sorted(aliases)


def get_or_create_org(
    db: Session, name: str, org_type: str = "unknown", allow_create: bool = True
) -> Optional[Organization]:
    norm = org_norm_key(name)
    if not norm:
        return None
    existing = db.execute(
        select(Organization).where(Organization.norm_name == norm)
    ).scalar_one_or_none()
    if existing:
        if existing.type == "unknown" and org_type != "unknown":
            existing.type = org_type
        return existing
    if not allow_create:
        return None
    org = Organization(name=name.strip(), norm_name=norm, type=org_type, meta={})
    db.add(org)
    db.flush()
    return org


def save_source(
    db: Session, result: SearchResult, query_used: str, full_text: Optional[str] = None
) -> Source:
    existing = db.execute(
        select(Source).where(Source.url == result.url)
    ).scalar_one_or_none()
    if existing:
        if full_text and not existing.full_text:
            existing.full_text = full_text
        return existing
    source = Source(
        url=result.url,
        title=result.title,
        snippet=result.snippet,
        full_text=full_text,
        provider=result.provider,
        query_used=query_used,
    )
    db.add(source)
    db.flush()
    return source


# --- status policy ---------------------------------------------------------
def derive_status(relationship_type: str, confidence: float) -> str:
    """Confidence tier (weak/candidate/strong); family_social never reaches strong."""
    t = tier(confidence)
    if relationship_type == "family_social" and t == "strong":
        return "candidate"
    return t


# --- edge upsert -----------------------------------------------------------
def add_edge_from_extraction(
    db: Session,
    subject: Person,
    edge: ExtractedEdge,
    depth: int,
    source: Optional[Source],
    counterpart,  # Person | Organization
) -> Optional[RelationshipEdge]:
    """Persist one ExtractedEdge, applying the (a,b,type,source_url) dedup rule."""
    is_person = edge.other_kind == "person"
    other_id = counterpart.id if is_person else None
    org_id = counterpart.id if not is_person else None
    source_id = source.id if source else None
    conf = edge.confidence_adjusted

    # Dedup rule: same (person_a, counterpart, relationship_type, source_url).
    existing = db.execute(
        select(RelationshipEdge).where(
            RelationshipEdge.person_a_id == subject.id,
            RelationshipEdge.person_b_id == other_id,
            RelationshipEdge.organization_id == org_id,
            RelationshipEdge.relationship_type == edge.relationship_type,
            RelationshipEdge.source_id == source_id,
        )
    ).scalar_one_or_none()
    if existing:
        if conf > (existing.confidence_raw or 0):
            existing.confidence_raw = conf
            existing.confidence_base = edge.confidence_base
            existing.status = derive_status(edge.relationship_type, conf)
            existing.signals = edge.signals.model_dump()
        return existing

    row = RelationshipEdge(
        person_a_id=subject.id,
        person_b_id=other_id,
        organization_id=org_id,
        relationship_type=edge.relationship_type,
        method=edge.method,
        evidence_snippet=edge.evidence_snippet,
        source_id=source_id,
        confidence_base=round(edge.confidence_base, 3),
        confidence_raw=round(conf, 3),
        signals=edge.signals.model_dump(),
        depth=depth,
        status=derive_status(edge.relationship_type, conf),
    )
    db.add(row)
    db.flush()
    return row
