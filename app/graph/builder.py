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

from sqlalchemy import func, or_, select
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
from . import disambiguate


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
def get_or_create_person(db: Session, name: str, qid: Optional[str] = None,
                         allow_create: bool = True,
                         identity_text: Optional[str] = None) -> Optional[Person]:
    """Resolve a person node, disambiguating homonyms by Wikidata QID.

    Identity rules:
      - qid given: same-QID node wins (authoritative merge across name variants);
        a name-match with NO qid ADOPTS this qid -- UNLESS the node's own
        already-accumulated evidence conflicts with `identity_text` (see
        _homonym_conflict), in which case it's treated like case 3 below; a
        name-match with a DIFFERENT qid is a distinct person (a homonym) ->
        a separate, QID-suffixed node.
      - no qid: fall back to the normalized-name key (today's behavior).

    `identity_text` is a short description of the identity `qid` refers to
    (e.g. a Wikipedia summary) -- the candidate side of the homonym check.
    Callers that don't have one (e.g. counterpart resolution during edge
    extraction, which never passes a qid at all) get the guard's old,
    unconditional-adopt behavior; there is nothing to check the name against.
    """
    norm = person_norm_key(name)
    if not norm:
        return None

    if qid:
        # 1) authoritative: an existing node already carrying this QID
        by_qid = db.execute(
            select(Person).where(Person.wikidata_qid == qid)
        ).scalar_one_or_none()
        if by_qid:
            _merge_aliases(by_qid, name)
            return by_qid
        # 2) a name-match with no QID yet -> normally it's this same person;
        #    adopt the QID -- unless the homonym guard says otherwise.
        by_name = db.execute(
            select(Person).where(Person.norm_name == norm)
        ).scalar_one_or_none()
        adopt = by_name is not None and not by_name.wikidata_qid
        if adopt and _homonym_conflict(db, by_name, identity_text):
            adopt = False
        if adopt:
            by_name.wikidata_qid = qid
            _merge_aliases(by_name, name)
            return by_name
        # 3) name-match exists but with a DIFFERENT QID, or the homonym guard
        #    just rejected adopting this QID onto it -> genuine homonym.
        #    Give the newcomer a QID-disambiguated key so they don't collide.
        if not allow_create:
            return None
        node_norm = norm if by_name is None else f"{norm}#{qid}"
        return _new_person(db, name, node_norm, qid)

    # no QID: plain name-key dedup
    existing = db.execute(
        select(Person).where(Person.norm_name == norm)
    ).scalar_one_or_none()
    if existing:
        _merge_aliases(existing, name)
        return existing
    if not allow_create:
        return None
    return _new_person(db, name, norm, None)


def _existing_evidence_signal(db: Session, person: Person) -> str:
    """A few of `person`'s own already-persisted edge-evidence snippets --
    used as an independent professional-domain signal by _homonym_conflict.

    Deliberately NOT a fresh search for `person`'s name: a search run to
    justify adopting `identity_text` could be the SAME search that produced
    `identity_text` in the first place, which would let a single fame-
    dominated lookup confirm itself. These snippets instead come from
    whatever OTHER subject's search first discovered this person as a
    counterpart, before this identity was ever proposed for it.
    """
    rows = db.execute(
        select(RelationshipEdge.evidence_snippet)
        .where(or_(RelationshipEdge.person_a_id == person.id,
                   RelationshipEdge.person_b_id == person.id))
        .limit(config.IDENTITY_SIGNAL_MAX_SNIPPETS)
    ).scalars()
    return " ".join(s for s in rows if s)


def _homonym_conflict(db: Session, existing: Person,
                      identity_text: Optional[str]) -> bool:
    """True when `existing`'s own accumulated evidence clearly anchors in a
    different professional world than `identity_text` (a description of the
    identity about to be adopted onto it) -- see graph.disambiguate.

    Silent (False) whenever there's nothing to compare: no identity_text was
    given, the feature is disabled, or `existing` has no prior evidence yet
    (a brand-new node adopts freely, same as before this guard existed).
    Records a `homonym_rejected` note in `existing.meta` on a real conflict --
    advisory only, not a permanent block: it doesn't stop `existing` from
    continuing to accumulate its own edges, and a later direct assignment of
    `wikidata_qid` (bypassing this path entirely, same as case 1 above) can
    still confirm the identity if a human decides the rejection was wrong.
    """
    if not identity_text or not config.IDENTITY_VERIFY_ENABLED:
        return False
    signal = _existing_evidence_signal(db, existing)
    if not signal:
        return False
    if not disambiguate.domain_conflict(signal, identity_text):
        return False
    meta = dict(existing.meta or {})
    meta["homonym_rejected"] = {"identity_text": identity_text[:300]}
    existing.meta = meta
    return True


def _new_person(db: Session, name: str, norm: str, qid: Optional[str]) -> Person:
    person = Person(
        canonical_name=name.strip(),
        norm_name=norm,
        wikidata_qid=qid,
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
