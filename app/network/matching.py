"""Tiered matching of local profiles against the public relationship graph.

Tiers (high precision; rejects broad-clue-only matches):
  1 exact_name     : normalized full name == public canonical name      -> 0.95
  2 name_company   : high fuzzy name + company/org overlap              -> 0.80-0.90
  3 name_school    : high fuzzy name + school/location overlap (weak)   -> 0.60-0.75
  -  fuzzy_name     : high fuzzy name only, no corroboration (weak)      -> 0.50
  4 org_overlap    : local company/school appears in graph (near-miss)  -> 0.40-0.60

Rejected by construction: same-city-only, same-industry-only, title-only,
generic school-only without name similarity.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import GraphMatch, LocalProfile, Organization, Person, RelationshipEdge
from ..utils.names import org_norm_key, person_norm_key
from .fuzzy import HIGH_SIMILARITY, name_similarity


def _scale(sim: float, lo: float, hi: float) -> float:
    t = (sim - HIGH_SIMILARITY) / (1.0 - HIGH_SIMILARITY) if sim > HIGH_SIMILARITY else 0.0
    return round(lo + (hi - lo) * min(max(t, 0.0), 1.0), 3)


class _PublicGraph:
    """Indexed view of the public graph used for matching."""

    def __init__(self, db: Session) -> None:
        self.people: List[Person] = list(db.execute(select(Person)).scalars())
        self.orgs: List[Organization] = list(db.execute(select(Organization)).scalars())
        self.org_by_id = {o.id: o for o in self.orgs}
        self.org_by_norm = {o.norm_name: o for o in self.orgs}

        # person_id -> set of connected org norm keys (and school subset)
        self.person_orgs: Dict[str, Set[str]] = defaultdict(set)
        self.person_schools: Dict[str, Set[str]] = defaultdict(set)
        for e in db.execute(
            select(RelationshipEdge).where(RelationshipEdge.organization_id.isnot(None))
        ).scalars():
            org = self.org_by_id.get(e.organization_id)
            if not org:
                continue
            self.person_orgs[e.person_a_id].add(org.norm_name)
            if org.type == "school":
                self.person_schools[e.person_a_id].add(org.norm_name)


def _overlap(local_values: List[str], public_norms: Set[str]) -> List[str]:
    hits = []
    for v in local_values or []:
        k = org_norm_key(v)
        if k and k in public_norms:
            hits.append(v)
    return hits


def _match_profile_to_people(profile: LocalProfile, pg: _PublicGraph) -> List[dict]:
    """Person-level matches (tiers 1-3 + fuzzy_name)."""
    out: List[dict] = []
    local_norm = profile.norm_name or person_norm_key(profile.canonical_name)

    for person in pg.people:
        # Tier 1 — exact name
        if local_norm and person.norm_name == local_norm:
            out.append(dict(
                person=person, match_type="exact_name", confidence=0.95,
                explanation=f"Exact normalized name match ('{person.canonical_name}').",
            ))
            continue

        sim = name_similarity(profile.canonical_name, person.canonical_name)
        if sim < HIGH_SIMILARITY:
            continue  # no name signal -> never match on company/school alone

        company_hits = _overlap(profile.companies, pg.person_orgs.get(person.id, set()))
        school_hits = _overlap(profile.schools, pg.person_schools.get(person.id, set()))

        if company_hits:
            out.append(dict(
                person=person, match_type="name_company",
                confidence=_scale(sim, 0.80, 0.90),
                explanation=(f"Fuzzy name match ({sim:.2f}) + company overlap "
                             f"({', '.join(company_hits)})."),
            ))
        elif school_hits:
            out.append(dict(
                person=person, match_type="name_school",
                confidence=_scale(sim, 0.60, 0.75), weak=True,
                explanation=(f"Fuzzy name match ({sim:.2f}) + school overlap "
                             f"({', '.join(school_hits)}). WEAK: corroborate further."),
            ))
        else:
            out.append(dict(
                person=person, match_type="fuzzy_name",
                confidence=0.50, weak=True,
                explanation=(f"High fuzzy name similarity ({sim:.2f}) only, no "
                             f"company/school corroboration. WEAK."),
            ))
    return out


def _match_profile_to_orgs(profile: LocalProfile, pg: _PublicGraph) -> List[dict]:
    """Tier 4 — organization-only bridges (near-miss, no person path)."""
    out: List[dict] = []
    seen: Set[str] = set()
    for value in (profile.companies or []) + (profile.schools or []):
        org = pg.org_by_norm.get(org_norm_key(value))
        if org and org.id not in seen:
            seen.add(org.id)
            out.append(dict(
                org=org, match_type="org_overlap", confidence=0.50,
                explanation=(f"Local org '{value}' appears in the public graph "
                             f"('{org.name}'). Near-miss bridge, not a person link."),
            ))
    return out


def run_matching(db: Session) -> List[GraphMatch]:
    """Recompute all graph_matches against the current public graph."""
    db.query(GraphMatch).delete()
    db.flush()

    pg = _PublicGraph(db)
    profiles = list(db.execute(select(LocalProfile)).scalars())
    created: List[GraphMatch] = []

    for profile in profiles:
        for m in _match_profile_to_people(profile, pg):
            gm = GraphMatch(
                local_profile_id=profile.id,
                public_person_id=m["person"].id,
                public_org_id=None,
                match_type=m["match_type"],
                confidence=round(m["confidence"], 3),
                explanation=m["explanation"],
            )
            db.add(gm)
            created.append(gm)
        for m in _match_profile_to_orgs(profile, pg):
            gm = GraphMatch(
                local_profile_id=profile.id,
                public_person_id=None,
                public_org_id=m["org"].id,
                match_type=m["match_type"],
                confidence=round(m["confidence"], 3),
                explanation=m["explanation"],
            )
            db.add(gm)
            created.append(gm)

    db.commit()
    return created
