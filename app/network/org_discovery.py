"""Discover people affiliated with an organization and add them to the local
network.

Builds a public graph seeded on the ORG name (web search, not the Wikipedia
person path), then promotes people DIRECTLY connected to the org seed into
local_profiles (tagged with the source org, connected to "You"). Only
candidate/strong edges are promoted — tangential, weak mentions are not added
to your network.

The temporary org public graph is cleared afterwards (its value now lives in
local_profiles), leaving the public graph clean for the next target search.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..graph import builder
from ..graph.expansion import expand_graph
from ..models import LocalEdge, LocalProfile, Person, RelationshipEdge
from ..utils.names import person_norm_key

_PROMOTABLE_STATUS = {"candidate", "strong"}


def discover_org_network(
    db: Session, org_name: str, depth: int = 1,
    source_tag: str = "org_discovery", progress=None,
) -> dict:
    """Return {discovered, promoted, updated} after enriching the local network."""
    # 1) build the org's public graph (seed is an ORG -> web search route)
    expand_graph(db, org_name, depth, progress=progress, seed_is_person=False)

    seed_norm = person_norm_key(org_name)
    seed = db.execute(
        select(Person).where(Person.norm_name == seed_norm)
    ).scalar_one_or_none()
    if seed is None:
        builder.reset_public_graph(db)
        return {"discovered": 0, "promoted": 0, "updated": 0}

    # 2) people DIRECTLY related to the org seed (candidate/strong edges only)
    related_ids = set()
    for e in db.execute(
        select(RelationshipEdge).where(
            RelationshipEdge.person_a_id == seed.id,
            RelationshipEdge.person_b_id.isnot(None),
        )
    ).scalars():
        if e.status in _PROMOTABLE_STATUS:
            related_ids.add(e.person_b_id)

    people = {p.id: p for p in db.execute(select(Person)).scalars()}
    promoted = updated = 0

    # 3) promote into local_profiles (connected to You)
    for pid in related_ids:
        person = people.get(pid)
        if person is None or person.id == seed.id:
            continue
        existing = db.execute(
            select(LocalProfile).where(LocalProfile.norm_name == person.norm_name)
        ).scalar_one_or_none()
        if existing:
            companies = set(existing.companies or [])
            if org_name not in companies:
                companies.add(org_name)
                existing.companies = sorted(companies)
                updated += 1
            continue
        lp = LocalProfile(
            canonical_name=person.canonical_name,
            norm_name=person.norm_name,
            aliases=person.aliases or [],
            companies=[org_name],
            titles=[], schools=[], locations=[],
            notes=f"Discovered via public search on '{org_name}'.",
            raw_row={"source": source_tag, "org": org_name},
        )
        db.add(lp)
        db.flush()
        db.add(LocalEdge(from_profile_id=None, to_profile_id=lp.id,
                         edge_type="org_affiliate", source=source_tag))
        promoted += 1

    db.commit()

    # 4) clear the temporary org public graph (data now lives in local_profiles)
    builder.reset_public_graph(db)
    return {"discovered": len(related_ids), "promoted": promoted, "updated": updated}
