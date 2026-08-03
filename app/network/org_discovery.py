"""Discover people affiliated with an organization and add them to the local
network.

Builds a public graph seeded on the ORG name (web search, not the Wikipedia
person path), then promotes people DIRECTLY connected to the org seed into
local_profiles (tagged with the source org, connected to "You"). Only
candidate/strong edges are promoted — tangential, weak mentions are not added
to your network.

The temporary org public graph is cleared afterwards (its value now lives in
local_profiles), leaving the public graph clean for the next target search --
but only when that graph is a private local file. See _clear_scratch_graph.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..graph import builder
from ..graph.expansion import expand_graph
from ..models import LocalEdge, LocalProfile, Person, RelationshipEdge
from ..utils.names import person_norm_key

_PROMOTABLE_STATUS = {"candidate", "strong"}


def _clear_scratch_graph(db: Session, progress=None) -> None:
    """Drop the throwaway org graph — but never on a SHARED database.

    This function treats the whole public graph as its own scratch space,
    which holds only while that graph is a private local file. On a team's
    Postgres the same wipe deletes every collaborator's work as a SIDE EFFECT
    of one person running `add-org-network` — destruction nobody asked for and
    nobody would think to guard against.

    Leaving the org graph behind instead is strictly the lesser problem: the
    graph is additive by design, so the residue is ordinary discovered data
    (it just wasn't asked for), and the promoted local_profiles are already
    committed by this point either way.
    """
    if builder.graph_is_shared():
        if progress:
            progress("  ℹ shared database — leaving the temporary org graph in "
                     "place rather than wiping everyone's data")
        return
    builder.reset_public_graph(db)


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
        _clear_scratch_graph(db)
        return {"discovered": 0, "promoted": 0, "updated": 0}

    # 2) people DIRECTLY related to the org seed (candidate/strong edges only).
    # Either orientation: person_a/person_b record which side happened to be
    # extracted first, not a direction, so matching person_a alone silently
    # dropped every tie discovered from the other end -- the same asymmetry
    # expansion._reuse_existing_neighbors was fixed for. It matters more now
    # that symmetric ties are stored once rather than mirrored (see
    # models.SYMMETRIC_RELATIONSHIP_TYPES).
    related_ids = set()
    for e in db.execute(
        select(RelationshipEdge).where(
            or_(RelationshipEdge.person_a_id == seed.id,
                RelationshipEdge.person_b_id == seed.id),
            RelationshipEdge.person_b_id.isnot(None),
        )
    ).scalars():
        if e.status in _PROMOTABLE_STATUS:
            other = e.person_b_id if e.person_a_id == seed.id else e.person_a_id
            if other:
                related_ids.add(other)

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
    _clear_scratch_graph(db, progress=progress)
    return {"discovered": len(related_ids), "promoted": promoted, "updated": updated}
