"""Backfill: re-check existing wikidata-colleagues edges against the org-size
gate in providers.wikidata.colleagues(), deleting any that no longer pass.

providers/wikidata.py's org-size gate only changes what NEW expansions write.
This cleans up what earlier runs already persisted before the gate existed --
e.g. the false "coworker" edges between people who share nothing but a
centuries-spanning employer like Harvard University.

    python -m scripts.backfill_wikidata_colleague_orgs
    python -m scripts.backfill_wikidata_colleague_orgs --dry-run
"""
from __future__ import annotations

import argparse

from app.db import SessionLocal
from app.graph import builder
from app.models import Person, RelationshipEdge, Source
from app.providers.orchestrator import SearchOrchestrator


def _subjects_with_colleague_edges(db):
    """(person_id, canonical_name, wikidata_qid) for every distinct subject
    with at least one relationship_edges row sourced from wikidata-colleagues."""
    return (
        db.query(Person.id, Person.canonical_name, Person.wikidata_qid)
        .join(RelationshipEdge, RelationshipEdge.person_a_id == Person.id)
        .join(Source, RelationshipEdge.source_id == Source.id)
        .filter(Source.provider == "wikidata-colleagues")
        .distinct()
        .all()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                         help="report what would be deleted without deleting")
    args = parser.parse_args()

    orch = SearchOrchestrator()
    db = SessionLocal()
    total_checked = total_deleted = 0
    try:
        subjects = _subjects_with_colleague_edges(db)
        print(f"{len(subjects)} subject(s) with wikidata-colleagues edges\n")

        for person_id, name, qid in subjects:
            if not qid:
                # Can't re-derive without the QID this provider keyed on.
                print(f"  SKIP {name}: no wikidata_qid on record")
                continue

            # Cache key is versioned (colleagues_v2), so this recomputes
            # under the new org-size gate rather than serving the old list.
            fresh = orch.wikidata.colleagues(qid)
            keep_names = {c["name"].lower() for c in fresh}

            existing = (
                db.query(RelationshipEdge, Person.canonical_name)
                .join(Source, RelationshipEdge.source_id == Source.id)
                .join(Person, Person.id == RelationshipEdge.person_b_id)
                .filter(RelationshipEdge.person_a_id == person_id,
                        Source.provider == "wikidata-colleagues")
                .all()
            )
            stale_ids = [edge.id for edge, other_name in existing
                         if other_name.lower() not in keep_names]
            total_checked += len(existing)

            if stale_ids:
                print(f"  {name}: {len(stale_ids)}/{len(existing)} edge(s) "
                      f"fail the new size gate")
                if not args.dry_run:
                    builder.delete_relationship_edges_with_retry(
                        db, RelationshipEdge.id.in_(stale_ids))
                    db.commit()
                total_deleted += len(stale_ids)

        verb = "would be deleted" if args.dry_run else "deleted"
        print(f"\n{total_checked} edge(s) checked, {total_deleted} {verb}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
