"""Delete first-degree edges that assert someone knows contacts they never uploaded.

network/ingest.backfill_graph_edges once bridged EVERY local_profile to the
operator running it, before profiles carried an owner. On a database holding two
people's LinkedIn exports that wrote one person's entire address book into the
other's first degree -- observed live: 2,152 linkedin_1st edges on "Abhimanyu
Sharma", of which 1,132 point at profiles he does not own (a student network:
"Undergraduate Researcher", "Finance Summer Intern", "Motion Picture Labs
Intern"). Ownership scoping stopped new ones; these were already written.

They are worse than noise. A linkedin_1st edge is the strongest claim in the
graph -- "these two personally know each other" -- so the pathfinder routes
through them by preference, and _route_exists will short-circuit an entire paid
walk on one. Every route through one is a warm intro that does not exist.

Dry run by default. --execute is the only thing that deletes, and it prints the
same summary first either way, because the number that matters is how many of
someone's OWN contacts survive.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.db import SessionLocal
from app.utils.names import person_norm_key

# Only this type. A discovered coworker/board tie to the same person may be
# perfectly real and independently sourced -- it is the "I uploaded them"
# assertion that is false here, not the person's existence in the graph.
BRIDGE_TYPE = "linkedin_1st"

# NOT EXISTS, not a LEFT JOIN to local_profiles filtered by owner_norm <>
# :owner: a contact shared across two operators' exports carries one
# local_profiles row PER uploader (that's the whole premise of the bug this
# script cleans up), so a LEFT JOIN fans out to one result row per owner. A
# row where the join happens to land on the OTHER owner then satisfies
# "owner_norm IS DISTINCT FROM :owner" and marks the edge doomed even when
# :owner also has their own legitimate row for the same contact -- silently
# queuing a real first-degree tie for deletion. NOT EXISTS asks the right
# question directly: does :owner have ANY row for this contact, regardless of
# who else does.
_SCOPE = """
      FROM relationship_edges e
      JOIN people pa ON pa.id = e.person_a_id
      JOIN people pb ON pb.id = e.person_b_id
     WHERE pa.norm_name = :owner
       AND e.relationship_type = :rtype
       AND NOT EXISTS (
             SELECT 1 FROM local_profiles lp
              WHERE lp.norm_name = pb.norm_name AND lp.owner_norm = :owner
           )
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("owner", help="the person whose first degree to clean, e.g. 'Abhimanyu Sharma'")
    ap.add_argument("--execute", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    owner = person_norm_key(args.owner)
    if not owner:
        print("unusable owner name")
        return 2

    db = SessionLocal()
    try:
        params = {"owner": owner, "rtype": BRIDGE_TYPE}
        total = db.execute(text(
            f"SELECT count(*) FROM relationship_edges e JOIN people pa ON pa.id=e.person_a_id "
            f"WHERE pa.norm_name=:owner AND e.relationship_type=:rtype"), params).scalar()
        doomed = db.execute(text(f"SELECT count(*) {_SCOPE}"), params).scalar()
        print(f"{args.owner}: {total} {BRIDGE_TYPE} edges, {doomed} assert contacts they do not own")
        print(f"  keeping {total - doomed}")

        # The displayed "owner=" is diagnostic only (any OTHER owner on record
        # for this contact, if there is one) -- it plays no part in the doom
        # decision above, which is NOT EXISTS against :owner alone.
        rows = db.execute(text(
            "SELECT pb.canonical_name, "
            "(SELECT owner_norm FROM local_profiles "
            "  WHERE norm_name = pb.norm_name AND owner_norm IS NOT NULL LIMIT 1) "
            f"{_SCOPE} LIMIT 10"), params).fetchall()
        print("\n  sample of what would go:")
        for r in rows:
            print(f"    {r[0][:40]:<40} owner={r[1] or '(none)'}")

        if not args.execute:
            print("\nDRY RUN — nothing deleted. Re-run with --execute.")
            return 0

        # Deleted by id rather than by the joined predicate: the scope query
        # reads through local_profiles, and expressing that as a DELETE...USING
        # would make the destructive statement a different statement from the
        # one just shown. Same rows, verifiably.
        ids = [r[0] for r in db.execute(text(f"SELECT e.id {_SCOPE}"), params).fetchall()]
        for i in range(0, len(ids), 500):
            db.execute(text("DELETE FROM relationship_edges WHERE id = ANY(:ids)"),
                       {"ids": ids[i:i + 500]})
        db.commit()
        print(f"\ndeleted {len(ids)} edges.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
