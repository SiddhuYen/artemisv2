"""Judge every uploaded contact once: would a search for this name return
anything, and which world does the row sit in?

Backfills LocalProfile.reach_profile for contacts imported before
extraction/contact_profiler existed. Costs no searches -- one batched Claude
pass over rows the operator already gave us.

Dry run by default; --execute writes. Re-running only profiles rows that don't
have one yet, so it converges instead of re-spending.

    python scripts/backfill_reach_profiles.py              # plan + cost
    python scripts/backfill_reach_profiles.py --execute
    python scripts/backfill_reach_profiles.py --execute --limit 50
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app import config
from app.db import SessionLocal
from app.extraction import contact_profiler
from app.models import LocalProfile


def _item(profile: LocalProfile) -> dict:
    return {
        "name": profile.canonical_name,
        "employer": (profile.companies or [None])[0],
        "title": (profile.titles or [None])[0],
        "school": (profile.schools or [None])[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="profile at most N")
    ap.add_argument("--reprofile", action="store_true",
                    help="also redo contacts that already have a profile")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        rows = list(db.execute(select(LocalProfile)).scalars())
        todo = [p for p in rows
                if args.reprofile or not isinstance(p.reach_profile, dict)]
        if args.limit:
            todo = todo[:args.limit]

        batches = (len(todo) + config.CONTACT_PROFILE_BATCH - 1) // max(
            1, config.CONTACT_PROFILE_BATCH)
        print(f"contacts        : {len(rows)}")
        print(f"already profiled: {len(rows) - len([p for p in rows if not isinstance(p.reach_profile, dict)])}")
        print(f"to profile      : {len(todo)}  (~{batches} batched calls, no searches)")
        print(f"claude active   : {contact_profiler.is_active()}")

        if not args.execute:
            print("\nDRY RUN — nothing written. Re-run with --execute.")
            return
        if not todo:
            print("\nnothing to do.")
            return

        verdicts = contact_profiler.profile([_item(p) for p in todo])
        written = 0
        for profile, verdict in zip(todo, verdicts):
            if verdict is None:
                continue          # no judgment reached; leave it unprofiled
            profile.reach_profile = verdict
            db.add(profile)
            written += 1
        db.commit()

        dist = collections.Counter(
            (p.reach_profile or {}).get("footprint") for p in todo
            if isinstance(p.reach_profile, dict))
        doms = collections.Counter(
            (p.reach_profile or {}).get("domain") for p in todo
            if isinstance(p.reach_profile, dict))
        print(f"\nprofiled  : {written} of {len(todo)}")
        print("footprint :", dict(dist))
        print("domain    :", dict(doms.most_common()))
    finally:
        db.close()


if __name__ == "__main__":
    main()
