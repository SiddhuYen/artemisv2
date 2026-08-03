"""Clear the DISCOVERED graph, keep everything a person gave us.

Why this exists: junk edges are not inert. connect._route_exists
short-circuits the entire paid walk the moment any traversable route exists,
so a wrong edge left over from testing does not merely rank badly -- it
suppresses the search that would replace it, permanently and silently, for
that pair. Fixes to the extractors are also never retroactive: the homonym
gate stops NEW bad OpenAlex clusters, it does not retract the ones written
before it existed. Periodically rebuilding the derived graph from inputs we
still hold is the only way to get a baseline that reflects today's code.

The split is by PROVENANCE, not by usefulness:

  DERIVED  -- everything Artemis inferred by searching the web. Reproducible:
              re-running the seed importers and the walks rebuilds it, mostly
              off the provider cache (artemis_cache.db, 30-day TTL), which
              this script never touches. That cache is where the money went;
              the graph is just what was computed from it.

  GIVEN    -- everything that came from a human: uploaded contacts, the owner
              profile, boards people arranged by hand. Not reproducible by any
              amount of searching. Never deleted here.

Safety properties, in order of how much they matter:

  1. Dry run by default. Deleting requires --execute.
  2. Every table is classified. A table in NEITHER list aborts the run rather
     than being cleared or skipped on a guess -- so adding a model without
     thinking about this script fails loudly instead of silently losing data.
  3. A backup is written before the first DELETE unless --no-backup, and
     --restore feeds it back. The trigger is meant to be reversible.

Usage:
    python scripts/reset_graph.py                    # dry run, shows the plan
    python scripts/reset_graph.py --execute          # back up, then clear
    python scripts/reset_graph.py --restore FILE     # put it back
"""
import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import inspect, select, text

from app import models  # noqa: F401  -- populates Base.metadata with every table
from app.db import Base, SessionLocal, engine

# Ordered child -> parent so plain DELETEs never trip a foreign key. (TRUNCATE
# ... CASCADE would be faster on Postgres but would also silently reach into
# tables this script has deliberately classified as GIVEN.)
DERIVED = [
    "relationship_edges",   # FK -> people, organizations, sources
    "graph_matches",        # FK -> people, local_profiles
    "candidate_paths",      # FK -> people
    "enrichment_tasks",     # FK -> enrichment_runs
    "enrichment_runs",
    "sources",
    "organizations",
    "people",
]

# Never touched. local_profiles/local_edges are the uploaded network -- the one
# thing here no search can reconstruct. boards/board_pages store hand-arranged
# canvases as self-contained JSON (no FK into people), so they survive a graph
# wipe intact; their node ids just stop resolving back to graph rows.
GIVEN = [
    "local_profiles",
    "local_edges",
    "owner_profiles",
    "boards",
    "board_pages",
]


def _classify(live_tables):
    """Every live table must be in exactly one list. Anything else aborts."""
    known = set(DERIVED) | set(GIVEN)
    unknown = sorted(set(live_tables) - known)
    if unknown:
        raise SystemExit(
            "ABORT: unclassified table(s): " + ", ".join(unknown) +
            "\nAdd each to DERIVED or GIVEN in this script. Refusing to guess "
            "-- guessing wrong here either destroys data or silently leaves "
            "stale rows behind.")
    return [t for t in DERIVED if t in live_tables]


def _counts(db, tables):
    return {t: db.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in tables}


def _backup(db, tables, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": datetime.now(timezone.utc).isoformat(),
               "database": str(engine.url.render_as_string(hide_password=True)),
               "tables": {}}
    for name in tables:
        table = Base.metadata.tables[name]
        rows = db.execute(select(table)).mappings().all()
        # default=str so datetime/UUID survive; JSON columns are already dicts
        # by the time SQLAlchemy's type layer has decoded them.
        payload["tables"][name] = [json.loads(json.dumps(dict(r), default=str))
                                   for r in rows]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def _restore(db, path: Path) -> dict:
    """Reload a backup. Inserts through SQLAlchemy Core table objects, NOT raw
    text() SQL: a JSON column round-trips as a Python dict, and psycopg2 cannot
    adapt a bare dict as a bind parameter -- so the raw-SQL version worked on
    SQLite and would have failed on Postgres, which is the only place this
    actually needs to run. Core applies each column's own type on the way in.
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    restored = {}
    # Parent -> child, the reverse of the delete order, so FKs resolve.
    for name in reversed(DERIVED):
        rows = payload["tables"].get(name) or []
        if not rows:
            continue
        table = Base.metadata.tables[name]
        db.execute(table.insert(), rows)
        restored[name] = len(rows)
    db.commit()
    return restored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="actually delete (default is a dry run)")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the pre-delete backup (not advised)")
    ap.add_argument("--backup-dir", default="scripts/backups")
    ap.add_argument("--restore", metavar="FILE",
                    help="reload a backup written by an earlier --execute")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        live = set(inspect(engine).get_table_names())
        targets = _classify(live)

        print(f"database : {engine.url.render_as_string(hide_password=True)}")

        if args.restore:
            restored = _restore(db, Path(args.restore))
            print("\n=== RESTORED ===")
            for t, n in restored.items():
                print(f"  {t:22} {n:>8}")
            return

        before = _counts(db, targets)
        keep = _counts(db, [t for t in GIVEN if t in live])

        print("\nWILL CLEAR (derived — rebuilt by re-running the importers/walks)")
        for t in targets:
            print(f"  {t:22} {before[t]:>8}")
        print("\nWILL KEEP (given — not reproducible by searching)")
        for t, n in keep.items():
            print(f"  {t:22} {n:>8}")

        if not args.execute:
            print("\nDRY RUN — nothing was changed. Re-run with --execute.")
            return

        if not args.no_backup:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = _backup(db, targets, Path(args.backup_dir) / f"graph-{stamp}.json.gz")
            print(f"\nbackup   : {out} ({out.stat().st_size / 1e6:.1f} MB)")

        for table in targets:
            db.execute(text(f"DELETE FROM {table}"))
        db.commit()

        after = _counts(db, targets)
        still_kept = _counts(db, [t for t in GIVEN if t in live])
        print("\n=== CLEARED ===")
        for t in targets:
            print(f"  {t:22} {before[t]:>8} -> {after[t]}")
        print("\n=== PRESERVED ===")
        for t, n in still_kept.items():
            flag = "OK" if n == keep[t] else f"CHANGED (was {keep[t]})"
            print(f"  {t:22} {n:>8}  {flag}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
