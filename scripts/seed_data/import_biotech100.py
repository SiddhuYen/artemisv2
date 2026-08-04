"""One-off import: biotech100_seed_connections.csv -> the shared graph.

Reuses the app's own identity-merge (builder.get_or_create_person) and edge
dedup (builder.add_edge_from_extraction) so these rows behave exactly like
live-discovered data -- future /connect runs will merge onto these same
nodes by norm_name instead of creating duplicates, and pathfinding just
walks them as ordinary RelationshipEdge rows (see connect._adjacency: no
special-casing by source).
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import select

from app.db import SessionLocal
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder
from app.models import Source

CSV_PATH = str(Path(__file__).resolve().parent / "biotech100_seed_connections.csv")

# The tech100/finance100/vc100 vocabulary, with ONE file-specific reading:
#
#   partner -> coworker. Both "partner" rows here are the Pfizer-BioNTech
#     vaccine partnership (Sahin <-> Bourla) -- a business relationship, so
#     coworker is right. Note this is the OPPOSITE of import_athletes.py,
#     where every "partner" row is romantic and maps to family_social. The
#     word carries no fixed meaning across these CSVs; it has to be read off
#     the evidence notes per file, which is why this comment exists rather
#     than a shared TYPE_MAP module.
TYPE_MAP = {
    "colleague": "coworker",
    "friend": "family_social",
    "family": "family_social",
    "cofounder": "cofounder",
    "executive": "employee",
    "board": "board_member",
    "spouse": "family_social",
    "ex-spouse": "family_social",
    "partner": "coworker",
    "early_employee": "employee",
    "mentor": "advisor",
    "protege": "advisor",
    "agent": "advisor",
    "manager": "advisor",
    "biographer": "author",
    "coauthor": "coauthor",
    "phd_advisor": "advisor",
    "phd_student": "student",
    "student": "student",
    "portfolio_founder": "investor",
    "investor": "investor",
    "predecessor": "appointee",
    "successor": "appointee",
}

FAMILY_SOCIAL_CONF = 0.80
PROFESSIONAL_CONF = 0.85

db = SessionLocal()
source_cache: dict[str, Source] = {}
stats = {"rows": 0, "edges_created": 0, "skipped": 0}
skipped_rows = []

try:
    people_before = db.query(builder.Person).count()

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(sys.argv) > 1:
        rows = rows[: int(sys.argv[1])]

    for row in rows:
        stats["rows"] += 1
        rtype = TYPE_MAP.get(row["relationship_type"])
        if rtype is None:
            stats["skipped"] += 1
            skipped_rows.append((row, f"unmapped relationship_type {row['relationship_type']!r}"))
            continue

        url = row["source_url"].strip()
        if url not in source_cache:
            existing = db.execute(
                select(Source).where(Source.url == url)
            ).scalars().first()
            if existing:
                source_cache[url] = existing
            else:
                src = Source(
                    url=url,
                    title=f"{row['connection_name']} — {row['relationship_type']} of {row['target_name']}",
                    snippet=row["evidence_note"].strip(),
                    provider="curated_seed",
                    query_used="",
                )
                db.add(src)
                db.flush()
                source_cache[url] = src
        source = source_cache[url]

        target = builder.get_or_create_person(
            db, row["target_name"], identity_text=row["target_context"])
        connection = builder.get_or_create_person(
            db, row["connection_name"], identity_text=row["connection_context"])
        if target is None or connection is None:
            stats["skipped"] += 1
            skipped_rows.append((row, "person resolution failed"))
            continue

        conf = FAMILY_SOCIAL_CONF if rtype == "family_social" else PROFESSIONAL_CONF
        edge = ExtractedEdge(
            person_a=target.canonical_name,
            person_b=connection.canonical_name,
            relationship_type=rtype,
            method="curated_seed_import",
            evidence_snippet=row["evidence_note"].strip(),
            source_url=url,
            confidence_base=conf,
            confidence_adjusted=conf,
            signals=EdgeSignals(trusted=True),
            other_kind="person",
        )
        result = builder.add_edge_from_extraction(
            db, subject=target, edge=edge, depth=0, source=source, counterpart=connection)
        if result is not None:
            stats["edges_created"] += 1

    builder.commit_with_retry(db)
    people_after = db.query(builder.Person).count()
    stats["people_created"] = people_after - people_before
    print("=== IMPORT COMPLETE ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    if skipped_rows:
        print("\n--- skipped rows ---")
        for row, reason in skipped_rows:
            print(f"{row['target_name']} -> {row['connection_name']}: {reason}")
finally:
    db.close()
