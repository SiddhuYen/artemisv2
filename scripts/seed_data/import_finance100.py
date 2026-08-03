"""Import finance100_seed_connections.csv into the shared graph.

relationship_type -> RELATIONSHIP_TYPES mapping (tech100 conventions, not
celeb100's -- here "partner" always means a business/co-ownership
partnership, e.g. Boehly/Eghbali co-owning Chelsea, not a romantic partner):
  friend/family/spouse/ex-spouse -> family_social
  partner/colleague             -> coworker
  cofounder                     -> cofounder
  executive                     -> employee
  successor/predecessor         -> appointee
  mentor/protege                -> advisor
  investor/portfolio_founder    -> investor
"""
import csv
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db import SessionLocal
from app.extraction.schemas import EdgeSignals, ExtractedEdge
from app.graph import builder
from app.models import Source

CSV_PATH = str(Path(__file__).resolve().parent / "finance100_seed_connections.csv")

TYPE_MAP = {
    "friend": "family_social",
    "family": "family_social",
    "spouse": "family_social",
    "ex-spouse": "family_social",
    "partner": "coworker",
    "colleague": "coworker",
    "cofounder": "cofounder",
    "executive": "employee",
    "successor": "appointee",
    "predecessor": "appointee",
    "mentor": "advisor",
    "protege": "advisor",
    "investor": "investor",
    "portfolio_founder": "investor",
}

CONF = 0.85

db = SessionLocal()
source_cache: dict[str, Source] = {}
stats = {"rows": 0, "edges_created": 0, "skipped": 0}
skipped_rows = []

try:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if not row.get("target_name", "").strip():
            continue
        stats["rows"] += 1
        rtype = TYPE_MAP.get(row["relationship_type"])
        if rtype is None:
            stats["skipped"] += 1
            skipped_rows.append((row, f"unmapped relationship_type {row['relationship_type']!r}"))
            continue

        url = row["source_url"].strip()
        if url not in source_cache:
            existing = db.execute(select(Source).where(Source.url == url)).scalars().first()
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

        target = builder.get_or_create_person(db, row["target_name"], identity_text=row["target_context"])
        connection = builder.get_or_create_person(db, row["connection_name"], identity_text=row["connection_context"])
        if target is None or connection is None:
            stats["skipped"] += 1
            skipped_rows.append((row, "person resolution failed"))
            continue

        edge = ExtractedEdge(
            person_a=target.canonical_name, person_b=connection.canonical_name,
            relationship_type=rtype, method="curated_seed_import",
            evidence_snippet=row["evidence_note"].strip(), source_url=url,
            confidence_base=CONF, confidence_adjusted=CONF,
            signals=EdgeSignals(trusted=True), other_kind="person",
        )
        result = builder.add_edge_from_extraction(
            db, subject=target, edge=edge, depth=0, source=source, counterpart=connection)
        if result is not None:
            stats["edges_created"] += 1

    builder.commit_with_retry(db)
    print("=== IMPORT COMPLETE ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    if skipped_rows:
        print("\n--- skipped ---")
        for row, reason in skipped_rows:
            print(f"{row['target_name']} -> {row['connection_name']}: {reason}")
finally:
    db.close()
