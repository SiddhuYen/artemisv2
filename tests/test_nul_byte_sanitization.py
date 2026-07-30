"""Regression coverage for the Postgres NUL-byte crash.

SQLite (used by these tests, see conftest.py) tolerates embedded NUL bytes
in text columns without complaint, so it can't reproduce the ValueError
Postgres raises at flush time. What we CAN verify here, against any backend,
is that builder.py strips NULs from every string it persists before the row
ever reaches the database -- so the Postgres flush never sees one.
"""
from app.extraction.schemas import ExtractedEdge
from app.graph import builder
from app.providers.base import SearchResult

NUL = "\x00"


def test_get_or_create_person_strips_nul_from_name(db):
    person = builder.get_or_create_person(db, f"Abhi{NUL}manyu Sharma")
    assert person is not None
    assert NUL not in person.canonical_name
    assert all(NUL not in a for a in (person.aliases or []))


def test_get_or_create_org_strips_nul_from_name(db):
    org = builder.get_or_create_org(db, f"Panthe{NUL}on Prep")
    assert org is not None
    assert NUL not in org.name


def test_save_source_strips_nul_from_every_field(db):
    result = SearchResult(
        title=f"Abhimanyu Sharma{NUL} | Pantheon Prep",
        url=f"https://example.com/profile{NUL}",
        snippet=f"Episode 297{NUL} | Abhimanyu Sharma",
        provider="serper",
    )
    source = builder.save_source(
        db, result, query_used=f'"Abhimanyu Sharma"{NUL} Pantheon',
        full_text=f"full page text{NUL} with embedded junk",
    )
    assert NUL not in source.title
    assert NUL not in source.url
    assert NUL not in source.snippet
    assert NUL not in source.query_used
    assert NUL not in source.full_text
    # the dedup lookup must key off the SAME sanitized url, or a NUL-bearing
    # and NUL-free copy of the same page would double-save as two rows
    again = builder.save_source(db, result, query_used="anything")
    assert again.id == source.id


def test_save_source_strips_nul_when_backfilling_existing_row(db):
    """Regression: the update branch (an already-saved Source whose full_text
    was empty gets backfilled on a later call) is a SEPARATE code path from
    the insert branch above and was missed in the first pass of this fix --
    it hit the exact same Postgres ValueError via an UPDATE instead of an
    INSERT."""
    bare = SearchResult(title="Someone", url="https://example.com/x",
                        snippet="A snippet", provider="serper")
    first = builder.save_source(db, bare, query_used="q")
    assert first.full_text is None

    backfilled = builder.save_source(
        db, bare, query_used="q", full_text=f"late-arriving text{NUL} with junk")
    assert backfilled.id == first.id
    assert NUL not in backfilled.full_text


def test_add_edge_from_extraction_strips_nul_from_evidence(db):
    subject = builder.get_or_create_person(db, "Subject Person")
    counterpart = builder.get_or_create_person(db, "Counterpart Person")
    edge = ExtractedEdge(
        person_a=subject.canonical_name,
        person_b=counterpart.canonical_name,
        relationship_type="coauthor",
        method=f"llm{NUL}_extraction",
        evidence_snippet=f"Subject Person coauthor of{NUL} Counterpart Person.",
        other_kind="person",
    )
    row = builder.add_edge_from_extraction(db, subject, edge, depth=0,
                                           source=None, counterpart=counterpart)
    assert row is not None
    assert NUL not in row.method
    assert NUL not in row.evidence_snippet
