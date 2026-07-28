"""Contacts (.vcf) import -> My Connections.

The browser parses the vCard and posts the chosen cards as rows; those rows go
through `ingest_rows`, the same tail the CSV upload uses. What matters here is
that a phone contact lands as a LocalProfile connected to "You", with the same
de-duplication a re-uploaded CSV gets.
"""
from sqlalchemy import select

from app.models import LocalEdge, LocalProfile
from app.network.ingest import ingest_rows


def _rows(*contacts):
    """Shape contacts the way the UI posts them (see /network/contacts/import)."""
    return [{
        "Name": c.get("name", ""),
        "Company": c.get("company", ""),
        "Position": c.get("title", ""),
        "Email Address": c.get("email", ""),
        "Notes": c.get("notes", ""),
    } for c in contacts]


def test_contacts_become_profiles_linked_to_you(db):
    stats = ingest_rows(db, _rows(
        {"name": "Ann Lee", "company": "Sequoia Capital", "title": "Partner",
         "email": "ann@sequoia.test", "notes": "Phone: +1 (415) 555-0142"},
        {"name": "Bob Ray", "notes": "Phone: +1 (212) 555-0175"},
    ))
    assert stats["created"] == 2
    assert stats["skipped"] == 0

    ann = db.execute(
        select(LocalProfile).where(LocalProfile.canonical_name == "Ann Lee")
    ).scalar_one()
    assert ann.companies == ["Sequoia Capital"]
    assert ann.titles == ["Partner"]
    assert ann.email == "ann@sequoia.test"
    assert ann.notes == "Phone: +1 (415) 555-0142"

    # every imported contact is a direct connection of "You" (from_profile_id NULL)
    edges = db.execute(select(LocalEdge)).scalars().all()
    assert len(edges) == 2
    assert all(e.from_profile_id is None for e in edges)


def test_reimport_merges_instead_of_duplicating(db):
    ingest_rows(db, _rows({"name": "Ann Lee", "company": "Sequoia Capital"}))
    # same person, re-shared later with more filled in
    stats = ingest_rows(db, _rows(
        {"name": "Ann Lee", "company": "Benchmark", "title": "Partner",
         "email": "ann@sequoia.test"}))

    assert stats["created"] == 0
    assert stats["updated"] == 1
    assert db.query(LocalProfile).count() == 1
    ann = db.execute(select(LocalProfile)).scalar_one()
    assert ann.companies == ["Benchmark", "Sequoia Capital"]  # merged, sorted
    assert ann.email == "ann@sequoia.test"


def test_same_name_different_email_stays_separate(db):
    ingest_rows(db, _rows({"name": "Chris Smith", "email": "chris@a.test"}))
    stats = ingest_rows(db, _rows({"name": "Chris Smith", "email": "chris@b.test"}))
    assert stats["created"] == 1
    assert db.query(LocalProfile).count() == 2


def test_nameless_card_is_skipped_not_stored(db):
    stats = ingest_rows(db, _rows(
        {"name": "", "email": "ghost@nowhere.test"},
        {"name": "Zara Okafor"},
    ))
    assert stats["created"] == 1
    assert stats["skipped"] == 1
    assert db.query(LocalProfile).count() == 1


def test_duplicate_cards_within_one_import_collapse(db):
    stats = ingest_rows(db, _rows(
        {"name": "Mei Ng", "email": "mei@bw.test"},
        {"name": "Mei Ng", "email": "mei@bw.test", "title": "Analyst"},
    ))
    assert stats["created"] == 1
    assert stats["updated"] == 1
    assert db.query(LocalProfile).count() == 1
    assert db.query(LocalEdge).count() == 1  # one "You" edge, not two
