"""Bridging uploaded LinkedIn contacts into the shared public graph.

Uploaded contacts used to be invisible to /connect and /discover: the
`owner_name` parameter that turns a LocalProfile into a real linkedin_1st
RelationshipEdge existed on the backend, but the frontend never sent it, so
zero such edges ever got created. Three things are covered here:
  1. ingest_rows(..., owner_name=...) creates the edge immediately.
  2. backfill_graph_edges retroactively bridges profiles imported without
     owner_name (i.e. every contact imported before this fix), and is
     idempotent.
  3. connect_people checks the existing graph FIRST, before any live search,
     so a bridged (or otherwise already-known) connection costs nothing to
     rediscover.
"""
from sqlalchemy import select

from app.graph import connect as C
from app.models import Person, RelationshipEdge
from app.network.ingest import backfill_graph_edges, ingest_rows
from app.utils.names import person_norm_key


def _rows(*contacts):
    return [{
        "Name": c.get("name", ""),
        "Company": c.get("company", ""),
        "URL": c.get("url", ""),
    } for c in contacts]


def _linkedin_edges(db):
    return db.execute(
        select(RelationshipEdge).where(RelationshipEdge.relationship_type == "linkedin_1st")
    ).scalars().all()


def test_ingest_with_owner_name_creates_graph_edge(db):
    stats = ingest_rows(db, _rows({"name": "Fred Volinsky", "company": "Epiphany Biosciences"}),
                        owner_name="Abhimanyu Sharma")
    assert stats["graph_edges"] == 1
    edges = _linkedin_edges(db)
    assert len(edges) == 1
    owner = db.get(Person, edges[0].person_a_id)
    contact = db.get(Person, edges[0].person_b_id)
    assert owner.norm_name == person_norm_key("Abhimanyu Sharma")
    assert contact.norm_name == person_norm_key("Fred Volinsky")


def test_ingest_without_owner_name_creates_no_graph_edge(db):
    """Today's default (no owner_name) -- ingestion stays scoped to
    LocalProfile only, same as before this fix."""
    stats = ingest_rows(db, _rows({"name": "Fred Volinsky"}))
    assert stats["graph_edges"] == 0
    assert _linkedin_edges(db) == []


def test_backfill_bridges_profiles_imported_without_owner_name(db):
    stats = ingest_rows(db, _rows({"name": "Fred Volinsky", "company": "Epiphany Biosciences"}))
    assert stats["graph_edges"] == 0
    assert _linkedin_edges(db) == []

    count = backfill_graph_edges(db, "Abhimanyu Sharma")

    assert count == 1
    edges = _linkedin_edges(db)
    assert len(edges) == 1
    contact = db.get(Person, edges[0].person_b_id)
    assert contact.norm_name == person_norm_key("Fred Volinsky")


def test_backfill_is_idempotent(db):
    ingest_rows(db, _rows({"name": "Fred Volinsky"}))
    backfill_graph_edges(db, "Abhimanyu Sharma")
    backfill_graph_edges(db, "Abhimanyu Sharma")
    assert len(_linkedin_edges(db)) == 1


def test_backfill_with_blank_owner_name_is_a_noop(db):
    ingest_rows(db, _rows({"name": "Fred Volinsky"}))
    assert backfill_graph_edges(db, "") == 0
    assert backfill_graph_edges(db, "   ") == 0
    assert _linkedin_edges(db) == []


def test_connect_finds_a_bridged_edge_without_touching_search(db, monkeypatch):
    """The core of 'go through uploaded connections first': once a
    linkedin_1st edge exists, connect_people must resolve it via the
    zero-cost _route_exists check alone -- neither the live direct-pair
    search nor the full neighborhood expansion should ever run."""
    ingest_rows(db, _rows({"name": "Fred Volinsky", "company": "Epiphany Biosciences"}),
               owner_name="Abhimanyu Sharma")

    def _boom(*a, **k):
        raise AssertionError("should not touch the live web when already connected")

    monkeypatch.setattr(C, "_direct_pair_search", _boom)
    monkeypatch.setattr(C, "_expand_both_concurrently", _boom)

    result = C.connect_people(db, "Abhimanyu Sharma", "Fred Volinsky", depth=1)

    assert result["connected"] is True
    assert result["hops"] == 1
    assert result["path"][-1]["relationship_from_previous"] == "linkedin_1st"
