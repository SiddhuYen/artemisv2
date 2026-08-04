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

from app.graph import builder
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


def test_backfill_does_not_bridge_rows_that_record_no_owner(db):
    """An unowned row cannot support a first-degree claim, so by default it is
    skipped rather than shared. local_profiles holds several people's exports;
    reading all of it wrote 1,188 edges from one operator to another's contacts
    -- and linkedin_1st is traversable, so _route_exists then answers through
    people the operator has never met."""
    stats = ingest_rows(db, _rows({"name": "Fred Volinsky", "company": "Epiphany Biosciences"}))
    assert stats["graph_edges"] == 0

    assert backfill_graph_edges(db, "Abhimanyu Sharma") == 0
    assert _linkedin_edges(db) == []


def test_claim_unowned_bridges_them_and_records_the_claim(db):
    """The pre-owner_norm migration path, kept but made explicit: claiming
    stamps the rows, so the assertion is auditable and the NEXT operator to run
    this cannot make it again."""
    from app.models import LocalProfile
    ingest_rows(db, _rows({"name": "Fred Volinsky", "company": "Epiphany Biosciences"}))

    count = backfill_graph_edges(db, "Abhimanyu Sharma", claim_unowned=True)

    assert count == 1
    claimed = db.query(LocalProfile).filter(
        LocalProfile.canonical_name == "Fred Volinsky").one()
    assert claimed.owner_norm == person_norm_key("Abhimanyu Sharma")
    # a second operator now gets nothing -- the row is spoken for
    assert backfill_graph_edges(db, "Someone Else", claim_unowned=True) == 0


def test_backfill_bridges_the_owner_s_own_rows(db):
    stats = ingest_rows(db, _rows({"name": "Fred Volinsky", "company": "Epiphany Biosciences"}),
                        owner_name="Abhimanyu Sharma")
    assert stats["graph_edges"] == 1

    count = backfill_graph_edges(db, "Abhimanyu Sharma")

    assert count == 1
    edges = _linkedin_edges(db)
    assert len(edges) == 1
    contact = db.get(Person, edges[0].person_b_id)
    assert contact.norm_name == person_norm_key("Fred Volinsky")


def test_backfill_is_idempotent(db):
    ingest_rows(db, _rows({"name": "Fred Volinsky"}), owner_name="Abhimanyu Sharma")
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

    result = C.connect_people(db, "Abhimanyu Sharma", "Fred Volinsky", depth=1,
                              owner_name="Abhimanyu Sharma")

    assert result["connected"] is True
    assert result["hops"] == 1
    assert result["path"][-1]["relationship_from_previous"] == "linkedin_1st"


# ── uploaded connections are private to whoever uploaded them ──────────────
# relationship_edges has no owner column: ingest.backfill_graph_edges turns a
# local_profile into a linkedin_1st edge and the conversion DROPS the owner, so
# once written, an edge to somebody else's contact was byte-for-byte identical
# to an edge to your own. On a database holding two people's LinkedIn exports
# that put 1,132 strangers into one operator's first degree -- and because
# linkedin_1st is the strongest claim in the graph, those outranked real ties
# AND suppressed the search that would have replaced them.
#
# Ownership is recovered at read time from local_profiles, which is already
# correct, rather than backfilled onto the edges.

def _two_operators(db):
    ingest_rows(db, _rows({"name": "Mine Contact", "company": "Acme"}),
                owner_name="Aa Owner")
    ingest_rows(db, _rows({"name": "Theirs Contact", "company": "Acme"}),
                owner_name="Bb Other")
    backfill_graph_edges(db, "Aa Owner")
    backfill_graph_edges(db, "Bb Other")
    db.commit()


def test_an_operator_reaches_their_own_uploaded_contact(db):
    _two_operators(db)
    assert C._route_exists(db, "Aa Owner", "Mine Contact", 3, "Aa Owner")


def test_an_operator_cannot_reach_somebody_elses_uploaded_contact(db):
    """The requirement. Bb's export is not Aa's network, however it got into
    the same database."""
    _two_operators(db)
    assert not C._route_exists(db, "Aa Owner", "Theirs Contact", 3, "Aa Owner")


def test_an_unidentified_caller_gets_no_uploaded_contacts_at_all(db):
    """Fails closed. person_a is whoever was typed into the box and proves
    nothing about who is asking, so an absent owner_name cannot be read as
    'the origin must be the owner'."""
    _two_operators(db)
    assert not C._route_exists(db, "Aa Owner", "Mine Contact", 3)


def test_a_profile_nobody_owns_is_private_to_nobody(db):
    """Imported before ownership existed, so there is no one it can be
    attributed to -- exactly the 1,132 edges that started this."""
    ingest_rows(db, _rows({"name": "Orphan Contact", "company": "Acme"}))
    backfill_graph_edges(db, "Aa Owner", claim_unowned=False)
    db.commit()
    assert not C._route_exists(db, "Aa Owner", "Orphan Contact", 3, "Aa Owner")


def test_the_gate_only_applies_to_uploaded_connections(db):
    """A discovered coworker tie is sourced from a page anyone could read; it is
    not a claim about anyone's address book and must stay walkable."""
    _two_operators(db)
    a = builder.get_or_create_person(db, "Aa Owner")
    far = builder.get_or_create_person(db, "Public Person")
    db.add(RelationshipEdge(person_a_id=a.id, person_b_id=far.id,
                            relationship_type="coworker", status="strong",
                            confidence_raw=0.8, method="test",
                            evidence_snippet="ev", signals={}))
    db.commit()
    assert C._route_exists(db, "Aa Owner", "Public Person", 3)
