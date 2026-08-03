"""A successful search must not be reported as an error.

Both bugs here have the same root: connect_people's Session holds a read
snapshot -- and identity-mapped objects -- from BEFORE the expansion it awaits,
while that expansion runs on other Sessions and ends by DELETING pruned nodes.

Bug 4 is the direct consequence: the final scoring read touches an instance
whose row is gone and raises ObjectDeletedError. It fires AFTER the stop
condition is met, so a route that was genuinely found comes back to the caller
as a hard failure.

Bug 2 is the same staleness reached through the cheap path: a direct-pair hit
short-circuits the expansion on the strength of a persisted edge, and the
pathfinder -- judging by the stricter _untraversable rule, and re-joining both
endpoints back to `people` -- then can't walk it, so the run answers "no path"
having skipped the work that would have found one.
"""
import pytest
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.graph import connect as C
from app.models import Person, RelationshipEdge
from app.utils.names import person_norm_key


@pytest.fixture
def db_isolated(tmp_path):
    """A Session with the DEPLOYED database's isolation, which the shared `db`
    fixture cannot provide.

    conftest's fixture uses StaticPool -- ONE connection shared by every
    Session -- so a "concurrent" writer's commit is visible to the main Session
    immediately and there is no stale snapshot to go stale. That is precisely
    what this bug is made of, so testing it there proves nothing (the same
    shape of masking the FK-violation test had to work around).

    A file-backed database in WAL mode gives each connection its own snapshot,
    which is how Postgres behaves and how the deployment actually runs.
    """
    from sqlalchemy import create_engine, event

    url = f"sqlite:///{tmp_path}/iso.db"
    engine = create_engine(url, future=True)

    @event.listens_for(engine, "connect")
    def _wal(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False,
                           expire_on_commit=False, future=True)
    session = Session()
    session.info["factory"] = Session
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _person(db, name):
    p = Person(canonical_name=name, norm_name=person_norm_key(name))
    db.add(p)
    db.flush()
    return p


def _edge(db, a, b, **kw):
    kw.setdefault("relationship_type", "coworker")
    kw.setdefault("status", "strong")
    kw.setdefault("confidence_raw", 0.8)
    kw.setdefault("evidence_snippet", "A and B worked together.")
    kw.setdefault("signals", {"sentence_cooccurrence": True})
    e = RelationshipEdge(person_a_id=a.id, person_b_id=b.id, **kw)
    db.add(e)
    db.commit()
    return e


def _other_session(db):
    """A second Session on the SAME engine — how the expansion's workers and
    the prune actually reach this database. On the isolated fixture this gets
    its own CONNECTION, and therefore its own snapshot, which is the point."""
    factory = db.info.get("factory")
    if factory is not None:
        return factory()
    return sessionmaker(bind=db.get_bind(), autoflush=False,
                        expire_on_commit=False, future=True)()


# ---------------------------------------------------------------------------
# The pre-scoring session reset (see connect_people)
#
# Scoped honestly. ObjectDeletedError was seen live at this point, but neither
# candidate mechanism reproduces -- a "stale snapshot" cannot exist at READ
# COMMITTED (test_a_session_is_not_reading_a_stale_snapshot proves the
# opposite), and expire-then-delete did not raise on attribute access. So these
# pin what the reset actually guarantees, and nothing more.
# ---------------------------------------------------------------------------
def test_a_session_is_not_reading_a_stale_snapshot(db_isolated):
    """Documents WHY the 'stale pre-cleanup snapshot' explanation is wrong, so
    nobody re-derives it. Both backends are READ COMMITTED: this Session sees
    another Session's committed write on its next statement, with no rollback."""
    db = db_isolated
    _person(db, "Ada End")
    db.commit()
    db.query(Person).all()          # open this Session's transaction
    assert db.in_transaction()

    worker = _other_session(db)
    try:
        worker.add(Person(canonical_name="Late Arrival",
                          norm_name=person_norm_key("Late Arrival")))
        worker.commit()
    finally:
        worker.close()

    assert db.query(Person).filter(
        Person.canonical_name == "Late Arrival").count() == 1, \
        "no rollback was needed to see it -- there is no snapshot to go stale"


def test_the_scoring_pass_runs_on_a_clean_session(db_isolated, monkeypatch):
    """What the reset does guarantee: whatever the expansion left in this
    Session's identity map is not what the answer is built from."""
    db = db_isolated
    a, b = _person(db, "Ada End"), _person(db, "Bo End")
    junk = _person(db, "Junk Node - LinkedIn | Top Voice")
    _edge(db, a, b)
    db.query(Person).all()
    assert junk in db, "precondition: the stale instance is held"

    def fake_expand(*args, **kwargs):
        worker = _other_session(db)
        try:
            worker.query(Person).filter(Person.id == junk.id).delete()
            worker.commit()
        finally:
            worker.close()
        return {}

    monkeypatch.setattr(C, "_route_exists", lambda *a, **k: False)
    monkeypatch.setattr(C, "_direct_pair_search", lambda *a, **k: (False, False))
    monkeypatch.setattr(C, "_expand_both_concurrently", fake_expand)
    monkeypatch.setattr(C, "_verified_routes", lambda db_, routes, *a, **k: routes)

    result = C.connect_people(db, "Ada End", "Bo End", depth=1)

    assert result["connected"] is True
    assert junk not in db, "the pruned node must not still be identity-mapped"


# ---------------------------------------------------------------------------
# Bug 2 — the direct-pair short-circuit must use the pathfinder's own rule
# ---------------------------------------------------------------------------
def _direct_hit_that_is_not_walkable(db, a, b):
    """What _direct_pair_search persists in the failing case: an edge the
    pathfinder rejects. Untyped with neither cooccurrence nor an explicit
    keyword is bare co-presence on one page -- _untraversable drops it."""
    _edge(db, a, b, relationship_type="unknown", status="weak",
          confidence_raw=0.1, signals={})


def test_a_confident_hit_the_pathfinder_cannot_walk_does_not_skip_expansion(db, monkeypatch):
    a, b = _person(db, "Paul Graham"), _person(db, "Sam Altman")
    _direct_hit_that_is_not_walkable(db, a, b)

    expanded = {"ran": False}

    def fake_expand(*args, **kwargs):
        expanded["ran"] = True
        return {}

    # `confident` is True — the evidence looked good — but the persisted edge
    # is one _untraversable rejects, which is the whole failure mode.
    monkeypatch.setattr(C, "_direct_pair_search", lambda *a, **k: (True, True))
    monkeypatch.setattr(C, "_expand_both_concurrently", fake_expand)
    monkeypatch.setattr(C, "_verified_routes", lambda db_, routes, *a, **k: routes)

    lines = []
    C.connect_people(db, "Paul Graham", "Sam Altman", depth=1, progress=lines.append)

    assert expanded["ran"] is True, \
        "must not skip the expansion for a route the pathfinder can't walk"
    assert any("can't walk it" in ln for ln in lines)


def test_a_confident_walkable_hit_still_short_circuits(db, monkeypatch):
    """The optimisation must survive the fix -- a real direct hit should still
    skip the expensive expansion."""
    a, b = _person(db, "Paul Graham"), _person(db, "Sam Altman")
    _edge(db, a, b)   # typed, strong, with cooccurrence: walkable

    expanded = {"ran": False}

    def fake_expand(*args, **kwargs):
        expanded["ran"] = True
        return {}

    monkeypatch.setattr(C, "_direct_pair_search", lambda *a, **k: (True, True))
    monkeypatch.setattr(C, "_expand_both_concurrently", fake_expand)
    monkeypatch.setattr(C, "_verified_routes", lambda db_, routes, *a, **k: routes)

    result = C.connect_people(db, "Paul Graham", "Sam Altman", depth=1)

    assert expanded["ran"] is False, "a walkable direct hit should still short-circuit"
    assert result["connected"] is True


def test_a_dangling_edge_whose_endpoint_was_pruned_is_not_a_route(db, monkeypatch):
    """The case `confident` alone cannot see: the evidence was fine, but a
    concurrent prune deleted an endpoint, so the edge no longer connects two
    people that exist. _route_exists joins both ends back to `people`."""
    a, b = _person(db, "Paul Graham"), _person(db, "Sam Altman")
    db.commit()   # NO edge yet, so stage 0 finds no route and we reach the
                  # direct-pair branch this test is actually about

    def direct_then_prune(*args, **kwargs):
        """Persist a good direct edge, then have a concurrent prune delete one
        of its endpoints -- which is what leaves the edge dangling."""
        worker = _other_session(db)
        try:
            worker.add(RelationshipEdge(
                person_a_id=a.id, person_b_id=b.id, relationship_type="coworker",
                status="strong", confidence_raw=0.8,
                evidence_snippet="Paul Graham appointed him as president.",
                signals={"sentence_cooccurrence": True}))
            worker.query(Person).filter(Person.id == b.id).delete()
            worker.commit()
        finally:
            worker.close()
        return (True, True)

    expanded = {"ran": False}

    def fake_expand(*args, **kwargs):
        expanded["ran"] = True
        return {}

    monkeypatch.setattr(C, "_direct_pair_search", direct_then_prune)
    monkeypatch.setattr(C, "_expand_both_concurrently", fake_expand)
    monkeypatch.setattr(C, "_verified_routes", lambda db_, routes, *a, **k: routes)

    C.connect_people(db, "Paul Graham", "Sam Altman", depth=1)
    assert expanded["ran"] is True, \
        "a dangling edge must not be mistaken for a walkable route"
