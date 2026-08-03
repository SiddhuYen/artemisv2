"""Regression tests for a real live bug: pruning a "junk" organization could
hit a genuine ForeignKeyViolation on Postgres. The two /connect sides run
concurrently, each in its own Session/connection, writing into the same
shared graph -- if side A decides an org is junk, deletes its edges, then
side B independently inserts a brand-new edge pointing at that SAME org
before side A's own node-delete runs, Postgres's real FK constraint
correctly rejects side A's delete. Confirmed live: a whole /connect job died
this way, over ONE contested organization out of a larger batch -- the
batched `DELETE FROM organizations WHERE id IN (...)` covering every
candidate failed as a single unit, taking every OTHER legitimately-junk org
down with it.

SQLite never enforces this constraint at all by default (PRAGMA foreign_keys
is never turned on for the app itself -- see db.py), so the identical race
was silently corrupting the graph there instead of ever raising anything.

Why these tests mock the exception instead of reproducing two real
concurrent writers: SQLite serializes ALL writers on a single whole-database
lock, so two genuinely overlapping open write transactions on the same file
are structurally impossible here -- confirmed empirically while writing this
file: a second, truly independent connection trying to commit while the
first's SAVEPOINT was still open just deadlocked on "database is locked"
instead of racing anything. That's not a test bug; it's exactly WHY
Postgres's much finer-grained MVCC locking is what let the real bug happen
in the first place, and exactly why SQLite silently never raised it. So,
same technique tests/test_prune_lock_retry.py already established for the
analogous SQLite-lock case: mock the DB-engine-level exception directly
(sqlalchemy.exc.IntegrityError, the real class a Postgres FK violation
raises through this ORM) rather than trying to force SQLite into a
concurrency mode it doesn't support.
"""
import pytest
from sqlalchemy.exc import IntegrityError

from app.graph import builder, expansion
from app.models import Organization, Person, RelationshipEdge
from app.utils.names import org_norm_key, person_norm_key


def _fk_violation() -> IntegrityError:
    return IntegrityError(
        "DELETE FROM organizations WHERE organizations.id = ?", {},
        Exception('FOREIGN KEY constraint failed'))


def _person(db, name: str) -> Person:
    p = Person(canonical_name=name, norm_name=person_norm_key(name))
    db.add(p)
    db.flush()
    return p


def _org(db, name: str) -> Organization:
    o = Organization(name=name, norm_name=org_norm_key(name))
    db.add(o)
    db.flush()
    return o


def _edge_to_org(db, person: Person, org: Organization) -> RelationshipEdge:
    e = RelationshipEdge(person_a_id=person.id, organization_id=org.id,
                         relationship_type="employee", status="candidate",
                         confidence_raw=0.5)
    db.add(e)
    db.flush()
    return e


def test_delete_node_with_retry_succeeds_immediately_when_nothing_contests_it(db):
    subject = _person(db, "Subject Person")
    org = _org(db, "Junk Org")
    _edge_to_org(db, subject, org)
    db.commit()

    deleted = builder.delete_node_with_retry(
        db, org, RelationshipEdge.organization_id == org.id)

    assert deleted is True
    assert db.query(RelationshipEdge).count() == 0
    # Not just "gone from the database" -- session.get() must not keep
    # returning a stale, already-deleted object from the identity map (the
    # bulk delete uses synchronize_session=False, which never updates it on
    # its own -- see delete_node_with_retry's own docstring on why the
    # explicit expunge on success is required).
    assert db.get(Organization, org.id) is None


def test_delete_node_with_retry_recovers_after_a_transient_fk_violation(db, monkeypatch):
    """Simulates the race resolving itself on retry: the underlying delete
    raises a real FK violation (IntegrityError) on the first attempt only --
    a stand-in for the OTHER /connect side's edge having landed in the gap
    -- and cleanly succeeds from the second attempt onward, same as it would
    once that one racing edge is no longer being freshly reintroduced."""
    subject = _person(db, "Subject Person")
    org = _org(db, "Junk Org")
    _edge_to_org(db, subject, org)
    db.commit()

    real_query = db.query
    calls = {"n": 0}

    class _FlakyOrgQuery:
        def __init__(self, inner, model):
            self._inner = inner
            self._model = model

        def filter(self, *args, **kwargs):
            self._inner = self._inner.filter(*args, **kwargs)
            return self

        def delete(self, synchronize_session=False):
            if self._model is Organization:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise _fk_violation()
            return self._inner.delete(synchronize_session=synchronize_session)

    monkeypatch.setattr(db, "query",
                        lambda model, *a, **k: _FlakyOrgQuery(real_query(model, *a, **k), model))
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)  # skip real sleeps

    deleted = builder.delete_node_with_retry(
        db, org, RelationshipEdge.organization_id == org.id)

    assert deleted is True, "the retry must win once the violation stops recurring"
    assert calls["n"] == 2, "must have hit the FK violation once, then retried"
    assert db.get(Organization, org.id) is None


def test_delete_node_with_retry_redoes_the_edge_delete_on_every_attempt(db, monkeypatch):
    """The fix's actual point: edges and the node are deleted in the SAME
    savepoint, so a retry redoes BOTH -- not just the node-delete -- which
    is what actually clears a freshly-raced-in edge before trying the
    node-delete again. Simulated here by having a NEW edge reappear after
    each edge-delete (standing in for the race), and confirming the
    edge-delete step really runs again on every attempt, not just once."""
    subject = _person(db, "Subject Person")
    org = _org(db, "Junk Org")
    edge_calls = {"n": 0}

    class _ReappearingEdgeQuery:
        def __init__(self, inner, model):
            self._inner = inner
            self._model = model

        def filter(self, *args, **kwargs):
            self._inner = self._inner.filter(*args, **kwargs)
            return self

        def delete(self, synchronize_session=False):
            if self._model is RelationshipEdge:
                edge_calls["n"] += 1
                result = self._inner.delete(synchronize_session=synchronize_session)
                if edge_calls["n"] == 1:
                    # the "race": a fresh edge lands right after this
                    # attempt's edge-delete ran, same shape as the OTHER
                    # /connect side's independent insert.
                    _edge_to_org(db, subject, org)
                return result
            if self._model is Organization:
                # only succeeds once nothing references the org anymore --
                # real_query (the unwrapped Session.query), not db.query,
                # since that's monkeypatched to return THIS same wrapper
                # class, which has no .count().
                still_referenced = real_query(RelationshipEdge).filter(
                    RelationshipEdge.organization_id == org.id).count() > 0
                if still_referenced:
                    raise _fk_violation()
            return self._inner.delete(synchronize_session=synchronize_session)

    real_query = db.query
    monkeypatch.setattr(
        db, "query",
        lambda model, *a, **k: _ReappearingEdgeQuery(real_query(model, *a, **k), model))
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    deleted = builder.delete_node_with_retry(
        db, org, RelationshipEdge.organization_id == org.id)

    assert deleted is True
    assert edge_calls["n"] == 2, "the edge-delete must run again on the retry, not just once"
    assert real_query(RelationshipEdge).count() == 0
    assert db.get(Organization, org.id) is None


def test_delete_node_with_retry_gives_up_gracefully_without_raising(db, monkeypatch):
    """If the violation never stops recurring, delete_node_with_retry must
    give up quietly after its bounded retries -- not raise and not corrupt
    anything -- so ONE permanently contested node can't fail the whole
    prune pass (see the _prune_invalid_nodes-level test below)."""
    subject = _person(db, "Subject Person")
    org = _org(db, "Junk Org")
    _edge_to_org(db, subject, org)
    db.commit()

    class _AlwaysContestedQuery:
        def __init__(self, inner, model):
            self._inner = inner
            self._model = model

        def filter(self, *args, **kwargs):
            self._inner = self._inner.filter(*args, **kwargs)
            return self

        def delete(self, synchronize_session=False):
            if self._model is Organization:
                raise _fk_violation()
            return self._inner.delete(synchronize_session=synchronize_session)

    real_query = db.query
    monkeypatch.setattr(
        db, "query",
        lambda model, *a, **k: _AlwaysContestedQuery(real_query(model, *a, **k), model))
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    deleted = builder.delete_node_with_retry(
        db, org, RelationshipEdge.organization_id == org.id, _retries=3)

    assert deleted is False
    # The org survives (never deleted) -- exactly the "left alone for a
    # future prune pass" outcome, not a crash and not data corruption.
    assert db.get(Organization, org.id) is not None


def test_prune_invalid_nodes_survives_one_permanently_contested_org(db, monkeypatch):
    """Integration-level proof that the batch is no longer all-or-nothing:
    one org that can never win the race must not prevent OTHER, genuinely
    junk nodes (a different org, and an unrelated junk person) from being
    cleaned up in the SAME call -- this is what the OLD single batched
    `DELETE FROM organizations WHERE id IN (...)` couldn't do (one contested
    id failed the whole statement, taking the rest of the batch down with
    it)."""
    subject = _person(db, "Subject Person")
    cursed_org = _org(db, "Cursed Org")
    clean_junk_org = _org(db, "Clean Junk Org")
    junk_person = _person(db, "Some Page - LinkedIn")  # fails the name-shape check
    _edge_to_org(db, subject, cursed_org)
    _edge_to_org(db, subject, clean_junk_org)
    db.commit()

    monkeypatch.setattr(expansion, "is_filtering_active", lambda: True)
    monkeypatch.setattr(expansion, "filter_entities", lambda names, kind: [])  # nothing valid -> both orgs are junk
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    class _CursedOrgQuery:
        def __init__(self, inner, model):
            self._inner = inner
            self._model = model

        def filter(self, *args, **kwargs):
            self._inner = self._inner.filter(*args, **kwargs)
            return self

        def delete(self, synchronize_session=False):
            if self._model is Organization:
                compiled = str(self._inner.statement.compile(
                    compile_kwargs={"literal_binds": True}))
                if cursed_org.id in compiled:
                    raise _fk_violation()
            return self._inner.delete(synchronize_session=synchronize_session)

    real_query = db.query
    monkeypatch.setattr(
        db, "query",
        lambda model, *a, **k: _CursedOrgQuery(real_query(model, *a, **k), model))

    removed = expansion._prune_invalid_nodes(db, protected_norms=set())

    # The cursed org survives -- permanently contested, left for later.
    assert db.get(Organization, cursed_org.id) is not None
    # But the clean junk org AND the unrelated junk person were still
    # removed in the SAME call -- the whole batch didn't die with it.
    assert db.get(Organization, clean_junk_org.id) is None
    assert db.get(Person, junk_person.id) is None
    assert removed == 2
