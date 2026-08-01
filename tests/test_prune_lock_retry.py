"""Retry-with-backoff for the bulk deletes in _prune_invalid_nodes.

Those deletes previously had zero resilience to a SQLite lock timeout --
confirmed live, repeatedly: /connect for a very public figure (Larry
Ellison) discovers hundreds of candidate organizations, Claude's entity
filter flags many as junk, and the resulting bulk DELETE FROM
relationship_edges is slow/large enough to still be contending with the
OTHER concurrent /connect side once SQLite's busy_timeout (5s, see
db.py's _tune_sqlite) gives up -- surfacing as a hard job failure instead
of a bounded, backed-off retry.
"""
from sqlalchemy.exc import OperationalError

from app.graph import builder
from app.models import Person, RelationshipEdge
from app.utils.names import person_norm_key


def _locked_error() -> OperationalError:
    return OperationalError("DELETE FROM relationship_edges WHERE ...", {},
                            Exception("database is locked"))


def _other_error() -> OperationalError:
    return OperationalError("DELETE FROM relationship_edges WHERE ...", {},
                            Exception("disk I/O error"))


def _person(db, name):
    p = Person(canonical_name=name, norm_name=person_norm_key(name))
    db.add(p)
    db.flush()
    return p


def _edge(db, a, b):
    e = RelationshipEdge(person_a_id=a.id, person_b_id=b.id,
                         relationship_type="coworker", status="candidate",
                         confidence_raw=0.5)
    db.add(e)
    db.flush()
    return e


def test_is_locked_matches_the_sqlite_lock_message():
    assert builder._is_locked(_locked_error()) is True
    assert builder._is_locked(_other_error()) is False


def test_delete_with_retry_succeeds_immediately_when_nothing_is_locked(db):
    a, b = _person(db, "Alpha"), _person(db, "Beta")
    _edge(db, a, b)
    db.commit()

    deleted = builder.delete_relationship_edges_with_retry(
        db, RelationshipEdge.person_a_id == a.id)

    assert deleted == 1
    assert db.query(RelationshipEdge).count() == 0


def test_delete_with_retry_recovers_from_transient_locks(db, monkeypatch):
    a, b = _person(db, "Alpha"), _person(db, "Beta")
    _edge(db, a, b)
    db.commit()

    real_query = db.query
    calls = {"n": 0}

    class _FlakyQuery:
        def __init__(self, inner):
            self._inner = inner

        def filter(self, *args, **kwargs):
            self._inner = self._inner.filter(*args, **kwargs)
            return self

        def delete(self, synchronize_session=False):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _locked_error()
            return self._inner.delete(synchronize_session=synchronize_session)

    monkeypatch.setattr(db, "query", lambda *a, **k: _FlakyQuery(real_query(*a, **k)))
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)  # skip real sleeps

    deleted = builder.delete_relationship_edges_with_retry(
        db, RelationshipEdge.person_a_id == a.id)

    assert calls["n"] == 3, "must have retried twice before succeeding on the third attempt"
    assert deleted == 1


def test_delete_with_retry_gives_up_after_exhausting_retries(db, monkeypatch):
    a, b = _person(db, "Alpha"), _person(db, "Beta")
    _edge(db, a, b)
    db.commit()

    class _AlwaysLockedQuery:
        def filter(self, *args, **kwargs):
            return self

        def delete(self, synchronize_session=False):
            raise _locked_error()

    monkeypatch.setattr(db, "query", lambda *a, **k: _AlwaysLockedQuery())
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    import pytest
    with pytest.raises(OperationalError):
        builder.delete_relationship_edges_with_retry(
            db, RelationshipEdge.person_a_id == a.id, _retries=3)


def test_delete_with_retry_reraises_a_non_lock_error_immediately(db, monkeypatch):
    a, b = _person(db, "Alpha"), _person(db, "Beta")
    _edge(db, a, b)
    db.commit()

    calls = {"n": 0}

    class _BrokenQuery:
        def filter(self, *args, **kwargs):
            return self

        def delete(self, synchronize_session=False):
            calls["n"] += 1
            raise _other_error()

    monkeypatch.setattr(db, "query", lambda *a, **k: _BrokenQuery())

    import pytest
    with pytest.raises(OperationalError):
        builder.delete_relationship_edges_with_retry(
            db, RelationshipEdge.person_a_id == a.id)

    assert calls["n"] == 1, "a non-lock error must not be retried at all"


def test_delete_with_retry_does_not_lose_a_sibling_edge_pending_in_the_same_transaction(db, monkeypatch):
    """The SAVEPOINT (not a bare retry) is the whole point: a failed-then-
    retried delete must not roll back OTHER work already added earlier in
    the same transaction."""
    a, b, c = _person(db, "Alpha"), _person(db, "Beta"), _person(db, "Gamma")
    _edge(db, a, b)          # will be deleted
    sibling = _edge(db, a, c)  # must survive the retry around the OTHER delete
    # deliberately NOT committed yet -- still pending in this same transaction,
    # exactly like _prune_invalid_nodes's own docstring describes.

    real_query = db.query
    calls = {"n": 0}

    class _FlakyQuery:
        def __init__(self, inner):
            self._inner = inner

        def filter(self, *args, **kwargs):
            self._inner = self._inner.filter(*args, **kwargs)
            return self

        def delete(self, synchronize_session=False):
            calls["n"] += 1
            if calls["n"] < 2:
                raise _locked_error()
            return self._inner.delete(synchronize_session=synchronize_session)

    monkeypatch.setattr(db, "query", lambda *a, **k: _FlakyQuery(real_query(*a, **k)))
    monkeypatch.setattr(builder, "_deadlock_backoff", lambda attempt: None)

    builder.delete_relationship_edges_with_retry(db, RelationshipEdge.person_b_id == b.id)
    db.commit()

    remaining = real_query(RelationshipEdge).all()
    assert [e.id for e in remaining] == [sibling.id], \
        "the sibling edge added earlier in the same transaction must survive the retry"
