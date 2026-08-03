"""hop_verify: the "deferred Claude verification stage" -- judges whether an
edge's OWN evidence actually supports the claimed relationship, independent
of any path it's walked in. Runs at path-assembly time in connect.py, only
against hops in a route that's already been found.
"""
from app import config
from app.graph import connect as C
from app.graph import hop_verify
from app.models import Person, RelationshipEdge
from app.utils.names import person_norm_key


def _person(db, name, **kw):
    p = Person(canonical_name=name, norm_name=person_norm_key(name), **kw)
    db.add(p)
    db.flush()
    return p


def _edge(db, a, b, rel="coworker", status="candidate", conf=0.5, evidence="some evidence"):
    e = RelationshipEdge(person_a_id=a.id, person_b_id=b.id, relationship_type=rel,
                         status=status, confidence_raw=conf, evidence_snippet=evidence)
    db.add(e)
    db.commit()
    return e


# ---------------------------------------------------------------------------
# hop_verify.verify -- unit tests, Claude call mocked
# ---------------------------------------------------------------------------
def test_verify_keeps_the_edge_when_the_feature_is_off(db, monkeypatch):
    monkeypatch.setattr(config, "CLAUDE_VERIFY_HOPS", False)
    a, b = _person(db, "A"), _person(db, "B")
    e = _edge(db, a, b)
    assert hop_verify.verify(db, e, "A", "B") is True
    assert e.verified_status is None  # never touched -- feature didn't run


def test_verify_keeps_the_edge_when_claude_is_unavailable(db, monkeypatch):
    monkeypatch.setattr(config, "CLAUDE_VERIFY_HOPS", True)
    monkeypatch.setattr(hop_verify, "claude_available", lambda: False)
    a, b = _person(db, "A"), _person(db, "B")
    e = _edge(db, a, b)
    assert hop_verify.verify(db, e, "A", "B") is True
    assert e.verified_status is None


def test_verify_keeps_the_edge_and_does_not_cache_when_the_call_fails(db, monkeypatch):
    """A failed call must never look the same as a checked-and-approved edge
    -- otherwise a transient outage would freeze a real edge's fate."""
    monkeypatch.setattr(config, "CLAUDE_VERIFY_HOPS", True)
    monkeypatch.setattr(hop_verify, "claude_available", lambda: True)
    monkeypatch.setattr(hop_verify, "call_json", lambda *a, **k: None)
    a, b = _person(db, "A"), _person(db, "B")
    e = _edge(db, a, b)
    assert hop_verify.verify(db, e, "A", "B") is True
    assert e.verified_status is None


def test_verify_marks_a_genuine_edge_without_touching_status(db, monkeypatch):
    monkeypatch.setattr(config, "CLAUDE_VERIFY_HOPS", True)
    monkeypatch.setattr(hop_verify, "claude_available", lambda: True)
    monkeypatch.setattr(hop_verify, "call_json", lambda *a, **k: {
        "genuine": True, "reason": "Evidence directly names both as coauthors.",
    })
    a, b = _person(db, "A"), _person(db, "B")
    e = _edge(db, a, b, status="candidate")
    assert hop_verify.verify(db, e, "A", "B") is True
    assert e.verified_status == "genuine"
    assert e.verified_at is not None
    assert e.status == "candidate"  # confidence tier is a separate axis, untouched


def test_verify_rejects_and_sets_the_existing_exclusion_status(db, monkeypatch):
    """A rejected verdict must flip the SAME status field connect._untraversable
    and network.paths already exclude on -- not just its own private column --
    so every existing consumer benefits without changes elsewhere."""
    monkeypatch.setattr(config, "CLAUDE_VERIFY_HOPS", True)
    monkeypatch.setattr(hop_verify, "claude_available", lambda: True)
    monkeypatch.setattr(hop_verify, "call_json", lambda *a, **k: {
        "genuine": False, "reason": "Evidence is about a third party, not these two.",
    })
    a, b = _person(db, "A"), _person(db, "B")
    e = _edge(db, a, b, status="strong")
    assert hop_verify.verify(db, e, "A", "B") is False
    assert e.verified_status == "rejected"
    assert e.verified_reason
    assert e.status == "rejected"


def test_verify_reuses_a_fresh_cached_verdict_without_calling_claude_again(db, monkeypatch):
    monkeypatch.setattr(config, "CLAUDE_VERIFY_HOPS", True)
    monkeypatch.setattr(hop_verify, "claude_available", lambda: True)
    a, b = _person(db, "A"), _person(db, "B")
    e = _edge(db, a, b)
    e.verified_status = "genuine"
    e.verified_at = hop_verify._now_iso()
    db.commit()

    def _boom(*a, **k):
        raise AssertionError("should not call Claude again -- verdict is fresh")
    monkeypatch.setattr(hop_verify, "call_json", _boom)
    assert hop_verify.verify(db, e, "A", "B") is True


def test_verify_reverses_a_stale_rejection_back_to_candidate(db, monkeypatch):
    """Confidence tier can't be recovered once overwritten by rejection, so a
    reversed verdict restores a safe traversable middle tier rather than
    staying excluded despite the new 'genuine' verdict."""
    monkeypatch.setattr(config, "CLAUDE_VERIFY_HOPS", True)
    monkeypatch.setattr(hop_verify, "claude_available", lambda: True)
    a, b = _person(db, "A"), _person(db, "B")
    e = _edge(db, a, b, status="rejected")
    e.verified_status = "rejected"
    e.verified_at = "2000-01-01T00:00:00+00:00"  # far past any TTL
    db.commit()
    monkeypatch.setattr(hop_verify, "call_json", lambda *a, **k: {
        "genuine": True, "reason": "New evidence review supports the connection.",
    })
    assert hop_verify.verify(db, e, "A", "B") is True
    assert e.status == "candidate"


# ---------------------------------------------------------------------------
# connect.py integration -- _verified_routes / connect_people warnings
# ---------------------------------------------------------------------------
def test_verified_routes_drops_a_route_with_a_rejected_hop(db, monkeypatch):
    a, b, c = _person(db, "A"), _person(db, "B"), _person(db, "C")
    e1 = _edge(db, a, b)
    e2 = _edge(db, b, c)
    person_by_id = {a.id: a, b.id: b, c.id: c}
    routes = [[(a.id, None), (b.id, e1), (c.id, e2)]]

    def fake_verify(_db, edge, _a, _b):
        return edge.id != e2.id  # e2 fails, e1 passes
    monkeypatch.setattr(hop_verify, "verify", fake_verify)

    kept = C._verified_routes(db, routes, person_by_id)
    assert kept == []


def test_verified_routes_keeps_a_route_when_every_hop_passes(db, monkeypatch):
    a, b = _person(db, "A"), _person(db, "B")
    e1 = _edge(db, a, b)
    person_by_id = {a.id: a, b.id: b}
    routes = [[(a.id, None), (b.id, e1)]]

    monkeypatch.setattr(hop_verify, "verify", lambda *a, **k: True)

    kept = C._verified_routes(db, routes, person_by_id)
    assert kept == routes


def test_connect_people_reason_distinguishes_rejected_candidates_from_no_path(db, monkeypatch):
    """'try a higher depth' would be misleading when routes existed but none
    passed verification -- more expansion can't fix a hop that failed on its
    own evidence."""
    a = _person(db, "Alpha Person")
    b = _person(db, "Beta Person")
    db.commit()

    monkeypatch.setattr(C, "_route_exists", lambda *args, **kw: False)
    monkeypatch.setattr(C, "_direct_pair_search", lambda *args, **kw: (False, False))
    monkeypatch.setattr(C, "_expand_both_concurrently", lambda *args, **kw: {})
    monkeypatch.setattr(C, "_diverse_paths",
                        lambda *args, **kw: [[(a.id, None), (b.id, object())]])
    monkeypatch.setattr(hop_verify, "claude_available", lambda: True)
    monkeypatch.setattr(config, "CLAUDE_VERIFY_HOPS", True)
    monkeypatch.setattr(C, "_verified_routes", lambda *args, **kw: [])

    result = C.connect_people(db, "Alpha Person", "Beta Person")
    assert result["connected"] is False
    assert "verification" in result["reason"]
    assert "higher depth" not in result["reason"]
