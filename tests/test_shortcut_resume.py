"""The "already connected" shortcut has to be self-correcting.

Every cheap check in connect_people -- the opening _route_exists, and the
re-check after the origin's own enrichment -- skips the entire paid walk on the
strength of edges that nothing has inspected yet. Hop verification inspects
them afterwards. When it rejects all of them, the skip was made on a false
premise, and the walk had already given up: it returned "no connection" having
never searched.

Observed on Sanjay Ghemawat -> Larry Page. A 0.39-confidence heuristic edge
(Ghemawat -> Eric Schmidt, extracted from a PDF whose evidence sentence names
Schmidt, Page and Brin but never Ghemawat) made a two-hop route appear to
exist. The walk was skipped, verification correctly rejected the edge, and the
caller got "no connection" for $0.0007 and zero searches -- the wrong answer,
reached by not looking.
"""
import pytest

from app import config
from app.graph import connect as C
from app.models import Person, RelationshipEdge
from app.utils.names import person_norm_key


def _person(db, name):
    p = Person(canonical_name=name, norm_name=person_norm_key(name))
    db.add(p)
    db.flush()
    return p


def _edge(db, a, b, rel="coworker", status="strong", conf=0.7):
    db.add(RelationshipEdge(person_a_id=a.id, person_b_id=b.id,
                            relationship_type=rel, status=status,
                            confidence_raw=conf, method="test",
                            evidence_snippet="ev",
                            signals={"sentence_cooccurrence": True}))


@pytest.fixture
def two_hop_graph(db):
    """A -- M -- B, where the A--M hop is the one verification will reject."""
    a, m, b = _person(db, "Aay End"), _person(db, "Emm Middle"), _person(db, "Bee End")
    _edge(db, a, m)
    _edge(db, m, b)
    db.commit()
    return a, m, b


@pytest.fixture(autouse=True)
def _no_paid_stages(monkeypatch):
    """Neither paid stage does anything unless a test says so. Each test
    asserts on whether they were CALLED, which is the behavior at issue."""
    monkeypatch.setattr(C, "_ensure_origin_enriched", lambda *a, **k: {})
    monkeypatch.setattr(C, "_direct_pair_search", lambda *a, **k: (False, False))
    monkeypatch.setattr(C, "_expand_both_concurrently", lambda *a, **k: {})


def _reject_everything(monkeypatch):
    """Hop verification that throws out every candidate route."""
    monkeypatch.setattr(C.config, "CLAUDE_VERIFY_HOPS", True)
    monkeypatch.setattr(C.hop_verify, "claude_available", lambda: True)
    monkeypatch.setattr(C, "_verified_routes", lambda *a, **k: [])


def test_rejected_shortcut_resumes_both_skipped_stages(db, two_hop_graph, monkeypatch):
    """The regression: a shortcut disproved by verification must send the walk
    back to the search it skipped, not end it."""
    _reject_everything(monkeypatch)
    calls = []
    monkeypatch.setattr(C, "_direct_pair_search",
                        lambda *a, **k: (calls.append("direct"), (False, False))[1])
    monkeypatch.setattr(C, "_expand_both_concurrently",
                        lambda *a, **k: (calls.append("expand"), {})[1])

    out = C.connect_people(db, "Aay End", "Bee End", depth=2)

    assert out["connected"] is False
    assert calls == ["direct", "expand"], (
        "a shortcut disproved by verification must resume BOTH stages it skipped")


def test_resume_runs_at_most_once(db, two_hop_graph, monkeypatch):
    """Verification persists its rejections, so a second pass cannot re-propose
    the disproved route -- but the guard must not depend on that alone, or a
    graph holding several bogus routes would loop the walk once per route."""
    _reject_everything(monkeypatch)
    expands = []
    monkeypatch.setattr(C, "_expand_both_concurrently",
                        lambda *a, **k: (expands.append(1), {})[1])

    C.connect_people(db, "Aay End", "Bee End", depth=2)

    assert len(expands) == 1


def test_no_resume_when_expansion_already_ran(db, monkeypatch):
    """If the walk DID search and its own edges were then rejected, repeating
    it would re-read the same pages for the same verdict. Only a walk that was
    skipped is worth resuming."""
    a, b = _person(db, "Cee End"), _person(db, "Dee End")
    _edge(db, a, b)
    db.commit()
    _reject_everything(monkeypatch)
    # Nothing pre-existing connects them, so the opening _route_exists is False
    # and expansion runs for real the first time.
    monkeypatch.setattr(C, "_route_exists", lambda *a, **k: False)
    expands = []
    monkeypatch.setattr(C, "_expand_both_concurrently",
                        lambda *a, **k: (expands.append(1), {})[1])

    C.connect_people(db, "Cee End", "Dee End", depth=2)

    assert len(expands) == 1, "must not expand twice when the first pass searched"


def test_no_resume_when_verification_is_off(db, two_hop_graph, monkeypatch):
    """With verification disabled nothing disproves the shortcut, so the route
    stands and the walk stays free -- the whole point of the shortcut."""
    monkeypatch.setattr(C.config, "CLAUDE_VERIFY_HOPS", False)
    expands = []
    monkeypatch.setattr(C, "_expand_both_concurrently",
                        lambda *a, **k: (expands.append(1), {})[1])

    out = C.connect_people(db, "Aay End", "Bee End", depth=2)

    assert out["connected"] is True
    assert expands == []


def test_reason_reports_the_real_candidate_count(db, two_hop_graph, monkeypatch):
    """It used to interpolate config.CONNECT_MAX_PATHS -- a constant -- so the
    message said "3 candidate route(s) found" no matter how many there were,
    including when there was one."""
    _reject_everything(monkeypatch)

    out = C.connect_people(db, "Aay End", "Bee End", depth=2)

    assert out["connected"] is False
    assert out["reason"].startswith("1 candidate route(s)"), out["reason"]
    # Guards the regression specifically: the constant it used to print is 3.
    assert config.CONNECT_MAX_PATHS != 1

