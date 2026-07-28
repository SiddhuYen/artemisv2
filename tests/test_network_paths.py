"""Synthetic-graph tests for the hop-surcharge / fame-penalty / mega-hub-degree
cost shaping added to _best_path. Each test builds a tiny in-memory graph where
the "before" behavior (no shaping) and "after" behavior (with shaping) provably
differ, rather than asserting on real-world data.

Uses the shared `db` fixture from tests/conftest.py.
"""
from app import config
from app.models import Person, RelationshipEdge
from app.network.paths import _PublicEdges, _best_path


def _person(db, name, qid=None):
    p = Person(canonical_name=name, norm_name=name.lower(), wikidata_qid=qid)
    db.add(p)
    db.flush()
    return p


def _edge(db, a, b, rel="cofounder", status="strong", conf=0.9):
    e = RelationshipEdge(person_a_id=a.id, person_b_id=b.id,
                         relationship_type=rel, status=status, confidence_raw=conf)
    db.add(e)
    db.flush()
    return e


def test_hop_surcharge_prefers_direct_over_weaker_relay(db):
    a = _person(db, "A")
    bridge = _person(db, "Bridge")
    d = _person(db, "D")
    # Direct edge is individually weaker than either relay hop, but the relay
    # costs two hops. Without HOP_SURCHARGE the relay's lower raw edge cost
    # wins; with it, one direct (if weaker) ask should win instead.
    _edge(db, a, d, status="candidate", conf=0.5)
    _edge(db, a, bridge, status="strong", conf=0.9)
    _edge(db, bridge, d, status="strong", conf=0.9)
    db.commit()

    pe = _PublicEdges(db)

    # Sanity: without any hop surcharge/node penalty, the 2-hop relay is cheaper.
    old_relay_cost = -__import__("math").log(0.9) * 2
    old_direct_cost = -__import__("math").log(0.5) + 0.3  # candidate penalty
    assert old_relay_cost < old_direct_cost, "test setup must make the relay cheaper pre-surcharge"

    path = _best_path(pe, a.id, d.id)
    hop_ids = [pid for pid, _ in path]
    assert hop_ids == [a.id, d.id], (
        f"expected the direct A->D hop to win once HOP_SURCHARGE is applied, got {hop_ids}")


def test_fame_penalty_prefers_non_notable_bridge(db):
    a = _person(db, "A")
    famous = _person(db, "Famous Bridge", qid="Q123")
    plain = _person(db, "Plain Bridge")
    d = _person(db, "D")
    # Two routes, identical edge costs on both sides — only the fame penalty
    # (famous.wikidata_qid is set) should break the tie.
    _edge(db, a, famous, status="strong", conf=0.9)
    _edge(db, famous, d, status="strong", conf=0.9)
    _edge(db, a, plain, status="strong", conf=0.9)
    _edge(db, plain, d, status="strong", conf=0.9)
    db.commit()

    pe = _PublicEdges(db)
    path = _best_path(pe, a.id, d.id)
    hop_ids = [pid for pid, _ in path]
    assert plain.id in hop_ids and famous.id not in hop_ids, (
        f"expected the non-notable bridge to be preferred, got {hop_ids}")


def test_fame_penalty_reranks_not_censors(db):
    a = _person(db, "A")
    famous = _person(db, "Famous Bridge", qid="Q123")
    d = _person(db, "D")
    # The ONLY route runs through a notable person — it must still be returned.
    _edge(db, a, famous, status="strong", conf=0.9)
    _edge(db, famous, d, status="strong", conf=0.9)
    db.commit()

    pe = _PublicEdges(db)
    path = _best_path(pe, a.id, d.id)
    assert path is not None, "a famous-only bridge must still be returned, never censored"
    hop_ids = [pid for pid, _ in path]
    assert famous.id in hop_ids


def test_mega_hub_degree_penalty_prefers_lower_degree_bridge(db):
    a = _person(db, "A")
    hub = _person(db, "Hub Bridge")
    quiet = _person(db, "Quiet Bridge")
    d = _person(db, "D")
    _edge(db, a, hub, status="strong", conf=0.9)
    _edge(db, hub, d, status="strong", conf=0.9)
    _edge(db, a, quiet, status="strong", conf=0.9)
    _edge(db, quiet, d, status="strong", conf=0.9)
    # Inflate hub's degree well past MEGA_HUB_DEGREE with unrelated edges.
    extra_needed = config.MEGA_HUB_DEGREE + 5
    for i in range(extra_needed):
        filler = _person(db, f"Filler{i}")
        _edge(db, hub, filler, status="strong", conf=0.9)
    db.commit()

    pe = _PublicEdges(db)
    assert pe.degree[hub.id] > config.MEGA_HUB_DEGREE
    assert pe.degree[quiet.id] <= config.MEGA_HUB_DEGREE

    path = _best_path(pe, a.id, d.id)
    hop_ids = [pid for pid, _ in path]
    assert quiet.id in hop_ids and hub.id not in hop_ids, (
        f"expected the lower-degree bridge to be preferred over the mega-hub, got {hop_ids}")
