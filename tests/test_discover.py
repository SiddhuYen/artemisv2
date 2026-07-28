"""Tests for discover_person ("/discover") -- expand one person's own public
network and rank everyone reachable, warmest/closest first. Unlike
connect_people (bridging toward a specific target), every reachable node here
is a suggestion, so the fame/hub penalty applies to listed nodes too, not just
nodes routed through.
"""
from app.graph import connect as C
from app.models import Person, RelationshipEdge
from app.utils.names import person_norm_key


def _person(db, name, **kw):
    p = Person(canonical_name=name, norm_name=person_norm_key(name), **kw)
    db.add(p)
    db.flush()
    return p


def _edge(db, a, b, rel="coworker", status="candidate", conf=0.5):
    e = RelationshipEdge(person_a_id=a.id, person_b_id=b.id,
                         relationship_type=rel, status=status, confidence_raw=conf)
    db.add(e)
    return e


def test_discover_reports_not_found_for_an_unknown_person(db):
    result = C.discover_person(db, "Nobody Here")
    assert result["found"] is False


def test_discover_lists_direct_and_second_hop_connections(db):
    root = _person(db, "Root Person")
    direct = _person(db, "Direct Person")
    indirect = _person(db, "Indirect Person")
    _edge(db, root, direct, rel="cofounder", conf=0.9)
    _edge(db, direct, indirect, rel="coworker", conf=0.6)
    db.commit()

    result = C.discover_person(db, "Root Person", depth=2)
    assert result["found"] is True
    assert result["person"] == "Root Person"
    labels = {c["label"]: c for c in result["connections"]}
    assert labels["Direct Person"]["hops"] == 1
    assert labels["Direct Person"]["relationship_type"] == "cofounder"
    assert labels["Indirect Person"]["hops"] == 2


def test_discover_respects_the_hop_depth_cap(db):
    root = _person(db, "Root Person")
    direct = _person(db, "Direct Person")
    indirect = _person(db, "Indirect Person")
    _edge(db, root, direct)
    _edge(db, direct, indirect)
    db.commit()

    result = C.discover_person(db, "Root Person", depth=1)
    labels = {c["label"] for c in result["connections"]}
    assert "Direct Person" in labels
    assert "Indirect Person" not in labels


def test_discover_excludes_the_root_person_from_their_own_results(db):
    root = _person(db, "Root Person")
    direct = _person(db, "Direct Person")
    _edge(db, root, direct)
    db.commit()

    result = C.discover_person(db, "Root Person", depth=2)
    assert all(c["label"] != "Root Person" for c in result["connections"])


def test_discover_ranks_closer_hops_before_farther_ones(db):
    root = _person(db, "Root Person")
    close = _person(db, "Close Person")
    mid = _person(db, "Mid Person")
    far = _person(db, "Far Person")
    _edge(db, root, close, conf=0.2)   # weakly evidenced but 1 hop
    _edge(db, root, mid, conf=0.9)     # strongly evidenced but also 1 hop
    _edge(db, mid, far, conf=0.9)      # 2 hops

    db.commit()
    result = C.discover_person(db, "Root Person", depth=2)
    hops = [c["hops"] for c in result["connections"]]
    assert hops == sorted(hops), "results must be grouped by hop distance first"
