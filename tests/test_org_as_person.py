"""An organization minted into `people` must not be walked as a person.

Observed on Justin Hotard -> Lip-Bu Tan, which returned:

    Justin Hotard -> Hewlett Packard Enterprise -> Lip-Bu Tan

with the second hop's evidence reading "Hewlett Packard Enterprise coworker of
Lip Bu Tan." The company was a Person row, processed=1, expanded as a human.

What makes this different from every other bad-edge case here is that NO
EVIDENCE CHECK CAN CATCH IT. Hop verification passed the first hop and was
right to -- it returned verified_status='genuine' because "Justin Hotard worked
at Hewlett Packard Enterprise in a leadership role" is true and well sourced.
The relationship is real; the TYPE is wrong. A person-to-employer affiliation
is being walked as though the employer were a person who knows people.

So the guard is structural: a name present in BOTH `people` and `organizations`
may be an endpoint, and may never be routed THROUGH.
"""
from app.graph import connect as C
from app.models import Organization, Person, RelationshipEdge
from app.utils.names import org_norm_key, person_norm_key


def _person(db, name):
    p = Person(canonical_name=name, norm_name=person_norm_key(name))
    db.add(p)
    db.flush()
    return p


def _org(db, name):
    o = Organization(name=name, norm_name=org_norm_key(name))
    db.add(o)
    db.flush()
    return o


def _edge(db, a, b):
    db.add(RelationshipEdge(person_a_id=a.id, person_b_id=b.id,
                            relationship_type="coworker", status="strong",
                            confidence_raw=0.85, method="test",
                            evidence_snippet="ev",
                            signals={"sentence_cooccurrence": True}))


def _graph_through_an_org(db):
    """A -- "HPE" -- B, where the middle is a company in both tables."""
    a, b = _person(db, "Justin Hotard"), _person(db, "Lip-Bu Tan")
    mid = _person(db, "Hewlett Packard Enterprise")
    _org(db, "Hewlett Packard Enterprise")
    _edge(db, a, mid)
    _edge(db, mid, b)
    db.commit()
    return a, mid, b


def test_the_collision_is_detected(db):
    a, mid, b = _graph_through_an_org(db)
    ids = C._org_shaped_person_ids(db)
    assert mid.id in ids
    assert a.id not in ids and b.id not in ids


def test_a_route_is_not_walked_through_an_org(db):
    """The regression itself."""
    a, mid, b = _graph_through_an_org(db)
    adj, by_id, _src, deg = C._adjacency(db)

    assert C._diverse_paths(adj, a.id, b.id, 4, 3, by_id, deg), \
        "sanity: without the guard this route exists"
    assert not C._diverse_paths(adj, a.id, b.id, 4, 3, by_id, deg,
                                excluded_intermediates=C._org_shaped_person_ids(db))


def test_route_exists_agrees_with_the_pathfinder(db):
    """The cheap check and the scoring pass must not disagree.

    A stage-0 "already connected" that the pathfinder then refuses to walk is
    the worst pair of outcomes at once: the paid search is skipped BECAUSE a
    route is believed found, and then nothing is returned. That is the same
    failure #53's second gate exists to prevent, reached a different way.
    """
    a, mid, b = _graph_through_an_org(db)
    assert C._route_exists(db, "Justin Hotard", "Lip-Bu Tan", 4) is False


def test_an_org_shaped_node_is_still_reachable_as_an_endpoint(db):
    """Blocking pass-through, not hiding the node. The collision says the two
    tables disagree, not which is wrong -- "Arnold Schwarzenegger" and "Steve
    Nash" are in this set too, as real people with a junk org row. Excluding
    them as intermediates costs a route; hiding them would lose the person."""
    a, mid, b = _graph_through_an_org(db)
    assert C._route_exists(db, "Justin Hotard", "Hewlett Packard Enterprise", 4) is True

    adj, by_id, _src, deg = C._adjacency(db)
    assert C._diverse_paths(adj, a.id, mid.id, 4, 3, by_id, deg,
                            excluded_intermediates=C._org_shaped_person_ids(db)), \
        "the org must still be reachable when it IS the endpoint asked for"


def test_a_clean_route_is_unaffected(db):
    """The guard must cost nothing when no node collides."""
    a, m, b = _person(db, "Aa One"), _person(db, "Mm Mid"), _person(db, "Bb Two")
    _edge(db, a, m)
    _edge(db, m, b)
    db.commit()

    adj, by_id, _src, deg = C._adjacency(db)
    assert C._diverse_paths(adj, a.id, b.id, 4, 3, by_id, deg,
                            excluded_intermediates=C._org_shaped_person_ids(db))
    assert C._route_exists(db, "Aa One", "Bb Two", 4) is True
