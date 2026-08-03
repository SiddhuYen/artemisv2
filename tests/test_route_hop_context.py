"""A route hop must say WHERE, not just what.

"coworker" answers what the tie is and leaves the obvious follow-up --
coworker where? -- unanswered. The place is never stored on the
person-person edge itself (organization_id is set only on person->org rows,
never alongside person_b_id), so connect.py recovers it by intersecting the
two endpoints' own org affiliations.
"""
from app.graph import connect as C
from app.models import Organization, Person, RelationshipEdge
from app.utils.names import person_norm_key, org_norm_key


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


def _org_edge(db, person, org, conf=0.8, status="strong", rel="employee"):
    db.add(RelationshipEdge(person_a_id=person.id, organization_id=org.id,
                            relationship_type=rel, status=status,
                            confidence_raw=conf,
                            signals={"sentence_cooccurrence": True}))


def test_shared_org_is_the_where_of_a_hop(db):
    a, b = _person(db, "Ada Bridge"), _person(db, "Bo Bridge")
    salesforce = _org(db, "Salesforce")
    _org_edge(db, a, salesforce)
    _org_edge(db, b, salesforce)
    db.commit()

    assert C._shared_orgs(C._org_affiliations(db), a.id, b.id) == ["Salesforce"]


def test_no_shared_org_yields_no_where(db):
    a, b = _person(db, "Ada Solo"), _person(db, "Bo Solo")
    _org_edge(db, a, _org(db, "Salesforce"))
    _org_edge(db, b, _org(db, "Oracle"))
    db.commit()

    assert C._shared_orgs(C._org_affiliations(db), a.id, b.id) == []


def test_shared_orgs_rank_by_the_weaker_side(db):
    """A place is only as good an answer as the shakier of the two
    affiliations behind it -- so a 0.9/0.2 pair must rank below a 0.6/0.6 one
    even though its best single edge is stronger."""
    a, b = _person(db, "Ada Rank"), _person(db, "Bo Rank")
    shaky, solid = _org(db, "Shaky Corp"), _org(db, "Solid Corp")
    _org_edge(db, a, shaky, conf=0.9)
    _org_edge(db, b, shaky, conf=0.2)
    _org_edge(db, a, solid, conf=0.6)
    _org_edge(db, b, solid, conf=0.6)
    db.commit()

    assert C._shared_orgs(C._org_affiliations(db), a.id, b.id) == \
        ["Solid Corp", "Shaky Corp"]


def test_rejected_org_edge_is_not_a_where(db):
    """'rejected' means reviewed and marked false -- the same rule that keeps
    an edge out of the route keeps it from explaining one."""
    a, b = _person(db, "Ada Ghost"), _person(db, "Bo Ghost")
    org = _org(db, "Phantom Inc")
    _org_edge(db, a, org)
    _org_edge(db, b, org, status="rejected")
    db.commit()

    assert C._shared_orgs(C._org_affiliations(db), a.id, b.id) == []


def test_connect_people_attaches_where_and_source_to_each_hop(db, monkeypatch):
    monkeypatch.setattr(C, "_route_exists", lambda *a, **k: True)
    a, b = _person(db, "Ada End"), _person(db, "Bo End")
    org = _org(db, "Salesforce")
    _org_edge(db, a, org)
    _org_edge(db, b, org)
    db.add(RelationshipEdge(person_a_id=a.id, person_b_id=b.id,
                            relationship_type="coworker", status="strong",
                            confidence_raw=0.7, method="co-mention in a filing",
                            evidence_snippet="Ada and Bo both led teams there.",
                            signals={"sentence_cooccurrence": True}))
    db.commit()

    result = C.connect_people(db, "Ada End", "Bo End", depth=1)

    assert result["connected"] is True
    hop = result["path"][1]
    assert hop["relationship_from_previous"] == "coworker"
    assert hop["via_orgs"] == ["Salesforce"]
    assert hop["method"] == "co-mention in a filing"
