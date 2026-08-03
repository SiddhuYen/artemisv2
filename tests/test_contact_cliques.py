"""Wave 0: structural edges derived from the contact export alone.

This runs BEFORE any enrichment and must never touch the network — the whole
point is that a freshly-imported export already asserts colleague ties and org
membership for free. The cases that matter are the ones where that assertion
is WRONG and has to be suppressed: a self-reported "Self-Employed" employer,
and a big-company directory artifact where 200 contacts share one employer and
are not colleagues in any useful sense.
"""
from sqlalchemy import select

from app import config
from app.models import Organization, RelationshipEdge
from app.network.cliques import materialize_contact_cliques
from app.network.ingest import ingest_rows
from app.utils.names import org_norm_key


def _rows(*contacts):
    return [{
        "Name": c.get("name", ""),
        "Company": c.get("company", ""),
        "School": c.get("school", ""),
    } for c in contacts]


def _edges(db, rel_type):
    return db.execute(
        select(RelationshipEdge).where(
            RelationshipEdge.relationship_type == rel_type)
    ).scalars().all()


def _names(db, edges):
    from app.models import Person
    people = {p.id: p.canonical_name for p in db.execute(select(Person)).scalars()}
    return {(people.get(e.person_a_id), people.get(e.person_b_id)) for e in edges}


def test_shared_employer_becomes_a_coworker_clique(db):
    ingest_rows(db, _rows(
        {"name": "Ada Lovelace", "company": "Analytical Engines"},
        {"name": "Grace Hopper", "company": "Analytical Engines"},
        {"name": "Alan Turing", "company": "Bletchley Systems"},
    ))
    counts = materialize_contact_cliques(db)

    # ONE row per pair. Both readers are undirected (connect._adjacency, and
    # expansion._reuse_existing_neighbors matches either endpoint), so the
    # mirrored row this used to write asserted nothing extra.
    assert counts["coworker_edges"] == 1
    assert _names(db, _edges(db, "coworker")) == {("Ada Lovelace", "Grace Hopper")}
    # the lone contact at the other employer gets membership but no clique
    assert counts["cliques"] == 1
    assert counts["membership_edges"] == 3


def test_membership_edges_link_contacts_to_their_employer(db):
    ingest_rows(db, _rows({"name": "Ada Lovelace", "company": "Analytical Engines"}))
    materialize_contact_cliques(db)

    org = db.execute(
        select(Organization).where(
            Organization.norm_name == org_norm_key("Analytical Engines"))
    ).scalar_one()
    assert org.type == "company"
    employee = _edges(db, "employee")
    assert len(employee) == 1
    assert employee[0].organization_id == org.id
    # trusted, so _prune_invalid_nodes never drops the operator's own contacts
    assert employee[0].signals["trusted"] is True
    assert employee[0].status == "strong"


def test_oversized_employer_keeps_membership_but_drops_the_clique(db):
    """200 contacts listing one big employer are a directory artifact, not a
    team — the clique would assert tens of thousands of false colleague ties."""
    size = config.CONTACT_CLIQUE_MAX + 1
    ingest_rows(db, _rows(*[
        {"name": f"Person Number{i}", "company": "Megacorp"} for i in range(size)
    ]))
    counts = materialize_contact_cliques(db)

    assert counts["skipped_oversize"] == 1
    assert counts["coworker_edges"] == 0
    assert counts["cliques"] == 0
    assert counts["membership_edges"] == size  # membership is still real


def test_employer_exactly_at_the_cap_still_cliques(db):
    ingest_rows(db, _rows(*[
        {"name": f"Person Number{i}", "company": "Smallco"}
        for i in range(config.CONTACT_CLIQUE_MAX)
    ]))
    counts = materialize_contact_cliques(db)

    n = config.CONTACT_CLIQUE_MAX
    assert counts["skipped_oversize"] == 0
    assert counts["coworker_edges"] == n * (n - 1) // 2   # one row per pair


def test_generic_employers_never_form_a_clique(db):
    """"Self-Employed" is one of the commonest values in a real export. Fusing
    every independent contractor the operator knows into one mutual-coworker
    blob is the single worst thing wave 0 could do."""
    ingest_rows(db, _rows(
        {"name": "Ada Lovelace", "company": "Self-Employed"},
        {"name": "Grace Hopper", "company": "self employed"},
        {"name": "Alan Turing", "company": "Freelance"},
        {"name": "Katherine Johnson", "company": "Retired"},
    ))
    counts = materialize_contact_cliques(db)

    assert counts["skipped_generic"] == 4
    assert counts["coworker_edges"] == 0
    assert counts["membership_edges"] == 0
    assert counts["organizations"] == 0
    assert db.execute(select(Organization)).scalars().all() == []


def test_shared_school_is_membership_only_never_a_clique(db):
    """A school is a directory, and the export carries no graduation year — two
    contacts who attended the same university 20 years apart are not connected
    by that fact."""
    ingest_rows(db, _rows(
        {"name": "Ada Lovelace", "school": "Cambridge"},
        {"name": "Grace Hopper", "school": "Cambridge"},
    ))
    counts = materialize_contact_cliques(db)

    assert counts["coworker_edges"] == 0
    assert counts["cliques"] == 0
    student = _edges(db, "student")
    assert len(student) == 2
    org = db.execute(
        select(Organization).where(Organization.norm_name == org_norm_key("Cambridge"))
    ).scalar_one()
    assert org.type == "school"


def test_rerunning_converges_instead_of_duplicating(db):
    """Stable synthetic source URLs + the (a, b, type, source) dedup rule mean
    a second pass adds nothing — the same property backfill_graph_edges has."""
    ingest_rows(db, _rows(
        {"name": "Ada Lovelace", "company": "Analytical Engines"},
        {"name": "Grace Hopper", "company": "Analytical Engines"},
    ))
    materialize_contact_cliques(db)
    before = len(db.execute(select(RelationshipEdge)).scalars().all())

    materialize_contact_cliques(db)
    after = len(db.execute(select(RelationshipEdge)).scalars().all())
    assert after == before


def test_no_contacts_is_a_clean_no_op(db):
    counts = materialize_contact_cliques(db)
    assert counts == {"organizations": 0, "membership_edges": 0,
                      "coworker_edges": 0, "cliques": 0,
                      "skipped_oversize": 0, "skipped_generic": 0}
