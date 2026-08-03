"""The operator's own identity, persisted server-side.

Before this, "who am I" lived only in the browser and was passed per-request.
Two things follow from persisting it: the operator's employer and school reach
ranking (nothing was sending them, so the shared-affiliation boost was dead
code in practice), and the operator joins their own org cluster in wave 0.
"""
from sqlalchemy import or_, select

from app.models import EnrichmentTask, Organization, Person, RelationshipEdge
from app.network.cliques import materialize_contact_cliques
from app.network.enrichment import plan_run
from app.network.ingest import ingest_rows
from app.network.owner import get_owner, owner_dict, upsert_owner
from app.network.ranking import score_contacts
from app.utils.names import org_norm_key, person_norm_key


def _rows(*contacts):
    return [{"Name": c.get("name", ""), "Company": c.get("company", ""),
             "School": c.get("school", "")} for c in contacts]


# --- storage ----------------------------------------------------------------

def test_a_profile_round_trips(db):
    upsert_owner(db, "gid1", name="Siddhu Yen", company="Pantheon Prep",
                 title="Founder", school="NYU")
    profile = get_owner(db, "gid1")
    assert (profile.name, profile.company, profile.school) == \
        ("Siddhu Yen", "Pantheon Prep", "NYU")
    assert owner_dict(profile)["configured"] is True


def test_a_partial_save_does_not_blank_the_rest(db):
    """Saving just the company must not wipe a name the operator set earlier."""
    upsert_owner(db, "gid1", name="Siddhu Yen", company="Pantheon Prep")
    upsert_owner(db, "gid1", company="Trinamix Inc")
    profile = get_owner(db, "gid1")
    assert profile.name == "Siddhu Yen"
    assert profile.company == "Trinamix Inc"


def test_profiles_are_scoped_per_owner_id(db):
    """Two operators on one deployment must not overwrite each other."""
    upsert_owner(db, "gid1", name="Alice Alpha", company="Alpha Corp")
    upsert_owner(db, "gid2", name="Bruno Beta", company="Beta Corp")
    assert get_owner(db, "gid1").name == "Alice Alpha"
    assert get_owner(db, "gid2").name == "Bruno Beta"


def test_an_absent_profile_is_a_normal_state_not_an_error(db):
    assert get_owner(db, "nobody") is None
    blank = owner_dict(None)
    assert blank["configured"] is False and blank["name"] == ""


def test_an_empty_name_never_overwrites_a_real_one(db):
    upsert_owner(db, "gid1", name="Siddhu Yen")
    upsert_owner(db, "gid1", name="", company="Pantheon Prep")
    assert get_owner(db, "gid1").name == "Siddhu Yen"


# --- what the profile actually buys ----------------------------------------

def test_the_owners_employer_boosts_colleagues_in_ranking(db):
    """The whole reason company/school are stored: nothing else ever supplied
    them, so this boost could not fire before."""
    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Pantheon Prep"},
        {"name": "Bb Stranger", "company": "Elsewhere Inc"},
    ))
    profile = upsert_owner(db, "gid1", name="Siddhu Yen", company="Pantheon Prep")
    scored = {c.display_name: c for c in score_contacts(
        db, owner_name=profile.name, owner_company=profile.company)}
    assert scored["Aa Colleague"].score > scored["Bb Stranger"].score


def test_the_owner_joins_their_own_employer_cluster(db):
    """Without this the operator sits outside every org cluster, joined only by
    linkedin_1st edges, understating the network they have at their own job."""
    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Pantheon Prep"},
        {"name": "Bb Colleague", "company": "Pantheon Prep"},
    ))
    owner = upsert_owner(db, "gid1", name="Siddhu Yen", company="Pantheon Prep")
    materialize_contact_cliques(db, owner=owner)

    owner_person = db.execute(select(Person).where(
        Person.norm_name == person_norm_key("Siddhu Yen"))).scalar_one()
    org = db.execute(select(Organization).where(
        Organization.norm_name == org_norm_key("Pantheon Prep"))).scalar_one()
    membership = db.execute(select(RelationshipEdge).where(
        RelationshipEdge.person_a_id == owner_person.id,
        RelationshipEdge.organization_id == org.id)).scalars().all()
    assert len(membership) == 1
    # Either orientation: a coworker tie is stored once now, and which endpoint
    # lands in person_a is just clique iteration order.
    coworkers = db.execute(select(RelationshipEdge).where(
        or_(RelationshipEdge.person_a_id == owner_person.id,
            RelationshipEdge.person_b_id == owner_person.id),
        RelationshipEdge.relationship_type == "coworker")).scalars().all()
    assert len(coworkers) == 2      # both colleagues


def test_wave_zero_without_an_owner_is_unchanged(db):
    """The owner is optional — every existing caller passes nothing."""
    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Pantheon Prep"},
        {"name": "Bb Colleague", "company": "Pantheon Prep"},
    ))
    counts = materialize_contact_cliques(db)
    assert counts["coworker_edges"] == 1      # the two contacts, one pair
    assert counts["membership_edges"] == 2


def test_the_owner_is_still_excluded_from_their_own_contact_ranking(db):
    """Being a clique participant must not make the operator a contact to
    enrich — they are the root of the network, not a member of it. (An export
    really can contain the operator: LinkedIn includes you in some formats.)"""
    ingest_rows(db, _rows({"name": "Siddhu Yen", "company": "Pantheon Prep"}))
    owner = upsert_owner(db, "gid1", name="Siddhu Yen", company="Pantheon Prep")
    materialize_contact_cliques(db, owner=owner)

    run = plan_run(db, "Siddhu Yen", owner_company="Pantheon Prep")
    names = [t.display_name for t in db.execute(
        select(EnrichmentTask).where(EnrichmentTask.run_id == run.id)
    ).scalars()]
    assert "Siddhu Yen" not in names
