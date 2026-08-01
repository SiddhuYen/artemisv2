"""Wave 2: sweeping a shared employer once instead of per-contact.

Expanding an org several contacts share reaches that organization's public
neighbourhood for ONE contact's worth of queries, which is the best coverage
per query available — so org tasks run first. The judgement calls that matter
are which orgs are worth it and how an org seed differs from a person seed.
"""
from app import config
from app.models import EnrichmentTask
from app.network.enrichment import plan_run
from app.network.executor import execute_run
from app.network.ingest import ingest_rows
from app.network.probe import ProbeResult
from app.network.ranking import org_sweep_candidates
from app.utils.names import org_norm_key


def _rows(*contacts):
    return [{"Name": c.get("name", ""), "Company": c.get("company", "")}
            for c in contacts]


def _at(company, n, start=0):
    return _rows(*[{"name": f"Person Number{i}", "company": company}
                   for i in range(start, start + n)])


def _has_footprint(name, context):
    return ProbeResult(True, 1, "stub")


def _tasks(db, run_id, kind=None):
    q = db.query(EnrichmentTask).filter_by(run_id=run_id)
    if kind:
        q = q.filter_by(kind=kind)
    return q.order_by(EnrichmentTask.rank).all()


# --- candidate selection ----------------------------------------------------

def test_an_employer_with_enough_contacts_is_a_sweep_candidate(db):
    ingest_rows(db, _at("Megacorp", 4))
    sweeps = org_sweep_candidates(db)
    assert [(s.name, s.contacts) for s in sweeps] == [("Megacorp", 4)]


def test_a_thinly_shared_employer_is_not_worth_a_sweep(db):
    """Below the threshold, sweeping those one or two contacts directly costs
    the same and returns their actual network, not the org's public face."""
    ingest_rows(db, _at("Tinyco", config.ENRICH_ORG_MIN_CONTACTS - 1))
    assert org_sweep_candidates(db) == []


def test_candidates_are_ranked_by_coverage(db):
    ingest_rows(db, _at("Bigco", 6) + _at("Midco", 4, start=100)
                + _at("Smallco", 3, start=200))
    assert [s.name for s in org_sweep_candidates(db)] == ["Bigco", "Midco", "Smallco"]


def test_generic_employers_are_never_swept(db):
    """"Self-Employed" is not an organization to expand — the same rule wave 0
    applies to cliques."""
    ingest_rows(db, _at("Self-Employed", 8))
    assert org_sweep_candidates(db) == []


def test_stealth_startup_is_not_an_employer(db):
    """LinkedIn's placeholder for declining to say where you work. Common
    enough in a real export (12 of 1,025) to rank third for a sweep, and
    expanding it would invent a company a dozen strangers all work at."""
    ingest_rows(db, _at("Stealth Startup", 12))
    assert org_sweep_candidates(db) == []


def test_one_contact_listing_an_employer_twice_votes_once(db):
    ingest_rows(db, [{"Name": "Solo Person", "Company": "Acme; Acme Inc."}])
    assert org_sweep_candidates(db, min_contacts=2) == []


def test_the_number_of_sweeps_is_capped(db):
    """Org tasks run FIRST, so an unbounded number would spend the whole wave-1
    budget before reaching a single real contact."""
    rows = []
    for i in range(config.ENRICH_ORG_MAX_SWEEPS + 5):
        rows += _at(f"Company{i}", 3, start=i * 10)
    ingest_rows(db, rows)
    assert len(org_sweep_candidates(db)) == config.ENRICH_ORG_MAX_SWEEPS


# --- planning ---------------------------------------------------------------

def test_org_tasks_are_planned_ahead_of_every_contact(db):
    ingest_rows(db, _at("Megacorp", 4))
    run = plan_run(db, "Siddhu Yen")
    tasks = _tasks(db, run.id)
    assert tasks[0].kind == "org"
    assert tasks[0].display_name == "Megacorp"
    assert all(t.kind == "contact" for t in tasks[1:])


def test_an_org_task_carries_no_disambiguating_context(db):
    """Context exists to pin a person against a namesake; an org name is
    already the thing being searched for."""
    ingest_rows(db, _at("Megacorp", 4))
    org = _tasks(db, plan_run(db, "Siddhu Yen").id, kind="org")[0]
    assert org.context is None
    assert org.norm_name == org_norm_key("Megacorp")
    assert org.score == 4.0          # coverage IS the score


def test_org_sweeps_can_be_switched_off(db, monkeypatch):
    monkeypatch.setattr(config, "ENRICH_ORG_SWEEPS_ENABLED", False)
    ingest_rows(db, _at("Megacorp", 4))
    assert _tasks(db, plan_run(db, "Siddhu Yen").id, kind="org") == []


# --- execution --------------------------------------------------------------

def test_an_org_seed_is_expanded_as_an_organization(db):
    """seed_is_person=False routes the seed through plain web search instead of
    the Wikipedia/Wikidata person path — expanding "Stripe" as a PERSON would
    look for a human called Stripe."""
    ingest_rows(db, _at("Megacorp", 4))
    run = plan_run(db, "Siddhu Yen")
    seen = {}

    def _expand(db_, name, depth, context, protected, should_stop, progress,
                is_person=True, silo_weights=None):
        seen[name] = is_person

    execute_run(db, run.id, limit=1, expand=_expand, probe=_has_footprint)
    assert seen == {"Megacorp": False}


def test_org_tasks_skip_the_probe(db):
    """The probe looks for every NAME token in a result, which is meaningless
    for an org — and a shared employer reliably has a web presence anyway."""
    ingest_rows(db, _at("Megacorp", 4))
    run = plan_run(db, "Siddhu Yen")

    def _expand(db_, name, depth, context, protected, should_stop, progress,
                is_person=True, silo_weights=None):
        pass

    def _probe(name, context):
        raise AssertionError("an org task must not be probed")

    counts = execute_run(db, run.id, limit=1, expand=_expand, probe=_probe)
    assert counts["done"] == 1


def test_a_completed_org_task_records_no_person(db):
    """An org sweep creates an Organization node, not a Person — leaving
    person_id set would point a task at whatever namesake happened to exist."""
    ingest_rows(db, _at("Megacorp", 4))
    run = plan_run(db, "Siddhu Yen")

    def _expand(db_, name, depth, context, protected, should_stop, progress,
                is_person=True, silo_weights=None):
        pass

    execute_run(db, run.id, limit=1, expand=_expand, probe=_has_footprint)
    org = _tasks(db, run.id, kind="org")[0]
    assert org.state == "done"
    assert org.person_id is None
