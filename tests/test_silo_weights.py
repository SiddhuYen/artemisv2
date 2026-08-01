"""Per-connection silo weights.

Expansion runs the same 9 silos over everyone — 36 queries whether the subject
is a senator or a college sophomore. Most of those are structurally hopeless:
asking the government silo about a high-schooler retrieves pages about other
people with the same name, which costs money AND is where a large share of the
junk edges come from.

What matters here is that the weights are DERIVED from real signals rather than
guessed, that a contact with no signals still gets the one silo their row
actually asserts (their employer), and that switching the feature off restores
exactly the old behavior.
"""
from app import config
from app.models import EnrichmentTask
from app.network.enrichment import plan_run
from app.network.executor import execute_run
from app.network.ingest import ingest_rows
from app.network.probe import ProbeResult
from app.network.silo_weights import initial_weights, query_budget
from app.silos import SILOS


def _rows(*contacts):
    return [{"Name": c.get("name", ""), "Company": c.get("company", ""),
             "Position": c.get("title", ""), "School": c.get("school", ""),
             "Email Address": c.get("email", "")} for c in contacts]


def _live(weights):
    """Silos that would actually run, given these weights."""
    return set(query_budget(weights))


def _cost(weights):
    return sum(query_budget(weights).values())


# --- the weights themselves -------------------------------------------------

def test_the_employer_silo_is_always_funded(db):
    """An employer is the one relationship an uploaded row actually asserts, so
    it is the only silo guaranteed to be about the right person."""
    for titles in ([], ["Intern"], ["Professor"], ["CEO"]):
        w = initial_weights(titles=titles, companies=["Acme"])
        assert w["company"] >= 1.0
        assert "company" in _live(w)


def test_a_contact_with_no_signals_runs_almost_nothing(db):
    """A bare name at an unremarkable employer: news, boards, publications and
    government are all near-certain to return someone else's pages."""
    w = initial_weights(titles=[], companies=["Acme"])
    assert _live(w) == {"company", "news"}
    assert _cost(w) < 10          # vs 36 unweighted


def test_seniority_funds_news_and_boards(db):
    """Press coverage and board seats are a senior-and-above phenomenon."""
    junior = initial_weights(titles=["Software Engineer"], companies=["Acme"])
    senior = initial_weights(titles=["Co-Founder & CEO"], companies=["Acme"])
    assert senior["news"] > junior["news"]
    assert senior["board_nonprofit"] > junior["board_nonprofit"]
    assert "board_nonprofit" in _live(senior)
    assert "board_nonprofit" not in _live(junior)


def test_an_academic_funds_publications_and_education(db):
    w = initial_weights(titles=["Professor of Computer Science"],
                        companies=["New York University"])
    assert {"publications", "education"} <= _live(w)
    assert "government" not in _live(w)


def test_a_political_role_funds_the_government_silo(db):
    civilian = initial_weights(titles=["Engineer"], companies=["Acme"])
    senator = initial_weights(titles=["Senator"], companies=["US Senate"])
    assert "government" not in _live(civilian)
    assert "government" in _live(senator)


def test_an_org_type_can_stand_in_for_a_title(db):
    """Someone at a foundation is board-relevant even with a blank title —
    plenty of export rows have no position at all."""
    w = initial_weights(titles=[], companies=["The Rockefeller Foundation"])
    assert "board_nonprofit" in _live(w)


def test_an_edu_email_domain_counts_as_an_academic_signal(db):
    w = initial_weights(titles=[], companies=["Acme"], email="a@stanford.edu")
    assert "education" in _live(w)


def test_family_and_friends_are_never_funded(db):
    """Paths run through colleagues, not relatives — expansion already
    down-weights family when ranking a frontier, and spending a professional
    network's query budget on genealogy is the same mistake one step earlier."""
    for titles in ([], ["CEO"], ["Professor"], ["Senator"]):
        live = _live(initial_weights(titles=titles, companies=["Acme"]))
        assert "family" not in live and "friends" not in live


def test_weights_only_name_silos_that_exist(db):
    keys = {s.key for s in SILOS}
    assert set(initial_weights(titles=["CEO"], companies=["Acme"])) <= keys


# --- the budget -------------------------------------------------------------

def test_no_weights_means_every_silo_at_full_allowance(db):
    """The unweighted path must be exactly today's behavior, so an existing
    caller (the CLI, /discover, connect_people) is unaffected."""
    budget = query_budget(None)
    assert set(budget) == {s.key for s in SILOS}
    assert set(budget.values()) == {config.MAX_QUERIES_PER_SILO}


def test_a_funded_silo_always_gets_at_least_one_query(db):
    """Rounding must never fund a silo with zero queries — that is just a more
    expensive way of skipping it."""
    budget = query_budget({s.key: config.ENRICH_SILO_MIN_WEIGHT for s in SILOS})
    assert budget and all(v >= 1 for v in budget.values())


def test_a_silo_is_never_over_funded(db):
    budget = query_budget({s.key: 99.0 for s in SILOS})
    assert set(budget.values()) == {config.MAX_QUERIES_PER_SILO}


# --- integration ------------------------------------------------------------

def test_weights_are_stored_on_the_task_at_plan_time(db):
    """Stored rather than recomputed so a plan is inspectable and reproducible
    — and so they can later be tuned from observed yield."""
    ingest_rows(db, _rows({"name": "Ada Lovelace", "company": "Acme",
                           "title": "Founder"}))
    plan_run(db, "Siddhu Yen")
    task = db.query(EnrichmentTask).filter_by(kind="contact").one()
    assert task.silo_weights["company"] > 1.0
    assert task.silo_weights["family"] < config.ENRICH_SILO_MIN_WEIGHT


def test_the_executor_hands_the_weights_to_expansion(db):
    ingest_rows(db, _rows({"name": "Ada Lovelace", "company": "Acme",
                           "title": "Professor"}))
    run = plan_run(db, "Siddhu Yen")
    seen = {}

    def _expand(db_, name, depth, context, protected, should_stop, progress,
                is_person=True, silo_weights=None):
        seen[name] = silo_weights

    execute_run(db, run.id, expand=_expand,
                probe=lambda n, c: ProbeResult(True, 1, "stub"))
    assert seen["Ada Lovelace"]["publications"] > 0.5


def test_weights_can_be_switched_off(db, monkeypatch):
    monkeypatch.setattr(config, "ENRICH_SILO_WEIGHTS_ENABLED", False)
    ingest_rows(db, _rows({"name": "Ada Lovelace", "company": "Acme"}))
    plan_run(db, "Siddhu Yen")
    task = db.query(EnrichmentTask).filter_by(kind="contact").one()
    assert task.silo_weights == {}      # -> query_budget(None) -> every silo
