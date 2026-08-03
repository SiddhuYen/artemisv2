"""Ranking contacts for enrichment, and persisting the resulting run plan.

Planning must cost nothing — no provider calls, no Claude — so every test here
runs without touching the network. The cases that matter are the ones where
enriching a contact would be wasteful (no web footprint to find) or actively
harmful (a bare name that would attach a namesake's network to the graph).
"""
from sqlalchemy import select

from app.models import EnrichmentRun, EnrichmentTask, Person
from app.network.enrichment import (
    RunConflict,
    cancel_run,
    pending_tasks,
    plan_run,
    tally,
)
from app.network.ingest import ingest_rows
from app.network.ranking import apply_notability, score_contacts


def _rows(*contacts):
    return [{
        "Name": c.get("name", ""),
        "Company": c.get("company", ""),
        "Position": c.get("title", ""),
        "School": c.get("school", ""),
        "Email Address": c.get("email", ""),
        "Url": c.get("url", ""),
    } for c in contacts]


def _by_name(scored):
    return {c.display_name: c for c in scored}


def _order(scored):
    return [c.display_name for c in scored if c.skip_reason is None]


# --- eligibility ------------------------------------------------------------

def test_contact_with_no_organization_is_skipped(db):
    """A bare name cannot be enriched safely: with nothing to disambiguate on,
    the searches would attach whichever notable namesake the web returns."""
    ingest_rows(db, _rows({"name": "John Smith"}))
    scored = score_contacts(db)
    assert _by_name(scored)["John Smith"].skip_reason == "no_context"


def test_title_alone_is_not_enough_context(db):
    """"VP of Engineering" narrows nothing — there are thousands — so it does
    not qualify as disambiguating context even though it scores well once an
    employer IS present."""
    ingest_rows(db, _rows({"name": "John Smith", "title": "VP of Engineering"}))
    assert _by_name(score_contacts(db))["John Smith"].skip_reason == "no_context"


def test_generic_employer_alone_is_skipped_distinctly(db):
    ingest_rows(db, _rows({"name": "John Smith", "company": "Self-Employed"}))
    assert _by_name(score_contacts(db))["John Smith"].skip_reason == "generic_only"


def test_school_alone_is_enough_context(db):
    ingest_rows(db, _rows({"name": "Ada Lovelace", "school": "Cambridge"}))
    contact = _by_name(score_contacts(db))["Ada Lovelace"]
    assert contact.skip_reason is None
    assert contact.context == "Cambridge"


def test_the_operator_is_not_ranked_as_their_own_contact(db):
    ingest_rows(db, _rows(
        {"name": "Siddhu Yen", "company": "Pantheon"},
        {"name": "Ada Lovelace", "company": "Analytical Engines"},
    ))
    scored = score_contacts(db, owner_name="Siddhu Yen")
    assert "Siddhu Yen" not in _by_name(scored)


# --- scoring ----------------------------------------------------------------

def test_seniority_outranks_an_untitled_contact(db):
    """Seniority is a proxy for web footprint: the silos query for board seats,
    funding and press, which a founder returns and a junior IC does not."""
    ingest_rows(db, _rows(
        {"name": "Junior Dev", "company": "Alpha Corp"},
        {"name": "Senior Founder", "company": "Beta Corp", "title": "Co-Founder & CEO"},
    ))
    assert _order(score_contacts(db))[0] == "Senior Founder"


def test_highest_seniority_tier_wins_and_does_not_stack(db):
    ingest_rows(db, _rows(
        {"name": "Aa Ceo", "company": "Alpha Corp", "title": "CEO"},
        {"name": "Bb Both", "company": "Beta Corp", "title": "CEO and Head of Product"},
    ))
    scored = _by_name(score_contacts(db))
    assert scored["Aa Ceo"].score == scored["Bb Both"].score


def test_shared_employer_with_the_operator_boosts(db):
    ingest_rows(db, _rows(
        {"name": "Aa Colleague", "company": "Pantheon"},
        {"name": "Bb Stranger", "company": "Elsewhere Inc"},
    ))
    scored = _by_name(score_contacts(db, owner_company="Pantheon"))
    assert scored["Aa Colleague"].score > scored["Bb Stranger"].score


def test_existing_public_evidence_boosts(db):
    """An edge the web produced is the best available predictor that another
    35 queries will also land. Edges the export itself created don't count —
    every contact has those by construction."""
    ingest_rows(db, _rows(
        {"name": "Aa Known", "company": "Alpha Corp"},
        {"name": "Bb Unknown", "company": "Beta Corp"},
    ), owner_name="Siddhu Yen")
    from app.extraction.schemas import EdgeSignals, ExtractedEdge
    from app.graph import builder
    from app.providers.base import SearchResult

    known = builder.get_or_create_person(db, "Aa Known")
    other = builder.get_or_create_person(db, "Some Journalist")
    source = builder.save_source(
        db, SearchResult("t", "https://example.com/a", "s", "web"), "q")
    builder.add_edge_from_extraction(db, known, ExtractedEdge(
        person_a="Aa Known", person_b="Some Journalist", other_kind="person",
        relationship_type="interview", confidence_base=0.7,
        confidence_adjusted=0.7, signals=EdgeSignals()), 0, source, other)
    db.commit()

    scored = _by_name(score_contacts(db))
    assert scored["Aa Known"].score > scored["Bb Unknown"].score


def test_public_evidence_check_is_scoped_to_the_given_contacts(db):
    """_people_with_public_evidence must only ever answer for the norm_names
    it's asked about, not scan the whole shared graph -- the graph is shared
    across every operator and grows with every run anyone does, so a plan for
    THIS operator's contacts must not pay for everyone else's history."""
    from app.extraction.schemas import EdgeSignals, ExtractedEdge
    from app.graph import builder
    from app.network.ranking import _people_with_public_evidence
    from app.providers.base import SearchResult
    from app.utils.names import person_norm_key

    in_scope = builder.get_or_create_person(db, "In Scope Person")
    out_of_scope = builder.get_or_create_person(db, "Out Of Scope Person")
    other = builder.get_or_create_person(db, "Some Journalist")
    source = builder.save_source(
        db, SearchResult("t", "https://example.com/a", "s", "web"), "q")
    for subject in (in_scope, out_of_scope):
        builder.add_edge_from_extraction(db, subject, ExtractedEdge(
            person_a=subject.canonical_name, person_b="Some Journalist",
            other_kind="person", relationship_type="interview",
            confidence_base=0.7, confidence_adjusted=0.7,
            signals=EdgeSignals()), 0, source, other)
    db.commit()

    found = _people_with_public_evidence(db, {person_norm_key("In Scope Person")})
    assert found == {person_norm_key("In Scope Person")}
    assert person_norm_key("Out Of Scope Person") not in found


def test_public_evidence_check_with_no_contacts_is_a_clean_no_op(db):
    from app.network.ranking import _people_with_public_evidence

    assert _people_with_public_evidence(db, set()) == set()


def test_second_contact_at_the_same_employer_is_damped(db):
    """Coverage, not popularity: the tenth person you know at one company opens
    almost no territory the first nine didn't, so a big employer must not eat
    the whole budget."""
    ingest_rows(db, _rows(
        {"name": "Aa One", "company": "Megacorp"},
        {"name": "Bb Two", "company": "Megacorp"},
        {"name": "Cc Three", "company": "Megacorp"},
        {"name": "Dd Solo", "company": "Tinyco"},
    ))
    order = _order(score_contacts(db))
    megacorp = {"Aa One", "Bb Two", "Cc Three"}
    # One Megacorp contact leads, but the sole Tinyco contact beats the rest.
    # WHICH Megacorp contact leads is deliberately not asserted: all three tie
    # before the decay runs, and pinning one of them only ever tested the
    # tie-break's alphabetical accident (see ranking._tiebreak for why that
    # ordering had to go).
    assert order[0] in megacorp
    assert order[1] == "Dd Solo"
    assert set(order[2:]) == megacorp - {order[0]}

    scored = _by_name(score_contacts(db))
    damped = sorted((scored[n].score for n in megacorp), reverse=True)
    assert damped[0] > damped[1] > damped[2], \
        "each additional contact at one employer must be damped further"
    assert scored["Dd Solo"].score > damped[1], \
        "the sole contact at another employer outranks the damped ones"


def test_ranking_is_deterministic_across_runs(db):
    ingest_rows(db, _rows(*[
        {"name": f"Person Number{i}", "company": f"Company{i % 3}"} for i in range(12)
    ]))
    assert _order(score_contacts(db)) == _order(score_contacts(db))


def test_notability_boost_is_applied_separately(db):
    """Kept out of score_contacts because it is the one signal that costs a
    provider call; the caller supplies the set."""
    ingest_rows(db, _rows(
        {"name": "Aa Famous", "company": "Alpha Corp"},
        {"name": "Bb Obscure", "company": "Beta Corp"},
    ))
    scored = score_contacts(db)
    before = _by_name(scored)["Aa Famous"].score
    boosted = _by_name(apply_notability(scored, {"Aa Famous"}))
    assert boosted["Aa Famous"].score > before
    assert boosted["Aa Famous"].score > boosted["Bb Obscure"].score


# --- run planning -----------------------------------------------------------

def test_plan_run_persists_a_ranked_plan(db):
    ingest_rows(db, _rows(
        {"name": "Aa Founder", "company": "Alpha Corp", "title": "Founder"},
        {"name": "Bb Nobody", "company": "Beta Corp"},
        {"name": "Cc Bare"},
    ))
    run = plan_run(db, "Siddhu Yen", depth=1)

    assert run.state == "planned"
    tasks = db.execute(
        select(EnrichmentTask).where(EnrichmentTask.run_id == run.id)
        .order_by(EnrichmentTask.rank)).scalars().all()
    assert [t.rank for t in tasks] == [1, 2, 3]
    assert tasks[0].display_name == "Aa Founder"
    assert tasks[0].context == "Alpha Corp"   # passed to expand_graph as seed_context
    assert tasks[-1].state == "skipped"       # the bare name sorts last
    assert tasks[-1].skip_reason == "no_context"
    assert run.counters["total"] == 3
    assert run.counters["pending"] == 2


def test_already_processed_contacts_start_done(db):
    """expansion._reuse_existing_neighbors makes re-expanding a processed node
    a no-op, so it should not occupy a scheduling slot — including when another
    operator sharing this graph is the one who expanded it."""
    ingest_rows(db, _rows({"name": "Ada Lovelace", "company": "Analytical Engines"}))
    from app.graph import builder
    person = builder.get_or_create_person(db, "Ada Lovelace")
    person.processed = 1
    db.commit()

    run = plan_run(db, "Siddhu Yen")
    assert tally(db, run.id)["done"] == 1
    assert pending_tasks(db, run.id) == []


def test_pending_tasks_come_back_in_rank_order(db):
    ingest_rows(db, _rows(
        {"name": "Aa Nobody", "company": "Alpha Corp"},
        {"name": "Bb Founder", "company": "Beta Corp", "title": "CEO"},
    ))
    run = plan_run(db, "Siddhu Yen")
    assert [t.display_name for t in pending_tasks(db, run.id)] == \
        ["Bb Founder", "Aa Nobody"]
    assert [t.display_name for t in pending_tasks(db, run.id, limit=1)] == ["Bb Founder"]


def test_a_second_run_is_refused_while_one_is_in_flight(db):
    """Two concurrent runs would race for the same build slots and enrich the
    same contacts twice."""
    ingest_rows(db, _rows({"name": "Ada Lovelace", "company": "Analytical Engines"}))
    run = plan_run(db, "Siddhu Yen")
    run.state = "running"
    db.commit()

    try:
        plan_run(db, "Siddhu Yen")
        assert False, "expected RunConflict"
    except RunConflict:
        pass


def test_a_planned_run_does_not_block_another_plan(db):
    """Only running/paused runs conflict — a plan that was never executed is
    just a stale preview."""
    ingest_rows(db, _rows({"name": "Ada Lovelace", "company": "Analytical Engines"}))
    plan_run(db, "Siddhu Yen")
    plan_run(db, "Siddhu Yen")
    assert len(db.execute(select(EnrichmentRun)).scalars().all()) == 2


def test_cancel_is_terminal_and_idempotent(db):
    ingest_rows(db, _rows({"name": "Ada Lovelace", "company": "Analytical Engines"}))
    run = plan_run(db, "Siddhu Yen")
    run.state = "running"
    db.commit()

    cancel_run(db, run)
    assert run.state == "cancelled"
    assert run.finished_at
    finished = run.finished_at
    cancel_run(db, run)
    assert run.finished_at == finished


def test_planning_with_no_contacts_is_a_clean_empty_run(db):
    run = plan_run(db, "Siddhu Yen")
    assert run.counters == {"total": 0}
    assert pending_tasks(db, run.id) == []


def test_planning_touches_no_person_rows_for_skipped_contacts(db):
    """Planning is free AND inert: it must not create graph nodes as a side
    effect, or a preview would mutate the shared graph."""
    ingest_rows(db, _rows({"name": "Cc Bare"}))
    before = len(db.execute(select(Person)).scalars().all())
    plan_run(db, "Siddhu Yen")
    assert len(db.execute(select(Person)).scalars().all()) == before
