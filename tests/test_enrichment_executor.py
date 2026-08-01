"""Executing an enrichment plan.

`expand` is injected throughout, so none of this touches the network — what is
under test is the loop's bookkeeping, not expansion itself. The properties that
matter are the ones a several-hour run depends on: it resumes after a restart,
it stops promptly when asked, and it never reports an unfinished plan as done.
"""
import time

from sqlalchemy import select

from app.models import EnrichmentRun, EnrichmentTask
from app.network.enrichment import plan_run, tally
from app.network.executor import execute_run
from app.network.ingest import ingest_rows
from app.network.probe import ProbeResult


def _rows(*contacts):
    return [{"Name": c.get("name", ""), "Company": c.get("company", ""),
             "Position": c.get("title", "")} for c in contacts]


def _plan(db, n=3, **kw):
    ingest_rows(db, _rows(*[
        {"name": f"Contact Number{i}", "company": f"Company{i}"} for i in range(n)
    ]))
    return plan_run(db, "Siddhu Yen", **kw)


def _recording_expand(log):
    def _expand(db, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
        log.append(name)
    return _expand


def _noop_expand(db, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
    pass


def _has_footprint(name, context):
    """Stub the probe so these tests exercise the LOOP, not the network.

    Not optional: probing is on by default (config.ENRICH_PROBE_ENABLED), so
    an execute_run without this would issue a real search per contact. The
    probe has its own tests in test_enrichment_probe.py.
    """
    return ProbeResult(True, 1, "stub")


def _state(db, run_id):
    """The run's state as PERSISTED. These tests share one session with the
    executor, and a cancel written through Core UPDATE leaves that session's
    identity map holding the pre-cancel copy — expire first or you assert
    against a stale cache rather than the database."""
    db.expire_all()
    return db.get(EnrichmentRun, run_id).state


def test_executes_every_pending_task_in_rank_order(db):
    run = _plan(db, n=3)
    seen = []
    counts = execute_run(db, run.id, probe=_has_footprint, expand=_recording_expand(seen))

    ranked = [t.display_name for t in db.execute(
        select(EnrichmentTask).where(EnrichmentTask.run_id == run.id)
        .order_by(EnrichmentTask.rank)).scalars()]
    assert seen == ranked
    assert counts["done"] == 3
    assert _state(db, run.id) == "done"


def test_the_contacts_context_is_passed_as_seed_context(db):
    """A bare name would let expansion attach a namesake's network; the
    employer is what keeps the queries pinned to the right person."""
    ingest_rows(db, _rows({"name": "John Smith", "company": "Analytical Engines"}))
    run = plan_run(db, "Siddhu Yen")
    seen = {}

    def _expand(db_, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
        seen[name] = context

    execute_run(db, run.id, probe=_has_footprint, expand=_expand)
    assert seen == {"John Smith": "Analytical Engines"}


def test_every_planned_contact_is_protected_from_the_prune(db):
    """expand_graph ends with a name-shape prune over the WHOLE graph. Losing
    the operator's own contacts to it would be silent data loss."""
    run = _plan(db, n=2)
    captured = []

    def _expand(db_, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
        captured.append(set(protected))

    execute_run(db, run.id, probe=_has_footprint, expand=_expand)
    assert all("contact number0" in p and "contact number1" in p for p in captured)


def test_limit_stops_early_and_leaves_the_run_resumable(db):
    """Wave 1 covers the top N; the rest is a later invocation over the same
    plan, so the run must land in `paused`, never `done`."""
    run = _plan(db, n=5)
    counts = execute_run(db, run.id, probe=_has_footprint, limit=2, expand=_noop_expand)

    assert counts["done"] == 2
    assert counts["pending"] == 3
    assert _state(db, run.id) == "paused"


def test_a_paused_run_resumes_where_it_stopped(db):
    run = _plan(db, n=5)
    first, second = [], []
    execute_run(db, run.id, probe=_has_footprint, limit=2, expand=_recording_expand(first))
    execute_run(db, run.id, probe=_has_footprint, expand=_recording_expand(second))

    assert len(first) == 2 and len(second) == 3
    assert not set(first) & set(second)   # nothing is paid for twice
    assert _state(db, run.id) == "done"
    assert tally(db, run.id)["done"] == 5


def test_a_failing_contact_is_recorded_and_the_run_continues(db):
    """One bad contact must not abort hours of remaining work."""
    run = _plan(db, n=3)

    def _expand(db_, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
        if name == "Contact Number1":
            raise RuntimeError("provider exploded")

    counts = execute_run(db, run.id, probe=_has_footprint, expand=_expand)
    assert counts["done"] == 2
    assert counts["failed"] == 1
    failed = db.execute(
        select(EnrichmentTask).where(EnrichmentTask.run_id == run.id,
                                     EnrichmentTask.state == "failed")
    ).scalars().one()
    assert "provider exploded" in failed.last_error
    assert failed.attempts == 1
    # a failed task is not pending, so the run is finished rather than paused
    assert _state(db, run.id) == "done"


def test_cancelling_mid_run_stops_before_the_next_contact(db):
    """Cancel is written by a DIFFERENT session (the HTTP handler), so this
    also pins that the executor re-reads live state rather than its cache."""
    run = _plan(db, n=5)
    seen = []

    def _expand(db_, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
        seen.append(name)
        if len(seen) == 2:
            db_.execute(
                EnrichmentRun.__table__.update()
                .where(EnrichmentRun.__table__.c.id == run.id)
                .values(state="cancelled"))
            db_.commit()

    execute_run(db, run.id, probe=_has_footprint, expand=_expand)
    assert len(seen) == 2
    assert _state(db, run.id) == "cancelled"
    assert tally(db, run.id)["pending"] == 3   # untouched work stays pending


def test_should_stop_reports_a_cancel_to_the_expansion_itself(db):
    """Within a contact, expand_graph polls should_stop at hop/node
    boundaries — that is what makes a cancel land in seconds instead of after
    the current contact's ~35 queries."""
    run = _plan(db, n=2)
    verdicts = []

    def _expand(db_, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
        verdicts.append(should_stop(db_))
        db_.execute(
            EnrichmentRun.__table__.update()
            .where(EnrichmentRun.__table__.c.id == run.id)
            .values(state="cancelled"))
        db_.commit()
        verdicts.append(should_stop(db_))

    execute_run(db, run.id, probe=_has_footprint, expand=_expand)
    assert verdicts == [False, True]


def test_budget_exhaustion_pauses_rather_than_completing(db):
    run = _plan(db, n=5, budget_s=0.05)

    def _slow(db_, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
        time.sleep(0.06)

    counts = execute_run(db, run.id, probe=_has_footprint, expand=_slow)
    assert counts["pending"] > 0
    assert _state(db, run.id) == "paused"


def test_a_run_in_a_terminal_state_cannot_be_started(db):
    run = _plan(db, n=1)
    execute_run(db, run.id, probe=_has_footprint, expand=_noop_expand)
    assert _state(db, run.id) == "done"
    try:
        execute_run(db, run.id, probe=_has_footprint, expand=_noop_expand)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_started_at_survives_a_resume(db):
    """It records when the run began, not when the latest wave did."""
    run = _plan(db, n=3)
    execute_run(db, run.id, probe=_has_footprint, limit=1, expand=_noop_expand)
    started = db.get(EnrichmentRun, run.id).started_at
    execute_run(db, run.id, probe=_has_footprint, expand=_noop_expand)
    assert db.get(EnrichmentRun, run.id).started_at == started


def test_only_one_caller_can_claim_a_run(db):
    """A double-clicked start button would otherwise spawn two executors over
    one plan, both pulling the same pending task and paying twice for it."""
    from app.network.executor import claim_run
    run = _plan(db, n=2)
    assert claim_run(db, run.id) is True
    assert claim_run(db, run.id) is False
    assert _state(db, run.id) == "running"


def test_an_empty_plan_completes_immediately(db):
    run = plan_run(db, "Siddhu Yen")
    counts = execute_run(db, run.id, probe=_has_footprint, expand=_noop_expand)
    assert counts == {"total": 0}
    assert _state(db, run.id) == "done"
