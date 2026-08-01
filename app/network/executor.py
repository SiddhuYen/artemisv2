"""Executing an enrichment plan — the part that actually spends money.

Consumes the ranked plan `enrichment.plan_run` produced: pull the best pending
task, expand that contact one hop, record what happened, repeat. Each contact
expanded at depth 1 is one slice of the operator's L2 layer.

Three properties this has to hold, none of which are optional:

  RESUMABLE. A run is hours long and a deploy will land in the middle of one.
  All state is in `EnrichmentTask`, committed after every task, so a restart
  picks up at the next pending task rather than re-paying for the first half.
  Nothing is held in memory that a restart would lose.

  INTERRUPTIBLE. Cancel/pause are observed BETWEEN tasks by re-reading the run
  row (the API handler writes it from a different session), and WITHIN a task
  via expand_graph's `should_stop`, so a cancel lands in seconds rather than
  after the current contact's ~35 queries finish.

  POLITE. Every task takes a BACKGROUND build ticket, so the run fills idle
  capacity and yields to interactive /connect the moment any arrives — see
  buildqueue.BuildQueue. Per-task rather than per-run: an interactive build
  should never wait on more than one contact's worth of work.

`expand` is injected so the whole loop is testable without touching the
network; it defaults to the real expand_graph.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from .. import config
from ..buildqueue import BUILDS, QueueFull
from ..models import EnrichmentRun, EnrichmentTask
from .enrichment import tally

# Run states from which execution may (re)start. "planned" is a fresh plan;
# "paused" is one that stopped on budget or by request and kept its progress.
STARTABLE = ("planned", "paused")
# Checked between tasks — anything else means someone else changed the run.
CONTINUABLE = "running"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_expand(db: Session, name: str, depth: int, context: str,
                    protected_norms, should_stop, progress,
                    is_person: bool = True, silo_weights=None) -> None:
    from ..graph.expansion import expand_graph

    # seed_is_person=False routes an ORG seed through plain web search instead
    # of the Wikipedia/Wikidata person path — the whole reason wave 2 can reuse
    # this function rather than needing its own expansion.
    expand_graph(db, name, depth, progress=progress, seed_context=context,
                 protected_norms=protected_norms, should_stop=should_stop,
                 seed_is_person=is_person, silo_weights=silo_weights)


def _run_state(bind, run_id: str) -> Optional[str]:
    """Read the run's state on a FRESH connection.

    Two reasons it cannot reuse the caller's Session, both of which make
    cancellation silently stop working rather than fail loudly:

      1. A Session holds its read transaction open until commit, so repeated
         reads inside one would keep returning the snapshot from before the
         cancel endpoint (a different session entirely) wrote it.
      2. expand_graph calls `should_stop` from its per-node WORKER threads,
         and a Session is not safe to share across threads.

    A short-lived connection sidesteps both and costs nothing at the hop/node
    boundaries where this is called.
    """
    with bind.connect() as conn:
        row = conn.execute(
            select(EnrichmentRun.state).where(EnrichmentRun.id == run_id)
        ).first()
    return row[0] if row else None


def _next_task(db: Session, run_id: str) -> Optional[EnrichmentTask]:
    """The best-ranked task still worth attempting.

    A `failed` task is included as long as it has attempts left
    (config.ENRICH_MAX_ATTEMPTS) — at ~35 outbound calls per contact, a
    transient provider timeout or rate-limit blip is an expected occurrence
    over an hours-long run, not a rare one, so 'failed' has to mean "try
    again", not "gone forever". Past the ceiling it stops being selected here
    and stays failed for good, so a contact that's PERMANENTLY broken (a name
    that reliably crashes the extractor) doesn't retry indefinitely.
    """
    return db.execute(
        select(EnrichmentTask)
        .where(
            EnrichmentTask.run_id == run_id,
            or_(
                EnrichmentTask.state == "pending",
                and_(EnrichmentTask.state == "failed",
                     EnrichmentTask.attempts < config.ENRICH_MAX_ATTEMPTS),
            ),
        )
        .order_by(EnrichmentTask.rank)
        .limit(1)
    ).scalars().first()


def claim_run(db: Session, run_id: str) -> bool:
    """Atomically move a run from planned/paused into running.

    True if THIS caller won the claim. A conditional UPDATE rather than a
    read-then-write because two rapid POST .../start calls (a double-clicked
    button) would otherwise both see `planned`, both spawn an executor thread,
    and both pull the same pending task — paying twice for every contact.

    The HTTP handler claims synchronously so it can answer 409 immediately,
    then hands the already-claimed run to the worker thread.
    """
    claimed = db.execute(
        update(EnrichmentRun)
        .where(EnrichmentRun.id == run_id, EnrichmentRun.state.in_(STARTABLE))
        .values(state=CONTINUABLE)
    ).rowcount == 1
    db.commit()
    return claimed


def _protected_norms(db: Session, run_id: str) -> set:
    """Every contact in the plan is exempt from expand_graph's final noise
    prune. Contacts usually carry a trusted edge already (linkedin_1st, or
    wave 0's coworker/employee) which protects them anyway, but that depends on
    the import having supplied owner_name — and losing the operator's own
    contacts to a name-shape heuristic is not a failure worth risking."""
    return set(db.execute(
        select(EnrichmentTask.norm_name).where(EnrichmentTask.run_id == run_id)
    ).scalars())


def _acquire_background_slot(db: Session, run_id: str,
                             progress=None) -> Optional[object]:
    """Wait for a background build slot. Returns None if the run stopped first.

    QueueFull here means interactive callers have filled the line; that is a
    reason to wait, not to fail the task — unlike an HTTP caller, this run has
    nowhere better to be.
    """
    bind = db.get_bind()
    while True:
        if _run_state(bind, run_id) != CONTINUABLE:
            return None
        try:
            ticket = BUILDS.reserve(background=True)
        except QueueFull:
            if progress:
                progress("  … build queue full; waiting for a slot")
            time.sleep(config.ENRICH_SLOT_POLL_S)
            continue

        class _Stopped(Exception):
            pass

        def _check() -> None:
            if _run_state(bind, run_id) != CONTINUABLE:
                raise _Stopped()

        try:
            BUILDS.acquire(ticket, check_cancel=_check)
            return ticket
        except _Stopped:
            BUILDS.release(ticket)
            return None


def _default_probe(name: str, context: str):
    from .probe import probe_footprint

    return probe_footprint(name, context)


def execute_run(db: Session, run_id: str, limit: int = 0, expand: Callable = None,
                progress=None, claimed: bool = False,
                probe: Callable = None) -> dict:
    """Run the plan. Returns the final counters.

    `limit` caps how many tasks this invocation attempts (wave 1 uses
    config.ENRICH_WAVE1_SIZE); 0 means "until the plan is exhausted". A run
    that stops with pending tasks left lands in `paused`, not `done`, because
    it is resumable and calling it done would be a lie.

    `claimed` says the caller already won claim_run (the HTTP handler does, so
    it can 409 synchronously). Direct callers leave it False and this claims.
    """
    from ..graph import builder

    expand = expand or _default_expand
    probe = probe or _default_probe
    run = db.get(EnrichmentRun, run_id)
    if run is None:
        raise LookupError(f"unknown enrichment run {run_id}")
    if not claimed and not claim_run(db, run_id):
        raise ValueError(f"run is {run.state}, expected one of {STARTABLE}")

    db.expire(run)          # pick up the state the claim just wrote
    run.error = None
    if not run.started_at:
        run.started_at = _now()
    db.commit()

    bind = db.get_bind()
    deadline = (time.monotonic() + run.budget_s) if run.budget_s else None
    protected = _protected_norms(db, run_id)
    attempted = 0

    try:
        while True:
            if limit and attempted >= limit:
                break
            if deadline is not None and time.monotonic() >= deadline:
                if progress:
                    progress(f"budget of {run.budget_s:.0f}s exhausted")
                break
            if _run_state(bind, run_id) != CONTINUABLE:
                break
            task = _next_task(db, run_id)
            if task is None:
                break

            ticket = _acquire_background_slot(db, run_id, progress=progress)
            if ticket is None:
                break
            try:
                task.state = "enriching"
                task.attempts = (task.attempts or 0) + 1
                task.updated_at = _now()
                db.commit()
                if progress:
                    progress(f"[{task.rank}] {task.display_name} "
                             f"({task.context or 'no context'})")

                # One query decides whether the other ~34 are worth paying
                # for. Inside the build slot on purpose: it is still an
                # outbound provider call and belongs under admission control.
                # Org tasks skip it: the probe is person-shaped (it looks for
                # every name token in a result), and an employer several
                # contacts share reliably has a web presence anyway.
                if config.ENRICH_PROBE_ENABLED and task.kind != "org":
                    result = probe(task.display_name, task.context or "")
                    if not result.has_footprint:
                        task.state = "probed_empty"
                        task.last_error = None
                        if progress:
                            progress("  ⊘ no web footprint — 1 query instead of ~35")
                        attempted += 1
                        continue

                # Budget bounds the CONTACT too, not just the queue: without
                # this a single slow contact can overrun the whole budget.
                # expand_graph passes its own (possibly worker-thread) session
                # here; we deliberately ignore it and read on a fresh
                # connection — see _run_state.
                def _should_stop(_session) -> bool:
                    if deadline is not None and time.monotonic() >= deadline:
                        return True
                    return _run_state(bind, run_id) != CONTINUABLE

                is_person = task.kind != "org"
                # Org tasks carry no weights: the weights are inferred from a
                # PERSON's title and employer, and an organization has neither.
                expand(db, task.display_name, run.depth, task.context or "",
                       protected, _should_stop, progress, is_person,
                       (task.silo_weights or None) if is_person else None)

                if is_person:
                    person = builder.get_or_create_person(
                        db, task.display_name, allow_create=False)
                    task.person_id = person.id if person is not None else None
                task.state = "done"
                task.last_error = None
            except Exception as exc:
                db.rollback()
                task = db.get(EnrichmentTask, task.id)
                if task is not None:
                    task.state = "failed"
                    task.last_error = f"{exc.__class__.__name__}: {exc}"
                if progress:
                    progress(f"  ⚠ {exc.__class__.__name__}: {exc} — task failed")
            finally:
                BUILDS.release(ticket)
                if task is not None:
                    task.updated_at = _now()
                # Commit per task: this is what makes a mid-run restart resume
                # instead of replaying, and it keeps the executor from holding
                # a write transaction open for hours.
                db.commit()
            attempted += 1

        return _finalize(db, run_id, progress=progress)
    except Exception as exc:
        db.rollback()
        run = db.get(EnrichmentRun, run_id)
        if run is not None:
            run.state = "failed"
            run.error = f"{exc.__class__.__name__}: {exc}"
            run.finished_at = _now()
            run.counters = tally(db, run_id)
            db.commit()
        raise


def _has_retryable_failures(db: Session, run_id: str) -> bool:
    """A `failed` task with attempts left is still work to do, even though its
    state isn't `pending` — see _next_task. Without this, a run that stops
    (limit/budget/pause) with its last remaining task sitting in a
    not-yet-exhausted `failed` state would be reported `done`, and `done`
    can never be (re)started — silently stranding a retryable task forever."""
    return db.execute(
        select(EnrichmentTask.id).where(
            EnrichmentTask.run_id == run_id,
            EnrichmentTask.state == "failed",
            EnrichmentTask.attempts < config.ENRICH_MAX_ATTEMPTS,
        ).limit(1)
    ).first() is not None


def _finalize(db: Session, run_id: str, progress=None) -> dict:
    """Settle the run's terminal state from what is actually left to do."""
    run = db.get(EnrichmentRun, run_id)
    counts = tally(db, run_id)
    if run is None:
        return counts

    # The live state, not this session's cached copy — a cancel arrives from a
    # different session and must not be overwritten by our stale 'running'.
    # Only `counters` is dirty here, so committing writes just that column.
    state = _run_state(db.get_bind(), run_id)
    if state == "cancelled":
        pass  # terminal already; the canceller wrote finished_at
    elif (counts.get("pending", 0) or counts.get("enriching", 0)
          or _has_retryable_failures(db, run_id)):
        # Stopped early — budget, wave limit, or a pause. Resumable, so saying
        # "done" here would misreport an unfinished plan as a complete one.
        run.state = "paused"
    else:
        run.state = "done"
        run.finished_at = _now()
    run.counters = counts
    db.commit()
    if progress:
        progress(f"run {run.state}: {counts}")
    return counts
