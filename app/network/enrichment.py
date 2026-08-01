"""Planning and tracking an initial-enrichment run.

A run is a ranked, budgeted, resumable plan for building out the operator's
second layer: L1 (their contacts) is already asserted by the export, and this
schedules the searches that discover L2 (who those contacts know).

Planning is deliberately separated from execution. `plan_run` costs nothing —
no searches, no Claude, no provider calls at all — so the operator (and we) can
see exactly which contacts would be enriched, in what order, and which are
excluded and why, BEFORE any money is spent. Execution consumes the plan.

Every task's state lives in the DB rather than in main.py's in-memory _JOBS: a
full run is hours of work, and a Render redeploy mid-run has to resume rather
than start from zero.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import config
from ..models import EnrichmentRun, EnrichmentTask
from .ranking import ScoredContact, org_sweep_candidates, score_contacts


class RunConflict(Exception):
    """A run is already in flight — caller should refuse rather than queue."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def recently_probed_empty(db: Session) -> Set[str]:
    """norm_names a recent run probed and found no web footprint for.

    Carries the verdict across runs, which is the whole reason the probe is
    worth recording rather than just acted on: without this, every replan
    would re-probe the same footprint-less contacts, and the saving would be
    per-run instead of permanent. Expires after config.ENRICH_PROBE_TTL_DAYS —
    people do acquire a web presence, just not weekly.
    """
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=config.ENRICH_PROBE_TTL_DAYS)).isoformat()
    return set(db.execute(
        select(EnrichmentTask.norm_name).where(
            EnrichmentTask.state == "probed_empty",
            EnrichmentTask.updated_at >= cutoff,
        )
    ).scalars())


def _task_state(contact: ScoredContact, probed_empty: Set[str]) -> str:
    if contact.skip_reason is not None:
        return "skipped"
    # Already expanded — by an earlier run, or by another operator sharing this
    # graph. expansion._reuse_existing_neighbors makes re-expanding it a no-op
    # that still costs a scheduling slot, so it starts life done.
    if contact.already_enriched:
        return "done"
    if contact.norm_name in probed_empty:
        return "probed_empty"
    return "pending"


def tally(db: Session, run_id: str) -> dict:
    """Task counts per state. Computed, never trusted from the cached copy on
    the run row — that one exists only so a list view avoids N aggregate
    queries."""
    rows = db.execute(
        select(EnrichmentTask.state, func.count())
        .where(EnrichmentTask.run_id == run_id)
        .group_by(EnrichmentTask.state)
    ).all()
    counts = {state: count for state, count in rows}
    counts["total"] = sum(count for _s, count in rows)
    return counts


def active_run(db: Session) -> Optional[EnrichmentRun]:
    return db.execute(
        select(EnrichmentRun).where(EnrichmentRun.state.in_(("running", "paused")))
    ).scalars().first()


def plan_run(db: Session, owner_name: str, owner_company: str = "",
             owner_school: str = "", depth: int = 1,
             budget_s: float = 0.0) -> EnrichmentRun:
    """Rank the operator's contacts and persist the resulting plan.

    Raises RunConflict when a run is already running or paused — two concurrent
    runs would race for the same build slots and enrich the same contacts twice.
    """
    existing = active_run(db)
    if existing is not None:
        raise RunConflict(f"run {existing.id} is already {existing.state}")

    run = EnrichmentRun(
        owner_name=owner_name.strip(),
        owner_company=(owner_company or "").strip() or None,
        owner_school=(owner_school or "").strip() or None,
        state="planned",
        depth=max(1, min(int(depth), 3)),
        budget_s=max(0.0, float(budget_s or 0.0)),
    )
    db.add(run)
    db.flush()

    # Wave 2 first: an org sweep reaches a whole organization's public
    # neighbourhood for one contact's worth of queries, so it is the highest
    # coverage per query available and must not sit behind 900 contacts.
    position = 0
    if config.ENRICH_ORG_SWEEPS_ENABLED:
        for sweep in org_sweep_candidates(db):
            position += 1
            db.add(EnrichmentTask(
                run_id=run.id, kind="org",
                display_name=sweep.name, norm_name=sweep.norm_name,
                context=None,          # an org name needs no disambiguation
                score=float(sweep.contacts),   # coverage IS the score
                rank=position, state="pending",
            ))

    scored = score_contacts(db, owner_name=owner_name,
                            owner_company=owner_company, owner_school=owner_school)
    probed_empty = recently_probed_empty(db)
    for contact in scored:
        position += 1
        db.add(EnrichmentTask(
            run_id=run.id, kind="contact",
            local_profile_id=contact.local_profile_id,
            display_name=contact.display_name,
            norm_name=contact.norm_name,
            context=contact.context or None,
            score=contact.score,
            silo_weights=contact.silo_weights or {},
            rank=position,
            state=_task_state(contact, probed_empty),
            skip_reason=contact.skip_reason,
        ))
    db.flush()
    run.counters = tally(db, run.id)
    db.commit()
    return run


def cancel_run(db: Session, run: EnrichmentRun) -> EnrichmentRun:
    """Stop a run. Tasks already done stay done — the graph keeps everything a
    cancelled run discovered, exactly like a killed build_yc_cache re-run."""
    if run.state in ("done", "cancelled", "failed"):
        return run
    run.state = "cancelled"
    run.finished_at = _now()
    run.counters = tally(db, run.id)
    db.commit()
    return run


def pending_tasks(db: Session, run_id: str, limit: int = 0) -> List[EnrichmentTask]:
    """The next tasks to execute, best-ranked first. Used by the executor."""
    stmt = (select(EnrichmentTask)
            .where(EnrichmentTask.run_id == run_id,
                   EnrichmentTask.state == "pending")
            .order_by(EnrichmentTask.rank))
    if limit > 0:
        stmt = stmt.limit(limit)
    return list(db.execute(stmt).scalars())


def run_dict(db: Session, run: EnrichmentRun, tasks_limit: int = 25) -> dict:
    """API shape: the run, live counters, and the head of the queue."""
    counts = tally(db, run.id)
    top = list(db.execute(
        select(EnrichmentTask)
        .where(EnrichmentTask.run_id == run.id,
               EnrichmentTask.state.in_(("pending", "enriching", "done")))
        .order_by(EnrichmentTask.rank)
        .limit(tasks_limit)
    ).scalars())
    return {
        "id": run.id,
        "owner_name": run.owner_name,
        "owner_company": run.owner_company,
        "owner_school": run.owner_school,
        "state": run.state,
        "depth": run.depth,
        "budget_s": run.budget_s,
        "counters": counts,
        "error": run.error,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "next_up": [task_dict(t) for t in top],
    }


def task_dict(task: EnrichmentTask) -> dict:
    return {
        "id": task.id,
        "rank": task.rank,
        "name": task.display_name,
        "context": task.context,
        "score": task.score,
        "silo_weights": task.silo_weights or {},
        "state": task.state,
        "skip_reason": task.skip_reason,
        "attempts": task.attempts,
        "last_error": task.last_error,
    }
