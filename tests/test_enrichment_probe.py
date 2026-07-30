"""The footprint probe: one query instead of ~35 for contacts the web has
never written about.

Most people in a real export are in this category, so the probe is the largest
single saving on a long-tail run — but a WRONG "no footprint" verdict silently
drops a real contact out of enrichment, so the failure modes matter as much as
the savings.
"""
from app import config
from app.models import EnrichmentTask
from app.network.enrichment import plan_run, recently_probed_empty, tally
from app.network.executor import execute_run
from app.network.ingest import ingest_rows
from app.network.probe import ProbeResult, probe_footprint, probe_query


class _Result:
    def __init__(self, title="", snippet="", url="https://example.com"):
        self.title, self.snippet, self.url = title, snippet, url


def _rows(*contacts):
    return [{"Name": c.get("name", ""), "Company": c.get("company", "")}
            for c in contacts]


# --- the probe itself -------------------------------------------------------

def test_a_named_hit_counts_as_a_footprint():
    hit = _Result(title="Ada Lovelace joins Analytical Engines as CTO")
    result = probe_footprint("Ada Lovelace", "Analytical Engines",
                             search=lambda q: [hit])
    assert result.has_footprint
    assert result.hits == 1


def test_no_results_is_no_footprint():
    result = probe_footprint("Ada Lovelace", "Analytical Engines",
                             search=lambda q: [])
    assert not result.has_footprint


def test_pages_about_the_company_alone_are_not_a_footprint():
    """Searching a person AT a company routinely returns pages about the
    company that never mention the person. Counting those would defeat the
    probe entirely — every contact with a real employer would pass."""
    company_pages = [
        _Result(title="Analytical Engines raises a Series B"),
        _Result(title="Analytical Engines careers page"),
    ]
    result = probe_footprint("Ada Lovelace", "Analytical Engines",
                             search=lambda q: company_pages)
    assert not result.has_footprint
    assert result.hits == 0


def test_a_partial_name_match_is_not_enough():
    """A page about some other Lovelace is not evidence about Ada Lovelace."""
    result = probe_footprint("Ada Lovelace", "Analytical Engines",
                             search=lambda q: [_Result(title="Lovelace Prize announced")])
    assert not result.has_footprint


def test_the_name_may_be_split_across_title_and_snippet():
    result = probe_footprint(
        "Ada Lovelace", "Analytical Engines",
        search=lambda q: [_Result(title="Ada, our new CTO",
                                  snippet="…Lovelace joins from…")])
    assert result.has_footprint


def test_a_provider_failure_passes_rather_than_condemns():
    """A transient outage must never permanently mark a real contact as having
    no web presence — failing open costs one wasted sweep, failing closed
    silently loses the contact for ENRICH_PROBE_TTL_DAYS."""
    def _boom(q):
        raise ConnectionError("provider down")

    assert probe_footprint("Ada Lovelace", "Analytical Engines",
                           search=_boom).has_footprint


def test_the_probe_query_is_one_the_full_sweep_would_issue_anyway():
    """That is what makes a PASSING probe close to free: the provider layer is
    cache-first, so the sweep reuses the cached response."""
    from app.silos import SILOS
    query = probe_query("Ada Lovelace", "Analytical Engines")
    expected = f"{SILOS[0].render_queries('Ada Lovelace')[0]} Analytical Engines"
    assert query == expected


# --- integration with the executor -----------------------------------------

def _plan(db, n=3):
    ingest_rows(db, _rows(*[
        {"name": f"Contact Number{i}", "company": f"Company{i}"} for i in range(n)
    ]))
    return plan_run(db, "Siddhu Yen")


def _noop_expand(db, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
    pass


def test_a_footprintless_contact_is_never_swept(db):
    run = _plan(db, n=3)
    swept = []

    def _expand(db_, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
        swept.append(name)

    def _probe(name, context):
        return ProbeResult(name != "Contact Number1", 0, "q")

    counts = execute_run(db, run.id, expand=_expand, probe=_probe)
    assert "Contact Number1" not in swept
    assert counts["probed_empty"] == 1
    assert counts["done"] == 2


def test_probing_empty_is_terminal_not_a_failure(db):
    """It is a real, cheap answer — not an error to retry, and not pending
    work that would keep the run out of `done`."""
    run = _plan(db, n=2)
    counts = execute_run(db, run.id, expand=_noop_expand,
                         probe=lambda n, c: ProbeResult(False, 0, "q"))
    assert counts.get("failed", 0) == 0
    assert counts["probed_empty"] == 2
    task = db.query(EnrichmentTask).filter_by(run_id=run.id).first()
    assert task.last_error is None


def test_the_verdict_survives_into_the_next_run(db):
    """Without this the saving would be per-run: every replan would re-probe
    the same footprint-less contacts forever."""
    run = _plan(db, n=2)
    execute_run(db, run.id, expand=_noop_expand,
                probe=lambda n, c: ProbeResult(False, 0, "q"))
    assert len(recently_probed_empty(db)) == 2

    later = plan_run(db, "Siddhu Yen")
    assert tally(db, later.id)["probed_empty"] == 2
    assert tally(db, later.id).get("pending", 0) == 0


def test_an_expired_verdict_is_re_probed(db):
    """People do acquire a web presence — a new job, a funding round."""
    run = _plan(db, n=1)
    execute_run(db, run.id, expand=_noop_expand,
                probe=lambda n, c: ProbeResult(False, 0, "q"))
    stale = db.query(EnrichmentTask).filter_by(run_id=run.id).one()
    stale.updated_at = "2020-01-01T00:00:00+00:00"
    db.commit()

    assert recently_probed_empty(db) == set()
    assert tally(db, plan_run(db, "Siddhu Yen").id)["pending"] == 1


def test_the_probe_can_be_switched_off(db, monkeypatch):
    monkeypatch.setattr(config, "ENRICH_PROBE_ENABLED", False)
    run = _plan(db, n=2)
    swept = []

    def _expand(db_, name, depth, context, protected, should_stop, progress, is_person=True, silo_weights=None):
        swept.append(name)

    def _never_called(name, context):
        raise AssertionError("probe must not run when disabled")

    execute_run(db, run.id, expand=_expand, probe=_never_called)
    assert len(swept) == 2
