"""contact_profiler: the target-independent half of "is this contact worth 35
queries" -- judged once from the uploaded row, stored, and reused by every
future /connect.

The tests that matter are the ones about a MISSING or malformed judgment. A
wrong verdict costs one contact's ranking; a verdict invented where none was
reached would silently mark a real person unsearchable for a month, because
these are cached.
"""
import pytest

from app import config
from app.extraction import contact_profiler
from app.models import LocalProfile
from app.network.ranking import BridgeTarget, score_contacts
from app.utils.names import person_norm_key


def _row(name, employer=None, title=None):
    return {"name": name, "employer": employer, "title": title, "school": None}


def _active(monkeypatch, payload):
    monkeypatch.setattr(config, "CONTACT_PROFILE_ENABLED", True)
    monkeypatch.setattr(contact_profiler, "claude_available", lambda: True)
    monkeypatch.setattr(contact_profiler, "call_json", lambda *a, **k: payload)
    # never touch the shared provider cache from a unit test
    monkeypatch.setattr(contact_profiler.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(contact_profiler.cache, "set", lambda *a, **k: None)


def test_verdicts_align_to_their_input_by_index(monkeypatch):
    _active(monkeypatch, {"results": [
        {"index": 2, "footprint": "none", "domain": "other", "why": "student club"},
        {"index": 1, "footprint": "individual", "domain": "company", "why": "founder"},
    ]})
    out = contact_profiler.profile([_row("A", "Acme", "Founder"),
                                    _row("B", "Chess Club", "Events Chair")])
    assert out[0]["footprint"] == "individual"
    assert out[1]["footprint"] == "none"


def test_a_missing_verdict_stays_none_rather_than_defaulting(monkeypatch):
    """These are cached for a month. Defaulting an unanswered row to a
    footprint would permanently mark a real contact unsearchable on one bad
    response, and the caller could no longer tell 'judged low' from 'not
    judged'."""
    _active(monkeypatch, {"results": []})
    assert contact_profiler.profile([_row("A", "Acme", "Founder")]) == [None]


def test_an_off_enum_footprint_is_not_a_judgment(monkeypatch):
    _active(monkeypatch, {"results": [
        {"index": 1, "footprint": "very important", "domain": "company", "why": "w"}]})
    assert contact_profiler.profile([_row("A")]) == [None]


def test_an_off_enum_domain_degrades_to_other(monkeypatch):
    """A bad domain is recoverable -- the footprint judgment is still usable --
    so unlike the footprint it degrades instead of voiding the verdict."""
    _active(monkeypatch, {"results": [
        {"index": 1, "footprint": "individual", "domain": "crypto vibes", "why": "w"}]})
    assert contact_profiler.profile([_row("A")])[0]["domain"] == "other"


def test_inactive_returns_all_none(monkeypatch):
    monkeypatch.setattr(config, "CONTACT_PROFILE_ENABLED", False)
    assert contact_profiler.profile([_row("A"), _row("B")]) == [None, None]
    assert contact_profiler.profile([]) == []


# ---------------------------------------------------------------------------
# ranking integration
# ---------------------------------------------------------------------------
def _contact(db, name, company, titles, reach=None):
    p = LocalProfile(canonical_name=name, norm_name=person_norm_key(name),
                     companies=[company], titles=titles, reach_profile=reach)
    db.add(p)
    db.flush()
    return p


def test_a_measured_footprint_outranks_a_title_regex(db):
    """The whole point. "Events Chair" at a student club scored founder-tier
    off the regex and outranked a real CTO; a judged footprint reverses it."""
    _contact(db, "Club Officer", "Nu Rho Psi - Epsilon Chapter", ["Events Chair"],
             reach={"footprint": "none", "domain": "other", "why": "campus chapter"})
    _contact(db, "Real Exec", "Pantheon", ["CTO"],
             reach={"footprint": "individual", "domain": "company", "why": "chief exec role"})
    db.commit()

    ranked = [c.display_name for c in score_contacts(db, target=BridgeTarget(name="X"))
              if c.skip_reason is None]
    assert ranked[0] == "Real Exec"


def test_the_title_regex_still_applies_to_unprofiled_contacts(db):
    """Backfill is incremental, so both kinds coexist. An unprofiled row must
    keep scoring the old way rather than dropping to zero."""
    _contact(db, "Unprofiled Founder", "Acme", ["Founder"])
    _contact(db, "Unprofiled Junior", "Acme Two", ["Associate"])
    db.commit()
    ranked = [c.display_name for c in score_contacts(db, target=BridgeTarget(name="X"))
              if c.skip_reason is None]
    assert ranked[0] == "Unprofiled Founder"


def test_footprint_replaces_the_title_score_rather_than_stacking(db):
    """Seniority is a PROXY for footprint. Once the footprint is measured,
    adding the proxy on top would double-count the same signal and let the
    regex keep overriding the measurement it stands in for."""
    _contact(db, "Founder Profiled", "Acme", ["Founder"],
             reach={"footprint": "org_only", "domain": "company", "why": "small shop"})
    db.commit()
    c = [x for x in score_contacts(db) if x.skip_reason is None][0]
    # base 1.0 + org_only 0.5 -- NOT + the 3.0 the title alone would have given
    assert c.score == pytest.approx(1.5)


def test_scored_contact_carries_footprint_and_domain(db):
    _contact(db, "X Person", "Acme", ["Founder"],
             reach={"footprint": "individual", "domain": "publications", "why": "w"})
    db.commit()
    c = [x for x in score_contacts(db) if x.skip_reason is None][0]
    assert (c.footprint, c.domain) == ("individual", "publications")


# ---------------------------------------------------------------------------
# tie-break -- a score tie must not become a permanent exclusion
# ---------------------------------------------------------------------------
def _many_tied(db, n=40):
    """n contacts that score identically: same title tier, distinct employers
    so the company decay doesn't separate them either."""
    for i in range(n):
        _contact(db, f"{chr(65 + i % 26)}{i:02d} Person", f"Employer {i}", ["Founder"])
    db.commit()


def test_a_tied_band_is_not_ordered_alphabetically(db):
    """The live failure: 175 contacts tied on one score, so the 15-contact
    shortlist was the alphabetical head of the band -- 12 of 15 names beginning
    'A' -- and the same 12 for every target."""
    _many_tied(db)
    top = [c for c in score_contacts(db, target=BridgeTarget(name="X"))
           if c.skip_reason is None][:15]
    names = [c.display_name for c in top]
    assert names != sorted(names), "ties must not fall back to alphabetical order"


def test_different_targets_sample_the_tied_band_differently(db):
    """Otherwise the same handful is considered for everyone and the rest of an
    equally-ranked band is unreachable for any target."""
    _many_tied(db)
    a = [c.display_name for c in score_contacts(db, target=BridgeTarget(name="Alpha"))
         if c.skip_reason is None][:15]
    b = [c.display_name for c in score_contacts(db, target=BridgeTarget(name="Beta"))
         if c.skip_reason is None][:15]
    assert a != b, "a tied band must be sampled per target, not fixed forever"


def test_the_same_target_plans_identically_twice(db):
    """The determinism the name sort provided has to survive: two runs of one
    target must produce the same plan."""
    _many_tied(db)
    t = BridgeTarget(name="Alpha")
    first = [c.display_name for c in score_contacts(db, target=t) if c.skip_reason is None]
    second = [c.display_name for c in score_contacts(db, target=t) if c.skip_reason is None]
    assert first == second


def test_the_tiebreak_never_reorders_across_scores(db):
    """It breaks ties only -- a higher score must still win outright."""
    _many_tied(db, n=10)
    # Genuinely higher, not just a different tier of the same size: shares the
    # target's own employer (+4.0), which is the strongest signal in the model.
    # ("individual" and a "Founder" title both score 3.0, so those alone tie.)
    _contact(db, "Zzz Topscorer", "Oracle", ["Founder"],
             reach={"footprint": "individual", "domain": "company", "why": "founder"})
    db.commit()
    ranked = [c for c in score_contacts(
        db, target=BridgeTarget(name="X", companies=["Oracle"]))
        if c.skip_reason is None]
    assert ranked[0].display_name == "Zzz Topscorer"
